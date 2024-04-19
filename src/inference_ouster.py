import torch
from models.semanticFCN import SemanticNetworkWithFPN
from dataset.definitions import color_map, class_names
from dataset.utils import build_normal_xyz
import numpy as np
import open3d as o3d
import cv2 
import copy
from ouster import client
from ouster import pcap
from contextlib import closing
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def visualize_semantic_segmentation_cv2(mask, class_colors):
    """
    Visualize semantic segmentation mask using class colors with cv2.

    Parameters:
    - mask: 2D NumPy array containing class IDs for each pixel.
    - class_colors: Dictionary mapping class IDs to BGR colors.

    Returns:
    - visualization: Colored semantic segmentation image in BGR format.
    """
    h, w = mask.shape
    visualization = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        visualization[mask == class_id] = color

    return visualization
nocs_model = SemanticNetworkWithFPN(resnet_type='resnet18', meta_channel_dim=6, num_classes=20)
nocs_model.load_state_dict(torch.load("/home/appuser/data/train_semantic/Resnet18/model_final.pth"))

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nocs_model.to(device)
nocs_model.eval()

print("num_params", count_parameters(nocs_model))

pcap_path = "/home/appuser/data/Inference/190324/0006/Ouster/OS-2-128-992317000331-2048x10.pcap"
metadata_path = "/home/appuser/data/Inference/190324/0006/Ouster/OS-2-128-992317000331-2048x10.json"

with open(metadata_path, 'r') as f:
    metadata = client.SensorInfo(f.read())

source = pcap.Pcap(pcap_path, metadata)
load_scan = lambda:  client.Scans(source)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
with closing(load_scan()) as stream:
    for i, scan in enumerate(stream):
        xyzlut = client.XYZLut(metadata)
        xyz = xyzlut(scan)
        xyz = client.destagger(stream.metadata, xyz)
        h,w,c = xyz.shape
        scalar_Id = np.array(range(0,h*w)).astype(np.int32)
        reflectivity_field = scan.field(client.ChanField.REFLECTIVITY)
        reflectivity_img = client.destagger(stream.metadata, reflectivity_field)/255.0
        range_field = scan.field(client.ChanField.RANGE)
        range_img = client.destagger(stream.metadata, range_field)
        color_img = cv2.applyColorMap(np.uint8(reflectivity_img), cv2.COLORMAP_PARULA)
        range_img = np.linalg.norm(xyz,axis=-1)
        normals = build_normal_xyz(xyz)
        
        reflectivity_img =  torch.as_tensor(reflectivity_img[...,None].transpose(2, 0, 1).astype("float32"))
        range_img =  torch.as_tensor(range_img[...,None].transpose(2, 0, 1).astype("float32"))
        xyz =  torch.as_tensor(xyz[...,0:3].transpose(2, 0, 1).astype("float32"))

        normals =  torch.as_tensor(normals.transpose(2, 0, 1).astype("float32"))
        
        range_img, reflectivity, xyz, normals = range_img[None,...].to(device), reflectivity_img[None,...].to(device), xyz[None,...].to(device), normals[None,...].to(device)
        

        start_time = time.time()
        start.record()
        outputs_semantic = nocs_model(torch.cat([range_img, reflectivity],axis=1), torch.cat([xyz, normals],axis=1))
        end.record()
        curr_time = (time.time()-start_time)*1000

        # Waits for everything to finish running
        torch.cuda.synchronize()

        
        print("inference took time: cpu: {} ms., cuda: {} ms.".format(curr_time,start.elapsed_time(end)))
        semseg_img = torch.argmax(outputs_semantic,dim=1)
         
        semantics_pred = (semseg_img).permute(0, 1, 2)[0,...].cpu().detach().numpy()
        xyz_img = (xyz).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
        normal_img = (normals.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()+1)/2
        reflectivity_img = (reflectivity).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
        reflectivity_img = np.uint8(255*np.concatenate(3*[reflectivity_img],axis=-1))
        prev_sem_pred = visualize_semantic_segmentation_cv2(semantics_pred, class_colors=color_map)

        cv2.imshow("inf", np.vstack((reflectivity_img,np.uint8(255*normal_img),prev_sem_pred)))
        
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            
            grid_size = 50
            step_size = 1
 
            # Create vertices for the grid lines
            lines = []
            for i in range(-grid_size, grid_size + step_size, step_size):
                lines.append([[i, -grid_size, 0], [i, grid_size, 0]])
                lines.append([[-grid_size, i, 0], [grid_size, i, 0]])

            # Create an Open3D LineSet
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.array(lines).reshape(-1, 3))
            line_set.lines = o3d.utility.Vector2iVector(np.arange(len(lines) * 2).reshape(-1, 2))
            line_set.translate((0,0,-1.7))
            #time.sleep(10)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_img.reshape(-1,3))
            pcd.colors = o3d.utility.Vector3dVector(np.float32(prev_sem_pred[...,::-1].reshape(-1,3))/255.0)



            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            o3d.visualization.draw_geometries([line_set, mesh, pcd])
