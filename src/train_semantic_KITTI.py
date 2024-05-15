import glob
import argparse
import torch
from torch.utils.data import DataLoader
from dataset.dataloader_semantic_KITTI import SemanticKitti
from models.semanticFCN import SemanticNetworkWithFPN
from models.losses import TverskyLoss, SemanticSegmentationLoss
import torch.optim as optim
import tqdm
import time
import numpy as np
import cv2
import os
import open3d as o3d
from dataset.definitions import color_map, class_names
from torch.utils.tensorboard import SummaryWriter

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

def main(args):
    # DataLoader
    data_path_train = [("/home/appuser/data/SemanticKitti/dataset/sequences/00/velodyne/{}.bin".format(str(i).zfill(6)),
                        "/home/appuser/data/SemanticKitti/dataset/sequences/00/labels/{}.label".format(str(i).zfill(6))) for i in range(0,4541)]
    
    data_path_train += [("/home/appuser/data/SemanticKitti/dataset/sequences/01/velodyne/{}.bin".format(str(i).zfill(6)),
                        "/home/appuser/data/SemanticKitti/dataset/sequences/01/labels/{}.label".format(str(i).zfill(6))) for i in range(0,1101)]
    
    # data_path_test = [("/workspace/data/SemanticKitti/data_odometry_velodyne/dataset/sequences/08/velodyne/{}.bin".format(str(i).zfill(6)),
    #                     "/workspace/data/SemanticKitti/data_odometry_velodyne/dataset/sequences/08/labels/{}.label".format(str(i).zfill(6))) for i in range(0,4071)]
    
    
    depth_dataset_train = SemanticKitti(data_path_train, rotate=args.rotate, flip=args.flip)
    dataloader_train = DataLoader(depth_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # depth_dataset_test = SemanticKitti(data_path_test, rotate=False, flip=False)
    # dataloader_test = DataLoader(depth_dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Depth Estimation Network
    nocs_model = SemanticNetworkWithFPN(backbone=args.model_type, meta_channel_dim=6, num_classes=20)
    print("num_params", count_parameters(nocs_model))
    
    # Define optimizer
    optimizer = optim.Adam(nocs_model.parameters(), lr=args.learning_rate)
    
    # Define loss functions
    criterion_dice = TverskyLoss()
    criterion_semantic = SemanticSegmentationLoss()
    
    # TensorBoard
    save_path ='/home/appuser/data/train_semantic/{}/'.format(args.model_type)
    writer = SummaryWriter(save_path)
    
    # Timer
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Training loop
    num_epochs = args.num_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nocs_model.to(device)
    
    for epoch in range(num_epochs):
        nocs_model.train()
        total_loss = 0.0
        # train one epoch
        for batch_idx, (range_img, reflectivity, xyz, normals, semantic) in enumerate(dataloader_train): #enumerate(tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            range_img, reflectivity, xyz, normals, semantic = range_img.to(device), reflectivity.to(device), xyz.to(device), normals.to(device), semantic.to(device)
    
            # run forward path
            start_time = time.time()
            start.record()
            outputs_semantic = nocs_model(torch.cat([range_img, reflectivity],axis=1), torch.cat([xyz, normals],axis=1))
            end.record()
            curr_time = (time.time()-start_time)*1000
    
            # Waits for everything to finish running
            torch.cuda.synchronize()
            
            # get losses
            loss_semantic = criterion_semantic(outputs_semantic, semantic, num_classes=20)
            loss_dice = criterion_dice(outputs_semantic, semantic, num_classes=20, alpha=0.9, beta=0.1)
            loss = loss_dice+loss_semantic
            
            print("inference took time: cpu: {} ms., cuda: {} ms. loss: {}".format(curr_time,start.elapsed_time(end), loss.item()))
            
            # get the most likely class
            semseg_img = torch.argmax(outputs_semantic,dim=1)
            
            if args.visualization:
                # visualize first sample in batch
                semantics_pred = (semseg_img).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                semantics_gt = (semantic[:,0,:,:]).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                xyz_img = (xyz).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                normal_img = (normals.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()+1)/2
                reflectivity_img = (reflectivity).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                reflectivity_img = np.uint8(255*np.concatenate(3*[reflectivity_img],axis=-1))
                prev_sem_pred = visualize_semantic_segmentation_cv2(semantics_pred, class_colors=color_map)
                prev_sem_gt = visualize_semantic_segmentation_cv2(semantics_gt, class_colors=color_map)
                cv2.imshow("inf", np.vstack((reflectivity_img,np.uint8(255*normal_img),prev_sem_pred,prev_sem_gt)))
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
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            
            # Log to TensorBoard
            if batch_idx % 10 == 0:
                step = epoch * len(dataloader_train) + batch_idx
                writer.add_scalar('Loss', loss.item(), step)
                writer.add_scalar('Semantic_Loss', loss_semantic.item(), step)
                writer.add_scalar('Dice_Loss', loss_dice.item(), step)
        
        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader_train)
        writer.add_scalar('Loss_EPOCH', avg_loss, epoch)
        print(f"Train Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")
    
    # # Save the trained model if needed
    torch.save(nocs_model.state_dict(), os.path.join(save_path, "model_final.pth"))
    
    # Close the TensorBoard writer
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train script for SemanticKitti')
    parser.add_argument('--model_type', type=str, default='resnet34',
                        help='Type of the model to be used (default: resnet34)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the model (default: 0.001)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs for training (default: 50)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training (default: 1)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of data loading workers (default: 1)')
    parser.add_argument('--rotate', action='store_true',
                        help='Whether to apply rotation augmentation (default: False)')
    parser.add_argument('--flip', action='store_true',
                        help='Whether to apply flip augmentation (default: False)')
    parser.add_argument('--visualization', action='store_true',
                        help='Toggle visualization during training (default: False)')
    args = parser.parse_args()

    main(args)

