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

def calculate_intersection_union(outputs, targets, num_classes):
    # Initialize IoU per class
    iou_per_class = torch.zeros(num_classes)
    intersection_per_class = torch.zeros(num_classes)
    union_per_class = torch.zeros(num_classes)

    for cls in range(num_classes):
        # Get predictions and targets for the current class
        pred_cls = (outputs == cls).float()
        target_cls = (targets == cls).float()
        
        # Calculate intersection and union
        intersection_per_class[cls] = (pred_cls * target_cls).sum()
        union_per_class[cls] = (pred_cls + target_cls).sum() - intersection_per_class[cls]
        
    return intersection_per_class, union_per_class

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
    
    data_path_train = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [f"{i:02}" for i in range(11) if i != 8] for bin_path in glob.glob(f"/home/appuser/data/SemanticKitti/dataset/sequences/{folder}/velodyne/*.bin")]
    data_path_test = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for bin_path in glob.glob(f"/home/appuser/data/SemanticKitti/dataset/sequences/08/velodyne/*.bin")]
    
    depth_dataset_train = SemanticKitti(data_path_train, rotate=args.rotate, flip=args.flip)
    dataloader_train = DataLoader(depth_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    depth_dataset_test = SemanticKitti(data_path_test, rotate=False, flip=False)
    dataloader_test = DataLoader(depth_dataset_test, batch_size=1, shuffle=False, num_workers=4)
    
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
            
        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader_train)
        writer.add_scalar('Loss_EPOCH', avg_loss, epoch)
        print(f"Train Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

        total_loss = 0.0
        inference_times = []
        total_loss_RMSE = 0.0
        num_classes = 20
        intersection = torch.zeros(num_classes)
        union = torch.zeros(num_classes)
        nocs_model.eval()

        if not (epoch % 10 == 0):
            continue
        #vid_writer = imageio.get_writer('/workspace/data/train_SemanticKitti/ResNet50/epoch_val_{}.mp4'.format(epoch))
        for batch_idx, (range_img, reflectivity, xyz, normals, semantic)  in enumerate(dataloader_test):
            range_img, reflectivity, xyz, normals, semantic = range_img.to(device), reflectivity.to(device), xyz.to(device), normals.to(device), semantic.to(device)
            start_time = time.time()

            # run forward path
            start_time = time.time()
            start.record()
            outputs_semantic = nocs_model(torch.cat([range_img, reflectivity],axis=1), torch.cat([xyz, normals],axis=1))
            end.record()
            curr_time = (time.time()-start_time)*1000
    
            # Waits for everything to finish running
            torch.cuda.synchronize()
            
            loss = loss_semantic
            
            outputs_semantic_argmax = torch.argmax(outputs_semantic,dim=1)
            semseg_img = outputs_semantic_argmax
            intersection_, union_ = calculate_intersection_union(outputs_semantic_argmax, semantic, num_classes=20)
            intersection += intersection_
            union += union_
            print("inference took {} ms. loss_semantic {}".format(start.elapsed_time(end), loss_semantic.item()))
            
            inference_times.append(start.elapsed_time(end))
            #KNN(xyz, semseg_img, xyz_tensor)        

            semantics_pred = (semseg_img).permute(0, 1, 2)[0,...].cpu().detach().numpy()
            semantics_gt = (semantic[:,0,:,:]).permute(0, 1, 2)[0,...].cpu().detach().numpy()
            xyz_img = (xyz).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
            
            total_loss += loss.item()
            
            prev_sem_pred = visualize_semantic_segmentation_cv2(semantics_pred, class_colors=color_map)
            prev_sem_gt = visualize_semantic_segmentation_cv2(semantics_gt, class_colors=color_map)
            
            if args.visualization:
                cv2.imshow("inf", np.vstack((prev_sem_pred,prev_sem_gt)))
                if (cv.waitKey(1) & 0xFF) == ord('q'):
                    
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

    
    
        iou_per_class = torch.zeros(num_classes)
        for cls in range(num_classes):
            # Avoid division by zero
            if union[cls] == 0:
                iou_per_class[cls] = float('nan')
            else:
                iou_per_class[cls] = float(intersection[cls]) / float(union[cls])
                
            writer.add_scalar('IoU_{}'.format(class_names[cls]), iou_per_class[cls].item()*100, epoch)
            
        mIoU = np.nanmean(iou_per_class[1:].numpy()) # ignore background class and ignore not available classes
        writer.add_scalar('mIoU_Test', mIoU*100, epoch)
        writer.add_scalar('Inference Time', np.median(inference_times), epoch)
        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader_test)
        avg_loss_RMSE = np.sqrt(total_loss_RMSE / len(dataloader_test))
        writer.add_scalar('Loss_Test', avg_loss, epoch)
        print(f"Test Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}, Average Loss RMSE: {avg_loss_RMSE}")
    # # Save the trained model if needed
    torch.save(nocs_model.state_dict(), os.path.join(save_path, "model_final.pth"))
    
    # Close the TensorBoard writer
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train script for SemanticKitti')
    parser.add_argument('--model_type', type=str, default='shufflenet_v2_x0_5',
                        help='Type of the model to be used (default: resnet34)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the model (default: 0.001)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs for training (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (default: 1)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of data loading workers (default: 1)')
    parser.add_argument('--rotate', action='store_true',
                        help='Whether to apply rotation augmentation (default: False)')
    parser.add_argument('--flip', action='store_true',
                        help='Whether to apply flip augmentation (default: False)')
    parser.add_argument('--visualization', action='store_true',
                        help='Toggle visualization during training (default: False)')
    args = parser.parse_args()

    main(args)


