
import torch

from models.losses import TverskyLoss, SemanticSegmentationLoss, LovaszSoftmax, DirichletSegmentationLoss, DirichletCalibrationLoss

import time
import numpy as np
import cv2
import os
import open3d as o3d
import tqdm

from torch.utils.tensorboard import SummaryWriter
from models.evaluator import SemanticSegmentationEvaluator

from torch.special import digamma

def aleatoric_uncertainty(alpha, eps=1e-10):
    """
    Approximates aleatoric uncertainty (expected entropy) from Dirichlet parameters.

    Args:
        alpha: Tensor of shape [B, C, H, W]

    Returns:
        Tensor of shape [B, H, W]
    """
    alpha0 = torch.sum(alpha, dim=1, keepdim=True) + eps
    term1 = digamma(alpha0 + 1)
    term2 = torch.sum((alpha * digamma(alpha + 1)), dim=1, keepdim=True) / alpha0
    expected_entropy = term1 - term2
    return expected_entropy.squeeze(1)

def get_predictive_entropy(alpha, eps=1e-10):
    """
    Computes predictive entropy H(E[p]) from Dirichlet parameters.
    
    Args:
        alpha: Tensor of shape [B, C, H, W], Dirichlet parameters per class

    Returns:
        Tensor of shape [B, H, W], entropy of expected class probabilities
    """
    S = torch.sum(alpha, dim=1, keepdim=True)       # Total concentration α₀
    p = alpha / (S + eps)                           # Expected class probabilities
    entropy = -torch.sum(p * torch.log(p + eps), dim=1)  # Entropy across classes
    return entropy

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer:
    def __init__(self, model, optimizer, cfg, scheduler=None, visualize=False, logging=False, test_mask=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        #time.sleep(3)
        
        self.visualize = visualize
        self.logging = logging
        
        # config
        self.cfg = cfg
        self.normals = cfg["model_settings"]["normals"]==True
        self.use_reflectivity = cfg["extras"]["use_reflectivity"]==True
        self.num_classes = cfg["extras"]["num_classes"]
        self.class_names = cfg["extras"]["class_names"]
        self.class_colors = cfg["extras"]["class_colors"]

        self.loss_function = cfg["extras"]["loss_function"]
        # TensorBoard
        self.save_path = cfg["extras"]["save_path"]
        self.writer = SummaryWriter(self.save_path)

        # Timer
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        # Loss
        if  self.loss_function == "Tversky":
            self.criterion_dice = TverskyLoss()
            self.criterion_semantic = SemanticSegmentationLoss()
        elif self.loss_function == "CE":
            self.criterion_semantic = SemanticSegmentationLoss()
        elif  self.loss_function == "Lovasz":
            self.criterion_lovasz = LovaszSoftmax()
        elif self.loss_function == "Dirichlet":
            self.criterion_unc = DirichletSegmentationLoss()
            self.criterion_smooth_calibration = DirichletCalibrationLoss()
        else:
            raise NotImplementedError

        # Evaluator
        self.evaluator = SemanticSegmentationEvaluator(self.num_classes, test_mask=test_mask)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_one_epoch(self, dataloder, epoch):
        self.model.train()
        total_loss = 0.0
        # train one epoch
        for batch_idx, (range_img, reflectivity, xyz, normals, semantic) in enumerate(tqdm.tqdm(dataloder, desc=f"Epoch {epoch + 1}/{self.num_epochs}")):
            range_img, reflectivity, xyz, normals, semantic = range_img.to(self.device), reflectivity.to(self.device), xyz.to(self.device), normals.to(self.device), semantic.to(self.device)
    
            # run forward path
            start_time = time.time()
            self.start.record()
            if self.use_reflectivity:
                input_img = torch.cat([range_img, reflectivity],axis=1)
            else:
                input_img = range_img
            if self.normals:
                outputs_semantic = self.model(input_img, torch.cat([xyz, normals],axis=1))
            else:
                outputs_semantic = self.model(input_img, xyz)
            self.end.record()
            curr_time = (time.time()-start_time)*1000
    
            # Waits for everything to finish running
            torch.cuda.synchronize()
            
            # get losses
            if  self.loss_function == "Tversky":
                loss_semantic = self.criterion_semantic(outputs_semantic, semantic, num_classes=self.num_classes)
                loss_dice = self.criterion_dice(outputs_semantic, semantic, num_classes=self.num_classes, alpha=0.9, beta=0.1)
                loss = loss_dice+loss_semantic
            elif  self.loss_function == "CE":
                loss = self.criterion_semantic(outputs_semantic, semantic, num_classes=self.num_classes)
            elif  self.loss_function == "Lovasz":
                loss = self.criterion_lovasz(outputs_semantic, semantic)
            elif self.loss_function == "Dirichlet":
                loss_nll = self.criterion_unc(outputs_semantic, semantic)
                loss_calibration = self.criterion_smooth_calibration(outputs_semantic, semantic)
                loss = loss_nll + loss_calibration
            else:
                raise NotImplementedError
            
            # get the most likely class
            semseg_img = torch.argmax(outputs_semantic,dim=1)
            

            if self.visualize:
                if self.loss_function == "Dirichlet":
                    pred_entropy = get_predictive_entropy(torch.nn.functional.softplus(outputs_semantic)+1)
                    pred_entropy = (pred_entropy).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                    epis_img = cv2.applyColorMap(np.uint8(255*np.maximum(2*((pred_entropy/np.log(self.num_classes))-0.5),0.0)), cv2.COLORMAP_TURBO)
                    print(f"pred_ent: {np.mean(pred_entropy)}, min_ent: {np.min(pred_entropy)}, max_ent: {np.max(pred_entropy)}, median_ent: {np.median(pred_entropy)}, normalized_ent: {np.mean(pred_entropy/np.log(self.num_classes))}")
                    # epis_img = cv2.applyColorMap(np.uint8(255*cv2.normalize(epistemic, epistemic, 0, 1, cv2.NORM_MINMAX)), cv2.COLORMAP_TURBO)
                # visualize first sample in batch
                semantics_pred = (semseg_img).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                semantics_gt = (semantic[:,0,:,:]).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                error_img = np.uint8(np.where(semantics_pred[...,None]!=semantics_gt[...,None], (0,0,255), (0,0,0)))
                xyz_img = (xyz).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                normal_img = (normals.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()+1)/2
                reflectivity_img = (reflectivity).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                reflectivity_img = np.uint8(255*np.concatenate(3*[reflectivity_img],axis=-1))
                prev_sem_pred = visualize_semantic_segmentation_cv2(semantics_pred, class_colors=self.class_colors)
                prev_sem_gt = visualize_semantic_segmentation_cv2(semantics_gt, class_colors=self.class_colors)
                if self.loss_function == "Dirichlet":
                    cv2.imshow("inf", np.vstack((reflectivity_img,np.uint8(255*normal_img),prev_sem_pred,prev_sem_gt,error_img, epis_img)))
                else:
                    cv2.imshow("inf", np.vstack((reflectivity_img,np.uint8(255*normal_img),prev_sem_pred,prev_sem_gt,error_img)))
                if (cv2.waitKey(1) & 0xFF) == ord('q'):

                    #time.sleep(10)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz_img.reshape(-1,3))
                    pcd.colors = o3d.utility.Vector3dVector(np.float32(prev_sem_pred[...,::-1].reshape(-1,3))/255.0)
        
                    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
                    o3d.visualization.draw_geometries([mesh, pcd])
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            total_loss += loss.item()
            
            # Log to TensorBoard
            if self.logging:
                if batch_idx % 10 == 0:
                    step = epoch * len(dataloder) + batch_idx
                    self.writer.add_scalar('Loss/Total', loss.item(), step)
                    if self.loss_function == "Dirichlet": 
                        self.writer.add_scalar('Loss/NLL', loss_nll.item(), step)
                        self.writer.add_scalar('Loss/Calibration', loss_calibration.item(), step)
                    #self.writer.add_scalar('Semantic_Loss', loss_semantic.item(), step)
                    #self.writer.add_scalar('Dice_Loss', loss_dice.item(), step)
        
        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloder)
        print(f"Train Epoch {epoch + 1}/{self.num_epochs}, Average Loss: {avg_loss}")
        
        # Log to TensorBoard
        if self.logging:
            self.writer.add_scalar('Loss/Epoch', avg_loss, epoch)
        

    def test_one_epoch(self, dataloder, epoch):
        inference_times = []
        self.model.eval()
        self.evaluator.reset()
        for batch_idx, (range_img, reflectivity, xyz, normals, semantic) in enumerate(dataloder):
            range_img, reflectivity, xyz, normals, semantic = range_img.to(self.device), reflectivity.to(self.device), xyz.to(self.device), normals.to(self.device), semantic.to(self.device)
            start_time = time.time()

            # run forward path
            start_time = time.time()
            self.start.record()
            if self.normals:
                outputs_semantic = self.model(torch.cat([range_img, reflectivity],axis=1), torch.cat([xyz, normals],axis=1))
            else:
                outputs_semantic = self.model(torch.cat([range_img, reflectivity],axis=1), xyz)
            self.end.record()
            curr_time = (time.time()-start_time)*1000
            
            # Waits for everything to finish running
            torch.cuda.synchronize()

            # log inference times
            inference_times.append(self.start.elapsed_time(self.end))
            
            outputs_semantic_argmax = torch.argmax(outputs_semantic,dim=1)

            # get the most likely class
            semseg_img = torch.argmax(outputs_semantic,dim=1)

            self.evaluator.update(outputs_semantic_argmax, semantic)
            
        mIoU, result_dict = self.evaluator.compute_final_metrics(class_names=self.class_names)
        print(f"Test Epoch {epoch + 1}/{self.num_epochs}, mIoU: {mIoU}")
        
        # Log to TensorBoard
        if self.logging:
            for cls in range(self.num_classes):
                self.writer.add_scalar('IoU_{}'.format(self.class_names[cls]), result_dict[self.class_names[cls]]*100, epoch)
            
            self.writer.add_scalar('mIoU_Test', mIoU*100, epoch)
            self.writer.add_scalar('Inference Time', np.median(inference_times), epoch)
        
        return mIoU
    
    def __call__(self, dataloder_train, dataloder_test): #, num_epochs=50, test_every_nth_epoch=1, save_every_nth_epoch=-1):
        self.num_epochs = self.cfg["train_params"]["num_epochs"]
        self.test_every_nth_epoch = self.cfg["logging_settings"]["test_every_nth_epoch"]
        self.save_every_nth_epoch = self.cfg["logging_settings"]["save_every_nth_epoch"]
        for epoch in range(self.num_epochs):
            # train one epoch
            self.train_one_epoch(dataloder_train, epoch)
            # test
            if epoch > 0 and epoch % self.test_every_nth_epoch == 0:
                mIoU = self.test_one_epoch(dataloder_test, epoch)
                # update scheduler based on rmse
                if not isinstance(self.scheduler, type(None)):
                    self.scheduler.step(mIoU)
            # save
            if self.logging:
                if self.save_every_nth_epoch >= 1 and epoch % self.save_every_nth_epoch:
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, f"model_{str(epoch).zfill(6)}.pt"))
        
        # run final test
        self.test_one_epoch(dataloder_test, epoch)
        # save last epoch
        if self.logging:
            torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_final.pt"))

