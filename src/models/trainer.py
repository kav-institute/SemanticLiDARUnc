
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
from models.probability_helper import get_predictive_entropy, get_aleatoric_uncertainty, get_epistemic_uncertainty
from torch.special import digamma

def add_horizontal_uncertainty_colorbar(image, num_classes, colormap=cv2.COLORMAP_TURBO, height=20,
                                        num_ticks=5, font_scale=0.7, thickness=1, color=(225, 225, 225)):
    """
    Adds a horizontal colorbar for uncertainty values (0 to log(num_classes)) with ticks and labels below the image.

    Args:
        image: Colored uncertainty image (H, W, 3) as BGR numpy array
        num_classes: Number of classes to compute max uncertainty = log(num_classes)
        colormap: OpenCV colormap to use
        height: Height of the colorbar in pixels
        num_ticks: Number of tick labels (e.g., 5 → 0, 0.5, 1, 1.5, 2)
        font_scale: Font size for labels
        thickness: Line and text thickness

    Returns:
        Concatenated image with labeled horizontal colorbar below
    """
    max_uncertainty = np.log(num_classes)
    width = image.shape[1]

    # Generate horizontal gradient (left = 0, right = max_uncertainty)
    gradient = np.linspace(0, max_uncertainty, width).astype(np.float32).reshape(1, -1)
    gradient_norm = np.clip((gradient / max_uncertainty) * 255.0, 0, 255).astype(np.uint8)
    gradient_resized = cv2.resize(gradient_norm, (width, height), interpolation=cv2.INTER_LINEAR)
    colorbar = cv2.applyColorMap(gradient_resized, colormap)

    bar_with_ticks = colorbar.copy()

    # Draw ticks and labels along width (x-axis)
    text_labels = ["Certain", "Confident", "Ambiguous", "Doubtful", "Uncertain"]
    for i in range(5):
        x = int(i * (width - 1) / (num_ticks - 1))
        value = i * max_uncertainty / (num_ticks - 1)
        label = text_labels[i]

        # Draw vertical tick mark
        #cv2.line(bar_with_ticks, (x, 0), (x, 20), color=color, thickness=thickness)

        # Get text size for horizontal centering
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if i <= 2:
            text_x = x #+ text_size[0]#// 2  # center label horizontally on tick
        elif i == 2:
            text_x = x #- text_size[0] // 2  # center label horizontally on tick
        else:
            text_x = x - text_size[0] #// 2  # center label horizontally on tick
        text_y = text_size[1]  # below tick mark

        # Put label text
        cv2.putText(bar_with_ticks, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, lineType=cv2.LINE_AA)

    # Concatenate colorbar below the image
    return np.concatenate((image, bar_with_ticks), axis=0)


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
        if self.visualize:
            cv2.namedWindow("inf", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("inf", width=1500, height=800)   # width=800, height=600

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
        if cfg["extras"].get("save_path", 0):
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
            if self.cfg["extras"]["with_calibration_loss"]:
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
                loss = loss_nll
                if self.cfg["extras"]["with_calibration_loss"]:
                    loss_calibration = self.criterion_smooth_calibration(outputs_semantic, semantic)
                    loss += loss_calibration
            else:
                raise NotImplementedError
            
            # get the most likely class
            semseg_img = torch.argmax(outputs_semantic,dim=1)
            

            if self.visualize:
                if self.loss_function == "Dirichlet":
                    colormap_unc = cv2.COLORMAP_MAGMA   # cv2.COLORMAP_TURBO or cv2.COLORMAP_MAGMA
                    # get alpha concentration parameters
                    alpha = torch.nn.functional.softplus(outputs_semantic) + 1
                    
                    # predictive entropy /total uncertainty (H)
                    pred_entropy = get_predictive_entropy(alpha)
                    pred_entropy = (pred_entropy).permute(0, 1, 2)[0,...].cpu().detach().numpy()    # [B,H,W] -> [B,W,H] and use first element in batch
                    
                    # normalization: maximum uncertainty with "flattest" Dirichlet, which is at α_j=1 ∀j:
                        # H = -∑_j (α_j / α₀) * ln(α_j / α₀)
                        # at α_j=1 ∀j -> H_max = -∑_j (1/K) * ln(1/K) = ln(K)
                    pred_entropy_norm = pred_entropy/np.log(self.num_classes)
                    
                    # pred_entropy_img = cv2.applyColorMap(np.uint8(255*np.maximum(2*(pred_entropy_norm - 0.5), 0.0)), cv2.COLORMAP_TURBO)
                    pred_entropy_img = cv2.applyColorMap(np.uint8(255*np.maximum(pred_entropy_norm, 0.0)), colormap=colormap_unc)
                    pred_entropy_img = add_horizontal_uncertainty_colorbar(pred_entropy_img, num_classes=self.num_classes, height=20, colormap=colormap_unc)
                    
                    # aleatoric uncertainty (AU)
                    aleatoric_uncertainty = get_aleatoric_uncertainty(alpha)
                    aleatoric_uncertainty = aleatoric_uncertainty.permute(0, 1, 2)[0,...].cpu().detach().numpy()
                    
                    # normalization: maximum uncertainty with "flattest" Dirichlet, which is at α_j=1 ∀j:
                        # AU = -∑_j (α_j / α₀) * [ψ(α_j + 1) − ψ(α₀ + 1)]
                        # at α_j=1 ∀j -> AU_max = -∑_j (1/K) * [ψ(1 + 1) − ψ(K + 1)] = -ψ(2) + ψ(K + 1)
                    aleatoric_uncertainty_norm = aleatoric_uncertainty / (digamma(torch.tensor(self.num_classes+1.)) - digamma(torch.tensor(2.)))
                    
                    au_img = cv2.applyColorMap(np.uint8(255*np.maximum(aleatoric_uncertainty_norm, 0.0)), colormap=colormap_unc)
                    au_img = add_horizontal_uncertainty_colorbar(au_img, num_classes=self.num_classes, height=20, colormap=colormap_unc)
                    
                    # epistemic uncertainty (EU)
                    epistemic_uncertainty = pred_entropy - aleatoric_uncertainty 
                    # or get_epistemic_uncertainty(alpha).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                    
                    # normalization: maximum uncertainty with "flattest" Dirichlet, which is at α_j=1 ∀j:
                        # as EU = H - AU -> EU_max = ln K − [ψ(K+1) − ψ(2)] 
                    epistemic_uncertainty_norm = epistemic_uncertainty / (np.log(self.num_classes) - ( digamma(torch.tensor(self.num_classes+1.)) - digamma(torch.tensor(2.)) ) )
                    
                    eu_img = cv2.applyColorMap(np.uint8(255*np.maximum(epistemic_uncertainty_norm, 0.0)), colormap=colormap_unc)
                    eu_img = add_horizontal_uncertainty_colorbar(eu_img, num_classes=self.num_classes, height=20, colormap=colormap_unc)
                    print("pred_entropy: {}, aleatoric_unc: {}, epistemic_unc: {}".format(pred_entropy.mean(), aleatoric_uncertainty.mean(), epistemic_uncertainty.mean()))
                    
                # visualize first sample in batch
                semantics_pred = (semseg_img).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                semantics_gt = (semantic[:,0,:,:]).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                error_img = np.uint8(np.where(semantics_pred[...,None]!=semantics_gt[...,None], (0,0,255), (0,0,0)))
                xyz_img = (xyz).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                #normal_img = (normals.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()+1)/2
                reflectivity_img = (reflectivity).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                reflectivity_img = np.uint8(255*np.concatenate(3*[reflectivity_img],axis=-1))
                prev_sem_pred = visualize_semantic_segmentation_cv2(semantics_pred, class_colors=self.class_colors)
                prev_sem_gt = visualize_semantic_segmentation_cv2(semantics_gt, class_colors=self.class_colors)
                if self.loss_function == "Dirichlet":
                    cv2.imshow("inf", np.vstack((reflectivity_img, prev_sem_pred, prev_sem_gt, error_img, pred_entropy_img, au_img, eu_img)))
                    # cv2.imshow("inf", np.vstack((reflectivity_img,np.uint8(255*normal_img),prev_sem_pred,prev_sem_gt,error_img, pred_entropy_img)))
                else:
                    normal_img = (normals.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()+1)/2
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
                        if self.cfg["extras"]["with_calibration_loss"]:
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

