import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.distributions import Dirichlet

def smooth_one_hot(targets, num_classes, smoothing=0.1):
    """
    Smooths one-hot encoded target labels for semantic segmentation.

    Args:
        targets: LongTensor of shape [B, 1, H, W], class indices
        num_classes: int, number of classes
        smoothing: float, amount of label smoothing (0.0 = no smoothing)

    Returns:
        FloatTensor of shape [B, C, H, W], smoothed one-hot labels
    """
    confidence = 1.0 - smoothing
    low_conf = smoothing / (num_classes - 1)

    B, _, H, W = targets.shape

    # Initialize full tensor with low confidence
    one_hot = torch.full(
        (B, num_classes, H, W),
        fill_value=low_conf,
        device=targets.device,
        dtype=torch.float
    )

    # Squeeze channel dimension to get [B, H, W]
    targets = targets.squeeze(1)

    # Scatter high confidence to correct class
    one_hot.scatter_(1, targets.unsqueeze(1), confidence)
    #one_hot = torch.nn.functional.softmax(one_hot, dim=1)
    return one_hot

class DirichletSegmentationLoss(nn.Module):
    def __init__(self, smooth=0.1, eps=0.01):
        self.smooth = smooth
        self.eps = eps
        super(DirichletSegmentationLoss, self).__init__()


    def forward(self, predicted_logits, target, num_classes=20):
        """
        Dirichlet Negative Log-Likelihood loss for semantic segmentation.

        Args:
            alpha: Tensor of shape [B, C, H, W], Dirichlet parameters (alpha > 0)
            target: Tensor of shape [B, C, H, W], one-hot encoded ground truth
            epsilon: Small constant for numerical stability

        Returns:
            Scalar loss value
        """
        alpha = torch.nn.functional.softplus(predicted_logits)+1
        # Ensure alpha is strictly positive
        alpha = alpha + self.eps

        # smooth the targets 
        target = smooth_one_hot(target,num_classes=num_classes,smoothing=self.smooth)

        # Compute log probability under the Dirichlet distribution
        dist = Dirichlet(alpha.permute(0, 2, 3, 1))  # [B, H, W, C] for torch.distributions
        log_prob = dist.log_prob(target.permute(0, 2, 3, 1))  # same shape

        # Loss is the negative log-likelihood
        loss = -log_prob.mean()

        return loss

class SemanticSegmentationLoss(nn.Module):
    def __init__(self):
        super(SemanticSegmentationLoss, self).__init__()

        # Assuming three classes
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predicted_logits, target, num_classes=20):
        # Flatten the predictions and the target
        predicted_logits_flat = predicted_logits.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        target_flat = target.view(-1)

        # Calculate the Cross-Entropy Loss
        loss = self.criterion(predicted_logits_flat, target_flat)

        return loss

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.9, beta=0.1, num_classes=20):
        targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).transpose(1, 4).squeeze(-1)   
        inputs = F.softmax(inputs, dim=1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        inputs, targets = self.prob_flatten(inputs, targets)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses
