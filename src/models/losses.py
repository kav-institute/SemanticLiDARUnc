import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function

### >>> Output-kind classifier (logits / probs / log_probs) <<< ###
@torch.no_grad()
def classify_output_kind(outputs: torch.Tensor, class_dim: int = 1, sample_fraction: float = 0.1):
    """
    Heuristic:
      - probs:   values in [0,1] and sum over class_dim ≈ 1 per pixel
      - log_probs: values <= 0 typically, and exp(outputs) behaves like probs
      - else: logits
    """
    x = outputs

    # ---- optional subsample over spatial positions ----
    if sample_fraction and sample_fraction < 1.0 and x.ndim > 2:
        # Move class dim to 1 so x is [B, C, ...spatial...]
        x_perm = x.movedim(class_dim, 1).contiguous()  # [B, C, spatial...]
        # Flatten spatial dims to S
        x_flat = x_perm.flatten(start_dim=2)           # [B, C, S]

        S = x_flat.size(-1)
        k = max(1, int(S * sample_fraction))
        idx = torch.randperm(S, device=x.device)[:k]   # indices in [0, S)

        x = x_flat[..., idx]                           # [B, C, k]
    else:
        # Ensure class dim is at 1 for the checks below
        x = x.movedim(class_dim, 1).contiguous()

    # ---- decide kind: probs / log_probs / logits ----
    # Probabilities? values in [0,1] and sums ≈ 1 per pixel
    in_range = (x.min() >= -1e-6) and (x.max() <= 1 + 1e-6)
    sums = x.sum(dim=1)  # [B, S or spatial product]
    if in_range and torch.allclose(sums, torch.ones_like(sums), atol=1e-3, rtol=1e-3):
        return 'probs'

    # Log-probs? typically <= 0; exp() behaves like probs
    if x.max() <= 1e-6:
        ex = x.exp()
        ex_sums = ex.sum(dim=1)
        if torch.allclose(ex_sums, torch.ones_like(ex_sums), atol=1e-3, rtol=1e-3):
            return 'log_probs'

    return 'logits'


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, outputs, labels, num_classes=20, model_act=None):
        labels = labels.long()
        C = outputs.shape[1]

        # remap invalid labels to ignore_index
        invalid = (labels < 0) | (labels >= C)
        if invalid.any():
            labels = torch.where(invalid, torch.full_like(labels, self.ignore_index), labels)

        # CrossEntropyLoss expects logits (no softmax). It internally does LogSoftmax+NLLLoss
        if model_act == 'logits':
            return nn.CrossEntropyLoss(ignore_index=self.ignore_index)(outputs, labels)
        elif model_act == 'probs':
            # CE expects logits; switch to NLL over log-probs instead
            return nn.NLLLoss(ignore_index=self.ignore_index)(torch.log(outputs.clamp_min(1e-8)), labels)
        elif model_act == 'log_probs':
            return nn.NLLLoss(ignore_index=self.ignore_index)(outputs, labels)
        else:
            raise ValueError(f"Unknown model_act: {model_act}")


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.9, beta=0.1, smooth=1.0, ignore_index=255, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, outputs, labels, num_classes=20, model_act='logits'):
        # outputs -> probs
        if model_act == 'logits':
            probs = F.softmax(outputs, dim=1)
        elif model_act == 'probs':
            probs = outputs
        elif model_act == 'log_probs':
            probs = outputs.exp()
        else:
            raise ValueError(f"Unknown model_act: {model_act}")

        labels = labels.long()

        # build mask for valid pixels
        valid = (labels >= 0) & (labels < num_classes)
        if self.ignore_index is not None:
            valid = valid & (labels != self.ignore_index)

        if not valid.any():
            # no valid pixels; return zero loss to avoid NaNs
            return probs.new_tensor(0.0, requires_grad=True)

        # set ignored positions to class 0 (temporary), then mask later
        safe_labels = torch.where(valid, labels, torch.zeros_like(labels))
        one_hot = F.one_hot(safe_labels, num_classes=num_classes).permute(0,3,1,2).float()

        # mask out invalid pixels by zeroing both probs and one_hot there
        valid_mask = valid.unsqueeze(1).float()  # [B,1,H,W]
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        TP = (probs * one_hot).sum(dims)
        FP = ((1 - one_hot) * probs).sum(dims)
        FN = (one_hot * (1 - probs)).sum(dims)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = 1 - tversky  # [C]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# -------- Lovasz-Softmax helpers --------
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

####
def flatten_probas(probas, labels, ignore=None):
    """flattens per-pixel class probabilities and labels, removing ignore pixels"""
    if probas.dim() == 4:  # [B,C,H,W]
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    elif probas.dim() == 5:  # [B,C,D,H,W]
        B, C, D, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
    else:
        raise ValueError("probas dim must be 4 or 5")
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    return probas[valid], labels[valid]

def lovasz_softmax_flat(probas, labels, classes='present', reduction="mean"):
    """
    probas: [P, C] class probabilities at each prediction (sum(row)=1)
    labels: [P] ground truth labels {0,..,C-1}
    """
    if probas.numel() == 0:
        # only void pixels
        return probas.new_tensor(0.)
    C = probas.size(1)
    losses = []
    class_to_sum = range(C) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if classes == 'present' and fg.sum() == 0:
            continue
        # class c probability
        pc = probas[:, c]
        # errors: margin for the Lovasz extension (1 for fg, 0 for bg)
        errors = (fg - pc).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if not losses:  # no present classes
        return probas.new_tensor(0.)
    
    losses = torch.stack(losses)
    if reduction == 'none':
        loss = losses
    elif reduction == 'sum':
        loss = losses.sum()
    else:
        loss = losses.mean()
    return loss

class LovaszSoftmaxStable(nn.Module):
    def __init__(self, ignore_index=255, classes='present'):
        super().__init__()
        self.ignore_index = ignore_index
        self.classes = classes

    def forward(self, outputs, labels, model_act=None):
        if model_act == 'logits':
            probs = F.softmax(outputs, dim=1)
        elif model_act == 'probs':
            probs = outputs
        elif model_act == 'log_probs':
            probs = outputs.exp()
        else:
            raise ValueError(f"Unknown model_act: {model_act}")
        #probs = torch.softmax(outputs, dim=1)
        probs_flat, labels_flat = flatten_probas(probs, labels.long(), ignore=self.ignore_index)
        return lovasz_softmax_flat(probs_flat, labels_flat, classes=self.classes)

# -------- Combined SalsaNext objective --------
class SalsaNextLoss(nn.Module):
    """
    Cross-Entropy (with class weights) + Lovasz-Softmax (probabilities),
    works for both plain and ADF models.
    """
    def __init__(self, class_weights=None, ignore_index=255, lovasz_classes='present'):
        super().__init__()
        # lovasz_classes:
            # ='all'    : always loops over all classes and averages their losses, even if a class doesn’t appear in the batch. This adds noise (not recommended)
            # ='present': means only classes present in the batch contribute to Lovasz (stable for sparse scenes).
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.lovasz = LovaszSoftmaxStable(ignore_index=ignore_index, classes=lovasz_classes)
        self.ignore_index = ignore_index

    def forward(self, outputs, labels):
        # Accept either logits tensor or (mean_logits, var_logits)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            logits = outputs[0]  # mean logits from ADF
        else:
            logits = outputs

        # 1) Weighted CE on logits
        loss_ce = self.ce(logits, labels.long())

        # 2) Lovasz-Softmax on probabilities
        probs = torch.softmax(logits, dim=1)
        loss_ls = self.lovasz(probs, labels.long())

        return loss_ce + loss_ls    #, {"ce": loss_ce.detach(), "lovasz": loss_ls.detach()}

from baselines.SalsaNext import adf
class SoftmaxHeteroscedasticLoss(torch.nn.Module):
    # L = 0.5 * ((y - mu)**2 / (var + eps)) + 0.5 * log(var + eps)
    def __init__(self, num_classes):
        super(SoftmaxHeteroscedasticLoss, self).__init__()
        self.adf_softmax = adf.Softmax(dim=1, keep_variance_fn=adf.keep_variance_fn)

    def forward(self, outputs, targets, eps=1e-5):
        mean, var = self.adf_softmax(*outputs)
        targets = torch.nn.functional.one_hot(targets.squeeze(1), num_classes=20).permute(0,3,1,2).float()

        precision = 1 / (var + eps)
        return torch.mean(0.5 * precision * (targets - mean) ** 2 + 0.5 * torch.log(var + eps))


### >>> Dirichlet <<< ###
from models.probability_helper import (
    to_alpha_concentrations, 
    alphas_to_Dirichlet, 
    smooth_one_hot
)
class DirichletNLLLoss(nn.Module):
    def __init__(self, smooth=0.1, eps=0.01):

        self.smooth = smooth
        self.eps = eps
        super().__init__()

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
        # get Dirichlet distribution of NN output logits and alpha concentration parameters
        alpha = to_alpha_concentrations(predicted_logits)
        dist = alphas_to_Dirichlet(alpha)
        
        # smooth the targets 
        target = smooth_one_hot(target,num_classes=num_classes,smoothing=self.smooth)

        # Compute log probability under the Dirichlet distribution
        log_prob = dist.log_prob(target.permute(0, 2, 3, 1))    # same shape

        # Loss is the negative log-likelihood
        loss = -log_prob.mean()

        return loss