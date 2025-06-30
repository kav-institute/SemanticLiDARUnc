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

def logits_to_DirichletDist(predicted_logits, eps=0.01):
    """
    Converts logits to Dirchlet alpha concentration parameters and distribution

    Args:
        predicted_logits (Tensor): Tensor of shape [B, C, H, W], Dirichlet parameters (alpha > 0)
        eps (float, optional): Small constant for numerical stability. Defaults to 0.01.

    Returns:
        tuple(Tensor, Tensor): 
            - dist: Dirichlet distribution of type torch.distributions.Dirichlet
            - alpha: alpha concentration parameters of shape [B, H, W, C] and
    """
    alpha = torch.nn.functional.softplus(predicted_logits)+1
    # Ensure alpha is strictly positive
    alpha = alpha + eps
    
    alpha = alpha.permute(0, 2, 3, 1)   # [B, H, W, C] for torch.distributions
    dist = Dirichlet(alpha)
    
    return dist, alpha

# Dirichlet
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
        # get Dirichlet distribution of NN output logits and alpha concentration parameters
        dist, alpha = logits_to_DirichletDist(predicted_logits, self.eps)
        
        # smooth the targets 
        target = smooth_one_hot(target,num_classes=num_classes,smoothing=self.smooth)

        # Compute log probability under the Dirichlet distribution
        log_prob = dist.log_prob(target.permute(0, 2, 3, 1))    # same shape

        # Loss is the negative log-likelihood
        loss = -log_prob.mean()

        return loss

# class DirichletCalibrationLoss(nn.Module):
#     def __init__(self, tau=0.03, eps=0.01, n_samples=32, num_classes=20, smooth=0.1, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
#         self.tau = tau
#         self.eps = eps
#         self.n_samples = n_samples
#         self.num_classes = num_classes
#         self.smooth=smooth
#         self.levels = levels
#         super(DirichletCalibrationLoss, self).__init__()
    
#     def soft_confidence(self, predicted_logits, target):
#         """
#         Differentiable analogue of the MC confidence value.
#         Smaller conf  ⇒ more confident.
        
#         Args:
#             predicted_logits: Tensor of shape [B, C, H, W] 
#                 -> alpha: Dirichlet parameters (alpha > 0) of shape [B, H, W, C]
#             target: Tensor of shape [B, C, H, W], one-hot encoded ground truth
#             epsilon: Small constant for numerical stability
#         """
#         # get Dirichlet distribution of NN output logits and alpha concentration parameters
#         dist, alpha = logits_to_DirichletDist(predicted_logits, self.eps)
        
#         # sample from Dirichlet distribution -> shape [n_samples, B, H, W, C]
#         samples = dist.rsample([self.n_samples])   
        
#         # smooth the targets and reshape to same shape as alpha
#         #target = smooth_one_hot(target,num_classes=self.num_classes,smoothing=self.smooth).permute(0, 2, 3, 1)
#         target = target.permute(0, 2, 3, 1)
#         #target_argmax = target.argmax(dim=-1, keepdim=True)
        
#         # 1) Expand target_argmax auf samples-Dim:
#         #    -> [n_samples, B, H, W, 1]
#         idx = target.unsqueeze(0).expand(self.n_samples, -1, -1, -1, -1)

#         # 2) Gather entlang der letzten Achse (C):
#         samples_gt = samples.gather(dim=-1, index=idx)
#         # samples_gt: [n_samples, B, H, W, 1]

#         # 3) Auf ein 4-D Tensor runterbrechen:
#         samples_gt = samples_gt.squeeze(-1)  # [n_samples, B, H, W]

#         p_mean = alpha / alpha.sum(-1, keepdim=True)                   # [B, H, W, C]
#         p_gt   = p_mean.gather(dim=-1, index=target)            # [B, H, W, 1]
#         p_gt   = p_gt.squeeze(-1)      

#         ind = torch.sigmoid((samples_gt - p_gt.unsqueeze(0)) / self.tau)    # 0/1 indicator for samples_gt > p_gt(mean)
#         conf = ind.mean(0)                                                  # [...]
#         return conf    
    
#     def coverage_penalty(self, conf, k=0.8, delta=0.01):
#         """
#         Encourage   P(conf ≤ k)  ≈  k    (coverage calibration)
#         Uses a triangular kernel of width 2δ around k.
#         """
#         # kernel weights w_i = max(0, 1 - |conf_i - k|/δ)
#         w = torch.relu(1 - torch.abs(conf - k) / delta)
#         # replaces (w > 0).float():
#         soft_mask = torch.sigmoid(w / delta)  
        
#         # replaces (conf <= k).float():
#         soft_indicator = torch.sigmoid((k - conf) / delta)
#         p_emp = (soft_indicator * w * soft_mask).sum() / ((w * soft_mask).sum() + self.eps)
#         # replaces p_emp = ((conf <= k).float() * w).sum() / w.sum()   # soft empirical CDF
#         return (p_emp - k).pow(2)       # MSE; differentiable
    
#     def forward(self, predicted_logits, target):
#         # 2) Smooth reliability penalty
#         conf  = self.soft_confidence(predicted_logits, target)   # [B,H,W]
#         cov_losses = []
#         for k in self.levels:
#             cov_losses.append( self.coverage_penalty(conf.flatten(), k=k) )
#         # coverage loss
#         loss = torch.stack(cov_losses).nanmean()
        
#         return loss
    
class DirichletCalibrationLoss(nn.Module):
    """
    Computes a differentiable approximation of the coverage calibration error
    based on MC samples from a Dirichlet distribution.

    Analytical metric:
        - Draw M samples p^{(m)} ~ Dir(alpha(x)).
        - Compute confidence c(x) = (1/M) * # of samples where p^{(m)}_y > mu_y,
            with mu = alpha / alpha_0 (expected class probabilities).
        - For nominal coverages k ∈ {0.1,...,0.9}, form empirical CDF F_emp(k) = P(c <= k).
        - Reliability scores: R_avg = 1 - ⟨|F_emp(k) - k|⟩_k, R_min = 1 - max_k |F_emp(k) - k|.
    Calibration is perfect if c ~ Uniform[0,1] so F_emp(k)=k for every (∀) k.
    """

    def __init__(self,
                tau: float = 0.03,
                learnable_tau: bool=True,
                delta: float = 0.01,
                learnable_delta:bool = True,
                eps: float = 1e-6,
                n_samples: int = 32,
                num_classes: int = 20,
                levels: list = None):
        super().__init__()
        # temperature for indicator smoothing
        tau_tensor = torch.tensor(tau, dtype=torch.float)   
        self.tau = nn.Parameter(tau_tensor, requires_grad=learnable_tau)

        # temperature for indicator smoothing
        self.delta = nn.Parameter(torch.tensor(delta, dtype=torch.float), requires_grad=learnable_delta)

        self.eps = eps          # numerical stability term
        self.n_samples = n_samples
        self.num_classes = num_classes
        # nominal coverage levels k at which to evaluate F_emp(k)
        self.levels = levels if levels is not None else [0.1 * i for i in range(1, 10)]

    def soft_confidence(self, predicted_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Step 1: Convert logits to Dirichlet parameters alpha(x) and sample M times.
        Step 2: Compute per-pixel confidence c(x) = (1/M) * Sum_m 1[p^{(m)}_y > mu_y].

        - predicted_logits: [B, C, H, W], unnormalized outputs
        - target: one-hot [B, C, H, W]
        Returns:
        - conf: [B, H, W] confidence values in [0,1]
        """
        # Analytical: alpha = f(logits) > 0; mu = alpha_y / Sum_j alpha_j
        dist, alpha = logits_to_DirichletDist(predicted_logits, self.eps)
        # Samples: {p^{(m)}} shape [M, B, H, W, C]
        samples = dist.rsample([self.n_samples])

        # Prepare ground-truth class indices: y(x)
        target_idx = target.permute(0, 2, 3, 1)

        # Expand indices to match samples shape: [M, B, H, W, 1]
        idx = target_idx.unsqueeze(0).expand(self.n_samples, -1, -1, -1, -1)

        # Gather p^{(m)}_y values: [M, B, H, W]
        samples_gt = samples.gather(dim=-1, index=idx).squeeze(-1)

        # Compute expected probability mu_y = alpha_y / alpha_0
        alpha_sum = alpha.sum(dim=-1, keepdim=True)       # [B,H,W,1]
        p_mean = alpha / alpha_sum                        # [B,H,W,C]
        # Gather mu_y to shape [B,H,W]
        p_gt = p_mean.gather(dim=-1, index=target_idx).squeeze(-1)

        # Indicator: 1[p^{(m)}_y > mu_y], smoothed by sigmoid
        # Sigmoid((p^{(m)}_y - mu_y)/τau) approximates the step function
        diff = (samples_gt - p_gt.unsqueeze(0)) / self.tau  # [M,B,H,W]
        ind = torch.sigmoid(diff)

        # Confidence c(x) = (1/M) * Sum_m ind
        conf = ind.mean(dim=0)  # [B,H,W]
        return conf

    def coverage_penalty(self, conf: torch.Tensor, k: float) -> torch.Tensor:
        """
        Step 3: Estimate empirical CDF F_emp(k) = P(conf <= k).
        We use a triangular kernel of width 2delta around k and soft indicators.

        - conf: flattened confidences shape [N]
        - k: nominal coverage level
        Returns:
        - (F_emp(k) - k)^2: squared deviation -> encourages F_emp(k) ≈ k
        """
        # Triangular weights: w_i = max(0, 1 - |conf_i - k|/delta)
        w = F.relu(1 - (conf - k).abs() / self.delta)
        # Soft mask ≈ indicator[w > 0]
        soft_mask = torch.sigmoid(w / self.delta)
        # Soft indicator ≈ 1[conf <= k]
        soft_indicator = torch.sigmoid((k - conf) / self.delta)

        # Empirical CDF: F_emp(k) approx. Sum_i soft_indicator * w * soft_mask / Sum_i w * soft_mask
        num = (soft_indicator * w * soft_mask).sum()
        den = (w * soft_mask).sum()
        p_emp = num / (den + self.eps)

        # Squared error at this level: (F̂(k) - k)^2
        return (p_emp - k).pow(2)

    def forward(self, predicted_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the average coverage calibration error over specified levels.
        """
        # Step 1+2: per-pixel confidence
        conf = self.soft_confidence(predicted_logits, target)  # [B,H,W]
        # Flatten spatial dims to vector
        conf_flat = conf.flatten()

        # Step 3: compute penalty for each k in levels
        losses = [self.coverage_penalty(conf_flat, k) for k in self.levels]
        # Average over levels
        loss = torch.stack(losses).mean()
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
