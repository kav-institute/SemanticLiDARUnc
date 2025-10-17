import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
            return nn.NLLLoss(ignore_index=self.ignore_index)(torch.log(outputs+1e-8), labels)
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

# -------- Lovasz-Softmax --------
class LovaszSoftmaxStable(nn.Module):
    def __init__(self, ignore_index=None, classes='present'):
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
        probs_flat, labels_flat = self.flatten_probas(probs, labels.long(), ignore=self.ignore_index)
        return self.lovasz_softmax_flat(probs_flat, labels_flat, classes=self.classes)
    
    def lovasz_grad(self, gt_sorted):
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

    def flatten_probas(self, probas, labels, ignore=None):
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

    def lovasz_softmax_flat(self, probas, labels, classes='present', reduction="mean"):
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
            grad = self.lovasz_grad(fg_sorted)
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


### >>> Dirichlet <<< ###
import math
import torch
import torch.nn as nn
from typing import Iterable, Union
from torch.special import digamma, gammaln as lgamma, polygamma

# ----------------- helpers -----------------

def _valid_mask(target: torch.Tensor, ignore_index: int | None) -> torch.Tensor:
    """
    Build a boolean mask of valid pixels.
    target: [B,H,W] or [B,1,H,W]
    """
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0]
    if ignore_index is None:
        return torch.ones_like(target, dtype=torch.bool)
    return (target != ignore_index)

def _beta_moment(a: torch.Tensor, b: torch.Tensor, q: float) -> torch.Tensor:
    """
    E[p^q] for Beta(a,b) = B(a+q,b) / B(a,b) computed in log-space:
    exp(lgamma(a+q) - lgamma(a) + lgamma(a+b) - lgamma(a+b+q))
    """
    return torch.exp(lgamma(a + q) - lgamma(a) + lgamma(a + b) - lgamma(a + b + q))

def _kl_map(alpha: torch.Tensor, alpha_prior: torch.Tensor) -> torch.Tensor:
    """
    KL(Dir(alpha) || Dir(alpha_prior)) per-pixel map.
    Shapes:
      alpha, alpha_prior: [B,C,H,W]
    Returns:
      [B,H,W]
    Formula:
      KL = logGamma(a0) - sum_i logGamma(ai)
           - (logGamma(a0p) - sum_i logGamma(aip))
           + sum_i (ai - aip) * (digamma(ai) - digamma(a0))
    where a0 = sum_i ai, a0p = sum_i aip.
    """
    a0  = alpha.sum(dim=1, keepdim=True)
    a0p = alpha_prior.sum(dim=1, keepdim=True)
    t1 = lgamma(a0) - lgamma(a0p)
    t2 = (lgamma(alpha_prior) - lgamma(alpha)).sum(dim=1, keepdim=True)
    t3 = ((alpha - alpha_prior) * (digamma(alpha) - digamma(a0))).sum(dim=1, keepdim=True)
    return (t1 + t2 + t3).squeeze(1)  # [B,H,W]

# ----------------- objectives (alpha-only) -----------------

def dce_from_alpha(alpha: torch.Tensor, target: torch.Tensor,
                   ignore_index: int | None = 0) -> torch.Tensor:
    """
    Dirichlet expected cross-entropy:
      E[-log p_y] = digamma(alpha0) - digamma(alpha_y)
    Returns masked mean scalar.
    """
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0]
    valid = _valid_mask(target, ignore_index)
    if valid.sum() == 0:
        return alpha.sum() * 0.0
    a0 = alpha.sum(dim=1)                                # [B,H,W]
    ay = alpha.gather(1, target.unsqueeze(1)).squeeze(1) # [B,H,W]
    per_pix = (digamma(a0) - digamma(ay))
    return (per_pix * valid.float()).sum() / valid.sum()

def nll_from_alpha(alpha: torch.Tensor, target: torch.Tensor,
                   num_classes: int, smoothing: float = 0.25,
                   ignore_index: int | None = 0, eps: float = 1e-8) -> torch.Tensor:
    """
    Dirichlet negative log-likelihood (density) at a point x on the simplex:
      -log Dir(x; alpha) = -( logGamma(a0) - sum_i logGamma(ai) + sum_i (ai - 1) * log(xi) )
    Here x is a smoothed one-hot target.
    Returns masked mean scalar.
    """
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0]  # [B,H,W]

    valid = _valid_mask(target, ignore_index)
    if valid.sum() == 0:
        return alpha.sum() * 0.0

    # Build smoothed one-hot x (strictly positive). Confidence on target index is (1 - smoothing).
    conf = 1.0 - smoothing
    low  = smoothing / (num_classes - 1)
    B, C, H, W = alpha.shape
    x = torch.full((B, C, H, W), low, device=alpha.device, dtype=alpha.dtype)
    x.scatter_(1, target.unsqueeze(1), conf)
    log_x = torch.log(x.clamp_min(eps))  # safe log

    # log Dir(alpha) at x
    a0   = alpha.sum(dim=1)                           # [B,H,W]
    logZ = lgamma(a0) - lgamma(alpha).sum(dim=1)      # [B,H,W]
    logp = logZ + ((alpha - 1.0) * log_x).sum(dim=1)  # [B,H,W]

    # masked mean of -logp
    return (-(logp) * valid.float()).sum() / valid.sum()

def nll_dirichlet_categorical(alpha: torch.Tensor, target: torch.Tensor,
                            ignore_index: int | None = 0, eps: float = 1e-12) -> torch.Tensor:
    # Dirichlet-categorical marginal likelihood for a one-hot label
    # -log E[p_y] = -log(alpha_y / alpha0)
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0]
    valid = _valid_mask(target, ignore_index)
    if valid.sum() == 0:
        return alpha.sum() * 0.0
    a0 = alpha.sum(dim=1)                                # [B,H,W]
    ay = alpha.gather(1, target.unsqueeze(1)).squeeze(1) # [B,H,W]
    per_pix = -(torch.log(ay + eps) - torch.log(a0 + eps))
    return (per_pix * valid.float()).sum() / valid.sum()

# class_weights: tensor[C], e.g. w_c = 1 / sqrt(freq_c + eps), then normalize to mean 1
def nll_dirichlet_categorical_weighted(
    alpha: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor,
    ignore_index: int | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    # alpha: [B,C,H,W]
    # target: [B,H,W] or [B,1,H,W] with int labels (0..C-1) and possibly ignore_index
    # class_weights: [C] (already mean-normalized, clipped)

    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0]            # [B,H,W]
    target = target.long()

    # valid mask
    valid = (target != ignore_index) if ignore_index is not None \
            else torch.ones_like(target, dtype=torch.bool)

    # safe target for indexing/gather (replace ignored indices with 0)
    tgt_safe = torch.where(valid, target, torch.zeros_like(target))

    # per-pixel -log p_hat_y = -log(alpha_y / alpha0)
    a0 = alpha.sum(dim=1)                                    # [B,H,W]
    ay = alpha.gather(1, tgt_safe.unsqueeze(1)).squeeze(1)   # [B,H,W]
    per = -(torch.log(ay + eps) - torch.log(a0 + eps))       # [B,H,W]

    # class weights (do not index at ignore_index)
    w_all = class_weights.to(alpha.device)
    w = w_all[tgt_safe].clamp_min(1e-3)                      # [B,H,W]

    # weighted mean over valid pixels
    per = per * w * valid
    num = per.sum()
    den = (w * valid).sum().clamp_min(1.0)
    return num / den

def imax_from_alpha(alpha: torch.Tensor, target: torch.Tensor,
                    p_moment: float = 2.0,
                    ignore_index: int | None = 0) -> torch.Tensor:
    """
    iMAX upper bound objective on ||y - p||_inf using Beta moments.
    For p ~ Dir(alpha), p_j ~ Beta(alpha_j, alpha0 - alpha_j).
      E[p_j^q] = B(alpha_j + q, alpha0 - alpha_j) / B(alpha_j, alpha0 - alpha_j)
      E[(1 - p_c)^q] = B((alpha0 - alpha_c) + q, alpha_c) / B(alpha0 - alpha_c, alpha_c)
    We compute:
      F = { E[(1 - p_c)^p] + sum_j E[p_j^p] - E[p_c^p] }^(1/p)
    Return masked mean scalar.
    """
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0]
    valid = _valid_mask(target, ignore_index)
    if valid.sum() == 0:
        return alpha.sum() * 0.0

    p   = float(p_moment)
    a0  = alpha.sum(dim=1)                                # [B,H,W]
    ac  = alpha.gather(1, target.unsqueeze(1)).squeeze(1) # [B,H,W]
    bc  = a0 - ac
    # E[(1 - p_c)^p]
    term_c = _beta_moment(bc, ac, p)
    # sum_j E[p_j^p] - E[p_c^p]
    a_all  = alpha
    b_all  = a0.unsqueeze(1) - a_all
    ep_all = _beta_moment(a_all, b_all, p)       # [B,C,H,W]
    ep_sum = ep_all.sum(dim=1)                   # [B,H,W]
    ep_c   = _beta_moment(ac, bc, p)             # [B,H,W]
    per_pix = (term_c + (ep_sum - ep_c) + 1e-12).pow(1.0 / p)
    return (per_pix * valid.float()).sum() / valid.sum()

# ----------------- regularizers (alpha-only) -----------------

def kl_evidence_from_alpha(alpha: torch.Tensor, s: float,
                            mask: torch.Tensor | None = None,
                            eps: float = 1e-8,
                            with_scaling: bool=True, scaling_force: float=1.0,
                            one_sided: bool = True, gate_width: float = 0.05  # gate_width ~ e.g., 0.05-0.15 of s
                            ) -> torch.Tensor:
    """
    Evidence prior: KL( Dir(alpha) || Dir(alpha_prior) ) with alpha_prior = s * p_hat,
    where p_hat = alpha / alpha0. This penalizes the total evidence alpha0 toward s
    while keeping the mean p_hat unchanged. KL only penalizes the magnitude and does not backprop through the prior.
    It prevents the KL from trying to reshape the mean and keeps it focused on the total evidence alpha0.
    
    If with_scaling=True, multiply per-pixel KL by (alpha0/s)^scaling_force for alpha0 > s (scaling_force=1),
    which strengthens the pull when evidence is too large.
    
    KL acts both ways (it will try to push up when a0 < s and down when a0 > s). 
    If you want a one-sided penalty (only act when a0 > s), add a gate with smooth transition with one_sided=True and add a gate width.
    
    If mask is provided (bool [B,H,W]), return masked mean; else global mean.
    """
    # total evidence
    a0    = alpha.sum(dim=1, keepdim=True) + eps
    
    # prior keeps the mean fixed; only total scale is penalized
    with torch.no_grad():
        p_hat = alpha / a0
        alpha_prior = float(s) * p_hat
        
    kl_map = _kl_map(alpha, alpha_prior)  # [B,H,W]
    
    if one_sided:
        # soft gate ~ 0 below s, ~1 above s
        width = gate_width * float(s)
        g = torch.sigmoid((a0 - float(s)) / (width + eps)).squeeze(1)  # [B,H,W]
        kl_map = kl_map * g
        
    if with_scaling:
        # amplify only when a0 > s; scaling_momentum=1.0 (adjust if you want stronger effect with scaling_force>1)
        scale = (a0.squeeze(1) / (float(s) + eps)).clamp_min(1.0).pow(scaling_force)
        kl_map = kl_map * scale

    if mask is None:
        return kl_map.mean()

    m = mask.float()
    return (kl_map * m).sum() / m.sum().clamp_min(1.0)

def kl_sym_from_alpha(alpha: torch.Tensor, c: float,
                      mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Symmetric prior: KL( Dir(alpha) || Dir(c * 1) ).
    Penalizes both mean shift (toward uniform) and evidence magnitude.
    If mask is provided (bool [B,H,W]), return masked mean; else global mean.
    """
    alpha_prior = torch.full_like(alpha, c)
    kl_map = _kl_map(alpha, alpha_prior)
    if mask is None:
        return kl_map.mean()
    return (kl_map * mask.float()).sum() / mask.float().sum().clamp_min(1.0)

def info_reg_from_alpha(alpha: torch.Tensor, target: torch.Tensor,
                        ignore_index: int | None = 0) -> torch.Tensor:
    """
    Information regularizer (Tsiligkaridis):
      R = 0.5 * sum_{j != c} (alpha_j - 1)^2 * [ trigamma(alpha_j) - trigamma(tilde_alpha0) ]
    where tilde_alpha0 = 1 + sum_{k != c} alpha_k.
    Returns masked mean scalar.
    """
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0]
    valid = _valid_mask(target, ignore_index)
    if valid.sum() == 0:
        return alpha.sum() * 0.0

    B, C, H, W = alpha.shape
    oh = torch.zeros((B, C, H, W), device=alpha.device, dtype=torch.bool)
    oh.scatter_(1, target.unsqueeze(1), True)   # one-hot ground truth
    mask_incorrect = ~oh
    tilde_a0 = 1.0 + (alpha * mask_incorrect.float()).sum(dim=1, keepdim=True)
    term = (alpha - 1.0)**2 * (polygamma(1, alpha) - polygamma(1, tilde_a0))
    per_pix = 0.5 * (term * mask_incorrect.float()).sum(dim=1)  # [B,H,W]
    return (per_pix * valid.float()).sum() / valid.sum()

# ----------------- single lightweight wrapper -----------------

class DirichletCriterion(nn.Module):
    """
    One instance, alpha-only API.
    You pass weights for whichever terms you want; terms with w == 0.0 are NOT computed.
    """
    def __init__(self, num_classes: int, ignore_index: int | None = 0,
                 eps: float = 1e-8, prior_concentration: float = 30.0,
                 p_moment: float = 2.0, kl_mode: str = "evidence", smoothing=0.25, nll_mode="density", with_acc_classWeights= False,
                 comp_gamma: float=2.0, comp_tau: float=0.6, comp_sigma: float=0.1, comp_normalize: bool=True, ema_momentum: float=0.99):
        super().__init__()
        assert kl_mode in ("evidence", "symmetric")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps
        self.prior_concentration = float(prior_concentration)
        self.p_moment = float(p_moment)
        self.kl_mode = kl_mode
        self.smoothing = smoothing
        self.nll_mode = nll_mode # | "density" or "dircat"
        self.with_acc_classWeights = with_acc_classWeights
        # comp parameters
            # - gamma (default 2.0): power on (1 - p_y). Bigger gamma focuses the penalty on very uncertain pixels (small p_y).
            # - tau (default 0.6): center of the sigmoid gate on p_y. The term ramps up when p_y < tau. 
            #     0.6 means "start engaging when top-class prob drops below ~60%".
            # - sigma (default 0.1): slope of that sigmoid ramp. Smaller sigma = sharper switch. 0.1 gives a smooth but decisive transition.
            # - s_target (default = prior_concentration): evidence target used by evidence-KL. 
            #     Using the same value here makes the complement term small when alpha0 >> s_target (i.e., the model already has strong evidence).
            # - normalize (default True): divides by log(C-1) so the term lies in [0,1]. This makes your global weight w_comp interpretable and stable across different C.
        self.comp_gamma = comp_gamma
        self.comp_tau = comp_tau
        self.comp_sigma = comp_sigma
        self.comp_normalize = comp_normalize
        self.ema_momentum = ema_momentum
        self.register_buffer("ema_counts", torch.zeros(num_classes, dtype=torch.float32))
        self.register_buffer("class_weights", torch.ones(num_classes, dtype=torch.float32))
    
    @torch.no_grad()
    def update_class_weights(self, labels, method="effective_num", beta=0.999,
                             clip_min=0.2, clip_max=5.0):
        # batch counts (int64!) from labels
        if labels.dim() == 4 and labels.size(1) == 1:
            labels = labels[:, 0]
        flat = labels.reshape(-1)
        if self.ignore_index is not None:
            flat = flat[flat != self.ignore_index]
        flat = flat.to(torch.int64)

        counts = torch.bincount(flat, minlength=self.num_classes).to(self.ema_counts)

        # EMA update on counts (float32)
        m = self.ema_momentum
        self.ema_counts.mul_(m).add_(counts.to(self.ema_counts.dtype), alpha=1.0 - m)

        # weights from EMA counts
        self.class_weights = compute_class_weights_from_counts(
            self.ema_counts, method=method, beta=beta,
            clip_min=clip_min, clip_max=clip_max
        )
        
    def forward(self, alpha: torch.Tensor, target: torch.Tensor,
                w_dce: float = 0.0, w_nll: float = 0.0, w_imax: float = 0.0,
                w_ir: float = 0.0, w_kl: float = 0.0, w_comp: float=0.0) -> tuple[torch.Tensor, dict]:
        """
        Returns (total_loss, dict_of_terms). Only computes terms with weight > 0.
        """
        # valid mask used by all target-dependent terms and to mask KL if desired
        if target.dim() == 4 and target.size(1) == 1:
            tgt_hw = target[:, 0]
        else:
            tgt_hw = target
        valid = _valid_mask(tgt_hw, self.ignore_index)

        terms = {}
        total = alpha.sum() * 0.0  # zero on correct device/dtype

        if w_dce > 0.0:
            val = dce_from_alpha(alpha, target, self.ignore_index)
            terms["dce"] = val; total = total + w_dce * val
        
        if w_nll > 0.0:
            if self.nll_mode == "dircat":
                if self.with_acc_classWeights:
                    val = nll_dirichlet_categorical_weighted(alpha, target, self.class_weights, ignore_index=self.ignore_index)
                else:
                    val = nll_dirichlet_categorical(alpha, target, self.ignore_index, self.eps)
            else:  # "density"
                val = nll_from_alpha(alpha, target, self.num_classes, self.smoothing,
                                     self.ignore_index, self.eps)
            terms["nll"] = val; total = total + w_nll * val

        if w_imax > 0.0:
            val = imax_from_alpha(alpha, target, self.p_moment, self.ignore_index)
            terms["imax"] = val; total = total + w_imax * val

        if w_ir > 0.0:
            val = info_reg_from_alpha(alpha, target, self.ignore_index)
            terms["ir"] = val; total = total + w_ir * val

        if w_kl > 0.0:
            if self.kl_mode == "evidence":
                # Mask KL by valid pixels so unlabeled regions do not bias it.
                val = kl_evidence_from_alpha(alpha, self.prior_concentration, mask=valid, eps=self.eps, 
                                            with_scaling=True, scaling_force=1.0, one_sided=True, gate_width=0.1)
            else:
                val = kl_sym_from_alpha(alpha, self.prior_concentration, mask=valid)
            terms["kl"] = val; total = total + w_kl * val
        
        if w_comp > 0.0:
            val = complement_kl_uniform_from_alpha(
                alpha, target, ignore_index=self.ignore_index, eps=self.eps,
                gamma=self.comp_gamma, tau=self.comp_tau, sigma=self.comp_sigma,
                s_target=None, normalize=self.comp_normalize   # s_target=self.prior_concentration or None
            )
            terms["comp"] = val; total = total + w_comp * val

        return total, terms

import math

@torch.no_grad()
def compute_class_weights_from_counts(
    counts: torch.Tensor,
    method: str = "effective_num",
    beta: float = 0.999,
    clip_min: float = 0.2,
    clip_max: float = 5.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    counts: [C] float or int, non-negative
    returns weights [C], mean over seen classes == 1.0
    """
    counts = counts.to(dtype=torch.float32)
    C = counts.numel()
    seen = counts > 0

    if method == "effective_num":
        # Cui et al. CVPR'19  w_c ∝ (1 - beta) / (1 - beta^{n_c})
        eff = 1.0 - torch.pow(torch.full_like(counts, beta), counts)
        w = (1.0 - beta) / (eff + eps)
    elif method == "inv_sqrt":
        w = 1.0 / torch.sqrt(counts + eps)
    elif method == "inv":
        w = 1.0 / (counts + eps)
    elif method == "median":
        w = torch.zeros_like(counts)
        if seen.any():
            med = counts[seen].median()
            w[seen] = med / (counts[seen] + eps)
    else:
        raise ValueError(f"Unknown method: {method}")

    # unseen classes -> weight 0
    w = torch.where(seen, w, torch.zeros_like(w))

    # normalize seen-class weights to mean 1
    if seen.any():
        w_seen_mean = w[seen].mean()
        w[seen] = w[seen] / (w_seen_mean + eps)

    # clamp to avoid extremes
    w.clamp_(clip_min, clip_max)
    return w

@torch.no_grad()
def compute_class_weights_from_labels(
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
    method: str = "effective_num",   # "effective_num" | "inv_sqrt" | "inv" | "median"
    beta: float = 0.999,             # used for "effective_num"
    clip_min: float = 0.2,
    clip_max: float = 5.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build class weights [C] from label indices [B,H,W] or [B,1,H,W].
    Mean(weight over seen classes) == 1.0. Unseen classes -> weight 0.
    """
    if labels.dim() == 4 and labels.size(1) == 1:
        labels = labels[:, 0]                 # [B,H,W]
    flat = labels.reshape(-1)

    if ignore_index is not None:
        flat = flat[flat != ignore_index]

    # IMPORTANT: bincount needs integer indices
    flat = flat.to(torch.int64)

    counts = torch.bincount(flat, minlength=num_classes).to(
        device=labels.device, dtype=torch.float32
    )
    return compute_class_weights_from_counts(
        counts, method=method, beta=beta,
        clip_min=clip_min, clip_max=clip_max, eps=eps
    )


# ============================================
# Complement KL-to-Uniform regularizer (non-negative)
# Acts on off-class conditional distribution to encourage uncertainty spread
# ============================================

def complement_kl_uniform_from_alpha(
    alpha: torch.Tensor, target: torch.Tensor,
    ignore_index: int = 0, eps: float = 1e-8,
    gamma: float = 2.0, tau: float = 0.6, sigma: float = 0.1,
    s_target: float | None = None, normalize: bool = True
) -> torch.Tensor:
    """
    Purpose
    -------
    Encourage uncertainty *spread* across off-classes when the model is unsure (low confidence i.e. p_y on true class).
    We minimize a gated KL divergence between the off-class conditional
    distribution and the uniform distribution over off-classes.

    Shapes
    ------
    alpha : [B,C,H,W], C >= 2, alpha_i > 0
    target: [B,H,W] or [B,1,H,W], integer labels in {0..C-1} or ignore_index
    returns: scalar tensor (mean over valid pixels)

    Definitions (ASCII math)
    ------------------------
    alpha0  = sum_i alpha_i
    p       = alpha / alpha0               # Dirichlet mean, p_i in (0,1), sum_i p_i = 1
    y       = ground-truth class index
    S       = 1 - p_y                      # total off-class mass
    tilde_p_j = p_j / S for j != y         # conditional over off-classes, sum_{j!=y} tilde_p_j = 1
    U_j     = 1 / (C - 1)                  # uniform over off-classes

    KL(tilde_p || U) = sum_{j!=y} tilde_p_j * log( tilde_p_j / U_j )
                     = sum_{j!=y} tilde_p_j * log(tilde_p_j) + log(C-1)
      range: [0, log(C-1)], 0 at tilde_p = Uniform

    Gating (when should the regularizer act?)
    -----------------------------------------
    We scale the term by w_total = w_uncert * w_evid, where

      w_uncert(p_y) = (1 - p_y)^gamma * sigmoid( (tau - p_y) / sigma )
        - (1 - p_y)^gamma        : polynomial emphasis on low p_y
        - sigmoid(...)           : soft step that turns on near p_y = tau
        - gamma (>=1)            : bigger -> focus more on very uncertain pixels
        - tau (in (0,1))         : larger -> turns on *earlier* (acts at higher p_y), turns on gate once p_y drops below around tau
        - sigma (>0)             : smaller -> sharper step around tau

      w_evid(alpha0) = s_target / (alpha0 + s_target)   (optional)
        - downweights when evidence alpha0 is already high
        - monotone: alpha0->0 => w_evid->1 ; alpha0->inf => w_evid->0

    Loss (per pixel) and averaging
    ------------------------------
    per_pix = w_total * KL_norm(tilde_p || U)
      where KL_norm = KL / log(C-1) if normalize=True, otherwise raw KL
    Output = mean(per_pix over valid pixels)

    Effects
    -------
    - Minimizing KL pushes the largest off-class tilde_p_j down and the smallest up,
      i.e., spreads off-class mass toward uniform when uncertainty is high.
    - Scale-invariant in alpha (depends on p only), so it does not inflate alpha0.
    - The optional evidence gate w_evid further suppresses the term when alpha0 is large.

    Parameter intuition (quick)
    ---------------------------
      gamma: 1.0 (broad), 2.0 (default), 3.0 (narrow to very low p_y)
      tau  : 0.6 (acts late), 0.7 (earlier), 0.8 (quite early; caution)
      sigma: 0.15 (soft ramp), 0.10 (crisp), 0.06 (very sharp; may jitter)
    Example strengths (gamma=2, tau=0.6, sigma=0.1; ignoring w_evid):
      p_y=0.90 -> w_uncert ~ 0.00047  (almost off)
      p_y=0.70 -> w_uncert ~ 0.024    (light)
      p_y=0.60 -> w_uncert ~ 0.080    (moderate)
      p_y=0.50 -> w_uncert ~ 0.183    (strong)
      p_y=0.30 -> w_uncert ~ 0.468    (very strong)

    Implementation
    --------------
    - We build a safe target index 'tgt_safe' for gather so ignore_index outside [0..C-1]
      never gets used for indexing.
    - 'valid' mask excludes ignore_index from the final average.
    - KL is normalized to [0,1] if normalize=True, making w_comp easier to tune.
    """

    # -- unify target shape and dtype --
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0]                 # [B,H,W]
    target = target.long()

    # -- valid pixels (exclude ignore_index) --
    valid = _valid_mask(target, ignore_index)  # [B,H,W] bool
    if valid.sum() == 0:
        return alpha.sum() * 0.0               # same device/dtype zero

    B, C, H, W = alpha.shape
    if C <= 2:
        # no complement distribution exists in binary case
        return alpha.sum() * 0.0

    # -- predictive mean probs p = alpha / alpha0 --
    a0 = alpha.sum(dim=1, keepdim=True) + eps     # [B,1,H,W]
    p  = alpha / a0                                # [B,C,H,W]

    # -- safe gather index: replace ignored labels by 0 to avoid OOB gather --
    tgt_safe = torch.where(valid, target, torch.zeros_like(target))   # [B,H,W]

    # p_y = gather p at GT class; clamp for log stability later
    py = p.gather(1, tgt_safe.unsqueeze(1)).clamp_min(eps)            # [B,1,H,W]

    # -- build off-class conditional tilde_p over C-1 classes --
    # mask out the GT channel
    oh = torch.zeros((B, C, H, W), device=p.device, dtype=torch.bool)
    oh.scatter_(1, tgt_safe.unsqueeze(1), True)        # one-hot: True at GT channel
    p_off = p.masked_fill(oh, 0.0)                     # zero GT prob
    S = (1.0 - py).clamp_min(eps)                      # total off-class mass
    tilde = p_off / S                                  # conditional probs, sum over j!=y equals 1

    # -- KL(tilde || U), U_j = 1/(C-1) --
    # KL = sum tilde * log(tilde / U) = sum tilde*log(tilde) + log(C-1)
    log_tilde = (tilde.clamp_min(eps)).log()
    kl_u = (tilde * log_tilde).sum(dim=1) + math.log(C - 1)    # [B,H,W], >= 0

    # optionally normalize KL to [0,1] by dividing by log(C-1)
    if normalize:
        kl_u = kl_u / math.log(C - 1)

    # -- gating by uncertainty and optional evidence --
    # w_uncert = (1 - p_y)^gamma * sigmoid((tau - p_y)/sigma)
    w_uncert = (1.0 - py).pow(gamma).squeeze(1) * torch.sigmoid((tau - py) / sigma).squeeze(1)

    # evidence gate w_evid = s / (alpha0 + s) in (0,1], monotone decreasing in alpha0
    if s_target is not None:
        s_val = float(s_target)
        w_evid = s_val / (a0.squeeze(1) + s_val)
        w = w_uncert * w_evid
    else:
        w = w_uncert

    # -- masked mean over valid pixels --
    per_pix = w * kl_u                                   # [B,H,W], >= 0
    return (per_pix * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

import math
import torch

def evidence_logspring(alpha: torch.Tensor,
                       s_per_class: float = 6.0,
                       window: float = 0.25,
                       eps: float = 1e-8) -> torch.Tensor:
    """
    Piecewise-quadratic spring on log(alpha0) with a dead-zone:
      a0 = sum_i alpha_i
      s  = C * s_per_class
      s_hi = s*(1+window), s_lo = s/(1+window)
      loss = [max(0, log(a0)-log(s_hi))]^2 + [max(0, log(s_lo)-log(a0))]^2
    This ONLY touches the scale (alpha0), not p = alpha/alpha0.
    """
    B, C, H, W = alpha.shape
    a0 = alpha.sum(dim=1, keepdim=True) + eps
    s  = C * float(s_per_class)
    s_hi = s * (1.0 + window)
    s_lo = s / (1.0 + window)

    log_a0 = a0.log()
    over   = torch.clamp(log_a0 - math.log(s_hi + eps), min=0.0)
    under  = torch.clamp(math.log(s_lo + eps) - log_a0, min=0.0)
    return (over**2 + under**2).mean()


def brier_dirichlet(alpha: torch.Tensor,
                    target: torch.Tensor,
                    ignore_index: int | None = None,
                    eps: float = 1e-12) -> torch.Tensor:
    """
    Expected Brier score under a Dirichlet predictive distribution.

    Math:
      For p ~ Dir(alpha), with alpha0 = sum_i alpha_i and p_hat_i = alpha_i / alpha0:
        E[p_i]   = p_hat_i
        E[p_i^2] = alpha_i (alpha_i + 1) / (alpha0 (alpha0 + 1))
                 = (alpha0 * p_hat_i^2 + p_hat_i) / (alpha0 + 1)

      Brier(target=y) = sum_i (p_i - y_i)^2
      E[Brier] = sum_i E[p_i^2] - 2 E[p_y] + 1

    Properties:
      * Proper scoring rule; improves probability calibration.
      * Very weak dependency on alpha0; does NOT push alpha0 to blow up.
      * Complements NLL (fit) and comp loss (confident-wrong suppression).
    """

    # --- INPUT SHAPES ---------------------------------------------------------
    # alpha : [B, C, H, W]   Dirichlet concentrations (> 0)
    # target: [B, H, W] or [B, 1, H, W] with integer labels in {0..C-1}
    # ignore_index: label value to mask from loss. Use None to disable masking.

    # --- NORMALIZE TARGET SHAPE ----------------------------------------------
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0]
    target = target.long()

    # --- BUILD VALID MASK -----------------------------------------------------
    if ignore_index is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        valid = (target != ignore_index)
    if valid.sum() == 0:
        # Return a zero tensor on the right device/dtype if nothing is valid
        return alpha.sum() * 0.0

    # --- COMPUTE p_hat = alpha / alpha0 --------------------------------------
    a0 = alpha.sum(dim=1, keepdim=True) + eps            # [B,1,H,W]
    p_hat = alpha / a0                                   # [B,C,H,W]

    # --- SUM OF E[p_i^2] OVER CLASSES ----------------------------------------
    # sum_i E[p_i^2] = (alpha0 * sum_i p_hat_i^2 + 1) / (alpha0 + 1)
    sum_ep2 = (a0 * (p_hat.pow(2).sum(dim=1, keepdim=True)) + 1.0) / (a0 + 1.0)

    # --- E[p_y] ---------------------------------------------------------------
    ep_y = p_hat.gather(1, target.unsqueeze(1))          # [B,1,H,W]

    # --- EXPECTED BRIER PER PIXEL --------------------------------------------
    per_pix = (sum_ep2 - 2.0 * ep_y + 1.0).squeeze(1)    # [B,H,W]

    # --- MEAN OVER VALID PIXELS ----------------------------------------------
    w = valid.float()
    return (per_pix * w).sum() / w.sum().clamp_min(1.0)

