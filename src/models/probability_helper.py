# probability_helper.py (safe for multi-process/DataLoader)
import math
from typing import Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.special import digamma
from scipy import special as scispecial

from utils.agg import mean_aggregator


# --------- pyplot loader (no top-level import, safe backend) ---------
def _get_pyplot(backend: str = "Agg"):
    """
    Import matplotlib.pyplot with a safe non-GUI backend.
    Call this ONLY inside functions that need plotting.
    """
    import matplotlib
    if matplotlib.get_backend().lower() != backend.lower():
        matplotlib.use(backend, force=True)
    import matplotlib.pyplot as plt
    return plt


# ---------------- Label smoothing & Dirichlet utilities ----------------

def smooth_one_hot(targets: torch.Tensor, num_classes: int, smoothing: float = 0.1) -> torch.Tensor:
    """
    targets: [B,1,H,W] -> returns [B,C,H,W] (float) with label smoothing.
    """
    confidence = 1.0 - smoothing
    low_conf = smoothing / (num_classes - 1)

    B, _, H, W = targets.shape
    one_hot = torch.full((B, num_classes, H, W), low_conf, device=targets.device, dtype=torch.float)
    targets = targets.squeeze(1)
    one_hot.scatter_(1, targets.unsqueeze(1), confidence)
    return one_hot


def to_alpha_concentrations(predicted_logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert logits [B,C,H,W] -> alpha > 0 via softplus + 1.
    """
    return torch.nn.functional.softplus(predicted_logits).clamp_min(eps) + 1.0


def alphas_to_Dirichlet(alpha: torch.Tensor) -> torch.distributions.Dirichlet:
    """
    alpha: [B,C,H,W] -> Dirichlet over last dim (returns dist on [B,H,W,C]).
    """
    return torch.distributions.Dirichlet(alpha.permute(0, 2, 3, 1))


# ---------------- Uncertainty measures (Dirichlet) ----------------

def get_predictive_entropy(alpha: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    H(E[p]) = -sum_j (α_j/α₀) log(α_j/α₀)  -> [B,H,W]
    """
    alpha_0 = alpha.sum(dim=1, keepdim=True) + eps
    exp_p = alpha / alpha_0
    return -(exp_p * torch.log(exp_p + eps)).sum(dim=1)

@mean_aggregator()
def get_predictive_entropy_norm(alpha: torch.Tensor, eps: float = 1e-10):
    assert alpha.dim() ==4, "Requires alpha to be of shape [B,C,H,W]"
    C = alpha.shape[1]
    return get_predictive_entropy(alpha, eps) / np.log(C)

def get_aleatoric_uncertainty(alpha: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    AU = -sum_j (α_j/α₀) [ψ(α_j+1) − ψ(α₀+1)] -> [B,H,W]
    """
    alpha_0 = alpha.sum(dim=1, keepdim=True) + eps
    term = digamma(alpha + 1) - digamma(alpha_0 + 1)
    exp_p = alpha / alpha_0
    return -(exp_p * term).sum(dim=1)

def get_aleatoric_uncertainty_norm(alpha: torch.Tensor, eps: float = 1e-10):
    """normalization: maximum uncertainty with "flattest" Dirichlet, which is at α_j=1 ∀j:
    AU = -∑_j (α_j / α₀) * [ψ(α_j + 1) − ψ(α₀ + 1)]
    at α_j=1 ∀j -> AU_max = -∑_j (1/K) * [ψ(1 + 1) − ψ(K + 1)] = -ψ(2) + ψ(K + 1)
    """
    assert alpha.dim() ==4, "Requires alpha to be of shape [B,C,H,W]"
    C = alpha.shape[1]
    return get_aleatoric_uncertainty(alpha, eps) / (digamma(torch.tensor(C+1.)) - digamma(torch.tensor(2.)))

def get_epistemic_uncertainty(alpha: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    I = H(E[p]) − AU  -> [B,H,W]
    """
    return get_predictive_entropy(alpha) - get_aleatoric_uncertainty(alpha)

def get_epistemic_uncertainty_norm(alpha: torch.Tensor, eps: float = 1e-10):
    """normalization: maximum uncertainty with "flattest" Dirichlet, which is at α_j=1 ∀j:
    as EU = H - AU -> EU_max = ln K − [ψ(K+1) − ψ(2)] 
    """
    assert alpha.dim() ==4, "Requires alpha to be of shape [B,C,H,W]"
    C = alpha.shape[1]
    return get_epistemic_uncertainty(alpha, eps) / (np.log(C) - ( digamma(torch.tensor(C+1.)) - digamma(torch.tensor(2.)) ) )

def get_dirichlet_uncertainty_imgs(alpha: torch.Tensor, idx: int=0):
    unc_tot = get_predictive_entropy_norm(alpha)[idx, ...]
    unc_a = get_aleatoric_uncertainty_norm(alpha)[idx, ...]
    unc_e = get_epistemic_uncertainty_norm(alpha)[idx, ...]
    
    import cv2
    colormap_unc = cv2.COLORMAP_TURBO
    unc_tot_img = cv2.applyColorMap(np.uint8(255*np.maximum(unc_tot , 0.0)), colormap=colormap_unc)
    unc_a_img   = cv2.applyColorMap(np.uint8(255*np.maximum(unc_a   , 0.0)), colormap=colormap_unc)
    unc_e_img   = cv2.applyColorMap(np.uint8(255*np.maximum(unc_e   , 0.0)), colormap=colormap_unc)
    
    return unc_tot_img, unc_a_img, unc_e_img

# ---------------- Reliability & calibration helpers ----------------

def compute_mc_reliability_bins(
    alpha: torch.Tensor,
    y_true: torch.Tensor,
    n_bins: int = 10,
    n_samples: int = 64,
    eps: float = 1e-10
):
    """
    MC confidence vs empirical accuracy binning for Dirichlet model.
    Returns hits, totals (np arrays on CPU).
    """
    B, C, H, W = alpha.shape
    dist = alphas_to_Dirichlet(alpha)                  # [B,H,W,C]
    samples = dist.sample((n_samples,))                # [M,B,H,W,C]
    max_samples = samples.argmax(dim=-1)               # [M,B,H,W]

    y_expanded = y_true.squeeze(1).unsqueeze(0).expand_as(max_samples)
    correct_emp = (max_samples == y_expanded).float().mean(dim=0)  # [B,H,W]
    conf_mc = correct_emp.flatten().clamp(min=eps, max=1.0 - eps)  # [N]

    probs_mean = alpha / alpha.sum(dim=1, keepdim=True)
    pred_1shot = probs_mean.argmax(dim=1).flatten()                # [N]
    correct_1shot = (pred_1shot == y_true.flatten()).float()

    bin_edges = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=alpha.device)
    bin_ids = torch.bucketize(conf_mc, bin_edges, right=False) - 1
    bin_ids = bin_ids.clamp_(0, n_bins - 1)

    totals = torch.zeros(n_bins, device=alpha.device)
    hits = torch.zeros(n_bins, device=alpha.device)
    ones = torch.ones_like(conf_mc)

    totals.scatter_add_(0, bin_ids, ones)
    hits.scatter_add_(0, bin_ids, correct_1shot)

    return hits.cpu().numpy(), totals.cpu().numpy()


def save_reliability_diagram(
    empirical_acc: np.ndarray,
    bin_centers:   np.ndarray,
    tot_counts:    np.ndarray,
    output_path:   str = "reliability_diagram.png",
    title:         str = 'Reliability diagram\n(dot area ∝ #pixels per confidence bin — sharpness)',
    xlabel:        str = 'Predicted confidence',
    ylabel:        str = 'Empirical accuracy',
    show:          bool = False,
    dpi:           int = 300,
):
    """
    Save a reliability diagram (no GUI). Safe in worker processes.
    """
    if len(bin_centers) == 0 or (tot_counts.max() if len(tot_counts) else 0) == 0:
        return
    plt = _get_pyplot("Agg")

    sizes = (tot_counts / tot_counts.max()) * 1000.0
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Perfect calibration')
    ax.scatter(bin_centers, empirical_acc, s=sizes, alpha=0.7, edgecolors='k', label='Empirical reliability')
    ax.fill_between(bin_centers, empirical_acc, 0, alpha=0.2)

    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc='upper left', markerscale=0.3)
    ax.grid(True, linestyle=':', linewidth=0.5)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ---------------- IoU between error and entropy-threshold mask ----------------

def compute_entropy_error_iou(
    entropy_norm:     torch.Tensor,
    error_mask:       torch.Tensor,
    thresholds:       Union[Sequence[float], torch.Tensor]
) -> torch.Tensor:
    """
    IoU between high-entropy mask and error mask for each threshold.
    entropy_norm: [H,W] or [B,H,W] in [0,1]
    error_mask:   same shape, 0/1
    returns: [T] (or [B,T])
    """
    device = entropy_norm.device
    thresholds = torch.as_tensor(thresholds, device=device, dtype=entropy_norm.dtype)

    if entropy_norm.ndim == 2:
        entropy = entropy_norm.unsqueeze(0)
        error   = error_mask.unsqueeze(0)
    else:
        entropy = entropy_norm
        error   = error_mask

    B, H, W = entropy.shape
    N = H * W
    entropy_flat = entropy.reshape(B, N)
    error_flat   = (error.reshape(B, N) > 0.5)

    T = thresholds.numel()
    thr = thresholds.view(1, T, 1)           # [1,T,1]
    e   = entropy_flat.view(B, 1, N)         # [B,1,N]
    pred = e > thr                            # [B,T,N]

    gt_err = error_flat.view(B, 1, N).expand_as(pred)
    inter = (pred & gt_err).sum(dim=2).to(torch.float32)  # [B,T]
    union = (pred | gt_err).sum(dim=2).to(torch.float32)  # [B,T]

    ious = torch.where(union > 0, inter / union, torch.zeros_like(inter))
    return ious.squeeze(0)


def plot_mIOU_errorEntropy(
    mean_ious: np.ndarray,
    thresholds: np.ndarray,
    output_path: str = "mean_iou_curve.png",
    show: bool = False,
    dpi: int = 300,
):
    """
    Save mean IoU vs entropy threshold curve (no GUI).
    """
    plt = _get_pyplot("Agg")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.set_ylim(0, 1); ax.set_xlim(0, 1)
    ax.plot(thresholds, mean_ious, marker='o', linestyle='-')
    ax.set_xlabel('Entropy threshold'); ax.set_ylabel('Mean IoU')
    ax.set_title('Mean IoU between Error Mask and Entropy-Threshold Mask')
    ax.grid(True, linestyle=':')
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ---------------- Entropy-as-error-probability reliability ----------------

def compute_entropy_reliability(
    entropy_norm: torch.Tensor,   # [B,H,W] in [0,1]
    error_mask:   torch.Tensor,   # [B,H,W] 0/1
    n_bins:       int = 10
):
    """
    Bin normalized entropy as predicted error prob.
    Returns (totals, errors, err_rate, ece).
    """
    B, H, W = entropy_norm.shape
    N = B * H * W
    h = entropy_norm.reshape(-1)
    e = error_mask.reshape(-1).float()

    edges   = torch.linspace(0.0, 1.0, n_bins + 1, device=h.device, dtype=h.dtype)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bins = torch.bucketize(h, edges, right=False).clamp(min=0, max=n_bins) - 1

    totals = torch.zeros(n_bins, device=h.device, dtype=h.dtype)
    errors = torch.zeros(n_bins, device=h.device, dtype=h.dtype)
    ones   = torch.ones_like(h)

    totals.scatter_add_(0, bins, ones)
    errors.scatter_add_(0, bins, e)

    err_rate = torch.where(totals > 0, errors / totals, torch.zeros_like(errors))
    ece = (totals / max(N, 1) * torch.abs(centers - err_rate)).sum()

    return (totals.detach().cpu().numpy(),
            errors.detach().cpu().numpy(),
            err_rate.detach().cpu().numpy(),
            float(ece))


# ---------------- Brier decomposition & variants ----------------

def _per_class_decomp(
    p_flat: torch.Tensor,
    y_flat: torch.Tensor,
    n_bins: int = 10,
    eps: float = 1e-10,
    with_binning: bool = True
):
    """
    Per-class Brier decomposition: returns BS, R, Z, U.
    """
    N = p_flat.size(0)
    y_bar = y_flat.mean()
    U = y_bar * (1 - y_bar)
    BS = torch.mean((p_flat - y_flat) ** 2)

    R = torch.tensor(0.0, device=p_flat.device)
    Z = torch.tensor(0.0, device=p_flat.device)

    if with_binning:
        bin_edges = torch.linspace(0, 1, n_bins + 1, device=p_flat.device, dtype=p_flat.dtype)
        bins = torch.bucketize(p_flat, bin_edges, right=False).clamp(min=0, max=n_bins) - 1

        n_b = torch.bincount(bins, minlength=n_bins).float().clamp(min=eps)
        sum_p = torch.zeros(n_bins, device=p_flat.device, dtype=p_flat.dtype).scatter_add_(0, bins, p_flat)
        sum_y = torch.zeros(n_bins, device=p_flat.device, dtype=p_flat.dtype).scatter_add_(0, bins, y_flat)

        p_b = sum_p / n_b
        y_b = sum_y / n_b

        R = torch.dot(n_b, (p_b - y_b) ** 2) / N
        Z = torch.dot(n_b, (y_b - y_bar) ** 2) / N

    return BS, R, Z, U


def brier_mc_decomp(
    alpha: torch.Tensor,
    y_true: torch.Tensor,
    n_samples: int = 64,
    use_exp: bool = False,
    with_binning: bool = True,
    n_bins: int = 10,
    eps: float = 1e-10
):
    """
    MC-based Brier decomposition via Dirichlet samples (or expectation if use_exp).
    Returns (totals_dict, factor).
    """
    B, C, H, W = alpha.shape
    N = B * H * W

    if use_exp:
        probs = alpha / alpha.sum(dim=1, keepdim=True)                  # [B,C,H,W]
        p_flat_perClass = probs.permute(0, 2, 3, 1).reshape(-1, C)      # [N,C]
        factor = N
        reps = 1
    else:
        dist = alphas_to_Dirichlet(alpha)
        samples = dist.sample((n_samples,))                             # [M,B,H,W,C]
        p_flat_perClass = samples.permute(0, 1, 3, 4, 2).reshape(-1, C) # [M*N,C]
        factor = N * n_samples
        reps = n_samples

    y = y_true.long().squeeze(1)                                        # [B,H,W]
    y_flat_perClass = (
        F.one_hot(y, num_classes=C)
        .view(B, H, W, C)
        .unsqueeze(0).expand(reps, B, H, W, C)
        .reshape(-1, C)
        .float()
    )

    totals = {'brier': 0., 'reliability': 0., 'resolution': 0., 'uncertainty': 0.}
    per_class = []
    for k in range(C):
        BS, R, Z, U = _per_class_decomp(p_flat_perClass[:, k], y_flat_perClass[:, k],
                                        n_bins=n_bins, eps=eps, with_binning=with_binning)
        totals['brier']       += BS
        totals['reliability'] += R
        totals['resolution']  += Z
        totals['uncertainty'] += U
        per_class.append({
            'BS': float(BS.item()),
            'reliability': float(R.item()),
            'resolution': float(Z.item()),
            'uncertainty': float(U.item()),
            'base_rate': float(y_flat_perClass[:, k].mean().item()),
        })
    totals['per_class'] = per_class
    return totals, factor


def brier_mc(alpha: torch.Tensor, y_true: torch.Tensor, n_samples: int = 64) -> torch.Tensor:
    """
    MC Brier score over Dirichlet samples. Returns scalar tensor.
    """
    B, C, H, W = alpha.shape
    dist = alphas_to_Dirichlet(alpha)
    samples = dist.sample((n_samples,)).permute(0, 1, 4, 2, 3)   # [M,B,C,H,W]
    y = y_true.long().squeeze(1)                                 # [B,H,W]
    y_oh = F.one_hot(y, num_classes=C).permute(0, 3, 1, 2).float()  # [B,C,H,W]
    y_oh = y_oh.unsqueeze(0).expand(n_samples, B, C, H, W)
    sq_err = (samples - y_oh) ** 2
    per_draw_bs = sq_err.sum(dim=2).mean(dim=(1, 2, 3))
    return per_draw_bs.mean()


# ---------------- Confidence measures & reliability curves ----------------

def mc_confidence_analytic(alpha: torch.Tensor, y_true: torch.Tensor, M: int = 64, eps: float = 1e-10) -> torch.Tensor:
    """
    conf = P(p_y > E[p_y]) via MC sampling; returns [B,1,H,W]
    """
    alpha_0 = alpha.sum(dim=1, keepdim=True).clamp_min(eps)
    alpha_y = torch.gather(alpha, dim=1, index=y_true)        # [B,1,H,W]

    dist = alphas_to_Dirichlet(alpha)
    samples = dist.sample((M,)).permute(0, 1, 4, 2, 3)        # [M,B,C,H,W]

    p_hat_y = alpha_y / alpha_0                                # [B,1,H,W]
    y_expand = y_true.unsqueeze(0).expand(M, -1, -1, -1, -1)   # [M,B,1,H,W]
    samples_gt = samples.gather(dim=2, index=y_expand)         # [M,B,1,H,W]

    conf = (samples_gt > p_hat_y.unsqueeze(0)).float().mean(0) # [B,1,H,W]
    return conf


def beta_confidence_analytic(alpha: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Exact conf = P(p_y > E[p_y]) via Beta marginal CDF; returns [B,1,H,W]
    """
    alpha_0 = alpha.sum(dim=1, keepdim=True).clamp_min(eps)
    alpha_y = torch.gather(alpha, dim=1, index=y_true)           # [B,1,H,W]
    p_hat_y = alpha_y / alpha_0                                  # [B,1,H,W]
    beta_b  = (alpha_0 - alpha_y).clamp_min(eps)                 # [B,1,H,W]

    a = alpha_y.squeeze(1).detach().cpu().numpy()
    b = beta_b.squeeze(1).detach().cpu().numpy()
    x = p_hat_y.squeeze(1).detach().cpu().numpy()

    cdf_at_mean = scispecial.betainc(a, b, x)                    # regularized incomplete beta
    conf = (1.0 - cdf_at_mean)[:, None, ...]                     # [B,1,H,W]
    return torch.from_numpy(conf).to(alpha.device, dtype=alpha.dtype)


def reliability_dirichlet(alpha: torch.Tensor, y_true: torch.Tensor, coverages, n_samples: int = 64, device: str = "cpu"):
    """
    Returns {'avg_RLS','min_RLS','F_hat','bins'} for MC-based reliability CDF.
    """
    alpha = alpha.to(device)
    y_true = y_true.to(device)
    conf = mc_confidence_analytic(alpha, y_true, M=n_samples)
    if not isinstance(conf, torch.Tensor):
        conf = torch.tensor(conf, device=device)

    conf = conf.flatten()
    bins = torch.as_tensor(coverages, dtype=conf.dtype, device=conf.device)
    F_hat = torch.stack([(conf <= k).float().mean() for k in bins])
    rel_error = torch.abs(F_hat - bins)

    return {
        "avg_RLS": float((1 - rel_error.mean()).item() * 100.0),
        "min_RLS": float((1 - rel_error.max()).item() * 100.0),
        "F_hat":   F_hat.detach().cpu().numpy(),
        "bins":    bins.detach().cpu().numpy(),
    }


def compute_ece_and_reliability(
    alpha: torch.Tensor,
    y_true: torch.LongTensor,
    n_bins: int = 10,
    eps: float = 1e-10
):
    """
    From Dirichlet means: returns (bin_confidence, bin_accuracy, ECE).
    """
    B, C, H, W = alpha.shape
    N = B * H * W

    alpha_0 = alpha.sum(dim=1, keepdim=True).clamp_min(eps)
    mean_probs = alpha / alpha_0

    pred_conf_max, pred_labels = mean_probs.max(dim=1, keepdim=True)  # [B,1,H,W]
    pred_conf_max = pred_conf_max.flatten().clamp(min=eps, max=1 - eps)
    correct = (pred_labels == y_true).float().flatten()

    bin_edges = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=alpha.device, dtype=alpha.dtype)
    bin_ids = torch.bucketize(pred_conf_max, bin_edges, right=False) - 1
    bin_ids = bin_ids.clamp_(0, n_bins - 1)

    counts = torch.bincount(bin_ids, minlength=n_bins).float()
    sum_conf = torch.bincount(bin_ids, weights=pred_conf_max, minlength=n_bins)
    sum_correct = torch.bincount(bin_ids, weights=correct, minlength=n_bins)

    nonzero = counts > 0
    confs = torch.full((n_bins,), float("nan"), dtype=alpha.dtype, device=alpha.device)
    accs  = torch.full((n_bins,), float("nan"), dtype=alpha.dtype, device=alpha.device)
    confs[nonzero] = sum_conf[nonzero] / counts[nonzero]
    accs[nonzero]  = sum_correct[nonzero] / counts[nonzero]

    weights = counts / max(N, 1)
    ece = torch.nansum(weights * torch.abs(accs - confs)).item()

    return np.asarray(confs.detach().cpu()), np.asarray(accs.detach().cpu()), ece
