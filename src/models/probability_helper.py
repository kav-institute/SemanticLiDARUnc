import math
from typing import Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.special import digamma
from scipy import special as scispecial
from utils.agg import mean_aggregator

# --------- Global parameter definitions ----------
_NORM_MODE = "max"  # default "max" | "ref"
_EPS: float = 1e-8          # default eps to avoid zero division
_T: float = 1.0             # default alpha concnetration temperature scaling

def set_norm_mode(mode: str):
    global _NORM_MODE
    if mode not in ("max", "ref"):
        raise ValueError(f"norm_mode must be 'max' or 'ref', got: {mode}")
    _NORM_MODE = mode
def get_norm_mode() -> str:
    return _NORM_MODE

def set_eps_value(eps: float):
    global _EPS
    _EPS = eps
def get_eps_value() -> float:
    return _EPS

def set_alpha_temperature(T: float):
    global _T
    _T = T
def get_alpha_temperature() -> float:
    return _T

# --------- pyplot loader (safe, no GUI) ---------

def _get_pyplot(backend: str = "Agg"):
    import matplotlib
    if matplotlib.get_backend().lower() != backend.lower():
        matplotlib.use(backend, force=True)
    import matplotlib.pyplot as plt
    return plt

# ---------------- Label smoothing & Dirichlet utilities ----------------
def smoothing_schedule(epoch, E, s0=0.20, s_min=0.05, frac=0.4):
    import math
    t = min(epoch/(frac*E), 1.0)
    return s_min + (s0 - s_min) * 0.5 * (1 + math.cos(math.pi * t))

def smooth_one_hot(targets: torch.Tensor, num_classes: int, smoothing: float = 0.25) -> torch.Tensor:
    confidence = 1.0 - smoothing
    low_conf = smoothing / (num_classes - 1)
    B, H, W = targets.shape
    one_hot = torch.full((B, num_classes, H, W), low_conf, device=targets.device, dtype=torch.float)
    one_hot.scatter_(1, targets.unsqueeze(1), confidence)
    return one_hot

# ---------------- Logits -> Dirichlet ----------------
def to_alpha_concentrations(predicted_logits: torch.Tensor, T: float | None = None, eps: float | None = None) -> torch.Tensor:
    """Convert logits [B,C,H,W] -> alpha > 0 via softplus + 1; temperature T damps evidence."""
    if T is None:
        T = get_alpha_temperature()
    if eps is None:
        eps = get_eps_value()
    return torch.nn.functional.softplus(predicted_logits / T) + 1.0 + eps

def alphas_to_Dirichlet(alpha: torch.Tensor) -> torch.distributions.Dirichlet:
    return torch.distributions.Dirichlet(alpha.permute(0, 2, 3, 1))

####################################################################
# ---------------- Uncertainty measures (Dirichlet) ----------------
####################################################################

# Core uncertainties

def get_predictive_entropy(alpha: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    if eps is None:
        eps = get_eps_value()
    alpha0 = alpha.sum(dim=1, keepdim=True) + eps
    p_hat = alpha / alpha0
    return -(p_hat * torch.log(p_hat + eps)).sum(dim=1)


def get_aleatoric_uncertainty(alpha: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    if eps is None:
        eps = get_eps_value()
    alpha0 = alpha.sum(dim=1, keepdim=True) + eps
    term = digamma(alpha + 1.0) - digamma(alpha0 + 1.0)
    p_hat = alpha / alpha0
    return -(p_hat * term).sum(dim=1)


def get_epistemic_uncertainty(alpha: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    if eps is None:
        eps = get_eps_value()
    return get_predictive_entropy(alpha, eps) - get_aleatoric_uncertainty(alpha, eps)

# Normalized variants

    # Helpers
def _au_ref(C: int, device) -> torch.Tensor:
    return digamma(torch.tensor(float(C + 1), device=device)) - digamma(torch.tensor(2.0, device=device))

def _eu_span_ref(C: int, device) -> torch.Tensor:
    return torch.tensor(math.log(C), device=device) - _au_ref(C, device)

    # Predictive Uncertainty/Entropy NORM
@mean_aggregator()
def get_predictive_entropy_norm(alpha: torch.Tensor, eps: float | None = None):
    if eps is None:
        eps = get_eps_value()
    C = alpha.shape[1]
    return get_predictive_entropy(alpha, eps) / math.log(C)

    # Aleatoric Uncertainty NORM
def get_aleatoric_uncertainty_norm(alpha: torch.Tensor, eps: float | None = None, mode: str | None = None):
    if eps is None:
        eps = get_eps_value()
    C = alpha.shape[1]
    AU = get_aleatoric_uncertainty(alpha, eps)
    m = get_norm_mode() if mode is None else mode
    if m == "max":
        return (AU / math.log(C)).clamp(0.0, 1.0)
    elif m == "ref":
        au_ref = _au_ref(C, alpha.device)
        eu_span = _eu_span_ref(C, alpha.device).clamp_min(eps)
        return ((AU - au_ref) / eu_span).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown mode: {m}")

    # Epistemic Uncertainty NORM
def get_epistemic_uncertainty_norm(alpha: torch.Tensor, eps: float | None = None, mode: str | None = None):
    if eps is None:
        eps = get_eps_value()
    C = alpha.shape[1]
    EU = get_epistemic_uncertainty(alpha, eps)
    m = get_norm_mode() if mode is None else mode
    if m == "max":
        return (EU / math.log(C)).clamp(0.0, 1.0)
    elif m == "ref":
        eu_span = _eu_span_ref(C, alpha.device).clamp_min(eps)
        return (EU / eu_span).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown mode: {m}")

# Fractions

def get_aleatoric_fraction(alpha: torch.Tensor, eps: float | None = None, min_h: float = None):
    if eps is None:
        eps = get_eps_value()
    if min_h is None:
        min_h = get_eps_value()
    H = get_predictive_entropy(alpha, eps)
    AU = get_aleatoric_uncertainty(alpha, eps)
    return (AU / torch.clamp(H, min=min_h)).clamp(0.0, 1.0)


def get_epistemic_fraction(alpha: torch.Tensor, eps: float | None = None, min_h: float = None):
    if eps is None:
        eps = get_eps_value()
    if min_h is None:
        min_h = get_eps_value()
    H = get_predictive_entropy(alpha, eps)
    EU = get_epistemic_uncertainty(alpha, eps)
    return (EU / torch.clamp(H, min=min_h)).clamp(0.0, 1.0)


def get_eu_minus_au_fraction(alpha: torch.Tensor, eps: float | None = None, min_h: float = None):
    if eps is None:
        eps = get_eps_value()
    if min_h is None:
        min_h = get_eps_value()
    AUf = get_aleatoric_fraction(alpha, eps, min_h)
    EUf = get_epistemic_fraction(alpha, eps, min_h)
    return (EUf - AUf).clamp(-1.0, 1.0)

# --------------- Visualization helpers ---------------

def _to_img_from_map_any(m: torch.Tensor, idx: int, clip=(0.02, 0.98), cmap=cv2.COLORMAP_TURBO):
    x = m[idx].detach().cpu().float().numpy()
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"map must be [H,W], got {x.shape}")
    lo, hi = np.quantile(x, clip[0]), np.quantile(x, clip[1])
    if hi <= lo:
        lo, hi = x.min(), x.max() + 1e-6
    x = np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)
    return cv2.applyColorMap((x * 255).astype(np.uint8), cmap)


def _to_img_signed_any(m: torch.Tensor, idx: int, clip=(-1.0, 1.0), cmap=cv2.COLORMAP_TURBO):
    x = m[idx].detach().cpu().float().numpy()
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"signed map must be [H,W], got {x.shape}")
    lo, hi = clip
    x = np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)
    return cv2.applyColorMap((x * 255).astype(np.uint8), cmap)


def build_uncertainty_layers(alpha: torch.Tensor, names: list, idx: int = 0, h_mask_thresh: float = 0.0, eps: float | None = None) -> dict:
    if eps is None:
        eps = get_eps_value()
    out = {}
    need_Hn = "H_norm" in names
    need_AUn = "AU_norm" in names
    need_EUn = "EU_norm" in names
    need_a0 = "alpha0" in names
    need_AUf = "AU_frac" in names
    need_EUf = "EU_frac" in names
    need_diff = "EU_minus_AU_frac" in names

    Hn = AUn = EUn = None
    if need_Hn: Hn = get_predictive_entropy_norm(alpha, eps)
    if need_AUn: AUn = get_aleatoric_uncertainty_norm(alpha, eps)
    if need_EUn: EUn = get_epistemic_uncertainty_norm(alpha, eps)

    if need_a0:
        a0 = alpha.sum(dim=1, keepdim=True) + eps
        out["alpha0"] = _to_img_from_map_any(a0, idx)

    if need_AUf or need_EUf or need_diff:
        H = get_predictive_entropy(alpha, eps)
        AU = get_aleatoric_uncertainty(alpha, eps)
        EU = H - AU
        denom = torch.clamp(H, min=1e-6)
        if need_AUf:
            AU_frac = (AU / denom).clamp(0.0, 1.0)
            out["AU_frac"] = _to_img_from_map_any(AU_frac, idx)
        if need_EUf:
            EU_frac = (EU / denom).clamp(0.0, 1.0)
            out["EU_frac"] = _to_img_from_map_any(EU_frac, idx)
        if need_diff:
            EU_frac = (EU / denom).clamp(0.0, 1.0)
            AU_frac = (AU / denom).clamp(0.0, 1.0)
            diff = (EU_frac - AU_frac).clamp(-1.0, 1.0)
            out["EU_minus_AU_frac"] = _to_img_signed_any(diff, idx, clip=(-1.0, 1.0))

    if need_Hn: out["H_norm"] = _to_img_from_map_any(Hn, idx)
    if need_AUn: out["AU_norm"] = _to_img_from_map_any(AUn, idx)
    if need_EUn: out["EU_norm"] = _to_img_from_map_any(EUn, idx)
    return out

# ---------------- Reliability & calibration helpers ----------------

def compute_mc_reliability_bins(alpha: torch.Tensor, y_true: torch.Tensor, n_bins: int = 10, n_samples: int = 64, eps: float | None = None):
    if eps is None:
        eps = get_eps_value()
    B, C, H, W = alpha.shape
    dist = alphas_to_Dirichlet(alpha)
    samples = dist.sample((n_samples,))                # [M,B,H,W,C]
    max_samples = samples.argmax(dim=-1)               # [M,B,H,W]

    y_expanded = y_true.squeeze(1).unsqueeze(0).expand_as(max_samples)
    correct_emp = (max_samples == y_expanded).float().mean(dim=0)  # [B,H,W]
    conf_mc = correct_emp.flatten().clamp(min=eps, max=1.0 - eps)

    probs_mean = alpha / alpha.sum(dim=1, keepdim=True)
    pred_1shot = probs_mean.argmax(dim=1).flatten()
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


def save_reliability_diagram(empirical_acc: np.ndarray, bin_centers: np.ndarray, tot_counts: np.ndarray, output_path: str = "reliability_diagram.png", title: str = 'Reliability diagram\n(dot area ∝ #pixels per confidence bin — sharpness)', xlabel: str = 'Predicted confidence', ylabel: str = 'Empirical accuracy', show: bool = False, dpi: int = 300):
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
    if show: plt.show()
    plt.close(fig)

# ---------------- IoU between error and entropy-threshold mask ----------------

def compute_entropy_error_iou(entropy_norm: torch.Tensor, error_mask: torch.Tensor, thresholds: Union[Sequence[float], torch.Tensor]) -> torch.Tensor:
    device = entropy_norm.device
    thresholds = torch.as_tensor(thresholds, device=device, dtype=entropy_norm.dtype)
    if entropy_norm.ndim == 2:
        entropy = entropy_norm.unsqueeze(0)
        error = error_mask.unsqueeze(0)
    else:
        entropy = entropy_norm; error = error_mask
    B, H, W = entropy.shape; N = H * W
    entropy_flat = entropy.reshape(B, N)
    error_flat = (error.reshape(B, N) > 0.5)
    T = thresholds.numel()
    thr = thresholds.view(1, T, 1)
    e = entropy_flat.view(B, 1, N)
    pred = e > thr
    gt_err = error_flat.view(B, 1, N).expand_as(pred)
    inter = (pred & gt_err).sum(dim=2).to(torch.float32)
    union = (pred | gt_err).sum(dim=2).to(torch.float32)
    ious = torch.where(union > 0, inter / union, torch.zeros_like(inter))
    return ious.squeeze(0)


def plot_mIOU_errorEntropy(mean_ious: np.ndarray, thresholds: np.ndarray, output_path: str = "mean_iou_curve.png", show: bool = False, dpi: int = 300):
    plt = _get_pyplot("Agg")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.set_ylim(0, 1); ax.set_xlim(0, 1)
    ax.plot(thresholds, mean_ious, marker='o', linestyle='-')
    ax.set_xlabel('Entropy threshold'); ax.set_ylabel('Mean IoU')
    ax.set_title('Mean IoU between Error Mask and Entropy-Threshold Mask')
    ax.grid(True, linestyle=':')
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

# ---------------- Entropy-as-error-probability reliability ----------------

def compute_entropy_reliability(entropy_norm: torch.Tensor, error_mask: torch.Tensor, n_bins: int = 10):
    B, H, W = entropy_norm.shape
    N = B * H * W
    h = entropy_norm.reshape(-1)
    e = error_mask.reshape(-1).float()

    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=h.device, dtype=h.dtype)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bins = torch.bucketize(h, edges, right=False).clamp(min=0, max=n_bins) - 1

    totals = torch.zeros(n_bins, device=h.device, dtype=h.dtype)
    errors = torch.zeros(n_bins, device=h.device, dtype=h.dtype)
    ones   = torch.ones_like(h)

    totals.scatter_add_(0, bins, ones)
    errors.scatter_add_(0, bins, e)

    err_rate = torch.where(totals > 0, errors / totals, torch.zeros_like(errors))
    ece = (totals / max(N, 1) * torch.abs(centers - err_rate)).sum()

    return (
        totals.detach().cpu().numpy(),
        errors.detach().cpu().numpy(),
        err_rate.detach().cpu().numpy(),
        float(ece),
    )
