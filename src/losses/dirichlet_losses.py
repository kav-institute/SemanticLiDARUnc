# dirichlet_losses.py
import math

import torch
import torch.nn as nn
from torch.special import gammaln as lgamma, digamma


# ---------------------------
# Helpers
# ---------------------------

from typing import Optional, Iterable, Union

def _valid_mask(
    target: torch.Tensor,
    ignore_index: Optional[Union[int, Iterable[int], torch.Tensor]]
) -> torch.Tensor:
    """
    Returns a boolean mask of valid pixels.
    target: [B,H,W] or [B,1,H,W] (int)
    ignore_index can be:
      - None: keep all
      - int: single label id to ignore
      - Iterable[int]: multiple label ids to ignore (list/tuple/set)
      - torch.BoolTensor same shape as target: True = keep, False = ignore
    """
    # normalize shape to [B,H,W]
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0]

    if ignore_index is None:
        return torch.ones_like(target, dtype=torch.bool)

    # boolean mask directly
    if torch.is_tensor(ignore_index):
        if ignore_index.dtype == torch.bool:
            # assume same shape as target
            return ignore_index
        else:
            # treat as tensor of ids
            ids = ignore_index.to(device=target.device, dtype=target.dtype)
            # torch.isin is available in modern PyTorch; fallback below if not
            if hasattr(torch, "isin"):
                return ~torch.isin(target, ids)
            else:
                mask_ign = torch.zeros_like(target, dtype=torch.bool)
                for idx in ids.tolist():
                    mask_ign |= (target == idx)
                return ~mask_ign

    # python int -> single id
    if isinstance(ignore_index, int):
        return (target != ignore_index)

    # iterable of ints
    if isinstance(ignore_index, Iterable):
        ids_list = list(ignore_index)
        if len(ids_list) == 0:
            return torch.ones_like(target, dtype=torch.bool)
        ids = torch.as_tensor(ids_list, device=target.device, dtype=target.dtype)
        if hasattr(torch, "isin"):
            return ~torch.isin(target, ids)
        else:
            mask_ign = torch.zeros_like(target, dtype=torch.bool)
            for idx in ids_list:
                mask_ign |= (target == idx)
            return ~mask_ign

    raise TypeError("ignore_index must be None, int, Iterable[int], or bool Tensor")



# ---------------------------
# 1) Dirichlet-categorical NLL: -log E[p_y] = -log(alpha_y / alpha0)
#    Scale-invariant in alpha (no push on alpha0).
# ---------------------------

class NLLDirichletCategorical(nn.Module):
    """
    Dirichlet-categorical marginal NLL for a one-hot label.
    Loss per pixel: -(log(alpha_y) - log(alpha0)).
    Scale-invariant in alpha (no pressure on alpha0).
    """
    def __init__(self, ignore_index: Optional[int] = 0, eps: float = 1e-12):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        alpha : [B,C,H,W], alpha_i > 0
        target: [B,H,W] or [B,1,H,W]
        returns scalar mean over valid pixels
        """
        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0]
        valid = _valid_mask(target, self.ignore_index)
        if valid.sum() == 0:
            return alpha.sum() * 0.0

        a0 = alpha.sum(dim=1)                              # [B,H,W]
        ay = alpha.gather(1, target.unsqueeze(1)).squeeze(1)  # [B,H,W]

        per = -(torch.log(ay + self.eps) - torch.log(a0 + self.eps))
        return (per * valid.float()).sum() / valid.sum().clamp_min(1.0)


# ---------------------------
# 2) Dirichlet Brier (expected).
#    Optionally scale-free via s_ref (replaces alpha0 by constant).
# ---------------------------

class BrierDirichlet(nn.Module):
    """
    Expected Brier score under Dirichlet predictive distribution.

    If s_ref is None:
        uses true alpha0, small residual dependence on alpha0.
    If s_ref is float:
        replaces alpha0 by constant s_ref in E[p_i^2], making it scale-free.
    """
    def __init__(self, ignore_index: Optional[int] = None,
                 s_ref: Optional[float] = None, eps: float = 1e-12):
        super().__init__()
        self.ignore_index = ignore_index
        self.s_ref = s_ref
        self.eps = eps

    def forward(self, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        alpha : [B,C,H,W], alpha_i > 0
        target: [B,H,W] or [B,1,H,W]
        returns scalar mean over valid pixels
        """
        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0]
        target = target.long()

        valid = _valid_mask(target, self.ignore_index)
        if valid.sum() == 0:
            return alpha.sum() * 0.0

        a0 = alpha.sum(dim=1, keepdim=True)                  # [B,1,H,W]
        p_hat = alpha / (a0 + self.eps)                      # [B,C,H,W]

        # sum_i E[p_i^2]
        sum_p2 = (p_hat * p_hat).sum(dim=1, keepdim=True)    # [B,1,H,W]
        if self.s_ref is None:
            sum_ep2 = (a0 * sum_p2 + 1.0) / (a0 + 1.0)
        else:
            s = torch.as_tensor(float(self.s_ref),
                                dtype=alpha.dtype, device=alpha.device)
            sum_ep2 = (s * sum_p2 + 1.0) / (s + 1.0)

        ep_y = p_hat.gather(1, target.unsqueeze(1))          # [B,1,H,W]
        per = (sum_ep2 - 2.0 * ep_y + 1.0).squeeze(1)        # [B,H,W]

        w = valid.float()
        return (per * w).sum() / w.sum().clamp_min(1.0)


# ---------------------------
# 3) Complement KL to Uniform on off-classes (uncertainty spread).
#    Scale-invariant in alpha; optional evidence gate uses a0.detach().
# ---------------------------

class ComplementKLUniform(nn.Module):
    """
    Encourages off-class conditional distribution to approach uniform
    when the model is uncertain (low p_y).

    Per-pixel:
        tilde = p_off / (1 - p_y)
        KL(tilde || U) in [0, log(C-1)], normalized to [0,1] if normalize=True.
        Weighted by w_uncert(p_y) and optional w_evid(a0).
    Optional evidence gate uses a0.detach() so it does not push alpha0.
    """
    def __init__(self,
                 ignore_index: Optional[int] = 0,
                 gamma: float = 2.0,
                 tau: float = 0.55,
                 sigma: float = 0.12,
                 s_target: Optional[float] = None,
                 normalize: bool = True,
                 eps: float = 1e-8):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.sigma = float(sigma)
        self.s_target = s_target           # if not None, only used for gating; alpha0 is detached
        self.normalize = bool(normalize)
        self.eps = eps

    def forward(self, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        alpha : [B,C,H,W], alpha_i > 0
        target: [B,H,W] or [B,1,H,W]
        returns scalar mean over valid pixels
        """
        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0]
        target = target.long()

        valid = _valid_mask(target, self.ignore_index)
        if valid.sum() == 0:
            return alpha.sum() * 0.0

        B, C, H, W = alpha.shape
        if C <= 2:
            return alpha.sum() * 0.0

        a0 = alpha.sum(dim=1, keepdim=True) + self.eps        # [B,1,H,W]
        p = alpha / a0                                        # [B,C,H,W]

        tgt_safe = torch.where(valid, target, torch.zeros_like(target))
        py = p.gather(1, tgt_safe.unsqueeze(1)).clamp_min(self.eps)  # [B,1,H,W]

        # mask out GT channel and build conditional off-class probs
        oh = torch.zeros((B, C, H, W), device=p.device, dtype=torch.bool)
        oh.scatter_(1, tgt_safe.unsqueeze(1), True)
        p_off = p.masked_fill(oh, 0.0)
        S = (1.0 - py).clamp_min(self.eps)
        tilde = p_off / S

        # KL(tilde || U), U_j = 1/(C-1)
        log_tilde = (tilde.clamp_min(self.eps)).log()
        kl_u = (tilde * log_tilde).sum(dim=1) + math.log(C - 1)  # [B,H,W]
        if self.normalize:
            kl_u = kl_u / math.log(C - 1)

        # Uncertainty gate
        w_uncert = ((1.0 - py).pow(self.gamma) *
                    torch.sigmoid((self.tau - py) / self.sigma)).squeeze(1)  # [B,H,W]

        # Optional evidence gate (detach a0 to avoid pushing alpha0)
        if self.s_target is not None:
            s_val = float(self.s_target)
            w_evid = s_val / (a0.detach().squeeze(1) + s_val)
            w = w_uncert * w_evid
        else:
            w = w_uncert

        per = w * kl_u
        return (per * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
