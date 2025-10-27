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


class NLLDirichletCategorical(nn.Module):
    """
    Dirichlet-categorical marginal NLL for a one-hot label.

    This is -log E[p_y] where E[p_y] = alpha_y / alpha0.
    So:
        loss = -( log(alpha_y) - log(alpha0) )

    This loss is scale-invariant in alpha:
    if you multiply all alpha_k by c>0, alpha0 also multiplies by c,
    alpha_y / alpha0 stays the same, and the loss is unchanged.
    That means there is no direct gradient incentive to blow up alpha0.

    alpha  : Dirichlet params > 0, shape [B,C,H,W]
    target : class ids, shape [B,H,W] or [B,1,H,W]
    """

    def __init__(self,
                 ignore_index: Optional[int] = None,
                 eps: float = 1e-12):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self,
                alpha: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:

        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0]
        target = target.long()

        valid = _valid_mask(target, self.ignore_index)  # [B,H,W] bool
        if valid.sum() == 0:
            return alpha.sum() * 0.0

        # alpha0 = sum_k alpha_k
        a0 = alpha.sum(dim=1)  # [B,H,W]

        # alpha_y = alpha for the ground truth class
        ay = alpha.gather(1, target.unsqueeze(1)).squeeze(1)  # [B,H,W]

        # per-pixel nll = -(log ay - log a0)
        per_pix = -(torch.log(ay + self.eps) - torch.log(a0 + self.eps))  # [B,H,W]

        w = valid.float()
        return (per_pix * w).sum() / w.sum().clamp_min(1.0)


class DigammaDirichletCE(nn.Module):
    """
    Expected cross-entropy under Dirichlet (Eq. 4 in Sensoy et al).

    L = psi(alpha0) - psi(alpha_y)

    where psi is the digamma function, psi(x) = d/dx log Gamma(x).

    This is E[-log p_y] for p ~ Dir(alpha).
    This loss is NOT scale-invariant in alpha: increasing alpha0
    (especially alpha_y) keeps lowering the loss.

    alpha  : Dirichlet params > 0, shape [B,C,H,W]
    target : class ids, shape [B,H,W] or [B,1,H,W]
    """

    def __init__(self,
                 ignore_index: Optional[int] = None,
                 eps: float = 1e-12):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps  # not heavily used here but good to have

    def forward(self,
                alpha: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:

        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0]
        target = target.long()

        valid = _valid_mask(target, self.ignore_index)  # [B,H,W] bool
        if valid.sum() == 0:
            return alpha.sum() * 0.0

        # alpha0 = sum_k alpha_k
        a0 = alpha.sum(dim=1)  # [B,H,W]

        # alpha_y = alpha for gt class
        ay = alpha.gather(1, target.unsqueeze(1)).squeeze(1)  # [B,H,W]

        # per-pixel CE risk = psi(alpha0) - psi(alpha_y)
        per_pix = torch.digamma(a0) - torch.digamma(ay)  # [B,H,W]

        w = valid.float()
        return (per_pix * w).sum() / w.sum().clamp_min(1.0)


# ---------------------------
#   Dirichlet Brier (expected).
#   Optionally scale-free via s_ref (replaces alpha0 by constant).
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


# # ---------------------------
# # 3) Complement KL to Uniform on off-classes (uncertainty spread).
# #    Scale-invariant in alpha; optional evidence gate uses a0.detach().
# # ---------------------------

class ComplementKLUniform(nn.Module):
    """
    Encourage off-class conditional distribution to be high-entropy when uncertain.

    Per-pixel:
      p      = alpha / sum(alpha)                  # predictive mean
      py     = p_y                                 # prob of GT class
      tilde  = p_off / (1 - py)                    # conditional over non-GT classes
      L_comp = KL(tilde || U), U_j = 1/(C-1)       # normalized to [0,1] if normalize=True

    Weighting:
      w_uncert(py) up-weights ambiguous pixels, but we DETACH py in the gate so
      the model cannot reduce the loss weight by changing py itself.

    Notes:
      - s_target is optional; if provided we use ONLY a detached a0 gate.
      - Scale-invariant in alpha; pairs well with shape+scale parametrization.
    """
    def __init__(self,
                 ignore_index: Optional[int] = 0,
                 gamma: float = 2.0,
                 tau: float = 0.55,
                 sigma: float = 0.12,
                 s_target: Optional[float] = None,
                 normalize: bool = True,
                 eps: float = 1e-8,
                 detach_uncert: bool = True):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.sigma = float(sigma)
        self.s_target = s_target
        self.normalize = bool(normalize)
        self.eps = eps
        self.detach_uncert = bool(detach_uncert)

    def forward(self, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # shapes
        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0]
        target = target.long()

        valid = _valid_mask(target, self.ignore_index)
        if valid.sum() == 0:
            return alpha.sum() * 0.0

        B, C, H, W = alpha.shape
        if C <= 2:
            return alpha.sum() * 0.0

        # predictive mean
        a0 = alpha.sum(dim=1, keepdim=True) + self.eps           # [B,1,H,W]
        p  = alpha / a0                                          # [B,C,H,W]

        # gather py at GT labels (guard invalid with zeros)
        tgt_safe = torch.where(valid, target, torch.zeros_like(target))
        py = p.gather(1, tgt_safe.unsqueeze(1)).clamp_min(self.eps)    # [B,1,H,W]

        # build p_off by zeroing the GT channel
        p_off = p.clone()
        p_off.scatter_(1, tgt_safe.unsqueeze(1), 0.0)            # zero GT prob
        denom = (1.0 - py).clamp_min(self.eps)
        tilde = p_off / denom                                    # conditional off-class

        # KL(tilde || U)
        log_tilde = (tilde.clamp_min(self.eps)).log()
        kl_u = (tilde * log_tilde).sum(dim=1) + math.log(C - 1)  # [B,H,W]
        if self.normalize:
            kl_u = kl_u / math.log(C - 1)

        # uncertainty gate (DETACHED py to avoid gaming the weight)
        py_gate = py.detach() if self.detach_uncert else py
        w_uncert = ((1.0 - py_gate).pow(self.gamma) *
                    torch.sigmoid((self.tau - py_gate) / self.sigma)).squeeze(1)  # [B,H,W]

        # optional evidence gate on a0, detached to avoid pushing alpha0
        if self.s_target is not None:
            s_val = float(self.s_target)
            w_evid = s_val / (a0.detach().squeeze(1) + s_val)   # [B,H,W]
            w = w_uncert * w_evid
        else:
            w = w_uncert

        per = w * kl_u
        wmask = valid.float()
        return (per * wmask).sum() / wmask.sum().clamp_min(1.0)


class DirichletMSELoss(nn.Module):
    """
    The expected square error (Brier-style Bayes risk) under a Dirichlet,
        Reference equation (5) in Sensoy et al 2018.
        This is the data fit term.
    """
    def __init__(self,
                 ignore_index: Optional[int] = None,
                 eps: float = 1e-8):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps
        
    def forward(self,
                alpha: torch.Tensor,
                target: torch.Tensor,
    ) -> torch.Tensor:
        """
        alpha: [B,C,H,W]
        target: [B,H,W] or [B,1,H,W]
        epoch_num: current epoch index
        anneal_step: anneal denominator for lambda_t

        returns scalar loss
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

        # ------------------------------------------------------------------
        # Data fit term (expected squared error under Dirichlet)
        # ------------------------------------------------------------------

        # alpha0 = sum_k alpha_k
        alpha0 = alpha.sum(dim=1, keepdim=True)        # [B,1,H,W]

        # p_hat = E[p] = alpha / alpha0
        p_hat = alpha / (alpha0 + self.eps)            # [B,C,H,W]

        # one-hot targets y_onehot: [B,C,H,W]
        # we only build for valid pixels to save memory if you want to,
        # but for clarity we just build full then mask at the end.
        y_onehot = torch.zeros_like(alpha)             # [B,C,H,W]
        y_onehot.scatter_(1, target.unsqueeze(1), 1.0)

        # squared error term: (y - p_hat)^2
        sq_err = (y_onehot - p_hat) ** 2               # [B,C,H,W]

        # Dirichlet predictive variance term:
        # var = alpha * (alpha0 - alpha) / (alpha0^2 * (alpha0 + 1))
        var = alpha * (alpha0 - alpha) / (
            (alpha0 * alpha0 + self.eps) * (alpha0 + 1.0)
        )                                              # [B,C,H,W]

        # per-pixel mse_like = sum_c [ sq_err + var ]
        mse_like = (sq_err + var).sum(dim=1)           # [B,H,W]
        
        # mean mse_like over valid pixels
        mse_mean = (mse_like * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
        return mse_mean