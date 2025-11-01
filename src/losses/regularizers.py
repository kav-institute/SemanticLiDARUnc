# helpers --------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Iterable, Union

def _valid_mask(
    target: torch.Tensor,
    ignore_index: Optional[Union[int, Iterable[int], torch.Tensor]]
) -> torch.Tensor:
    """
    Build a boolean mask of valid pixels.
    target: [B,H,W] or [B,1,H,W] (int)
    ignore_index can be:
      - None: keep all
      - int: single id to ignore
      - Iterable[int]: multiple ids to ignore
      - Bool tensor same shape as target (True=keep, False=ignore)
    """
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0]

    if ignore_index is None:
        return torch.ones_like(target, dtype=torch.bool)

    if torch.is_tensor(ignore_index):
        if ignore_index.dtype == torch.bool:
            return ignore_index
        ids = ignore_index.to(device=target.device, dtype=target.dtype)
        if hasattr(torch, "isin"):
            return ~torch.isin(target, ids)
        mask_ign = torch.zeros_like(target, dtype=torch.bool)
        for idx in ids.tolist():
            mask_ign |= (target == idx)
        return ~mask_ign

    if isinstance(ignore_index, int):
        return (target != ignore_index)

    if isinstance(ignore_index, Iterable):
        ids_list = list(ignore_index)
        if len(ids_list) == 0:
            return torch.ones_like(target, dtype=torch.bool)
        ids = torch.as_tensor(ids_list, device=target.device, dtype=target.dtype)
        if hasattr(torch, "isin"):
            return ~torch.isin(target, ids)
        mask_ign = torch.zeros_like(target, dtype=torch.bool)
        for idx in ids_list:
            mask_ign |= (target == idx)
        return ~mask_ign

    raise TypeError("ignore_index must be None, int, Iterable[int], or bool Tensor")

def _mean_over_valid(x: torch.Tensor,
                     mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Reduce x by mean. If mask is provided (bool [B,H,W]), average only over
    valid locations. x can be [B,H,W] or [B,C,H,W]; mask is broadcast on C.
    """
    eps = 1e-8
    if mask is None:
        return x.mean()
    m = mask.float()
    if x.dim() == 4:
        m = m.unsqueeze(1)                 # [B,1,H,W] -> broadcast over C
    num = (m * 1.0).sum().clamp_min(eps)
    return (x * m).sum() / num

# ---------------------------------------------------------------------
# Logit regularizer with optional masking
# ---------------------------------------------------------------------

class LogitRegularizer(nn.Module):
    """
    Hinge-squared on raw logits z to prevent very large values.

    If threshold is None:
        L = mean( z^2 ) over (valid) elements
    Else:
        L = mean( max(0, z - thr)^2 ) over (valid) elements

    Notes
    -----
    - If your dataset has many ignore pixels, pass a mask (or target+ignore_index)
      so unlabeled areas do not dominate this term.
    """
    def __init__(self, threshold: Optional[float] = None,
                 ignore_index: Optional[Union[int, Iterable[int], torch.Tensor]] = None):
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index

    def forward(self,
                logits: torch.Tensor,
                *,
                mask: Optional[torch.Tensor] = None,
                target: Optional[torch.Tensor] = None               
        ) -> torch.Tensor:
        # Build mask from target if needed
        if mask is None and target is not None:
            mask = _valid_mask(target, ignore_index=self.ignore_index)

        if self.threshold is None:
            per = logits.pow(2)                    # [B,C,H,W]
        else:
            per = F.relu(logits - float(self.threshold)).pow(2)

        return _mean_over_valid(per, mask)

# ---------------------------------------------------------------------
# Evidence regularizers with optional masking
# ---------------------------------------------------------------------

class EvidenceRegBand(nn.Module):
    """
    Two-sided log spring on a0 = sum_k alpha_k with a dead-zone band.

    No penalty inside [ s*(1-band), s*(1+band) ].
    Outside the band:
        over  = relu( log(a0 / (s*(1+band))) )
        under = relu( log((s*(1-band)) / a0) )
        L = mean_over_valid( over^2 + under^2 )
    """
    def __init__(self, s_target: float, band: float = 0.10, ignore_index: Optional[Union[int, Iterable[int], torch.Tensor]] = None):
        super().__init__()
        self.s = float(s_target)
        self.band = float(band)
        self.ignore_index = ignore_index

    def forward(self,
                alpha: torch.Tensor,
                *,
                mask: Optional[torch.Tensor] = None,
                target: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        if mask is None and target is not None:
            mask = _valid_mask(target, ignore_index=self.ignore_index)

        a0 = alpha.sum(dim=1) + 1e-8               # [B,H,W]
        s_hi = self.s * (1.0 + self.band)
        s_lo = self.s * (1.0 - self.band)
        over  = F.relu(torch.log(a0 / s_hi))        # >0 only when a0 > s_hi
        under = F.relu(torch.log(s_lo / a0))        # >0 only when a0 < s_lo
        per = over.pow(2) + under.pow(2)            # [B,H,W]
        return _mean_over_valid(per, mask)

class EvidenceReg(nn.Module):
    """
    Direct regularizer on total evidence a0 = sum_k alpha_k.

    Modes
    -----
    "log_squared":
        L = mean_over_valid( log(a0 / s)^2 )
        Smooth and symmetric in log-space.

    "one_sided":
        L = mean_over_valid( relu(a0 - s*(1+margin))^2 )
        Only penalizes when a0 is above the soft cap.

    "l2":
        L = mean_over_valid( (a0 - s)^2 )
        Simple quadratic in linear space.

    Tip
    ---
    If "log_squared" feels too weak when a0 is high, consider either:
      - increase the weight, or
      - switch to "l2", or
      - multiply by (a0/s) for scale-corrected strength.
    """
    def __init__(self, s_target: float, mode: str = "log_squared", margin: float = 0.1,
                 scale_correct: bool = False,
                 ignore_index: Optional[Union[int, Iterable[int], torch.Tensor]] = None):  # set True to strengthen at high a0
        super().__init__()
        self.s_target = float(s_target)
        self.mode = mode
        self.margin = float(margin)
        self.scale_correct = bool(scale_correct)
        self.ignore_index = ignore_index

    def forward(self,
                alpha: torch.Tensor,
                *,
                mask: Optional[torch.Tensor] = None,
                target: Optional[torch.Tensor] = None,
               ) -> torch.Tensor:
        if mask is None and target is not None:
            mask = _valid_mask(target, ignore_index=self.ignore_index)

        a0 = alpha.sum(dim=1) + 1e-8               # [B,H,W]
        s  = self.s_target

        if self.mode == "log_squared":
            lr = torch.log(a0 / s)                 # log-ratio
            per = lr.pow(2)
            if self.scale_correct:
                # optional: stronger push when a0 >> s
                per = (a0 / s) * per
            return _mean_over_valid(per, mask)

        elif self.mode == "one_sided":
            cap = s * (1.0 + self.margin)
            per = F.relu(a0 - cap).pow(2)
            return _mean_over_valid(per, mask)

        else:  # "l2"
            per = (a0 - s).pow(2)
            return _mean_over_valid(per, mask)


import math
import torch
import torch.nn as nn

class WrongLowEvidence(nn.Module):
    """
    Trim total evidence a0 = C + s *only* on WRONG predictions (argmax != y).

    Per-pixel:
        wrong = 1[argmax(p) != y]                    (no grad)
        m = p_pred - p_y                             (confidence margin, no grad)
        gate_m = 1  (or sigmoid((m - margin)/k))
        L = wrong * gate_m * relu(log(a0) - log(C + s_low))^2

    Notes:
      - Works in log-space to be scale-stable.
      - Affects *only* the scale head: a0 = C + s, and d a0 / d(shape logits) = 0.
      - Set s_low=0 to pull wrongs toward the uniform-prior evidence (a0 ≈ C).
    """
    def __init__(self,
                 ignore_index=None,
                 s_low: float = 0.0,
                 margin: float = 0.05,      # require a bit of confidence gap
                 soft_margin_k: float = 0.08,  # 0 => hard gate; >0 => smooth
                 eps: float = 1e-8):
        super().__init__()
        self.ignore_index = ignore_index
        self.s_low = float(s_low)
        self.margin = float(margin)
        self.k = float(soft_margin_k)
        self.eps = float(eps)

    def forward(self, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # normalize target to [B,H,W]
        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0]
        target = target.long()

        valid = _valid_mask(target, self.ignore_index)
        if valid.sum() == 0:
            return alpha.sum() * 0.0

        B, C, H, W = alpha.shape
        a0 = alpha.sum(dim=1, keepdim=True).clamp_min(self.eps)      # [B,1,H,W]
        p  = (alpha / a0)                                            # [B,C,H,W]

        # ---- gates computed on detached probs (no grad) ----
        with torch.no_grad():
            p_det = p.detach()
            # hard error mask via argmax
            pred = p_det.argmax(dim=1)                               # [B,H,W]
            wrong_mask = (pred != target)                            # bool
            # confidence margin m = p_pred - p_y
            py   = p_det.gather(1, target.unsqueeze(1)).clamp_min(self.eps)  # [B,1,H,W]
            pmax = p_det.max(dim=1, keepdim=True).values.clamp_min(self.eps) # [B,1,H,W]
            m = (pmax - py).squeeze(1)                                       # [B,H,W] >=0 if wrong

            # margin gate
            if self.margin > 0.0:
                if self.k > 0.0:
                    gate_m = torch.sigmoid((m - self.margin) / self.k)
                else:
                    gate_m = (m > self.margin).float()
            else:
                gate_m = torch.ones_like(m, dtype=p.dtype)

            gate_wrong = wrong_mask.float() * gate_m * valid.float()         # [B,H,W]

        # ---- squared hinge in log(a0) above log(C + s_low) ----
        C = alpha.shape[1]  # Number of classes
        a0_log = a0.log().squeeze(1)                                   # [B,H,W]
        target_log = math.log(C + self.s_low + self.eps)
        per = torch.relu(a0_log - target_log).pow(2) * gate_wrong

        denom = gate_wrong.sum().clamp_min(1.0)  # average over active wrong pixels
        return per.sum() / denom

class KL_offClasses_to_uniform(nn.Module):
    """ 
    KL divergence term that penalizes evidence for the
    non-true classes, pushing alpha for wrong classes toward 1.
    """
    
    def __init__(self,
                 ignore_index: Optional[int] = None,
                 with_conf_weighting: bool = False,
                 gamma: float = 1.0,
                 eps: float=1e-8):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps
        self.with_conf_weighting = with_conf_weighting
        self.gamma = gamma
    
    @staticmethod
    def _dirichlet_kl_to_uniform(alpha_tilde: torch.Tensor,
                             eps: float = 1e-12) -> torch.Tensor:
        """
        KL( Dir(alpha_tilde) || Dir(1,...,1) ), per sample / per pixel.
        alpha_tilde: [N,C] with all entries > 0

        We drop constant terms that do not depend on alpha_tilde. This is fine for
        training because constants have zero gradient.

        Formula:
        KL = logGamma(sum_k a_k) - sum_k logGamma(a_k)
            + sum_k (a_k - 1) * (digamma(a_k) - digamma(sum_k a_k))

        Additionally multiplies the per-pixel off-class Dirichlet KL by an
        error-aware weight:
            w_i = (1 - p_hat_y[i]) ** gamma
        where p_hat_y = alpha_y / sum_k alpha_k.
        Effect: focus KL pressure on wrong/uncertain pixels and back off when the
        model is confident-and-correct. gamma > 1.0 increases the focus on uncertain pixels.

        returns kl_per_sample: [N]
        """
        # shape
        a = alpha_tilde.clamp_min(eps)           # [N,C]
        sum_a = a.sum(dim=1, keepdim=True)       # [N,1]

        term1 = torch.lgamma(sum_a) - torch.lgamma(a).sum(dim=1, keepdim=True)
        term2 = ((a - 1.0) *
                (torch.digamma(a) - torch.digamma(sum_a))).sum(dim=1, keepdim=True)

        kl = term1 + term2                       # [N,1]
        return kl.squeeze(1)                     # [N]
    
    def forward(self, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0]
        target = target.long()

        valid = _valid_mask(target, self.ignore_index)
        if valid.sum() == 0:
            return alpha.sum() * 0.0
        
        # one-hot targets y_onehot: [B,C,H,W]
        # we only build for valid pixels to save memory if you want to,
        # but for clarity we just build full then mask at the end.
        y_onehot = torch.zeros_like(alpha)             # [B,C,H,W]
        y_onehot.scatter_(1, target.unsqueeze(1), 1.0)
        
        # alpha_tilde = y + (1 - y) * alpha
        # This "removes" evidence for the true class so KL only punishes
        # spurious evidence for non-true classes.
        alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha  # [B,C,H,W]

        # flatten valid pixels so we compute KL only there
        # shape after view: [N_valid, C]
        B, C, H, W = alpha.shape
        valid_flat = valid.view(B, -1)                     # [B,H*W]
        alpha_tilde_flat = alpha_tilde.permute(0, 2, 3, 1).reshape(-1, C)   # make class layer last and flatten on elements
        alpha_tilde_flat = alpha_tilde_flat[valid_flat.reshape(-1)]

        assert alpha_tilde_flat.numel() != 0, "no off classes present?! Check KL function"
        
        kl_each = self._dirichlet_kl_to_uniform(alpha_tilde_flat,
                                            eps=self.eps)  # [N_valid]

        # ---- compute per-pixel weight w = (1 - p_hat_y)**gamma ----
        if self.with_conf_weighting:
            a0 = alpha.sum(dim=1, keepdim=True)                     # [B,1,H,W]
            p_hat = alpha / (a0 + self.eps)                         # [B,C,H,W]
            p_hat_y = p_hat.gather(1, target.unsqueeze(1)).squeeze(1)  # [B,H,W]
            one_minus = (1.0 - p_hat_y).clamp(0.0, 1.0)
            w_pix = one_minus ** self.gamma                         # [B,H,W]
            # match weights to flattened valid pixels
            w_flat = w_pix.view(B, -1)[valid_flat].reshape(-1)      # [N_valid]
            w_flat = w_flat.detach()                             # no grad on weighting
        
            kl_mean = (kl_each * w_flat).sum() / w_flat.sum().clamp_min(1.0) 
        else:
            kl_mean = kl_each.mean()

        return kl_mean