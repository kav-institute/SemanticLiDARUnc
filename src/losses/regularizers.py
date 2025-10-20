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
