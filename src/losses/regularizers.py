import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitRegularizer(nn.Module):
    """
    Hinge-squared regularizer on raw logits to prevent very large z.

    If threshold is None:
        L = mean( z^2 )                        # plain L2 on all logits
    Else (hinge):
        L = mean( max(0, z - threshold)^2 )    # only penalize z above threshold

    Why this helps:
        Since alpha = 1 + softplus(z/T), very large z produce very large alpha.
        Putting a hinge in logit space caps "how easy" it is to create huge
        alpha by pushing z beyond the threshold.

    Parameters
    ----------
    threshold : float or None
        Logit value where the hinge starts. Choose using your evidence budget,
        e.g. z_thr from logit_threshold_for_alpha_cap(s_total, K, m, margin, T).
        If None, the penalty is global L2 on z.
    """
    def __init__(self, threshold: float | None = None):
        super().__init__()
        self.threshold = threshold

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits : [B, C, H, W] (pre-softplus)
        returns a scalar tensor
        """
        if self.threshold is None:
            # Global L2 on all logits: L = mean(z^2)
            return (logits ** 2).mean()
        else:
            # Hinge-L2 above threshold: L = mean( max(0, z - thr)^2 )
            return (torch.relu(logits - self.threshold) ** 2).mean()

class EvidenceRegBand(nn.Module):
    """
    Keep a0 = sum_k alpha_k near target s with a two-sided spring in log-space.

    Band idea:
      No penalty if a0 is inside [ s*(1-band), s*(1+band) ].
      Outside the band, penalize the log-ratio smoothly.

    Loss:
      over  = relu( log(a0 / (s*(1+band))) )
      under = relu( log((s*(1-band)) / a0) )
      L = mean( over^2 + under^2 )

    Why this helps:
      - Two-sided: pushes down if a0 too high, up if a0 too low.
      - Log-space: multiplicative errors are treated evenly.
      - Dead-zone avoids fighting the shape losses when a0 is "good enough".
    """
    def __init__(self, s_target: float, band: float = 0.10):
        super().__init__()
        self.s = float(s_target)
        self.band = float(band)

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        a0 = alpha.sum(dim=1) + 1e-8               # [B,H,W]
        s_hi = self.s * (1.0 + self.band)
        s_lo = self.s * (1.0 - self.band)
        over  = F.relu(torch.log(a0 / s_hi))        # >0 only when a0 > s_hi
        under = F.relu(torch.log(s_lo / a0))        # >0 only when a0 < s_lo
        return (over.pow(2) + under.pow(2)).mean()
    
class EvidenceReg(nn.Module):
    """
    Direct regularizer on total evidence alpha0 = sum_k alpha_k.

    Modes (all averaged over pixels):
      1) "log_squared":
            L = mean( log(alpha0 / s_target)^2 )
         Properties:
            - Penalizes multiplicative deviation of alpha0 from s_target.
            - Symmetric: pushes down if alpha0 > s_target, pushes up if alpha0 < s_target.
            - Smooth and scale-aware; works well across a wide range of scales.

      2) "one_sided":
            L = mean( max(0, alpha0 - s_target*(1 + margin))^2 )
         Properties:
            - Only pushes down when alpha0 exceeds the soft cap s_target*(1+margin).
            - No penalty if alpha0 is at or below the cap.
            - Use this when you only want to stop blow-ups, not lift low alpha0.

      3) "l2":
            L = mean( (alpha0 - s_target)^2 )
         Properties:
            - Simple quadratic around s_target in linear space.

    Parameters
    ----------
    s_target : float
        Desired target for alpha0 (total evidence). For a "mean-preserving"
        KL prior you typically choose s_target via your Beta coverage solver.
    mode : {"log_squared", "one_sided", "l2"}
        Which penalty to use (see above).
    margin : float
        Only used in "one_sided". The cap is s_target*(1 + margin).
        Example: margin = 0.10 permits ~10% overshoot before penalty.
    """
    def __init__(self, s_target: float, mode: str = "log_squared", margin: float = 0.0):
        super().__init__()
        self.s_target = float(s_target)
        self.mode = mode
        self.margin = float(margin)

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        alpha : [B, C, H, W], alpha > 0
        returns a scalar tensor
        """
        # alpha0 per pixel: [B, H, W]
        a0 = alpha.sum(dim=1)

        if self.mode == "log_squared":
            # L = mean( log(alpha0 / s)^2 )
            # log-ratio is numerically stable for wide ranges of scales.
            log_ratio = torch.log(a0 / self.s_target)
            return (log_ratio ** 2).mean()

        elif self.mode == "one_sided":
            # L = mean( max(0, alpha0 - s*(1+m))^2 ), where m is margin
            cap = self.s_target * (1.0 + self.margin)
            penalty = torch.relu(a0 - cap) ** 2
            return penalty.mean()

        else:  # "l2"
            # L = mean( (alpha0 - s)^2 )
            return ((a0 - self.s_target) ** 2).mean()