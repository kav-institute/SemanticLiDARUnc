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

    def forward(self, outputs, targets, eps=1e-8):
        mean, var = self.adf_softmax(*outputs)
        targets = torch.nn.functional.one_hot(targets.squeeze(1), num_classes=20).permute(0,3,1,2).float()

        precision = 1 / (var + eps)
        nll = torch.mean(0.5 * precision * (targets - mean) ** 2 + 0.5 * torch.log(var + eps))
        return nll


### >>> Dirichlet <<< ###
from models.probability_helper import (
    to_alpha_concentrations, 
    alphas_to_Dirichlet, 
    smooth_one_hot
)
import torch
import torch.nn as nn
from torch.special import digamma, gammaln as lgamma
from models.probability_helper import get_eps_value

def _beta_moment(a: torch.Tensor, b: torch.Tensor, q: float) -> torch.Tensor:
    """
    E[p^q] for Beta(a,b), computed in log-space:
    E[p^q] = B(a+q, b) / B(a, b) = exp( lgamma(a+q)-lgamma(a) + lgamma(a+b)-lgamma(a+b+q) )
    """
    return torch.exp(lgamma(a + q) - lgamma(a) + lgamma(a + b) - lgamma(a + b + q))

class DirichletLoss(nn.Module):
    """Data-term objectives with optional separate KL prior regularizer.
        objective: "nll" | "dce" | "imax"
        kl_prior_mode: "evidence" | "symmetric"
    
    'Information Aware Max-Norm Dirichlet Networks for Predictive Uncertainty Estimation (Tsiligkaridis)'
    https://doi.org/10.48550/arXiv.1910.04819
        - nll: negative log-marginal likelihood
            Dirichlet over a smoothed one-hot: fits the entire Dirichlet to match a target pseudo-distribution. 
            It can encourage very peaky Dirichlets (high alpha_0) unless regularized carefully; 
            when labels are noisy, the NLL can over-trust the target.
        - dce: dirichlet cross entropy/ Bayes risk of the cross-entropy loss
            DCE[E[-log p_y] = digamma(alpha_0) - digamma(alpha_y)
            targets only the expected negative log-prob of the correct class; it doesn't micromanage off-class structure
        - imax: Info-Aware Max-Norm loss, From paper
            Let p ~ Dir(alpha), alpha0 = sum_j alpha_j, correct class = c.
            We want the Bayes risk of ||y - p||_∞. Using Jensen + Lp relaxation:
            R_p  =  E[ ||y - p||_∞ ]   <=   { E[ (1 - p_c)^p ] + sum_{j≠c} E[ p_j^p ] }^(1/p)

            For a Dirichlet, each marginal p_j ~ Beta(a=alpha_j, b=alpha0 - alpha_j).

            Beta q-th moment:
            E[ p_j^q ] = B(a + q, b) / B(a, b) = exp( lgamma(a+q) - lgamma(a) + lgamma(a+b) - lgamma(a+b+q) )

            By symmetry:
            E[ (1 - p_c)^p ] = B(b_c + p, a_c) / B(b_c, a_c)   with a_c=alpha_c, b_c=alpha0 - alpha_c.

            So the per-pixel "imax" objective is:

            F = { E[(1 - p_c)^p] + sum_{j≠c} E[p_j^p] }^(1/p)

            We minimize the masked mean of F over pixels.
            Penalizes the largest competing mass instead. imax deliberately does not push the correct class to 1.0 as aggressively as NLL.
    """
    def __init__(self,
                num_classes: int = 20,
                objective: str = "nll",
                smoothing: float = 0.25,
                temperature: float = 1.0,
                prior_concentration: float = 3.0,
                # keep kl_weight here only for reference; apply it in the trainer
                kl_weight: float = 0.0,
                eps: float | None = None,
                kl_prior_mode: str = "evidence",
                ignore_index: int | None = 0,
                p_moment: float = 4.0):
        super().__init__()
        assert objective in ("nll", "dce", "imax")
        assert kl_prior_mode in ("evidence", "symmetric")
        self.num_classes = num_classes
        self.objective = objective
        self.smoothing = smoothing
        self.temperature = temperature
        self.prior_concentration = prior_concentration
        self.kl_weight = kl_weight    # not used internally anymore; just stored
        self.eps = eps
        self.kl_prior_mode = kl_prior_mode
        self.ignore_index = ignore_index
        self.p_moment = float(p_moment)

    # ----- internals -----
    @staticmethod
    def _kl_map(alpha: torch.Tensor, alpha_prior: torch.Tensor) -> torch.Tensor:
        a0  = alpha.sum(dim=1, keepdim=True)
        a0p = alpha_prior.sum(dim=1, keepdim=True)
        t1 = lgamma(a0) - lgamma(a0p)
        t2 = (lgamma(alpha_prior) - lgamma(alpha)).sum(dim=1, keepdim=True)
        t3 = ((alpha - alpha_prior) * (digamma(alpha) - digamma(a0))).sum(dim=1, keepdim=True)
        return t1 + t2 + t3  # [B,1,H,W]

    @staticmethod
    def _dce_perpix(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a0 = alpha.sum(dim=1)
        ay = alpha.gather(1, y.unsqueeze(1)).squeeze(1)
        return (digamma(a0) - digamma(ay))  # [B,H,W]

    def _imax_perpix(self, alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p   = self.p_moment
        a0  = alpha.sum(dim=1, keepdim=False)             # [B,H,W]
        a_c = alpha.gather(1, y.unsqueeze(1)).squeeze(1)  # [B,H,W]
        b_c = a0 - a_c                                    # [B,H,W]
        # E[(1 - p_c)^p]
        term_c = _beta_moment(b_c, a_c, p)
        # sum_j E[p_j^p] - E[p_c^p]
        a_all  = alpha                                    # [B,C,H,W]
        b_all  = a0.unsqueeze(1) - a_all                  # [B,C,H,W]
        ep_all = _beta_moment(a_all, b_all, p)            # [B,C,H,W]
        ep_sum = ep_all.sum(dim=1)                        # [B,H,W]
        ep_c   = _beta_moment(a_c, b_c, p)                # [B,H,W]
        F = (term_c + (ep_sum - ep_c) + 1e-12).pow(1.0 / p)
        return F

    # ----- public split API -----
    @torch.no_grad()
    def _valid_mask(self, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0]
        if self.ignore_index is None:
            return torch.ones_like(target, dtype=torch.bool)
        return (target != self.ignore_index)

    def data_from_alpha(self, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute data term only.
        Returns a scalar (mean over valid pixels).
        """
        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0]
        valid = self._valid_mask(target)
        if valid.sum() == 0:
            return alpha.sum() * 0.0

        if self.objective == "dce":
            per_pix = self._dce_perpix(alpha, target)                 # [B,H,W]
            return (per_pix * valid.float()).sum() / valid.sum()

        if self.objective == "nll":
            dist   = alphas_to_Dirichlet(alpha)
            tgt_sm = smooth_one_hot(torch.where(valid, target, 0),
                                    num_classes=self.num_classes,
                                    smoothing=self.smoothing)         # [B,C,H,W]
            logp   = dist.log_prob(tgt_sm.permute(0, 2, 3, 1))        # [B,H,W]
            return (-(logp) * valid.float()).sum() / valid.sum()

        # "imax"
        per_pix = self._imax_perpix(alpha, target)                    # [B,H,W]
        return (per_pix * valid.float()).sum() / valid.sum()

    def kl_from_alpha(self, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute KL prior term only.
        Returns a scalar (mean over valid pixels).
        """
        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0]
        valid = self._valid_mask(target)
        if valid.sum() == 0:
            return alpha.sum() * 0.0

        if self.kl_prior_mode == "symmetric":
            # Symmetric prior:
            #   alpha_prior = c * 1   (each class has the same concentration = prior_concentration)
            #   => p_hat_prior = 1/C
            #
            # Effect:
            #   KL(Dir(alpha) || Dir(c*1)) penalizes BOTH:
            #     (i) mean shift: pushes predicted mean p_hat_j = alpha_j / alpha0 towards uniform 1/C
            #    (ii) evidence magnitude: discourages alpha0 = sum_j alpha_j from growing too large
            #
            # Good for: strong regularization, stabilizing training if predictions collapse.
            # Risk: can hurt IoU on rare classes, since it pulls means toward uniform.
            alpha_prior = torch.full_like(alpha, self.prior_concentration)
        else:  # "evidence", tries to make the sum match self.prior_concentration while keeping the same mean, confidence regularizer
            # Evidence prior:
            #   alpha_prior = s * p_hat,   with p_hat = alpha / alpha0
            #   => p_hat_prior = p_hat   (same mean as the prediction, only "strength" = s)
            #
            # Effect:
            #   KL(Dir(alpha) || Dir(s * p_hat)) approx penalizes |alpha0 - s|,
            #   i.e. how far the total evidence diverges from the target prior_concentration,
            #   while leaving the MEAN p_hat intact.
            #
            # Good for: controlling overconfident alpha0 growth (calibration) without
            # interfering with class proportions learned by the data term.
            # This plays nicely with iMAX (mean-shaping) + Lovasz (IoU).
            with torch.no_grad():
                a0    = alpha.sum(dim=1, keepdim=True) + self.eps
                p_hat = alpha / a0
                alpha_prior = self.prior_concentration * p_hat

        kl_map = self._kl_map(alpha, alpha_prior).squeeze(1)          # [B,H,W]
        return (kl_map * valid.float()).sum() / valid.sum()

    # ----- convenience wrappers (alpha or logits) -----
    def data_from_logits(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        alpha = to_alpha_concentrations(logits, T=self.temperature, eps=self.eps)
        return self.data_from_alpha(alpha, target)

    def kl_from_logits(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        alpha = to_alpha_concentrations(logits, T=self.temperature, eps=self.eps)
        return self.kl_from_alpha(alpha, target)

    # ----- legacy wrappers (keep if other code still calls them) -----
    def forward_from_alpha(self, alpha: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.data_from_alpha(alpha, target)
        kl   = self.kl_from_alpha(alpha, target)
        return data, kl

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = to_alpha_concentrations(logits, T=self.temperature, eps=self.eps)
        return self.forward_from_alpha(alpha, target)
    

class DirichletNLLLoss(nn.Module):
    """Probabilistic loss for semantic segmentation with Dirichlet outputs.
    Supports two objectives:
      - "nll":  NLL of a smoothed one-hot target under Dir(alpha).
      - "dce":  Dirichlet cross-entropy E[-log p_y] = psi(alpha0) - psi(alpha_y).  (Recommended)

    Adds a KL regularizer KL[ Dir(alpha) || Dir(prior) ] with a symmetric prior to
    control evidence magnitude (alpha0) and prevent overconfidence.
    """
    def __init__(self, smooth=0.05, eps=1e-8):

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