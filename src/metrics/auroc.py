# ---------- AUROC Aggregator (adds score_override) ----------
import numpy as np
import torch
from torch.special import digamma
import math
import matplotlib.pyplot as plt

class AUROCAggregator:
    """
    AUROC for error detection in segmentation.

    mode: 'alpha' | 'logits' | 'probs'
    score: 'entropy' | 'entropy_norm' | 'mi' | 'mi_norm' | '1-maxprob'
    score_override: optional [B,H,W] tensor; if given, it's used verbatim.
    """
    def __init__(self, mode="alpha", score="entropy_norm",
                 ignore_index=None, max_samples=None, seed=0, eps=1e-12):
        assert mode in {"alpha","logits","probs"}
        assert score in {"entropy","entropy_norm","mi","mi_norm","1-maxprob"}
        self.mode, self.score = mode, score
        self.ignore_index = ignore_index
        self.max_samples = max_samples
        self.rng = np.random.default_rng(seed)
        self.eps = float(eps)
        self._scores = torch.empty(0, dtype=torch.float32)
        self._is_error = torch.empty(0, dtype=torch.uint8)
        self._seen = 0

    def reset(self):
        self._scores = torch.empty(0, dtype=torch.float32)
        self._is_error = torch.empty(0, dtype=torch.uint8)
        self._seen = 0

    @torch.no_grad()
    def _to_probs(self, preds: torch.Tensor) -> torch.Tensor:
        if self.mode == "alpha":
            a0 = preds.sum(dim=1, keepdim=True)
            p = preds / (a0 + self.eps)
        elif self.mode == "logits":
            p = preds.softmax(dim=1)
        else:
            p = preds.clamp_min(0)
            p = p / p.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return p

    @torch.no_grad()
    def _uncertainty_score(self, preds: torch.Tensor) -> torch.Tensor:
        if self.score in {"entropy","entropy_norm","1-maxprob"} or self.mode != "alpha":
            p = self._to_probs(preds)
            if self.score == "1-maxprob":
                return 1.0 - p.max(dim=1).values
            H = -(p.clamp_min(self.eps) * p.clamp_min(self.eps).log()).sum(dim=1)
            return H / math.log(p.size(1)) if self.score == "entropy_norm" else H
        else:
            # Dirichlet MI
            alpha = preds
            a0 = alpha.sum(dim=1, keepdim=True) + self.eps
            p = alpha / a0
            H = -(p.clamp_min(self.eps) * p.clamp_min(self.eps).log()).sum(dim=1)
            term = digamma(alpha + 1.0) - digamma(a0 + 1.0)
            EH = -(p * term).sum(dim=1)
            MI = H - EH
            return MI / math.log(alpha.size(1)) if self.score == "mi_norm" else MI

    @staticmethod
    def _roc_from_scores(scores: np.ndarray, is_error: np.ndarray):
        order = np.argsort(-scores)
        y = is_error[order].astype(np.float64)
        s = scores[order].astype(np.float64)
        P, N = y.sum(), y.size - y.sum()
        if P == 0 or N == 0:
            return None, None, None, float("nan")
        tps = np.cumsum(y); fps = np.cumsum(1.0 - y)
        tpr = np.concatenate(([0.0], tps / P, [1.0]))
        fpr = np.concatenate(([0.0], fps / N, [1.0]))
        thr = np.concatenate(([np.inf], s, [-np.inf]))
        auroc = float(np.trapz(tpr, fpr))
        return fpr, tpr, thr, auroc

    @torch.no_grad()
    def update(self, preds: torch.Tensor, labels: torch.Tensor,
            score_override: torch.Tensor | None = None):
        """
        preds : [B,C,H,W]   (alpha/logits/probs depending on mode)
        labels: [B,H,W] or [B,1,H,W]
        score_override: [B,H,W] optional custom uncertainty (e.g., MC-MI)
        """
        # shapes
        assert preds.dim() == 4 and (labels.dim() == 3 or (labels.dim() == 4 and labels.size(1) == 1)), \
            "labels must be [B,H,W] or [B,1,H,W]"

        # squeeze labels if needed
        if labels.dim() == 4:  # [B,1,H,W] -> [B,H,W]
            labels = labels[:, 0]

        # probs and predictions
        p = self._to_probs(preds)                 # [B,C,H,W]
        pred = p.argmax(dim=1)                    # [B,H,W]

        lab = labels.long()
        valid = torch.ones_like(lab, dtype=torch.bool) if self.ignore_index is None else (lab != self.ignore_index)
        if not valid.any():
            return

        # 1) uncertainty score on valid pixels
        if score_override is None:
            score_map = self._uncertainty_score(preds)   # [B,H,W]
        else:
            score_map = score_override                   # [B,H,W]
        score = score_map[valid].reshape(-1).to(torch.float32)

        # 2) error flags on the SAME valid pixels
        is_err = (pred != lab)
        is_err = is_err[valid].reshape(-1).to(torch.uint8)

        # reservoir-style cap
        if self.max_samples is None:
            self._scores   = torch.cat([self._scores,   score.cpu()], dim=0)
            self._is_error = torch.cat([self._is_error, is_err.cpu()], dim=0)
            self._seen += score.numel()
            return

        n_new = score.numel()
        self._seen += n_new
        if self._scores.numel() < self.max_samples:
            take = min(self.max_samples - self._scores.numel(), n_new)
            if take < n_new:
                idx = torch.from_numpy(self.rng.choice(n_new, size=take, replace=False))
                score, is_err = score[idx], is_err[idx]
            self._scores   = torch.cat([self._scores,   score.cpu()], dim=0)
            self._is_error = torch.cat([self._is_error, is_err.cpu()], dim=0)
        else:
            p_keep = min(1.0, float(self.max_samples) / float(self._seen + 1e-9))
            keep = torch.from_numpy(self.rng.random(n_new) < p_keep)
            if keep.any():
                score, is_err = score[keep], is_err[keep]
                replace_idx = torch.from_numpy(
                    self.rng.choice(self.max_samples, size=score.numel(), replace=False)
                )
                self._scores[replace_idx]   = score.cpu()
                self._is_error[replace_idx] = is_err.cpu()

        # optional safety check
        assert self._scores.numel() == self._is_error.numel(), \
            f"length mismatch: scores={self._scores.numel()} vs errs={self._is_error.numel()}"


    def compute(self, save_plot_path: str | None = None, title: str = "ROC: error detection", dpi: int = 200):
        if self._scores.numel() == 0:
            return float("nan"), {}
        scores = self._scores.numpy()
        is_error = self._is_error.numpy()
        fpr, tpr, thr, auroc = self._roc_from_scores(scores, is_error)
        if np.isnan(auroc):
            return auroc, {}
        if save_plot_path is not None:
            fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=dpi)
            ax.plot([0,1], [0,1]); ax.plot(fpr, tpr)
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title(f"{title}\nAUROC = {auroc:.4f}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout(); fig.savefig(save_plot_path, bbox_inches="tight", dpi=dpi); plt.close(fig)
        return auroc, {"fpr": fpr, "tpr": tpr, "thresholds": thr}
