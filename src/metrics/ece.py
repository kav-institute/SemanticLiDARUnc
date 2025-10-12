# ------------------------------------------------------------
# Expected Calibration Error (ECE) Aggregator
# - Accumulates per-batch data
# - Computes ECE at epoch end
# - Saves a reliability diagram
# ------------------------------------------------------------
import math, numpy as np, torch
import matplotlib.pyplot as plt
import pandas as pd

class ECEAggregator:
    """
    Expected Calibration Error for segmentation.

    Mode:
      - 'alpha'  : preds are Dirichlet alphas [B,C,H,W]; p = alpha / alpha0
      - 'logits' : preds are logits [B,C,H,W]; p = softmax(logits)
      - 'probs'  : preds are probs [B,C,H,W]; p used as-is (re-normalized for safety)

    API:
      ece = ece_eval.compute(save_plot_path="...", title="...")[0]
      ece_eval.reset()
    """

    def __init__(
        self,
        n_bins: int = 15,
        mode: str = "alpha",
        ignore_index: int | None = None,
        max_samples: int | None = None,  # cap memory; None = keep all
        seed: int = 0,
        eps: float = 1e-12,
    ):
        assert mode in {"alpha", "logits", "probs"}
        assert n_bins >= 2
        self.n_bins = int(n_bins)
        self.mode = mode
        self.ignore_index = ignore_index
        self.max_samples = max_samples
        self.rng = np.random.default_rng(seed)
        self.eps = float(eps)
        self._conf = torch.empty(0, dtype=torch.float32)
        self._correct = torch.empty(0, dtype=torch.uint8)
        self._seen = 0

    def reset(self):
        self._conf = torch.empty(0, dtype=torch.float32)
        self._correct = torch.empty(0, dtype=torch.uint8)
        self._seen = 0

    # ---------- internal: convert preds -> probs ----------
    @torch.no_grad()
    def _to_probs(self, preds: torch.Tensor) -> torch.Tensor:
        if self.mode == "alpha":
            a0 = preds.sum(dim=1, keepdim=True)
            p = preds / (a0 + self.eps)
        elif self.mode == "logits":
            p = preds.softmax(dim=1)
        else:  # 'probs'
            p = preds.clamp_min(0)
            p = p / p.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return p

    @torch.no_grad()
    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        preds : [B,C,H,W] (alpha/logits/probs depending on mode)
        labels: [B,H,W] (int class ids)
        """
        assert preds.dim() == 4 and labels.dim() == 3
        p = self._to_probs(preds)  # [B,C,H,W]

        conf, pred = p.max(dim=1)               # [B,H,W], [B,H,W]
        lab = labels.long()

        if self.ignore_index is not None:
            valid = (lab != self.ignore_index)
        else:
            valid = torch.ones_like(lab, dtype=torch.bool)

        if not valid.any():
            return

        conf = conf[valid].detach().to(torch.float32).view(-1).clamp_(0, 1)
        corr = (pred[valid].view(-1) == lab[valid].view(-1)).to(torch.uint8)

        # Accumulate with optional reservoir-style cap (like your other class)
        if self.max_samples is None:
            self._conf = torch.cat([self._conf, conf.cpu()], dim=0)
            self._correct = torch.cat([self._correct, corr.cpu()], dim=0)
            self._seen += conf.numel()
            return

        n_new = conf.numel()
        self._seen += n_new
        if self._conf.numel() < self.max_samples:
            take = min(self.max_samples - self._conf.numel(), n_new)
            if take < n_new:
                idx = torch.from_numpy(self.rng.choice(n_new, size=take, replace=False))
                conf, corr = conf[idx], corr[idx]
            self._conf = torch.cat([self._conf, conf.cpu()], dim=0)
            self._correct = torch.cat([self._correct, corr.cpu()], dim=0)
        else:
            p_keep = min(1.0, float(self.max_samples) / float(self._seen + 1e-9))
            keep = torch.from_numpy(self.rng.random(n_new) < p_keep)
            if keep.any():
                conf, corr = conf[keep], corr[keep]
                replace_idx = torch.from_numpy(
                    self.rng.choice(self.max_samples, size=conf.numel(), replace=False)
                )
                self._conf[replace_idx] = conf.cpu()
                self._correct[replace_idx] = corr.cpu()

    # ---------- compute ECE ----------
    def _bin_edges(self) -> np.ndarray:
        edges = np.linspace(0.0, 1.0, self.n_bins + 1, dtype=np.float32)
        edges[0] = 0.0; edges[-1] = 1.0
        return edges

    def _stats_df(self) -> pd.DataFrame:
        if self._conf.numel() == 0:
            return pd.DataFrame(columns=["low","high","center","n","pct","acc","conf"])
        conf = self._conf.numpy()
        corr = self._correct.numpy().astype(np.float32)
        edges = self._bin_edges()
        n     = np.histogram(conf, bins=edges)[0].astype(int)
        acc_s = np.histogram(conf, bins=edges, weights=corr)[0]
        conf_s= np.histogram(conf, bins=edges, weights=conf)[0]
        acc = np.divide(acc_s, n, out=np.full_like(acc_s, np.nan, dtype=float), where=n>0)
        avg_conf = np.divide(conf_s, n, out=np.full_like(conf_s, np.nan, dtype=float), where=n>0)
        pct = 100.0 * n / max(1, conf.size)
        centers = 0.5*(edges[:-1] + edges[1:])
        return pd.DataFrame({
            "low": edges[:-1], "high": edges[1:], "center": centers,
            "n": n, "pct": pct, "acc": acc, "conf": avg_conf
        })

    def compute(self, save_plot_path: str | None = None, title: str = "Reliability Diagram", dpi: int = 200):
        """
        Returns (ece_scalar, stats_df). If save_plot_path is given, saves the plot.
        """
        stats = self._stats_df()
        if stats.empty or stats["n"].sum() == 0:
            return float("nan"), stats

        w = stats["n"].to_numpy().astype(np.float64)
        w = w / max(1, w.sum())
        gap = np.abs(stats["acc"].to_numpy() - stats["conf"].to_numpy())
        ece = float(np.nansum(w * gap))

        if save_plot_path is not None:
            fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=dpi)
            x = stats["center"].to_numpy()

            # perfect calibration
            ax.plot([0, 1], [0, 1], label="perfect calibration")

            # accuracy per bin
            acc = np.nan_to_num(stats["acc"].to_numpy(), nan=0.0)
            ax.plot(x, acc, marker="o", label="accuracy")

            # mean confidence per bin
            conf = np.nan_to_num(stats["conf"].to_numpy(), nan=0.0)
            ax.plot(x, conf, marker="x", linestyle="--", label="avg. confidence")

            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("Confidence (bin center)")
            ax.set_ylabel("Accuracy / Avg. Confidence")
            ax.set_title(f"{title}\nECE = {ece:.4f}")
            ax.grid(True, alpha=0.3)

            # NEW: legend
            ax.legend(loc="lower right", frameon=True)

            fig.tight_layout()
            fig.savefig(save_plot_path, bbox_inches="tight", dpi=dpi)
            plt.close(fig)

        return ece, stats
