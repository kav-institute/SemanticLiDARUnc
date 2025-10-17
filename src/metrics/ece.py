# ------------------------------------------------------------
# Expected Calibration Error (ECE) for segmentation
# ------------------------------------------------------------
import numpy as np, torch, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

class ECEAggregator:
    """
    Computes top-label ECE for segmentation.

    mode:
      - 'alpha'  : preds are Dirichlet alphas [B,C,H,W]; p = alpha / alpha0
      - 'logits' : preds are logits          [B,C,H,W]; p = softmax(logits)
      - 'probs'  : preds are probs           [B,C,H,W]; p re-normalized for safety

    binning: 'uniform' (equal-width) | 'adaptive' (equal-mass)
    plot_style: 'classic' | 'classic+hist' | 'gap'

    API:
      (ece, mce), stats = ece_eval.compute(save_plot_path="...", title="...")
      ece_eval.reset()
    """
    def __init__(self, n_bins=15, mode="alpha", ignore_index=None,
                 max_samples=None, seed=0, eps=1e-12,
                 binning: str = "uniform", plot_style: str = "classic"):
        assert binning in {"uniform", "adaptive"}
        assert plot_style in {"classic", "classic+hist", "gap"}
        assert mode in {"alpha", "logits", "probs"}
        assert n_bins >= 2
        self.n_bins = int(n_bins)
        self.mode = mode
        self.ignore_index = ignore_index
        self.max_samples = max_samples
        self.rng = np.random.default_rng(seed)
        self.eps = float(eps)
        self.binning = binning
        self.plot_style = plot_style
        self._conf = torch.empty(0, dtype=torch.float32)
        self._correct = torch.empty(0, dtype=torch.bool)
        self._seen = 0

    def reset(self):
        self._conf = torch.empty(0, dtype=torch.float32)
        self._correct = torch.empty(0, dtype=torch.bool)
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

        conf, pred = p.max(dim=1)         # [B,H,W], [B,H,W]
        lab = labels.long()

        valid = (lab != self.ignore_index) if self.ignore_index is not None \
                else torch.ones_like(lab, dtype=torch.bool)
        if not valid.any():
            return

        conf = conf[valid].detach().to(torch.float32).view(-1).clamp_(0, 1)
        corr = (pred[valid].view(-1) == lab[valid].view(-1))  # bool

        # Accumulate with optional reservoir cap
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

    # ---------- bin edges ----------
    def _bin_edges(self) -> np.ndarray:
        if self.binning == "uniform" or self._conf.numel() == 0:
            edges = np.linspace(0.0, 1.0, self.n_bins + 1, dtype=np.float32)
        else:
            # equal-mass (adaptive) binning from empirical CDF of confidences
            q = np.linspace(0.0, 1.0, self.n_bins + 1, dtype=np.float32)
            edges = np.quantile(self._conf.numpy(), q)
            edges[0], edges[-1] = 0.0, 1.0
            # guard against duplicates when many identical confidences
            edges = np.unique(edges)
            if edges.size < self.n_bins + 1:
                edges = np.linspace(0.0, 1.0, self.n_bins + 1, dtype=np.float32)
        edges[0] = 0.0; edges[-1] = 1.0
        return edges

    # ---------- per-bin stats ----------
    def _stats_df(self) -> pd.DataFrame:
        if self._conf.numel() == 0:
            return pd.DataFrame(columns=["low","high","center","width","n","pct","acc","conf"])
        conf = self._conf.numpy()
        corr = self._correct.numpy().astype(np.float32)
        edges = self._bin_edges()
        n       = np.histogram(conf, bins=edges)[0].astype(int)
        acc_s   = np.histogram(conf, bins=edges, weights=corr)[0]
        conf_s  = np.histogram(conf, bins=edges, weights=conf)[0]
        acc     = np.divide(acc_s, n, out=np.full_like(acc_s, np.nan, dtype=float), where=n>0)
        avg_conf= np.divide(conf_s, n, out=np.full_like(conf_s, np.nan, dtype=float), where=n>0)
        pct     = 100.0 * n / max(1, conf.size)
        lows, highs = edges[:-1], edges[1:]
        centers = 0.5 * (lows + highs)
        widths  = highs - lows
        return pd.DataFrame({
            "low": lows, "high": highs, "center": centers, "width": widths,
            "n": n, "pct": pct, "acc": acc, "conf": avg_conf
        })

    # ---------- compute ECE & plot ----------
    def compute(self, save_plot_path: str | None = None,
                title: str = "Reliability Diagram", dpi: int = 200):
        """
        Returns ((ece, mce), stats_df). If save_plot_path is given, saves the plot.
        """
        stats = self._stats_df()
        if stats.empty or stats["n"].sum() == 0:
            return (float("nan"), float("nan")), stats

        w   = stats["n"].to_numpy().astype(np.float64)
        w   = w / max(1, w.sum())
        acc = np.nan_to_num(stats["acc"].to_numpy(), nan=0.0)
        conf= np.nan_to_num(stats["conf"].to_numpy(), nan=0.0)
        gap = np.abs(acc - conf)
        ece = float(np.sum(w * gap))
        mce = float(np.max(gap))

        if save_plot_path is not None:
            fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=dpi)
            x      = stats["center"].to_numpy()
            widths = stats["width"].to_numpy()

            if self.plot_style in {"classic", "classic+hist"}:
                ax.plot([0, 1], [0, 1], label="perfect calibration", linewidth=2)
                ax.plot(x, acc, marker="o", label="accuracy")
                ax.plot(x, conf, marker="x", linestyle="--", label="avg. confidence")
                ax.set_xlabel("Confidence (bin center)")
                ax.set_ylabel("Accuracy / Avg. Confidence")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
                ax.set_title(f"{title}\nECE={ece:.4f}  |  MCE={mce:.4f}")
                if self.plot_style == "classic+hist":
                    ax2 = ax.twinx()
                    widths = (stats["high"] - stats["low"]).to_numpy()  # exact bin widths (works for adaptive too)
                    mass = stats["n"].to_numpy() / max(1, int(self._conf.numel()))
                    ax2.bar(stats["center"].to_numpy(), mass, width=widths*0.9, alpha=0.25, color="#6baed6", edgecolor="none")

                    ax2.set_ylim(0, 1)                                   # fix to [0, 1]
                    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
                    ax2.set_ylabel("Bin mass (% of samples)", color="gray")
                    ax2.tick_params(axis="y", colors="gray")
                ax.legend(loc="lower right", frameon=True)

            elif self.plot_style == "gap":
                signed = conf - acc  # >0 over-confident, <0 under-confident
                colors = np.where(signed >= 0, "tab:red", "tab:green")
                ax.axhline(0.0, color="k", linewidth=1)
                ax.bar(x, signed, width=widths*0.9, color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Confidence (bin center)")
                ax.set_ylabel("conf - acc  (positive = over-confident)")
                ax.set_title(f"{title}\nECE={ece:.4f}  |  MCE={mce:.4f}")
                ax.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(save_plot_path, bbox_inches="tight", dpi=dpi)
            plt.close(fig)

        return (ece, mce), stats



# # ------------------------------------------------------------
# # Expected Calibration Error (ECE) Aggregator
# # - Accumulates per-batch data
# # - Computes ECE at epoch end
# # - Saves a reliability diagram
# # ------------------------------------------------------------
# import math, numpy as np, torch
# import matplotlib.pyplot as plt
# import pandas as pd

# class ECEAggregator:
#     """
#     Expected Calibration Error for segmentation.

#     Mode:
#       - 'alpha'  : preds are Dirichlet alphas [B,C,H,W]; p = alpha / alpha0
#       - 'logits' : preds are logits [B,C,H,W]; p = softmax(logits)
#       - 'probs'  : preds are probs [B,C,H,W]; p used as-is (re-normalized for safety)

#     API:
#       ece = ece_eval.compute(save_plot_path="...", title="...")[0]
#       ece_eval.reset()
#     """

#     # binning: 'uniform' (your current) or 'adaptive' (equal-mass)
#     # plot_style: 'classic' | 'classic+hist' | 'gap'
#     def __init__(self, n_bins=15, mode="alpha", ignore_index=None,
#                 max_samples=None, seed=0, eps=1e-12,
#                 binning: str = "uniform", plot_style: str = "classic"):
#         ...
#         assert binning in {"uniform", "adaptive"}
#         assert plot_style in {"classic", "classic+hist", "gap"}
#         assert mode in {"alpha", "logits", "probs"}
#         assert n_bins >= 2
#         self.binning = binning
#         self.plot_style = plot_style
#         self.n_bins = int(n_bins)
#         self.mode = mode
#         self.ignore_index = ignore_index
#         self.max_samples = max_samples
#         self.rng = np.random.default_rng(seed)
#         self.eps = float(eps)
#         self._conf = torch.empty(0, dtype=torch.float32)
#         self._correct = torch.empty(0, dtype=torch.uint8)
#         self._seen = 0

#     def _bin_edges(self) -> np.ndarray:
#         if self.binning == "uniform" or self._conf.numel() == 0:
#             edges = np.linspace(0.0, 1.0, self.n_bins + 1, dtype=np.float32)
#         else:
#             # equal-mass (adaptive) binning from empirical CDF of confidences
#             q = np.linspace(0.0, 1.0, self.n_bins + 1, dtype=np.float32)
#             edges = np.quantile(self._conf.numpy(), q)
#             edges[0], edges[-1] = 0.0, 1.0
#             # guard against duplicates when many identical confidences
#             edges = np.unique(edges)
#             if edges.size < self.n_bins + 1:
#                 edges = np.linspace(0.0, 1.0, self.n_bins + 1, dtype=np.float32)
#         edges[0] = 0.0; edges[-1] = 1.0
#         return edges

#     def reset(self):
#         self._conf = torch.empty(0, dtype=torch.float32)
#         self._correct = torch.empty(0, dtype=torch.uint8)
#         self._seen = 0

#     # ---------- internal: convert preds -> probs ----------
#     @torch.no_grad()
#     def _to_probs(self, preds: torch.Tensor) -> torch.Tensor:
#         if self.mode == "alpha":
#             a0 = preds.sum(dim=1, keepdim=True)
#             p = preds / (a0 + self.eps)
#         elif self.mode == "logits":
#             p = preds.softmax(dim=1)
#         else:  # 'probs'
#             p = preds.clamp_min(0)
#             p = p / p.sum(dim=1, keepdim=True).clamp_min(self.eps)
#         return p

#     @torch.no_grad()
#     def update(self, preds: torch.Tensor, labels: torch.Tensor):
#         """
#         preds : [B,C,H,W] (alpha/logits/probs depending on mode)
#         labels: [B,H,W] (int class ids)
#         """
#         assert preds.dim() == 4 and labels.dim() == 3
#         p = self._to_probs(preds)  # [B,C,H,W]

#         conf, pred = p.max(dim=1)               # [B,H,W], [B,H,W]
#         lab = labels.long()

#         if self.ignore_index is not None:
#             valid = (lab != self.ignore_index)
#         else:
#             valid = torch.ones_like(lab, dtype=torch.bool)

#         if not valid.any():
#             return

#         conf = conf[valid].detach().to(torch.float32).view(-1).clamp_(0, 1)
#         corr = (pred[valid].view(-1) == lab[valid].view(-1)).to(torch.uint8)

#         # Accumulate with optional reservoir-style cap (like your other class)
#         if self.max_samples is None:
#             self._conf = torch.cat([self._conf, conf.cpu()], dim=0)
#             self._correct = torch.cat([self._correct, corr.cpu()], dim=0)
#             self._seen += conf.numel()
#             return

#         n_new = conf.numel()
#         self._seen += n_new
#         if self._conf.numel() < self.max_samples:
#             take = min(self.max_samples - self._conf.numel(), n_new)
#             if take < n_new:
#                 idx = torch.from_numpy(self.rng.choice(n_new, size=take, replace=False))
#                 conf, corr = conf[idx], corr[idx]
#             self._conf = torch.cat([self._conf, conf.cpu()], dim=0)
#             self._correct = torch.cat([self._correct, corr.cpu()], dim=0)
#         else:
#             p_keep = min(1.0, float(self.max_samples) / float(self._seen + 1e-9))
#             keep = torch.from_numpy(self.rng.random(n_new) < p_keep)
#             if keep.any():
#                 conf, corr = conf[keep], corr[keep]
#                 replace_idx = torch.from_numpy(
#                     self.rng.choice(self.max_samples, size=conf.numel(), replace=False)
#                 )
#                 self._conf[replace_idx] = conf.cpu()
#                 self._correct[replace_idx] = corr.cpu()

#     # ---------- compute ECE ----------
#     # def _bin_edges(self) -> np.ndarray:
#     #     edges = np.linspace(0.0, 1.0, self.n_bins + 1, dtype=np.float32)
#     #     edges[0] = 0.0; edges[-1] = 1.0
#     #     return edges

#     def _stats_df(self) -> pd.DataFrame:
#         if self._conf.numel() == 0:
#             return pd.DataFrame(columns=["low","high","center","n","pct","acc","conf"])
#         conf = self._conf.numpy()
#         corr = self._correct.numpy().astype(np.float32)
#         edges = self._bin_edges()
#         n     = np.histogram(conf, bins=edges)[0].astype(int)
#         acc_s = np.histogram(conf, bins=edges, weights=corr)[0]
#         conf_s= np.histogram(conf, bins=edges, weights=conf)[0]
#         acc = np.divide(acc_s, n, out=np.full_like(acc_s, np.nan, dtype=float), where=n>0)
#         avg_conf = np.divide(conf_s, n, out=np.full_like(conf_s, np.nan, dtype=float), where=n>0)
#         pct = 100.0 * n / max(1, conf.size)
#         centers = 0.5*(edges[:-1] + edges[1:])
#         return pd.DataFrame({
#             "low": edges[:-1], "high": edges[1:], "center": centers,
#             "n": n, "pct": pct, "acc": acc, "conf": avg_conf
#         })
        
#     def compute(self, save_plot_path: str | None = None,
#                 title: str = "Reliability Diagram", dpi: int = 200):
#         stats = self._stats_df()
#         if stats.empty or stats["n"].sum() == 0:
#             return float("nan"), stats

#         w   = stats["n"].to_numpy().astype(np.float64)
#         w   = w / max(1, w.sum())
#         acc = np.nan_to_num(stats["acc"].to_numpy(), nan=0.0)
#         conf= np.nan_to_num(stats["conf"].to_numpy(), nan=0.0)
#         gap = np.abs(acc - conf)
#         ece = float(np.sum(w * gap))
#         mce = float(np.max(gap))  # maximum calibration error

#         if save_plot_path is not None:
#             import matplotlib.pyplot as plt
#             fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=dpi)
#             x = stats["center"].to_numpy()

#             if self.plot_style in {"classic", "classic+hist"}:
#                 ax.plot([0, 1], [0, 1], label="perfect calibration", linewidth=2)
#                 ax.plot(x, acc, marker="o", label="accuracy")
#                 ax.plot(x, conf, marker="x", linestyle="--", label="avg. confidence")
#                 ax.set_xlabel("Confidence (bin center)"); ax.set_ylabel("Accuracy / Avg. Confidence")
#                 ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
#                 ttl = f"{title}\nECE={ece:.4f}  |  MCE={mce:.4f}"
#                 ax.set_title(ttl)

#                 if self.plot_style == "classic+hist":
#                     ax2 = ax.twinx()
#                     ax2.bar(x, stats["n"].to_numpy() / max(1, int(self._conf.numel())),
#                             width=(x[1]-x[0] if len(x)>1 else 1.0)*0.9,
#                             alpha=0.25, edgecolor="none")
#                     ax2.set_ylabel("Bin mass (fraction)", color="gray")
#                     ax2.tick_params(axis='y', colors='gray')

#                 ax.legend(loc="lower right", frameon=True)

#             elif self.plot_style == "gap":
#                 # signed gap: positive => over-confident (conf > acc), negative => under-confident
#                 signed = conf - acc
#                 colors = np.where(signed >= 0, "tab:red", "tab:green")
#                 ax.axhline(0.0, color="k", linewidth=1)
#                 ax.bar(x, signed, width=(x[1]-x[0] if len(x)>1 else 1.0)*0.9, color=colors)
#                 ax.set_xlim(0, 1)
#                 ax.set_xlabel("Confidence (bin center)")
#                 ax.set_ylabel("conf - acc  (positive = over-confident)")
#                 ax.set_title(f"{title}\nECE={ece:.4f}  |  MCE={mce:.4f}")
#                 ax.grid(True, alpha=0.3)

#             fig.tight_layout()
#             fig.savefig(save_plot_path, bbox_inches="tight", dpi=dpi)
#             plt.close(fig)

#         # return both; callers can ignore MCE if not needed
#         return (ece, mce), stats

#     # def compute(self, save_plot_path: str | None = None, title: str = "Reliability Diagram", dpi: int = 200):
#     #     """
#     #     Returns (ece_scalar, stats_df). If save_plot_path is given, saves the plot.
#     #     """
#     #     stats = self._stats_df()
#     #     if stats.empty or stats["n"].sum() == 0:
#     #         return float("nan"), stats

#     #     w = stats["n"].to_numpy().astype(np.float64)
#     #     w = w / max(1, w.sum())
#     #     gap = np.abs(stats["acc"].to_numpy() - stats["conf"].to_numpy())
#     #     ece = float(np.nansum(w * gap))

#     #     if save_plot_path is not None:
#     #         fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=dpi)
#     #         x = stats["center"].to_numpy()

#     #         # perfect calibration
#     #         ax.plot([0, 1], [0, 1], label="perfect calibration")

#     #         # accuracy per bin
#     #         acc = np.nan_to_num(stats["acc"].to_numpy(), nan=0.0)
#     #         ax.plot(x, acc, marker="o", label="accuracy")

#     #         # mean confidence per bin
#     #         conf = np.nan_to_num(stats["conf"].to_numpy(), nan=0.0)
#     #         ax.plot(x, conf, marker="x", linestyle="--", label="avg. confidence")

#     #         ax.set_xlim(0, 1); ax.set_ylim(0, 1)
#     #         ax.set_xlabel("Confidence (bin center)")
#     #         ax.set_ylabel("Accuracy / Avg. Confidence")
#     #         ax.set_title(f"{title}\nECE = {ece:.4f}")
#     #         ax.grid(True, alpha=0.3)

#     #         # NEW: legend
#     #         ax.legend(loc="lower right", frameon=True)

#     #         fig.tight_layout()
#     #         fig.savefig(save_plot_path, bbox_inches="tight", dpi=dpi)
#     #         plt.close(fig)

#     #     return ece, stats
