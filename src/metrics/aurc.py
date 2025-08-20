import math
import numpy as np
import torch

# ---------- core RC utilities ----------

def rc_curve_stats(risks: np.ndarray, confids: np.ndarray):
    assert risks.ndim == 1 and confids.ndim == 1 and risks.size == confids.size
    n = risks.size
    idx = np.argsort(confids)  # ascending confidence → discard from left

    coverage = n
    error_sum = float(risks[idx].sum())

    coverages = [coverage / n]
    selective_risks = [error_sum / n]
    weights = []

    tmp_w = 0
    for i in range(0, n - 1):
        coverage -= 1
        error_sum -= float(risks[idx[i]])
        tmp_w += 1
        if i == 0 or confids[idx[i]] != confids[idx[i - 1]]:
            coverages.append(coverage / n)
            selective_risks.append(error_sum / (n - 1 - i))
            weights.append(tmp_w / n)
            tmp_w = 0

    if tmp_w > 0:
        coverages.append(0.0)
        selective_risks.append(selective_risks[-1])
        weights.append(tmp_w / n)

    return np.array(coverages), np.array(selective_risks), np.array(weights)


def aurc_from_risks_confids(risks: np.ndarray, confids: np.ndarray):
    coverages, rc_risks, weights = rc_curve_stats(risks, confids)
    aurc = float(np.sum((rc_risks[:-1] + rc_risks[1:]) * 0.5 * weights))
    n = risks.size
    selective_risks_opt = np.cumsum(np.sort(risks)) / np.arange(1, n + 1)
    aurc_opt = float(selective_risks_opt.sum() / n)
    eaurc = aurc - aurc_opt
    return aurc, eaurc, coverages, rc_risks


def entropy_from_probs(probs: torch.Tensor, eps=1e-12):
    """Normalized entropy from probs [B,C,H,W] → [B,H,W] in [0,1]."""
    C = probs.shape[1]
    p = probs.clamp_min(eps)
    ent = -(p * torch.log(p)).sum(dim=1)            # [B,H,W]
    ent = ent / math.log(C)
    return ent


# ===== flatten (pixel-level across the whole batch) =====

@torch.no_grad()
def _flatten_batch(
    outputs_semantic: torch.Tensor,  # [B,C,H,W] softmax probs
    semantic: torch.Tensor,          # [B,1,H,W] GT labels
    ent_mc: torch.Tensor = None,     # [B,H,W] (optional)
    ignore_index: int = 255,
    use_max_prob_confidence: bool = False
):
    pred = outputs_semantic.argmax(dim=1)  # [B,H,W]
    gt   = semantic.squeeze(1)             # [B,H,W]
    valid = gt != ignore_index
    error_mask = (pred != gt) & valid

    if use_max_prob_confidence:
        conf = outputs_semantic.max(dim=1).values  # [B,H,W]
    else:
        if ent_mc is None:
            ent_mc = entropy_from_probs(outputs_semantic)
        conf = 1.0 - ent_mc.clamp(0, 1)           # [B,H,W]

    confids = conf[valid].float().cpu().numpy()
    risks   = error_mask[valid].to(torch.float32).cpu().numpy()
    return risks, confids, valid, pred  # pred, valid useful for visualization


# ======= BATCH GETTERS =======

def compute_batch_uncertainty_metrics(
    outputs_semantic: torch.Tensor,  # [B,C,H,W]
    semantic: torch.Tensor,          # [B,1,H,W]
    ent_mc: torch.Tensor = None,     # [B,H,W] normalized
    ignore_index: int = 255,
    use_max_prob_confidence: bool = False,
    ks = (1,2,5,10,20,30,40,50)
):
    """Return AURC/E-AURC and full curve data for ONE BATCH (all pixels in it)."""
    risks, confids, valid, pred = _flatten_batch(
        outputs_semantic, semantic, ent_mc, ignore_index, use_max_prob_confidence
    )
    aurc, eaurc, coverages, rc_risks = aurc_from_risks_confids(risks, confids)

    # Top-k% error recall data
    idx = np.argsort(confids)  # low conf first (high uncertainty)
    risks_sorted = risks[idx]
    total_err = risks.sum()
    recalls = []
    for k in ks:
        m = max(1, int(risks.size * k / 100))
        recalls.append(float(risks_sorted[:m].sum() / max(total_err, 1)))

    return {
        "AURC": aurc,
        "EAURC": eaurc,
        "coverages": coverages,     # x for RC curve
        "rc_risks": rc_risks,       # y for RC curve
        "ks": np.asarray(ks),       # x for recall curve
        "recalls": np.asarray(recalls),
        "num_pixels": int(risks.size),
        "num_errors": int(total_err),
        # extras that help visualize
        "valid_mask_shape": tuple(valid.shape),
    }


# ======= BATCH VISUALIZERS =======

def plot_batch_rc_curves(batch_metrics, title_prefix="Batch", save_path=None, dpi=150, show=False, close=True):
    import matplotlib.pyplot as plt

    # Risk–Coverage
    fig = plt.figure()
    plt.plot(batch_metrics["coverages"], batch_metrics["rc_risks"])
    plt.xlabel("Coverage (fraction kept)")
    plt.ylabel("Risk (error rate on kept)")
    plt.title(f"{title_prefix} Risk–Coverage")
    plt.grid(True, linestyle=":")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    if close:
        plt.close(fig)

    # Top-k% Error Recall
    fig = plt.figure()
    plt.plot(batch_metrics["ks"], batch_metrics["recalls"], marker="o")
    plt.xlabel("Top-k% most-uncertain pixels")
    plt.ylabel("Recall of errors")
    plt.title(f"{title_prefix} Uncertainty Error-Recall")
    plt.grid(True, linestyle=":")
    fig.tight_layout()
    if save_path:
        stem = save_path.rsplit(".", 1)[0]
        fig.savefig(f"{stem}_error_recall.png", bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    if close:
        plt.close(fig)


def visualize_batch_maps(
    ent_mc: torch.Tensor,           # [B,H,W] normalized entropy
    outputs_semantic: torch.Tensor, # [B,C,H,W]
    semantic: torch.Tensor,         # [B,1,H,W]
    index: int = 0,                 # which item in the batch to show
    ignore_index: int = 255,
    vmax_entropy: float = 1.0,
    save_path: str | None = None,
    dpi: int = 150,
    show: bool = False,
    close: bool = True
):
    """Quick visual maps for ONE item in the batch: entropy, prediction, errors."""
    import matplotlib.pyplot as plt
    pred = outputs_semantic.argmax(dim=1)  # [B,H,W]
    gt   = semantic.squeeze(1)             # [B,H,W]
    valid = gt != ignore_index
    err = (pred != gt) & valid

    e = ent_mc[index].detach().cpu().numpy()
    p = pred[index].detach().cpu().numpy()
    g = gt[index].detach().cpu().numpy()
    ev = err[index].detach().cpu().numpy()

    fig, axs = plt.subplots(1, 4, figsize=(16,4))
    im0 = axs[0].imshow(e, vmin=0.0, vmax=vmax_entropy)
    axs[0].set_title("Entropy (normalized)")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    axs[1].imshow(p, interpolation="nearest")
    axs[1].set_title("Prediction (argmax)")

    axs[2].imshow(g, interpolation="nearest")
    axs[2].set_title("Ground truth")

    axs[3].imshow(ev, cmap="gray", interpolation="nearest")
    axs[3].set_title("Error mask")
    for ax in axs: ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    if close:
        plt.close(fig)


# ---------- dataset-level aggregator ----------

class UncertaintyAggregator:
    """
    Stream batches with add_batch(...), then call finalize() to get AURC/E-AURC
    and optional plots. Works at pixel level across the *whole* eval set.

    Memory controls:
      - By default we store ALL pixels (exact AURC). This can be large.
      - Set `reservoir_size` to keep at most that many random pixels
        (approximate AURC with uniform subsampling).

    Example:
      agg = UncertaintyAggregator(ignore_index=255, use_max_prob_confidence=False,
                                  reservoir_size=2_000_000, seed=0)
      for probs, labels in loader:
          agg.add_batch(probs, labels)             # (optionally ent_mc=...)
      res = agg.finalize(make_plots=True, save_dir=".../eval", epoch=epoch)
    """

    def __init__(self,
                 ignore_index: int = 255,
                 use_max_prob_confidence: bool = False,
                 reservoir_size: int | None = None,
                 seed: int | None = None):
        self.ignore_index = int(ignore_index)
        self.use_max_prob_confidence = bool(use_max_prob_confidence)

        # Exact mode (lists) or memory-bounded reservoir (np arrays)
        self.reservoir_size = None if reservoir_size is None else int(reservoir_size)
        self._rng = np.random.RandomState(seed) if seed is not None else np.random

        # storage
        self._risks_list = []     # used when reservoir_size is None
        self._confids_list = []
        self._risks_res = None    # used when reservoir_size is not None
        self._confids_res = None

    def _append_exact(self, risks_np: np.ndarray, conf_np: np.ndarray):
        self._risks_list.append(risks_np)
        self._confids_list.append(conf_np)

    def _append_reservoir(self, risks_np: np.ndarray, conf_np: np.ndarray):
        if self._risks_res is None:
            if risks_np.size <= self.reservoir_size:
                self._risks_res = risks_np.copy()
                self._confids_res = conf_np.copy()
                return
            # initialize by downsampling first chunk
            idx = self._rng.choice(risks_np.size, size=self.reservoir_size, replace=False)
            self._risks_res = risks_np[idx]
            self._confids_res = conf_np[idx]
            return

        # merge with existing reservoir, then downsample back to reservoir_size
        merged_r = np.concatenate([self._risks_res, risks_np], axis=0)
        merged_c = np.concatenate([self._confids_res, conf_np], axis=0)
        if merged_r.size <= self.reservoir_size:
            self._risks_res, self._confids_res = merged_r, merged_c
        else:
            idx = self._rng.choice(merged_r.size, size=self.reservoir_size, replace=False)
            self._risks_res = merged_r[idx]
            self._confids_res = merged_c[idx]

    @torch.no_grad()
    def add_batch(
        self,
        outputs_semantic: torch.Tensor,  # [B,C,H,W] softmax probs
        semantic: torch.Tensor,          # [B,1,H,W] GT labels
        ent_mc: torch.Tensor = None      # optional [B,H,W] normalized entropy
    ):
        assert outputs_semantic.ndim == 4 and semantic.ndim == 4 and semantic.shape[1] == 1
        B, C, H, W = outputs_semantic.shape

        # prediction & valid mask
        pred = outputs_semantic.argmax(dim=1)  # [B,H,W]
        gt   = semantic.squeeze(1)             # [B,H,W]
        valid = gt != self.ignore_index
        error_mask = (pred != gt) & valid

        # confidence
        if self.use_max_prob_confidence:
            max_prob = outputs_semantic.max(dim=1).values  # [B,H,W]
            confids_t = max_prob
        else:
            if ent_mc is None:
                ent_mc = entropy_from_probs(outputs_semantic)  # [B,H,W]
            confids_t = 1.0 - ent_mc.clamp(0, 1)               # [B,H,W]

        # flatten valid pixels across the batch → CPU numpy
        conf_np = confids_t[valid].float().cpu().numpy()
        risk_np = error_mask[valid].to(torch.float32).cpu().numpy()

        if self.reservoir_size is None:
            self._append_exact(risk_np, conf_np)
        else:
            self._append_reservoir(risk_np, conf_np)

    def _materialize_arrays(self):
        if self.reservoir_size is None:
            if not self._risks_list:
                raise RuntimeError("No batches added. Call add_batch(...) first.")
            risks   = np.concatenate(self._risks_list, axis=0)
            confids = np.concatenate(self._confids_list, axis=0)
        else:
            if self._risks_res is None:
                raise RuntimeError("No batches added. Call add_batch(...) first.")
            risks, confids = self._risks_res, self._confids_res
        return risks, confids

    def finalize(self,
                 make_plots: bool = True,
                 title_suffix: str = "",
                 save_dir: str | None = None,
                 epoch: int | None = None,
                 filename_prefix: str = "",
                 dpi: int = 150,
                 show: bool = False,
                 close: bool = True):
        risks, confids = self._materialize_arrays()

        aurc, eaurc, coverages, rc_risks = aurc_from_risks_confids(risks, confids)

        rc_path = er_path = None
        if make_plots:
            import os
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                tag = (f"{filename_prefix}epoch_{int(epoch):06d}_" if epoch is not None else filename_prefix)
                rc_path = os.path.join(save_dir, f"{tag}rc_curve.png")
                er_path = os.path.join(save_dir, f"{tag}uncertainty_error_recall.png")

            self._plot_rc_curve(coverages, rc_risks,
                                title=f"Risk-Coverage (dataset){title_suffix}",
                                save_path=rc_path, dpi=dpi, show=show, close=close)
            self._plot_error_recall(risks, confids,
                                    title=f"Uncertainty Error-Recall{title_suffix}",
                                    save_path=er_path, dpi=dpi, show=show, close=close)

        return {
            "AURC": aurc,
            "EAURC": eaurc,
            "num_pixels": int(risks.size),
            "rc_curve_path": rc_path,
            "error_recall_path": er_path,
        }

    # -------- plotting --------
    @staticmethod
    def _plot_rc_curve(coverages, rc_risks, title="Risk-Coverage", save_path=None, dpi=150, show=False, close=True):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(coverages, rc_risks)
        plt.xlabel("Coverage (fraction kept)")
        plt.ylabel("Risk (error rate on kept)")
        plt.title(title)
        plt.grid(True, linestyle=":")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        if show:
            plt.show()
        if close:
            plt.close(fig)

    @staticmethod
    def _plot_error_recall(risks, confids, ks=(1,2,5,10,20,30,40,50),
                           title="Uncertainty Error-Recall", save_path=None, dpi=150, show=False, close=True):
        import numpy as np, matplotlib.pyplot as plt
        idx = np.argsort(confids)
        risks_sorted = risks[idx]
        total_err = risks.sum()
        recalls = []
        for k in ks:
            m = max(1, int(risks.size * k / 100))
            recalls.append(float(risks_sorted[:m].sum() / max(total_err, 1)))
        fig = plt.figure()
        plt.plot(ks, recalls, marker="o")
        plt.xlabel("Top-k% most-uncertain pixels")
        plt.ylabel("Recall of errors")
        plt.title(title)
        plt.grid(True, linestyle=":")
        fig.tight_layout()
        if save_path:
            # if a single path is provided, we save with that name
            fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        if show:
            plt.show()
        if close:
            plt.close(fig)
