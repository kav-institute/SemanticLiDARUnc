import torch
import numpy as np

class_names = {
  0 : "unlabeled", # False
  1 : "car", # True
  2 : "bicycle", # False 
  3 : "motorcycle", # False
  4 : "truck", # True
  5 : "other-vehicle", # False
  6: "person", # True
  7: "bicyclist", # False
  8: "motorcyclist", # False
  9: "road", # True
  10: "parking", # True
  11: "sidewalk", # True
  12: "other-ground", # False
  13: "building", # True
  14: "fence", # True
  15: "vegetation", # True
  16: "trunk", # True
  17: "terrain", # True
  18: "pole", # True
  19: "traffic-sign", # True
}

import torch, numpy as np

class IoUEvaluator:
    def __init__(self, num_classes: int, device="cpu"):
        self.C = num_classes
        self.device = device
        self.reset()

    def reset(self):
        self.confmat = torch.zeros((self.C, self.C), dtype=torch.long, device=self.device)
        # Rows = GT, Cols = Pred

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds/targets: [B,H,W] int64 with values in [0, C-1] (you can include unlabeled here).
        We accumulate ALL pixels; ignoring can be applied later at compute time.
        """
        preds   = preds.to(self.device).view(-1)
        targets = targets.to(self.device).view(-1)

        # Drop anything out of range just in case
        ok = (targets >= 0) & (targets < self.C) & (preds >= 0) & (preds < self.C)
        if ok.any():
            idx = targets[ok] * self.C + preds[ok]
            cm  = torch.bincount(idx, minlength=self.C*self.C).reshape(self.C, self.C)
            self.confmat += cm

    def compute(self,
                class_names,
                test_mask=None,          # list/bool tensor length C; which classes to average
                ignore_gt=None,          # list of GT labels to ignore entirely (e.g., [0, 255])
                reduce="mean",
                ignore_th=None           # drop classes with IoU<ignore_th from the average (optional)
               ):
        cm = self.confmat.clone().double()

        # 1) Remove *GT rows* you want to ignore (so unlabeled pixels don’t penalize anything)
        if ignore_gt:
            rows = torch.tensor(ignore_gt, dtype=torch.long, device=cm.device)
            rows = rows[(rows >= 0) & (rows < self.C)]
            cm[rows, :] = 0.0

        # 2) Per-class IoU from confusion matrix
        TP = cm.diag()                     # true positives
        FP = cm.sum(0) - TP                # column sum minus TP
        FN = cm.sum(1) - TP                # row sum minus TP
        denom = TP + FP + FN               # union
        iou = torch.full((self.C,), float("nan"), dtype=torch.float64, device=cm.device)
        valid = denom > 0
        iou[valid] = TP[valid] / denom[valid]

        # 3) Reporting / averaging mask (class-level)
        if test_mask is None:
            test_mask = torch.ones(self.C, dtype=torch.bool, device=cm.device)
        else:
            test_mask = torch.as_tensor(test_mask, dtype=torch.bool, device=cm.device)
            if test_mask.numel() != self.C: raise ValueError("test_mask length != num_classes")

        # Optional IoU threshold filter for the average
        if ignore_th is not None:
            avg_mask = test_mask & torch.isfinite(iou) & (iou >= ignore_th)
        else:
            avg_mask = test_mask & torch.isfinite(iou)

        # Per-class dict
        out = {}
        for k in range(self.C):
            name = class_names[k] if (isinstance(class_names, list) or isinstance(class_names, dict)) else class_names[str(k)]
            out[name] = float(iou[k].item()) if torch.isfinite(iou[k]) else float("nan")

        # mIoU
        if avg_mask.any():
            vals = iou[avg_mask].cpu().numpy()
            mIoU = float(np.mean(vals)) if reduce == "mean" else float(np.median(vals))
        else:
            mIoU = float("nan")
        out["mIoU"] = mIoU
        return mIoU, out


class SemanticSegmentationEvaluator:

    def __init__(self, num_classes, test_mask=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
        self.use_th = False
        if all(x == 1 for x in test_mask):
            self.use_th = True
        
        self.num_classes = num_classes
        self.test_mask = test_mask
        self.reset()

    def reset(self):
        self.intersection_per_class = torch.zeros(self.num_classes)
        self.union_per_class = torch.zeros(self.num_classes)


    def update(self, outputs, targets):

        """Update metrics with a new batch of data."""
        intersection, union = self.compute_scores(outputs, targets)
        self.intersection_per_class += intersection
        self.union_per_class += union

    def compute_scores(self, outputs, targets):
        intersection_per_class = torch.zeros(self.num_classes)
        union_per_class = torch.zeros(self.num_classes)

        for cls in range(self.num_classes):
            # Get predictions and targets for the current class
            pred_cls = (outputs == cls).float()
            target_cls = (targets == cls).float()
            
            # Calculate intersection and union
            intersection_per_class[cls] = self.test_mask[cls]*(pred_cls * target_cls).sum()
            union_per_class[cls] = self.test_mask[cls] * ((pred_cls + target_cls).sum() - intersection_per_class[cls])
            
        return intersection_per_class, union_per_class


    def compute_final_metrics(self, class_names, reduce="mean", ignore_th=0.01): # 0.01
        """Compute final metrics after processing all batches."""
        return_dict = {}
        iou_per_class = torch.zeros(self.num_classes)
        for cls in range(self.num_classes):
            # Avoid division by zero
            if self.union_per_class[cls] == 0:
                iou_per_class[cls] = float('nan')
            else:
                iou_per_class[cls] = float(self.intersection_per_class[cls]) / float(self.union_per_class[cls])
            # only get iou for valid classes
            if self.test_mask[cls] == 0:
                iou_per_class[cls] = float('nan')
            try:
                return_dict[class_names[cls]] = iou_per_class[cls].item()
            except:
                return_dict[class_names[str(cls)]] = iou_per_class[cls].item()
        if self.use_th:
            if reduce=="mean":
                mIoU = np.nanmean(np.where(iou_per_class.numpy()<ignore_th, np.NaN, iou_per_class.numpy()))
            elif reduce=="median":
                mIoU = np.nanmedian(np.where(iou_per_class.numpy()<ignore_th, np.NaN, iou_per_class.numpy()))
            else:
                raise NotImplementedError
        else:
            if reduce=="mean":
                mIoU = np.nanmean(iou_per_class.numpy())
            elif reduce=="median":
                mIoU = np.nanmedian(iou_per_class.numpy())
            else:
                raise NotImplementedError
        return_dict["mIoU"] = mIoU
        return mIoU, return_dict
    

# -------------------------------------------
# Uncertainty per class: aggregator + boxplot
# -------------------------------------------
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class UncertaintyPerClassAggregator:
    """
    Aggregates per-pixel uncertainty by class across batches.
    - labels: int tensor of shape (B,H,W) with values in [0, num_classes-1]
    - uncertainty: float tensor of shape (B,H,W), e.g., normalized entropy in [0,1]
    - Optional per-class cap (reservoir-like) to avoid unbounded memory.
    """
    def __init__(self, num_classes: int, max_per_class: int | None = None, seed: int = 0):
        self.num_classes = num_classes
        self.max_per_class = max_per_class
        self.rng = np.random.default_rng(seed)
        # Store on CPU to avoid GPU RAM growth
        self._values = [torch.empty(0, dtype=torch.float32) for _ in range(num_classes)]
        self._seen_counts = [0 for _ in range(num_classes)]  # for reservoir-like sampling

    def reset(self):
        self._values = [torch.empty(0, dtype=torch.float32) for _ in range(self.num_classes)]
        self._seen_counts = [0 for _ in range(self.num_classes)]

    @torch.no_grad()
    def update(self, labels: torch.Tensor, uncertainty: torch.Tensor):
        """
        labels: (B,H,W) or (1,64,512) int
        uncertainty: (B,H,W) float
        """
        assert labels.shape == uncertainty.shape, "labels and uncertainty must have same shape"
        # Work on CPU tensors for aggregation
        lab = labels.detach().to(torch.int64).cpu()
        unc = uncertainty.detach().to(torch.float32).cpu()

        # Flatten once
        lab_flat = lab.reshape(-1)
        unc_flat = unc.reshape(-1)

        # For each class, pull values with a boolean mask (vectorized)
        for c in range(self.num_classes):
            mask = (lab_flat == c)
            if not mask.any():
                self._seen_counts[c] += 0
                continue
            vals = unc_flat[mask]  # 1D tensor
            self._seen_counts[c] += int(vals.numel())

            if self.max_per_class is None:
                # Just append
                self._values[c] = torch.cat([self._values[c], vals], dim=0)
            else:
                # Keep at most max_per_class samples per class (approx reservoir)
                cur = self._values[c]
                if cur.numel() < self.max_per_class:
                    take = min(self.max_per_class - cur.numel(), vals.numel())
                    if take < vals.numel():
                        idx = torch.from_numpy(self.rng.choice(vals.numel(), size=take, replace=False))
                        vals = vals[idx]
                    self._values[c] = torch.cat([cur, vals], dim=0)
                else:
                    # Replace a random subset proportional to overflow
                    n_new = vals.numel()
                    # Probability of replacement for each incoming sample:
                    # p = max_per_class / seen_count so far (approximation)
                    p = min(1.0, float(self.max_per_class) / float(self._seen_counts[c] + 1e-9))
                    if p <= 0:
                        continue
                    keep_mask = torch.from_numpy(self.rng.random(n_new) < p)
                    if keep_mask.any():
                        to_insert = vals[keep_mask]
                        # Replace equal number of random existing indices
                        replace_idx = torch.from_numpy(
                            self.rng.choice(self.max_per_class, size=to_insert.numel(), replace=False)
                        )
                        cur[replace_idx] = to_insert
                        self._values[c] = cur  # in-place modified

    def as_dataframe(self, class_names: list, ignore_ids=()):
        """Return a long DataFrame with columns: class_id, class, uncertainty."""
        rows = []
        ignore_set = set(ignore_ids)
        for c in range(self.num_classes):
            if c in ignore_set:
                continue
            v = self._values[c]
            if v.numel() == 0:
                continue
            rows.append(pd.DataFrame({
                "class_id": c,
                "class": class_names[c],
                "uncertainty": v.numpy()
            }))
        if not rows:
            return pd.DataFrame(columns=["class_id", "class", "uncertainty"])
        return pd.concat(rows, ignore_index=True)

    def plot_boxplot(
        self,
        class_names: list,
        color_map: dict,
        ignore_ids=(),
        figsize=(18, 6),
        title="Per-class uncertainty (boxplot)",
        y_label="Normalized uncertainty",
        showfliers=False,
        save_path: str | None = None,
        dpi: int = 200
    ):
        df = self.as_dataframe(class_names, ignore_ids=ignore_ids)
        if df.empty:
            print("No data to plot.")
            return

        # Order classes as provided (excluding ignored)
        order_ids = [c for c in range(self.num_classes) if c not in set(ignore_ids) and (df["class_id"] == c).any()]
        order = [class_names[c] for c in order_ids]

        # Build seaborn palette aligned with order
        palette = [np.array(color_map[c]) / 255.0 for c in order_ids]

        sns.set_style("whitegrid")
        plt.figure(figsize=figsize)
        ax = sns.boxplot(
            data=df,
            x="class",
            y="uncertainty",
            order=order,
            palette=palette,
            showfliers=showfliers,
            linewidth=1.2,
            whis=1.5  # standard Tukey
        )
        ax.set_title(title, fontsize=18, pad=16, weight="bold")
        ax.set_xlabel("Class", fontsize=13)
        ax.set_ylabel(y_label, fontsize=13)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=11)
        ax.tick_params(axis="y", labelsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Boxplot saved to {save_path}")
        plt.show()

    def plot_ridgeline(
        self,
        class_names: list,
        color_map: dict,
        ignore_ids=(),
        figsize=(14, 9),
        title="Normalized Uncertainty per Class (Ridgeline)",
        x_label="Normalized uncertainty",
        bw_adjust: float = 0.9,     # tweak smoothness (smaller = wigglier, larger = smoother)
        fill_alpha: float = 0.9,
        line_width: float = 1.0,
        save_path: str | None = None,
        dpi: int = 200,
    ):
        """
        Ridgeline density plot of per-class uncertainty.
        Requires seaborn >= 0.11 for kdeplot(fill=...).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        df = self.as_dataframe(class_names, ignore_ids=ignore_ids)
        if df.empty:
            print("No data to plot.")
            return

        # Order classes as provided (excluding ignored) and that actually exist in df
        valid = set(df["class_id"].unique().tolist())
        order_ids = [c for c in range(self.num_classes) if c not in set(ignore_ids) and c in valid]
        order_labels = [class_names[c] for c in order_ids]
        palette = [np.array(color_map[c]) / 255.0 for c in order_ids]

        # Build one small axes per class (stacked), shared x, tight vertical spacing
        n = len(order_ids)
        height_ratios = [1] * n
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows=n, ncols=1, height_ratios=height_ratios, hspace=0.05)

        x_min, x_max = 0.0, 1.0  # uncertainties are normalized
        for i, (cid, cname, color) in enumerate(zip(order_ids, order_labels, palette)):
            ax = fig.add_subplot(gs[i, 0], sharex=fig.axes[0] if i > 0 else None)

            vals = df.loc[df["class_id"] == cid, "uncertainty"].to_numpy()
            if vals.size < 2:
                # Not enough data for KDE; fall back to a thin histogram bar
                ax.hist(vals, bins=10, range=(x_min, x_max), density=True, alpha=fill_alpha, color=color)
            else:
                sns.kdeplot(
                    vals,
                    ax=ax,
                    bw_adjust=bw_adjust,
                    fill=True,
                    clip=(x_min, x_max),
                    linewidth=line_width,
                    color=color,
                )

            # Aesthetics: ridgeline look
            ax.set_ylabel(cname, rotation=0, ha="right", va="center", labelpad=25, fontsize=11)
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            # thin baseline
            ax.axhline(0, lw=0.6, color="black", alpha=0.3)

            if i < n - 1:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(x_label, fontsize=12)

        fig.suptitle(title, fontsize=18, weight="bold", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Ridgeline plot saved to {save_path}")

        plt.show()
    
    def plot_ridgeline_fast(
        self,
        class_names: list,
        color_map: dict,
        ignore_ids=(),
        figsize=(14, 9),
        title="Normalized Uncertainty per Class (Ridgeline)",
        x_label="Normalized uncertainty",
        bins: int = 2048,              # resolution of the density grid (increase for smoother)
        bandwidth: str | float = "silverman",  # "silverman" | "scott" | float (in x-units)
        fill_alpha: float = 0.9,
        line_width: float = 1.0,
        save_path: str | None = None,
        dpi: int = 200,
    ):
        """
        Fast ridgeline using histogram + Gaussian convolution (uses ALL samples).
        Boundary bias handled via reflection at [0,1].
        """
        import numpy as np
        import matplotlib.pyplot as plt

        ignore = set(ignore_ids)
        order_ids = [c for c in range(self.num_classes) if c not in ignore and self._values[c].numel() > 0]
        if not order_ids:
            print("No data to plot."); return
        order_labels = [class_names[c] for c in order_ids]
        palette = [np.array(color_map[c]) / 255.0 for c in order_ids]

        x0, x1 = 0.0, 1.0
        dx = (x1 - x0) / bins
        x_centers = np.linspace(x0 + 0.5*dx, x1 - 0.5*dx, bins)

        def _bandwidth(vals: np.ndarray) -> float:
            n = vals.size
            if isinstance(bandwidth, (int, float)):
                return float(bandwidth)
            s = float(vals.std(ddof=1)) if n > 1 else 1e-3
            iqr = float(np.subtract(*np.percentile(vals, [75, 25]))) if n > 1 else 0.0
            sigma = max(min(s, iqr/1.34) if iqr > 0 else s, 1e-6)
            if bandwidth == "silverman":
                return 0.9 * sigma * n ** (-1/5)
            elif bandwidth == "scott":
                return 1.06 * sigma * n ** (-1/5)
            else:
                raise ValueError("bandwidth must be 'silverman', 'scott', or a float.")

        def _gauss_kernel_sigma_bins(h: float) -> np.ndarray:
            # convert bandwidth in x-units to std in bin units
            sigma_bins = max(h / dx, 1e-6)
            half = int(np.ceil(3 * sigma_bins))
            kx = np.arange(-half, half + 1, dtype=np.float32)
            k = np.exp(-0.5 * (kx / sigma_bins) ** 2)
            k /= k.sum()
            return k

        # layout
        n = len(order_ids)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows=n, ncols=1, height_ratios=[1]*n, hspace=0.05)

        for i, (cid, cname, color) in enumerate(zip(order_ids, order_labels, palette)):
            ax = fig.add_subplot(gs[i, 0], sharex=fig.axes[0] if i > 0 else None)

            vals = self._values[cid].numpy()
            n_pts = vals.size
            if n_pts < 2:
                # draw a tiny spike or skip gracefully
                counts, _ = np.histogram(vals, bins=bins, range=(x0, x1), density=False)
                density = counts / max(n_pts * dx, 1.0)
            else:
                # histogram of all samples (exact mass)
                counts, _ = np.histogram(vals, bins=bins, range=(x0, x1), density=False)

                # Gaussian kernel based on data-driven bandwidth
                h = _bandwidth(vals)
                k = _gauss_kernel_sigma_bins(h)

                # reflection padding to reduce boundary bias
                half = (len(k) - 1) // 2
                if half > 0:
                    left_pad = counts[1:half+1][::-1] if half <= len(counts)-1 else counts[::-1][:half]
                    right_pad = counts[-half-1:-1][::-1] if half <= len(counts)-1 else counts[::-1][:half]
                    pad = np.concatenate([left_pad, counts, right_pad])
                    smooth = np.convolve(pad, k, mode="same")[len(left_pad):-len(right_pad) if half>0 else None]
                else:
                    smooth = counts

                density = smooth / (n_pts * dx)  # convert to pdf (area ≈ 1)

            # draw filled ridgeline
            ax.fill_between(x_centers, 0.0, density, color=color, alpha=fill_alpha, linewidth=0)
            ax.plot(x_centers, density, color=color, linewidth=line_width)

            # aesthetics
            ax.set_ylabel(cname, rotation=0, ha="right", va="center", labelpad=25, fontsize=11)
            ax.set_yticks([])
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.axhline(0, lw=0.6, color="black", alpha=0.3)
            # if i < n - 1:
            #     ax.set_xlabel(""); ax.set_xticklabels([])
            # else:
            #     ax.set_xlabel(x_label, fontsize=12)
            import matplotlib.ticker as mticker

            ax.set_xlim(0.0, 1.0)  # uncertainties are normalized
            if i < n - 1:
                # hide only the visibility of bottom labels on this axes
                ax.tick_params(axis='x', which='both', labelbottom=False)
                ax.set_xlabel("")
            else:
                ax.set_xlabel(x_label, fontsize=12)
                ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=11))
                ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.1f}"))
                ax.tick_params(axis='x', labelsize=11)

        fig.suptitle(title, fontsize=18, weight="bold", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Ridgeline plot saved to {save_path}")
        plt.show()


##########################################################
### --- Bar chart for uncertainty vs iou per class --- ###
##########################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_iou_sorted_by_uncertainty(
    unc_agg,                 # your UncertaintyPerClassAggregator instance
    result_dict: dict,       # from compute_final_metrics(...)
    class_names: list,       # list(class_names.values())
    color_map: dict,         # {class_id: [R,G,B]}
    ignore_ids=(0,),         # ignore unlabeled by default
    figsize=(18, 6),
    title="mIoU per class, sorted by mean uncertainty",
    y_label="mIoU",
    save_path: str | None = None,
    dpi: int = 200
):
    # 1) Gather per-class mean uncertainty from aggregator
    df_unc = unc_agg.as_dataframe(class_names, ignore_ids=ignore_ids)
    if df_unc.empty:
        print("No uncertainty data available to plot.")
        return
    mean_unc = (
        df_unc.groupby(["class_id", "class"], as_index=False)
              .agg(mean_uncertainty=("uncertainty", "mean"))
    )

    # 2) Build iou per class from result_dict
    rows = []
    for cid in mean_unc["class_id"]:
        cname = class_names[cid]
        iou = result_dict.get(cname, np.nan)
        rows.append({"class_id": cid, "class": cname, "iou": iou})
    df_iou = pd.DataFrame(rows)

    # 3) Merge and filter: must have both mean_unc and finite iou
    df = mean_unc.merge(df_iou, on=["class_id", "class"], how="left")
    df = df[np.isfinite(df["iou"])]

    # 4) Sort by mean uncertainty (ascending: certain → uncertain)
    df = df.sort_values("mean_uncertainty", ascending=True).reset_index(drop=True)

    # 5) Palette aligned to sorted order
    palette = [np.array(color_map[cid]) / 255.0 for cid in df["class_id"]]

    # 6) Plot
    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df,
        x="class",
        y="iou",
        palette=palette
    )

    # aesthetics
    ax.set_title(title, fontsize=18, pad=16, weight="bold")
    ax.set_xlabel("Class (sorted by mean uncertainty: low → high)", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0.0, 1.0)

    # 7) Overall mIoU line
    overall = float(result_dict.get("mIoU", np.nan))
    if np.isfinite(overall):
        ax.axhline(overall, ls="--", lw=2, alpha=0.8, color="black")
        ax.text(
            0.99, overall + 0.01, f"mIoU = {overall:.3f}",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=12, fontweight="bold"
        )

    # optional: annotate bar values
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.2f}",
                    (p.get_x() + p.get_width()/2., h),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved bar chart to {save_path}")
    plt.show()


# ------------------------------------------------------------
# Accuracy vs (normalized) predictive-uncertainty aggregator
# (robust histogram-based; colored by bin percentage)
# ------------------------------------------------------------
import math, numpy as np, torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, to_rgb
from matplotlib.cm import get_cmap, ScalarMappable
import matplotlib.patheffects as patheffects
import pandas as pd

class UncertaintyAccuracyAggregator:
    """
    Aggregates (uncertainty, correctness) pairs and plots accuracy per
    uncertainty bin. Uses histogramming (no pandas groupby) so empty
    bins cannot show a bogus accuracy.
    """

    def __init__(self, max_samples: int | None = None, seed: int = 0):
        self.max_samples = max_samples
        self.rng = np.random.default_rng(seed)
        self._uncert = torch.empty(0, dtype=torch.float32)   # values in [0,1]
        self._correct = torch.empty(0, dtype=torch.uint8)    # 1 = correct
        self._seen = 0

    def reset(self):
        self._uncert = torch.empty(0, dtype=torch.float32)
        self._correct = torch.empty(0, dtype=torch.uint8)
        self._seen = 0

    @torch.no_grad()
    def update(self, labels: torch.Tensor, preds: torch.Tensor,
               uncertainty: torch.Tensor, ignore_ids=()):
        assert labels.shape == preds.shape == uncertainty.shape, "shapes must match"
        lab = labels.detach().to(torch.int64).cpu().reshape(-1)
        prd = preds.detach().to(torch.int64).cpu().reshape(-1)
        unc = uncertainty.detach().to(torch.float32).cpu().reshape(-1).clamp_(0, 1)

        if ignore_ids:
            mask = ~torch.isin(lab, torch.tensor(list(ignore_ids), dtype=torch.int64))
            if not mask.any():
                return
            lab, prd, unc = lab[mask], prd[mask], unc[mask]

        corr = (lab == prd).to(torch.uint8)

        if self.max_samples is None:
            self._uncert = torch.cat([self._uncert, unc], dim=0)
            self._correct = torch.cat([self._correct, corr], dim=0)
            self._seen += unc.numel()
            return

        # Reservoir-like cap
        n_new = unc.numel()
        self._seen += n_new
        if self._uncert.numel() < self.max_samples:
            take = min(self.max_samples - self._uncert.numel(), n_new)
            if take < n_new:
                idx = torch.from_numpy(self.rng.choice(n_new, size=take, replace=False))
                unc, corr = unc[idx], corr[idx]
            self._uncert = torch.cat([self._uncert, unc], dim=0)
            self._correct = torch.cat([self._correct, corr], dim=0)
        else:
            p = min(1.0, float(self.max_samples) / float(self._seen + 1e-9))
            keep = torch.from_numpy(self.rng.random(n_new) < p)
            if keep.any():
                unc, corr = unc[keep], corr[keep]
                replace_idx = torch.from_numpy(
                    self.rng.choice(self.max_samples, size=unc.numel(), replace=False)
                )
                self._uncert[replace_idx] = unc
                self._correct[replace_idx] = corr

    # ---------- compute ----------
    def _ensure_arrays(self):
        u = self._uncert.numpy()
        c = self._correct.numpy().astype(np.float32)
        return u, c

    def make_bins(self, num_bins: int | None = None, bin_width: float | None = None,
                  bin_edges: np.ndarray | None = None) -> np.ndarray:
        """
        Return strictly increasing edges covering [0,1].
        Priority: bin_edges > bin_width > num_bins.
        """
        if bin_edges is not None:
            edges = np.asarray(bin_edges, dtype=np.float32)
        elif bin_width is not None:
            K = max(1, int(round(1.0 / float(bin_width))))
            edges = np.linspace(0.0, 1.0, K + 1, dtype=np.float32)
        else:
            K = int(num_bins) if num_bins is not None else 10
            edges = np.linspace(0.0, 1.0, K + 1, dtype=np.float32)
        edges[0] = 0.0; edges[-1] = 1.0
        assert np.all(np.diff(edges) > 0), "bin edges must be strictly increasing"
        return edges

    def binned_accuracy(self, num_bins: int = 10, bin_width: float | None = None,
                        bin_edges: np.ndarray | None = None) -> pd.DataFrame:
        """
        Robust binning via numpy.histogram. Returns DataFrame with:
        [low, high, label, n, pct, accuracy]
        Empty bins: n=0, accuracy=NaN, pct=0.
        """
        u, c = self._ensure_arrays()
        if u.size == 0:
            return pd.DataFrame(columns=["low","high","label","n","pct","accuracy"])
        edges = self.make_bins(num_bins=num_bins, bin_width=bin_width, bin_edges=bin_edges)
        n    = np.histogram(u, bins=edges)[0].astype(int)
        csum = np.histogram(u, bins=edges, weights=c)[0]
        acc  = np.divide(csum, n, out=np.full_like(csum, np.nan, dtype=float), where=n>0)
        pct  = 100.0 * n / max(1, u.size)

        lows, highs = edges[:-1], edges[1:]
        labels = [f"[{l:.2f}, {h:.2f})" if i < len(lows)-1 else f"[{l:.2f}, {h:.2f}]"
                  for i,(l,h) in enumerate(zip(lows, highs))]

        return pd.DataFrame({
            "low": lows, "high": highs, "label": labels,
            "n": n, "pct": pct, "accuracy": acc
        })

    # ---------- plots ----------
    def plot_accuracy_vs_uncertainty_bins(
        self,
        num_bins: int = 10,
        bin_width: float | None = None,      # e.g., 0.05 for 20 bins
        bin_edges: np.ndarray | None = None,
        figsize=(14, 5),
        title="Pixel Accuracy vs Predictive-Uncertainty (binned)",
        x_label="Normalized predictive-entropy bin",
        y_label="Accuracy",
        show_percent_on_bars: bool = True,   # set False to hide all % labels (use colorbar only)
        annotate_min_pct: float = 0.1,      # do NOT print a label if bin percentage < this (in %)
        annotate_every: int = 1,             # label every Nth bar (1 = all)
        percent_fmt: str = "{:.1f}%",
        save_path: str | None = None,        # <<< save here if provided
        show: bool = False,                  # <<< set True to also display; default is save-only
        close_fig: bool = True,              # close after saving to free memory
        dpi: int = 200,
        cmap_name: str = "viridis",
        color_norm: str = "linear",            # "auto" | "linear" | "log"
    ):
        stats = self.binned_accuracy(num_bins=num_bins, bin_width=bin_width, bin_edges=bin_edges)
        if stats.empty or stats["n"].sum() == 0:
            print("No data to plot.")
            return

        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize, LogNorm
        from matplotlib.cm import get_cmap, ScalarMappable
        import matplotlib.patheffects as patheffects
        import numpy as np

        # --- color mapping by percentage in bin ---
        cmap = get_cmap(cmap_name)
        pct = stats["pct"].to_numpy()

        if color_norm == "linear_rel":
            norm = Normalize(vmin=0.0, vmax=max(1.0, float(pct.max())))
        if color_norm == "linear":
            norm = Normalize(vmin=0.0, vmax=100.0)
        elif color_norm == "log":
            vmin = max(1e-3, float(pct[pct > 0].min()) if (pct > 0).any() else 1e-3)
            norm = LogNorm(vmin=vmin, vmax=max(vmin * 10, float(pct.max() or 1.0)))
        else:  # auto
            vmax = float(pct.max() or 1.0)
            vmin_pos = float(pct[pct > 0].min()) if (pct > 0).any() else 1.0
            if vmax / max(vmin_pos, 1e-3) >= 20:
                norm = LogNorm(vmin=max(1e-3, vmin_pos), vmax=max(vmax, vmin_pos * 10))
            else:
                norm = Normalize(vmin=0.0, vmax=max(1.0, vmax))

        colors = cmap(norm(pct))
        empty = (stats["n"].to_numpy() == 0)
        colors[empty, 3] = 0.25  # lower alpha for empty bins

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        heights = np.nan_to_num(stats["accuracy"].to_numpy(), nan=0.0)
        bars = ax.bar(stats["label"].to_list(), heights,
                    color=colors, edgecolor="black", linewidth=0.8)

        ax.set_ylim(0.0, 1.0)
        ax.set_title(title, fontsize=18, weight="bold", pad=10)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.tick_params(axis="x", rotation=45)

        # overall accuracy dashed line
        overall = float(self._correct.float().mean()) if self._correct.numel() else float("nan")
        if np.isfinite(overall):
            ax.axhline(overall, ls="--", lw=2, color="black", alpha=0.85)
            ax.text(len(stats) - 0.5, overall + 0.012, f"overall = {overall:.3f}",
                    ha="right", va="bottom", fontsize=12, fontweight="bold")

        # ---- annotate percentages (optional) ----
        if show_percent_on_bars:
            for i, (pbar, pct_i, n_i) in enumerate(zip(bars, stats["pct"].to_list(), stats["n"].to_list())):
                # skip empty bins and tiny percentages; honor annotate_every
                if (n_i == 0): continue
                if (pct_i < float(annotate_min_pct)) or (i % max(1, int(annotate_every)) != 0):
                    r, g, b, _ = pbar.get_facecolor()
                    L = 0.2126 * r + 0.7152 * g + 0.0722 * b  # luminance → choose black/white text
                    txt_color = "black" if L > 0.6 else "white"
                    ax.text(
                        pbar.get_x() + pbar.get_width() / 2.0, 0.015, f"<{float(annotate_min_pct):.1f}%",
                        ha="center", va="bottom", fontsize=9, color=txt_color, clip_on=False,
                        path_effects=[patheffects.withStroke(linewidth=1.2,
                                    foreground=("black" if txt_color == "white" else "white"))]
                    )
                    continue

                r, g, b, _ = pbar.get_facecolor()
                L = 0.2126 * r + 0.7152 * g + 0.0722 * b  # luminance → choose black/white text
                txt_color = "black" if L > 0.6 else "white"
                ax.text(
                    pbar.get_x() + pbar.get_width() / 2.0, 0.015, percent_fmt.format(pct_i),
                    ha="center", va="bottom", fontsize=9, color=txt_color, clip_on=False,
                    path_effects=[patheffects.withStroke(linewidth=1.2,
                                foreground=("black" if txt_color == "white" else "white"))]
                )

        # colorbar for % of points
        sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("Percentage of points (%)", rotation=90)

        fig.tight_layout()

        # ----- save/show/close -----
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        # elif close_fig:
        #     plt.close(fig)

        return fig, ax


