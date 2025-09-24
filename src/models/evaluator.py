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

class SemanticSegmentationEvaluator:

    def __init__(self, num_classes, test_mask=[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]):
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
