import torch

class ECEAggregator:
    """
    Streaming Expected Calibration Error (ECE) + reliability diagram aggregator
    for dense segmentation.

    Usage:
      ece = ECEAggregator(n_bins=15, ignore_index=255, title="Reliability Diagram")
      for probs, labels in loader:   # probs: [B,C,H,W] (softmax), labels: [B,1,H,W] or [B,H,W]
          ece.add_batch(probs, labels)
      res = ece.finalize(make_plot=True, save_dir=".../eval", epoch=epoch)

    Returns from finalize():
    {
        "ECE": float,
        "bin_counts": np.ndarray[n_bins],
        "bin_accuracy": np.ndarray[n_bins],
        "bin_confidence": np.ndarray[n_bins],
        "save_path": Optional[str]
    }
    """

    def __init__(self, n_bins: int = 15, ignore_index: int = 255, title: str = "Reliability Diagram"):
        self.n_bins = int(n_bins)
        self.ignore_index = int(ignore_index)
        self.title = title

        # Buffers start on CPU; we’ll move them to the right device on first use.
        self.bin_edges = torch.linspace(0, 1, self.n_bins + 1)              # float32
        self.tot_counts = torch.zeros(self.n_bins, dtype=torch.long)        # long
        self.correct_counts = torch.zeros(self.n_bins, dtype=torch.long)    # long
        self.confidence_sums = torch.zeros(self.n_bins, dtype=torch.float32)

    # --- helpers ---
    @torch.no_grad()
    def _ensure_device(self, device: torch.device):
        """Keep internal buffers on the same device as incoming tensors."""
        if self.bin_edges.device != device:
            self.bin_edges = self.bin_edges.to(device)
            self.tot_counts = self.tot_counts.to(device)
            self.correct_counts = self.correct_counts.to(device)
            self.confidence_sums = self.confidence_sums.to(device)

    @property
    def bin_centers(self):
        edges = self.bin_edges
        return 0.5 * (edges[:-1] + edges[1:])

    @torch.no_grad()
    def reset(self):
        """Clear accumulated counts (keeps configuration)."""
        self.tot_counts.zero_()
        self.correct_counts.zero_()
        self.confidence_sums.zero_()

    # --- streaming API ---
    @torch.no_grad()
    def add_batch(self, probs: torch.Tensor, labels: torch.Tensor):
        """
        Accumulate per-bin counts from one batch.

        Args:
            probs  : [B,C,H,W] softmax probabilities
            labels : [B,1,H,W] or [B,H,W] integer labels
        """
        device = probs.device
        self._ensure_device(device)

        if labels.device != device:
            labels = labels.to(device, non_blocking=True)

        if labels.ndim == 4:
            labels = labels.squeeze(1)
        labels = labels.long()

        # mask ignore
        valid = labels != self.ignore_index
        if not valid.any():
            return

        # Flatten valid pixels: probs -> [N,C], labels -> [N]
        p = probs.permute(0, 2, 3, 1)[valid]   # [N,C]
        y = labels[valid]                      # [N]

        conf, pred = p.max(dim=1)              # [N], [N]
        correct = (pred == y)                  # [N], bool

        # Bin assignment on the SAME device as buffers
        idx = torch.bucketize(conf, self.bin_edges, right=False) - 1
        idx = idx.clamp_(0, self.n_bins - 1)

        # Tally counts
        ones = torch.ones_like(conf, dtype=torch.long)
        self.tot_counts.index_add_(0, idx, ones)
        self.correct_counts.index_add_(0, idx, correct.to(torch.long))
        self.confidence_sums.index_add_(0, idx, conf)

    @torch.no_grad()
    def finalize(self,
                make_plot: bool = True,
                save_dir: str | None = None,
                epoch: int | None = None,
                filename: str | None = None,
                dpi: int = 150,
                show: bool = False,
                close: bool = True):
        """
        Compute ECE and (optionally) save a reliability diagram.

        Args:
            make_plot : whether to build a reliability diagram
            save_dir  : directory to save the figure (created if needed)
            epoch     : if provided, filename defaults to f"ece_epoch_{epoch:06d}.png"
            filename  : custom filename (overrides epoch-based naming)
            dpi       : output figure DPI
            show      : call plt.show() (blocks in interactive backends)
            close     : close the figure after saving/showing

        Returns:
            dict with ECE and per-bin arrays + save_path (if saved)
        """
        # Per-bin accuracy & confidence
        valid_bins = self.tot_counts > 0
        bin_acc = torch.zeros_like(self.confidence_sums)
        bin_conf = torch.zeros_like(self.confidence_sums)

        bin_acc[valid_bins] = self.correct_counts[valid_bins].float() / self.tot_counts[valid_bins].float()
        bin_conf[valid_bins] = self.confidence_sums[valid_bins] / self.tot_counts[valid_bins].float()

        # Expected Calibration Error (weighted |acc - conf|)
        total = self.tot_counts.sum().clamp_min(1).float()
        weights = torch.zeros_like(self.confidence_sums)
        weights[valid_bins] = self.tot_counts[valid_bins].float() / total
        ece = (weights * (bin_acc - bin_conf).abs()).sum().item()

        save_path = None
        if make_plot:
            import os
            import matplotlib.pyplot as plt

            vb = valid_bins.cpu().numpy()
            centers = self.bin_centers.cpu().numpy()
            bc = centers[vb]
            ba = bin_acc.detach().cpu().numpy()[vb]

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.plot([0, 1], [0, 1], '--', linewidth=1)
            if len(bc) > 0:
                ax.plot(bc, ba, marker='o')
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{self.title}\nECE = {ece:.4f}')
            fig.tight_layout()

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                if filename is None:
                    filename = f"ece_epoch_{int(epoch):06d}.png" if epoch is not None else "ece.png"
                save_path = os.path.join(save_dir, filename)
                fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

            if show:
                plt.show()
            if close:
                plt.close(fig)

        return {
            "ECE": ece,
            "bin_counts": self.tot_counts.detach().cpu().numpy(),
            "bin_accuracy": bin_acc.detach().cpu().numpy(),
            "bin_confidence": bin_conf.detach().cpu().numpy(),
            "save_path": save_path,
        }
