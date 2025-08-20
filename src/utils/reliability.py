# utils/reliability.py
import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def reliability_diagram_from_probs(
    probs,              # [B,C,H,W] probabilities
    labels,             # [B,1,H,W] or [B,H,W]
    ignore_index=255,
    n_bins=15,
    title="Reliability Diagram",
    save_path: str | None = None,
    show: bool = False
):
    if labels.ndim == 4:
        labels = labels.squeeze(1)
    labels = labels.long()

    valid = labels != ignore_index
    if valid.sum() == 0:
        raise ValueError("No valid pixels for reliability computation.")

    probs = probs.permute(0, 2, 3, 1)[valid]   # [N,C]
    labels_v = labels[valid]
    confidences, preds = probs.max(dim=1)
    accuracies = (preds == labels_v).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    bin_accs, bin_confs = [], []

    N = confidences.numel()
    for i in range(n_bins):
        l, r = bin_boundaries[i], bin_boundaries[i+1]
        mask = (confidences >= l) & (confidences <= r) if i == 0 else (confidences > l) & (confidences <= r)
        if mask.any():
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            w = mask.float().mean()
            ece = ece + w * (bin_acc - bin_conf).abs()
            bin_accs.append(bin_acc.item())
            bin_confs.append(bin_conf.item())

    fig = plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1],'--',linewidth=1)
    if bin_confs:
        plt.plot(bin_confs, bin_accs, marker='o')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'{title}\nECE = {ece.item():.4f}')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    return ece.item()
