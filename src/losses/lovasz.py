import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Lovasz-Softmax --------
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