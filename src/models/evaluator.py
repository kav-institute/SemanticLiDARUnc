import torch
import numpy as np

class SemanticSegmentationEvaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
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
            intersection_per_class[cls] = (pred_cls * target_cls).sum()
            union_per_class[cls] = (pred_cls + target_cls).sum() - intersection_per_class[cls]
            
        return intersection_per_class, union_per_class


    def compute_final_metrics(self, class_names, reduce="mean", ignore_th=0.1):
        """Compute final metrics after processing all batches."""
        return_dict = {}
        iou_per_class = torch.zeros(self.num_classes)
        for cls in range(self.num_classes):
            # Avoid division by zero
            if self.union_per_class[cls] == 0:
                iou_per_class[cls] = float('nan')
            else:
                iou_per_class[cls] = float(self.intersection_per_class[cls]) / float(self.union_per_class[cls])
            return_dict[class_names[cls]] = iou_per_class[cls].item()
                
        if reduce=="mean":
            mIoU = np.nanmean(np.where(iou_per_class.numpy()<=ignore_th, np.NaN, iou_per_class.numpy()))
        elif reduce=="median":
            mIoU = np.nanmedian(np.where(iou_per_class.numpy()<=ignore_th, np.NaN, iou_per_class.numpy()))
        else:
            raise NotImplementedError
        return_dict["mIoU"] = mIoU
        return mIoU, return_dict