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
