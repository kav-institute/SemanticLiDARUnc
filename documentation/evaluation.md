# 📊 SemanticSegmentationEvaluator

The `SemanticSegmentationEvaluator` class is designed to compute **mean Intersection over Union (mIoU)** and per-class IoU for semantic segmentation tasks.

## 🔧 Initialization

```python
SemanticSegmentationEvaluator(num_classes, test_mask)
Parameters
Name	Type	Description
num_classes	int	Total number of classes in the segmentation task.
test_mask	list	A binary list indicating which classes should be evaluated (1 = include).
```

If all values in test_mask are 1, a flag use_th is activated to enable thresholding during mIoU computation.

## ✅ Example Usage
```python
evaluator = SemanticSegmentationEvaluator(num_classes=20)
evaluator.update(preds, labels)
mIoU, scores = evaluator.compute_final_metrics(class_names)
print(f"mIoU: {mIoU:.2f}")
print(scores)
```

## 📌 Notes

test_mask allows selective evaluation on relevant classes (e.g., ignoring "unlabeled").

When use_th is True, classes with IoU below ignore_th are excluded from the mIoU calculation.

The evaluator supports both mean and median mIoU aggregation for robustness.


## 🏷️ Classes Ignored for Evaluation (0000–0008)

| ID  | Class Name       | 0000 | 0001 | 0002 | 0003 | 0004 | 0005 | 0006 | 0007 | 0008 |
|-----|------------------|------|------|------|------|------|------|------|------|------|
| 0   | unlabeled        | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   |
| 1   | car              | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |
| 2   | bicycle          | ❌   | ❌   | ✅   | ✅   | ✅   | ✅   | ❌   | ✅   | ✅   |
| 3   | motorcycle       | ❌   | ❌   | ✅   | ✅   | ✅   | ✅   | ❌   | ❌   | ❌   |
| 4   | truck            | ✅   | ❌   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ❌   |
| 5   | other-vehicle    | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   |
| 6   | person           | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |
| 7   | bicyclist        | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   |
| 8   | motorcyclist     | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   |
| 9   | road             | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |
| 10  | parking          | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |
| 11  | sidewalk         | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |
| 12  | other-ground     | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   |
| 13  | building         | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |
| 14  | fence            | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |
| 15  | vegetation       | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |
| 16  | trunk            | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |
| 17  | terrain          | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |
| 18  | pole             | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |
| 19  | traffic-sign     | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |
