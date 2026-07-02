# Section 12.6: Non-Maximum Suppression

> **Source:** Prosise, Ch. 12 - NMS section (R-CNN intro + YOLO post-processing)  
> **Prerequisites:** [Section 12.1](./section-01-from-classification-to-detection.md) | [Section 12.5](./section-05-yolov3-in-practice.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [deep-learning](../../GLOSSARY.md#deep-learning)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Duplicate Box Problem

Modern detectors are **over-eager**. A single zebra might produce dozens of overlapping bounding boxes - each with a slightly different position and confidence score. Without post-processing, visualization becomes unreadable and downstream logic (counting, tracking) breaks.

**Non-Maximum Suppression (NMS)** is the standard fix. Prosise calls it *"a crucial element of virtually all modern object detection systems."*

> **Humorous analogy:** NMS is the bouncer at a club who says "only the most confident person per group of hugging friends gets in."

> **In plain English:** Keep the best box per object; throw away overlapping duplicates.

---

## Intersection over Union (IoU)

NMS groups boxes by **overlap**. Overlap is measured with **Intersection over Union**:

$$
\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}
$$
> **Readable form:** IoU = area of overlap divided by area of union of two boxes

| IoU | Interpretation |
|-----|----------------|
| 0.0 | No overlap |
| 0.5 | Typical NMS threshold (Prosise default) |
| 1.0 | Identical boxes |

```python
def iou(box_a, box_b):
    """box = (xmin, ymin, xmax, ymax)"""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
```

---

## NMS Algorithm (Greedy)

Prosise describes NMS for one class (e.g., zebra):

1. Sort all boxes by **confidence score** (descending)
2. Take the highest-scoring box - **keep it**
3. Remove all remaining boxes with IoU > threshold with the kept box
4. Repeat until no boxes remain

For **two zebras** (Fig. 12-2 in Prosise): boxes cluster into two groups by spatial overlap; NMS keeps the best box in each group.

```
Boxes: [Z1a:0.95, Z1b:0.88, Z1c:0.82, Z2a:0.91, Z2b:0.85]
IoU threshold = 0.5

Step 1: Keep Z1a (0.95). Suppress Z1b, Z1c (overlap Z1a).
Step 2: Keep Z2a (0.91). Suppress Z2b.
Result: 2 boxes for 2 zebras.
```

---

## Multi-Class NMS

In practice, run NMS **per class** (or use class-aware variants):

```python
def nms_per_class(detections, iou_threshold=0.5):
    """detections: list of dicts with box, score, class_id"""
    by_class = {}
    for d in detections:
        by_class.setdefault(d['class_id'], []).append(d)
    kept = []
    for cls, dets in by_class.items():
        kept.extend(greedy_nms(dets, iou_threshold))
    return kept
```

YOLO's `decode_predictions` applies NMS internally after decoding grid outputs.

---

## IoU Threshold as Hyperparameter

Prosise notes IoU threshold is tunable:

| Threshold | Behavior |
|-----------|----------|
| **Low (0.3)** | Aggressive suppression - may merge nearby same-class objects |
| **0.5** | Common default |
| **High (0.7)** | Keeps more overlapping boxes - risk of duplicates |

**Overlapping same-class instances** (one zebra behind another) may require threshold adjustment - a known edge case.

---

## Confidence Threshold vs IoU Threshold

Two separate knobs:

| Parameter | Controls |
|-----------|----------|
| **Confidence (`min_score`)** | Which boxes enter NMS at all |
| **IoU (`nms_threshold`)** | Which surviving boxes suppress each other |

Pipeline:

```
Raw predictions → filter by min_score → NMS → final detections
```

Tuning both is essential ([Section 12.5](./section-05-yolov3-in-practice.md) dog example used `min_score`; NMS handles duplicate person boxes).

---

## NMS in R-CNN and YOLO Pipelines

| Architecture | Where NMS applies |
|--------------|-------------------|
| R-CNN family | After SVM/classifier scores; optionally on RPN proposals |
| YOLO | After `decode_predictions` on multi-scale outputs |
| Custom Vision API | Server-side (you receive filtered results) |

Faster R-CNN may run NMS **twice** - on proposals and final detections ([Section 12.3](./section-03-faster-r-cnn-and-mask-r-cnn.md)).

---

## Soft-NMS (Brief)

Greedy NMS **hard-deletes** overlapping boxes. **Soft-NMS** decays scores instead of removing boxes - better for crowded scenes. Modern frameworks often offer `soft_nms` as an option.

---

## Implementing Greedy NMS

```python
def greedy_nms(detections, iou_threshold=0.5):
    """detections sorted by score descending."""
    dets = sorted(detections, key=lambda d: d['score'], reverse=True)
    kept = []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        dets = [d for d in dets if iou(best['box'], d['box']) < iou_threshold]
    return kept
```

Production code uses vectorized NumPy or GPU kernels for speed when thousands of candidates exist pre-NMS.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Skipping NMS entirely | Box soup on every object | Always post-process |
| NMS across all classes together | Class A suppresses Class B | Run per-class NMS |
| IoU threshold too low for crowds | Under-counting objects | Raise threshold or use Soft-NMS |
| Only tuning confidence | Duplicate boxes remain | Tune `nms_threshold` too |
| Wrong box format in IoU | Nonsense overlap values | Standardize on (xmin, ymin, xmax, ymax) |

---

## Self-Check

1. What problem does NMS solve?  
   *Multiple overlapping detections for the same object instance.*
2. Write IoU in words.  
   *Overlap area divided by union area of two boxes.*
3. What is a typical IoU threshold?  
   *0.5 (Prosise default).*
4. How does NMS handle two objects of the same class that do not overlap?  
   *Keeps the highest-confidence box in each spatial cluster.*
5. What's the difference between confidence threshold and IoU threshold?  
   *Confidence filters low-quality detections; IoU removes duplicates among survivors.*

---

## Exercises

1. Implement `iou` and `greedy_nms` from scratch. Test on synthetic overlapping boxes.
2. Run YOLO with `min_score=0.3` but vary NMS IoU (0.3, 0.5, 0.7). Count final boxes on one image.
3. Draw two zebras with overlapping boxes on paper. Trace NMS steps by hand.
4. Research Soft-NMS - when would you prefer it over greedy NMS?

---

## mAP and NMS Connection

COCO evaluation uses IoU thresholds (0.50:0.05:0.95) for mAP - NMS IoU at inference affects which boxes reach evaluation. Tuning NMS for demo visualization differs from tuning for benchmark leaderboard scores; document your threshold in production configs.

---

## torchvision NMS Example

```python
import torch
import torchvision

boxes = torch.tensor([[100, 100, 200, 200], [105, 105, 205, 205], [300, 300, 400, 400]], dtype=torch.float32)
scores = torch.tensor([0.9, 0.85, 0.92])
keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
print(keep)  # tensor([0, 2]) - suppresses overlapping box 1
```

---

## References

- Prosise, Ch. 12 (NMS + Fig. 12-2)
- Bodla et al., "Soft-NMS" - [https://arxiv.org/abs/1704.04503](https://arxiv.org/abs/1704.04503)
- [ torchvision.ops.nms ](https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Next:** [Section 12.7 - Azure Custom Vision](./section-07-azure-custom-vision.md)



