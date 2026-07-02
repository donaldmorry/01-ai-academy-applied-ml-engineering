# Section 12.8: Choosing a Detection Architecture

> **Source:** Prosise, Ch. 12 - mAP, FPS, two-stage vs single-shot, edge vs cloud  
> **Prerequisites:** [Sections 12.1-12.7](./section-01-from-classification-to-detection.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [deep-learning](../../GLOSSARY.md#deep-learning)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Architecture Decision

You now know **R-CNN family** (accurate, slower), **YOLO** (fast, slightly lower mAP), **Mask R-CNN** (instance masks), **NMS** post-processing, and **Custom Vision** (managed training). Production engineering picks among them using **accuracy, latency, cost, privacy, and maintainability** - not leaderboard hype alone.

> **Humorous analogy:** Choosing a detector is like choosing transportation: Faster R-CNN is a cargo plane (heavy, precise), YOLO is a motorcycle (fast, nimble), Custom Vision is rideshare (someone else drives).

> **In plain English:** Match the detector to your FPS budget, label budget, and whether data can leave your network.

---

## Detection Metrics

### Intersection over Union (IoU)

From [Section 12.6](./section-06-non-maximum-suppression.md):

$$
\text{IoU}(B_{\text{pred}}, B_{\text{gt}}) = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})}
$$
> **Readable form:** IoU = overlap area divided by union area; 1.0 = perfect box, 0 = no overlap

A prediction is a **true positive** if IoU ≥ threshold (typically 0.5) and class is correct.

### Precision & Recall (per class)

$$
\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}
$$
> **Readable form:** precision = fraction of predictions that are correct; recall = fraction of ground-truth objects found

### Average Precision (AP) and mAP

**AP** - area under precision-recall curve for one class at fixed IoU threshold.

**mAP** - mean AP across all classes:

$$
\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} AP_c
$$
> **Readable form:** mAP = average of per-class AP scores across C object classes - standard COCO benchmark metric

COCO also reports mAP@[.5:.95] - average over IoU thresholds 0.5 to 0.95 in steps of 0.05 - stricter than mAP@0.5 alone.

---

## Speed: FPS and Latency

| Architecture | Typical GPU FPS | Typical CPU FPS | Notes |
|--------------|-----------------|-----------------|-------|
| Faster R-CNN | 5-15 | <1 | High mAP |
| Mask R-CNN | 3-10 | <1 | + segmentation |
| YOLOv3 | 20-45 | 1-5 | Good balance |
| YOLOv8n | 60-120+ | 5-15 | Nano for edge |
| Custom Vision (cloud) | N/A | N/A | Network latency ~100-500ms |

**Frames per second:**

$$
\text{FPS} = \frac{1}{t_{\text{infer}} + t_{\text{pre}} + t_{\text{post}}}
$$
> **Readable form:** FPS = inverse of total time for preprocess + inference + postprocess per frame

Real-time video often needs ≥15 FPS; autonomous driving targets 30+ FPS.

---

## Two-Stage vs Single-Shot

| Aspect | Two-stage (Faster R-CNN) | Single-shot (YOLO) |
|--------|--------------------------|---------------------|
| Pipeline | Proposals → classify each | Grid cells predict boxes + classes |
| Accuracy (mAP) | Higher on COCO | Competitive; YOLOv8 near SOTA |
| Speed | Slower | Faster |
| Small objects | Better with FPN | Improved in recent versions |
| Training complexity | Higher | Moderate |
| Best for | Offline analysis, max accuracy | Video, edge, robotics |

Prosise: R-CNN family for accuracy showcase (Mask R-CNN ONNX); YOLO for real-time driving narrative.

---

## Decision Matrix

| Requirement | Recommended approach |
|-------------|---------------------|
| No ML team, <50 classes, fast MVP | Azure Custom Vision |
| 30+ FPS on Jetson / Raspberry Pi | YOLOv8n export TensorRT |
| Need pixel masks (background swap) | Mask R-CNN / YOLO-seg |
| Data cannot leave premises | Self-hosted YOLO or exported Custom Vision |
| 80 COCO classes sufficient | Pretrained YOLO - no training |
| Rare industrial defect | Custom Vision or fine-tuned YOLO |
| Research / max mAP | Detectron2, Faster R-CNN + FPN |

---

## Pretrained vs Custom Training

**Pretrained COCO** ([Section 12.5](./section-05-yolov3-in-practice.md)):

- 80 classes (person, car, dog, …)
- Zero training cost
- Wrong if your class ∉ COCO

**Fine-tune YOLO:**

- Start from COCO weights
- Train on your labeled dataset
- Needs GPU + labeling effort

**Custom Vision:**

- Managed fine-tuning
- Less control over architecture

Transfer learning math is the same idea as [Chapter 10](../chapter-10-convolutional-neural-networks/section-07-fine-tuning-pretrained-models.md) - backbone features generalize; head adapts to new classes.

---

## Edge vs Cloud Inference

| Factor | Edge (local YOLO) | Cloud (Custom Vision API) |
|--------|-------------------|---------------------------|
| Latency | Low after load | Network dependent |
| Privacy | Data stays local | Images transmitted |
| Cost at scale | Fixed hardware | Per-call pricing |
| Updates | You redeploy | Microsoft retrains service |
| Offline | Works | Requires connectivity |

Hybrid: train in cloud, **export ONNX**, run on edge container ([Chapter 14](../chapter-14-azure-cognitive-services/section-02-azure-setup-and-authentication.md)).

---

## Model Size vs Accuracy

| YOLO variant | Params (approx) | mAP (COCO) | Use |
|--------------|-----------------|------------|-----|
| YOLOv8n | 3M | lower | Mobile |
| YOLOv8s | 11M | medium | Balanced edge |
| YOLOv8m | 26M | higher | Server GPU |
| YOLOv8x | 68M | highest | Offline batch |

$$
\text{efficiency} = \frac{\text{mAP}}{\text{params} \times t_{\text{infer}}}
$$
> **Readable form:** efficiency score = accuracy divided by model size times inference time - higher is better for resource-constrained deployment

---

## Post-Processing Tuning

Same model, different deployment behavior via:

| Hyperparameter | Effect |
|----------------|--------|
| Confidence threshold $\tau_{\text{conf}}$ | Higher → fewer boxes, higher precision |
| NMS IoU threshold | Lower → more aggressive duplicate removal |
| Input resolution | Higher → better small objects, slower |

Always tune on **validation set** representative of production camera angles.

---

## Monitoring in Production

Track:

- mAP drift (re-label monthly sample)
- Inference latency p95
- False positive rate (user reports)
- Class imbalance in detections

Trigger retraining when precision drops below SLA.

---

## Course 1 → Course 3 Path

| Topic here | Advanced coverage |
|------------|-------------------|
| YOLOv3 | YOLOv5-v11, anchor-free detectors |
| mAP | COCO evaluation protocol细节 |
| Mask R-CNN | Panoptic segmentation, SAM |
| Custom Vision | Full Azure MLOps (Course 2) |

---

## Self-Check

1. What is mAP and why not use plain accuracy for detection?
2. When would you pick Faster R-CNN over YOLO?
3. What IoU threshold typically defines a true positive?
4. Name three factors in the edge vs cloud decision.
5. When is pretrained COCO YOLO insufficient?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 12 - architecture comparison
- [COCO detection metrics](https://cocodataset.org/#detection-eval)
- [Redmon et al., YOLO, 2015](https://arxiv.org/abs/1506.02640)
- [Lin et al., Focal Loss / RetinaNet](https://arxiv.org/abs/1708.02002)
- [Ultralytics YOLOv8 docs](https://docs.ultralytics.com/)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 12.7](./section-07-azure-custom-vision.md) | **Next:** [Lab 12](./section-lab-12-object-detection-in-the-wild.md)



