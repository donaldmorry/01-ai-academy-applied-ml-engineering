# Section 12.4: Single-Shot Detectors & YOLO

> **Source:** Prosise, Ch. 12 - YOLO section  
> **Prerequisites:** [Section 12.3](./section-03-faster-r-cnn-and-mask-r-cnn.md)  
> **Glossary:** [deep-learning](../../GLOSSARY.md#deep-learning) | [neural-network](../../GLOSSARY.md#neural-network) | [classification](../../GLOSSARY.md#classification)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Speed Problem with Two-Stage Detectors

The R-CNN family ([Sections 12.2](./section-02-r-cnn-and-fast-r-cnn.md)-[12.3](./section-03-faster-r-cnn-and-mask-r-cnn.md)) delivers top accuracy but runs a **multi-stage pipeline**: proposals, RoI processing, classification, regression, NMS. Self-driving cars and robotics need **real-time** inference - often 30+ FPS.

> **Humorous analogy:** Two-stage detectors are a fine-dining tasting menu - exquisite, but you will miss your flight. YOLO is street food you eat while walking.

> **In plain English:** YOLO reframes detection as one regression problem: pixels in, boxes and classes out, single forward pass.

---

## You Only Look Once (2015)

Redmon et al. proposed **YOLO** - **You Only Look Once**. Instead of proposing regions then classifying each, YOLO divides the image into a grid and predicts bounding boxes and class probabilities **directly** from one [neural network](../../GLOSSARY.md#neural-network) pass (Prosise, Fig. 12-8).

From the original paper (quoted in Prosise):

> *We reframe object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities.*

YOLO's base network achieved **45 FPS** on a Titan X (2015) - more than twice the mAP of other real-time systems at the time.

---

## How YOLO Works (High Level)

1. Resize input image (e.g., $416 \times 416$)
2. Single CNN extracts feature maps at multiple scales
3. Each grid cell predicts **B bounding boxes** with confidence scores
4. Each box predicts class probabilities (conditional on objectness)
5. **NMS** filters duplicate boxes ([Section 12.6](./section-06-non-maximum-suppression.md))

$$
\text{confidence} = P(\text{object}) \times P(\text{class} \mid \text{object})
$$
> **Readable form:** detection confidence = probability of object times probability of class given object

**Anchor boxes** (borrowed from Faster R-CNN) let each cell predict boxes of different shapes and sizes.

---

## Grid Cells and Multi-Scale Detection

YOLOv3 (Prosise's focus) uses **three detection scales** with grids $13 \times 13$, $26 \times 26$, and $52 \times 52$ - detecting small, medium, and large objects respectively. Each cell predicts **9 anchors** (3 scales × 3 aspect ratios).

| Scale | Grid | Typical objects |
|-------|------|-----------------|
| Coarse | 13×13 | Cars, people (far) |
| Medium | 26×26 | Mid-size objects |
| Fine | 52×52 | Small objects |

---

## YOLO vs R-CNN Family

| Aspect | Faster R-CNN | YOLO |
|--------|--------------|------|
| Stages | 2 (RPN + head) | 1 |
| CNN passes | 1 (+ RPN) | 1 |
| Typical FPS | 5-15 | 30-150+ |
| Small objects | Strong (with FPN) | Historically weaker |
| Training | Complex | End-to-end simpler |

YOLO's weakness: **small, crowded objects** - improved in v3+ but still a consideration (Prosise notes the dog in `abby-lady.jpg` was detected below default 0.9 confidence).

---

## YOLO Versions

Prosise notes YOLOv1 through YOLOv7, plus variants (PP-YOLO, YOLO9000). **YOLOv3** was the last version Joseph Redmon contributed to and serves as the reference in the book's `keras-yolo3` / `yolov3.py` helper.

Modern production often uses **Ultralytics YOLOv8+** - same single-shot philosophy, better accuracy.

---

## Unified Loss (Conceptual)

YOLO trains with a combined loss over grid predictions:

$$
L = \lambda_{\text{coord}} L_{\text{box}} + L_{\text{obj}} + \lambda_{\text{noobj}} L_{\text{noobj}} + L_{\text{class}}
$$
> **Readable form:** total loss = weighted box error + objectness loss + background penalty + classification loss

Box loss typically uses IoU-based metrics; class loss is cross-entropy.

---

## When YOLO Wins

- **Live video** - surveillance, sports analytics, drones
- **Edge deployment** - Jetson, mobile NPUs
- **Latency-sensitive robotics** - obstacle avoidance at 30 FPS
- **Simpler training pipeline** - one network, one loss

When you need **instance masks** or maximum mAP on tiny objects, consider Mask R-CNN ([Section 12.3](./section-03-faster-r-cnn-and-mask-r-cnn.md)).

---

## Code Sketch: YOLO Philosophy

```python
# Conceptual - full runnable code in Section 12.5
def yolo_inference(image, model, decode_fn, nms_fn):
    """Single forward pass detection."""
    tensor = preprocess(image)          # resize, normalize
    raw_preds = model.predict(tensor)   # one CNN pass
    boxes = decode_fn(raw_preds)        # grid → pixel coordinates
    return nms_fn(boxes)                # suppress overlaps
```

---

## Connection to Classification CNNs

YOLO's backbone is a standard CNN (Darknet-53 in v3) - the same [deep learning](../../GLOSSARY.md#deep-learning) building blocks from [Chapter 10](../chapter-10-convolutional-neural-networks/README.md). Detection heads replace the single softmax layer with per-cell regression outputs.

---

## Prosise Ch. 12 Narrative

Prosise positions YOLO after R-CNN/Mask R-CNN to show the **accuracy-speed tradeoff**. Engineers rarely choose in the abstract - they benchmark FPS and mAP on *their* hardware and data ([Section 12.8](./section-08-choosing-a-detection-architecture.md)).

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Expecting YOLO to match Mask R-CNN mAP on tiny objects | Missed detections in crowds | Lower confidence threshold, larger input, or different architecture |
| Forgetting NMS | Hundreds of duplicate boxes | Always run NMS post-processing |
| Wrong input size | Distorted boxes | Match training resolution (416×416 for Prosise's YOLOv3) |
| Comparing FPS without same hardware | Misleading benchmarks | Document GPU, batch size, model variant |
| Ignoring anchor design | Poor box shapes | Use default COCO anchors unless retraining |

---

## Self-Check

1. How does YOLO differ from Faster R-CNN in pipeline stages?  
   *YOLO uses one stage; Faster R-CNN uses RPN + detection head.*
2. What does "You Only Look Once" mean computationally?  
   *One full CNN forward pass per image.*
3. Why does YOLOv3 use three grid scales?  
   *To detect objects at small, medium, and large sizes.*
4. What is the role of anchor boxes in YOLO?  
   *Templates for box shape/size at each grid cell.*
5. When is YOLO the wrong choice?  
   *When maximum mask precision or highest mAP on tiny objects outweighs speed.*

---

## Exercises

1. Read the original YOLO paper abstract and list three claims about speed vs accuracy.
2. Draw a $13 \times 13$ grid on a $416 \times 416$ image. How many pixels per cell?
3. Compare parameter counts: YOLOv3-tiny vs Faster R-CNN ResNet50 (look up model cards).
4. List three real-world applications where 30+ FPS detection is mandatory.

---

## YOLOv3 Multi-Scale Grids (Prosise Detail)

YOLOv3 extracts features from three backbone layers, producing detection grids at **13×13**, **26×26**, and **52×52** - roughly corresponding to large, medium, and small objects. Nine anchors per cell (three scales × three aspect ratios) yield dense predictions collapsed by NMS.

From the YOLOv1 paper introduction (quoted in Prosise):

- **45 FPS** base network on Titan X without batching
- **150+ FPS** fast variant
- **< 25 ms** latency for streaming video
- **2× mAP** vs other real-time systems of the era

---

## Anchor Boxes at Each Grid Cell

At each cell $(i, j)$, YOLO predicts offsets relative to anchor templates:

$$
b_x = \sigma(t_x) + c_x, \quad b_y = \sigma(t_y) + c_y
$$
> **Readable form:** predicted box center = sigmoid of network output plus grid cell index

The network learns $t_x, t_y, t_w, t_h$ and class logits per anchor - thousands of candidates collapsed by NMS to one box per object.

---

## YOLOv2-v8 Evolution (Brief)

| Version | Notable change |
|---------|----------------|
| YOLOv2 | Anchor boxes, batch norm |
| YOLOv3 | Multi-scale predictions |
| YOLOv4+ | Bag of freebies, CSPDarknet |
| YOLOv8 | Anchor-free options, ultralytics API |

[Section 12.5](./section-05-yolov3-in-practice.md) uses ultralytics - same family, modern weights.

---

## References

- Redmon et al., "You Only Look Once" - [https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640)
- Redmon & Farhadi, "YOLOv3" - [https://arxiv.org/abs/1804.02767](https://arxiv.org/abs/1804.02767)
- Prosise, Ch. 12
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Next:** [Section 12.5 - YOLOv3 in Practice](./section-05-yolov3-in-practice.md)



