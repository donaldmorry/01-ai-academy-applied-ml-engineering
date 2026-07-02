# Section 12.3: Faster R-CNN & Mask R-CNN

> **Source:** Prosise, Ch. 12 - Faster R-CNN and Mask R-CNN sections  
> **Prerequisites:** [Section 12.2](./section-02-r-cnn-and-fast-r-cnn.md)  
> **Glossary:** [deep-learning](../../GLOSSARY.md#deep-learning) | [neural-network](../../GLOSSARY.md#neural-network) | [classification](../../GLOSSARY.md#classification)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Faster R-CNN: Proposals Go Neural

**Faster R-CNN** (Ren et al., 2016) asked: *if everything else is a [neural network](../../GLOSSARY.md#neural-network), why not region proposals too?* It introduced the **Region Proposal Network (RPN)** - a small conv net sliding over the shared feature map, predicting objectness and box offsets at each anchor location (Prosise, Fig. 12-5).

```
Input image
    ↓
Shared CNN backbone → feature map F
    ↓                    ↓
    RPN               Detection head
 (proposals)      (RoI Align + cls + bbox)
    ↓                    ↓
  ~300 proposals → final detections
```

> **Humorous analogy:** Faster R-CNN is a security guard who scans the whole lobby once (backbone), whistles at suspicious corners (RPN), then only frisks the flagged spots (RoI head).

> **In plain English:** One GPU pass proposes regions and classifies them - proposals become nearly free.

Prosise reports Faster R-CNN runs **10× faster** than Fast R-CNN with **near real-time** detection and typically higher accuracy because the RPN learns better regions than Selective Search.

---

## Region Proposal Network (RPN)

The RPN places **anchor boxes** at each spatial location on the feature map. At every anchor it predicts:

1. **Objectness score** - object vs background
2. **Box deltas** - refine anchor to proposal $(\Delta x, \Delta y, \Delta w, \Delta h)$

Typical anchor configuration:

```
Scales: 128², 256², 512² pixels
Aspect ratios: 1:1, 1:2, 2:1
→ 9 anchors per feature-map location
```

For a $50 \times 38$ feature map: $50 \times 38 \times 9 \approx 17{,}100$ anchors - most filtered during training/inference.

$$
p_i = \sigma(\mathbf{w}_{\text{obj}}^\top \mathbf{a}_i)
$$
> **Readable form:** objectness probability for anchor i = sigmoid of dot product of objectness weights with anchor feature

---

## RPN Training Labels

Each anchor is labeled:

| Condition | Label |
|-----------|-------|
| IoU with any GT $\geq 0.7$ | Positive (object) |
| IoU with all GT $< 0.3$ | Negative (background) |
| Otherwise | Ignored |

Positive anchors train bbox regression toward matched ground-truth boxes. Hard negative mining balances the overwhelming background majority.

---

## RoI Align vs RoI Pool

Fast R-CNN's **RoI Pooling** quantizes coordinates - misalignment hurts pixel-level tasks. **Mask R-CNN** (He et al., 2017) replaces it with **RoI Align**:

- Use **bilinear interpolation** at sampled points - no harsh rounding
- Preserves spatial precision for mask prediction

For detection-only Faster R-CNN, RoI Pool often suffices; instance segmentation demands RoI Align.

---

## Faster R-CNN Detection Head

After RPN proposes ~300 top-scoring regions:

1. **RoI Align** extracts $7 \times 7$ features per proposal
2. **FC layers** → class probabilities (including background)
3. **BBox head** → refined coordinates

Inference pipeline:

```
RPN proposals → NMS on proposals → RoI head → class scores → NMS on detections
```

Two NMS stages: one for proposals, one for final boxes ([Section 12.6](./section-06-non-maximum-suppression.md)).

---

## Mask R-CNN: Instance Segmentation

**Mask R-CNN** extends Faster R-CNN with a parallel **mask branch** - a small FCN predicting $28 \times 28$ binary mask per RoI (Prosise, Fig. 12-6):

$$
L = L_{\text{cls}} + L_{\text{box}} + L_{\text{mask}}
$$
> **Readable form:** total loss = classification + box regression + pixel-wise mask loss

**Instance segmentation** distinguishes *individual* object instances - two overlapping people get separate masks, unlike semantic segmentation which labels all "person" pixels one color.

Prosise demonstrates a pretrained **MaskRCNN-12-int8.onnx** model (ResNet50 backbone, COCO-trained, 80 classes) via ONNX Runtime - detecting people, backpacks, and even low-confidence false positives on logos.

```python
# Prosise Ch. 12 pattern - ONNX Mask R-CNN
from mask import preprocess, annotate_image
import onnxruntime as rt
from PIL import Image

image = Image.open('Data/adam.jpg')
image_data = preprocess(image)
session = rt.InferenceSession('Data/MaskRCNN-12-int8.onnx')
input_name = session.get_inputs()[0].name
result = session.run(None, {input_name: image_data})

boxes, labels, scores, masks = result[0], result[1], result[2], result[3]
annotate_image(image, boxes, labels, scores, masks)  # default min_confidence=0.7
```

The `change_background` helper composites segmented foreground onto a new background - the same technique Zoom uses for virtual backgrounds.

---

## Mask Resolution Tradeoff

Prosise notes masks are only **28 × 28** ("soft masks") - when upscaled to full image resolution, edges look torn. Higher-resolution masks require retraining the entire model on COCO-scale data.

---

## Running Mask R-CNN (torchvision)

```python
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights)
model.eval()

# outputs[0]['boxes'], ['labels'], ['scores'], ['masks']  shape (N,1,28,28)
```

---

## When to Choose Faster/Mask R-CNN

| Strength | Limitation |
|----------|------------|
| High mAP on COCO | Slower than YOLO - ~5-15 FPS typical |
| Precise boxes | Heavier model, more GPU memory |
| Mask R-CNN: pixel masks | Complex training pipeline |
| Strong on small objects (with FPN) | Two-stage latency for real-time video |

Use when **accuracy beats speed** - medical imaging, offline video analysis.

Use YOLO ([Section 12.4](./section-04-single-shot-detectors-and-yolo.md)) when **real-time** matters.

---

## Architecture Timeline

```
2014  R-CNN          ── accurate, impractically slow
2015  Fast R-CNN     ── shared backbone
2016  Faster R-CNN   ── neural proposals (RPN)
2017  Mask R-CNN     ── + instance masks
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Using RoI Pool for segmentation | Jagged mask boundaries | Switch to RoI Align |
| Ignoring low-confidence detections | Cluttered annotations | Tune `min_confidence` (Prosise uses 0.7) |
| Expecting 28×28 masks at full resolution | Blocky edges | Accept limitation or use higher-res model |
| Skipping proposal NMS | Duplicate RoIs | NMS at both RPN and detection stages |
| Loading ONNX without preprocess | Wrong input shape | Follow model's BGR, stride-32 resize rules |

---

## Self-Check

1. What does the RPN predict at each anchor location?  
   *Objectness score and bounding-box deltas.*
2. Why are anchors needed at multiple scales and aspect ratios?  
   *Objects vary in size and shape across the image.*
3. How does instance segmentation differ from semantic segmentation?  
   *Instance segmentation separates individual objects; semantic labels all pixels of a class alike.*
4. What is RoI Align, and why does Mask R-CNN need it?  
   *Bilinear sampling without quantization - preserves pixel alignment for masks.*
5. When would you prefer Faster R-CNN over YOLO?  
   *When accuracy and mask detail matter more than FPS.*

---

## Exercises

1. Run Prosise's Mask R-CNN ONNX demo on `adam.jpg`. List all detections above 70% confidence.
2. Lower `min_confidence` to 0.5 and count false positives. What classes confuse the model?
3. Try `change_background` with a custom background image. Document aspect-ratio distortion.
4. Compare inference time: Mask R-CNN ONNX vs `yolov3` predict on the same image.

---

## References

- Ren et al., "Faster R-CNN" - [https://arxiv.org/abs/1506.01497](https://arxiv.org/abs/1506.01497)
- He et al., "Mask R-CNN" - [https://arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)
- Prosise, Ch. 12 (ONNX Mask R-CNN exercise)
- [ONNX Model Zoo - Mask R-CNN](https://github.com/onnx/models)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Next:** [Section 12.4 - Single-Shot Detectors & YOLO](./section-04-single-shot-detectors-and-yolo.md)
