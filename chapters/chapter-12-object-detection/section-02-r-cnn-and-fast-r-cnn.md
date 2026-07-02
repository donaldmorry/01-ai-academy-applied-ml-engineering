# Section 12.2: R-CNN & Fast R-CNN

> **Source:** Prosise, Ch. 12 - R-CNN and Fast R-CNN sections  
> **Prerequisites:** [Section 12.1](./section-01-from-classification-to-detection.md) | [Chapter 10 - CNNs](../chapter-10-convolutional-neural-networks/README.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [deep-learning](../../GLOSSARY.md#deep-learning) | [neural-network](../../GLOSSARY.md#neural-network) | [svm](../../GLOSSARY.md#svm)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The R-CNN Breakthrough (2014)

Before **R-CNN** (Regions with CNN features), object detection relied on hand-crafted features (HOG, SIFT) and shallow classifiers. Girshick et al. showed that **CNN features** extracted from proposed image regions dramatically outperformed the old pipeline - but at a brutal computational cost.

R-CNN established the **two-stage** template that Faster R-CNN and Mask R-CNN still refine today:

```
1. Propose candidate regions (where might objects be?)
2. Warp each region to fixed size
3. Run CNN on each region independently
4. Classify + refine box with SVM/linear layers
```

> **Humorous analogy:** R-CNN is a detective who photocopies 2,000 random crop-outs of a crime-scene photo, sends each to a forensics lab, and only then decides which crops contain suspects. Accurate - exhausting.

> **In plain English:** R-CNN asks "what's in this patch?" thousands of times per image.

---

## Region Proposals: Selective Search

R-CNN's first stage uses **Selective Search** - a classical algorithm (no neural net) that merges superpixels hierarchically to propose ~2,000 **region proposals** per image. Each proposal is a bounding box that *might* contain an object, keyed on color, texture, shape, and size similarities (Prosise, Fig. 12-3).

| Property | Typical value |
|----------|---------------|
| Proposals per image | ~2,000 |
| Proposal method | Selective Search |
| CNN per proposal | AlexNet forward pass |

**Bottleneck:** 2,000 regions × CNN forward pass = **hours per image** on 2014 GPUs. Training was a three-stage pipeline (CNN fine-tune, [SVM](../../GLOSSARY.md#svm) train, bbox regressor train).

---

## R-CNN Pipeline Detail

For each training image with proposals $R = \{r_1, \ldots, r_N\}$, $N \approx 2000$:

1. **Warp** each region $r_i$ to $227 \times 227$ (AlexNet input)
2. **Forward pass** through CNN → feature vector $\mathbf{f}_i \in \mathbb{R}^{4096}$
3. **Train linear SVM** per class on $\mathbf{f}_i$
4. **Train bbox regressor** to refine proposal coordinates

At inference, class score for region $i$ and class $c$:

$$
\text{score}(r_i, c) = \mathbf{w}_c^\top \mathbf{f}_i + b_c
$$
> **Readable form:** class score = dot product of class weights with CNN features plus bias

Only high-scoring regions proceed to NMS ([Section 12.6](./section-06-non-maximum-suppression.md)).

---

## Why Separate CNN Passes Hurt

The fundamental inefficiency: **overlapping proposals recompute nearly identical convolutions**. A car proposal and a slightly shifted car proposal both run a full CNN.

```
Image 800×600
  → 2000 warped 227×227 crops
  → 2000 × full CNN forward passes
  → ~47 seconds/image (2014 hardware)
```

Real-time video at 30 FPS needs **33 ms per frame**. R-CNN was research-grade accuracy, not production video.

---

## Fast R-CNN (2015): Share the Backbone

**Fast R-CNN** fixed the redundancy by running the CNN **once** on the full image (Prosise, Fig. 12-4):

```
Input image
    ↓
Single CNN forward pass → feature map (e.g., 50×38×512)
    ↓
Region proposals mapped onto feature map
    ↓
RoI Pooling → fixed-size feature per region
    ↓
FC layers → class + bbox refinement (Δx, Δy, Δw, Δh)
```

> **In plain English:** Read the book once, then zoom into interesting paragraphs - don't re-read the whole book per highlight.

Prosise reports Fast R-CNN trains **an order of magnitude faster** than R-CNN, predicts **two orders of magnitude faster**, and is slightly more accurate.

---

## RoI Pooling

**Region of Interest (RoI) Pooling** converts a variable-size region on the feature map into a fixed-size tensor (e.g., $7 \times 7 \times 512$):

1. Project proposal coordinates from image space to feature-map space (divide by stride, e.g., 16)
2. Divide the RoI into a $7 \times 7$ grid
3. **Max-pool** each grid cell

$$
\text{RoIPool}(F, r) \in \mathbb{R}^{7 \times 7 \times d}
$$
> **Readable form:** RoI Pooling takes feature map F and region r, outputting a fixed 7×7×d tensor regardless of region size

Prosise's ROI pooling example: an $8 \times 16$ region reduced to $4 \times 4$ by dividing into a grid and taking the max in each cell - simple, fast, works at any aspect ratio.

```python
# Conceptual - use tf.image.crop_and_resize or torchvision.ops.roi_pool in production
def roi_pool_concept(feature_map, roi_xyxy, output_size=(7, 7)):
    """Illustrative RoI pooling - divide region into bins, max-pool each."""
    x1, y1, x2, y2 = roi_xyxy
    bin_h = (y2 - y1) / output_size[0]
    bin_w = (x2 - x1) / output_size[1]
    # Each bin: max over feature_map cells in that sub-rectangle
    return f"pooled {output_size} from region {roi_xyxy}"
```

---

## Multi-Task Loss in Fast R-CNN

Fast R-CNN trains with a combined loss per RoI:

$$
L = L_{\text{cls}} + \lambda \, L_{\text{box}}
$$
> **Readable form:** total loss = classification loss + λ times bounding-box regression loss

- **Classification loss** - softmax cross-entropy over classes + background
- **Box regression loss** - smooth L1 on $(\Delta x, \Delta y, \Delta w, \Delta h)$ for positive RoIs only

Background regions contribute only to classification - teaching the model to reject bad proposals.

---

## Bounding Box Regression

Proposals are approximate. A **bbox regressor** predicts adjustments:

$$
\begin{aligned}
\hat{x} &= x + \Delta x \cdot w \\
\hat{y} &= y + \Delta y \cdot h \\
\hat{w} &= w \cdot \exp(\Delta w) \\
\hat{h} &= h \cdot \exp(\Delta h)
\end{aligned}
$$
> **Readable form:** refined center = old center plus scaled deltas; refined size = old size times exponential of size deltas

Training targets are computed relative to ground-truth boxes for positive IoU matches (typically IoU $\geq 0.5$).

---

## Remaining Bottleneck: Selective Search

Fast R-CNN eliminated per-region CNN cost but still depended on **Selective Search** for proposals - ~2 seconds per image on CPU. The neural network was fast; the proposal algorithm was not.

[Section 12.3](./section-03-faster-r-cnn-and-mask-r-cnn.md) replaces Selective Search with a **Region Proposal Network (RPN)** - making proposals learnable and GPU-accelerated.

---

## Fast R-CNN in Modern Frameworks

```python
# PyTorch - Fast R-CNN superseded by Faster R-CNN in torchvision
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# images = [tensor CHW, 0-1 normalized]
# outputs = model(images)
# outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores']
```

Prosise's Chapter 12 emphasizes **understanding the evolution** - you rarely implement Fast R-CNN from scratch, but the RoI pooling + shared backbone pattern appears everywhere.

---

## R-CNN Family Comparison (So Far)

| Model | Proposals | CNN passes | Era |
|-------|-----------|------------|-----|
| R-CNN | Selective Search | ~2000 per image | 2014 |
| Fast R-CNN | Selective Search | 1 per image | 2015 |
| Faster R-CNN | RPN (neural) | 1 per image | 2016 |

Accuracy improved alongside speed - a rare win-win driven by shared computation.

---

## Connection to Chapter 10 CNNs

The backbone in Fast R-CNN is a standard CNN (VGG, ResNet) from [Chapter 10](../chapter-10-convolutional-neural-networks/README.md):

- Early conv layers → low-level edges, textures
- Deep layers → semantic feature map at reduced resolution
- Detection heads sit **on top of** these features rather than replacing them

Transfer learning applies: ImageNet-pretrained backbones initialize detection training.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Confusing R-CNN with single-pass CNN classification | One label per image | Detection needs proposals + per-region classification |
| Ignoring proposal overlap cost in R-CNN | Absurd training time | Use Fast R-CNN shared backbone or modern detectors |
| Wrong coordinate space for RoI pooling | Garbage boxes | Map image coords to feature-map coords via stride |
| Skipping background class training | False positives everywhere | Train background RoIs in classification head |
| Expecting real-time from Selective Search | 2+ s proposal stage | Move to RPN ([Section 12.3](./section-03-faster-r-cnn-and-mask-r-cnn.md)) |

---

## Self-Check

1. What are region proposals, and roughly how many does R-CNN use per image?  
   *~2000 candidate bounding boxes from Selective Search.*
2. Why is running the CNN once per image faster than once per proposal?  
   *Shared convolution over the full image; overlapping regions reuse computation.*
3. What does RoI Pooling produce, and why must the output size be fixed?  
   *A fixed 7×7 (typical) tensor per region so FC layers have consistent input shape.*
4. What two losses does Fast R-CNN optimize?  
   *Classification (softmax) and bounding-box regression (smooth L1).*
5. What component of Fast R-CNN was still slow on CPU?  
   *Selective Search region proposals.*

---

## Exercises

1. Read Prosise's ROI pooling note (8×16 → 4×4 grid). Implement a toy `roi_pool` on a random NumPy array and one arbitrary region.
2. Plot ~100 Selective Search boxes on an image using `cv2.ximgproc.segmentation.createSelectiveSearchSegmentation` (if available) or describe why proposals cluster on textured regions.
3. Compare parameter count: AlexNet run 2000× vs one ResNet50 forward pass. Which dominates wall-clock time?
4. Sketch the data flow from raw image to final detections in Fast R-CNN, labeling where NMS applies.

---

## Engineering Note

When benchmarking YOLO against two-stage detectors, report **hardware**, **input resolution**, **batch size**, and **confidence/NMS thresholds** - otherwise FPS comparisons mislead stakeholders. Prosise's Ch. 12 exercises use 416×416 and default `decode_predictions` thresholds; document yours in lab reports.

---

## References

- Girshick et al., "Rich feature hierarchies…" (R-CNN) - [https://arxiv.org/abs/1311.2524](https://arxiv.org/abs/1311.2524)
- Girshick, "Fast R-CNN" - [https://arxiv.org/abs/1504.08083](https://arxiv.org/abs/1504.08083)
- Prosise, *Applied ML and AI for Engineers*, Ch. 12
- [Scikit-Learn SVM Guide](https://scikit-learn.org/stable/chapters/svm.html) - R-CNN's original stage-3 classifier
- [GLOSSARY.md](../../GLOSSARY.md) - [classification](../../GLOSSARY.md#classification), [deep-learning](../../GLOSSARY.md#deep-learning)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Next:** [Section 12.3 - Faster R-CNN & Mask R-CNN](./section-03-faster-r-cnn-and-mask-r-cnn.md)

