# Section 12.1: From Classification to Detection

> **Source:** Prosise, Ch. 12 - "Object Detection" / localization + classification  
> **Prerequisites:** [Chapter 10 - CNNs](../chapter-10-convolutional-neural-networks/README.md) | [Chapter 11 - Face Detection](../chapter-11-face-detection-recognition/README.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [deep-learning](../../GLOSSARY.md#deep-learning) | [neural-network](../../GLOSSARY.md#neural-network)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Problem: "What" vs "What and Where"

[Image classification](../chapter-10-convolutional-neural-networks/section-01-why-cnns-for-images.md) answers a single question: *What is in this image?* A CNN trained on ImageNet might label a street scene as "city" or "highway" - useful, but incomplete.

**Object detection** answers two questions for every object instance:

1. **What** is it? (class label)
2. **Where** is it? (bounding box coordinates)

> **Humorous analogy:** Classification is like reading a restaurant menu and guessing the dish from a single bite. Detection is like a food critic walking through a buffet, naming every dish *and* pointing at each plate. One photo, many objects, many boxes.

> **In plain English:** Detection = find all objects + draw rectangles around them + label each rectangle.

---

## Why Classification CNNs Fall Short

In [Chapter 10](../chapter-10-convolutional-neural-networks/README.md), training images are carefully cropped: one object, centered, fixed size. Real-world photos violate every assumption:

| Classification assumption | Real-world reality |
|---------------------------|-------------------|
| One object per image | Dozens of cars, people, signs in one frame |
| Object centered and scaled | Objects at any position, any scale |
| Fixed input dimensions | Same object can be 10×10 or 500×500 pixels |
| Whole-image label | Need per-object labels |

A self-driving car's forward camera sees pedestrians, traffic lights, and vehicles simultaneously (Prosise, Fig. 12-1). A classifier might say "street scene" but cannot report *three cars at these coordinates*.

[Face detection](../chapter-11-face-detection-recognition/README.md) is a **special case** of object detection - one class ("face"), but the same localization challenge applies.

---

## The Detection Output Format

Every modern detector returns a list of detections. Each detection typically contains:

$$
\text{detection} = (x_{\min}, y_{\min}, x_{\max}, y_{\max}, \text{class}, \text{confidence})
$$
> **Readable form:** detection = (left, top, right, bottom pixel coordinates, class name or ID, confidence score from 0 to 1)

**Bounding box conventions:**

- **Corner format:** $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$ - two opposite corners
- **Center format:** $(c_x, c_y, w, h)$ - center point plus width and height

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, xmin, ymin, xmax, ymax, label, score, color='red'):
  rect = patches.Rectangle(
      (xmin, ymin), xmax - xmin, ymax - ymin,
      linewidth=2, edgecolor=color, facecolor='none')
  ax.add_patch(rect)
  ax.text(xmin, ymin - 4, f'{label} ({score:.0%})',
          color='white', fontsize=9, fontweight='bold',
          bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(plt.imread('street_scene.jpg'))
draw_box(ax, 120, 80, 340, 280, 'person', 0.97)
draw_box(ax, 400, 150, 620, 350, 'car', 0.94)
draw_box(ax, 50, 30, 90, 120, 'traffic light', 0.88)
ax.set_axis_off()
plt.tight_layout()
```

---

## Detection vs Segmentation

Computer vision tasks form a hierarchy of increasing spatial detail:

| Task | Output | Example |
|------|--------|---------|
| **Classification** | One label per image | "Dog" |
| **Object detection** | Boxes + labels per instance | 2 dogs, 1 frisbee with coordinates |
| **Semantic segmentation** | Pixel-level class map | Every pixel labeled grass/sky/dog |
| **Instance segmentation** | Pixel mask per object instance | Separate mask for each dog |

Detection sits in the sweet spot for many engineering applications: autonomous driving, retail shelf analytics, wildlife monitoring, industrial defect inspection. You get location without the compute cost of full pixel masks (covered in [Section 12.3](./section-03-faster-r-cnn-and-mask-r-cnn.md)).

---

## The Two Sub-Problems

Every detector must solve:

### 1. Localization - Where are candidate objects?

Strategies evolved over time:

- **Sliding window:** Run a classifier at every position and scale - brute force, slow
- **Region proposals:** Propose ~2,000 candidate boxes (selective search, RPN) - [Section 12.2](./section-02-r-cnn-and-fast-r-cnn.md)
- **Single-shot regression:** Predict all boxes in one forward pass - [Section 12.4](./section-04-single-shot-detectors-and-yolo.md)

### 2. Classification - What is in each region?

Once a region is proposed, a [CNN](../chapter-10-convolutional-neural-networks/README.md) (or shared backbone) classifies the contents. Modern systems unify both steps in one network.

---

## Confidence Scores and Thresholds

Each detection carries a **confidence score** $s \in [0, 1]$:

$$
s = P(\text{object}) \times P(\text{class} \mid \text{object})
$$
> **Readable form:** confidence = probability there is an object times probability of the predicted class given an object is present

Low-confidence detections are usually discarded:

```python
def filter_detections(detections, min_confidence=0.5):
  return [d for d in detections if d['score'] >= min_confidence]
```

**Tradeoff:** Higher threshold → fewer false positives, more missed objects. Lower threshold → more detections, more duplicates (handled by [NMS in Section 12.6](./section-06-non-maximum-suppression.md)).

---

## Pretrained Detectors: The COCO Dataset

State-of-the-art detectors are trained on **COCO** (Common Objects in Context):

- 200,000+ images
- 1.5M+ object instances
- **80 object classes** (person, bicycle, car, dog, …)

Pretrained models (YOLO, Mask R-CNN) detect these 80 classes out of the box. Custom classes require retraining or services like [Azure Custom Vision](./section-07-azure-custom-vision.md).

```python
# COCO class IDs are 1-indexed in many frameworks
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    # ... 68 more classes
]
```

---

## Real-World Applications

| Domain | Detection task |
|--------|----------------|
| Autonomous vehicles | Pedestrians, vehicles, signs in real time |
| Retail | Product recognition on shelves |
| Security | Package detection on porches |
| Healthcare | Cell detection in tissue samples |
| Wildlife | Species counting in camera traps |

Prosise emphasizes the engineering reality: you rarely train detectors from scratch. You run pretrained models, fine-tune on domain data, or use cloud tools when speed-to-value matters.

---

## How This Chapter Unfolds

| Section | Topic |
|--------|-------|
| 12.2-12.3 | R-CNN family: accuracy-first, two-stage |
| 12.4-12.5 | YOLO: speed-first, single-shot |
| 12.6 | Non-maximum suppression post-processing |
| 12.7 | Azure Custom Vision for custom classes |
| 12.8 | Architecture selection guide |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Using a classifier on uncropped scenes | One vague label | Switch to a detector |
| Ignoring confidence threshold | Thousands of spurious boxes | Tune `min_score` (start 0.5-0.7) |
| Expecting COCO model to find custom SKUs | Zero detections | Train custom model ([Section 12.7](./section-07-azure-custom-vision.md)) |
| Confusing detection with segmentation | Boxes only, no pixel masks | Use Mask R-CNN for instance masks |

---

## Self-Check

1. What two questions does object detection answer that classification does not?
2. Why can't a standard ImageNet CNN localize multiple objects?
3. What information does each detection output contain?
4. What is the COCO dataset, and why do pretrained detectors use it?
5. How does face detection relate to general object detection?

---

## Exercises

1. Find a busy street photo online. List all object classes you can identify and estimate bounding boxes manually (rough rectangles on paper or in an image editor).
2. Load a pretrained detector (YOLO or `ultralytics`) and print detections with scores below 0.3 - how many false positives appear?
3. Compare inference time of a classification CNN vs a detector on the same image. Record FPS for each.

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 12 (intro)
- [COCO Dataset](https://cocodataset.org/)
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [GLOSSARY.md](../../GLOSSARY.md) - [classification](../../GLOSSARY.md#classification), [deep-learning](../../GLOSSARY.md#deep-learning)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Next:** [Section 12.2 - R-CNN & Fast R-CNN](./section-02-r-cnn-and-fast-r-cnn.md)

---

## Assessment Practice

Use the shared [Assessment Appendix](../../ASSESSMENT_APPENDIX.md) for concept audits, worked examples, implementation checks, experiment logs, oral-exam prompts, and deliverable checklists.
