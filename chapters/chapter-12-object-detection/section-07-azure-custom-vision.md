# Section 12.7: Azure Custom Vision

> **Source:** Prosise, Ch. 12 - Custom Vision object detection, training, REST API deployment  
> **Prerequisites:** [Section 12.6](./section-06-non-maximum-suppression.md) | [Chapter 07 - Operationalizing](../chapter-07-operationalizing-models/README.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [deep-learning](../../GLOSSARY.md#deep-learning)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The "Easier Way" for Custom Detection

Training YOLO or Faster R-CNN from scratch demands GPUs, labeled datasets, and hyperparameter expertise. Prosise's engineer-first message: when the job is **custom object detection for business value**, Azure **Custom Vision** offers a managed path - upload images, draw bounding boxes in a browser, train on cloud GPUs, consume via REST API or export ONNX/Core ML.

Chapter 12 introduces Custom Vision before Chapter 14's full Cognitive Services tour. The marine biologist example trains a detector for **Narwhals** and **Belugas**.

> **Humorous analogy:** Training YOLO yourself is building a boat. Custom Vision is calling a whale-watching charter - you still spot narwhals, but someone else maintains the engine.

> **In plain English:** Tag 30+ images in the portal, click Train, call Prediction API from Python. No Darknet config required.

---

## Custom Vision vs Self-Hosted YOLO

| Factor | Custom Vision | Self-hosted YOLO |
|--------|---------------|------------------|
| Setup time | Hours | Days-weeks |
| Labeling UI | Built-in | External tool (CVAT, LabelImg) |
| GPU needed locally | No | Yes for training |
| Domain classes | Your tags | Your classes |
| Latency | Network round-trip | Local inference |
| Cost | Free tier + per-call | Compute + engineering time |
| Privacy | Images uploaded to Azure | Data stays on-prem |

---

## Workflow Overview

```
1. Create Custom Vision project (Object Detection)
2. Create tags (e.g., narwhal, beluga)
3. Upload images → draw bounding boxes per tag
4. Train (Quick vs Advanced domain)
5. Evaluate precision/recall in portal
6. Publish iteration → get Prediction endpoint + key
7. Call REST API from Python / C# / Swift
```

Minimum viable dataset Prosise implies: **30+ images** per class with varied angles and lighting.

---

## Portal Setup

1. Sign in at [customvision.ai](https://www.customvision.ai/)
2. **New Project** → type **Object Detection**
3. Select domain:
   - **General (compact)** - edge export, faster
   - **General (advanced)** - higher accuracy, cloud API
4. Add **Tags** matching your object classes

Bounding box rules:

- Tight boxes around entire object
- Include partial occlusions as separate examples
- Tag every instance in multi-object images

---

## Training Iterations

Each **Train** click creates an **iteration** with metrics:

| Metric | Meaning |
|--------|---------|
| **Precision** | Of predicted boxes, fraction correct |
| **Recall** | Of ground-truth objects, fraction found |
| **Average Precision (AP)** | Per-class detection quality |

Custom Vision uses a **compact detector** architecture optimized for export - not full YOLOv3, but sufficient for many enterprise cases.

**Quick Training** - fewer images, faster, lower ceiling. **Advanced** - benefits from 50+ images per tag.

---

## Prediction REST API

After **Publish**, retrieve:

- `PREDICTION_ENDPOINT`
- `PREDICTION_KEY`
- `PROJECT_ID`
- `PUBLISHED_NAME`

```python
import requests

PREDICTION_ENDPOINT = 'https://YOUR_REGION.api.cognitive.microsoft.com/'
PREDICTION_KEY = 'YOUR_KEY'
PROJECT_ID = 'YOUR_PROJECT_ID'
PUBLISHED_NAME = 'Iteration1'

url = (f'{PREDICTION_ENDPOINT}customvision/v3.0/Prediction/'
       f'{PROJECT_ID}/detect/iterations/{PUBLISHED_NAME}/image')

headers = {
    'Prediction-Key': PREDICTION_KEY,
    'Content-Type': 'application/octet-stream',
}

with open('test_whale.jpg', 'rb') as f:
    response = requests.post(url, headers=headers, data=f.read())

results = response.json()
for pred in results['predictions']:
    if pred['probability'] < 0.5:
        continue
    tag = pred['tagName']
    prob = pred['probability']
    bb = pred['boundingBox']
    print(f'{tag}: {prob:.2f} at left={bb["left"]:.2f}, top={bb["top"]:.2f}, '
          f'w={bb["width"]:.2f}, h={bb["height"]:.2f}')
```

Bounding boxes are **normalized** (0-1 relative to image dimensions).

---

## Drawing Predictions in Python

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('test_whale.jpg')
h, w = img.shape[:2]
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

for pred in results['predictions']:
    if pred['probability'] < 0.6:
        continue
    bb = pred['boundingBox']
    x1 = int(bb['left'] * w)
    y1 = int(bb['top'] * h)
    x2 = int((bb['left'] + bb['width']) * w)
    y2 = int((bb['top'] + bb['height']) * h)
    cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(rgb, f"{pred['tagName']} {pred['probability']:.2f}",
                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

plt.imshow(rgb)
plt.axis('off')
plt.show()
```

---

## SDK Alternative

```python
# pip install azure-cognitiveservices-vision-customvision
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

credentials = ApiKeyCredentials(in_headers={'Prediction-key': PREDICTION_KEY})
predictor = CustomVisionPredictionClient(PREDICTION_ENDPOINT, credentials)

with open('test_whale.jpg', 'rb') as f:
    results = predictor.detect_image(PROJECT_ID, PUBLISHED_NAME, f.read())

for pred in results.predictions:
    print(pred.tag_name, pred.probability)
```

SDK handles endpoint path construction - preferred for production ([Chapter 14](../chapter-14-azure-cognitive-services/README.md)).

---

## Export Options

Custom Vision supports **export** for offline inference:

| Format | Use case |
|--------|----------|
| **TensorFlow** | Python / mobile |
| **Core ML** | iOS |
| **ONNX** | Cross-platform ([Chapter 07](../chapter-07-operationalizing-models/section-05-onnx-export-and-inference.md)) |
| **Docker** | Containerized edge |

Export includes sample inference code - aligns with Prosise's ONNX theme from Mask R-CNN.

---

## Labeling Quality Checklist

- [ ] Every object instance tagged
- [ ] Boxes not excessively loose
- [ ] Train/test split respects image diversity (portal can suggest)
- [ ] Hard negatives (similar non-target objects) included
- [ ] Re-train after adding 20+ new images

Poor labels dominate failure more than architecture choice.

---

## Integration with NMS Concepts

Custom Vision applies internal post-processing similar to [Section 12.6](./section-06-non-maximum-suppression.md) - overlapping predictions filtered. Client-side, you may apply additional confidence filtering:

$$
\text{keep box } b \text{ if } p(b) \geq \tau_{\text{conf}}
$$
> **Readable form:** keep detections whose class probability exceeds confidence threshold τ

---

## Prosise's Marine Biology Example

Training narwhal/beluga detector:

1. Collect Arctic wildlife photos (book GitHub repo provides samples)
2. Tag species with bounding boxes
3. Train and publish
4. Run batch prediction on held-out images
5. Compare to COCO-pretrained YOLO (which lacks narwhal class)

**Key insight:** COCO's 80 classes don't include niche industrial objects - Custom Vision fills the gap.

---

## Cost & Free Tier

- Custom Vision training: limited free iterations per month
- Prediction API: free tier with daily call cap
- Export: no per-inference cloud cost - you host the model

Monitor usage in Azure Portal - set budget alerts ([Chapter 14](../chapter-14-azure-cognitive-services/section-08-production-considerations.md)).

---

## When Custom Vision Is Enough

| Scenario | Custom Vision fit |
|----------|-------------------|
| Product SKU on conveyor belt | Excellent |
| Safety equipment compliance | Excellent |
| Real-time autonomous driving | Insufficient - need custom YOLO + sensors |
| Medical imaging | Regulatory review + possibly insufficient |
| 100+ FPS on edge GPU | Export compact model or use YOLO |

---

## Self-Check

1. What project type must you select for bounding-box training?
2. How are Custom Vision bounding boxes formatted in API responses?
3. What is the difference between Quick and Advanced domains?
4. Name two export formats Custom Vision supports.
5. Why doesn't a COCO-pretrained YOLO detect narwhals out of the box?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 12 - Custom Vision tutorial
- [Azure Custom Vision documentation](https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/)
- [Custom Vision Python SDK](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-vision-customvision-readme)
- [Section 12.8 - Architecture selection](./section-08-choosing-a-detection-architecture.md)
- [Chapter 14 - Azure Cognitive Services](../chapter-14-azure-cognitive-services/README.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 12.6](./section-06-non-maximum-suppression.md) | **Next:** [Section 12.8](./section-08-choosing-a-detection-architecture.md)



