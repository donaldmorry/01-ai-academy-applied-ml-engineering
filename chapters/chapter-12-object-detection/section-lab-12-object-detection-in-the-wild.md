# Lab 12: Object Detection in the Wild

> **Prerequisites:** Sections [12.1](./section-01-from-classification-to-detection.md)-[12.8](./section-08-choosing-a-detection-architecture.md)  
> **Estimated time:** 5-7 hours  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [deep-learning](../../GLOSSARY.md#deep-learning)  
> **Math:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Objectives

By the end of this lab you will:

1. Run **YOLO** (v3/v5/v8 via ultralytics) on diverse images with pretrained COCO weights
2. Visualize bounding boxes, class labels, and confidence scores
3. Experiment with **confidence threshold** and **NMS IoU** - document effects
4. Train or use a published **Azure Custom Vision** object detection model
5. Call the **Custom Vision Prediction API** from Python
6. Produce a **comparison table**: setup effort, accuracy, latency for YOLO vs Custom Vision
7. Optionally run detection on a short video clip

> **Humorous briefing:** One track uses a neural network that has seen 80 COCO classes. The other uses a cloud service you trained to find one very specific thing - safety vest, plant species, or your coffee mug. Compare like an engineer, not a fanboy.

---

## Setup

```python
# pip install ultralytics opencv-python matplotlib numpy requests
import warnings; warnings.filterwarnings('ignore')
import time
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

RANDOM_STATE = 42
OUT = Path('lab12_results'); OUT.mkdir(exist_ok=True)
IMAGES = Path('lab12_images')  # street, wildlife, indoor scenes
```

Place 10+ test images in `lab12_images/`. Book GitHub repo Chapter 12 folder has samples.

---

## Part A: YOLO Inference (90 min)

### A1. Load pretrained model

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # nano - fast; try yolov8s.pt for accuracy
print(model.names)  # 80 COCO class names
```

### A2. Single image inference

```python
def run_yolo(image_path, conf=0.25, iou=0.45):
    t0 = time.perf_counter()
    results = model.predict(str(image_path), conf=conf, iou=iou, verbose=False)
    latency_ms = (time.perf_counter() - t0) * 1000
    return results[0], latency_ms

img_path = list(IMAGES.glob('*.jpg'))[0]
result, latency = run_yolo(img_path)
print(f'Latency: {latency:.1f} ms, detections: {len(result.boxes)}')

annotated = result.plot()  # BGR numpy array
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig(OUT / 'yolo_single.png', dpi=120)
plt.show()
```

### A3. Threshold experiment

```python
conf_values = [0.1, 0.25, 0.5, 0.7]
fig, axes = plt.subplots(1, len(conf_values), figsize=(16, 4))

for ax, conf in zip(axes, conf_values):
    r, _ = run_yolo(img_path, conf=conf, iou=0.45)
    ann = r.plot()
    ax.imshow(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
    ax.set_title(f'conf={conf}, n={len(r.boxes)}')
    ax.axis('off')
plt.savefig(OUT / 'conf_sweep.png', dpi=120)
plt.show()
```

Record observations: low conf → more false positives; high conf → missed objects.

### A4. Batch on diverse scenes

```python
records = []
for path in IMAGES.glob('*.jpg'):
    r, lat = run_yolo(path, conf=0.35, iou=0.45)
    records.append({
        'file': path.name,
        'detections': len(r.boxes),
        'latency_ms': round(lat, 1),
        'classes': [r.names[int(c)] for c in r.boxes.cls] if len(r.boxes) else [],
    })
    r.save(filename=str(OUT / f'yolo_{path.stem}.jpg'))

Path(OUT / 'yolo_batch.json').write_text(json.dumps(records, indent=2))
```

### A5. Video (stretch)

```python
# model.predict(source='traffic.mp4', save=True, conf=0.35)
```

---

## Part B: Azure Custom Vision (90 min)

### B1. Portal training (manual step)

1. Create **Object Detection** project at [customvision.ai](https://www.customvision.ai/)
2. Define tag (e.g., `hard_hat`, `narwhal`, `defect`)
3. Upload **30+ images**, draw bounding boxes
4. **Train** → **Publish** iteration
5. Copy Prediction endpoint, key, project ID, published name

*Skip portal if using book repo's pretrained whale model files.*

### B2. REST API call

```python
import requests

PREDICTION_ENDPOINT = 'https://YOUR_REGION.api.cognitive.microsoft.com/'
PREDICTION_KEY = 'YOUR_KEY'
PROJECT_ID = 'YOUR_PROJECT_ID'
PUBLISHED_NAME = 'Iteration1'

def predict_custom_vision(image_path):
    url = (f'{PREDICTION_ENDPOINT}customvision/v3.0/Prediction/'
           f'{PROJECT_ID}/detect/iterations/{PUBLISHED_NAME}/image')
    headers = {
        'Prediction-Key': PREDICTION_KEY,
        'Content-Type': 'application/octet-stream',
    }
    t0 = time.perf_counter()
    with open(image_path, 'rb') as f:
        resp = requests.post(url, headers=headers, data=f.read())
    latency_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    return resp.json(), latency_ms

cv_result, cv_lat = predict_custom_vision('custom_test.jpg')
print(f'Custom Vision latency: {cv_lat:.1f} ms')
for p in cv_result['predictions']:
    if p['probability'] > 0.5:
        print(f"  {p['tagName']}: {p['probability']:.2f}")
```

### B3. Visualize Custom Vision boxes

```python
def draw_cv_predictions(image_path, predictions, min_prob=0.5):
    bgr = cv2.imread(str(image_path))
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for p in predictions:
        if p['probability'] < min_prob:
            continue
        bb = p['boundingBox']
        x1 = int(bb['left'] * w)
        y1 = int(bb['top'] * h)
        x2 = int((bb['left'] + bb['width']) * w)
        y2 = int((bb['top'] + bb['height']) * h)
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rgb, f"{p['tagName']} {p['probability']:.2f}",
                    (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return rgb

plt.imshow(draw_cv_predictions('custom_test.jpg', cv_result['predictions']))
plt.axis('off')
plt.savefig(OUT / 'custom_vision.png', dpi=120)
plt.show()
```

---

## Part C: Comparison Table (45 min)

Fill `lab12_results/comparison.md`:

| Criterion | YOLO (pretrained COCO) | Azure Custom Vision (your model) |
|-----------|------------------------|--------------------------------|
| Setup time (hours) | | |
| Training data needed | 0 (pretrained) | 30+ tagged images |
| Custom class support | Fine-tune required | Native |
| Avg inference latency (ms) | | |
| GPU required | Optional (faster) | No (cloud) |
| Privacy (data leaves network?) | No | Yes (training + API) |
| Detected your custom class? | Y/N | Y/N |
| Subjective quality (1-5) | | |

Add 2-3 sentences: **when would you choose each** for a workplace safety camera project?

---

## Part D: NMS Experiment (30 min)

From [Section 12.6](./section-06-non-maximum-suppression.md), vary IoU while holding conf fixed:

```python
for iou in [0.3, 0.45, 0.6, 0.8]:
    r, _ = run_yolo(img_path, conf=0.35, iou=iou)
    print(f'IoU={iou}: {len(r.boxes)} boxes after NMS')
```

Explain duplicate box behavior in your notebook.

---

## Submission Checklist

| Item | Location |
|------|----------|
| YOLO batch outputs | `lab12_results/yolo_*.jpg` |
| Confidence sweep figure | `conf_sweep.png` |
| Custom Vision visualization | `custom_vision.png` |
| Comparison table | `comparison.md` |
| Batch metrics JSON | `yolo_batch.json` |

---

## Grading Rubric (Self-Assessment)

| Criterion | Points |
|-----------|--------|
| YOLO on 10+ images with latency logged | 25 |
| Confidence + NMS experiments documented | 20 |
| Custom Vision API integration | 25 |
| Comparison table with written recommendation | 20 |
| Code runs without hard-coded secrets in repo | 10 |

Store API keys in environment variables:

```python
import os
PREDICTION_KEY = os.environ['CUSTOM_VISION_KEY']
```

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 12 - YOLO + Custom Vision
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Azure Custom Vision Prediction API](https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/quickstarts/object-detection)
- [Section 12.8 - Architecture choice](./section-08-choosing-a-detection-architecture.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 12.8](./section-08-choosing-a-detection-architecture.md) | **Next:** [Chapter 13 - NLP](../chapter-13-natural-language-processing/README.md)