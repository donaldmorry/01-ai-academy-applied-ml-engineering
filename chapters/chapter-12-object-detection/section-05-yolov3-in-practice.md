# Section 12.5: YOLOv3 in Practice

> **Source:** Prosise, Ch. 12 - "YOLOv3 and Keras" section  
> **Prerequisites:** [Section 12.4](./section-04-single-shot-detectors-and-yolo.md) | [Chapter 10 - CNNs](../chapter-10-convolutional-neural-networks/README.md)  
> **Glossary:** [deep-learning](../../GLOSSARY.md#deep-learning) | [neural-network](../../GLOSSARY.md#neural-network) | [classification](../../GLOSSARY.md#classification)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## From Darknet to Keras

YOLO was originally implemented in **Darknet**. Prosise uses the **keras-yolo3** project (Huynh Ngoc Anh / experiencor) wrapped in a helper file `yolov3.py` - simplified for **inference** with COCO pretrained weights.

> **Humorous analogy:** `yolov3.py` is a rental car with GPS pre-programmed to 80 COCO destinations. You don't build the engine; you drive it.

> **In plain English:** Download weights, load model, preprocess image, predict, decode boxes, draw results.

---

## Setup: Files and Weights

Prosise's exercise requires:

| File | Purpose |
|------|---------|
| `yolov3.py` | Model builder, `WeightReader`, `decode_predictions`, `annotate_image` |
| `yolov3.weights` | COCO-trained Darknet weights |
| `Data/xian.jpg`, `Data/abby-lady.jpg` | Sample images |

Input size: **416 × 416** pixels (YOLOv3 standard in this port).

```python
from yolov3 import *

model = make_yolov3_model()
weight_reader = WeightReader('Data/yolov3.weights')
weight_reader.load_weights(model)
model.summary()
```

---

## Loading and Displaying an Image

```python
import matplotlib.pyplot as plt
%matplotlib inline

image = plt.imread('Data/xian.jpg')
width, height = image.shape[1], image.shape[0]
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'xticks': [], 'yticks': []})
ax.imshow(image)
```

The Xian city-wall biking photo contains two people and two bicycles - a good multi-object test scene.

---

## Preprocessing and Prediction

```python
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

x = load_img('Data/xian.jpg', target_size=(YOLO3.width, YOLO3.height))
x = img_to_array(x) / 255.0
x = np.expand_dims(x, axis=0)
y = model.predict(x)
```

`predict` returns raw tensors at **three scales** (8×8, 16×16, 32×32 grid cells). These must be **decoded** into pixel-coordinate bounding boxes.

---

## Decoding Predictions

`decode_predictions` (inspired by Keras's `decode_predictions`) maps network output to `BoundingBox` objects:

```python
boxes = decode_predictions(y, width, height)

for box in boxes:
    print(f'({box.xmin}, {box.ymin}), ({box.xmax}, {box.ymax}), '
          f'{box.label}, {box.score}')
```

Prosise's output on `xian.jpg`:

```
(692, 232), (1303, 1490), person, 0.997
(1314, 327), (1920, 1496), person, 0.996
(716, 786), (1277, 1634), bicycle, 0.992
(1210, 845), (2397, 1600), bicycle, 0.996
```

---

## COCO Class Labels

YOLOv3 detects **80 COCO classes** - built into `YOLO3.labels`:

```python
print(len(YOLO3.labels))  # 80
# person, bicycle, car, ... wine glass, etc.
```

Pretrained COCO models work out of the box for common objects but **not** for domain-specific classes (license plates, SKUs, cancer cells).

---

## Visualizing Detections

```python
annotate_image('Data/xian.jpg', boxes)
```

`annotate_image` draws rectangles, class names, and confidence percentages on the image.

---

## Confidence Threshold Tuning

On `abby-lady.jpg`, the model detected the girl and laptop at high confidence but **missed the dog** at the default `min_score=0.9`. Lowering the threshold reveals more detections:

```python
boxes = decode_predictions(y, width, height, min_score=0.55)
annotate_image('Data/abby-lady.jpg', boxes)
```

At 0.55, the dog and sofa appear - along with potential false positives.

| `min_score` | Effect |
|-------------|--------|
| **High (0.9)** | Few boxes, high precision, missed objects |
| **Medium (0.5-0.7)** | Balanced for most demos |
| **Low (0.3)** | Many boxes, recall up, noisy |

---

## Scaling Boxes Back to Original Resolution

`decode_predictions` needs original `width` and `height` because the network sees 416×416 but boxes must map to full-resolution coordinates:

$$
x_{\text{orig}} = x_{416} \times \frac{W_{\text{orig}}}{416}
$$
> **Readable form:** original x coordinate = network x times width scale factor

The helper handles this internally - always pass original dimensions.

---

## Video Inference Pattern

```python
import cv2

cap = cv2.VideoCapture('street.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    # preprocess frame → predict → decode_predictions(y, w, h)
    # draw boxes on frame → cv2.imshow(...)
cap.release()
```

Prosise focuses on images; video is the same loop with FPS budgeting ([Section 12.8](./section-08-choosing-a-detection-architecture.md)).

---

## Modern Alternative: Ultralytics

```python
# pip install ultralytics
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model('Data/xian.jpg')
results[0].show()
```

Ultralytics simplifies loading, NMS, and visualization - same YOLO philosophy, less boilerplate than raw keras-yolo3.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Forgetting to normalize pixels (/255) | Random detections | Match training preprocessing |
| Wrong `target_size` | Distorted aspect ratio | Use `YOLO3.width/height` constants |
| Skipping `decode_predictions` | Meaningless raw tensors | Always decode + NMS |
| Default 0.9 threshold on hard images | "Model missed obvious object" | Lower `min_score`, check raw scores |
| Expecting custom classes from COCO weights | No detections | Train custom model ([Section 12.7](./section-07-azure-custom-vision.md)) |

---

## Self-Check

1. What input resolution does Prosise's YOLOv3 Keras port use?  
   *416 × 416.*
2. What does `decode_predictions` require besides raw model output?  
   *Original image width and height for coordinate scaling.*
3. How many COCO classes are built into `YOLO3.labels`?  
   *80.*
4. Why did the dog not appear at default settings in `abby-lady.jpg`?  
   *Confidence was below the default 0.9 threshold.*
5. What three scales does YOLOv3 predict at?  
   *Corresponding to 13×13, 26×26, and 52×52 grids on the feature maps.*

---

## Exercises

1. Run YOLOv3 (or Ultralytics) on three personal photos. Record detections per class.
2. Sweep `min_score` from 0.3 to 0.95 on one image. Plot precision vs number of boxes qualitatively.
3. Time `model.predict` for 100 runs. Compute average latency in ms.
4. Try an image with no COCO objects (abstract art). What false positives appear?

---

## References

- [keras-yolo3 GitHub](https://github.com/experiencor/keras-yolo3)
- Prosise, Ch. 12 - YOLOv3 and Keras
- [COCO Classes List](https://cocodataset.org/#explore)
- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Next:** [Section 12.6 - Non-Maximum Suppression](./section-06-non-maximum-suppression.md)



