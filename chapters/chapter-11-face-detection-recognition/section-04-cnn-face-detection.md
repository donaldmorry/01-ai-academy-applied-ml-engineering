# Section 11.4: CNN Face Detection

> **Source:** Prosise, Ch. 11 - MTCNN, multitask cascaded CNNs, OpenCV DNN face detectors  
> **Prerequisites:** [Section 11.3](./section-03-opencv-face-pipeline.md) | [Chapter 10 - CNNs](../chapter-10-convolutional-neural-networks/README.md)  
> **Glossary:** [deep-learning](../../GLOSSARY.md#deep-learning) | [neural-network](../../GLOSSARY.md#neural-network) | [classification](../../GLOSSARY.md#classification)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Why Upgrade from Haar?

[Section 11.2](./section-02-viola-jones-and-haar-cascades.md) showed that Haar cascades are fast but brittle - profile views, occlusion, low light, and small faces cause missed detections and false positives. **Convolutional neural networks** learn features from data instead of hand-crafted Haar rectangles.

Prosise highlights **MTCNN** (Multitask Cascaded Convolutional Networks, Zhang et al., 2016) as the deep-learning upgrade: three CNN stages in cascade, plus **facial landmarks** (eyes, nose, mouth) useful for alignment in [Section 11.3](./section-03-opencv-face-pipeline.md).

> **Humorous analogy:** Haar cascades are a metal detector at the beach - great for coins, useless for jewelry in mud. MTCNN is the diver with goggles who actually looks at what you found.

> **In plain English:** CNN detectors cost more compute but handle pose, scale, and lighting far better than classical cascades.

---

## MTCNN Three-Stage Cascade

| Stage | Network | Role |
|-------|---------|------|
| **P-Net** | Shallow CNN | Propose candidate face regions at multiple scales |
| **R-Net** | Deeper CNN | Refine proposals, reject false positives |
| **O-Net** | Deepest CNN | Final filter + output 5 facial landmarks |

Each stage is a **multitask** head - three outputs per candidate:

1. **Classification** - face vs not-face probability
2. **Bounding-box regression** - refine box coordinates
3. **Landmark regression** - eye/nose/mouth positions

Classification score at O-Net:

$$
p(\text{face} \mid \mathbf{x}) = \sigma\left(\mathbf{w}^\top \mathbf{f}(\mathbf{x}) + b\right)
$$
> **Readable form:** face probability = sigmoid of a linear function of CNN feature vector f(x)

Box refinement adjusts proposal $(x, y, w, h)$ by learned offsets $\Delta$:

$$
\hat{x} = x + w \cdot \Delta_x, \quad \hat{y} = y + h \cdot \Delta_y
$$
> **Readable form:** refined box = original box plus scaled offset predictions from the network

---

## MTCNN in Python

```python
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

detector = MTCNN()

image = plt.imread('group_photo.jpg')
faces = detector.detect_faces(image)

print(f'Detected {len(faces)} faces')
for i, face in enumerate(faces):
    x, y, w, h = face['box']
    conf = face['confidence']
    print(f'  Face {i}: conf={conf:.3f}, box=({x},{y},{w},{h})')
    print(f'    keypoints: {face["keypoints"]}')

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(image)
for face in faces:
    if face['confidence'] < 0.9:
        continue
    x, y, w, h = face['box']
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='lime', lw=2))
    for name, (kx, ky) in face['keypoints'].items():
        ax.plot(kx, ky, 'ro', markersize=4)
ax.axis('off')
plt.title('MTCNN Detections + Landmarks')
plt.show()
```

Prosise's Amsterdam photo example: MTCNN found faces Haar missed and also detected a statue reflection - tune with `min_face_size` and confidence threshold.

---

## Filtering Detections

| Filter | Parameter | Effect |
|--------|-----------|--------|
| Confidence | `if conf > 0.9` | Drop uncertain boxes |
| Minimum size | `MTCNN(min_face_size=40)` | Ignore tiny background faces |
| NMS | Built into pipeline | Merge overlapping boxes |

```python
detector = MTCNN(min_face_size=60)
faces = [f for f in detector.detect_faces(image) if f['confidence'] >= 0.95]
```

---

## Landmark-Based Alignment

Landmarks enable **similarity transform** alignment - rotate and scale so eyes sit at canonical positions before embedding:

```python
import cv2
import numpy as np

# Canonical eye positions for 112×112 ArcFace input (example)
REF_EYES = np.float32([[38.2946, 51.6963], [73.5318, 51.5014]])

def align_face(img, keypoints, output_size=(112, 112)):
    src = np.float32([
        keypoints['left_eye'],
        keypoints['right_eye'],
    ])
    dst = REF_EYES * (output_size[0] / 112.0)
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    aligned = cv2.warpAffine(img, M, output_size, borderValue=0)
    return aligned
```

Alignment reduces variance from head tilt - critical for [Section 11.5](./section-05-face-embeddings-and-arcface.md).

---

## OpenCV DNN Face Detector

OpenCV ships a **ResNet-based SSD** face detector - no MTCNN install required:

```python
import cv2
import numpy as np

prototxt = 'deploy.prototxt'
weights = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, weights)

img = cv2.imread('portrait.jpg')
h, w = img.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

CONF_THRESHOLD = 0.7
faces = []
for i in range(detections.shape[2]):
    conf = detections[0, 0, i, 2]
    if conf < CONF_THRESHOLD:
        continue
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    x1, y1, x2, y2 = box.astype(int)
    faces.append((x1, y1, x2 - x1, y2 - y1))
```

Download prototxt/weights from OpenCV's model zoo. Slightly less landmark detail than MTCNN but integrates cleanly with existing OpenCV pipelines.

---

## Haar vs MTCNN vs OpenCV DNN

| Criterion | Haar cascade | MTCNN | OpenCV DNN SSD |
|-----------|--------------|-------|----------------|
| Speed (CPU) | Fastest | Slowest | Moderate |
| Pose invariance | Low | High | High |
| Landmarks | No | Yes (5 points) | No |
| GPU benefit | Minimal | Large | Moderate |
| Dependencies | OpenCV only | `mtcnn` package | OpenCV + model files |

**Rule of thumb:** Haar for embedded real-time preview; MTCNN/DNN when accuracy matters.

---

## Prosise's `extract_faces` Pattern

Chapter 11 wraps detection + square crop + confidence filter:

```python
from PIL import Image, ImageOps
from mtcnn.mtcnn import MTCNN
import numpy as np

def extract_faces(path, min_confidence=0.9, crop_square=True):
    pil = ImageOps.exif_transpose(Image.open(path))
    image = np.array(pil)
    faces = MTCNN().detect_faces(image)
    faces = [f for f in faces if f['confidence'] >= min_confidence]
    crops = []
    for f in faces:
        x, y, w, h = f['box']
        if crop_square:
            side = max(w, h)
            cx, cy = x + w // 2, y + h // 2
            x, y = cx - side // 2, cy - side // 2
            w = h = side
        crops.append(Image.fromarray(image[y:y+h, x:x+w]))
    return crops
```

Resize to 224×224 for ResNet-based recognizers or 112×112 for ArcFace.

---

## Connection to Object Detection

Face detection is a **special case of object detection** ([Chapter 12](../chapter-12-object-detection/README.md)) with one class ("face") and heavy engineering for speed. YOLO-face and RetinaFace extend general detectors to faces at scale.

---

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| No faces found | Image too large / faces too small | Downscale or lower `min_face_size` |
| Many false positives | Threshold too low | Raise confidence cutoff |
| Jittery webcam boxes | Frame-to-frame noise | Temporal smoothing / tracking |
| Wrong colors in display | BGR vs RGB confusion | Convert before matplotlib |

---

## Self-Check

1. What are the three stages of MTCNN called?
2. What three tasks does each MTCNN stage perform?
3. Why are facial landmarks useful before recognition?
4. How does OpenCV DNN face detection differ from Haar cascades?
5. What parameters filter out small or low-confidence MTCNN detections?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 11 - MTCNN, OpenCV DNN
- [Zhang et al., MTCNN, 2016](https://arxiv.org/abs/1604.02878)
- [MTCNN Python package](https://github.com/ipazc/mtcnn)
- [OpenCV DNN face detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)
- [Chapter 12 - Object Detection](../chapter-12-object-detection/README.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 11.3](./section-03-opencv-face-pipeline.md) | **Next:** [Section 11.5](./section-05-face-embeddings-and-arcface.md)
