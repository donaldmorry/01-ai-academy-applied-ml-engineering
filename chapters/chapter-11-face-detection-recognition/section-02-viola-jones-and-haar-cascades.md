# Section 11.2: Viola-Jones & Haar Cascades

> **Source:** Prosise, Ch. 11 - integral images, AdaBoost cascades, OpenCV Haar detector  
> **Prerequisites:** [Section 11.1](./section-01-detection-vs-recognition.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [feature](../../GLOSSARY.md#feature)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Classical Face Detector

Before [deep learning](../../GLOSSARY.md#deep-learning) dominated vision, **Viola and Jones (2001)** made real-time face detection practical. Their method combines:

1. **Haar-like features** - rectangular intensity differences
2. **Integral images** - O(1) rectangle sums
3. **AdaBoost** - select weak classifiers
4. **Cascade** - reject non-faces early

OpenCV ships pretrained Haar cascades - Prosise uses `cv2.CascadeClassifier` as the fast baseline before CNN detectors.

> **Humorous analogy:** The cascade is a bouncer line at a club with increasingly picky doormen. Most patches get rejected at door one ("not a face"); survivors face tougher checks.

> **In plain English:** Slide a window across the image; at each position, run a sequence of cheap tests. If any test fails, skip - it's not a face.

---

## Haar-Like Features

Each feature compares sum of pixels in white vs black rectangles:

```
Edge feature:     Line feature:      Center-surround:
┌──┬──┐           ┌────┬────┐        ┌────┬────┐
│██│  │           │████│    │        │    │████│
│██│  │           │████│    │        │████│    │
└──┴──┘           └────┴────┘        └────┴────┘
```

Feature value:

$$
f = \sum_{\text{white}} I - \sum_{\text{black}} I
$$
> **Readable form:** Haar feature = sum of pixel intensities in white rectangles minus sum in black rectangles

Thousands of candidate features exist at each position and scale.

---

## Integral Image

Precompute summed-area table $II$ so any axis-aligned rectangle sum is four lookups:

$$
II(x,y) = \sum_{x' \le x,\, y' \le y} I(x', y')
$$
> **Readable form:** The integral image stores the cumulative pixel sum up to each location, making rectangular Haar features fast to compute.

Rectangle sum:

$$
\sum_{(x,y) \in R} I(x,y) = II(D) - II(B) - II(C) + II(A)
$$
> **Readable form:** rectangle pixel sum = bottom-right corner sum − adjustments using three other corners of the integral image

Enables millions of Haar evaluations per second - critical for real-time sliding window.

---

## AdaBoost Weak Classifiers

Each stage trains a **weak learner** (decision stump on one Haar feature) via AdaBoost to minimize detection error. Weighted combination:

$$
h(x) = \text{sign}\left(\sum_t \alpha_t h_t(x)\right)
$$
> **Readable form:** strong classifier = weighted vote of many simple threshold rules on Haar features

Each cascade stage can be understood as a boosted classifier, but the full detector is not one pooled vote across every feature. It is a sequence of pass/fail stages: fail early and the window is rejected immediately; pass all stages and the window becomes a detection candidate. Prosise links to boosting theory from Course 2 - selected features emphasize hard examples.

---

## Cascade Architecture

Stages arranged sequentially - each must pass to continue:

| Stage | False accept rate | Detection rate |
|-------|-------------------|----------------|
| 1 | High reject of negatives | ~100% of faces pass |
| 2 | Stricter | ~99% |
| ... | ... | ... |
| K | Very strict | ~90% overall |

**Goal:** reject ~99% of non-face windows at stage 1 with tiny compute.

Overall detection:

$$
D = \prod_{k=1}^{K} d_k, \quad F = \prod_{k=1}^{K} f_k
$$
> **Readable form:** total detection rate = product of per-stage detection rates; total false accept = product of per-stage false accepts

---

## OpenCV Implementation

```python
import cv2
import matplotlib.pyplot as plt

# Load pretrained frontal face cascade
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

img_bgr = cv2.imread('group_photo.jpg')
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,   # pyramid scale between window sizes
    minNeighbors=5,    # votes needed per detection
    minSize=(30, 30),  # smallest face
    flags=cv2.CASCADE_SCALE_IMAGE,
)

print(f'Detected {len(faces)} faces')

img_out = img_bgr.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(img_out, (x, y), (x+w, y+h), (0, 255, 0), 2)

img_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.title('Haar Cascade Detections')
plt.show()
```

---

## Key Hyperparameters

| Parameter | Effect if increased | Effect if decreased |
|-----------|---------------------|---------------------|
| `scaleFactor` | Faster, may miss small faces | Slower, more scales |
| `minNeighbors` | Fewer false positives | More detections, more FPs |
| `minSize` | Ignores tiny faces | More compute |

Tune on your image resolution - passport photos vs wide crowd shots differ.

---

## Multi-Scale Sliding Window

Detector scans image pyramid:

```
Original 800×600
Scale 1.0 → window 30×30..800×600
Scale 1.1 → smaller effective window
Scale 1.21 → ...
```

Computational cost grows with image size - Haar remains fast on  VGA, struggles on 4K without downscaling.

---

## Strengths & Weaknesses

| Strength | Weakness |
|----------|----------|
| Extremely fast on CPU | Frontal faces only (default cascade) |
| No GPU required | Poor on profile, occlusion, low light |
| Tiny model (XML file) | High false positives on non-face patterns |
| Great teaching tool | Outperformed by CNN detectors |

**Profile faces:** try `haarcascade_profileface.xml`. **Eyes:** `haarcascade_eye.xml` for rough localization.

---

## Side-by-Side with Chapter 10 CNNs

| Aspect | Haar cascade | CNN detector ([Section 11.4](./section-04-cnn-face-detection.md)) |
|--------|--------------|----------------------------------------------------------|
| Features | Hand-crafted Haar | Learned conv filters |
| Training | Pretrained XML | Pretrained weights |
| Pose invariance | Low | High |
| Speed on CPU | Excellent | Moderate |

Prosise: start Haar for pipeline mechanics, upgrade to CNN when accuracy matters.

---

## Detection Output for Pipeline

```python
def detect_faces_haar(image_path, padding=0.2):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    crops = []
    h_img, w_img = gray.shape
    for (x, y, w, h) in faces:
        pad_w, pad_h = int(w * padding), int(h * padding)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w_img, x + w + pad_w)
        y2 = min(h_img, y + h + pad_h)
        crops.append(img[y1:y2, x1:x2])
    return img, faces, crops
```

Pass crops to [Section 11.3](./section-03-opencv-face-pipeline.md).

---

## Historical Note

Viola-Jones enabled webcam filters, early digital cameras, and OpenCV tutorials for two decades. Understanding cascades clarifies why modern detectors still use **multi-stage rejection** concepts in neural form.

---

## Self-Check

1. What does an integral image accelerate?
2. What is the role of `minNeighbors` in `detectMultiScale`?
3. Why do Haar cascades struggle with profile views?
4. What file format stores OpenCV's pretrained cascades?
5. How does a cascade reduce average compute per window?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 11 - Viola-Jones
- [Viola & Jones, 2001](https://www.merl.com/publications/docs/TR2004-043.pdf)
- [OpenCV CascadeClassifier](https://docs.opencv.org/4.x/d7/d8b/tutorial_py_face_detection.html)
- [Haar cascades XML files](https://github.com/opencv/opencv/tree/master/data/haarcascades)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 11.1](./section-01-detection-vs-recognition.md) | **Next:** [Section 11.3](./section-03-opencv-face-pipeline.md)


