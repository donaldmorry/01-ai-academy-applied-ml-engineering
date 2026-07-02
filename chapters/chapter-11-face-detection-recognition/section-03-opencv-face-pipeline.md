# Section 11.3: OpenCV Face Pipeline

> **Source:** Prosise, Ch. 11 - detect, crop, resize, color space, preprocessing for recognition  
> **Prerequisites:** [Section 11.2](./section-02-viola-jones-and-haar-cascades.md)  
> **Glossary:** [feature](../../GLOSSARY.md#feature) | [classification](../../GLOSSARY.md#classification)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## From Bounding Box to Recognizer Input

Detection gives rough rectangles. Recognition models expect **standardized face chips** - fixed size, consistent color space, optional alignment. Prosise's OpenCV pipeline:

```
BGR image → grayscale (detector) → bounding boxes → crop → resize → RGB/BGR for embedder
```

Skipping preprocessing is a top cause of "my face recognizer doesn't work" bugs.

> **Humorous analogy:** Detection finds the potato; preprocessing peels and dices it so every dish gets uniform cubes. Serving the recognizer a whole muddy potato yields mashed confusion.

> **In plain English:** Detect, expand the crop slightly, resize to model input (e.g., 160×160), convert colors correctly, then pass to ArcFace.

---

## Color Spaces

OpenCV loads **BGR** by default (historical). Most embedding models expect **RGB**:

```python
import cv2
import numpy as np

img_bgr = cv2.imread('portrait.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

print(img_bgr.shape)  # (H, W, 3)
```

| Stage | Color | Why |
|-------|-------|-----|
| Haar detection | Grayscale | Cascade trained on intensity |
| Visualization | RGB | Matplotlib expects RGB |
| ArcFace / InsightFace | RGB | Pretrained convention |
| Legacy OpenCV LBPH | Grayscale | Local binary patterns |

Always document which convention your pipeline uses.

---

## Complete Detection + Crop Function

```python
import cv2
from pathlib import Path

class FacePipeline:
    def __init__(self, cascade_name='haarcascade_frontalface_default.xml',
                 target_size=(160, 160), padding=0.15):
        path = cv2.data.haarcascades + cascade_name
        self.detector = cv2.CascadeClassifier(path)
        self.target_size = target_size
        self.padding = padding

    def detect(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Histogram equalization helps uneven lighting
        gray = cv2.equalizeHist(gray)
        return self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    def crop_face(self, image_bgr, box):
        x, y, w, h = box
        H, W = image_bgr.shape[:2]
        pad_x = int(w * self.padding)
        pad_y = int(h * self.padding)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(W, x + w + pad_x), min(H, y + h + pad_y)
        return image_bgr[y1:y2, x1:x2]

    def preprocess_chip(self, face_bgr):
        face = cv2.resize(face_bgr, self.target_size, interpolation=cv2.INTER_AREA)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return face_rgb

    def process_image(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(image_path)
        boxes = self.detect(img)
        chips = []
        for box in boxes:
            crop = self.crop_face(img, box)
            chip = self.preprocess_chip(crop)
            chips.append({'box': box, 'chip': chip, 'crop_bgr': crop})
        return img, chips
```

---

## Histogram Equalization

Boosts contrast in shadows - helps Haar on dim photos:

```python
gray_eq = cv2.equalizeHist(gray)
```

For color images before embedding, **CLAHE** on L channel in LAB space is gentler:

```python
lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l = clahe.apply(l)
face_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
```

Use sparingly - over-equalization looks unnatural.

---

## Resize Interpolation

| Method | When |
|--------|------|
| `INTER_AREA` | Downsampling (large crop → 160×160) |
| `INTER_LINEAR` | General |
| `INTER_CUBIC` | Upsampling small faces (slower) |

Recognition models want **square** inputs - non-square crops stretch unless you pad to square first:

```python
def pad_to_square(img):
    h, w = img.shape[:2]
    size = max(h, w)
    pad_top = (size - h) // 2
    pad_left = (size - w) // 2
    return cv2.copyMakeBorder(img, pad_top, size-h-pad_top,
                              pad_left, size-w-pad_left,
                              cv2.BORDER_CONSTANT, value=[0,0,0])
```

---

## Multiple Faces per Image

```python
pipeline = FacePipeline(target_size=(112, 112))
img, chips = pipeline.process_image('family.jpg')

for i, item in enumerate(chips):
    plt.subplot(1, len(chips), i+1)
    plt.imshow(item['chip'])
    plt.title(f'Face {i}'); plt.axis('off')
```

Each chip gets its own embedding in [Section 11.6](./section-06-building-a-face-recognition-system.md).

---

## Batch Processing a Folder

```python
from pathlib import Path

def extract_gallery_chips(root_dir, pipeline):
    gallery = {}
    for person_dir in Path(root_dir).iterdir():
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        gallery[name] = []
        for img_path in person_dir.glob('*.jpg'):
            _, chips = pipeline.process_image(img_path)
            for c in chips:
                gallery[name].append(c['chip'])
    return gallery
```

Expect **multiple chips per file** if group photos appear in enrollment - usually pick largest face or first detection.

---

## Webcam Loop Skeleton

```python
cap = cv2.VideoCapture(0)
pipeline = FacePipeline()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    for (x, y, w, h) in pipeline.detect(frame):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Faces', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

Recognition overlays name labels after embedding match ([Lab 11](./section-lab-11-face-detection-and-recognition-pipeline.md)).

---

## Quality Filters Before Recognition

Reject bad chips to avoid garbage embeddings:

| Check | Rule of thumb |
|-------|---------------|
| Minimum size | Box width ≥ 40 px |
| Blur | Laplacian variance < threshold → skip |
| Aspect ratio | 0.7 < w/h < 1.4 |
| Confidence | CNN detector score ([Section 11.4](./section-04-cnn-face-detection.md)) |

```python
def is_blurry(img_gray, thresh=80.0):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var() < thresh
```

---

## Alignment Preview (5-Point)

When landmarks available (MTCNN), similarity transform to template eye positions:

$$
T = \text{estimateAffinePartial2D}(\text{src\_landmarks}, \text{dst\_template})
$$
> **Readable form:** alignment transform = 2D affine map moving eyes/nose to standard coordinates

InsightFace and MTCNN bundles handle this internally - Prosise OpenCV path uses resize-only baseline.

---

## Standard Input Sizes by Model

| Model | Input size | Normalization |
|-------|------------|---------------|
| ArcFace (InsightFace) | 112×112 | mean/std per channel |
| FaceNet | 160×160 | [-1, 1] |
| OpenCV LBPH | any grayscale | intensity |

Mismatching size/norm destroys accuracy - read model docs.

---

## Persisting Processed Chips

```python
import numpy as np

# Save enrollment chips for debugging
np.save('debug_chip.npy', chip_rgb)
cv2.imwrite('debug_chip.jpg', cv2.cvtColor(chip_rgb, cv2.COLOR_RGB2BGR))
```

---

## Pipeline Checklist

1. ✅ Correct BGR ↔ RGB conversions
2. ✅ Grayscale only for Haar
3. ✅ Padding on crops
4. ✅ Square resize to model size
5. ✅ One face per chip for enrollment portraits
6. ✅ Quality filters on blurry/small detections

---

## Self-Check

1. Why detect on grayscale but embed in RGB?
2. What does `padding` on the bounding box prevent?
3. When use `INTER_AREA` vs `INTER_CUBIC`?
4. Why pad to square before resize?
5. What preprocessing does histogram equalization help?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 11 - OpenCV pipeline
- [OpenCV Color conversions](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html)
- [OpenCV Geometric transformations](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
- [Section 11.2](./section-02-viola-jones-and-haar-cascades.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 11.2](./section-02-viola-jones-and-haar-cascades.md) | **Next:** [Section 11.4](./section-04-cnn-face-detection.md)



