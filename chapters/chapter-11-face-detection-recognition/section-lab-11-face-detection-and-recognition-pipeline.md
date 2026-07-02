# Lab 11: Face Detection & Recognition Pipeline

> **Prerequisites:** Sections [11.1](./section-01-detection-vs-recognition.md)-[11.8](./section-08-ethics-and-deployment.md)  
> **Estimated time:** 4-6 hours  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [deep-learning](../../GLOSSARY.md#deep-learning)  
> **Math:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Objectives

Build a complete face system following Prosise Chapter 11:

1. **Detect** faces with Haar cascades and MTCNN - compare speed and accuracy  
2. **Enroll** a gallery of 5+ people (3-5 photos each) with ArcFace embeddings  
3. **Recognize** probes via nearest-neighbor matching with tuned threshold  
4. **Evaluate** open-set behavior with strangers - plot TAR/FAR vs threshold  
5. **Document** ethical constraints and deployment limits

> **Humorous briefing:** You are the bouncer, the ID checker, and the privacy officer - same notebook, three hats.

---

## Setup

```python
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn.mtcnn import MTCNN

# Optional: pip install arcface opencv-python mtcnn
try:
    from arcface import ArcFace
    embedder = ArcFace.ArcFace()
except ImportError:
    raise ImportError('Install arcface: pip install arcface')

DATA_DIR = Path('data/lab11')
GALLERY_DIR = DATA_DIR / 'gallery'   # subfolders per person
GROUP_DIR = DATA_DIR / 'group_photos'
STRANGER_DIR = DATA_DIR / 'strangers'  # people NOT in gallery
OUT_DIR = Path('lab11_results')
OUT_DIR.mkdir(exist_ok=True)
SEED = 42
```

**Data layout:**

```
data/lab11/
  gallery/
    alice/  photo1.jpg ...
    bob/
    ...
  group_photos/  party.jpg office.jpg
  strangers/     unknown1.jpg ...
```

Use your own photos with consent, or cropped LFW/Olivetti subsets for practice.

---

## Part A: Haar Detection (45 min)

*Follow [Sections 11.2](./section-02-viola-jones-and-haar-cascades.md) & [11.3](./section-03-opencv-face-pipeline.md).*

```python
def detect_haar(image_path, scale=1.1, min_neighbors=5, min_size=(40, 40)):
    img_bgr = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    boxes = cascade.detectMultiScale(
        gray, scaleFactor=scale, minNeighbors=min_neighbors, minSize=min_size)
    return img_bgr, boxes

def draw_boxes_rgb(img_bgr, faces, title='Haar'):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img_rgb)
    for (x, y, w, h) in faces:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='lime', lw=2))
    ax.set_title(f'{title}: {len(faces)} faces')
    ax.axis('off')
    plt.savefig(OUT_DIR / f'haar_{Path(image_path).stem}.png', dpi=120)
    plt.show()

for p in list(GROUP_DIR.glob('*.jpg'))[:3]:
    img, faces = detect_haar(p)
    draw_boxes_rgb(img, faces)
```

**Tune `minNeighbors`** - record false positives vs missed faces in a table.

| Image | minNeighbors | Detected | False positives | Missed |
|-------|--------------|----------|-----------------|--------|
| party.jpg | 3 | | | |
| party.jpg | 20 | | | |

---

## Part B: MTCNN Detection (45 min)

*Follow [Section 11.4](./section-04-cnn-face-detection.md).*

```python
detector = MTCNN()

def detect_mtcnn(image_path, conf_thresh=0.9, min_face_size=40):
    pil = ImageOps.exif_transpose(Image.open(image_path).convert('RGB'))
    img = np.array(pil)
    faces = detector.detect_faces(img)
    faces = [f for f in faces if f['confidence'] >= conf_thresh
             and f['box'][2] >= min_face_size]
    return img, faces

def draw_mtcnn(img, faces):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    for f in faces:
        x, y, w, h = f['box']
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='red', lw=2))
        ax.text(x, y - 4, f"{f['confidence']:.2f}", color='yellow', fontsize=8)
    ax.set_title(f'MTCNN: {len(faces)} faces')
    ax.axis('off')
    plt.show()

for p in list(GROUP_DIR.glob('*.jpg'))[:3]:
    img, faces = detect_mtcnn(p)
    draw_mtcnn(img, faces)
```

**Deliverable:** Side-by-side note - which detector found profile/small faces?

---

## Part C: Crop & Preprocess Utility (30 min)

```python
def square_crop(img, box):
    x, y, w, h = box
    if w > h:
        x += (w - h) // 2; w = h
    elif h > w:
        y += (h - w) // 2; h = w
    return img[y:y+h, x:x+w]

def face_chip(img, face, size=224):
    chip = square_crop(img, face['box'])
    chip = cv2.resize(chip, (size, size), interpolation=cv2.INTER_AREA)
    return chip
```

---

## Part D: Enrollment Gallery (60 min)

*Follow [Sections 11.5](./section-05-face-embeddings-and-arcface.md) & [11.6](./section-06-building-a-face-recognition-system.md).*

```python
GALLERY = {}  # name -> centroid embedding (512,)

for person_dir in sorted(GALLERY_DIR.iterdir()):
    if not person_dir.is_dir():
        continue
    embs = []
    for img_path in person_dir.glob('*.*'):
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        pil = ImageOps.exif_transpose(Image.open(img_path).convert('RGB'))
        pil = pil.resize((224, 224))
        embs.append(embedder.calc_emb(np.array(pil)))
    if embs:
        GALLERY[person_dir.name] = np.mean(embs, axis=0)
        print(f"Enrolled {person_dir.name}: {len(embs)} photos")

print(f'Gallery size: {len(GALLERY)} people')
```

Require **≥5 people**, **≥3 photos** each.

---

## Part E: Recognition + Threshold Sweep (90 min)

*Follow [Section 11.7](./section-07-open-set-classification.md).*

```python
def match(probe_emb, gallery):
    best_name, best_sim = 'UNKNOWN', -1.0
    for name, centroid in gallery.items():
        sim = float(cosine_similarity([probe_emb], [centroid])[0, 0])
        if sim > best_sim:
            best_name, best_sim = name, sim
    return best_name, best_sim

def collect_similarities(gallery_dir, gallery, label_is_genuine=True):
    sims, labels = [], []
    for person_dir in gallery_dir.iterdir():
        if not person_dir.is_dir():
            continue
        genuine = person_dir.name in gallery
        for img_path in person_dir.glob('*.*'):
            pil = ImageOps.exif_transpose(Image.open(img_path).convert('RGB'))
            pil = pil.resize((224, 224))
            emb = embedder.calc_emb(np.array(pil))
            pred, sim = match(emb, gallery)
            if label_is_genuine and genuine:
                sims.append(float(cosine_similarity([emb], [gallery[person_dir.name]])[0, 0]))
                labels.append(1)
            elif not label_is_genuine or not genuine:
                sims.append(sim if genuine else sim)
                labels.append(1 if (genuine and pred == person_dir.name) else 0)
    return sims, labels
```

Build **genuine** list (correct person similarity) and **impostor** list (strangers or wrong match):

```python
genuine_sims, impostor_sims = [], []

# Held-out: last photo per enrolled person as probe
for name, centroid in GALLERY.items():
    paths = sorted((GALLERY_DIR / name).glob('*.jpg'))
    if len(paths) < 2:
        continue
    probe = embedder.calc_emb(np.array(
        ImageOps.exif_transpose(Image.open(paths[-1]).convert('RGB')).resize((224, 224))))
    genuine_sims.append(float(cosine_similarity([probe], [centroid])[0, 0]))

for sp in STRANGER_DIR.glob('*.*'):
    if sp.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
        continue
    probe = embedder.calc_emb(np.array(
        ImageOps.exif_transpose(Image.open(sp).convert('RGB')).resize((224, 224))))
    _, sim = match(probe, GALLERY)
    impostor_sims.append(sim)

thresholds = np.arange(0.30, 0.65, 0.02)
rows = []
for t in thresholds:
    tar = np.mean(np.array(genuine_sims) >= t) if genuine_sims else 0
    far = np.mean(np.array(impostor_sims) >= t) if impostor_sims else 0
    rows.append({'threshold': t, 'TAR': tar, 'FAR': far})
df_thresh = pd.DataFrame(rows)
df_thresh.to_csv(OUT_DIR / 'threshold_sweep.csv', index=False)

plt.figure(figsize=(8, 5))
plt.plot(df_thresh['FAR'], df_thresh['TAR'], 'o-')
plt.xlabel('FAR (impostor accept rate)')
plt.ylabel('TAR (genuine accept rate)')
plt.title('Face recognition operating curve')
plt.grid(alpha=0.3)
plt.savefig(OUT_DIR / 'tar_far_curve.png', dpi=120)
plt.show()

OPERATING_THRESHOLD = 0.45  # pick from curve - document choice
```

---

## Part F: Annotate Group Photos (45 min)

```python
def label_group_photo(path, gallery, embedder, threshold=0.45):
    img, faces = detect_mtcnn(path, conf_thresh=0.85)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    for f in faces:
        chip = face_chip(img, f)
        emb = embedder.calc_emb(chip)
        name, sim = match(emb, gallery)
        if sim < threshold:
            name = f'UNKNOWN ({sim:.2f})'
        else:
            name = f'{name} ({sim:.2f})'
        x, y, w, h = f['box']
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='cyan', lw=2))
        ax.text(x, y - 2, name, color='white', fontsize=9,
                bbox=dict(facecolor='black', alpha=0.6))
    ax.axis('off')
    plt.savefig(OUT_DIR / f'labeled_{Path(path).stem}.png', dpi=120)
    plt.show()

for gp in GROUP_DIR.glob('*.jpg'):
    label_group_photo(gp, GALLERY, embedder, OPERATING_THRESHOLD)
```

---

## Part G: Ethics Statement (30 min)

*Follow [Section 11.8](./section-08-ethics-and-deployment.md).*

Write `lab11_results/ethics_statement.md` (½ page):

1. Who provided enrollment photos and was consent obtained?  
2. Chosen threshold and implied FAR/TAR - acceptable for your use case?  
3. Known bias/limitation risks  
4. Would you deploy this without liveness detection? Why or why not?

---

## Deliverables Checklist

| Item | Path |
|------|------|
| Haar vs MTCNN comparison table | notebook / markdown |
| Enrolled gallery (≥5 people) | `data/lab11/gallery/` |
| TAR/FAR curve | `lab11_results/tar_far_curve.png` |
| Threshold CSV | `lab11_results/threshold_sweep.csv` |
| Annotated group photos | `lab11_results/labeled_*.png` |
| Ethics statement | `lab11_results/ethics_statement.md` |
| Single notebook | `lab11_face_pipeline.ipynb` |

---

## Stretch Goals

1. **Webcam demo** - `cv2.VideoCapture(0)` + MTCNN every N frames  
2. **Haar vs MTCNN latency** - time 100 frames on CPU  
3. **VGGFace softmax** - compare closed-set mislabels on strangers vs ArcFace open-set  
4. **ONNX export** - embedder via InsightFace for edge deployment preview

---

## Self-Check

1. Why evaluate strangers separately from enrolled accuracy?  
2. What happens if `OPERATING_THRESHOLD` is too low? Too high?  
3. Why square-crop before embedding?  
4. Name one ethical reason Meta stopped auto face tagging.

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 11 - full pipeline lab
- [MTCNN](https://github.com/ipazc/mtcnn) | [ArcFace package](https://pypi.org/project/arcface/)
- [OpenCV Haar cascades](https://docs.opencv.org/4.x/d7/d8b/tutorial_py_face_detection.html)
- [Chapter 11 README](./README.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 11.8](./section-08-ethics-and-deployment.md)  
**Next:** [Chapter 12 - Object Detection](../chapter-12-object-detection/README.md)