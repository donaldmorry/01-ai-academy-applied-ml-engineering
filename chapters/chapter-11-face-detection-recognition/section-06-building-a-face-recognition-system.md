# Section 11.6: Building a Face Recognition System

> **Source:** Prosise, Ch. 11 - end-to-end pipeline, VGGFace transfer learning, `label_faces`  
> **Prerequisites:** [Section 11.5](./section-05-face-embeddings-and-arcface.md) | [Section 11.3](./section-03-opencv-face-pipeline.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [deep-learning](../../GLOSSARY.md#deep-learning) | [keras](../../GLOSSARY.md#keras)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Full Pipeline

Prosise Chapter 11 culminates in a **detect → crop → preprocess → classify → annotate** system:

```
Group photo
    → MTCNN detection ([Section 11.4](./section-04-cnn-face-detection.md))
    → Square crop + resize 224×224 ([Section 11.3](./section-03-opencv-face-pipeline.md))
    → VGGFace + custom softmax head OR ArcFace embedding match
    → Draw box + name label on image
```

Real deployments add logging, access control APIs, and threshold tuning ([Section 11.7](./section-07-open-set-classification.md)) - but the core loop matches Prosise's `label_faces` tutorial.

> **Humorous analogy:** You're running a nightclub bouncer line - detect faces at the door, check IDs against the VIP list, and only stamp wrists above a confidence cutoff.

> **In plain English:** Wire detection, recognition, and visualization into one callable function you can run on any photo folder.

---

## Dataset Design: Ordinary People, Not Celebrities

Prosise deliberately avoids LFW celebrities for the capstone - eight photos each of three people (Jeff, Lori, Abby) across age, glasses, and pose. Images extracted with `extract_faces` from [Section 11.4](./section-04-cnn-face-detection.md).

| Design choice | Rationale |
|---------------|-----------|
| 8 photos / person | Enough variation; still tiny for deep learning |
| 50/50 train/val split (4 train, 4 val) | Stress-tests generalization |
| `Samples/` folder of uncropped group shots | Tests detection + recognition together |
| Small Dense head (8 neurons) | Limits overfitting on 12 training images |

**Folder layout:**

```
Faces/
  Jeff/     # 8 cropped 224×224 portraits
  Lori/
  Abby/
  Samples/  # group photos for inference demo
data/vggface.h5  # pretrained bottleneck weights
```

---

## Loading Enrollment Images

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

def load_images_from_path(path, label):
    images, labels = [], []
    for file in os.listdir(path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(path, file)
            arr = image.img_to_array(
                image.load_img(img_path, target_size=(224, 224))
            )
            images.append(arr)
            labels.append(label)
    return images, labels

def show_images(images, n=8):
    fig, axes = plt.subplots(1, min(n, len(images)), figsize=(20, 3))
    if len(images) == 1:
        axes = [axes]
    for ax, img in zip(axes, images[:n]):
        ax.imshow(img / 255.0)
        ax.set(xticks=[], yticks=[])
    plt.show()

x, y = [], []
for idx, person in enumerate(['Jeff', 'Lori', 'Abby']):
    imgs, labs = load_images_from_path(f'Faces/{person}', idx)
    show_images(imgs)
    x += imgs
    y += labs
```

---

## Preprocess for ResNet50 / VGGFace

VGGFace expects ResNet50 preprocessing (BGR mean subtraction), not simple `/255`:

```python
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input

faces = preprocess_input(np.array(x))
labels = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(
    faces, labels, train_size=0.5, stratify=labels, random_state=0
)
print(x_train.shape, np.unique(y_train, return_counts=True))
```

---

## Transfer Learning Model (Prosise Architecture)

Small head prevents memorizing 12 training chips:

```python
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Resizing, Flatten, Dense

base_model = load_model('data/vggface.h5')
base_model.trainable = False

model = Sequential([
    Resizing(224, 224),
    base_model,
    Flatten(),
    Dense(8, activation='relu'),      # narrow - Prosise's anti-overfit trick
    Dense(3, activation='softmax')    # Jeff, Lori, Abby
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
```

Train with tiny batch size - only 12 training samples:

```python
hist = model.fit(x_train, y_train,
                 validation_data=(x_test, y_test),
                 batch_size=2, epochs=25, verbose=1)
```

Plot train vs validation accuracy - expect high training, moderate validation gap on such a small set.

---

## Detection Helper: `get_face`

Square crop from MTCNN box (matches enrollment preprocessing):

```python
def get_face(np_image, face):
    x1, y1, w, h = face['box']
    if w > h:
        x1 = x1 + ((w - h) // 2)
        w = h
    elif h > w:
        y1 = y1 + ((h - w) // 2)
        h = w
    x2, y2 = x1 + w, y1 + h
    return np_image[y1:y2, x1:x2]
```

---

## The `label_faces` Function

Prosise's production-style inference wrapper:

```python
from mtcnn.mtcnn import MTCNN
from PIL import Image, ImageOps
from matplotlib.patches import Rectangle

def label_faces(path, model, names,
                face_threshold=0.9, prediction_threshold=0.9,
                show_outline=True, figsize=(12, 8)):
    # Fix EXIF orientation
    pil_image = Image.open(path)
    pil_image = ImageOps.exif_transpose(pil_image)
    np_image = np.array(pil_image)

    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={'xticks': [], 'yticks': []})
    ax.imshow(np_image)

    detector = MTCNN()
    faces = detector.detect_faces(np_image)
    faces = [f for f in faces if f['confidence'] >= face_threshold]

    for face in faces:
        x, y, w, h = face['box']
        chip = get_face(np_image, face)
        chip = preprocess_input(np.array(image.array_to_img(chip)))
        preds = model.predict(np.expand_dims(chip, axis=0), verbose=0)
        confidence = float(np.max(preds))

        if confidence >= prediction_threshold:
            idx = int(np.argmax(preds))
            if show_outline:
                ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                       edgecolor='red', linewidth=2))
            ax.text(x + w / 2, y, f'{names[idx]} ({confidence:.1%})',
                    color='white', ha='center', va='bottom', fontweight='bold',
                    bbox=dict(facecolor='red', alpha=0.8, pad=2))
    plt.show()
```

**Two thresholds:**
- `face_threshold` - MTCNN detection confidence
- `prediction_threshold` - softmax max prob before labeling (open-set precursor)

---

## Running on Sample Photos

```python
labels = ['Jeff', 'Lori', 'Abby']
label_faces('Faces/Samples/Sample-1.jpg', model, labels)
label_faces('Faces/Samples/Sample-2.jpg', model, labels,
            prediction_threshold=0.85)
```

Expect correct labels on enrolled people; strangers may be mislabeled as Jeff/Lori/Abby with high softmax confidence - motivation for [Section 11.7](./section-07-open-set-classification.md).

---

## Embedding-Based Alternative

Replace softmax head with [ArcFace gallery](./section-05-face-embeddings-and-arcface.md):

```python
from arcface import ArcFace
af = ArcFace.ArcFace()

GALLERY = {}
for name, folder in [('Jeff', 'Faces/Jeff'), ('Lori', 'Faces/Lori'), ('Abby', 'Faces/Abby')]:
    embs = [af.calc_emb(np.array(Image.open(f'{folder}/{f}')))
            for f in os.listdir(folder) if f.endswith('.jpg')]
    GALLERY[name] = np.mean(embs, axis=0)
```

Same `label_faces` loop, but compare `af.calc_emb(chip)` to gallery centroids with cosine similarity.

---

## System Components Checklist

| Component | Technology | Section |
|-----------|------------|--------|
| Detection | MTCNN or Haar | 11.2, 11.4 |
| Alignment | Landmarks (optional) | 11.4 |
| Preprocess | Square crop, 224×224, preprocess_input | 11.3 |
| Recognizer | VGGFace softmax or ArcFace | 11.5, this section |
| Thresholds | Detection + prediction | 11.7 |
| Ethics review | Consent, bias audit | 11.8 |

---

## Persistence for Deployment

```python
model.save('models/face_recognizer.keras')
import json
meta = {'names': labels, 'face_threshold': 0.9,
        'prediction_threshold': 0.9, 'input_size': 224}
with open('models/face_recognizer_meta.json', 'w') as f:
    json.dump(meta, f)
```

Reload with `load_model` + metadata - same pattern as [Chapter 07](../chapter-07-operationalizing-models/README.md) and [Section 9.8](../chapter-09-neural-networks/section-08-callbacks-training-control-and-model-persistence.md).

---

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| No boxes drawn | `face_threshold` too high | Lower to 0.85 |
| Wrong names on strangers | Closed-set softmax | Switch to embeddings + threshold |
| All labels "Jeff" | Class imbalance / overfit | More photos, smaller head, dropout |
| Sideways faces missed | EXIF not fixed | `ImageOps.exif_transpose` |
| Blurry small faces | No `min_face_size` | Filter or upsample |

---

## Self-Check

1. Why does Prosise use only 8 neurons in the Dense layer?
2. What are the two thresholds in `label_faces`?
3. Why `preprocess_input` instead of dividing by 255?
4. How does the Samples folder differ from enrollment folders?
5. What happens if you pass a photo of yourself to the closed-set model?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 11 - end-to-end tutorial, `label_faces`
- [Keras-vggface / VGGFace2 weights](https://github.com/rcmalli/keras-vggface)
- [MTCNN package](https://github.com/ipazc/mtcnn)
- [Chapter 10 - Transfer Learning](../chapter-10-convolutional-neural-networks/section-05-transfer-learning-with-resnet50v2.md)
- [Section 11.5](./section-05-face-embeddings-and-arcface.md) | [Section 11.7](./section-07-open-set-classification.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 11.5](./section-05-face-embeddings-and-arcface.md) | **Next:** [Section 11.7](./section-07-open-set-classification.md)