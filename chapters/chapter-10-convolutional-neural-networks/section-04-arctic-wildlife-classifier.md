# Section 10.4: Arctic Wildlife Classifier

> **Source:** Prosise, Ch. 10 - custom image dataset, directory structure, training from scratch  
> **Prerequisites:** [Section 10.3](./section-03-building-a-cnn-in-keras.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [feature](../../GLOSSARY.md#feature) | [overfitting](../../GLOSSARY.md#overfitting)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Prosise Case Study

Chapter 10 trains a CNN on **Arctic wildlife** photographs - a multi-class [classification](../../GLOSSARY.md#classification) problem with natural backgrounds, variable lighting, and limited samples per species. Typical classes:

| Class | Visual cues | Common confusions |
|-------|-------------|-------------------|
| Polar bear | White fur, ice, large body | Walrus on ice |
| Penguin | Black/white plumage, upright | - |
| Walrus | Tusks, brown skin, bulky | Polar bear |
| Arctic fox | Small, bushy tail, white/gray coat | Polar bear cub at distance |

> **Humorous analogy:** You're building a zoologist bot for a cruise ship - passengers keep photographing the same four animals, but every photo has different clouds, angles, and someone wearing a red jacket in the background.

> **In plain English:** Organize photos in folders, load them with Keras, train the CNN from [Section 10.3](./section-03-building-a-cnn-in-keras.md), and evaluate honestly on held-out images.

---

## Directory Layout

Keras expects one subfolder per class:

```
arctic_wildlife/
├── train/
│   ├── arctic_fox/
│   │   ├── fox_001.jpg
│   │   └── ...
│   ├── penguin/
│   ├── polar_bear/
│   └── walrus/
├── val/
│   ├── arctic_fox/
│   ├── penguin/
│   ├── polar_bear/
│   └── walrus/
└── test/
    ├── arctic_fox/
    ├── penguin/
    ├── polar_bear/
    └── walrus/
```

**Rules:**
- No spaces in folder names (use underscores)
- Supported formats: JPG, PNG, BMP
- Aim for **stratified splits** - equal proportion of each class in train/val/test
- Prosise suggests ~70% train, 15% val, 15% test

```python
from pathlib import Path

root = Path('arctic_wildlife/train')
for cls in sorted(root.iterdir()):
    if cls.is_dir():
        n = len(list(cls.glob('*.*')))
        print(f'{cls.name}: {n} images')
```

---

## Stratified Split Script

If you start from one folder per class:

```python
import shutil
import random
from pathlib import Path

random.seed(42)
SRC = Path('arctic_raw')
DST = Path('arctic_wildlife')
SPLITS = {'train': 0.70, 'val': 0.15, 'test': 0.15}

for split in SPLITS:
    for cls_dir in SRC.iterdir():
        (DST / split / cls_dir.name).mkdir(parents=True, exist_ok=True)

for cls_dir in SRC.iterdir():
    images = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png'))
    random.shuffle(images)
    n = len(images)
    n_train = int(n * SPLITS['train'])
    n_val = int(n * SPLITS['val'])
    for i, img in enumerate(images):
        if i < n_train:
            split = 'train'
        elif i < n_train + n_val:
            split = 'val'
        else:
            split = 'test'
        shutil.copy(img, DST / split / cls_dir.name / img.name)
```

---

## Loading with ImageDataGenerator

Prosise uses `flow_from_directory` - scales pixels and batches automatically:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'arctic_wildlife/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
)

val_gen = val_datagen.flow_from_directory(
    'arctic_wildlife/val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)

print(train_gen.class_indices)  # {'arctic_fox': 0, ...}
class_names = list(train_gen.class_indices.keys())
```

**`class_indices`** maps folder names to integers - save this mapping for inference.

---

## Modern Alternative: tf.keras.utils.image_dataset_from_directory

TensorFlow 2.x native pipeline:

```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    'arctic_wildlife/train',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'arctic_wildlife/val',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Normalize
normalization = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization(x), y))
```

Both approaches are valid; Prosise's text follows `ImageDataGenerator`.

---

## Visualize the Data

```python
import matplotlib.pyplot as plt

images, labels = next(train_gen)
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    idx = int(labels[i].argmax())
    plt.title(class_names[idx])
    plt.axis('off')
plt.suptitle('Arctic Wildlife Training Batch')
plt.tight_layout()
plt.show()
```

Inspect for mislabeled files, corrupt images, and class imbalance.

---

## Train from Scratch

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model = build_small_cnn(input_shape=(*IMG_SIZE, 3), num_classes=len(class_names))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
]

steps_per_epoch = train_gen.samples // BATCH_SIZE
val_steps = val_gen.samples // BATCH_SIZE

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=50,
    callbacks=callbacks,
)
```

**Target:** >80% validation accuracy per chapter README - achievable with clean data and 100+ images per class.

---

## Class Imbalance

If polar bears dominate the dataset:

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

labels = train_gen.classes
weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight = dict(enumerate(weights))
print(class_weight)

history = model.fit(train_gen, ..., class_weight=class_weight)
```

Or collect more photos of underrepresented species - often better than weighting alone.

---

## steps_per_epoch vs samples

| Generator API | Epoch definition |
|---------------|------------------|
| `flow_from_directory` | `steps = samples // batch_size` |
| `image_dataset_from_directory` | One epoch = one full pass (default) |

Off-by-one errors in steps cause under-training - verify `train_gen.samples`.

---

## Error Analysis: Misclassified Gallery

```python
import numpy as np

val_gen.reset()
probs = model.predict(val_gen, verbose=1)
y_pred = np.argmax(probs, axis=1)
y_true = val_gen.classes

wrong_idx = np.where(y_pred != y_true)[0][:12]
val_gen.reset()
images, _ = next(val_gen)  # reload carefully - better: iterate filenames

# Manual loop over val folder for gallery
from tensorflow.keras.preprocessing import image

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for ax, i in zip(axes.ravel(), wrong_idx[:12]):
    # load image i from val set...
    pass
plt.suptitle('Misclassified Arctic Wildlife')
plt.show()
```

Document **why** errors happen: occlusion, blur, wrong species in background.

---

## Baseline Comparison

Before celebrating CNN accuracy, compare to a classical baseline:

```python
# Flatten + logistic regression on same data (slow but instructive)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Extract flattened training features via generator (subsample if needed)
```

| Model | Val accuracy (typical) |
|-------|------------------------|
| Logistic on pixels | 40-55% |
| HOG + SVM | 55-70% |
| CNN scratch | 75-85% |
| CNN + augmentation | 80-88% |

CNN wins because it learns spatial [features](../../GLOSSARY.md#feature) - not because ML magic bypasses data limits.

---

## Overfitting Signals

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| train acc 95%, val 60% | [Overfitting](../../GLOSSARY.md#overfitting) | Augmentation, Dropout, smaller model |
| both acc ~25% (4 classes) | Not learning | LR, labels, broken paths |
| val loss increases | Stop training | EarlyStopping already helps |

[Section 10.6](./section-06-data-augmentation.md) is the next lever before jumping to transfer learning.

---

## Persist Artifacts

```python
import json

model.save('models/arctic_scratch.keras')
with open('models/class_indices.json', 'w') as f:
    json.dump(train_gen.class_indices, f, indent=2)

with open('models/training_history.json', 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
```

Production systems need model + label map + preprocessing spec (128×128, rescale 1/255).

---

## Data Collection Tips (Real Projects)

1. **Minimum** ~50 images per class to start; 200+ for reliable scratch training
2. Include diverse poses, seasons, and backgrounds
3. Remove near-duplicate frames from video bursts
4. Verify EXIF orientation - some JPGs display rotated
5. Hold out entire **photo sessions** for test if possible (reduces leakage)

---

## Self-Check

1. What folder structure does `flow_from_directory` require?
2. Why save `class_indices`?
3. How do you compute `steps_per_epoch`?
4. What validation accuracy target does the chapter set?
5. Which species pair is most often confused?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 10 - Arctic wildlife dataset
- [Keras ImageDataGenerator](https://keras.io/api/preprocessing/image/)
- [tf.keras.utils.image_dataset_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory)
- [Section 10.3](./section-03-building-a-cnn-in-keras.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 10.3](./section-03-building-a-cnn-in-keras.md) | **Next:** [Section 10.5](./section-05-transfer-learning-with-resnet50v2.md)