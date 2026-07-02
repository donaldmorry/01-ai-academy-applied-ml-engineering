# Lab 10: Arctic Wildlife & Transfer Learning

> **Prerequisites:** Sections [10.1](./section-01-why-cnns-for-images.md)-[10.8](./section-08-cnns-for-audio-spectrograms.md)  
> **Estimated time:** 5-7 hours  
> **Glossary:** [deep-learning](../../GLOSSARY.md#deep-learning) | [classification](../../GLOSSARY.md#classification) | [overfitting](../../GLOSSARY.md#overfitting)  
> **Math:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Objectives

By the end of this lab you will:

1. Train a **custom CNN from scratch** on Arctic wildlife images with data augmentation
2. Achieve **≥80% validation accuracy** on the scratch model
3. Build a **ResNet50V2 transfer learning** pipeline that beats scratch in fewer epochs
4. **Fine-tune** the pretrained base and compare metrics side-by-side
5. Produce training curves, confusion matrices, and a misclassified image gallery
6. Document training time, parameter counts, and when transfer learning is worth it
7. **(Stretch)** Classify a short audio clip via mel spectrogram + small CNN

> **Humorous briefing:** You are the ship's computer vision officer. Four animals, hundreds of tourist photos, one ResNet that's seen more cats than you've seen snow. Prove transfer learning isn't cheating - it's engineering.

---

## Setup

```python
# pip install tensorflow numpy pandas matplotlib seaborn scikit-learn librosa
import warnings; warnings.filterwarnings('ignore')
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DATA_DIR = Path('arctic_wildlife')
SCRATCH_SIZE = (128, 128)
RESNET_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4

OUT = Path('lab10_results'); OUT.mkdir(exist_ok=True)
MODELS = Path('lab10_models'); MODELS.mkdir(exist_ok=True)
```

Download or organize the Prosise Arctic wildlife dataset into `train/`, `val/`, `test/` subfolders ([Section 10.4](./section-04-arctic-wildlife-classifier.md)).

---

## Part A: Data Exploration (30 min)

### A1. Count images per class

```python
for split in ['train', 'val', 'test']:
    print(f'\n=== {split} ===')
    for cls in sorted((DATA_DIR / split).iterdir()):
        if cls.is_dir():
            n = len(list(cls.glob('*.*')))
            print(f'  {cls.name}: {n}')
```

### A2. Visual grid

```python
from tensorflow.keras.preprocessing import image

fig, axes = plt.subplots(4, 6, figsize=(12, 8))
for row, cls in enumerate(sorted((DATA_DIR/'train').iterdir())):
    if not cls.is_dir(): continue
    imgs = list(cls.glob('*.jpg'))[:6]
    for col, p in enumerate(imgs):
        axes[row, col].imshow(image.load_img(p, target_size=SCRATCH_SIZE))
        axes[row, col].set_title(cls.name, fontsize=8)
        axes[row, col].axis('off')
plt.suptitle('Arctic Wildlife Samples')
plt.tight_layout()
plt.savefig(OUT / 'a2_sample_grid.png', dpi=120)
plt.show()
```

**Deliverable:** `lab10_results/a2_sample_grid.png`

---

## Part B: Scratch CNN + Augmentation (90 min)

### B1. Generators

```python
train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
)
plain = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    DATA_DIR/'train', target_size=SCRATCH_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True, seed=RANDOM_STATE,
)
val_gen = plain.flow_from_directory(
    DATA_DIR/'val', target_size=SCRATCH_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False,
)
test_gen = plain.flow_from_directory(
    DATA_DIR/'test', target_size=SCRATCH_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False,
)
class_names = list(train_gen.class_indices.keys())
with open(MODELS/'class_indices.json', 'w') as f:
    json.dump(train_gen.class_indices, f, indent=2)
```

### B2. Build & train scratch model

Use `build_small_cnn` from [Section 10.3](./section-03-building-a-cnn-in-keras.md).

```python
def build_small_cnn(input_shape, num_classes):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])

scratch = build_small_cnn((*SCRATCH_SIZE, 3), NUM_CLASSES)
scratch.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
scratch.summary()

t0 = time.time()
hist_scratch = scratch.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_data=val_gen,
    validation_steps=val_gen.samples // BATCH_SIZE,
    epochs=50,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4),
    ],
)
scratch_time = time.time() - t0
scratch.save(MODELS / 'scratch_cnn.keras')
print(f'Scratch training time: {scratch_time/60:.1f} min')
print(f'Best val accuracy: {max(hist_scratch.history["val_accuracy"]):.4f}')
```

**Gate:** val accuracy ≥ 0.80 - if not, increase augmentation or train longer.

### B3. Plot curves

```python
def plot_history(hist, title, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(hist.history['loss'], label='train')
    ax1.plot(hist.history['val_loss'], label='val')
    ax1.set_title(f'{title} - Loss'); ax1.legend()
    ax2.plot(hist.history['accuracy'], label='train')
    ax2.plot(hist.history['val_accuracy'], label='val')
    ax2.set_title(f'{title} - Accuracy'); ax2.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.show()

plot_history(hist_scratch, 'Scratch CNN', OUT / 'b3_scratch_curves.png')
```

---

## Part C: ResNet50V2 Transfer Learning (90 min)

### C1. ResNet generators (224×224, no rescale - preprocess in model)

```python
train_gen_rn = ImageDataGenerator(
    rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
    zoom_range=0.1, horizontal_flip=True, fill_mode='nearest',
).flow_from_directory(DATA_DIR/'train', target_size=RESNET_SIZE,
    batch_size=16, class_mode='categorical', shuffle=True, seed=RANDOM_STATE)

val_gen_rn = ImageDataGenerator().flow_from_directory(
    DATA_DIR/'val', target_size=RESNET_SIZE, batch_size=16,
    class_mode='categorical', shuffle=False)
```

### C2. Phase 1 - frozen base

```python
base = ResNet50V2(weights='imagenet', include_top=False, input_shape=(*RESNET_SIZE, 3))
base.trainable = False

inp = layers.Input(shape=(*RESNET_SIZE, 3))
x = preprocess_input(inp)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
resnet = models.Model(inp, out)
resnet.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
               loss='categorical_crossentropy', metrics=['accuracy'])

t0 = time.time()
hist_resnet_p1 = resnet.fit(
    train_gen_rn, validation_data=val_gen_rn, epochs=20,
    callbacks=[EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)],
)
p1_time = time.time() - t0
```

### C3. Phase 2 - fine-tune top layers

```python
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

resnet.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
               loss='categorical_crossentropy', metrics=['accuracy'])

t0 = time.time()
hist_resnet_p2 = resnet.fit(
    train_gen_rn, validation_data=val_gen_rn, epochs=15,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    initial_epoch=len(hist_resnet_p1.epoch),
)
resnet_time = p1_time + (time.time() - t0)
resnet.save(MODELS / 'resnet50v2_finetuned.keras')
plot_history(hist_resnet_p2, 'ResNet Fine-tuned', OUT / 'c3_resnet_curves.png')
```

---

## Part D: Evaluation & Comparison (60 min)

### D1. Test set metrics

```python
def evaluate_model(model, gen, name):
    gen.reset()
    probs = model.predict(gen, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = gen.classes
    acc = (y_pred == y_true).mean()
    print(f'\n=== {name} test accuracy: {acc:.4f} ===')
    print(classification_report(y_true, y_pred, target_names=class_names))
    return y_true, y_pred, acc

yt_s, yp_s, acc_scratch = evaluate_model(scratch, test_gen, 'Scratch')
# Rebuild test_gen for ResNet size
test_gen_rn = ImageDataGenerator().flow_from_directory(
    DATA_DIR/'test', target_size=RESNET_SIZE, batch_size=16,
    class_mode='categorical', shuffle=False)
yt_r, yp_r, acc_resnet = evaluate_model(resnet, test_gen_rn, 'ResNet50V2')
```

### D2. Comparison table

```python
comparison = pd.DataFrame([
    {'model': 'Scratch CNN', 'test_acc': acc_scratch, 'train_time_min': scratch_time/60,
     'trainable_params': scratch.count_params()},
    {'model': 'ResNet50V2', 'test_acc': acc_resnet, 'train_time_min': resnet_time/60,
     'trainable_params': sum(tf.keras.backend.count_params(w) for w in resnet.trainable_weights)},
])
print(comparison.to_string(index=False))
comparison.to_csv(OUT / 'd2_comparison.csv', index=False)
```

### D3. Confusion matrices side-by-side

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, yt, yp, title in [
    (axes[0], yt_s, yp_s, 'Scratch'), (axes[1], yt_r, yp_r, 'ResNet')]:
    cm = confusion_matrix(yt, yp)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(title); ax.set_ylabel('True'); ax.set_xlabel('Pred')
plt.tight_layout()
plt.savefig(OUT / 'd3_confusion_matrices.png', dpi=120)
plt.show()
```

### D4. Misclassified gallery (ResNet)

Document 6-12 wrong predictions with filenames and analysis.

---

## Part E: Written Analysis (30 min)

Answer in markdown cells:

1. Did ResNet beat scratch on accuracy **and** training time?
2. Which species pair confuses both models most?
3. When would you **not** use transfer learning (deployment constraints)?
4. What augmentations helped most?

---

## Part F: Stretch - Audio CNN (optional, 60 min)

Follow [Section 10.8](./section-08-cnns-for-audio-spectrograms.md):

1. Collect 20+ clips per class (bird, wind, engine, walrus) × 3s WAV
2. Train `build_audio_cnn` for 30 epochs
3. Save spectrogram visualization + accuracy

---

## Deliverables Checklist

| Item | Path |
|------|------|
| Scratch model | `lab10_models/scratch_cnn.keras` |
| ResNet model | `lab10_models/resnet50v2_finetuned.keras` |
| Class indices JSON | `lab10_models/class_indices.json` |
| Training curves | `lab10_results/b3_*.png`, `c3_*.png` |
| Comparison CSV | `lab10_results/d2_comparison.csv` |
| Confusion matrices | `lab10_results/d3_confusion_matrices.png` |
| Misclassified gallery | `lab10_results/d4_misclassified.png` |
| Written analysis | Notebook markdown |

---

## Rubric (Self-Assessment)

| Criterion | Met? |
|-----------|------|
| Scratch val acc ≥ 80% | ☐ |
| ResNet test acc > scratch test acc | ☐ |
| Side-by-side metrics table | ☐ |
| Training curves plotted | ☐ |
| Misclassification analysis | ☐ |
| Models saved and reloadable | ☐ |

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 10 - lab case study
- [TensorFlow CNN tutorial](https://www.tensorflow.org/tutorials/images/cnn)
- [TensorFlow transfer learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- Sections [10.1](./section-01-why-cnns-for-images.md)-[10.8](./section-08-cnns-for-audio-spectrograms.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 10.8](./section-08-cnns-for-audio-spectrograms.md) | **Next:** [Chapter 11](../chapter-11-face-detection-recognition/README.md)