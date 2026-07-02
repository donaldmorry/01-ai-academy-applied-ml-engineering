# Section 9.6: Multi-Class Classification - Face Recognition

> **Source:** Prosise, Ch. 9 - "Multiclass Classification with Neural Networks" & "Training a Neural Network to Recognize Faces"  
> **Prerequisites:** [Section 9.5](./section-05-binary-classification-credit-card-fraud-detection.md) | [Chapter 05 Section 5.7](../chapter-05-support-vector-machines/section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [neural-network](../../GLOSSARY.md#neural-network)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Binary → Multi-Class: Three Changes

Prosise's modification checklist:

| Component | Binary | Multi-class (K classes) |
|-----------|--------|-------------------------|
| Output neurons | 1 | **K** (one per class) |
| Output activation | `sigmoid` | **`softmax`** |
| Loss | `binary_crossentropy` | **`sparse_categorical_crossentropy`** |

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

K = 5  # number of people

model = Sequential([
    Dense(128, activation='relu', input_shape=(2,)),
    Dense(K, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

**Labels:** integers `0` to `K-1` - no one-hot encoding needed with `sparse_categorical_crossentropy`.

---

## Softmax

For logits $\mathbf{z} = (z_1, \ldots, z_K)$:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$
> **Readable form:** each output is a positive number; all K outputs sum to 1.0 - a probability distribution over classes

```python
import numpy as np

def softmax(z):
    exp_z = np.exp(z - z.max())  # numerical stability
    return exp_z / exp_z.sum()

probs = softmax(np.array([2.0, 0.5, -1.0, 0.1]))
print(probs, probs.sum())  # sums to 1.0
```

---

## Synthetic Four-Class Demo (Prosise Fig. 9-4)

```python
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X, y = make_blobs(n_samples=400, centers=4, cluster_std=1.2, random_state=42)
X = X.astype(np.float32)
y = y.astype(np.int32)

model = Sequential([
    Dense(128, activation='relu', input_shape=(2,)),
    Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(X, y, epochs=40, batch_size=10, validation_split=0.2, verbose=0)
```

### Predictions

```python
probs = model.predict(np.array([[0.2, 0.8]], dtype=np.float32), verbose=0)
print(f"Class probabilities: {probs[0]}")
# e.g. [0.02, 0.00005, 0.05, 0.93] - class 3 wins

pred_class = np.argmax(probs, axis=1)
print(f"Predicted class: {pred_class[0]}")
```

---

## `sparse_` vs `categorical` Cross-Entropy

| Loss | Label format |
|------|--------------|
| `sparse_categorical_crossentropy` | Integer column: `[0, 2, 1, 3, ...]` |
| `categorical_crossentropy` | One-hot matrix via `keras.utils.to_categorical` |

```python
from tensorflow.keras.utils import to_categorical

y_onehot = to_categorical(y, num_classes=4)
# [[1,0,0,0], [0,0,1,0], ...]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fit with y_onehot instead of y
```

Prosise prefers **sparse** - simpler labels, same math.

---

## Labeled Faces in the Wild (LFW)

Chapter 5 trained an SVM on Olivetti; Chapter 9 uses **LFW** - famous people, built into scikit-learn:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from collections import Counter

faces = fetch_lfw_people(min_faces_per_person=100, resize=0.4)
image_count = faces.images.shape[0]
image_height = faces.images.shape[1]
image_width = faces.images.shape[2]
class_count = len(faces.target_names)

print(faces.target_names)
print(f"Images: {image_count}, shape: {faces.images.shape}")
print(f"Classes: {class_count}")
```

**Prosise's filter:** `min_faces_per_person=100` → **5 people**, **1,140 images** (47×62 pixels).

### Visualize

```python
fig, axes = plt.subplots(3, 8, figsize=(18, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.images[i], cmap='gist_gray')
    ax.set(xticks=[], yticks=[],
           xlabel=faces.target_names[faces.target[i]])
plt.tight_layout()
plt.show()
```

### Class balance

```python
counts = Counter(faces.target)
names = {faces.target_names[k]: counts[k] for k in counts}
pd.DataFrame.from_dict(names, orient='index', columns=['count']).plot(kind='bar')
plt.title('Images per person (before balancing)')
plt.show()
```

George W. Bush dominates - imbalance hurts classifiers.

---

## Balance to 100 Images Per Person

```python
mask = np.zeros(faces.target.shape, dtype=bool)
for target in np.unique(faces.target):
    mask[np.where(faces.target == target)[0][:100]] = True

X_faces = faces.data[mask]
y_faces = faces.target[mask]
print(X_faces.shape, y_faces.shape)  # (500, 2914), (500,)
```

---

## Preprocess & Split

```python
from sklearn.model_selection import train_test_split

# Normalize pixels to [0, 1]
face_images = X_faces / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    face_images, y_faces,
    train_size=0.8,
    stratify=y_faces,
    random_state=0
)
```

**Prosise note:** dividing by 255 vs `StandardScaler` gave similar results - try both on your hardware.

---

## Face Recognition Network

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

input_dim = image_width * image_height  # 47 * 62 = 2914

model = Sequential([
    Dense(512, activation='relu', input_shape=(input_dim,)),
    Dense(class_count, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=20,
    verbose=1
)
```

Flattened pixels → Dense layers. [Chapter 10](../chapter-10-convolutional-neural-networks/README.md) replaces this with CNNs that respect spatial structure.

---

## Training Curves

```python
import seaborn as sns
sns.set_theme()

hist = history.history
epochs = range(1, len(hist['accuracy']) + 1)

plt.figure(figsize=(8, 4))
plt.plot(epochs, hist['accuracy'], '-', label='Training accuracy')
plt.plot(epochs, hist['val_accuracy'], ':', label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('LFW Face Recognition - Dense Network')
plt.legend(loc='lower right')
plt.show()
```

**Typical pattern (Prosise):** training accuracy → ~100%, validation **80-85%** - sign of [overfitting](../../GLOSSARY.md#overfitting). [Section 9.7](./section-07-dropout-and-regularization.md) addresses this.

---

## Confusion Matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

y_prob = model.predict(X_test, verbose=0)
y_pred = y_prob.argmax(axis=1)

print(classification_report(y_test, y_pred, target_names=faces.target_names))

fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=faces.target_names,
    cmap='Blues', xticks_rotation='vertical',
    colorbar=False, ax=ax
)
plt.tight_layout()
plt.show()
```

**Analysis questions (Prosise):**
- How often is George W. Bush confused with Colin Powell?
- Would 128 hidden neurons match 512?

---

## Compare to Chapter 05 SVM

| Approach | Features | Typical accuracy |
|----------|----------|------------------|
| SVM + HOG (Chapter 05) | Engineered gradients | ~88%+ on Olivetti |
| Dense NN on pixels (this section) | Raw flattened pixels | ~80-85% val on LFW subset |
| CNN (Chapter 10) | Learned spatial filters | Best |

```python
# Different datasets - comparison is directional, not apples-to-apples
# Olivetti: 40 classes, 400 images
# LFW subset: 5 classes, 500 images
```

Neural networks on **raw pixels** without convolutions leave performance on the table - try width sweep (128 vs 512) per [Section 9.3](./section-03-network-sizing-and-architecture.md).

---

## Self-Check

1. Why does softmax replace sigmoid for multi-class output?
2. When do you use `sparse_categorical_crossentropy` vs `categorical_crossentropy`?
3. Why balance LFW to 100 images per person?
4. Why does training accuracy hit ~100% while validation stalls at ~85%?

---

## What's Next

[Section 9.7](./section-07-dropout-and-regularization.md) introduces **dropout** to close the train/validation gap on this face model.

---

**Previous:** [Section 9.5](./section-05-binary-classification-credit-card-fraud-detection.md)  
**Next:** [Section 9.7 - Dropout & Regularization](./section-07-dropout-and-regularization.md)



