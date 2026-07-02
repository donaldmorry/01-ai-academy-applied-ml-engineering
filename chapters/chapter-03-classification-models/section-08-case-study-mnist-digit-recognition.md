# Section 3.8: Case Study - MNIST Digit Recognition

> **Source:** Prosise, Ch. 3 - "Recognizing Handwritten Digits"  
> **Prerequisites:** [Section 3.1](./section-01-classification-fundamentals.md) | [Section 1.5](../chapter-01-machine-learning/section-05-supervised-learning-k-nearest-neighbors.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [k-nearest-neighbors](../../GLOSSARY.md#k-nearest-neighbors)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## From Tabular Rows to 784 Pixels

Titanic had a dozen columns. Fraud had 30. **MNIST** gives you **28×28 grayscale images** - **784** [features](../../GLOSSARY.md#feature) per digit - and **10 classes** (0 through 9).

Prosise uses MNIST to introduce **multi-class [classification](../../GLOSSARY.md#classification)** and algorithms that scale to high dimensions: [k-NN](../../GLOSSARY.md#k-nearest-neighbors) and SVM. You'll hit **>95% test accuracy** without neural networks - [Chapter 09](../chapter-09-neural-networks/README.md) and [Chapter 10](../chapter-10-convolutional-neural-networks/README.md) will crush that later with CNNs.

> **Humorous analogy:** MNIST is the "Hello, World" of computer vision - if your digit classifier can't read a 7, don't pitch autonomous vehicles yet.

---

## The Dataset

| Split | Samples | Labels |
|-------|---------|--------|
| Train | 60,000 | 0-9 |
| Test | 10,000 | 0-9 |

Each image: 28×28 pixels, values 0-255 (uint8).

```python
import pandas as pd
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data, mnist.target.astype(int)

print(X.shape)   # (70000, 784)
print(y.shape)   # (70000,)
print(f"Pixel range: {X.min()} - {X.max()}")
print(f"Class balance:\n{pd.Series(y).value_counts().sort_index()}")
```

**Balanced classes** (~6,000-7,000 per digit) - [accuracy](../../GLOSSARY.md#accuracy) is meaningful here, unlike [fraud](./section-07-case-study-credit-card-fraud-detection.md).

---

## Step 1: Visualize Digits

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Label: {y[i]}')
    ax.axis('off')
plt.suptitle('First 10 MNIST Samples')
plt.tight_layout()
plt.show()
```

Humans read these instantly. The machine sees a 784-dimensional vector.

---

## Step 2: Train/Test Split

`fetch_openml` merges train+test; use standard split:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, train_size=60000, random_state=42, stratify=y
)
```

Or slice the canonical split: first 60k train, last 10k test (Prosise convention).

---

## Step 3: Scale Pixels (Critical for k-NN and SVM)

Pixel values 0-255 dominate distance metrics. Scale to [0, 1] or standardize:

$$
x'_{j} = \frac{x_j}{255}
$$
> **Readable form:** scaled pixel = original pixel divided by 255

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0
```

For pipelines:

```python
scale_pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    # classifier here
])
```

> **In plain English:** Without scaling, a bright pixel at 200 drowns out a dim pixel at 3 in Euclidean distance - like measuring NYC distances in millimeters for one axis and miles for another.

---

## Step 4: k-Nearest Neighbors (Prosise Baseline)

From [Section 1.5](../chapter-01-machine-learning/section-05-supervised-learning-k-nearest-neighbors.md) - classify by majority vote of $k$ nearest training images in 784-D space.

$$
\hat{y} = \text{mode}\left(y_{i_1}, y_{i_2}, \ldots, y_{i_k}\right)
$$
> **Readable form:** predicted digit = most common label among the k closest training images

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, metric='euclidean')

t0 = time.time()
knn.fit(X_train_scaled, y_train)
train_time = time.time() - t0

t0 = time.time()
y_pred_knn = knn.predict(X_test_scaled)
pred_time = time.time() - t0

print(f"k-NN (k=3) accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"Train time: {train_time:.1f}s | Predict time: {pred_time:.1f}s")
```

| k | Typical test accuracy | Notes |
|---|----------------------|-------|
| 1 | ~97% | Noisy, slow |
| 3 | ~97.5% | Prosise sweet spot |
| 5 | ~97% | Smoother votes |
| 11 | ~96% | Over-smoothing |

**Curse of dimensionality:** 784 dimensions makes "nearest" neighbors less meaningful - yet k-NN still works surprisingly well on MNIST.

```python
# Confusion matrix - which digits get confused?
cm = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.colorbar()
plt.title('k-NN Confusion Matrix')
plt.show()
# Common errors: 4↔9, 3↔8, 5↔6
```

---

## Step 5: Logistic Regression (Multi-Class OvR)

Ten one-vs-rest binary classifiers from [Section 3.2](./section-02-logistic-regression.md):

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(
    max_iter=1000,
    multi_class='ovr',
    solver='lbfgs',
    n_jobs=-1,
    random_state=42,
)

log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
print(f"Logistic OvR accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
```

Linear boundaries in 784-D - fast, ~92-94% accuracy. Not state-of-art but instructive.

---

## Step 6: Support Vector Machine (Prosise's Strong Classic)

SVM finds maximum-margin hyperplanes; with **RBF kernel**, boundaries become nonlinear:

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \| \mathbf{x}_i - \mathbf{x}_j \|^2 \right)
$$
> **Readable form:** RBF kernel = e raised to (negative gamma × squared distance between two images)

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Subsample for grid search speed (optional on slow machines)
subsample = 10000
idx = np.random.RandomState(42).choice(len(X_train_scaled), subsample, replace=False)

param_grid = {
    'C': [1, 5, 10],
    'gamma': ['scale', 0.001, 0.01],
}

svm = SVC(kernel='rbf', random_state=42)
gs = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
gs.fit(X_train_scaled[idx], y_train[idx])
print(f"Best params: {gs.best_params_}")
print(f"CV accuracy: {gs.best_score_:.4f}")

best_svm = gs.best_estimator_
best_svm.fit(X_train_scaled, y_train)
y_pred_svm = best_svm.predict(X_test_scaled)
print(f"SVM test accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
```

Prosise typically reports **~98%+** with tuned RBF SVM - beating k-NN and logistic.

| Algorithm | MNIST test accuracy (typical) | Train speed | Predict speed |
|-----------|------------------------------|-------------|---------------|
| k-NN (k=3) | ~97% | Fast | **Slow** (scan 60k) |
| Logistic OvR | ~93% | Medium | Fast |
| SVM RBF | **~98%** | **Slow** | Medium |
| CNN (later chapters) | 99%+ | Slow | Fast |

---

## Step 7: Random Forest on Pixels (Optional Comparison)

```python
from sklearn.ensemble import RandomForestClassifier

# Subsample - full 60k × 784 is heavy
rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                            n_jobs=-1, random_state=42)
rf.fit(X_train_scaled[:20000], y_train[:20000])
print(f"RF accuracy (20k train): {accuracy_score(y_test, rf.predict(X_test_scaled)):.4f}")
```

Trees on raw pixels work (~97%) but ignore spatial structure - CNNs exploit locality ([Chapter 10](../chapter-10-convolutional-neural-networks/README.md)).

---

## Step 8: Multi-Class Metrics

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_svm, digits=4))
```

| Average | When |
|---------|------|
| `macro` | Care equally about digit "1" and "9" |
| `weighted` | Weight by class frequency (≈ accuracy for balanced) |

Per-class recall shows weak digits (often 5, 8, 9).

---

## Step 9: Inspect Misclassifications

```python
wrong = np.where(y_pred_svm != y_test)[0][:12]
fig, axes = plt.subplots(3, 4, figsize=(10, 7))
for ax, i in zip(axes.ravel(), wrong):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'True:{y_test[i]} Pred:{y_pred_svm[i]}')
    ax.axis('off')
plt.suptitle('SVM Misclassifications')
plt.tight_layout()
plt.show()
```

Ambiguous handwriting - even humans disagree. Model errors are often reasonable.

---

## Step 10: Full Pipeline (Production Pattern)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

mnist_pipe = Pipeline([
    ('scale', MinMaxScaler()),
    ('svm', SVC(kernel='rbf', C=5, gamma='scale', random_state=42)),
])

mnist_pipe.fit(X_train, y_train)  # scaler learns 0-255 range
print(f"Pipeline accuracy: {mnist_pipe.score(X_test, y_test):.4f}")
```

Deploy note: save `mnist_pipe` with joblib in [Chapter 07](../chapter-07-operationalizing-models/README.md).

---

## Flattened Pixels vs Convolution (Preview)

MNIST treats each pixel as independent feature - ignores that **nearby pixels correlate**.

```
Flattened:  [p1, p2, p3, ..., p784]  → dot products / distances

CNN:        2D grid → local filters detect edges, curves, loops
```

| Approach | Inductive bias | MNIST accuracy |
|----------|----------------|----------------|
| Flatten + SVM | None | ~98% |
| Flatten + k-NN | Local similarity | ~97% |
| CNN | Spatial locality | 99%+ |

Prosise's Ch. 3 stops at classical ML - you're ready for [Chapter 09](../chapter-09-neural-networks/README.md).

---

## Dimensionality Note (Link to Chapter 06)

784 features is manageable; 784 irrelevant features hurt. [PCA](../chapter-06-principal-component-analysis/README.md) can reduce dimensions before k-NN/SVM:

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pca_pipe = Pipeline([
    ('scale', MinMaxScaler()),
    ('pca', PCA(n_components=50, random_state=42)),
    ('svm', SVC(kernel='rbf', C=5)),
])
pca_pipe.fit(X_train, y_train)
print(f"PCA(50)+SVM: {pca_pipe.score(X_test, y_test):.4f}")
```

Often **faster** with small accuracy tradeoff.

---

## Chapter Arc Summary

| Case study | Classes | Key section |
|------------|---------|------------|
| [Titanic](./section-06-case-study-titanic-survival.md) | 2 | Encoding, trees, interpretability |
| [Fraud](./section-07-case-study-credit-card-fraud-detection.md) | 2 (rare) | Metrics, thresholds, imbalance |
| **MNIST** | **10** | Multi-class, scaling, k-NN/SVM |

---

## Self-Check

1. Why scale pixel values before k-NN?
2. How does multi-class logistic regression (OvR) work?
3. Which is slower at prediction: k-NN or SVM?
4. Why do 4 and 9 get confused?

---

## References

- [OpenML: mnist_784](https://www.openml.org/d/554)
- [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/)
- [Scikit-Learn: fetch_openml](https://scikit-learn.org/stable/chapters/generated/sklearn.datasets.fetch_openml.html)
- [Scikit-Learn: KNeighborsClassifier](https://scikit-learn.org/stable/chapters/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Scikit-Learn: SVC](https://scikit-learn.org/stable/chapters/generated/sklearn.svm.SVC.html)
- Prosise, *Applied ML and AI for Engineers*, Ch. 3 - MNIST
- [StatQuest: SVM](https://www.youtube.com/watch?v=efR1C6CvhmE)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 3.7](./section-07-case-study-credit-card-fraud-detection.md) | **Next:** [Lab 03](./section-lab-03-three-class-classification-sprint.md)



