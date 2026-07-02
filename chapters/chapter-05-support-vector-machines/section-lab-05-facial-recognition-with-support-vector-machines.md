# Lab 05: Facial Recognition with Support Vector Machines

> **Prerequisites:** Sections [5.1](./section-01-svm-intuition-maximum-margin-and-support-vectors.md)-[5.8](./section-08-svm-production-notes-when-to-deploy-and-onnx-bridge.md)  
> **Estimated time:** 4-6 hours  
> **Glossary:** [svm](../../GLOSSARY.md#svm) | [kernel-trick](../../GLOSSARY.md#kernel-trick) | [cross-validation](../../GLOSSARY.md#cross-validation)  
> **Math:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Objectives

By the end of this lab you will:

1. Build a **face identification system** on the Olivetti Faces dataset (Prosise Ch. 5)
2. Extract **HOG features** and compare against raw pixel baselines
3. Train a tuned **RBF SVM** inside a leakage-free `Pipeline` with `GridSearchCV`
4. Achieve **≥ 88% test accuracy** on held-out faces (40-class problem)
5. Analyze confusion patterns - which subjects get mixed up?
6. Compare **linear vs RBF** kernel training time and accuracy
7. Persist the production pipeline with `joblib` and metadata JSON
8. Optional: export **LinearSVC** baseline to ONNX

> **Humorous briefing:** You have 400 grayscale faces and a Swiss-watch classifier. Your job: teach the machine to tell 40 people apart without mistaking Bob for Carol. Deep learning is banned - this is 2005 airport security cosplay.

---

## Setup

```python
# pip install numpy pandas matplotlib seaborn scikit-learn scikit-image joblib
import warnings; warnings.filterwarnings('ignore')
import json, time, joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score)

from skimage.feature import hog
from skimage import exposure

RANDOM_STATE = 42; np.random.seed(RANDOM_STATE)
MODEL_DIR = Path('lab05_models'); MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path('lab05_results'); RESULTS_DIR.mkdir(exist_ok=True)
```

---

## Part A: Data Exploration (30 min)

*Follow [Section 5.7](./section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md) - Olivetti overview.*

### A1. Load and inspect

```python
faces = fetch_olivetti_faces()
X_pixels, y = faces.data, faces.target
images = faces.images

print(f"Samples: {len(y)}, Features (pixels): {X_pixels.shape[1]}")
print(f"Classes: {len(np.unique(y))}, images/class: {len(y) // len(np.unique(y))}")
print(f"Pixel range: [{X_pixels.min():.3f}, {X_pixels.max():.3f}]")
```

**Tasks:**
1. Plot a 4×10 grid of sample faces (one row per person for first 4 people)
2. Compute mean pixel intensity per person - any outlier lighting?
3. Record class balance (should be perfectly uniform: 10 each)

### A2. Stratified split

```python
X_train_px, X_test_px, y_train, y_test = train_test_split(
    X_pixels, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
images_train, images_test = train_test_split(
    images, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(y_train)}, Test: {len(y_test)}")
```

**Deliverable:** `results/a1_exploration.png` - face grid figure.

---

## Part B: HOG Feature Extraction (45 min)

### B1. Implement HOG extractor

```python
def extract_hog_batch(imgs, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2)):
    """Return (n_samples, n_hog_features) matrix."""
    return np.array([
        hog(img, orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            feature_vector=True)
        for img in imgs
    ])

X_train_hog = extract_hog_batch(images_train)
X_test_hog = extract_hog_batch(images_test)
print(f"HOG shape: train {X_train_hog.shape}, test {X_test_hog.shape}")
```

### B2. Visualize HOG

```python
fd, hog_vis = hog(images_train[0], orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), visualize=True)
hog_vis = exposure.rescale_intensity(hog_vis, in_range=(0, 10))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(images_train[0], cmap='gray'); axes[0].set_title('Original')
axes[1].imshow(hog_vis, cmap='gray'); axes[1].set_title('HOG')
plt.savefig(RESULTS_DIR / 'b2_hog_visualization.png', dpi=120)
plt.show()
```

**Tasks:**
1. Report HOG dimensionality vs raw pixels (4096)
2. In 2-3 sentences: what structure does HOG emphasize?

---

## Part C: Pixel Baseline SVM (45 min)

*Unscaled vs scaled - see [Section 5.4](./section-04-normalization-and-pipelining-for-svms.md).*

### C1. Quick baseline (no tuning)

```python
for scaled in [False, True]:
    if scaled:
        pipe = Pipeline([('scaler', StandardScaler()),
                         ('svc', SVC(kernel='rbf', gamma='scale', C=10))])
    else:
        pipe = Pipeline([('svc', SVC(kernel='rbf', gamma='scale', C=10))])
    pipe.fit(X_train_px, y_train)
    acc = pipe.score(X_test_px, y_test)
    print(f"Pixels, scaled={scaled}: test accuracy = {acc:.3f}")
```

### C2. Grid search on pixels

```python
pixel_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf')),
])
param_grid = {
    'svc__C': [1, 10, 100, 1000],
    'svc__gamma': [0.001, 0.01, 0.1, 'scale'],
}

pixel_search = GridSearchCV(
    pixel_pipe, param_grid, cv=5, n_jobs=-1, verbose=1
)
pixel_search.fit(X_train_px, y_train)

print(f"Best pixel params: {pixel_search.best_params_}")
print(f"Pixel test accuracy: {pixel_search.score(X_test_px, y_test):.3f}")
```

**Deliverable:** Table comparing scaled vs unscaled (Part C1).

---

## Part D: HOG + Tuned RBF SVM - Main Model (60 min)

*Primary deliverable - Prosise Ch. 5 pattern.*

### D1. GridSearchCV on HOG

```python
hog_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf')),
])

hog_search = GridSearchCV(
    hog_pipe, param_grid, cv=5, scoring='accuracy',
    n_jobs=-1, refit=True, verbose=1
)
hog_search.fit(X_train_hog, y_train)

best_hog_model = hog_search.best_estimator_
y_pred = best_hog_model.predict(X_test_hog)

print(f"Best HOG params: {hog_search.best_params_}")
print(f"Best CV score: {hog_search.best_score_:.3f}")
print(f"HOG test accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

### D2. Confusion matrix & heatmap

```python
fig, ax = plt.subplots(figsize=(12, 10))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
plt.title('HOG + RBF SVM - Confusion Matrix')
plt.savefig(RESULTS_DIR / 'd2_confusion_matrix.png', dpi=120)
plt.show()

cm = confusion_matrix(y_test, y_pred)
np.fill_diagonal(cm, 0)
i, j = np.unravel_index(cm.argmax(), cm.shape)
print(f"Most confused pair: true={i}, pred={j}, count={cm[i,j]}")

# C-gamma heatmap
pivot = pd.DataFrame(hog_search.cv_results_).pivot_table(
    values='mean_test_score', index='param_svc__gamma', columns='param_svc__C')
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
plt.title('GridSearchCV - HOG + RBF SVM')
plt.savefig(RESULTS_DIR / 'd3_grid_heatmap.png', dpi=120)
plt.show()
```

**Tasks:** List 3 confused pairs; show 2 misclassified faces; report `n_support_ / n_train`.

---

## Part E: Comparisons & Production (60 min)

### E1. Linear vs RBF + feature table

```python
for kernel in ['linear', 'rbf']:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('svc', SVC(kernel=kernel, C=10, gamma='scale'))])
    t0 = time.perf_counter()
    scores = cross_val_score(pipe, X_train_hog, y_train, cv=5)
    print(f"{kernel}: CV={scores.mean():.3f}±{scores.std():.3f}, time={time.perf_counter()-t0:.1f}s")

# Optional PCA baseline
pca_search = GridSearchCV(
    Pipeline([('scaler', StandardScaler()), ('pca', PCA(100, whiten=True)),
              ('svc', SVC(kernel='rbf'))]),
    param_grid, cv=5, n_jobs=-1)
pca_search.fit(X_train_px, y_train)
print(f"PCA+SVM test: {pca_search.score(X_test_px, y_test):.3f}")
```

Fill this table in your report:

| Feature | Best params | CV acc | Test acc |
|---------|-------------|--------|----------|
| Pixels | | | |
| HOG | | | |
| PCA(100)+pixels | | | |

### E2. Save model + smoke test

```python
joblib.dump(best_hog_model, MODEL_DIR / 'face_svm_hog_v1.joblib')
svc_fitted = best_hog_model.named_steps['svc']
json.dump({
    'best_params': hog_search.best_params_,
    'test_accuracy': float(accuracy_score(y_test, y_pred)),
    'n_support_vectors': int(svc_fitted.support_vectors_.shape[0]),
}, open(MODEL_DIR / 'face_svm_hog_v1.json', 'w'), indent=2)

loaded = joblib.load(MODEL_DIR / 'face_svm_hog_v1.joblib')
assert np.array_equal(loaded.predict(X_test_hog), y_pred)
```

Optional ONNX: export `LinearSVC` pipeline via `skl2onnx` ([Section 5.8](./section-08-svm-production-notes-when-to-deploy-and-onnx-bridge.md)).

---

## Part F: Written Analysis (30 min)

Answer in `lab05_report.md`:

1. Support vector count and fraction of training data
2. Scaling impact on pixel baseline
3. HOG vs pixels - which generalized better and why?
4. Top confused pairs - glasses, lighting, expression?
5. Deploy this SVM vs CNN ([Chapter 10](../chapter-10-convolutional-neural-networks/README.md))?
6. Link to [SVR Section 2.5](../chapter-02-regression-models/section-05-support-vector-regression-svr.md)

**Target:** ≥ 88% test accuracy | leakage-free pipeline | joblib + JSON saved

| Issue | Fix |
|-------|-----|
| ~2.5% accuracy | Add `StandardScaler` |
| Slow grid search | `n_jobs=-1`, smaller grid |
| HOG shape error | Use `images`, not flat `X_pixels` |

---

## References

- [Section 5.7 - Facial Recognition](./section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md)
- [Section 5.3 - Hyperparameter Tuning](./section-03-hyperparameter-tuning-c-gamma-and-gridsearchcv.md)
- [Section 5.8 - Production Notes](./section-08-svm-production-notes-when-to-deploy-and-onnx-bridge.md)
- [Section 2.5 - SVR](../chapter-02-regression-models/section-05-support-vector-regression-svr.md)
- Prosise, Ch. 5
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 5.8](./section-08-svm-production-notes-when-to-deploy-and-onnx-bridge.md) | **Next Chapter:** [Chapter 06 - PCA](../chapter-06-principal-component-analysis/README.md)