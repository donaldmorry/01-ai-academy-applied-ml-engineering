# Section 5.7: Facial Recognition with SVM - HOG & Olivetti Faces

> **Source:** Prosise, Ch. 5 - "Facial Recognition with Support Vector Machines" (Olivetti faces)  
> **Prerequisites:** [Section 5.5](./section-05-svm-classification-svc-for-binary-and-multiclass.md) | [Section 5.3](./section-03-hyperparameter-tuning-c-gamma-and-gridsearchcv.md)  
> **Glossary:** [svm](../../GLOSSARY.md#svm) | [classification](../../GLOSSARY.md#classification) | [feature](../../GLOSSARY.md#feature)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Why Faces + SVM?

Prosise's Chapter 5 capstone: recognize **which person** appears in a grayscale face image. This is classic [multiclass classification](../../GLOSSARY.md#classification) with:

- **High-dimensional inputs** - 64×64 pixels = 4,096 [features](../../GLOSSARY.md#feature) per image
- **Relatively few samples** - 10 images × 40 people = 400 total
- **Nonlinear decision boundaries** - faces differ subtly in geometry

[SVMs](../../GLOSSARY.md#svm) with RBF kernels excel here: maximum margin in a kernel space, sparse support vectors, strong accuracy without neural networks. [Chapter 11](../chapter-09-neural-networks/README.md) CNNs will surpass this later - but Prosise wants you to see **classical vision + SVM** first.

> **Humorous analogy:** You're building airport security from 2005 - no deep learning, just gradients, histograms, and a very patient SVM reading 4,096 numbers per face.

> **In plain English:** Extract numeric features from face images, scale them, train a tuned RBF SVM, and evaluate who gets confused with whom.

---

## The Olivetti Faces Dataset

Scikit-learn bundles the AT&T Olivetti Faces corpus:

| Property | Value |
|----------|-------|
| Subjects | 40 people |
| Images per person | 10 |
| Total images | 400 |
| Resolution | 64 × 64 pixels (grayscale) |
| Labels | Person ID 0-39 |

```python
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import numpy as np

faces = fetch_olivetti_faces()
X_pixels, y = faces.data, faces.target
images = faces.images  # (400, 64, 64) for visualization

print(f"Feature matrix: {X_pixels.shape}")  # (400, 4096)
print(f"Labels: {y.shape}, classes: {np.unique(y).size}")
print(f"Pixel range: [{X_pixels.min():.2f}, {X_pixels.max():.2f}]")

fig, axes = plt.subplots(4, 10, figsize=(12, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(images[i], cmap='gray')
    ax.set_title(f'{y[i]}', fontsize=7)
    ax.axis('off')
plt.suptitle('Olivetti Faces - 40 subjects × 10 images')
plt.tight_layout()
plt.show()
```

Each person has varied expressions, lighting, and accessories - realistic small-data challenge.

---

## Feature Strategy 1: Raw Pixels (Prosise Baseline)

Flattened pixels work with scaling + RBF SVM:

$$
\mathbf{x} \in \mathbb{R}^{4096}, \quad x_j \in [0, 1]
$$
> **Readable form:** each image becomes a 4096-dimensional vector of normalized pixel intensities

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

X_train, X_test, y_train, y_test = train_test_split(
    X_pixels, y, test_size=0.25, random_state=42, stratify=y
)

pixel_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf')),
])

param_grid = {
    'svc__C': [1, 10, 100, 1000],
    'svc__gamma': [0.001, 0.01, 0.1, 'scale'],
}

grid = GridSearchCV(pixel_pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print(f"Best: {grid.best_params_}")
print(f"Test accuracy: {grid.score(X_test, y_test):.3f}")
```

Expect **~85-95%** test accuracy depending on split and tuning - strong for 40 classes.

---

## Feature Strategy 2: HOG Descriptors

**Histogram of Oriented Gradients (HOG)** captures edge orientation distributions - robust to lighting shifts. Used in pedestrian detection for years; Prosise and OpenCV tutorials pair HOG + SVM routinely.

**Intuition:** Divide image into cells. In each cell, histogram gradient directions (edges). Concatenate histograms → compact [feature](../../GLOSSARY.md#feature) vector emphasizing **shape** over raw intensity.

```python
from skimage.feature import hog
from skimage import exposure

def extract_hog_features(images, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2)):
    """Compute HOG feature vector per face image."""
    features = []
    for img in images:
        fd = hog(
            img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False,
            feature_vector=True,
        )
        features.append(fd)
    return np.array(features)

X_hog = extract_hog_features(images)
print(f"HOG feature shape: {X_hog.shape}")  # ~(400, 1764) for 64×64 default

# Visualize HOG for one face
fd, hog_image = hog(images[0], orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(images[0], cmap='gray'); ax1.set_title('Original')
ax2.imshow(hog_image_rescaled, cmap='gray'); ax2.set_title('HOG visualization')
plt.show()
```

### Train SVM on HOG

```python
X_train_h, X_test_h, y_train, y_test = train_test_split(
    X_hog, y, test_size=0.25, random_state=42, stratify=y
)

hog_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf')),
])

hog_grid = GridSearchCV(hog_pipe, param_grid, cv=5, n_jobs=-1)
hog_grid.fit(X_train_h, y_train)
print(f"HOG test accuracy: {hog_grid.score(X_test_h, y_test):.3f}")
```

HOG often matches or **beats** raw pixels with **fewer dimensions** - better generalization per parameter.

---

## PCA Optional Acceleration

[Chapter 06](../chapter-06-principal-component-analysis/README.md) covers PCA fully. Preview for speed:

```python
from sklearn.decomposition import PCA

pca_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100, whiten=True)),
    ('svc', SVC(kernel='rbf')),
])

pca_grid = GridSearchCV(pca_pipe, param_grid, cv=5, n_jobs=-1)
pca_grid.fit(X_train, y_train)
print(f"PCA+SVM test accuracy: {pca_grid.score(X_test, y_test):.3f}")
```

**Order:** scale → PCA → SVM. Whitening (`whiten=True`) equalizes component variances - helps RBF kernel.

---

## Evaluation: Who Gets Confused?

```python
best_model = hog_grid.best_estimator_
y_pred = best_model.predict(X_test_h)

print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title('Olivetti HOG+SVM Confusion Matrix')
plt.show()
```

**Analysis tasks:**
- Which person IDs have lowest recall?
- Are confusions symmetric (A→B and B→A)?
- Do misclassified images share expression or lighting?

```python
# Show misclassified examples
misclassified = X_test_h[y_pred != y_test]
true_labels = y_test[y_pred != y_test]
pred_labels = y_pred[y_pred != y_test]
print(f"Misclassified: {len(misclassified)} / {len(y_test)}")
```

---

## Linear vs RBF on Faces

```python
from sklearn.model_selection import cross_val_score

for kernel in ['linear', 'rbf']:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel=kernel, C=10, gamma='scale')),
    ])
    scores = cross_val_score(pipe, X_hog, y, cv=5)
    print(f"{kernel:8s} CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

RBF typically wins on faces; linear is faster baseline.

---

## Prosise Ch. 5 Workflow Summary

```
1. Load Olivetti faces (400 images, 40 classes)
2. Extract features: pixels or HOG
3. train_test_split (stratified)
4. Pipeline: StandardScaler → (optional PCA) → SVC
5. GridSearchCV over C and gamma
6. classification_report + confusion matrix
7. Inspect errors - similar-looking subjects
```

This is the template for **any** high-dim small-$n$ vision problem before deep learning.

---

## Comparison to MNIST (Chapter 03)

| | MNIST digits | Olivetti faces |
|---|-------------|----------------|
| Classes | 10 | 40 |
| Features | 784 pixels | 4096 pixels / ~1764 HOG |
| Samples | 70,000 | 400 |
| SVM viability | Subsample needed | Full dataset OK |
| Section | [3.8](../chapter-03-classification-models/section-08-case-study-mnist-digit-recognition.md) | This section |

Faces are **harder per class** (intra-class variance) with **far fewer examples** - tuning $C$ and $\gamma$ matters more.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| No scaling on pixels/HOG | ~random accuracy | `StandardScaler` in pipeline |
| `GridSearchCV` without pipeline | Leaked scaling | Wrap scaler inside |
| HOG on unnormalized uint8 | Wrong gradient scale | Use `images` float in [0,1] from sklearn |
| Expecting 100% on 40 classes | Unrealistic | 90%+ is excellent |
| PCA before scaling | Brightness dominates PCs | Scale first |

---

## Self-Check

1. Why are SVMs a good match for Olivetti faces?  
   *High-dimensional features, small sample size, nonlinear class boundaries - RBF SVM handles this well.*
2. What does HOG capture that raw pixels miss?  
   *Edge orientation histograms - shape/gradient structure, more lighting-invariant.*
3. Why stratify the train/test split?  
   *Each person appears equally in train and test (10 images each).*
4. What hyperparameters matter most?  
   *`C` and `gamma` for RBF - tune jointly via grid search.*

---

## Exercises

1. Beat 90% test accuracy with HOG + tuned RBF SVM. Document best params.
2. Compare pixel vs HOG feature pipelines on identical splits. Which generalizes better?
3. Reduce to 20 people (labels 0-19). How much does accuracy improve? Why?

---

## References

- [Scikit-Learn Olivetti Faces](https://scikit-learn.org/stable/chapters/generated/sklearn.datasets.fetch_olivetti_faces.html)
- [scikit-image HOG](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html)
- Dalal & Triggs, "Histograms of Oriented Gradients for Human Detection"
- Prosise, Ch. 5 - facial recognition walkthrough
- [Section 3.8](../chapter-03-classification-models/section-08-case-study-mnist-digit-recognition.md) - MNIST parallel
- [GLOSSARY.md](../../GLOSSARY.md) - [svm](../../GLOSSARY.md#svm), [classification](../../GLOSSARY.md#classification)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 5.6](./section-06-svr-regression-from-tube-to-production.md) | **Next:** [Section 5.8 - Production Notes](./section-08-svm-production-notes-when-to-deploy-and-onnx-bridge.md)



