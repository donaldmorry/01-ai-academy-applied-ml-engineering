# Section 6.1: The Curse of Dimensionality

> **Source:** Prosise, Ch. 6 - "Understanding Principal Component Analysis" (motivation)  
> **Glossary:** [pca](../../GLOSSARY.md#pca) | [feature](../../GLOSSARY.md#feature) | [overfitting](../../GLOSSARY.md#overfitting)  
> **Math:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Why Engineers Care About Dimensions

Imagine a spreadsheet where every row is a sensor reading and every column is a measurement channel. A grayscale face image? Each pixel is a column. A vibration snapshot from four accelerometers sampled at 1 kHz? Hundreds of columns per second. A credit card transaction with 29 engineered features? Still "high" when fraud is one row in 284,000.

**[Principal Component Analysis (PCA)](../../GLOSSARY.md#pca)** is Prosise's "best-kept secret in machine learning" - but before we open that toolbox, we need to understand *why* high dimensions hurt in the first place. PCA is the antidote; the curse of dimensionality is the disease.

> **Analogy:** A 1,000-column dataset is like a warehouse with 1,000 doors. Most doors lead to the same room (redundant information). A few doors open to unique rooms (signal). PCA finds the shortest tour that visits the rooms that actually matter - and boards up the rest.

---

## What "High-Dimensional" Means

There is no universal cutoff, but engineers routinely hit pain when:

| Situation | Example | Why it hurts |
|-----------|---------|--------------|
| **Features ≫ intuition** | 2,914 pixels per face (LFW) | Cannot plot, hard to debug |
| **Features ≈ samples** | 30 features, 569 patients (breast cancer) | Models become unstable |
| **Correlated columns** | Temperature sensors on the same engine block | Redundant math, inflated variance |
| **Sparse coverage** | 1,000 features, 200 rows | Empty corners of feature space |

Prosise offers a practical rule of thumb: aim for **at least five times as many rows as columns** for reliable supervised learning. If you cannot collect more rows, **reduce columns** - PCA is one way to do that without throwing away most of the signal.

---

## The Curse, Stated Plainly

As dimensionality grows, several unpleasant things happen - even before you pick an algorithm:

### 1. Distance Concentration

In high dimensions, the distance between *any two random points* becomes surprisingly similar. [k-NN](../chapter-01-machine-learning/section-05-supervised-learning-k-nearest-neighbors.md) relies on "near" and "far." When everything is medium-far, neighbors stop being meaningful.

### 2. Volume Explodes

A unit hypercube in $d$ dimensions has volume 1. A hypersphere inscribed inside it has volume that **shrinks toward zero** as $d$ increases. Data that looks dense in 2D becomes needle-thin in 100D - most of the space is empty.

$$
V_{\text{sphere}} = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)} r^d
$$
> **Readable form:** hypersphere volume grows with dimension $d$ in the exponent - but compared to the enclosing cube, the sphere occupies a vanishing fraction of the space.

**Humor break:** In 100 dimensions, your training data is the equivalent of a few sesame seeds floating in a stadium. Good luck finding a nearest neighbor.

### 3. Redundancy and Correlation

Real sensors do not vary independently. Bearing 1 vibration rises when Bearing 2 does. Adjacent pixels in a face photo are nearly identical. Extra columns add **noise in the optimization** without adding **information in the physics**.

PCA's first move is to rotate the coordinate system so new axes are **uncorrelated** - eliminating duplicate stories the data tells.

### 4. Overfitting Risk

With many [features](../../GLOSSARY.md#feature) and few samples, models can memorize:

$$
\text{parameters to learn} \approx \text{features} \times \text{model complexity}
$$
> **Readable form:** more knobs than data points → the model twists knobs to fit noise → [overfitting](../../GLOSSARY.md#overfitting).

Reducing dimensions before [SVM](../chapter-05-support-vector-machines/README.md) or logistic regression is not laziness - it is regularization by geometry.

---

## Dropping Columns vs Smart Reduction

Prosise's opening 2D example (Figures 6-1 through 6-3) makes the point viscerally. Suppose each point has coordinates $(x, y)$:

| Naive approach | What you lose |
|----------------|---------------|
| Drop $x$, keep $y$ | A vertical smear - structure gone |
| Drop $y$, keep $x$ | A horizontal smear - structure gone |
| **[PCA](../../GLOSSARY.md#pca) to 1D** | Project onto the axis of **maximum spread** - retains >95% of variance in his example |

The bad approaches amputate an entire measurement. PCA asks: *which single direction best preserves the cloud's shape?*

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

# Correlated 2D blob - stand-in for Prosise's Figure 6-1
X, _ = make_blobs(n_samples=200, centers=1, cluster_std=1.5, random_state=0)
X = X @ np.array([[1.0, 0.8], [0.2, 1.2]])  # shear → correlation

# Naive: keep only column 0
naive_1d = X[:, [0]]

# Smart: PCA to 1 component
pca = PCA(n_components=1)
pca_1d = pca.fit_transform(X)

print(f"Variance retained by PCA: {pca.explained_variance_ratio_[0]:.1%}")
```

---

## When PCA Helps (Preview of the Chapter)

Prosise lists engineering wins that motivate the whole chapter:

1. **Visualization** - 1,000 columns → 2 or 3 for plotting
2. **Noise filtering** - discard low-variance directions where noise hides
3. **Anonymization** - rotate data so original feature meanings are obscured
4. **Anomaly detection** - abnormal rows reconstruct poorly after compression
5. **Sample-to-feature ratio** - 1,000 columns → 100 while keeping ~90% information

You will implement each of these before [Chapter 07](../chapter-07-operationalizing-models/README.md).

---

## PCA Is Linear - Know the Limits

PCA finds **straight-line** combinations of features. If your data lies on a curved Swiss-roll manifold in 3D, PCA may mash layers together. Nonlinear alternatives (kernel PCA, t-SNE, UMAP - Course 2) exist for that reason. For many sensor, image, and tabular problems, linear PCA is shockingly effective - Prosise's face and fraud examples are proof.

---

## Connection to Prior Chapters

| Prior topic | Link to dimensionality |
|-------------|------------------------|
| [Chapter 05 - SVM](../chapter-05-support-vector-machines/README.md) | High-D kernel spaces need normalized, sometimes reduced inputs |
| [Facial recognition](../chapter-05-support-vector-machines/section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md) | 4,096 HOG features - PCA preview appeared there |
| [Credit card fraud](../chapter-03-classification-models/README.md) | 29 anonymized features - PCA can anonymize further |
| [k-means](../chapter-01-machine-learning/section-04-unsupervised-learning-k-means-clustering.md) | Clustering in 1,000D is slow and distance-weird |

---

## Practical Checklist Before PCA

1. **Ask:** Do I have too many columns, correlated columns, or a plotting problem?
2. **Scale** if units differ (PCA is variance-based - dollars dominate cents otherwise). Use `StandardScaler` in a [Pipeline](../chapter-05-support-vector-machines/section-04-normalization-and-pipelining-for-svms.md).
3. **Center** data - `PCA` centers by default (`with_mean=True`).
4. **Do not** apply PCA to the full dataset before cross-validation if PCA is part of model selection - fit PCA inside each fold (leakage section from Chapter 05).

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| PCA on raw mixed units | First component = "richest currency" | `StandardScaler` first |
| Assuming PCA always helps prediction | Worse test accuracy | PCA is unsupervised - it maximizes variance, not label separation |
| Confusing rows and columns | Weird shapes after transform | PCA reduces **features** (columns), not samples (rows) |
| Using PCA for causal claims | "Component 3 causes failure" | Components are mathematical mixtures, not physical sensors |

---

## Self-Check

1. State Prosise's rows-to-columns rule of thumb in your own words.  
2. Why is dropping a single raw column worse than PCA projection in correlated data?  
3. Name two problems from the curse of dimensionality besides overfitting.  
4. Why must features be scaled before PCA when units differ?

---

## Exercises

1. Generate `make_classification(n_features=50, n_informative=5)` with 500 samples. Train logistic regression with and without `PCA(n_components=5)`. Compare test accuracy and training time.
2. Plot pairwise correlations of 10 features from `load_breast_cancer()`. How many feature pairs have $|r| > 0.8$?
3. For a 100×100 random grayscale image (10,000 features, 1 sample), explain why PCA is awkward. What would you do instead for compression?

---

## Sample-to-Feature Ratio (Prosise's Rule)

Prosise recommends at least **five rows per column** for stable supervised learning. When you cannot collect more samples, reduce features instead of hoping the model "figures it out."

$$
\frac{n_{\text{samples}}}{n_{\text{features}}} \geq 5
$$
> **Readable form:** sample-to-feature ratio should be at least 5 to 1.

Example: 200 patients × 30 features → ratio ≈ 6.7 (acceptable). 200 patients × 1,000 genes → ratio 0.2 (danger zone). PCA can shrink 1,000 → 40 while retaining ~90% variance, improving the effective ratio without new lab work.

```python
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
ratio = X.shape[0] / X.shape[1]
print(f"Samples: {X.shape[0]}, features: {X.shape[1]}, ratio: {ratio:.1f}")
```

---

## Distance Concentration Demo

```python
import numpy as np
from sklearn.metrics import pairwise_distances

def mean_pairwise_distance(n_samples, n_dims, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_dims))
    d = pairwise_distances(X)
    # exclude diagonal
    return d[np.triu_indices(n_samples, k=1)].mean()

for d in [2, 10, 100, 500]:
    print(f"dim={d:3d}  mean pairwise dist ≈ {mean_pairwise_distance(100, d):.3f}")
```

As dimension grows, distances concentrate - [k-NN](../chapter-01-machine-learning/section-05-supervised-learning-k-nearest-neighbors.md) neighbors become less distinctive.

---

## References

- [Scikit-Learn: PCA](https://scikit-learn.org/stable/chapters/decomposition.html#pca)
- [Zakaria Jaadi - Step-by-Step PCA](https://towardsdatascience.com/a-step-by-step-explanation-of-principal-component-analysis-f73627259f8d)
- Prosise, *Applied ML and AI for Engineers*, Ch. 6 (intro)
- [StatQuest: PCA](https://www.youtube.com/watch?v=FgakZw6K1QQ)
- [GLOSSARY.md](../../GLOSSARY.md) - [pca](../../GLOSSARY.md#pca), [feature](../../GLOSSARY.md#feature)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Next:** [Section 6.2 - PCA Theory](./section-02-pca-theory-covariance-eigenvectors-and-principal-components.md)



