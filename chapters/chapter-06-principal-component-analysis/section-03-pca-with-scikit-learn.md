# Section 6.3: PCA with Scikit-Learn

> **Source:** Prosise, Ch. 6 - `PCA`, `fit_transform`, `explained_variance_ratio_`  
> **Glossary:** [pca](../../GLOSSARY.md#pca) | [explained-variance](../../GLOSSARY.md#explained-variance)  
> **Math:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Good News (Prosise's Words)

You do not have to implement eigendecomposition by hand. `sklearn.decomposition.PCA` does the numerical gymnastics. Your job: **choose $k$**, **fit on the right data**, and **interpret** `explained_variance_ratio_`.

> **Humor:** PCA is the machine learning equivalent of a Michelin kitchen - you plate the results; sklearn sweats over the linear algebra in the back.

---

## Minimal Workflow

Prosise's canonical pattern:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=5, random_state=0)
X_reduced = pca.fit_transform(X)
```

| Call | When |
|------|------|
| `fit(X)` | Learn means, subspace from training data |
| `transform(X)` | Project new data into learned subspace |
| `fit_transform(X)` | Fit + project in one step |
| `inverse_transform(Z)` | Reconstruct approximate original |

**Rule:** `fit` on training only. `transform` validation and test with the **same** `pca` object.

---

## Key Attributes After `fit`

```python
pca = PCA(n_components=150, random_state=0)
Z = pca.fit_transform(faces.data)  # LFW: 2914 → 150

print(pca.n_components_)           # 150
print(pca.explained_variance_ratio_[:5])
print(pca.explained_variance_ratio_.sum())  # ~0.948 for LFW
print(pca.components_.shape)       # (150, 2914) - each row is a PC axis
print(pca.mean_.shape)             # (2914,) - column means
```

| Attribute | Meaning |
|-----------|---------|
| `explained_variance_ratio_` | Fraction of total variance per component |
| `explained_variance_` | Absolute variance (not normalized) |
| `components_` | Principal axes in feature space |
| `singular_values_` | SVD singular values |
| `n_components_` | Actual count (may differ from constructor - see below) |
| `noise_variance_` | Minka estimate when `n_components='mle'` |

---

## Three Ways to Set `n_components`

### 1. Integer - fixed count

```python
PCA(n_components=26)  # Prosise fraud example: 29 → 26
```

### 2. Float in $(0, 1)$ - variance threshold

```python
PCA(0.8, random_state=0)  # keep enough PCs to explain 80% variance
pca.fit(noisy_faces)
print(pca.n_components_)  # sklearn picks count - 179 for noisy LFW
```

> **Readable form:** pass 0.8 → retain the smallest number of components whose cumulative explained variance ≥ 80%.

### 3. `None` - keep all components

```python
PCA(n_components=None)  # all min(n_samples, n_features) components
```

Useful for scree plots ([Section 6.4](./section-04-choosing-the-number-of-components.md)) and anonymization ([Section 6.6](./section-06-anonymization-and-privacy-with-pca.md)).

---

## LFW Face Compression (Prosise Reproduction)

```python
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

faces = fetch_lfw_people(min_faces_per_person=100)
print(faces.data.shape)  # (n_people, 2914)

pca = PCA(n_components=150, random_state=0)
Z = pca.fit_transform(faces.data)
X_hat = pca.inverse_transform(Z).reshape(-1, 62, 47)

fig, axes = plt.subplots(3, 8, figsize=(18, 10))
for ax, img, i in zip(axes.flat, X_hat, range(24)):
    ax.imshow(img, cmap='gist_gray')
    ax.set(xticks=[], yticks=[],
           xlabel=faces.target_names[faces.target[i]])
plt.tight_layout()
```

**Takeaway:** 95% of dimensions removed; faces still recognizable. Blur = discarded variance.

---

## Pipelines with StandardScaler

PCA is sensitive to scale. When features use different units, scale first:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=50)),
    ('clf', SVC(kernel='rbf'))
])
pipe.fit(X_train, y_train)
```

This mirrors [Chapter 05](../chapter-05-support-vector-machines/section-04-normalization-and-pipelining-for-svms.md) sections - **never** fit `StandardScaler` on the full dataset before splitting.

---

## Whitening

```python
PCA(n_components=20, whiten=True)
```

After rotation, divides each component by $\sqrt{\lambda_k}$ so variances become 1. Decorrelates *and* equalizes scale - sometimes helps distance-based models. **Caution:** components with tiny eigenvalues get amplified (noise boost).

---

## `random_state` - When It Matters

`PCA` uses randomized SVD when `svd_solver='randomized'` (default for large matrices with low `n_components`). Set `random_state` for reproducibility. Full `svd_solver='full'` is deterministic but slower on wide matrices.

```python
PCA(n_components=100, svd_solver='full')      # exact, slower
PCA(n_components=100, svd_solver='randomized', random_state=0)  # fast
```

---

## Incremental and Sparse Variants

| Class | Use when |
|-------|----------|
| `PCA` | Fits in memory |
| `IncrementalPCA` | Batch/streaming data |
| `SparsePCA` | Sparse loadings (interpretability) |
| `KernelPCA` | Nonlinear boundaries (Course 2 preview) |

```python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=50, batch_size=100)
for batch in np.array_split(X_big, 10):
    ipca.partial_fit(batch)
Z = ipca.transform(X_big)
```

---

## Transform vs Inverse - Shapes

If $\mathbf{X}$ has shape `(n_samples, n_features)`:

$$
\mathbf{Z} = \text{transform}(\mathbf{X}) \quad \Rightarrow \quad \text{shape } (n_{\text{samples}}, k)
$$
> **Readable form:** The equation projects data onto directions that preserve the most variance in a lower-dimensional representation.

$$
\hat{\mathbf{X}} = \text{inverse\_transform}(\mathbf{Z}) \quad \Rightarrow \quad \text{shape } (n_{\text{samples}}, n_{\text{features}})
$$
> **Readable form:** The equation projects data onto directions that preserve the most variance in a lower-dimensional representation.

Reconstruction error per sample (foundation for [Section 6.8](./section-08-case-study-bearing-failure-and-credit-card-fraud-detection.md)):

$$
\text{MSE}_i = \frac{1}{m}\sum_{j=1}^{m}(x_{ij} - \hat{x}_{ij})^2
$$
> **Readable form:** per-sample reconstruction error = mean squared difference between original and PCA-reconstructed features.

```python
import numpy as np

def reconstruction_mse(X, pca):
    Z = pca.transform(X)
    X_hat = pca.inverse_transform(Z)
    return np.mean((X - X_hat) ** 2, axis=1)
```

---

## Prosise's One-Liner Summary

```python
pca = PCA(n_components=5)
x = pca.fit_transform(x)
# ...
x = pca.inverse_transform(x)  # approximate restore - not identical
```

That inverse is the engine behind noise filtering, fraud detection, and bearing monitoring in this chapter.

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Fitting PCA on train+test together | Fit on train only |
| Ignoring `n_components_` when using float | Print it - sklearn chooses count |
| Expecting `inverse_transform` to be exact | Only exact when keeping all components |
| PCA on categorical one-hot without scaling | Scale or consider other methods |

---

## Self-Check

1. What does `explained_variance_ratio_.sum()` tell you after fitting with `n_components=150`?  
2. Difference between `n_components=0.8` and `n_components=80`?  
3. Why put `StandardScaler` before `PCA` in a pipeline?  
4. What shape is `pca.components_` for 150 components on 2914 features?

---

## Exercises

1. Reproduce Prosise's LFW grid: original vs 50 vs 150 vs 500 components. Plot cumulative variance vs visual quality.
2. Wrap `PCA(0.95)` + `LogisticRegression` in a `Pipeline` on `load_digits()`. Compare accuracy to raw pixels.
3. Time `svd_solver='full'` vs `'randomized'` on a wide random matrix (1000 features, 5000 samples).

---

## References

- [Scikit-Learn: PCA](https://scikit-learn.org/stable/chapters/generated/sklearn.decomposition.PCA.html)
- [Scikit-Learn: Decomposition user guide](https://scikit-learn.org/stable/chapters/decomposition.html)
- Prosise, *Applied ML and AI for Engineers*, Ch. 6
- [GLOSSARY.md](../../GLOSSARY.md) - [pca](../../GLOSSARY.md#pca), [explained-variance](../../GLOSSARY.md#explained-variance)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 6.2 - PCA Theory](./section-02-pca-theory-covariance-eigenvectors-and-principal-components.md)  
**Next:** [Section 6.4 - Choosing Components](./section-04-choosing-the-number-of-components.md)



