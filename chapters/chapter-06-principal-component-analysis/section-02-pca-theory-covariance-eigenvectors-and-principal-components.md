# Section 6.2: PCA Theory - Covariance, Eigenvectors, and Principal Components

> **Source:** Prosise, Ch. 6 - "Understanding Principal Component Analysis"  
> **Glossary:** [pca](../../GLOSSARY.md#pca) | [eigenvector](../../GLOSSARY.md#eigenvector) | [explained-variance](../../GLOSSARY.md#explained-variance)  
> **Math:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Geometric Story

[PCA](../../GLOSSARY.md#pca) answers one question: **In what direction does the data vary the most?** That direction becomes the **first principal component**. The second component is the direction of maximum *remaining* variance, forced to be **orthogonal** (perpendicular) to the first. The third is orthogonal to both, and so on.

Prosise's Figures 6-1 through 6-3 show a 2D scatter cloud with two arrows - the longer arrow is the primary component; the shorter is secondary. Reducing to 1D means projecting every point onto the long arrow. You lose the "width" of the cloud but keep its "length."

> **Analogy:** PCA is like finding the best angle to photograph a tilted ellipse so it looks as wide as possible in the frame. The camera angle that maximizes apparent width is PC1. Rotate 90° for PC2.

---

## Step-by-Step Mathematics

Given a centered data matrix $\mathbf{X} \in \mathbb{R}^{n \times m}$ ($n$ samples, $m$ features), PCA proceeds as follows.

### Step 1: Center the Data

Subtract the mean of each column:

$$
\tilde{\mathbf{X}} = \mathbf{X} - \mathbf{1}\boldsymbol{\mu}^\top
$$
> **Readable form:** each feature column has its mean subtracted so the cloud sits at the origin.

`sklearn.decomposition.PCA` does this automatically when `with_mean=True` (default).

### Step 2: Covariance Matrix

The sample covariance matrix $\mathbf{C}$ captures how each feature varies with every other:

$$
\mathbf{C} = \frac{1}{n-1} \tilde{\mathbf{X}}^\top \tilde{\mathbf{X}}
$$
> **Readable form:** covariance matrix = scaled product of centered data transpose times centered data - entry $(i,j)$ measures how feature $i$ and feature $j$ move together.

Diagonal entries are variances; off-diagonals are covariances. Highly correlated features produce large off-diagonal values - PCA will merge that shared story into one component.

### Step 3: Eigendecomposition

Solve:

$$
\mathbf{C} \mathbf{v}_k = \lambda_k \mathbf{v}_k
$$
> **Readable form:** covariance matrix times eigenvector $k$ equals eigenvalue $k$ times the same eigenvector.

- $\mathbf{v}_k$ - the $k$-th **[eigenvector](../../GLOSSARY.md#eigenvector)** (principal axis direction)
- $\lambda_k$ - the corresponding eigenvalue (variance along that axis)

Eigenvectors are sorted by decreasing $\lambda_k$. The first eigenvector points along PC1.

### Step 4: Project

Transformed coordinates:

$$
\mathbf{Z} = \tilde{\mathbf{X}} \mathbf{V}_k
$$
> **Readable form:** projected data = centered data multiplied by the matrix of top-$k$ eigenvectors.

Each row of $\mathbf{Z}$ is a sample in the new low-dimensional space.

---

## Explained Variance

The **[explained variance](../../GLOSSARY.md#explained-variance)** of component $k$:

$$
\text{EVR}_k = \frac{\lambda_k}{\sum_{j=1}^{m} \lambda_j}
$$
> **Readable form:** explained variance ratio for component $k$ = that component's eigenvalue divided by the sum of all eigenvalues.

Prosise's LFW example: PC1 explains ~18%, PC2 ~15%, and the first 150 components sum to **94.8%** of total variance - hence faces stay recognizable after compression.

---

## SVD: What Scikit-Learn Actually Computes

You rarely call `np.linalg.eig` on $\mathbf{C}$ directly. Numerically stable PCA uses **Singular Value Decomposition (SVD)**:

$$
\tilde{\mathbf{X}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top
$$
> **Readable form:** centered data = U times singular values times V-transpose.

Singular values squared (normalized) give eigenvalues of the covariance matrix. `PCA` wraps this - you get the same subspace without building a huge $\mathbf{C}$ when $m$ is large.

---

## Worked Micro-Example (2 Features)

```python
import numpy as np
from sklearn.decomposition import PCA

# Tiny correlated dataset
X = np.array([
    [2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2],
    [3.1, 3.0], [2.3, 2.7], [2.0, 1.6], [1.0, 1.1],
    [1.5, 1.6], [1.1, 0.9]
])

pca = PCA(n_components=2)
Z = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Principal axes (components_):\n", pca.components_)
print("Means (center):", pca.mean_)
```

**Interpretation:**
- `components_[0]` - direction of PC1 in original feature space
- `explained_variance_ratio_[0]` - fraction of total variance along PC1
- `mean_` - column means subtracted during fitting

---

## Reconstruction and Information Loss

Projection to $k$ dimensions, then mapping back:

$$
\hat{\mathbf{X}} = \mathbf{Z} \mathbf{V}_k^\top + \mathbf{1}\boldsymbol{\mu}^\top
$$
> **Readable form:** reconstructed data = low-D coordinates times eigenvector matrix, plus means added back.

Discarded components are gone forever. `inverse_transform` returns the **best rank-$k$ approximation** in mean-squared-error sense - not the original $\mathbf{X}$ unless $k = m$.

Prosise's LFW demo: 2,914 → 150 → 2,914 dimensions. Faces blur slightly; identities remain. That blur is the lost 5.2% variance.

---

## Properties You Should Know

| Property | Meaning |
|----------|---------|
| **Orthogonal components** | $\text{corr}(\text{PC}_i, \text{PC}_j) = 0$ for $i \neq j$ |
| **Ordered variance** | $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_m$ |
| **Linear combinations** | Each PC is a weighted sum of original features |
| **Unsupervised** | Labels are ignored - variance, not class separation, is maximized |
| **Unique up to sign** | $\mathbf{v}$ and $-\mathbf{v}$ are the same axis |

---

## PCA vs Other Matrix Factorizations

| Method | Relation |
|--------|----------|
| **Eigendecomposition of $\mathbf{C}$** | Classical textbook PCA |
| **SVD of $\tilde{\mathbf{X}}$** | What sklearn uses |
| **NMF** | Nonnegative factors - parts-based, not rotation |
| **Autoencoder** | Nonlinear PCA - Course 3 |

---

## Whitening (Preview)

`PCA(whiten=True)` scales each component to unit variance after rotation:

$$
z_k^{\text{white}} = \frac{z_k}{\sqrt{\lambda_k}}
$$
> **Readable form:** whitened component $k$ = projected value divided by the square root of its eigenvalue.

Useful before some classifiers; can amplify noise in near-zero eigenvalue directions. Covered in [Section 6.3](./section-03-pca-with-scikit-learn.md).

---

## Humor: The Eigenvector Support Group

*"I don't know where I'm pointing, but $\mathbf{C}\mathbf{v} = \lambda\mathbf{v}$, so I'm clearly important."* - Every first principal component at a party where correlated features complain about redundancy.

---

## Common Mistakes

| Mistake | Reality |
|---------|---------|
| "Eigenvector = feature" | Eigenvector is a **direction** - a recipe mixing all features |
| "More components = always better" | Diminishing returns; tail components are often noise |
| Forgetting centering | First PC points toward the mean offset, not true spread |
| Confusing $\mathbf{V}$ with $\mathbf{Z}$ | `components_` are axes; `transform` output is coordinates |

---

## Self-Check

1. What matrix does PCA eigendecompose (conceptually)?  
2. Why are principal components uncorrelated?  
3. If PC1 explains 60% and PC2 explains 25%, how much remains for PC3 onward?  
4. What does `inverse_transform` *not* recover?

---

## Exercises

1. Manually compute the 2×2 covariance of the micro-example above. Compare eigenvalues to `pca.explained_variance_`.
2. Fit PCA on `load_iris()` with 2 components. Plot `components_` as arrows over a scatter of sepal length vs width.
3. Show that reconstruction error $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$ equals the sum of variances of discarded components (for centered data).

---

## References

- [Scikit-Learn PCA API](https://scikit-learn.org/stable/chapters/generated/sklearn.decomposition.PCA.html)
- [Jaadi - Step-by-Step PCA](https://towardsdatascience.com/a-step-by-step-explanation-of-principal-component-analysis-f73627259f8d)
- Prosise, *Applied ML and AI for Engineers*, Ch. 6
- [Gilbert Strang - SVD lecture (MIT OCW)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- [GLOSSARY.md](../../GLOSSARY.md) - [pca](../../GLOSSARY.md#pca), [eigenvector](../../GLOSSARY.md#eigenvector), [explained-variance](../../GLOSSARY.md#explained-variance)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 6.1 - Curse of Dimensionality](./section-01-the-curse-of-dimensionality.md)  
**Next:** [Section 6.3 - PCA with Scikit-Learn](./section-03-pca-with-scikit-learn.md)
