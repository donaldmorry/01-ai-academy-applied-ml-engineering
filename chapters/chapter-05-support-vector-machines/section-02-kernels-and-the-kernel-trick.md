# Section 5.2: Kernels & the Kernel Trick

> **Source:** Prosise, Ch. 5 - kernel functions, nonlinear SVMs  
> **Prerequisites:** [Section 5.1](./section-01-svm-intuition-maximum-margin-and-support-vectors.md) | [Section 2.5](../chapter-02-regression-models/section-05-support-vector-regression-svr.md)  
> **Glossary:** [kernel-trick](../../GLOSSARY.md#kernel-trick) | [svm](../../GLOSSARY.md#svm) | [feature](../../GLOSSARY.md#feature)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## When a Straight Line Fails

The maximum-margin story in [Section 5.1](./section-01-svm-intuition-maximum-margin-and-support-vectors.md) assumes a **linear** hyperplane works. But many datasets are not linearly separable in the original [features](../../GLOSSARY.md#feature):

- XOR pattern in 2D
- Circular clusters (points inside vs outside a ring)
- Sinusoidal decision boundaries in regression ([Section 2.5](../chapter-02-regression-models/section-05-support-vector-regression-svr.md))

**Naive fix:** manually engineer polynomial terms - $x_1^2$, $x_1 x_2$, $x_2^2$ - then run linear SVM in the expanded space.

**Problem:** For degree-$d$ polynomials in $p$ features, the expanded space has $\binom{p+d}{d}$ dimensions. With 784 MNIST pixels and degree 3, you cannot materialize that matrix.

The **[kernel trick](../../GLOSSARY.md#kernel-trick)** computes dot products in the high-dimensional space **without ever building it**.

> **Humorous analogy:** You can't list every spice combination in the universe, but you can *taste* whether two dishes use similar blends. Kernels taste similarity without opening the pantry.

> **In plain English:** A kernel function $K(\mathbf{x}_i, \mathbf{x}_j)$ measures how alike two examples are in a hidden high-dimensional space. The SVM only needs those pairwise similarities - not the expanded coordinates.

---

## The Math: From Dots to Kernels

In the dual formulation, the SVM decision function depends on inner products $\mathbf{x}_i^T \mathbf{x}_j$. Suppose we map inputs to $\phi(\mathbf{x})$ in a higher-dimensional space. The decision function uses:

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)
$$
> **Readable form:** kernel value = dot product of the two mapped feature vectors

**Mercer's condition:** A function $K$ is a valid kernel if the Gram matrix $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ is positive semi-definite for any finite set of points. Scikit-Learn's built-in kernels satisfy this.

The prediction becomes:

$$
f(\mathbf{x}) = \sum_{i \in SV} \alpha_i y_i \, K(\mathbf{x}_i, \mathbf{x}) + b
$$
> **Readable form:** prediction = weighted sum of kernel similarities to support vectors, plus bias

Same support-vector sparsity - only a subset of training rows contribute.

---

## The Three Kernels You'll Use Daily

### 1. Linear Kernel

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j
$$
> **Readable form:** linear kernel = ordinary dot product of the two feature vectors

**When to use:**
- High-dimensional sparse data (text TF-IDF from [Chapter 04](../chapter-04-text-classification/README.md))
- Already-separable data in original space
- Millions of features where RBF is too slow

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=50, random_state=42)
linear_svc = SVC(kernel='linear', C=1.0)
linear_svc.fit(X, y)
```

**Prosise note:** Linear SVM often matches or beats logistic regression on text - with the same scaling requirements.

### 2. RBF (Gaussian) Kernel - The Default for Tabular & Images

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)
$$
> **Readable form:** RBF kernel = exponential of negative gamma times squared Euclidean distance

- $\gamma > 0$ controls how far each training point's influence reaches
- $\gamma$ large → each point is a tiny bump → wiggly boundary ([overfitting](../../GLOSSARY.md#overfitting))
- $\gamma$ small → smooth boundary spanning many points ([underfitting](../../GLOSSARY.md#underfitting))

| $\gamma$ | Effect | Analogy |
|----------|--------|---------|
| **High** | Local bumps around each SV | Flashlight beam - only nearby points matter |
| **Low** | Broad smooth hills | Floodlight - distant points still influence |
| **`scale`** (sklearn default) | $\gamma = 1 / (n_{\text{features}} \cdot \text{Var}(X))$ | Auto-adjusts to feature count |

```python
rbf_svc = SVC(kernel='rbf', C=10.0, gamma='scale')
rbf_svc.fit(X, y)
print(f"Support vectors: {rbf_svc.n_support_.sum()} / {len(y)}")
```

### 3. Polynomial Kernel

$$
K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \, \mathbf{x}_i^T \mathbf{x}_j + r)^d
$$
> **Readable form:** polynomial kernel = (gamma times dot product plus constant) raised to degree d

| Parameter | Meaning | Typical value |
|-----------|---------|---------------|
| `degree` ($d$) | Polynomial order | 2 or 3 (rarely > 3) |
| `gamma` | Scale of dot product | `'scale'` |
| `coef0` ($r$) | Constant term | 0 or 1 |

```python
poly_svc = SVC(kernel='poly', degree=3, gamma='scale', coef0=1)
poly_svc.fit(X, y)
```

**When to use:** Problems where feature interactions are known to be polynomial (some physics/engineering signals). In practice, **RBF beats poly** on most tabular benchmarks - but Prosise includes poly for completeness.

---

## Kernel Comparison on Nonlinear Data

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

kernels = {
    'linear': SVC(kernel='linear', C=1.0),
    'rbf': SVC(kernel='rbf', C=10.0, gamma='scale'),
    'poly (d=3)': SVC(kernel='poly', degree=3, gamma='scale', coef0=1),
}

for name, clf in kernels.items():
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f"{name:12s} accuracy: {acc:.3f}")
```

Expected pattern on moons: **linear fails (~85%)**, **RBF and poly succeed (~97%+)**.

---

## Visualizing Decision Boundaries

```python
def plot_decision_boundary(pipe, X, y, title):
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = pipe.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    plt.title(title)
    plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
    plt.show()

rbf_pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale'))
rbf_pipe.fit(X_train, y_train)
plot_decision_boundary(rbf_pipe, X_test, y_test, 'RBF SVM on Moons')
```

The RBF boundary curves smoothly around the crescent shapes - impossible for a single linear hyperplane in 2D.

---

## Custom Kernels (Advanced)

Scikit-Learn accepts a callable kernel:

```python
def my_kernel(X, Y):
    """Precomputed dot product squared - equivalent to poly degree 2 with specific params."""
    return np.dot(X, Y.T) ** 2

# svc = SVC(kernel=my_kernel)
```

Or pass a **precomputed** Gram matrix for very small datasets:

```python
from sklearn.metrics.pairwise import rbf_kernel

K_train = rbf_kernel(X_train, X_train, gamma=0.5)
svc_pre = SVC(kernel='precomputed', C=1.0)
svc_pre.fit(K_train, y_train)

K_test = rbf_kernel(X_test, X_train, gamma=0.5)
y_pred = svc_pre.predict(K_test)
```

Useful when computing kernels is expensive but reusable (e.g., string kernels). For 99% of engineering work, stick to `linear`, `rbf`, `poly`.

---

## Kernel Selection Cheat Sheet

| Data type | First choice | Second choice | Avoid |
|-----------|-------------|---------------|-------|
| Text (TF-IDF, sparse) | `linear` | - | `rbf` on raw sparse (slow) |
| Tabular, < 50k rows | `rbf` | `linear` if high-dim | `poly` degree > 3 |
| Image features (HOG, pixels) | `rbf` | `linear` after PCA | Untuned `gamma` |
| Already linear trend | `linear` | - | High-$\gamma$ RBF |
| MNIST / digits | `rbf` | `linear` + PCA | k-NN at scale |

Tuning $C$ and $\gamma$: [Section 5.3](./section-03-hyperparameter-tuning-c-gamma-and-gridsearchcv.md).

---

## Connection to SVR (Regression)

The same kernels power **Support Vector Regression** in [Section 2.5](../chapter-02-regression-models/section-05-support-vector-regression-svr.md) and [Section 5.6](./section-06-svr-regression-from-tube-to-production.md). Classification maximizes margin between classes; regression fits an $\epsilon$-tube. The kernel machinery is identical:

$$
K_{\text{RBF}}(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)
$$
> **Readable form:** The RBF kernel gives nearby vectors high similarity and drives the score toward zero as distance grows.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| RBF on unscaled features | One feature dominates distance | `StandardScaler` in pipeline ([Section 5.4](./section-04-normalization-and-pipelining-for-svms.md)) |
| `gamma` too high | 100% train, poor test | Reduce $\gamma$ or $C$; use `GridSearchCV` |
| Polynomial degree 5+ | Numerical blow-up, slow | Cap at $d \leq 3$ |
| Linear kernel on moons/circles | Stuck at ~85% accuracy | Switch to `rbf` |
| Custom kernel not PSD | Solver errors or nonsense | Verify Mercer condition |

---

## Self-Check

1. What does the kernel trick avoid?  
   *Explicitly constructing high-dimensional feature maps - only pairwise similarities are needed.*
2. Why is RBF the default for nonlinear tabular data?  
   *Flexible smooth boundaries, only two extra hyperparameters ($C$, $\gamma$), works well with scaling.*
3. When is linear kernel preferred over RBF?  
   *High-dimensional sparse text, very large $n \times p$ where RBF Gram matrix is prohibitive.*
4. What does $\gamma$ control in the RBF kernel?  
   *Inverse width of influence - higher gamma means more local, complex boundaries.*

---

## Exercises

1. On `make_circles(noise=0.1)`, compare linear vs RBF. Plot boundaries side by side.
2. Fix $C=1$ and sweep $\gamma \in [0.01, 0.1, 1, 10, 100]$ on moons. Plot train vs test accuracy.
3. Train `SVC(kernel='poly', degree=2)` and `degree=5` on the same data. Which overfits?

---

## References

- [Scikit-Learn: Kernel Methods](https://scikit-learn.org/stable/chapters/metrics.html#kernels)
- [Scikit-Learn SVC](https://scikit-learn.org/stable/chapters/generated/sklearn.svm.SVC.html)
- Schölkopf & Smola, *Learning with Kernels*
- Prosise, Ch. 5 - kernel selection for face recognition
- [GLOSSARY.md](../../GLOSSARY.md) - [kernel-trick](../../GLOSSARY.md#kernel-trick), [svm](../../GLOSSARY.md#svm)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 5.1](./section-01-svm-intuition-maximum-margin-and-support-vectors.md) | **Next:** [Section 5.3 - Hyperparameter Tuning](./section-03-hyperparameter-tuning-c-gamma-and-gridsearchcv.md)
