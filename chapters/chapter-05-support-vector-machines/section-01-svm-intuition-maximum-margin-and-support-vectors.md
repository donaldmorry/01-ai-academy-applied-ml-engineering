# Section 5.1: SVM Intuition - Maximum Margin & Support Vectors

> **Source:** Prosise, Ch. 5 - "Support Vector Machines" / geometric intuition  
> **Prerequisites:** [Chapter 03 - Classification](../chapter-03-classification-models/README.md) | [Section 1.5](../chapter-01-machine-learning/section-05-supervised-learning-k-nearest-neighbors.md)  
> **Glossary:** [svm](../../GLOSSARY.md#svm) | [classification](../../GLOSSARY.md#classification) | [hyperparameter](../../GLOSSARY.md#hyperparameter)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Problem: Drawing the Best Line

Imagine two groups of points on a scatter plot - spam vs ham emails measured by two numeric features, or two Iris species plotted by petal length and width. Many lines could separate them. Which one should you pick?

[Logistic regression](../chapter-03-classification-models/section-02-logistic-regression.md) finds *a* boundary by maximizing likelihood. Decision trees carve axis-aligned rectangles. **[Support Vector Machines (SVMs)](../../GLOSSARY.md#svm)** ask a different question:

> *Which separating hyperplane leaves the **widest possible gap** between classes?*

That gap is the **margin**. The points that touch the margin - the closest training examples to the boundary - are **support vectors**. They alone define the decision surface.

> **Humorous analogy:** Picture two rival pizza shops on a street. The SVM draws the property line exactly midway between the closest customers of each shop - not where most people stand, but where the **tightest squeeze** happens. Move one loyal customer, and the whole border might shift.

> **In plain English:** SVMs care about the hardest examples near the boundary, not the easy points deep inside each cluster.

---

## Hyperplanes in Feature Space

In 2D, the decision boundary is a line. In $p$ dimensions, it is a **hyperplane**:

$$
\mathbf{w}^T \mathbf{x} + b = 0
$$
> **Readable form:** hyperplane = all points where (weight vector dot feature vector) + bias equals zero

- $\mathbf{w}$ - normal vector perpendicular to the hyperplane (which direction it faces)
- $b$ - bias (how far the plane sits from the origin)
- $\mathbf{x}$ - a feature vector for one example

**Classification rule** for binary labels $y \in \{-1, +1\}$:

$$
f(\mathbf{x}) = \text{sign}(\mathbf{w}^T \mathbf{x} + b)
$$
> **Readable form:** predicted class = positive if (w dot x + b) is positive, negative otherwise

Points on the positive side get class +1; negative side get −1.

---

## The Margin: Why Width Matters

Given a separating hyperplane, compute the perpendicular distance from any point $\mathbf{x}_i$ to the plane:

$$
d_i = \frac{|y_i(\mathbf{w}^T \mathbf{x}_i + b)|}{\|\mathbf{w}\|}
$$
> **Readable form:** distance = absolute value of (label times prediction) divided by length of weight vector

The **margin** is the minimum distance among all training points:

$$
\text{margin} = \min_i \frac{y_i(\mathbf{w}^T \mathbf{x}_i + b)}{\|\mathbf{w}\|}
$$
> **Readable form:** margin = smallest signed distance from any labeled point to the hyperplane

**Hard-margin SVM** (perfectly separable data) solves:

$$
\min_{\mathbf{w}, b} \quad \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \quad \forall i
$$
> **Readable form:** minimize weight magnitude, subject to every point being on the correct side at least one margin-width away

Minimizing $\|\mathbf{w}\|^2$ **maximizes** the margin - a classic convex optimization problem with a unique solution (when data is separable).

| Concept | Role |
|---------|------|
| Hyperplane | Decision boundary |
| Margin | Width of the "no man's land" between classes |
| Support vectors | Training points with $y_i(\mathbf{w}^T \mathbf{x}_i + b) = 1$ - on the margin |
| $\mathbf{w}$, $b$ | Learned parameters |

---

## Support Vectors: The VIP List

After training, most training rows are irrelevant for prediction. Only **support vectors** matter - typically a small fraction of the dataset.

**Real-life analogy:** A university admissions committee might review thousands of applications, but the borderline cases (GPA 3.49 vs 3.51) determine where the cutoff lands. Everyone far above or below is unchanged by a small cutoff shift.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=80, centers=2, random_state=42, cluster_std=1.2)
y_svm = np.where(y == 0, -1, 1)  # SVM convention: labels -1 and +1

clf = svm.SVC(kernel='linear', C=1000)  # high C ≈ hard margin
clf.fit(X, y_svm)

print(f"Total training points: {len(y_svm)}")
print(f"Support vectors: {clf.n_support_}")
print(f"Support vector indices (first 5): {clf.support_[:5]}")
```

**Prediction** for a new point uses only support vectors:

$$
f(\mathbf{x}) = \sum_{i \in SV} \alpha_i y_i \, K(\mathbf{x}_i, \mathbf{x}) + b
$$
> **Readable form:** prediction = sum over support vectors of (alpha × label × kernel similarity to that vector) + bias

For a linear kernel, this collapses to $\mathbf{w}^T \mathbf{x} + b$ - but the support-vector form generalizes to kernels in [Section 5.2](./section-02-kernels-and-the-kernel-trick.md).

---

## Visualizing the Margin

```python
def plot_svm_margin(clf, X, y):
  """Plot 2D linear SVM with margin lines."""
  ax = plt.gca()
  ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=60)
  ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
             s=200, facecolors='none', edgecolors='lime', linewidths=2,
             label='Support vectors')

  xlim = ax.get_xlim()
  xx = np.linspace(xlim[0], xlim[1], 100)
  w = clf.coef_[0]
  b = clf.intercept_[0]
  slope = -w[0] / w[1]
  intercept = -b / w[1]
  ax.plot(xx, slope * xx + intercept, 'k-', linewidth=2, label='Decision boundary')

  # Margin boundaries: w·x + b = ±1
  margin = 1 / np.sqrt(np.sum(w ** 2))
  ax.plot(xx, slope * xx + intercept + margin / np.cos(np.arctan(slope)), 'k--', alpha=0.5)
  ax.plot(xx, slope * xx + intercept - margin / np.cos(np.arctan(slope)), 'k--', alpha=0.5)
  ax.legend()
  plt.title('Linear SVM: Maximum Margin')
  plt.show()

plot_svm_margin(clf, X, y_svm)
```

The dashed lines show where $y_i(\mathbf{w}^T \mathbf{x}_i + b) = \pm 1$ - support vectors sit on them.

---

## Hard Margin vs Soft Margin (Preview)

Real data is **noisy**. Outliers and overlap mean no hyperplane separates all points perfectly. **Soft-margin SVM** introduces slack variables $\xi_i \geq 0$ allowing misclassification:

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i
$$

$$
\text{subject to} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i
$$
> **Readable form:** minimize (weight magnitude) + C × (sum of slack penalties), allowing some points inside the margin or on the wrong side

The **[hyperparameter](../../GLOSSARY.md#hyperparameter)** $C$ controls the tradeoff:

| $C$ | Behavior |
|-----|----------|
| **Large** | Narrow margin tolerated only with high penalty → fits training data tightly, risk of [overfitting](../../GLOSSARY.md#overfitting) |
| **Small** | Allows more slack → wider effective margin, smoother boundary, risk of [underfitting](../../GLOSSARY.md#underfitting) |

Full treatment of $C$, $\gamma$, and grid search: [Section 5.3](./section-03-hyperparameter-tuning-c-gamma-and-gridsearchcv.md).

---

## SVM vs Other Classifiers (Geometric View)

| Algorithm | What it optimizes | Boundary shape |
|-----------|-------------------|----------------|
| [Logistic regression](../chapter-03-classification-models/section-02-logistic-regression.md) | Log-likelihood | Linear (logistic curve in 1D) |
| [k-NN](../chapter-01-machine-learning/section-05-supervised-learning-k-nearest-neighbors.md) | Local majority vote | Flexible, jagged |
| [Decision tree](../chapter-03-classification-models/section-05-tree-based-classifiers.md) | Information gain | Axis-aligned steps |
| **SVM** | Maximum margin | Linear or kernel-shaped |

SVMs shine when you believe a **wide margin** between classes is meaningful - common in high-dimensional text and image feature spaces.

---

## Prosise Ch. 5: Why SVMs After Trees and Text

Prosise places SVMs in Part I after classification fundamentals because:

1. **Engineers need kernel methods** before deep learning takes over vision tasks
2. **Face recognition** (Olivetti faces, [Section 5.7](./section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md)) is a compact capstone with clear accuracy gains from RBF kernels
3. **Pipelines + scaling** discipline transfers directly to every distance-sensitive algorithm

His narrative: start with geometric intuition (this section), add kernels for nonlinear data, then deploy on real image features.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Expecting SVM to find *any* separating line | Same as logistic regression | Remember: SVM maximizes **margin**, not probability |
| Ignoring mislabeled outliers with hard margin | Wild hyperplane tilt | Use soft margin ($C < \infty$) |
| Training on unscaled features | Margin dominated by large-scale columns | [Section 5.4](./section-04-normalization-and-pipelining-for-svms.md) |
| Assuming all points matter equally | Confusion about model size | Only support vectors define $f(\mathbf{x})$ |
| Using SVM on 5M rows without sampling | Hours of training | Subsample or switch algorithms ([Section 5.8](./section-08-svm-production-notes-when-to-deploy-and-onnx-bridge.md)) |

---

## Self-Check

1. What is the margin in one sentence?  
   *The perpendicular distance between the decision hyperplane and the nearest training points of each class.*
2. Why are they called support vectors?  
   *They "support" or define the hyperplane - removing a non-support vector doesn't change the model.*
3. What does increasing $C$ do?  
   *Penalizes margin violations more heavily → tighter fit to training labels.*
4. How is SVM different from k-NN geometrically?  
   *SVM finds a global optimal hyperplane; k-NN uses local neighborhoods with no explicit margin.*

---

## Exercises

1. Generate `make_blobs` with `cluster_std=2.5` so classes overlap. Plot linear SVM with $C \in \{0.01, 1, 1000\}$. How do support vector counts change?
2. Identify support vectors manually: for a fitted `SVC`, verify `clf.support_vectors_` matches rows where dual coefficients $\alpha_i > 0$.
3. Compare test accuracy of linear SVM vs [k-NN](../chapter-01-machine-learning/section-05-supervised-learning-k-nearest-neighbors.md) on the Iris dataset (2 features only). Which has fewer "active" training points at prediction time?

---

## References

- [Scikit-Learn SVM Guide](https://scikit-learn.org/stable/chapters/svm.html)
- [LIBSVM: A Library for Support Vector Machines](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- Prosise, *Applied ML and AI for Engineers*, Ch. 5
- [StatQuest: SVM](https://www.youtube.com/watch?v=efR1C6CvhmE)
- [GLOSSARY.md](../../GLOSSARY.md) - [svm](../../GLOSSARY.md#svm), [classification](../../GLOSSARY.md#classification)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Next:** [Section 5.2 - Kernels & the Kernel Trick](./section-02-kernels-and-the-kernel-trick.md)
