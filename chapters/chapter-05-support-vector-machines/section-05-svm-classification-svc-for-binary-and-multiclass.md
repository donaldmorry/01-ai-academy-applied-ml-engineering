# Section 5.5: SVM Classification - SVC for Binary & Multiclass

> **Source:** Prosise, Ch. 5 - `SVC` classifier, multiclass strategies  
> **Prerequisites:** [Section 5.4](./section-04-normalization-and-pipelining-for-svms.md) | [Chapter 03](../chapter-03-classification-models/README.md)  
> **Glossary:** [svm](../../GLOSSARY.md#svm) | [classification](../../GLOSSARY.md#classification) | [accuracy](../../GLOSSARY.md#accuracy)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## SVC: The Workhorse Classifier

`sklearn.svm.SVC` implements the [support vector machine](../../GLOSSARY.md#svm) for [classification](../../GLOSSARY.md#classification). It wraps the LIBSVM solver - fast, battle-tested, and the default choice for small-to-medium datasets with up to tens of thousands of samples.

**Binary classification** finds a hyperplane (or kernel boundary) separating two classes. **Multiclass** extends this via decomposition strategies built into scikit-learn.

> **In plain English:** `SVC` is what you import when you have labeled categories and want maximum-margin classification with optional nonlinear kernels.

---

## Binary Classification Walkthrough

Prosise's spam/ham and fraud examples are binary. The SVM predicts $y \in \{-1, +1\}$ internally; sklearn maps to your `[0, 1]` labels transparently.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = load_breast_cancer()
X, y = data.data, data.target  # binary: malignant vs benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=10, gamma='scale', probability=False)
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
print(classification_report(y_test, y_pred, target_names=data.target_names))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=data.target_names)
plt.title('Breast Cancer - RBF SVC')
plt.show()
```

### Decision function

For binary SVC, `decision_function` returns signed distance to the hyperplane:

$$
f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b \quad \text{(linear case)}
$$
> **Readable form:** score = dot product of weights and features plus bias; sign determines class

```python
scores = clf.decision_function(X_test)
print(f"Score range: [{scores.min():.2f}, {scores.max():.2f}]")
# Positive → class 1, negative → class 0
```

Larger $|f(\mathbf{x})|$ = more confident prediction.

---

## Multiclass: One-vs-Rest (OvR)

With $K > 2$ classes, sklearn trains **$K$ binary classifiers** - each separates one class from all others:

$$
f_k(\mathbf{x}) = \sum_{i \in SV_k} \alpha_i^{(k)} y_i K(\mathbf{x}_i, \mathbf{x}) + b_k
$$
> **Readable form:** each class k has its own SVM score; pick the class with highest score

```python
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

iris_svc = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovr')
)
iris_svc.fit(X_train, y_train)
print(classification_report(y_test, iris_svc.predict(X_test),
                            target_names=iris.target_names))
```

**Default:** `decision_function_shape='ovr'` for multiclass.

| Strategy | sklearn param | How it works | When |
|----------|---------------|--------------|------|
| **One-vs-Rest** | `ovr` | $K$ classifiers | Default, scales OK |
| **One-vs-One** | `ovo` | $K(K-1)/2$ classifiers | Sometimes better on small $K$ |

```python
ovo_svc = SVC(kernel='rbf', decision_function_shape='ovo')
```

For Iris (3 classes), OvO trains 3 classifiers; OvR trains 3. Difference is usually small.

---

## Probability Estimates: `probability=True`

SVMs are not probabilistic by default. Setting `probability=True` enables Platt scaling (sigmoid calibration on CV scores):

```python
svc_prob = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=10, probability=True)
)
svc_prob.fit(X_train, y_train)
proba = svc_prob.predict_proba(X_test)
print(proba[:3])  # rows sum to 1.0
```

**Cost:** 5-fold internal CV during `fit` - **slower training**. Use only when you need `predict_proba` (threshold tuning, ROC curves from [Section 3.3](../chapter-03-classification-models/section-03-classification-accuracy-measures.md)).

---

## LinearSVC: Fast Linear Alternative

For high-dimensional sparse text or large $n$:

```python
from sklearn.svm import LinearSVC

linear_clf = make_pipeline(
    StandardScaler(with_mean=False),  # sparse matrices: don't center
    LinearSVC(C=1.0, max_iter=10000)
)
```

| | `SVC(kernel='linear')` | `LinearSVC` |
|---|------------------------|-------------|
| Solver | LIBSVM | liblinear |
| Large $n$ | Slow | Fast |
| `predict_proba` | Via `probability=True` | No (use `CalibratedClassifierCV`) |
| Sparse input | Possible | Optimized |

[Chapter 04](../chapter-04-text-classification/README.md) often pairs `LinearSVC` or logistic regression with TF-IDF.

---

## MNIST Digits: Multiclass at Scale

From [Section 3.8](../chapter-03-classification-models/section-08-case-study-mnist-digit-recognition.md) - 10 classes, 784 features:

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data.astype(np.float64), mnist.target.astype(int)

# Subsample for SVM speed (full 60k is slow)
idx = np.random.RandomState(42).choice(len(y), 10000, replace=False)
X_sub, y_sub = X[idx], y[idx]

X_train, X_test, y_train, y_test = train_test_split(
    X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub
)

mnist_pipe = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=10, gamma=0.001)  # lower gamma for high-dim
)
mnist_pipe.fit(X_train, y_train)
print(f"MNIST subset accuracy: {mnist_pipe.score(X_test, y_test):.3f}")
```

Expect **~95%+** on 10k subsample with tuned RBF - Prosise's MNIST SVM benchmark.

---

## Hyperparameters Recap for SVC

```python
SVC(
    C=1.0,                    # soft-margin penalty
    kernel='rbf',             # 'linear', 'poly', 'rbf', 'sigmoid'
    gamma='scale',            # RBF/poly: influence radius
    degree=3,                 # poly only
    coef0=0.0,                # poly/sigmoid only
    class_weight=None,        # or 'balanced' or dict
    decision_function_shape='ovr',
    probability=False,
    random_state=42,
)
```

Full tuning workflow: [Section 5.3](./section-03-hyperparameter-tuning-c-gamma-and-gridsearchcv.md). Pipeline pattern: [Section 5.4](./section-04-normalization-and-pipelining-for-svms.md).

---

## SVC vs Other Classifiers

| | Logistic Regression | Random Forest | SVC (RBF) |
|---|---------------------|---------------|-----------|
| Nonlinear boundary | Manual features | Yes | Yes (kernel) |
| Scaling | Recommended | Optional | **Required** |
| `predict_proba` | Native | Native | Needs `probability=True` |
| Training speed (50k rows) | Fast | Moderate | Slow |
| High-dim sparse text | Good | OK | Use `LinearSVC` |

---

## Inspecting Support Vectors

```python
svc = iris_svc.named_steps['svc']
print(f"Support vectors per class: {svc.n_support_}")
print(f"Total support vectors: {svc.support_vectors_.shape[0]} / {len(y_train)}")

# Support vector indices in training set
print(f"First 5 SV indices: {svc.support_[:5]}")
```

Typically 10-40% of training points become support vectors with default $C$.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| `SVC` on 100k+ rows | Hours to train | Subsample, `LinearSVC`, or SGD |
| Multiclass without scaling | Poor OvR boundaries | Pipeline with scaler |
| `probability=True` everywhere | 5× slower fit | Enable only when needed |
| Ignoring `class_weight` on imbalanced data | Majority class wins | `class_weight='balanced'` |
| Wrong label dtype | Obscure errors | Integer labels 0..K-1 |

---

## Self-Check

1. How does multiclass OvR work?  
   *Train K binary SVMs, each class vs rest; predict class with highest decision score.*
2. What does `decision_function` return?  
   *Signed distance to the separating hyperplane (higher = more confident for that class in OvR).*
3. When use `LinearSVC` over `SVC(kernel='linear')`?  
   *Large datasets or sparse high-dimensional features (text).*
4. Why is scaling mandatory for `SVC` with RBF?  
   *RBF kernel uses Euclidean distance; unscaled features distort distances.*

---

## Exercises

1. Train OvR vs OvO on Iris with identical $C$, $\gamma$. Compare accuracy and `n_support_` totals.
2. On breast cancer, plot decision function histogram for each true class.
3. Subsample MNIST to 5k/1k train/test. Grid-search `C` and `gamma`; beat 92% test accuracy.

---

## References

- [Scikit-Learn SVC](https://scikit-learn.org/stable/chapters/generated/sklearn.svm.SVC.html)
- [Scikit-Learn LinearSVC](https://scikit-learn.org/stable/chapters/generated/sklearn.svm.LinearSVC.html)
- [Section 3.8](../chapter-03-classification-models/section-08-case-study-mnist-digit-recognition.md) - MNIST case study
- Prosise, Ch. 5 & Ch. 3 (MNIST)
- [GLOSSARY.md](../../GLOSSARY.md) - [svm](../../GLOSSARY.md#svm), [classification](../../GLOSSARY.md#classification)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 5.4](./section-04-normalization-and-pipelining-for-svms.md) | **Next:** [Section 5.6 - SVR Regression](./section-06-svr-regression-from-tube-to-production.md)
