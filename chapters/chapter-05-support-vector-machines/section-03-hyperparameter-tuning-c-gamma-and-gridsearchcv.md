# Section 5.3: Hyperparameter Tuning - C, Gamma & GridSearchCV

> **Source:** Prosise, Ch. 5 - tuning SVMs for production accuracy  
> **Prerequisites:** [Section 5.2](./section-02-kernels-and-the-kernel-trick.md) | [Section 2.7](../chapter-02-regression-models/section-07-model-comparison-and-cross-validation.md)  
> **Glossary:** [hyperparameter](../../GLOSSARY.md#hyperparameter) | [cross-validation](../../GLOSSARY.md#cross-validation) | [svm](../../GLOSSARY.md#svm)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## SVMs Don't Train Themselves

Unlike [ordinary least squares](../../GLOSSARY.md#ordinary-least-squares), SVMs expose **[hyperparameters](../../GLOSSARY.md#hyperparameter)** you must choose **before** fitting:

| Parameter | Applies to | Controls |
|-----------|------------|----------|
| `C` | All kernels | Margin violations vs margin width (soft margin) |
| `gamma` | RBF, poly | Influence radius of each training point |
| `kernel` | - | `linear`, `rbf`, `poly` |
| `degree` | `poly` only | Polynomial order |
| `class_weight` | Imbalanced data | Penalty per class |

Bad choices produce garbage - a wide-margin underfit or a memorizing overfit. Prosise's face-recognition example hinges on finding the right $C$ and $\gamma$ via grid search.

> **Humorous analogy:** $C$ and $\gamma$ are the SVM's personality sliders. Low $C$ = chill professor ("close enough"); high $C$ + high $\gamma$ = obsessive tutor who memorizes every typo in the textbook.

> **In plain English:** You cannot eyeball optimal SVM settings. Use [cross-validation](../../GLOSSARY.md#cross-validation) on a grid of candidates and pick the winner.

---

## The C Parameter in Depth

Soft-margin objective ([Section 5.1](./section-01-svm-intuition-maximum-margin-and-support-vectors.md)):

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i
$$
> **Readable form:** minimize (weight magnitude) + C × (total slack for margin violations)

| $C$ | Margin behavior | Training fit | Risk |
|-----|-----------------|--------------|------|
| **0.01** | Very wide, tolerant | Misses outliers | [Underfitting](../../GLOSSARY.md#underfitting) |
| **1.0** | Balanced default | Moderate | Baseline |
| **100** | Tight, few violations | Tracks training closely | [Overfitting](../../GLOSSARY.md#overfitting) |
| **10000** | Near hard margin | Memorizes noise | Severe overfitting |

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = make_classification(n_samples=400, n_features=20, n_informative=10,
                             n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

for C in [0.01, 0.1, 1, 10, 1000]:
    pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=C, gamma='scale'))
    pipe.fit(X_train, y_train)
    print(f"C={C:6g}  train={pipe.score(X_train, y_train):.3f}  test={pipe.score(X_test, y_test):.3f}")
```

Watch train accuracy climb toward 100% as $C$ grows while test accuracy peaks then falls.

---

## The Gamma Parameter (RBF & Polynomial)

For the RBF kernel:

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)
$$
> **Readable form:** RBF similarity drops exponentially as squared distance grows; gamma sets how fast it drops

**Scikit-Learn default `gamma='scale'`:**

$$
\gamma_{\text{scale}} = \frac{1}{n_{\text{features}} \cdot \text{Var}(X)}
$$
> **Readable form:** scale gamma = 1 divided by (number of features times variance of X)

| $\gamma$ | Decision boundary | Support vectors |
|----------|-------------------|-----------------|
| Very small | Almost linear, smooth | Few |
| `scale` | Data-adaptive | Moderate |
| Very large | Jagged, local bumps | Many (nearly all points) |

**Rule of thumb:** Search $\gamma$ on a log scale: `[1e-4, 1e-3, 1e-2, 0.1, 1]` after scaling.

$C$ and $\gamma$ **interact**. High $C$ + high $\gamma$ = extreme overfitting. Tune them **together**, not one at a time.

---

## GridSearchCV: The Engineer's Workflow

From [Section 2.7](../chapter-02-regression-models/section-07-model-comparison-and-cross-validation.md), `GridSearchCV` exhaustively evaluates hyperparameter combinations with k-fold [cross-validation](../../GLOSSARY.md#cross-validation):

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', random_state=42)
)

param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': ['scale', 0.01, 0.1, 1],
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    refit=True,
    verbose=1,
)

grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.3f}")
print(f"Test score: {grid.score(X_test, y_test):.3f}")
```

**Critical:** Put `StandardScaler` **inside** the pipeline so each CV fold fits the scaler on training folds only - no leakage ([Section 5.4](./section-04-normalization-and-pipelining-for-svms.md)).

### Pipeline parameter naming

When the estimator is nested, prefix with the step name + double underscore:

```python
# Pipeline steps: ('scaler', StandardScaler()), ('svc', SVC())
# Parameters: scaler__*, svc__C, svc__gamma, svc__kernel
```

---

## Visualizing the C-Gamma Landscape

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.DataFrame(grid.cv_results_)
pivot = results.pivot_table(
    values='mean_test_score',
    index='param_svc__gamma',
    columns='param_svc__C'
)

plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
plt.title('GridSearchCV: Mean CV Accuracy')
plt.xlabel('$C$'); plt.ylabel('$\\gamma$')
plt.show()
```

Dark cells = high accuracy. Look for **broad plateaus** (stable choices) not sharp peaks (fragile overfit).

---

## Coarse-to-Fine Search Strategy

Prosise recommends a two-stage search for face recognition:

**Stage 1 - coarse log grid:**

```python
coarse_grid = {
    'svc__C': [0.1, 1, 10, 100, 1000],
    'svc__gamma': [1e-4, 1e-3, 1e-2, 0.1, 1],
}
```

**Stage 2 - refine around best:**

```python
# If best was C=10, gamma=0.01:
fine_grid = {
    'svc__C': [3, 10, 30, 100],
    'svc__gamma': [0.005, 0.01, 0.02, 0.05],
}
```

This saves compute vs one giant grid.

---

## Kernel Selection in the Same Grid

Compare kernels fairly in one search:

```python
param_grid = [
    {'svc__kernel': ['linear'], 'svc__C': [0.1, 1, 10, 100]},
    {'svc__kernel': ['rbf'], 'svc__C': [0.1, 1, 10, 100],
     'svc__gamma': ['scale', 0.01, 0.1]},
    {'svc__kernel': ['poly'], 'svc__degree': [2, 3],
     'svc__C': [0.1, 1, 10], 'svc__gamma': ['scale']},
]

grid = GridSearchCV(make_pipeline(StandardScaler(), SVC()), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_)
```

Each dict in the list is a **sub-grid** - only relevant params are evaluated per kernel.

---

## Scoring Metrics Beyond Accuracy

| Problem | `scoring` | Why |
|---------|-----------|-----|
| Balanced classes | `'accuracy'` | Default OK |
| Imbalanced | `'f1'`, `'f1_macro'` | [Section 3.3](../chapter-03-classification-models/section-03-classification-accuracy-measures.md) |
| Fraud / spam | `'average_precision'` | Precision-focused |
| Regression (SVR) | `'neg_mean_absolute_error'` | [Section 5.6](./section-06-svr-regression-from-tube-to-production.md) |

```python
GridSearchCV(pipe, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
```

---

## RandomizedSearchCV for Large Grids

When combinations explode ($5 \times 5 \times 5 = 125$ fits × 5 folds = 625 models):

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_dist = {
    'svc__C': loguniform(1e-2, 1e3),
    'svc__gamma': loguniform(1e-4, 1e0),
}

random_search = RandomizedSearchCV(
    pipe, param_dist, n_iter=30, cv=5, random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)
```

Sample 30 random $(C, \gamma)$ pairs instead of testing all 25+ combinations.

---

## class_weight for Imbalanced Data

```python
SVC(kernel='rbf', class_weight='balanced')  # inversely proportional to class frequency
```

Or manual weights:

```python
SVC(class_weight={0: 1, 1: 10})  # penalize misclassifying class 1 ten times more
```

Include in grid search when classes are skewed.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Tuning on test set | Optimistic test score | Hold out test; tune on train/CV only |
| Scaling outside pipeline | Leaked statistics | `make_pipeline(StandardScaler(), SVC())` |
| Tuning $C$ only, fixing $\gamma$ | Suboptimal RBF | Joint grid over both |
| Linear grid on log-scale params | Misses optimum between decades | Use `[0.01, 0.1, 1, 10, 100]` or `loguniform` |
| Ignoring train-test gap after search | Overfit to CV noise | Check `best_estimator_` on held-out test |
| `n_jobs=1` on 16-core machine | Hours of waiting | `n_jobs=-1` |

---

## Self-Check

1. What happens when $C \to \infty$?  
   *Approaches hard margin - no violations allowed, risk of overfitting on noisy data.*
2. Why tune $C$ and $\gamma$ together?  
   *They interact: same $C$ behaves differently at different $\gamma$ values.*
3. Why must the scaler be inside `GridSearchCV`?  
   *Each CV fold must fit scaler on training portion only.*
4. When use `RandomizedSearchCV` over `GridSearchCV`?  
   *Large or continuous search spaces where exhaustive grid is too expensive.*

---

## Exercises

1. Run a $5 \times 5$ grid on Iris (3-class). Plot heatmap of CV accuracy. Is the best region broad or narrow?
2. Repeat with **unscaled** data in the pipeline (remove `StandardScaler`). How much does best CV score drop?
3. Compare wall-clock time: `GridSearchCV` with `n_jobs=1` vs `n_jobs=-1` on 2000 samples.

---

## References

- [Scikit-Learn GridSearchCV](https://scikit-learn.org/stable/chapters/generated/sklearn.model_selection.GridSearchCV.html)
- [Scikit-Learn: Tuning SVM](https://scikit-learn.org/stable/chapters/svm.html#parameters-of-the-rbf-kernel)
- Prosise, Ch. 5 - grid search on Olivetti faces
- [GLOSSARY.md](../../GLOSSARY.md) - [hyperparameter](../../GLOSSARY.md#hyperparameter), [cross-validation](../../GLOSSARY.md#cross-validation)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 5.2](./section-02-kernels-and-the-kernel-trick.md) | **Next:** [Section 5.4 - Normalization & Pipelining](./section-04-normalization-and-pipelining-for-svms.md)



