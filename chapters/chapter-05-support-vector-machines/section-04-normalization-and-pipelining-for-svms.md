# Section 5.4: Normalization & Pipelining for SVMs

> **Source:** Prosise, Ch. 5 - feature scaling before SVM; sklearn `Pipeline`  
> **Prerequisites:** [Section 5.3](./section-03-hyperparameter-tuning-c-gamma-and-gridsearchcv.md) | [Section 4.8](../chapter-04-text-classification/section-08-pipelines-and-production-text-ml.md)  
> **Glossary:** [feature](../../GLOSSARY.md#feature) | [svm](../../GLOSSARY.md#svm) | [hyperparameter](../../GLOSSARY.md#hyperparameter)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Why SVMs Demand Scaled Features

[SVMs](../../GLOSSARY.md#svm) depend on **distances** and **dot products**. The RBF kernel:

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)
$$
> **Readable form:** RBF similarity uses squared Euclidean distance between feature vectors

If one [feature](../../GLOSSARY.md#feature) spans 0-10,000 (income in USD) and another spans 0-1 (age in decades), distance is dominated by income. The SVM margin tilts toward the large-scale column - even if age is more predictive.

The linear term $\mathbf{w}^T \mathbf{x}$ has the same problem: unpenalized large-magnitude features get small weights but still dominate the product.

> **Humorous analogy:** Measuring "closeness" between people using height (meters) and bank balance (dollars) without scaling is like saying two people are similar because their net worth differs by only 2 meters.

> **In plain English:** Always scale features before SVM. No exceptions for RBF or polynomial kernels. Linear SVM on scaled data is strongly recommended.

---

## StandardScaler: Zero Mean, Unit Variance

**Most common choice for SVM:**

$$
x'_j = \frac{x_j - \mu_j}{\sigma_j}
$$
> **Readable form:** scaled feature = (original minus mean) divided by standard deviation

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# WRONG: fit on all data
# scaler = StandardScaler().fit(X)
# X_train_s = scaler.transform(X_train)  # leakage if fit on full X

# RIGHT: pipeline handles per-fold fitting
svm_pipe = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=10, gamma='scale')
)
svm_pipe.fit(X_train, y_train)
print(f"Scaled SVM accuracy: {svm_pipe.score(X_test, y_test):.3f}")
```

### Demonstrating the scaling impact

```python
unscaled = SVC(kernel='rbf', C=10, gamma='scale')
unscaled.fit(X_train, y_train)
scaled_pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale'))
scaled_pipe.fit(X_train, y_train)

print(f"Unscaled: {unscaled.score(X_test, y_test):.3f}")
print(f"Scaled:   {scaled_pipe.score(X_test, y_test):.3f}")
```

On Wine, unscaled accuracy often drops **10-20 percentage points**.

---

## MinMaxScaler: Bounded [0, 1]

$$
x'_j = \frac{x_j - \min_j}{\max_j - \min_j}
$$
> **Readable form:** scaled feature = (original minus minimum) divided by (maximum minus minimum)

| Scaler | Output range | Best for |
|--------|-------------|----------|
| `StandardScaler` | Unbounded, ~N(0,1) | SVM default, Gaussian kernels |
| `MinMaxScaler` | [0, 1] | Image pixels 0-255, bounded sensors |
| `RobustScaler` | Median/IQR based | Heavy outliers (less common for SVM) |

```python
from sklearn.preprocessing import MinMaxScaler

# MNIST-style pixels (Chapter 03)
pixel_pipe = make_pipeline(
    MinMaxScaler(),
    SVC(kernel='rbf', C=5, gamma='scale')
)
```

Prosise scales Olivetti face pixels before SVM in Ch. 5. Either `MinMaxScaler` or `StandardScaler` works on grayscale images - **consistency** matters more than the exact choice.

---

## make_pipeline vs Pipeline

```python
from sklearn.pipeline import Pipeline, make_pipeline

# Explicit names - needed for ColumnTransformer routing
pipe_a = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf')),
])

# Shorthand - auto-generates step names 'standardscaler', 'svc'
pipe_b = make_pipeline(StandardScaler(), SVC(kernel='rbf'))

# Access steps
print(pipe_b.named_steps['svc'].C)  # after fit, inspect fitted estimator
```

For SVM-only workflows, `make_pipeline` is sufficient. Use named `Pipeline` when combining multiple column types ([Section 3.4](../chapter-03-classification-models/section-04-categorical-data-and-feature-encoding.md)).

---

## Preventing Data Leakage

The leakage trap from [Section 4.8](../chapter-04-text-classification/section-08-pipelines-and-production-text-ml.md) applies equally to numeric scaling:

```
WRONG ORDER:
1. StandardScaler.fit(all_data)
2. train_test_split
3. SVC.fit → test metrics are optimistic

CORRECT ORDER:
1. train_test_split
2. Pipeline.fit(train) → scaler learns μ, σ from train only
3. Pipeline.score(test) → scaler transforms test with train statistics
```

With [cross-validation](../../GLOSSARY.md#cross-validation):

```python
from sklearn.model_selection import cross_val_score

pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

Each fold: scaler fits on 4/5 of training data, SVM trains on scaled 4/5, evaluates on scaled held-out 1/5.

---

## Full Pipeline with GridSearchCV

From [Section 5.3](./section-03-hyperparameter-tuning-c-gamma-and-gridsearchcv.md) - the production pattern:

```python
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', random_state=42)
)

param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': ['scale', 0.01, 0.1],
}

grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"Test: {best_model.score(X_test, y_test):.3f}")
```

**No separate scaler object to forget.** `best_model` includes fitted scaler + SVM.

---

## Persisting the Pipeline

```python
import joblib

joblib.dump(best_model, 'svm_wine_pipeline.joblib')

loaded = joblib.load('svm_wine_pipeline.joblib')
# Production: loaded.predict(new_samples) - scaling happens automatically
```

Same pattern as text pipelines in [Lab 04](../chapter-04-text-classification/section-lab-04-sentiment-spam-and-recommendations.md).

---

## PCA + Scaler + SVM (Face Recognition Preview)

[Section 5.7](./section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md) chains dimensionality reduction:

```python
from sklearn.decomposition import PCA

face_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=150, whiten=True)),
    ('svc', SVC(kernel='rbf', C=1000, gamma='scale')),
])
```

**Order matters:**
1. Scale pixels (or HOG features)
2. PCA on scaled data
3. SVM on PCA components

Never PCA before scaling on image pixels - high-variance bright pixels dominate components.

---

## ColumnTransformer for Mixed Data

When some columns are numeric (scale) and others categorical (encode):

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

numeric_features = ['age', 'income']
categorical_features = ['city']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
])

full_pipe = Pipeline([
    ('prep', preprocessor),
    ('svc', SVC(kernel='rbf')),
])
```

SVM still requires all inputs on comparable scales **after** preprocessing.

---

## Inspection: What the Scaler Learned

```python
scaler = svm_pipe.named_steps['standardscaler']
print("Means:", scaler.mean_[:5])
print("Stds:", scaler.scale_[:5])

# Verify transformed training data ~ N(0,1)
X_train_scaled = scaler.transform(X_train)
print(pd.DataFrame(X_train_scaled).describe().loc[['mean', 'std']])
```

Means near 0, stds near 1 on training data confirm correct fitting.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| No scaling with RBF | Random-level accuracy | `StandardScaler` in pipeline |
| `fit_transform` on full dataset | Leaked test statistics | Pipeline + proper split |
| Saving SVC without scaler | Production predictions wrong | `joblib.dump(pipeline)` |
| Different scaler at inference | Train/serve skew | Single persisted pipeline |
| Scaling labels | Nonsense regression | Scale **features** only |
| PCA before scaling on images | Components chase brightness | Scale first |

---

## Self-Check

1. Why does RBF SVM fail on unscaled mixed-magnitude features?  
   *Distance is dominated by large-scale features, distorting the kernel.*
2. What does `make_pipeline` guarantee during CV?  
   *Preprocessor refits on each training fold only.*
3. When prefer `MinMaxScaler` over `StandardScaler`?  
   *Bounded inputs like pixels 0-255 where you want a fixed range.*
4. Why save the whole pipeline, not just `SVC`?  
   *Inference must apply the same scaling learned during training.*

---

## Exercises

1. Load Iris. Compare `StandardScaler` vs `MinMaxScaler` vs no scaler with RBF SVM. Tabulate test accuracy.
2. Deliberately leak: fit scaler on full X, then split. Compare "leaked" test score vs pipeline score.
3. Build a 3-step pipeline: `StandardScaler` → `PCA(n_components=10)` → `SVC`. Tune only `svc__C` with `GridSearchCV`.

---

## References

- [Scikit-Learn Pipeline](https://scikit-learn.org/stable/chapters/compose.html)
- [Scikit-Learn StandardScaler](https://scikit-learn.org/stable/chapters/generated/sklearn.preprocessing.StandardScaler.html)
- [Section 4.8](../chapter-04-text-classification/section-08-pipelines-and-production-text-ml.md) - pipeline patterns
- Prosise, Ch. 5 - scaling face features
- [GLOSSARY.md](../../GLOSSARY.md) - [feature](../../GLOSSARY.md#feature), [svm](../../GLOSSARY.md#svm)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 5.3](./section-03-hyperparameter-tuning-c-gamma-and-gridsearchcv.md) | **Next:** [Section 5.5 - SVM Classification](./section-05-svm-classification-svc-for-binary-and-multiclass.md)



