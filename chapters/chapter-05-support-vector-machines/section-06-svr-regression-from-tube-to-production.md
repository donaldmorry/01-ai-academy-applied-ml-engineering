# Section 5.6: SVR Regression - From ε-Tube to Production

> **Source:** Prosise, Ch. 5 (classification focus) + Ch. 2 (SVR regression); deep dive in [Section 2.5](../chapter-02-regression-models/section-05-support-vector-regression-svr.md)  
> **Prerequisites:** [Section 5.2](./section-02-kernels-and-the-kernel-trick.md) | [Section 2.5](../chapter-02-regression-models/section-05-support-vector-regression-svr.md)  
> **Glossary:** [regression](../../GLOSSARY.md#regression) | [svm](../../GLOSSARY.md#svm) | [kernel-trick](../../GLOSSARY.md#kernel-trick)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Same Family, Different Loss

[Section 2.5](../chapter-02-regression-models/section-05-support-vector-regression-svr.md) introduced **Support Vector Regression (SVR)** in the regression chapter - enough to compare against [gradient boosting](../../GLOSSARY.md#gradient-boosting) on taxi fares. This section **connects** that foundation to the full SVM toolkit you've built in Sections 5.1-5.5.

| | SVC (classification) | SVR (regression) |
|---|------------------------|------------------|
| Goal | Maximize margin between classes | Fit function within ε-tube |
| Loss | Hinge (margin violations) | ε-insensitive |
| Output | Class label | Continuous $\hat{y}$ |
| Key params | `C`, `gamma`, `kernel` | `C`, `epsilon`, `gamma`, `kernel` |
| sklearn class | `SVC` | `SVR` |

> **Humorous analogy:** SVC builds a fence between dog parks and cat cafes. SVR builds a highway lane and says "stay within the shoulders - we only ticket drivers who veer outside ε meters."

> **In plain English:** SVR uses the same [kernel trick](../../GLOSSARY.md#kernel-trick) and support-vector sparsity as classification SVMs, but optimizes for continuous targets with a tolerance tube.

---

## The ε-Insensitive Loss (Review)

Prediction (linear kernel):

$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b
$$
> **Readable form:** predicted value = dot product of weights and features plus bias

Loss for residual $r = y - \hat{y}$:

$$
L_\epsilon(r) = \max(0, |r| - \epsilon)
$$
> **Readable form:** loss = zero if error within epsilon; otherwise absolute error minus epsilon

Training objective:

$$
\min_{\mathbf{w}, b} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} L_\epsilon(y_i - \hat{y}_i)
$$
> **Readable form:** minimize weight magnitude plus C times sum of epsilon-insensitive losses

Points **inside** the tube ($|y_i - \hat{y}_i| \leq \epsilon$) contribute **zero** gradient - only boundary violators become support vectors.

---

## SVR Hyperparameters

| Parameter | Role | Tuning guidance |
|-----------|------|-----------------|
| `C` | Penalty for points outside tube | High = tight fit; low = smooth |
| `epsilon` | Tube half-width (target units) | 0.1 for standardized targets; use domain units for fares |
| `gamma` | RBF influence (same as SVC) | Tune with `GridSearchCV` |
| `kernel` | `linear`, `rbf`, `poly` | `rbf` default for nonlinear |

**`epsilon` in real units:** For taxi fares in USD, $\epsilon = 2.0$ means errors within **2 USD** are ignored. For log-transformed fares, $\epsilon = 0.05$ is a very different tube.

---

## Code: SVR Pipeline (Canonical Pattern)

Always mirror classification - scale first:

```python
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)
X = np.random.uniform(0, 10, (800, 3))
y = np.sin(X[:, 0]) * 2 + X[:, 1] * 0.3 + np.random.normal(0, 0.2, 800)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svr_pipe = make_pipeline(
    StandardScaler(),
    SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale')
)
svr_pipe.fit(X_train, y_train)
y_pred = svr_pipe.predict(X_test)

print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

### Grid search for SVR

```python
param_grid = {
    'svr__C': [0.1, 1, 10, 100],
    'svr__epsilon': [0.01, 0.1, 0.5],
    'svr__gamma': ['scale', 0.1, 1],
}

search = GridSearchCV(
    make_pipeline(StandardScaler(), SVR(kernel='rbf')),
    param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
)
search.fit(X_train, y_train)
print(f"Best: {search.best_params_}, MAE: {-search.best_score_:.4f}")
```

Note `scoring='neg_mean_absolute_error'` - sklearn maximizes score, so MAE is negated.

---

## Link to Section 2.5: Taxi Fares

[Section 2.5](../chapter-02-regression-models/section-05-support-vector-regression-svr.md) places SVR in the regression chapter narrative. Prosise suggests comparing `SVR` against `GradientBoostingRegressor` in [Section 2.8](../chapter-02-regression-models/section-08-taxi-fare-prediction-end-to-end-project.md):

```python
from sklearn.ensemble import GradientBoostingRegressor

# Same features and split as Section 2.8
gbm = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
gbm.fit(X_train, y_train)
# gbm_mae = mean_absolute_error(y_test, gbm.predict(X_test))

# svr_pipe from above on identical data
# Compare MAE in USD - GBM usually wins on tabular fares
```

| Criterion | SVR | GBM |
|-----------|-----|-----|
| Tabular fare RMSE | Often higher | Often lower |
| Training 1M rows | Impractical | Feasible |
| Smooth predictions | Yes | Stairsteps |
| High-dim features (faces) | Excellent | Good |

**Takeaway from Chapter 02:** SVR is a **specialist**, not the default tabular regressor. Chapter 05 explains **why** it still matters for kernelized high-dimensional problems.

---

## NuSVR: Alternative Formulation

`NuSVR` reparameterizes with $\nu \in (0, 1]$ controlling the fraction of support vectors:

```python
from sklearn.svm import NuSVR

nu_svr = make_pipeline(StandardScaler(), NuSVR(nu=0.5, C=1.0, kernel='rbf'))
nu_svr.fit(X_train, y_train)
```

Less common in practice than `SVR` with `epsilon` - but useful when you want explicit control over support vector fraction.

---

## Visualizing the ε-Tube

```python
import matplotlib.pyplot as plt

X_1d = np.sort(X_train[:, 0].reshape(-1, 1), axis=0)
y_1d_train = y_train[np.argsort(X_train[:, 0])]

svr_1d = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, epsilon=0.3))
svr_1d.fit(X_train[:, 0].reshape(-1, 1), y_train)

X_plot = np.linspace(X[:, 0].min(), X[:, 0].max(), 200).reshape(-1, 1)
y_plot = svr_1d.predict(X_plot)
eps = 0.3

plt.scatter(X_train[:, 0], y_train, alpha=0.4, label='Training')
plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='SVR prediction')
plt.fill_between(X_plot.ravel(), y_plot - eps, y_plot + eps, alpha=0.2, color='red', label='ε-tube')
plt.legend()
plt.title('SVR ε-Tube (1D projection)')
plt.show()
```

---

## SVR vs Linear Regression vs Random Forest

| | OLS | SVR (RBF) | Random Forest |
|---|-----|-----------|---------------|
| Outliers | Sensitive (squared loss) | ε-tube dampens | Robust |
| Nonlinearity | Feature engineering | [Kernel](../../GLOSSARY.md#kernel-trick) | Automatic |
| Scaling | Recommended | **Required** | Optional |
| Interpretability | Coefficients | Opaque | Feature importance |
| Speed (n=100k) | Fast | Slow | Moderate |

---

## When Chapter 05 SVR Beats Chapter 02 Baselines

From Prosise's perspective, keep SVR in your toolkit when:

1. **Medium $n$, high $p$** - face HOG features ([Section 5.7](./section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md))
2. **Smooth function** required - physical sensors, calibrated scores
3. **Kernel alignment** - you already tuned RBF SVC on related data; reuse $\gamma$ priors
4. **Outlier-heavy** continuous targets - ε-tube reduces leverage of extremes vs [MSE](../../GLOSSARY.md#mse)

Avoid SVR when:

- Millions of rows ([Section 5.8](./section-08-svm-production-notes-when-to-deploy-and-onnx-bridge.md))
- Need fast iteration in notebooks
- Tabular with complex interactions - [gradient boosting](../../GLOSSARY.md#gradient-boosting) wins

---

## Cross-Chapter Study Path

```
Chapter 02, Section 2.5  →  ε-tube intuition, taxi fare teaser
         ↓
Chapter 05, Sections 5.1-5.5  →  kernels, C, gamma, pipelines, SVC
         ↓
Chapter 05, Section 5.6 (this)  →  SVR with full SVM machinery
         ↓
Lab 05  →  classification capstone (faces); optional SVR extension
```

Re-read [Section 2.5](../chapter-02-regression-models/section-05-support-vector-regression-svr.md) after this section - the kernel math should click harder.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Unscaled SVR | Wild MAE | `StandardScaler` in pipeline |
| `epsilon` in wrong units | Tube too wide/narrow | Match target scale |
| Expecting SVR to beat GBM on fares | Disappointment | Use right tool per dataset |
| `SVR` on 500k rows | Overnight training | Subsample or linear models |
| Confusing `C` with `epsilon` | `C` = penalty; `epsilon` = tube width | Tune both via grid |

---

## Self-Check

1. How does SVR loss differ from SVC hinge loss?  
   *SVR uses ε-insensitive loss on continuous residuals; SVC penalizes margin violations for class labels.*
2. What does `epsilon` control in target units?  
   *Half-width of the no-penalty tube around predictions.*
3. Why revisit SVR after learning kernels?  
   *Same kernel machinery applies; tuning C/gamma transfers directly.*
4. When did Prosise suggest GBM over SVR?  
   *Taxi fare tabular regression - trees capture feature interactions better.*

---

## Exercises

1. Re-run [Section 2.5](../chapter-02-regression-models/section-05-support-vector-regression-svr.md) code with `GridSearchCV` from [Section 5.3](./section-03-hyperparameter-tuning-c-gamma-and-gridsearchcv.md). Did MAE improve?
2. Compare `SVR(kernel='linear')` vs `SVR(kernel='rbf')` on the synthetic sinusoid above.
3. Optional: load taxi fare features from Section 2.8. Report MAE for SVR vs GBM on identical split.

---

## References

- [Section 2.5 - Support Vector Regression](../chapter-02-regression-models/section-05-support-vector-regression-svr.md) - **primary regression treatment**
- [Scikit-Learn SVR](https://scikit-learn.org/stable/chapters/generated/sklearn.svm.SVR.html)
- [Section 5.2](./section-02-kernels-and-the-kernel-trick.md) - kernels
- Prosise, Ch. 2 & Ch. 5
- [GLOSSARY.md](../../GLOSSARY.md) - [regression](../../GLOSSARY.md#regression), [svm](../../GLOSSARY.md#svm)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 5.5](./section-05-svm-classification-svc-for-binary-and-multiclass.md) | **Next:** [Section 5.7 - Facial Recognition](./section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md)



