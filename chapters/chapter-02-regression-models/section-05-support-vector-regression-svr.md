# Section 2.5: Support Vector Regression (SVR)

> **Source:** Prosise, Ch. 2 - Support Vector Machines (regression context); full SVM in [Chapter 05](../chapter-05-support-vector-machines/README.md)  
> **Prerequisites:** [Section 2.4](./section-04-ensemble-methods-random-forests-and-gradient-boosting.md) | [Glossary: regression](../../GLOSSARY.md#regression)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## A Tube Instead of a Line

[Linear regression](../../GLOSSARY.md#linear-regression) penalizes every error via [MSE](../../GLOSSARY.md#mse). **Support Vector Regression (SVR)** says: *"Errors inside an ε-tube don't count. Only worry about points outside."*

The prediction function (linear kernel) is:

$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b
$$
> **Readable form:** predicted value = (dot product of weight vector and feature vector) + bias

### ε-insensitive loss

For each point, define the residual $r_i = y_i - \hat{y}_i$. SVR uses:

$$
L_\epsilon(r) = \max(0, |r| - \epsilon)
$$
> **Readable form:** loss = zero if error is within epsilon; otherwise error minus epsilon

Points with $|y_i - \hat{y}_i| \leq \epsilon$ contribute **zero** loss - they're "close enough."

**Real-life analogy:** Grading on a curve with mercy - if you're within ±2 points of the target score, you get full credit. Only way-off answers hurt your grade.

> **In plain English:** SVR draws a prediction line (or curved surface with kernels) and wraps a slack zone around it. Points inside the zone are "close enough." The model focuses energy on the hard cases at the margins - like SVM classification in [Chapter 05](../chapter-05-support-vector-machines/README.md) focusing on support vectors.

---

## **Milestone: Sparse Solutions from Support Vectors**

Only training points **on or outside** the ε-tube affect the final model. At prediction time, you don't need the entire dataset - just the **support vectors**.

| Algorithm | What gets stored after training |
|-----------|--------------------------------|
| [Linear regression](../../GLOSSARY.md#linear-regression) | All weights (uses every row indirectly) |
| [Random forest](../../GLOSSARY.md#random-forest) | Hundreds of trees |
| SVR | Support vectors + kernel params |

This sparsity is elegant - but on million-row taxi CSVs, SVR training can be **slow** compared to tree ensembles.

---

## The C Parameter: Tube Width vs Violations

Training balances:

$$
\min_{\mathbf{w}, b} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} L_\epsilon(y_i - \hat{y}_i)
$$
> **Readable form:** minimize (weight magnitude) + C × (sum of epsilon-insensitive losses)

- **Higher $C$** → narrower tube tolerance for violations → tighter fit, risk of [overfitting](../../GLOSSARY.md#overfitting)
- **Lower $C$** → wider slack → smoother model, risk of [underfitting](../../GLOSSARY.md#underfitting)

| Param | Role |
|-------|------|
| `C` | Tradeoff: wider tube vs fewer violations (higher = tighter fit) |
| `epsilon` | Width of insensitive tube (in target units - USD for fares) |
| `gamma` (RBF) | Influence radius of single training example |

> **Humorous analogy:** `C` is how picky your GPS is - low C = "eh, close enough"; high C = "RECALCULATING" at every block.

---

## Kernels: Straight Lines in Curved Worlds

Linear SVR fails on circular or sinusoidal data. The **kernel trick** maps features to higher dimensions where a linear separator works - without you building the huge feature matrix explicitly.

| Kernel | When to use |
|--------|-------------|
| `linear` | High-dimensional sparse text (later chapters) |
| `rbf` (Gaussian) | Default for nonlinear tabular data |
| `poly` | Polynomial relationships (degree rarely > 3) |

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2) \quad \text{(RBF)}
$$
> **Readable form:** RBF kernel = exponential of negative gamma times squared distance between two feature vectors

> **In plain English:** RBF kernel asks "how similar is this new trip to each training trip?" Nearby trips influence the prediction more.

---

## Code: Always Pipeline with Scaling

SVR is **sensitive to feature scale**. Distance in miles (0-30) vs passenger count (1-6) will dominate unless normalized.

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

np.random.seed(42)
X = np.random.uniform(0, 10, (500, 2))
y = np.sin(X[:, 0]) * 3 + X[:, 1] * 0.5 + np.random.normal(0, 0.3, 500)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVR REQUIRES scaling - always pipeline it
svr_rbf = make_pipeline(
    StandardScaler(),
    SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale')
)
svr_rbf.fit(X_train, y_train)
print(f"SVR (RBF) MAE: {mean_absolute_error(y_test, svr_rbf.predict(X_test)):.3f}")

svr_linear = make_pipeline(
    StandardScaler(),
    SVR(kernel='linear', C=1.0, epsilon=0.1)
)
svr_linear.fit(X_train, y_train)
print(f"SVR (linear) MAE: {mean_absolute_error(y_test, svr_linear.predict(X_test)):.3f}")
```

### Quick grid search for SVR

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'svr__C': [0.1, 1, 10],
    'svr__epsilon': [0.01, 0.1, 0.5],
    'svr__gamma': ['scale', 0.1],
}

search = GridSearchCV(
    make_pipeline(StandardScaler(), SVR(kernel='rbf')),
    param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
)
search.fit(X_train, y_train)
print(f"Best SVR params: {search.best_params_}")
```

---

## SVR on Taxi Fares (Prosise's Suggestion)

Prosise suggests swapping `GradientBoostingRegressor` for `SVR` in the taxi example - worth comparing in [Section 2.8](./section-08-taxi-fare-prediction-end-to-end-project.md).

```python
from sklearn.ensemble import GradientBoostingRegressor

# Same features as Section 2.8 - expect GBM to win on RMSE for tabular fares
gbm = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
# Compare MAE in USD against SVR pipeline on identical train/test split
```

On tabular fare data, **[gradient boosting](../../GLOSSARY.md#gradient-boosting) usually wins**, but SVR shines on:

- **Medium-sized** datasets (thousands-tens of thousands of rows)
- **High-dimensional** problems (e.g., facial recognition features in Chapter 05)
- Cases where you want a **smooth** predictor (vs tree stairsteps)

---

## SVR vs Other Regressors

| | Linear Regression | SVR (RBF) | Random Forest |
|---|------------------|-----------|---------------|
| Nonlinearity | Manual features | Kernel | Automatic |
| Scaling needed | Recommended | **Required** | No |
| Speed on 1M rows | Fast | Slow | Moderate |
| Interpretability | Coefficients | Opaque | Feature importance |
| Outliers | Sensitive | ε-tube helps | Robust |

---

## When to Use SVR

| ✅ | ❌ |
|----|-----|
| Medium datasets, high dimensions | Millions of rows (slow) |
| Need kernel flexibility | Need feature importance scores |
| Complement to tree ensembles | Default first choice for tabular |
| Smooth decision boundary desired | Need fast iteration in notebooks |

---

## Connection to Chapter 05 (Full SVM Treatment)

This section gives you **regression-flavored** SVR - enough to compare against trees on taxi fares. [Chapter 05](../chapter-05-support-vector-machines/README.md) goes deeper:

- Margin classification and support vectors in detail
- Polynomial and custom kernels
- Face recognition and digit classification case studies from Prosise

Think of SVR here as a **sampler plate** before the full SVM banquet.

| Topic | This section (2.5) | Chapter 05 |
|-------|-------------------|-----------|
| ε-insensitive loss | ✅ | ✅ |
| Kernel trick | Overview | Full derivation |
| Classification SVM | Mentioned | Primary focus |
| Taxi fare comparison | ✅ | - |

---

## Common Mistakes

1. **Forgetting `StandardScaler`** - SVR with raw `trip_distance` and `hour` will misbehave
2. **`epsilon` in wrong units** - 0.01 USD vs 0.01 log-fare are very different tubes
3. **Expecting SVR to win on every tabular problem** - trees often beat it on fares
4. **Huge `C` without CV** - memorizes noise
5. **Confusing SVR with SVM classifier** - same family, different loss (ε-tube vs margin)

---

## Self-Check

1. Why must SVR features be scaled?  
   *Distance-based kernels and penalty terms treat large-scale features as more important.*
2. What does ε control?  
   *Width of the "no penalty" zone around predictions.*
3. How is SVR related to SVM classification?  
   *Both use support vectors and kernels; regression uses ε-insensitive loss instead of class margins.*
4. Why might Prosise still deploy GBM over SVR for taxi fares?  
   *Better accuracy on tabular interactions, faster training at scale.*

---

## References

- [Scikit-Learn SVR](https://scikit-learn.org/stable/chapters/generated/sklearn.svm.SVR.html)
- [Scikit-Learn SVM Guide](https://scikit-learn.org/stable/chapters/svm.html)
- [LIBSVM Documentation](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- Prosise, *Applied ML and AI for Engineers*, Ch. 2 & Ch. 5
- [StatQuest: SVM](https://www.youtube.com/watch?v=efR1C6CvhmE)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 2.4](./section-04-ensemble-methods-random-forests-and-gradient-boosting.md) | **Next:** [Section 2.6 - Accuracy Measures](./section-06-regression-accuracy-measures.md)
