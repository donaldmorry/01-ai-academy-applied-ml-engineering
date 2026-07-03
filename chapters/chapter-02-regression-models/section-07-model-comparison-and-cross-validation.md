# Section 2.7: Model Comparison & Cross-Validation

> **Source:** Prosise, Ch. 2 - cross-validation & model selection themes  
> **Prerequisites:** [Section 2.6](./section-06-regression-accuracy-measures.md) | [Glossary: cross-validation](../../GLOSSARY.md#cross-validation)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## One Split Can Lie to You

You train on January-June data, test on July. Lucky month? Your model looks genius. Unlucky? Fired.

**[Cross-validation](../../GLOSSARY.md#cross-validation) (CV)** rotates through multiple train/test splits so metrics reflect **typical** performance, not one lottery draw.

> **In plain English:** Instead of one pop quiz, give five pop quizzes on different chapters. Average grade is a fairer measure of understanding.

Prosise stresses this before declaring [gradient boosting](../../GLOSSARY.md#gradient-boosting) the taxi fare winner - a single 80/20 split can flatter the wrong model.

---

## **Milestone: k-Fold Cross-Validation**

Split data into $k$ equal folds. For each fold $i$:

1. Train on the other $k-1$ folds
2. Evaluate on fold $i$
3. Record metric

$$
\text{CV Score} = \frac{1}{k}\sum_{i=1}^{k} \text{metric}_i
$$
> **Readable form:** cross-validation score = average of the metric across all k fold evaluations

Also report **standard deviation** across folds - high std means unstable model or too little data.

| $k$ | Tradeoff |
|-----|----------|
| 3 | Fast, noisy estimate |
| 5 | Prosise's common choice |
| 10 | Smoother, more compute |
| $n$ (LOOCV) | Expensive, low bias |

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

np.random.seed(42)
n = 800
distance = np.random.uniform(0, 25, n)
passengers = np.random.randint(1, 5, n)
fare = 5 + 1.5 * distance + 0.8 * passengers + np.random.normal(0, 3, n)
X = np.column_stack([distance, passengers])
y = fare

models = {
  'LinearRegression': LinearRegression(),
  'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
  'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
  'SVR (RBF)': make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10)),
}

print(f"{'Model':<20} {'CV RMSE (mean)':<18} {'CV RMSE (std)'}")
print("-" * 50)

for name, model in models.items():
    # neg_root_mean_squared_error because sklearn maximizes scores
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    rmse_scores = -scores
    print(f"{name:<20} {rmse_scores.mean():.2f} USD          ±{rmse_scores.std():.2f}")
```

---

## Hyperparameter Tuning with GridSearchCV

**[Hyperparameters](../../GLOSSARY.md#hyperparameter)** aren't learned from data - you choose them before `fit`.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 8, 12],
    'learning_rate': [0.05, 0.1],
}

gbm = GradientBoostingRegressor(random_state=42)
search = GridSearchCV(
    gbm, param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
search.fit(X, y)

print(f"Best params: {search.best_params_}")
print(f"Best CV RMSE: {-search.best_score_:.2f} USD")
```

> **Humorous analogy:** GridSearch is trying every combo on a coffee machine - grind size × water temp × brew time - until you find the cup that doesn't taste like regret.

### RandomizedSearchCV (when grid is huge)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 12),
    'learning_rate': uniform(0.01, 0.2),
}
rand_search = RandomizedSearchCV(
    gbm, param_dist, n_iter=20, cv=5,
    scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1
)
rand_search.fit(X, y)
```

Sample random combos instead of exhaustive grid - often finds 95% of the gain in 20% of the time.

---

## Bias-Variance Tradeoff (Regression View)

$$
\text{Expected Test Error} \approx \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}
$$
> **Readable form:** expected test error ≈ squared bias + variance + noise you can never model away

| | High Bias | High Variance |
|---|-----------|---------------|
| **Symptom** | Bad on train AND test | Great on train, bad on test |
| **Cause** | Model too simple | Model too complex |
| **Fix** | More features, complex model | [Regularization](../../GLOSSARY.md#regularization), more data, simpler model |
| **Example** | [Linear regression](../../GLOSSARY.md#linear-regression) on curved data | Deep tree, `max_depth=None` |

> **In plain English (again):** **Bias** = always wrong in the same direction (blurry glasses). **Variance** = wildly different answers depending on which training data you saw (mood swings).

Formal theory in [Course 2, Chapter 19](https://github.com/donaldmorry/02-ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-19-learning-from-examples/README.md) and [Course 3, Chapter 05](https://github.com/donaldmorry/03-ai-academy-deep-learning-foundations/blob/main/chapters/chapter-05-machine-learning-basics/README.md).

---

## Nested Cross-Validation (Avoid Leakage When Tuning)

**Problem:** If you GridSearch on the same CV folds you report, metrics are optimistically biased.

**Fix:** Outer CV for performance estimate, inner CV for tuning:

```
Outer fold 1:  [====train====|test|]
                 └─ inner GridSearch on train only
```

Prosise's taxi chapter uses a simpler workflow (tune on full training set with CV, evaluate once on held-out test) - acceptable when you have a **separate final test set** you touch only once.

---

## Time-Based Splits for Taxi Data

Random `train_test_split` shuffles months together. For fare data with **seasonality**, consider:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
  # train always before test in time order
  pass
```

Prosise's published TLC walkthrough often uses random splits for pedagogy; production fare models should respect time.

---

## Model Selection Workflow

```
1. Establish mean baseline (Section 2.1)
2. Try linear regression (fast, interpretable)
3. Try random forest (strong default)
4. Try gradient boosting (often best on tabular)
5. Compare with 5-fold CV RMSE
6. Tune winner with GridSearchCV
7. FINAL evaluation on held-out test set (touch once!)
8. Analyze residuals (Section 2.6)
```

**Golden rule:** Never tune [hyperparameters](../../GLOSSARY.md#hyperparameter) using the test set. CV uses training data only. Test set is the **final exam** - open the envelope once.

### Pipelines in CV (critical for SVR)

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('svr', SVR(kernel='rbf'))
])
# cross_val_score(pipe, X, y) - scaler refits inside each fold, no leakage
```

---

## When Is GridSearch Overkill?

| Situation | Recommendation |
|-----------|----------------|
| Exploring first models | Default hyperparameters + CV |
| RF with 100 trees | Often good enough |
| GBM for production | GridSearch or RandomizedSearch |
| Linear regression | No grid needed (OLS is closed-form) |
| Deadline in 2 hours | RF defaults, ship baseline comparison |

---

## Common Mistakes

1. **Tuning on test data** - inflated metrics, production disappointment
2. **Single split decisions** - always use CV for model selection
3. **Forgetting to scale SVR inside CV** - fit scaler on full data before CV leaks info
4. **Ignoring CV std** - model with 3.0 ± 2.5 USD RMSE is unstable
5. **Comparing models on different metrics** - stick to one primary scorer

---

## Self-Check

1. Why use 5 folds instead of 1 train/test split?  
   *More reliable estimate of generalization; less variance from one lucky split.*
2. What does high CV standard deviation indicate?  
   *Model or data slice sensitivity - performance varies a lot across folds.*
3. When is GridSearchCV overkill?  
   *Early exploration with strong defaults (e.g., RF with 100 trees).*
4. Why pipeline SVR with StandardScaler in CV?  
   *Scaler must be fit only on training portion of each fold.*

---

## References

- [Scikit-Learn Cross-Validation](https://scikit-learn.org/stable/chapters/cross_validation.html)
- [Scikit-Learn GridSearchCV](https://scikit-learn.org/stable/chapters/generated/sklearn.model_selection.GridSearchCV.html)
- [Scikit-Learn TimeSeriesSplit](https://scikit-learn.org/stable/chapters/generated/sklearn.model_selection.TimeSeriesSplit.html)
- Prosise, *Applied ML and AI for Engineers*, Ch. 2 - cross-validation introduction
- [StatQuest: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 2.6](./section-06-regression-accuracy-measures.md) | **Next:** [Section 2.8 - Taxi Fare Project](./section-08-taxi-fare-prediction-end-to-end-project.md)
