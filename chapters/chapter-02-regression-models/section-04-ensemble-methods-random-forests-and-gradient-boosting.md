# Section 2.4: Ensemble Methods - Random Forests & Gradient Boosting

> **Source:** Prosise, Ch. 2 - "Random Forests" & "Gradient-Boosting Machines"  
> **Prerequisites:** [Section 2.3](./section-03-decision-trees-for-regression.md) | [Glossary: random forest](../../GLOSSARY.md#random-forest)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Wisdom of Crowds for Algorithms

**Real-life analogy:** One doctor's diagnosis might be wrong. Five independent specialists voting? Much more reliable. **Ensemble learning** combines many [models](../../GLOSSARY.md#model) into one stronger predictor.

Two star players in tabular [regression](../../GLOSSARY.md#regression):
1. **[Random Forest](../../GLOSSARY.md#random-forest)** - many trees in parallel, averaged
2. **[Gradient Boosting](../../GLOSSARY.md#gradient-boosting)** - trees in sequence, each fixing prior mistakes

Prosise's taxi fare chapter treats ensembles as the **accuracy ceiling** for structured data - after you've tried [linear regression](../../GLOSSARY.md#linear-regression) as a fast baseline.

---

## Bagging: Bootstrap Aggregating (Random Forest's Foundation)

**Bagging** trains many models on random bootstrap samples (draw rows **with replacement**), then averages predictions.

$$
\hat{y}_{\text{bag}} = \frac{1}{B}\sum_{b=1}^{B} f_b(\mathbf{x})
$$
> **Readable form:** bagged prediction = average of B models, each trained on a different random sample of rows

Why it helps: individual trees have **high variance** (small data changes → different tree). Averaging cancels much of that noise.

> **Humorous analogy:** One weather forecaster panics at every cloud. Ten forecasters who trained on different years of data? The panic averages out.

---

## Random Forest: Democracy of Trees

### How it works

1. Train hundreds of decision trees on **bootstrap samples** (random rows with replacement)
2. At each split, consider only a **random subset of features** (decorrelates trees)
3. **Average** all tree predictions

$$
\hat{y}_{\text{RF}} = \frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{x})
$$
> **Readable form:** random forest prediction = (tree₁ prediction + tree₂ prediction + … + tree_B prediction) divided by B

> **In plain English:** Grow 200 slightly different trees (different data slices, different feature choices). Ask them all "what's the fare?" Take the average. Outliers that fool one tree rarely fool all 200.

### Code

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

np.random.seed(42)
n = 1000
distance = np.random.uniform(0, 25, n)
passengers = np.random.randint(1, 5, n)
fare = 5 + 1.5 * distance + 0.8 * passengers + np.random.normal(0, 3, n)

X = np.column_stack([distance, passengers])
y = fare

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

mae = mean_absolute_error(y_test, rf.predict(X_test))
print(f"RF Test MAE: {mae:.2f} USD")

# Feature importance - which inputs matter most?
for name, imp in zip(['distance', 'passengers'], rf.feature_importances_):
    print(f"  {name}: {imp:.3f}")
```

**Milestone:** Random forests are the **default first try** for structured/tabular data in industry - robust, parallelizable, hard to break.

### Random Forest hyperparameters

| Parameter | Effect | Tuning tip |
|-----------|--------|------------|
| `n_estimators` | More trees → stabler, slower | 100-500 typical |
| `max_depth` | Deeper → more variance per tree | 8-16 for fares |
| `max_features` | Fewer → more diversity | `'sqrt'` default |
| `min_samples_leaf` | Larger → smoother leaves | Raise if overfitting |

---

## Gradient Boosting: The Perfectionist

### How it works

1. Train a small tree on residuals (errors)
2. Add it to the ensemble
3. Repeat - each new tree focuses on what previous trees got wrong

$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})
$$
> **Readable form:** model at round m = previous model + (learning rate × new small tree)

where $\eta$ is **learning rate** (shrinkage) and $h_m$ is the new weak learner.

The new tree $h_m$ is fit to **negative gradients** of the loss - for squared error, that's essentially the residual $y - F_{m-1}(\mathbf{x})$.

> **In plain English:** Student takes a practice test (tree 1), misses questions. Studies only missed topics (tree 2), retakes. Repeat. Final score = sum of all improvement rounds.

> **Humorous analogy:** Random forest is a committee meeting where everyone votes at once. Gradient boosting is a relay race where each runner only sprints the distance the previous runner fell short.

### Code

```python
from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,      # Prosise: stochastic boosting reduces overfitting
    random_state=42
)
gbm.fit(X_train, y_train)

print(f"GBM Test MAE: {mean_absolute_error(y_test, gbm.predict(X_test)):.2f} USD")
```

### Learning rate vs number of trees

| `learning_rate` | `n_estimators` | Tradeoff |
|-----------------|----------------|----------|
| High (0.3) | Fewer trees needed | Fast but can overshoot |
| Low (0.05) | Many trees needed | Slower, often more accurate |

Rule of thumb: if you halve `learning_rate`, you may need ~2× trees for similar fit.

---

## XGBoost / LightGBM (Industry Standard)

Prosise notes Scikit-Learn's `GradientBoostingRegressor` works, but production often uses:

```python
# pip install xgboost
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
print(f"XGBoost MAE: {mean_absolute_error(y_test, xgb_model.predict(X_test)):.2f} USD")
```

| Library | Strength |
|---------|----------|
| XGBoost | Mature, GPU support, widely deployed |
| LightGBM | Fast on large data, leaf-wise growth |
| CatBoost | Strong on categorical features |

All three appear in Kaggle tabular competitions and many production fare/pricing systems.

---

## Random Forest vs Gradient Boosting

| | Random Forest | Gradient Boosting |
|---|--------------|-----------------|
| Training | Parallel (fast) | Sequential (slower) |
| [Overfitting](../../GLOSSARY.md#overfitting) risk | Lower | Higher if untuned |
| Typical accuracy | Very good | Often best |
| Tuning effort | Minimal | Significant |
| Feature importance | Built-in | Built-in |
| Prosise taxi fares | Strong | Often **wins** |

> **In plain English (again):** RF = reliable SUV. GBM = race car - faster lap times but needs a skilled driver ([hyperparameter](../../GLOSSARY.md#hyperparameter) tuning).

---

## When Ensembles Beat Linear Models

Prosise's taxi fare dataset: distance drives price, but **interactions** matter (long trip + rush hour + airport). Linear models miss interactions unless you hand-engineer them. Trees discover them automatically.

Example interaction the forest learns without you typing `distance * is_rush`:

```
IF distance > 15 AND is_rush = 1 → higher leaf mean than distance alone predicts
```

```python
# Linear model needs explicit cross-term:
# from sklearn.preprocessing import PolynomialFeatures  # or manual column
# X['distance_x_rush'] = X['distance'] * X['is_rush']
```

Ensembles save feature-engineering labor - at the cost of interpretability.

---

## Stacking (Preview)

Advanced teams sometimes **stack** models: train RF, GBM, and SVR; feed their predictions into a meta [linear regression](../../GLOSSARY.md#linear-regression). Prosise keeps the core chapter on RF + GBM; stacking is bonus material for competitions.

---

## Common Mistakes

1. **One giant GBM with `learning_rate=1.0`** - overshoots, [overfits](../../GLOSSARY.md#overfitting)
2. **Comparing RF and GBM on one lucky split** - use [cross-validation](../../GLOSSARY.md#cross-validation) ([Section 2.7](./section-07-model-comparison-and-cross-validation.md))
3. **Ignoring `subsample`** - stochastic boosting (0.8) often generalizes better
4. **Expecting coefficient-style explanations** - use `feature_importances_`, SHAP, or partial dependence
5. **Training on leaked future data** - time-based splits matter for fare surge patterns

---

## Self-Check

1. Why does averaging trees reduce variance?  
   *Independent errors cancel; no single quirky tree dominates.*
2. What does `learning_rate=0.1` do in gradient boosting?  
   *Shrinks each tree's contribution - smaller steps, less overshoot.*
3. Why use `subsample=0.8`?  
   *Trains each tree on 80% of rows - adds randomness, reduces overfitting.*
4. Which ensemble does Prosise typically pick for taxi fares?  
   *Gradient boosting (or XGBoost) - often edges RF on tabular accuracy.*

---

## References

- [Scikit-Learn RandomForestRegressor](https://scikit-learn.org/stable/chapters/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Scikit-Learn GradientBoostingRegressor](https://scikit-learn.org/stable/chapters/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- Prosise, *Applied ML and AI for Engineers*, Ch. 2 - Random Forests & GBM
- [StatQuest: Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
- [StatQuest: Gradient Boosting](https://www.youtube.com/watch?v=3CC4N4z2GJ4)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 2.3](./section-03-decision-trees-for-regression.md) | **Next:** [Section 2.5 - SVR](./section-05-support-vector-regression-svr.md)



