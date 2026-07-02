# Section 2.6: Regression Accuracy Measures

> **Source:** Prosise, Ch. 2 - "Accuracy Measures for Regression Models"  
> **Prerequisites:** [Section 2.2](./section-02-linear-regression.md) | [Glossary: MAE](../../GLOSSARY.md#mae)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## You Can't Improve What You Don't Measure

A model that predicts taxi fares within **500 USD** of truth is useless. One within **2 USD** is gold. **Metrics translate math into business decisions.**

For [regression](../../GLOSSARY.md#regression), every metric compares true values $y_i$ to predictions $\hat{y}_i$.

Prosise treats metric choice as a **stakeholder conversation**: executives understand "typically off by 3 dollars" more than "MSE = 9."

---

## The Core Four

### 1. MSE - Mean Squared Error

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$
> **Readable form:** MSE = (1/n) × sum of squared differences between actual and predicted values

- Squares errors → **large mistakes hurt more**
- Units: squared dollars (awkward to interpret for fares)
- What [ordinary least squares](../../GLOSSARY.md#ordinary-least-squares) minimizes

### 2. RMSE - Root Mean Squared Error

$$
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$
> **Readable form:** RMSE = square root of MSE - back in original units (USD for fares)

- Same units as target (USD for fares, °F for temperature)
- **Prosise's go-to** for comparing [regression](../../GLOSSARY.md#regression) models
- One 50 USD mistake counts like twenty-five 10 USD mistakes (because $50^2 = 2500$ vs $10^2 = 100$)

### 3. MAE - Mean Absolute Error

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$
> **Readable form:** MAE = (1/n) × sum of absolute differences between actual and predicted values

- Average dollar error, no squaring
- **Robust to outliers** - one crazy trip doesn't dominate
- Easiest metric to explain to non-technical managers

### 4. R² - Coefficient of Determination

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
$$
> **Readable form:** R-squared = 1 minus (model's squared errors divided by baseline mean predictor's squared errors)

- Fraction of variance explained
- $R^2 = 1$: perfect; $R^2 = 0$: no better than guessing mean; **negative**: worse than mean

> **In plain English:** R² answers "how much better am I than the dumb mean baseline from [Section 2.1](./section-01-introduction-to-regression.md)?"

---

## **Milestone: Pick Metrics That Match Business Pain**

| Scenario | Prefer | Why |
|----------|--------|-----|
| All errors cost equally | [MAE](../../GLOSSARY.md#mae) | Easy to explain: "off by 3 USD on average" |
| Big errors are catastrophic | [RMSE](../../GLOSSARY.md#rmse) | Penalizes huge misses |
| Executive dashboard | [R²](../../GLOSSARY.md#r-squared) | Single 0-1 summary |
| Outlier-heavy data (fraud fares) | MAE | Don't let one 500 USD trip dominate |
| Comparing models in sklearn CV | neg_RMSE | Built-in, higher-is-better convention |

**Real-life analogy:** MAE is average speed camera fines. RMSE is when speeding 40 over hurts WAY more than 5 over.

### RMSE ≥ MAE (always)

Jensen's inequality guarantees $\text{RMSE} \geq \text{MAE}$ with equality only when all errors have the same magnitude. Wider gap → more outlier-driven mistakes.

---

## Bonus Metrics

### MAPE - Mean Absolute Percentage Error

$$
\text{MAPE} = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$
> **Readable form:** MAPE = average absolute percent error relative to true value

Useful for cross-product comparisons ("5% off on average"). **Avoid when $y_i$ can be near zero** - division explodes.

### Explained Variance

```python
from sklearn.metrics import explained_variance_score
# 1.0 = perfect; can be negative if model is very bad
```

Similar spirit to $R^2$ but doesn't penalize constant bias in predictions the same way.

---

## Code: Compare Metrics

```python
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)

y_true = np.array([12.0, 15.0, 18.0, 50.0, 14.0])  # 50.0 = outlier trip
y_pred = np.array([13.0, 14.5, 17.0, 30.0, 15.0])

mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f} USD")
print(f"MAE:  {mae:.2f} USD")
print(f"R²:   {r2:.3f}")
```

### Metric function cheat sheet

```python
from sklearn.metrics import get_scorer

# sklearn CV uses 'neg_' prefix because it maximizes scores
scorers = ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']
for s in scorers:
    print(s, "→", get_scorer(s))
```

---

## Residual Analysis - The Detective Work

**Residual** $e_i = y_i - \hat{y}_i$. Plot them!

```python
import matplotlib.pyplot as plt

residuals = y_true - y_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Residuals vs predicted - should look like random cloud
axes[0].scatter(y_pred, residuals, alpha=0.7)
axes[0].axhline(0, color='r', linestyle='--')
axes[0].set_xlabel('Predicted fare (USD)')
axes[0].set_ylabel('Residual (USD)')
axes[0].set_title('Residuals vs Predicted')

# Histogram - should be roughly bell-shaped
axes[1].hist(residuals, bins=20, edgecolor='black')
axes[1].set_xlabel('Residual (USD)')
axes[1].set_title('Residual Distribution')
plt.tight_layout()
plt.show()
```

### Red flags

| Pattern | Meaning | Fix |
|---------|---------|-----|
| Funnel shape | Model worse at high fares | Log transform on target |
| Curved pattern | Nonlinearity missed | Trees, polynomials |
| One huge residual | Outlier | Investigate or remove |
| Non-zero mean residual | Systematic bias | Add feature, check leakage |

> **In plain English (again):** Good residuals look like TV static - random noise. Patterns in residuals mean your model is systematically wrong somewhere.

---

## What R² Does NOT Tell You

- **Causation** - high $R^2$ doesn't mean features cause the target
- **Extrapolation safety** - great $R^2$ in 0-20 mile range says nothing about 100-mile trips
- **Fairness** - aggregate $R^2$ can hide terrible performance on minority groups
- **Business acceptability** - $R^2 = 0.85$ with RMSE = 15 USD may still be too sloppy for pricing

### Is RMSE = 4 USD good?

**Trick question.** Compare to:

1. Mean baseline RMSE
2. Current manual process error
3. Competitor / industry benchmark

A 4 USD RMSE on 8 USD average fares is painful. On 80 USD airport trips, it may be fine.

---

## Prosise's Evaluation Workflow

```
1. Report MAE in USD (manager-friendly)
2. Report RMSE for model comparison (Prosise default)
3. Report R² for variance-explained narrative
4. Plot residuals before declaring victory
5. Never tune on test set metrics alone
```

---

## Common Mistakes

1. **Reporting only $R^2$** - hides dollar error magnitude
2. **Using MSE in presentations** - "your error is 81 squared dollars" confuses everyone
3. **Ignoring residual plots** - pretty metrics, broken assumptions
4. **Comparing RMSE across different target scales** - log-fare RMSE ≠ USD RMSE
5. **Optimizing MAE but reporting RMSE** - pick one primary metric for selection

---

## Self-Check

1. RMSE ≥ MAE always. Why?  
   *Squaring inflates large errors more than absolute value does.*
2. Can $R^2$ be negative? What does that mean?  
   *Yes - model is worse than always predicting the mean.*
3. A fare model has RMSE = 4 USD. Is that good?  
   *Only relative to baseline and business tolerance.*
4. Which metric does OLS minimize?  
   *[MSE](../../GLOSSARY.md#mse) (equivalently RMSE).*

---

## References

- [Scikit-Learn Model Evaluation](https://scikit-learn.org/stable/chapters/model_evaluation.html#regression-metrics)
- [Scikit-Learn mean_absolute_error](https://scikit-learn.org/stable/chapters/generated/sklearn.metrics.mean_absolute_error.html)
- Prosise, *Applied ML and AI for Engineers*, Ch. 2 - Accuracy Measures
- [Wikipedia: Coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination)
- [StatQuest: R-squared](https://www.youtube.com/watch?v=2AQkLW5aKaE)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 2.5](./section-05-support-vector-regression-svr.md) | **Next:** [Section 2.7 - Model Comparison](./section-07-model-comparison-and-cross-validation.md)



