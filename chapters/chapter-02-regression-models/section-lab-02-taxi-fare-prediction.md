# Lab 02: Taxi Fare Prediction

> **Prerequisites:** Sections [2.1](./section-01-introduction-to-regression.md)-[2.8](./section-08-taxi-fare-prediction-end-to-end-project.md)  
> **Estimated time:** 3-4 hours  
> **Glossary:** [../../GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Objectives

By the end of this lab you will:

1. Complete an end-to-end [regression](../../GLOSSARY.md#regression) pipeline on real (or realistic) taxi data
2. Beat the mean baseline with at least two algorithms
3. Use [cross-validation](../../GLOSSARY.md#cross-validation) to select the best [model](../../GLOSSARY.md#model)
4. Produce a business-ready recommendation memo with errors in **USD**

This lab mirrors Prosise Chapter 2's taxi fare project - the same capstone you'll extend when operationalizing models in [Chapter 07](../chapter-07-operationalizing-models/README.md).

---

## Setup

```python
# Recommended environment
# pip install pandas numpy scikit-learn matplotlib seaborn jupyter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

Create a Jupyter notebook named `lab-02-taxi-fares.ipynb` and work through parts A-E below.

---

## Part A: Data (45 min)

### Option 1 - Real NYC TLC data

Download from [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Use yellow taxi files; extract or map columns to match Prosise's schema (`fare_amount`, `trip_distance`, `passenger_count`, `pickup_datetime`).

### Option 2 - Synthetic fallback

Use NYC TLC data or this synthetic fallback:

```python
np.random.seed(42)
n = 5000
df = pd.DataFrame({
    'trip_distance': np.random.exponential(3, n).clip(0.1, 50),
    'passenger_count': np.random.randint(1, 5, n),
    'pickup_datetime': pd.date_range('2023-01-01', periods=n, freq='5min'),
})
df['hour'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_rush'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
df['fare_amount'] = (
    3.0 + 2.5 * df['trip_distance']
    + 0.5 * df['passenger_count']
    + 1.5 * df['is_rush']
    + np.random.normal(0, 2, n)
).clip(2.5, 150)
```

### Tasks

- Plot `fare_amount` vs `trip_distance` with labeled axes (USD, miles)
- Compute `df.describe()` and correlation of numeric columns
- Identify and remove outliers - **document your rules in markdown cells**
  - Example: `fare_amount <= 0`, `fare_amount > 200`, `trip_distance > 100`
- Report final row count before/after cleaning

> **In plain English:** Cleaning is not "optional preprocessing." One bad row can drag [linear regression](../../GLOSSARY.md#linear-regression) coefficients like a shopping cart with a stuck wheel.

---

## Part B: Models (90 min)

### Train/test split

```python
from sklearn.model_selection import train_test_split

features = ['trip_distance', 'passenger_count', 'hour',
            'day_of_week', 'is_weekend', 'is_rush']
target = 'fare_amount'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
```

### Mean baseline (required first!)

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

baseline = y_train.mean()
y_pred_base = np.full(len(y_test), baseline)

def report_metrics(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label:22s}  RMSE={rmse:.2f} USD  MAE={mae:.2f} USD  R²={r2:.3f}")

report_metrics(y_test, y_pred_base, 'Mean Baseline')
```

### Train and compare

| Model | Required |
|-------|----------|
| Mean baseline | ✅ |
| [LinearRegression](../../GLOSSARY.md#linear-regression) | ✅ |
| [RandomForest](../../GLOSSARY.md#random-forest) | ✅ |
| [GradientBoosting](../../GLOSSARY.md#gradient-boosting) | ✅ |
| SVR (optional) | ⭐ bonus |

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=RANDOM_STATE),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE),
    'SVR': make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10)),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    report_metrics(y_test, model.predict(X_test), name)
```

Build a **metric comparison table** (pandas DataFrame) for your deliverables.

**Success criterion:** At least two models beat mean baseline on test [MAE](../../GLOSSARY.md#mae).

---

## Part C: Cross-Validation (45 min)

Run 5-fold CV on your top two models from Part B.

```python
from sklearn.model_selection import cross_val_score

for name, model in top_two_models.items():
    scores = cross_val_score(
        model, X, y, cv=5,
        scoring='neg_root_mean_squared_error', n_jobs=-1
    )
    rmse = -scores
    print(f"{name}: CV RMSE = {rmse.mean():.2f} USD ± {rmse.std():.2f} USD")
```

**Answer in your notebook:**

- Which model wins on average?
- Is the std deviation small enough to trust the winner?
- Does the CV winner match your test-set winner? If not, why might that happen?

---

## Part D: Residuals (30 min)

Plot residuals vs predicted for the winner.

```python
best_model.fit(X_train, y_train)  # if not already fitted on full train
y_pred = best_model.predict(X_test)
residuals = y_test - y_pred

plt.figure(figsize=(8, 4))
plt.scatter(y_pred, residuals, alpha=0.4)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted fare (USD)')
plt.ylabel('Residual (USD)')
plt.title(f'Residuals - {winner_name}')
plt.show()
```

Write **2-3 sentences** on any patterns (funnel, curve, clusters). Link to [Section 2.6](./section-06-regression-accuracy-measures.md) red-flag table.

---

## Part E: Memo (30 min)

Write a half-page memo for a non-technical manager:

1. **Problem:** What are we predicting?
2. **Best model:** Name and why (accuracy + practicality)
3. **Accuracy:** "Typical error is X USD" (use MAE)
4. **Limitations:** What could go wrong? (missing tolls, airports, outliers)
5. **Next step:** Mention [Chapter 07](../chapter-07-operationalizing-models/README.md) deployment / ONNX export

### Memo template

```
TO: Product Director, TaxiCo
FROM: [Your name], ML Engineering
RE: Fare Estimator - Model Recommendation

We built a regression model to predict trip fares in USD before pickup...

Recommended model: [name]
Expected accuracy: MAE of [X] USD on held-out test data
...
```

---

## Grading Rubric (Self-Assessment)

| Criterion | Points | Done? |
|-----------|--------|-------|
| EDA plots + documented cleaning | 20 | ☐ |
| Mean baseline reported | 10 | ☐ |
| ≥3 models compared with RMSE/MAE/R² | 25 | ☐ |
| 5-fold CV on top 2 models | 20 | ☐ |
| Residual plot + interpretation | 15 | ☐ |
| Manager memo | 10 | ☐ |

---

## Common Pitfalls

1. **Fitting on full data before split** - scalers and outlier stats leak test info
2. **SVR without StandardScaler** - poor results, wrong conclusion about SVR
3. **Celebrating test metrics after 50 GridSearch iterations on same test set**
4. **Memo says "high R²" without dollar error** - managers need USD
5. **Forgetting glossary links** when writing markdown reflections

> **Humorous analogy:** Shipping a model without a baseline comparison is like a chef saying "this soup is good" without ever tasting water.

---

## Deliverables Checklist

- [ ] Jupyter notebook with all code and markdown explanations
- [ ] Metric comparison table (all models + baseline)
- [ ] Residual plots for winning model
- [ ] Recommendation memo (Part E)
- [ ] Answers to Chapter 02 self-assessment questions from Sections 2.1-2.8

---

## Stretch Goals (Optional)

- Tune `GradientBoostingRegressor` with `GridSearchCV` ([Section 2.7](./section-07-model-comparison-and-cross-validation.md))
- Add `trip_distance * is_rush` interaction feature - does linear model improve?
- Try `log1p(fare_amount)` as target - compare RMSE in log space vs back-transformed USD
- Export best model with `joblib` - preview for Chapter 07

---

## References

- [NYC TLC Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [Scikit-Learn Regression](https://scikit-learn.org/stable/supervised_learning.html#regression)
- [Kaggle: NYC Taxi Fare Prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction)
- Prosise, *Applied ML and AI for Engineers*, Ch. 2
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Next Chapter:** [03 - Classification Models](../chapter-03-classification-models/README.md)