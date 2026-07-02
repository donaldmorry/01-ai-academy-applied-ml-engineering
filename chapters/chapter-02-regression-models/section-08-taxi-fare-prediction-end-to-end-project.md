# Section 2.8: Taxi Fare Prediction - End-to-End Project

> **Source:** Prosise, Ch. 2 - "Using Regression to Predict Taxi Fares"  
> **Prerequisites:** Sections [2.1](./section-01-introduction-to-regression.md)-[2.7](./section-07-model-comparison-and-cross-validation.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Business Story

You're the ML engineer at a taxi company. Management wants a **fare estimator** in the rider app: enter pickup, dropoff, passenger count → see predicted price before the trip.

**Data:** NYC TLC trip records - millions of real rides with distance, time, location, and actual fare.

**This section walks Prosise's full Chapter 2 pipeline** with enhancements and links to later deployment in [Chapter 07](../chapter-07-operationalizing-models/README.md).

| Stakeholder question | Your answer comes from |
|---------------------|------------------------|
| "How accurate is it?" | [MAE](../../GLOSSARY.md#mae) in USD |
| "Which model?" | CV comparison ([Section 2.7](./section-07-model-comparison-and-cross-validation.md)) |
| "Can we ship it?" | Test set + residual audit |

---

## Step 1: Load & Explore

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Download from: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
df = pd.read_csv('data/taxi-fares.csv', parse_dates=['pickup_datetime'])

print(df.shape)
print(df.describe())
print(df.isnull().sum())

# Prosise: visualize before modeling
sns.scatterplot(data=df.sample(5000, random_state=42),
                x='trip_distance', y='fare_amount', alpha=0.3)
plt.xlabel('Trip distance (mi)')
plt.ylabel('Fare (USD)')
plt.title('Fare vs Distance (sample)')
plt.show()
```

### What to look for in EDA

| Check | Why |
|-------|-----|
| `fare_amount` distribution | Long tail? Consider log transform |
| Correlation with distance | Dominant feature? |
| Null counts | Drop or impute before split |
| Date range | Seasonality, policy changes |

---

## Step 2: Clean Data (Outliers Are Liars)

```python
# Remove impossible fares and distances - Prosise's rules
df = df[df['fare_amount'] > 0]
df = df[df['fare_amount'] < 200]
df = df[df['trip_distance'] > 0]
df = df[df['trip_distance'] < 100]
df = df.dropna(subset=['fare_amount', 'trip_distance'])

print(f"After cleaning: {len(df):,} rows")
```

> **In plain English:** One corrupted GPS reading showing a 500-mile, 0.01 USD trip will wreck [linear regression](../../GLOSSARY.md#linear-regression). Delete obvious garbage before training.

Document every filter - auditors will ask why rows disappeared.

---

## Step 3: Feature Engineering

```python
df['hour'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Rush hour flag - fares spike
df['is_rush'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

features = ['trip_distance', 'passenger_count', 'hour', 'day_of_week',
            'is_weekend', 'is_rush']
target = 'fare_amount'

X = df[features]
y = df[target]
```

**Milestone insight from Prosise:** `trip_distance` is the dominant driver - but time features capture surge patterns trees and boosting exploit.

### Optional enhancements (beyond Prosise baseline)

```python
# df['month'] = df['pickup_datetime'].dt.month
# df['distance_per_passenger'] = df['trip_distance'] / df['passenger_count'].clip(1)
# Pickup/dropoff zone IDs if available - huge lift, more engineering
```

---

## Step 4: Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Golden rule:** Split **before** any statistic computed on the full dataset (means, scalers, outlier thresholds).

---

## Step 5: Train & Compare Models

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)
    print(f"{name:22s}  RMSE={rmse:5.2f} USD  MAE={mae:5.2f} USD  R²={r2:.3f}")
    return model, pred

models = [
    ('Mean Baseline', None),  # handled separately
    ('LinearRegression', LinearRegression()),
    ('RandomForest', RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)),
    ('GradientBoosting', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
    ('SVR', make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10, epsilon=0.5))),
]

baseline = y_train.mean()
baseline_pred = np.full_like(y_test, baseline)
print(f"{'Mean Baseline':22s}  RMSE={np.sqrt(mean_squared_error(y_test, baseline_pred)):5.2f} USD  "
      f"MAE={mean_absolute_error(y_test, baseline_pred):5.2f} USD")

best_model = None
for name, model in models[1:]:
    m, _ = evaluate(name, model, X_train, y_train, X_test, y_test)
    if name == 'GradientBoosting':  # Prosise: GBM often wins
        best_model = m
```

> **Prosise's typical result:** [Gradient boosting](../../GLOSSARY.md#gradient-boosting) edges out [random forest](../../GLOSSARY.md#random-forest) and crushes [linear regression](../../GLOSSARY.md#linear-regression) on this dataset.

---

## Step 6: Cross-Validate the Winner

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    best_model, X, y, cv=5,
    scoring='neg_root_mean_squared_error', n_jobs=-1
)
print(f"5-Fold CV RMSE: {(-scores).mean():.2f} USD ± {(-scores).std():.2f} USD")
```

If CV RMSE is much worse than test RMSE, your test split may have been lucky - trust CV for model selection.

---

## Step 7: Tune the Winner (Optional but Prosise-Aligned)

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
}

search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1
)
search.fit(X_train, y_train)
best_model = search.best_estimator_
print(f"Tuned GBM: {search.best_params_}")
```

Re-evaluate tuned model on **X_test** only once at the end.

---

## Step 8: Business Interpretation

```python
# What does a 10-mile Tuesday 8 AM rush-hour trip cost?
sample = pd.DataFrame([{
    'trip_distance': 10.0,
    'passenger_count': 2,
    'hour': 8,
    'day_of_week': 1,
    'is_weekend': 0,
    'is_rush': 1,
}])

predicted_fare = best_model.predict(sample)[0]
print(f"Predicted fare: {predicted_fare:.2f} USD")
```

Write the **recommendation memo** Prosise implies:

1. **Problem:** Predict fare in USD before trip starts
2. **Best model:** Gradient boosting (or tuned variant) and why
3. **Accuracy:** "Typical error is X USD" ([MAE](../../GLOSSARY.md#mae))
4. **Known failure modes:** airport trips, tolls not in features, extreme distances
5. **Next step:** export to ONNX ([Chapter 07](../chapter-07-operationalizing-models/README.md))

---

## Step 9: Residual Analysis

```python
y_pred = best_model.predict(X_test)
residuals = y_test - y_pred

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.3, s=5)
plt.axhline(0, color='r')
plt.xlabel('Predicted (USD)'); plt.ylabel('Residual (USD)')

plt.subplot(1, 2, 2)
plt.scatter(X_test['trip_distance'], residuals, alpha=0.3, s=5)
plt.axhline(0, color='r')
plt.xlabel('Distance (mi)'); plt.ylabel('Residual (USD)')
plt.tight_layout()
plt.show()
```

Funnel shape at high fares? Consider `log(fare_amount)` as target and report errors in original USD via back-transform.

---

## Step 10: Save the Model

```python
import joblib

joblib.dump(best_model, 'models/taxi_fare_gbm.joblib')
# Later: Chapter 07 converts to ONNX for mobile deployment
```

---

## Common Project Mistakes

1. **Cleaning after split** - test leakage via global statistics
2. **Skipping mean baseline** - can't justify model complexity
3. **Reporting test RMSE after heavy tuning on same test set** - optimistic
4. **Ignoring tolls/zones** - residuals cluster geographically
5. **Deploying linear model because it's "simpler"** - when GBM wins by 30% MAE

---

## Chapter 02 Summary

You can now:

- ✅ Frame [regression](../../GLOSSARY.md#regression) problems and beat mean baselines
- ✅ Train [linear](../../GLOSSARY.md#linear-regression), tree, [forest](../../GLOSSARY.md#random-forest), [boosting](../../GLOSSARY.md#gradient-boosting), and SVR models
- ✅ Measure with [RMSE](../../GLOSSARY.md#rmse), [MAE](../../GLOSSARY.md#mae), [R²](../../GLOSSARY.md#r-squared)
- ✅ Select models with [cross-validation](../../GLOSSARY.md#cross-validation)
- ✅ Deliver an end-to-end business project

**Next chapter:** [Classification](../chapter-03-classification-models/README.md) - when the answer isn't a number but a **category**.

---

## References

- [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- Prosise, *Applied ML and AI for Engineers*, Ch. 2 - full taxi fare walkthrough
- [Kaggle: NYC Taxi Fare Prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction)
- [ONNX Model Export](https://onnx.ai/) - preview for Chapter 07
- [Scikit-Learn model persistence](https://scikit-learn.org/stable/model_persistence.html)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 2.7](./section-07-model-comparison-and-cross-validation.md) | **Lab:** [lab-02.md](./section-lab-02-taxi-fare-prediction.md)