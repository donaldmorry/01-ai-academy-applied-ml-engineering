# Section 9.4: Regression with Neural Networks - Taxi Fare Prediction

> **Source:** Prosise, Ch. 9 - "Using a Neural Network to Predict Taxi Fares"  
> **Prerequisites:** [Section 9.3](./section-03-network-sizing-and-architecture.md) | [Chapter 02 Section 2.8](../chapter-02-regression-models/section-08-taxi-fare-prediction-end-to-end-project.md)  
> **Glossary:** [regression](../../GLOSSARY.md#regression) | [mae](../../GLOSSARY.md#mae) | [r-squared](../../GLOSSARY.md#r-squared) | [neural-network](../../GLOSSARY.md#neural-network)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Same Problem, New Algorithm

Chapter 2 solved NYC taxi fare prediction with [linear regression](../../GLOSSARY.md#linear-regression), decision trees, and random forests. Chapter 9 revisits it with a [neural network](../../GLOSSARY.md#neural-network) - same business question, different learner.

**Goal:** predict `fare_amount` from day-of-week, hour, and trip distance.

> **In plain English:** Three numbers go in; one dollar amount comes out. The network learns nonlinear combinations (rush-hour × distance) without you hand-coding interaction terms.

---

## Load & Prepare Data (Prosise Pipeline)

```python
import pandas as pd
import numpy as np
from math import sqrt

df = pd.read_csv('data/taxi-fares.csv', parse_dates=['pickup_datetime'])
print(df.shape)
df.head()
```

### Feature engineering (matches Chapter 2)

```python
# Single-passenger trips only
df = df[df['passenger_count'] == 1].copy()
df = df.drop(['key', 'passenger_count'], axis=1)

# Vectorized alternative to Prosise's iterrows loop
dt = df['pickup_datetime']
df['day_of_week'] = dt.dt.weekday          # 0=Monday
df['pickup_time'] = dt.dt.hour             # 0-23

x_miles = (df['dropoff_longitude'] - df['pickup_longitude']) * 54.6
y_miles = (df['dropoff_latitude'] - df['pickup_latitude']) * 69.0
df['distance'] = np.sqrt(x_miles**2 + y_miles**2)

df.drop(['pickup_datetime', 'pickup_longitude', 'pickup_latitude',
         'dropoff_longitude', 'dropoff_latitude'], axis=1, inplace=True)

# Outlier removal - Prosise bounds
df = df[(df['distance'] > 1.0) & (df['distance'] < 10.0)]
df = df[(df['fare_amount'] > 0.0) & (df['fare_amount'] < 50.0)]

print(f"Clean rows: {len(df):,}")
df.head()
```

**Features:** `day_of_week`, `pickup_time`, `distance`  
**Target:** `fare_amount`

---

## Build the Regression Network

Prosise uses two hidden layers of 512 ReLU neurons:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

tf.random.set_seed(42)

model = Sequential([
    Dense(512, activation='relu', input_shape=(3,)),
    Dense(512, activation='relu'),
    Dense(1)  # linear output - regression
])

model.compile(optimizer='adam', loss='mae', metrics=['mae'])
model.summary()
```

| Choice | Why |
|--------|-----|
| `loss='mae'` | [MAE](../../GLOSSARY.md#mae) in dollars - interpretable |
| No output activation | Fare can be any positive real value |
| `adam` | Prosise's default optimizer |

---

## Train

```python
X = df.drop('fare_amount', axis=1).values.astype(np.float32)
y = df['fare_amount'].values.astype(np.float32)

history = model.fit(
    X, y,
    validation_split=0.2,
    epochs=100,
    batch_size=100,
    verbose=1
)
```

With ~38,000 samples and `batch_size=100`, each epoch runs ~380 gradient updates.

### Plot training curves (Fig. 9-1 style)

```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

hist = history.history
epochs = range(1, len(hist['mae']) + 1)

plt.figure(figsize=(8, 4))
plt.plot(epochs, hist['mae'], '-', label='Training MAE')
plt.plot(epochs, hist['val_mae'], ':', label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error ($)')
plt.title('Taxi Fare - Training vs Validation')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()

final_val_mae = hist['val_mae'][-1]
print(f"Final validation MAE: ${final_val_mae:.2f}")
```

**Prosise's benchmark:** validation MAE ≈ **2.25 USD** - predictions within about 2.25 USD on average.

---

## R² Score with Scikit-Learn

Keras doesn't compute $R^2$ natively. Use scikit on predictions:

$$
R^2 = 1 - \frac{\sum_{i}(y_i - \hat{y}_i)^2}{\sum_{i}(y_i - \bar{y})^2}
$$
> **Readable form:** R² = 1 minus (your errors) divided by (errors of always guessing the mean). 1.0 is perfect; 0 means no better than the mean.

```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

y_pred = model.predict(X, verbose=0).flatten()

print(f"R²:   {r2_score(y, y_pred):.4f}")      # Prosise: ~0.75
print(f"MAE:  ${mean_absolute_error(y, y_pred):.2f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y, y_pred)):.2f}")
```

---

## Make Predictions

Friday 5 PM, 2-mile trip (`day_of_week=4`, `hour=17`, `distance=2.0`):

```python
friday = model.predict(np.array([[4, 17, 2.0]], dtype=np.float32), verbose=0)
saturday = model.predict(np.array([[5, 17, 2.0]], dtype=np.float32), verbose=0)

print(f"Friday 5pm, 2 mi:   ${friday[0,0]:.2f}")
print(f"Saturday 5pm, 2 mi: ${saturday[0,0]:.2f}")
```

**Discussion question:** Does Saturday predict higher or lower? Does that match NYC pricing patterns (weekend surcharges, traffic)?

---

## Compare to Chapter 02 Baselines

| Model (Chapter 02) | Typical test MAE | Notes |
|-------------------|------------------|-------|
| Linear regression | ~$3-4 | Misses nonlinear surge |
| Random forest | ~$2.0-2.5 | Strong tabular baseline |
| **NN (this section)** | ~$2.25 val MAE | Competitive, not magic |

```python
# Optional: load your Chapter 02 random forest for side-by-side
# from joblib import load
# rf = load('models/taxi_rf.joblib')
# rf_mae = mean_absolute_error(y_test, rf.predict(X_test))
```

**Key insight from Prosise:** Neural networks can fit nonlinear patterns, but on **small tabular** data, gradient boosting or random forests often match or beat them with less tuning.

---

## Experiments (Prosise Homework)

### 1. Embrace randomness

```python
r2_scores = []
for seed in range(5):
    tf.random.set_seed(seed)
    m = Sequential([
        Dense(512, activation='relu', input_shape=(3,)),
        Dense(512, activation='relu'),
        Dense(1)
    ])
    m.compile(optimizer='adam', loss='mae', metrics=['mae'])
    m.fit(X, y, validation_split=0.2, epochs=50, batch_size=100, verbose=0)
    r2_scores.append(r2_score(y, m.predict(X, verbose=0).flatten()))
print(f"R² across 5 runs: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
```

### 2. Vary hidden width

Try `128`, `256`, `512`, `1024` - average over 3 seeds each.

### 3. Vary batch size

```python
for bs in [32, 100, 256, 512]:
    # time per epoch vs final val_mae
    pass
```

Larger batches → faster epochs; accuracy may rise or fall.

### 4. Tiny network (16 neurons)

```python
tiny = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),
    Dense(1)
])
tiny.compile(optimizer='adam', loss='mae', metrics=['mae'])
tiny.summary()  # ~65 parameters - expect much worse R²
```

---

## Improvements Beyond Prosise

| Technique | When to try |
|-----------|-------------|
| `StandardScaler` on features | Distance dominates scale |
| Early stopping | [Section 9.8](./section-08-callbacks-training-control-and-model-persistence.md) |
| Dropout | If val MAE diverges from train - [Section 9.7](./section-07-dropout-and-regularization.md) |
| Log-transform target | Long-tailed fare distribution |

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)
# Retrain on X_scaled - compare val MAE
```

---

## Production Checklist

1. Save preprocessing (scaler, feature list) with the model
2. Validate on a **held-out test set** not used in `validation_split`
3. Monitor MAE in production - data drift breaks fare models
4. Compare to simpler baselines before deploying NN complexity

---

## Self-Check

1. Why `loss='mae'` instead of `mse` for stakeholder communication?
2. What does validation MAE ≈ $2.25 mean for a rider?
3. Why might R² ≈ 0.75 be acceptable but not stellar?
4. When would you stop using a neural network and ship random forest instead?

---

## What's Next

[Section 9.5](./section-05-binary-classification-credit-card-fraud-detection.md) changes two things - sigmoid output and `binary_crossentropy` - to detect credit card fraud from [Chapter 03](../chapter-03-classification-models/section-07-case-study-credit-card-fraud-detection.md).

---

**Previous:** [Section 9.3](./section-03-network-sizing-and-architecture.md)  
**Next:** [Section 9.5 - Binary Classification: Fraud Detection](./section-05-binary-classification-credit-card-fraud-detection.md)



