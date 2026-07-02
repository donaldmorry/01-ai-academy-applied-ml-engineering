# Lab 09: Neural Network Triathlon

> **Prerequisites:** Sections [9.1](./section-01-keras-and-tensorflow-setup.md)-[9.8](./section-08-callbacks-training-control-and-model-persistence.md)  
> **Estimated time:** 5-7 hours  
> **Glossary:** [neural-network](../../GLOSSARY.md#neural-network) | [keras](../../GLOSSARY.md#keras) | [dropout](../../GLOSSARY.md#dropout) | [early-stopping](../../GLOSSARY.md#early-stopping)  
> **Math:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Objectives

Train three Keras models (taxi regression, fraud binary, LFW multi-class) with dropout, EarlyStopping, and ModelCheckpoint. Plot curves, save/reload models, and compare to Part I baselines (RF, logistic, SVM).

> **Humorous briefing:** Three datasets, three problem types, one API - a neural network triathlon with overfitting laps.

---

## Setup

```python
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, r2_score, classification_report,
    roc_auc_score, ConfusionMatrixDisplay, accuracy_score)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_lfw_people

SEED = 42; tf.random.set_seed(SEED); np.random.seed(SEED); sns.set_theme()
DATA_DIR, OUT_DIR, MODEL_DIR = Path('data'), Path('lab09_results'), Path('models/lab09')
OUT_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(parents=True, exist_ok=True)
```

---

## Shared Utilities

```python
def make_callbacks(name, monitor, mode='min'):
    d = MODEL_DIR / name; d.mkdir(exist_ok=True)
    return [EarlyStopping(monitor=monitor, patience=15, restore_best_weights=True, mode=mode),
            ModelCheckpoint(str(d / 'best.keras'), monitor=monitor, save_best_only=True, mode=mode),
            ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=5, min_lr=1e-6)]

def plot_history(hist, metric, title, fname):
    vk = f'val_{metric}'; ep = range(1, len(hist.history[metric]) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(ep, hist.history[metric], '-', label=f'Train {metric}')
    if vk in hist.history: plt.plot(ep, hist.history[vk], ':', label=f'Val {metric}')
    plt.xlabel('Epoch'); plt.ylabel(metric); plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(OUT_DIR / fname, dpi=120, bbox_inches='tight'); plt.show()
```

---

## Part A: Taxi Fare Regression (90 min)

*Follow [Section 9.4](./section-04-regression-with-neural-networks-taxi-fare-prediction.md). Baseline: [Chapter 02](../chapter-02-regression-models/section-08-taxi-fare-prediction-end-to-end-project.md).*

### A1. Load & engineer features

```python
df = pd.read_csv(DATA_DIR / 'taxi-fares.csv', parse_dates=['pickup_datetime'])
df = df[df['passenger_count'] == 1].copy()
df = df.drop(['key', 'passenger_count'], axis=1)

dt = df['pickup_datetime']
df['day_of_week'] = dt.dt.weekday
df['pickup_time'] = dt.dt.hour
x_m = (df['dropoff_longitude'] - df['pickup_longitude']) * 54.6
y_m = (df['dropoff_latitude'] - df['pickup_latitude']) * 69.0
df['distance'] = np.sqrt(x_m**2 + y_m**2)

df.drop(['pickup_datetime', 'pickup_longitude', 'pickup_latitude',
         'dropoff_longitude', 'dropoff_latitude'], axis=1, inplace=True)
df = df[(df['distance'] > 1) & (df['distance'] < 10)]
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 50)]

X = df.drop('fare_amount', axis=1).values.astype(np.float32)
y = df['fare_amount'].values.astype(np.float32)
print(f"Taxi rows: {len(y):,}")
```

### A2. Build & train NN

```python
taxi_model = Sequential([
    Dense(512, activation='relu', input_shape=(3,)),
    Dropout(0.1),
    Dense(256, activation='relu'),
    Dropout(0.1),
    Dense(1)
])
taxi_model.compile(optimizer='adam', loss='mae', metrics=['mae'])

taxi_hist = taxi_model.fit(
    X, y, validation_split=0.2, epochs=150, batch_size=100,
    callbacks=make_callbacks('taxi', 'val_mae', mode='min'),
    verbose=1
)
plot_history(taxi_hist, 'mae', 'Taxi Fare NN', 'a_taxi_curves.png')
```

### A3. Evaluate & save

```python
y_pred = taxi_model.predict(X, verbose=0).flatten()
nn_mae = mean_absolute_error(y, y_pred)
nn_r2 = r2_score(y, y_pred)
print(f"NN - MAE: ${nn_mae:.2f}, R²: {nn_r2:.4f}")

taxi_model.save(MODEL_DIR / 'taxi' / 'final.keras')
taxi_model.save(MODEL_DIR / 'taxi' / 'savedmodel')  # SavedModel dir

loaded = load_model(MODEL_DIR / 'taxi' / 'final.keras')
assert np.allclose(loaded.predict(X[:3], verbose=0), taxi_model.predict(X[:3], verbose=0))
```

### A4. Random forest baseline

```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
rf.fit(X_tr, y_tr)
rf_mae = mean_absolute_error(y_te, rf.predict(X_te))
rf_r2 = r2_score(y_te, rf.predict(X_te))
print(f"RF - MAE: ${rf_mae:.2f}, R²: {rf_r2:.4f}")
```

**Deliverable A:** Record MAE/R² for NN vs RF in your comparison table.

---

## Part B: Fraud Detection (90 min)

*Follow [Section 9.5](./section-05-binary-classification-credit-card-fraud-detection.md). Baseline: [Chapter 03](../chapter-03-classification-models/section-07-case-study-credit-card-fraud-detection.md).*

### B1. Load & split

```python
fraud_df = pd.read_csv(DATA_DIR / 'creditcard.csv')
X_f = fraud_df.drop(['Time', 'Class'], axis=1).values.astype(np.float32)
y_f = fraud_df['Class'].values.astype(np.float32)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_f, y_f, test_size=0.2, stratify=y_f, random_state=SEED)
print(f"Fraud rate train: {y_tr.mean():.4f}")
```

### B2. Train fraud NN

```python
fraud_model = Sequential([
    Dense(128, activation='relu', input_shape=(29,)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
fraud_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

fraud_hist = fraud_model.fit(
    X_tr, y_tr, validation_data=(X_te, y_te),
    epochs=30, batch_size=100,
    callbacks=make_callbacks('fraud', 'val_loss', mode='min'),
    verbose=1
)
plot_history(fraud_hist, 'loss', 'Fraud NN Loss', 'b_fraud_curves.png')
```

### B3. Metrics (NOT accuracy alone)

```python
y_prob = fraud_model.predict(X_te, verbose=0).flatten()
y_pred = (y_prob > 0.5).astype(int)

print(classification_report(y_te, y_pred, target_names=['Legit', 'Fraud']))
print(f"AUC-ROC: {roc_auc_score(y_te, y_prob):.4f}")

ConfusionMatrixDisplay.from_predictions(
    y_te, y_pred, display_labels=['Legitimate', 'Fraudulent'], cmap='Blues')
plt.savefig(OUT_DIR / 'b_fraud_cm.png', dpi=120, bbox_inches='tight')
plt.show()

fraud_model.save(MODEL_DIR / 'fraud' / 'final.keras')
```

### B4. Logistic regression baseline

```python
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])
lr.fit(X_tr, y_tr)
lr_prob = lr.predict_proba(X_te)[:, 1]
lr_auc = roc_auc_score(y_te, lr_prob)
print(f"Logistic AUC: {lr_auc:.4f}")
```

**Deliverable B:** Fraud recall, AUC vs logistic baseline, confusion matrix PNG.

---

## Part C: Face Recognition (2 hours)

*Follow [Sections 9.6](./section-06-multi-class-classification-face-recognition.md) & [9.7](./section-07-dropout-and-regularization.md). Baseline: [Chapter 05](../chapter-05-support-vector-machines/section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md).*

### C1. Load LFW, balance, split

```python
faces = fetch_lfw_people(min_faces_per_person=100, resize=0.4)
h, w = faces.images.shape[1], faces.images.shape[2]
input_dim = h * w
n_classes = len(faces.target_names)

mask = np.zeros(faces.target.shape, dtype=bool)
for t in np.unique(faces.target):
    mask[np.where(faces.target == t)[0][:100]] = True

X_face = faces.data[mask] / 255.0
y_face = faces.target[mask]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_face, y_face, test_size=0.2, stratify=y_face, random_state=SEED)
print(faces.target_names, X_tr.shape)
```

### C2. Train with dropout

```python
face_model = Sequential([
    Dense(512, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(n_classes, activation='softmax')
])
face_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])

face_hist = face_model.fit(
    X_tr, y_tr, validation_data=(X_te, y_te),
    epochs=100, batch_size=20,
    callbacks=make_callbacks('face', 'val_accuracy', mode='max'),
    verbose=1
)
plot_history(face_hist, 'accuracy', 'Face NN', 'c_face_curves.png')
```

### C3. Evaluate

```python
y_prob = face_model.predict(X_te, verbose=0)
y_pred = y_prob.argmax(axis=1)
face_acc = accuracy_score(y_te, y_pred)
print(f"Face NN accuracy: {face_acc:.3f}")
print(classification_report(y_te, y_pred, target_names=faces.target_names))

ConfusionMatrixDisplay.from_predictions(
    y_te, y_pred, display_labels=faces.target_names,
    xticks_rotation=90, cmap='Blues')
plt.savefig(OUT_DIR / 'c_face_cm.png', dpi=120, bbox_inches='tight')
plt.show()

face_model.save(MODEL_DIR / 'face' / 'final.keras')
```

### C4. SVM baseline (scaled pixels)

```python
svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED))
])
svm_pipe.fit(X_tr, y_tr)
svm_acc = accuracy_score(y_te, svm_pipe.predict(X_te))
print(f"SVM (RBF) accuracy: {svm_acc:.3f}")
```

**Deliverable C:** Train vs val accuracy gap before/after dropout (run once without dropout for comparison).

---

## Part D-E: Justification & Comparison (45 min)

Per task, document width/depth choice, dropout rate, whether early stopping fired, and NN vs baseline outcome ([Sections 9.3](./section-03-network-sizing-and-architecture.md), [9.7](./section-07-dropout-and-regularization.md)).

```python
results = pd.DataFrame([
    {'Task': 'Taxi', 'Metric': 'MAE ($)', 'NN': nn_mae, 'Baseline': rf_mae},
    {'Task': 'Fraud', 'Metric': 'AUC', 'NN': roc_auc_score(y_te, y_prob), 'Baseline': lr_auc},
    {'Task': 'Faces', 'Metric': 'Accuracy', 'NN': face_acc, 'Baseline': svm_acc},
])
results.to_csv(OUT_DIR / 'comparison_table.csv', index=False)
print(results)
```

**Deliverables:** curves (`a_taxi_curves.png`, `b_fraud_cm.png`, `c_face_cm.png`), saved models in `models/lab09/`, SavedModel at `models/lab09/taxi/savedmodel/`, architecture notes, single notebook `lab09_neural_triathlon.ipynb`.

---

**Previous:** [Section 9.8](./section-08-callbacks-training-control-and-model-persistence.md)  
**Next:** [Chapter 10 - Convolutional Neural Networks](../chapter-10-convolutional-neural-networks/README.md)