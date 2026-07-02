# Section 9.5: Binary Classification - Credit Card Fraud Detection

> **Source:** Prosise, Ch. 9 - "Binary Classification with Neural Networks" & "Training a Neural Network to Detect Credit Card Fraud"  
> **Prerequisites:** [Section 9.4](./section-04-regression-with-neural-networks-taxi-fare-prediction.md) | [Chapter 03 Section 3.7](../chapter-03-classification-models/section-07-case-study-credit-card-fraud-detection.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [accuracy](../../GLOSSARY.md#accuracy) | [overfitting](../../GLOSSARY.md#overfitting)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Regression → Binary: Two Changes

Prosise's recipe for converting a regression network to a **binary classifier**:

| Component | Regression (taxi) | Binary classification (fraud) |
|-----------|-------------------|-------------------------------|
| Output layer | `Dense(1)` - no activation | `Dense(1, activation='sigmoid')` |
| Loss | `mae` / `mse` | `binary_crossentropy` |
| Metrics | `mae` | `accuracy` (plus recall/AUC manually) |
| Prediction | Raw dollar value | Probability in $(0, 1)$ |

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(512, activation='relu', input_shape=(3,)),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')  # ← the two key changes start here
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

> **In plain English:** Sigmoid squashes the output to a probability. Binary cross-entropy punishes confident wrong answers exponentially hard.

---

## Binary Cross-Entropy (Log Loss)

For true label $y \in \{0, 1\}$ and predicted probability $\hat{p}$:

$$
L = -\big[y \log(\hat{p}) + (1-y)\log(1-\hat{p})\big]
$$
> **Readable form:** if the true label is 1 and you predict 0.9, loss is small (−log 0.9 ≈ 0.04). If you predict 0.0001, loss is huge (−log 0.0001 = 4).

**Prosise's intuition:** cross-entropy pats the optimizer on the back when close, **slaps** it when far wrong - driving weights aggressively toward correct probabilities.

---

## Nonlinear Demo (Prosise Fig. 9-3)

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X = X.astype(np.float32)
y = y.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                 epochs=40, batch_size=10, verbose=0)
```

### Predicting class labels

```python
prob = model.predict(np.array([[-0.5, 0.0]], dtype=np.float32), verbose=0)
print(f"Probability positive class: {prob[0, 0]:.3f}")

# Threshold at 0.5
label = (prob > 0.5).astype('int32')
print(f"Predicted class: {label[0, 0]}")
```

`predict_classes` was removed in modern Keras - use threshold or `np.argmax` for multi-class.

---

## Fraud Dataset Setup

Same [ULB credit card fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset from [Section 3.7](../chapter-03-classification-models/section-07-case-study-credit-card-fraud-detection.md):

$$
\text{Fraud rate} = \frac{492}{284{,}808} \approx 0.17\%
$$
> **Readable form:** fewer than 2 frauds per 1,000 transactions - accuracy is a trap

```python
import pandas as pd

df = pd.read_csv('data/creditcard.csv')
print(df.shape)  # (284807, 31)
print(df['Class'].value_counts())
```

| Column | Description |
|--------|-------------|
| `Time` | Seconds since first transaction (drop for modeling) |
| `Amount` | Transaction amount |
| `V1`-`V28` | PCA-transformed features |
| `Class` | 0 = legitimate, 1 = fraud |

---

## Train/Test Split (Explicit - Not `validation_split`)

Prosise splits manually for a clean confusion matrix on held-out data:

```python
from sklearn.model_selection import train_test_split

X = df.drop(['Time', 'Class'], axis=1).values.astype(np.float32)
y = df['Class'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train fraud rate: {y_train.mean():.4f}")
```

**Three-way split (best practice):** train / validation / test. Prosise uses test data as `validation_data` during training - acceptable because validation doesn't update weights, but a third holdout is safer for final reporting.

---

## Fraud Detection Network

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

tf.random.set_seed(0)

model = Sequential([
    Dense(128, activation='relu', input_dim=29),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=100,
    verbose=1
)
```

Smaller network than taxi (128 vs 512) - 29 PCA features, different problem scale.

---

## Plot Accuracy - Read Carefully

```python
import matplotlib.pyplot as plt

hist = history.history
epochs = range(1, len(hist['accuracy']) + 1)

plt.figure(figsize=(8, 4))
plt.plot(epochs, hist['accuracy'], '-', label='Training accuracy')
plt.plot(epochs, hist['val_accuracy'], ':', label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Fraud NN - Accuracy Curves (Misleading Alone!)')
plt.legend(loc='lower right')
plt.show()
```

**Val accuracy ≈ 0.9994 looks amazing - it's not.** A model predicting "all legitimate" scores ~99.83% [accuracy](../../GLOSSARY.md#accuracy).

---

## Confusion Matrix - The Truth

```python
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             roc_auc_score, precision_recall_curve)

y_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_prob > 0.5).astype(int)

print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=['Legitimate', 'Fraudulent'],
    cmap='Blues', xticks_rotation='vertical'
)
plt.show()

print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
```

**Prosise's typical result:**
- Legitimate: ~99.99% correct (6 misclassified out of ~57K)
- Fraud caught: ~73%+ recall
- Business logic: banks prefer false negatives (missed fraud) over false positives (declined legit charges) - but not infinitely

---

## Class Weights for Imbalance

Without resampling, tell `fit()` to care more about fraud:

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=100,
    class_weight={0: 1.0, 1: 0.01}  # Prosise experiment - penalize legit errors more
)
```

| `class_weight` effect | Legitimate errors | Fraud recall |
|-----------------------|-------------------|--------------|
| Default (balanced implicitly by frequency) | Low | Moderate (~73%) |
| `{0: 1.0, 1: 0.01}` | Near zero | **Decreases** |

> **Prosise's point:** `{0: 1.0, 1: 0.01}` means you care **more** about legit classification - fraud recall drops. Tune weights to match business cost matrix, not maximize accuracy.

Better starting point for catching fraud:

```python
from sklearn.utils.class_weight import compute_class_weight

weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = {0: weights[0], 1: weights[1]}
print(class_weight)
```

---

## Compare to Chapter 03 Logistic Regression

```python
# Baseline from Chapter 03 - illustrative
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, class_weight='balanced'))
])
lr_pipe.fit(X_train, y_train)
lr_prob = lr_pipe.predict_proba(X_test)[:, 1]

print(f"Logistic AUC: {roc_auc_score(y_test, lr_prob):.4f}")
print(f"NN AUC:       {roc_auc_score(y_test, y_prob):.4f}")
```

Neural networks can capture nonlinear fraud patterns in PCA space - but logistic regression with `class_weight='balanced'` is a fierce baseline.

---

## Threshold Tuning

Default 0.5 is rarely optimal for fraud:

```python
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Pick threshold for minimum recall target
target_recall = 0.80
idx = np.argmax(recalls >= target_recall)
chosen_thresh = thresholds[idx] if idx < len(thresholds) else 0.5
y_pred_tuned = (y_prob >= chosen_thresh).astype(int)
print(f"Threshold: {chosen_thresh:.3f}")
print(classification_report(y_test, y_pred_tuned))
```

---

## Self-Check

1. What two changes convert a regression NN to binary classification?
2. Why is 99.94% accuracy suspicious on this dataset?
3. What does `class_weight={0: 1.0, 1: 0.01}` prioritize?
4. Which metric should you report to stakeholders instead of accuracy?

---

## What's Next

[Section 9.6](./section-06-multi-class-classification-face-recognition.md) swaps sigmoid for **softmax** and trains on the LFW face dataset from [Chapter 05](../chapter-05-support-vector-machines/section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md).

---

**Previous:** [Section 9.4](./section-04-regression-with-neural-networks-taxi-fare-prediction.md)  
**Next:** [Section 9.6 - Multi-Class Classification: Face Recognition](./section-06-multi-class-classification-face-recognition.md)
