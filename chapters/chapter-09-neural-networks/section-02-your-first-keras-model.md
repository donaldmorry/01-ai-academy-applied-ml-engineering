# Section 9.2: Your First Keras Model

> **Source:** Prosise, Ch. 9 - "Building Neural Networks with Keras and TensorFlow"  
> **Prerequisites:** [Section 9.1](./section-01-keras-and-tensorflow-setup.md)  
> **Glossary:** [neural-network](../../GLOSSARY.md#neural-network) | [epoch](../../GLOSSARY.md#epoch) | [keras](../../GLOSSARY.md#keras)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Three-Step Recipe

Every Keras model follows the same lifecycle Prosise demonstrates in Chapter 9:

```
1. Define  → Sequential + Dense layers
2. Compile → optimizer + loss + metrics
3. Fit     → epochs, batch_size, validation
```

> **In plain English:** Stack layers, tell Keras how to measure success, then show it data until the error stops falling.

---

## Step 1: Define the Architecture

Recreate the Chapter 8 three-layer perceptron - 2 inputs, 3 hidden neurons (ReLU), 1 linear output:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

tf.random.set_seed(42)

model = Sequential()
model.add(Dense(3, activation='relu', input_dim=2))  # hidden layer
model.add(Dense(1))                                   # output - regression, no activation

model.summary()
```

**Key details from Prosise:**

- You rarely add an explicit input layer - `input_dim=2` on the first `Dense` creates it implicitly
- **ReLU** (`activation='relu'`) in hidden layers: $\text{ReLU}(z) = \max(0, z)$
- Output layer for **regression** has **no activation** - raw linear prediction

$$
\hat{y} = w_3 \cdot \text{ReLU}(w_2 \cdot \text{ReLU}(w_1 \mathbf{x} + b_1) + b_2) + b_3
$$
> **Readable form:** inputs pass through two ReLU transformations, then a final weighted sum produces the prediction

### Modern list syntax (equivalent)

```python
model = Sequential([
    Dense(3, activation='relu', input_shape=(2,)),
    Dense(1)
])
```

---

## Step 2: Compile

```python
model.compile(
    optimizer='adam',
    loss='mae',
    metrics=['mae']
)
```

| Argument | Role | Regression choice | Classification (preview) |
|----------|------|-------------------|--------------------------|
| `optimizer` | Updates weights via backprop | `'adam'` (Prosise's default) | `'adam'` |
| `loss` | What to minimize | `'mae'` or `'mse'` | `'binary_crossentropy'` / `'sparse_categorical_crossentropy'` |
| `metrics` | Logged each epoch (not used for gradients) | `['mae']` | `['accuracy']` |

**String shortcuts:** `optimizer='adam'` equals `optimizer=keras.optimizers.Adam()`. Use the long form to customize:

```python
from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=2e-5),  # fine-tuning rate - Ch. 13 preview
    loss='mae',
    metrics=['mae']
)
```

Inside `compile()`, Keras builds a TensorFlow computation graph optimized for training.

---

## Step 3: Fit

```python
# Synthetic regression data for demonstration
np.random.seed(42)
n = 1000
X = np.random.randn(n, 2).astype(np.float32)
y = (3 * X[:, 0] - 2 * X[:, 1] + 1 + 0.1 * np.random.randn(n)).astype(np.float32)

history = model.fit(
    X, y,
    epochs=100,
    batch_size=100,
    validation_split=0.2,
    verbose=1
)
```

### `fit()` parameters explained

| Parameter | Meaning |
|-----------|------|
| `epochs` | Full passes through training data |
| `batch_size` | Samples per gradient update before backprop |
| `validation_split=0.2` | Last 20% of data for validation each epoch |
| `validation_data=(X_val, y_val)` | Alternative: explicit validation set |
| `verbose` | 0=silent, 1=progress bar, 2=one line per epoch |

**Batch size tradeoff (Prosise):** larger batches → faster epochs, not always better accuracy. Experiment; don't assume smaller is always better.

**`validation_split` caveat:** takes the **last** fraction without shuffling. For ordered time-series, shuffle first or use `train_test_split`.

---

## Reading the History Object

```python
hist = history.history
print(hist.keys())  # dict: 'loss', 'mae', 'val_loss', 'val_mae'

epochs_range = range(1, len(hist['mae']) + 1)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(epochs_range, hist['mae'], '-', label='Training MAE')
plt.plot(epochs_range, hist['val_mae'], ':', label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Training vs Validation - Prosise Fig. 9-1 style')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Interpreting curves

| Pattern | Diagnosis |
|---------|-----------|
| Both loss ↓, val ≈ train | Healthy learning |
| Train ↓, val flat or ↑ | [Overfitting](../../GLOSSARY.md#overfitting) - [Section 9.7](./section-07-dropout-and-regularization.md) |
| Both high, flat early | [Underfitting](../../GLOSSARY.md#underfitting) - more capacity or epochs |
| Val better than train | Unusual; check data leakage or tiny val set |

> **Prosise's rule:** You care about **validation** metrics - that's performance on unseen data.

---

## Predict

```python
sample = np.array([[2.0, 2.0]], dtype=np.float32)
prediction = model.predict(sample, verbose=0)
print(f"Predicted: {prediction[0, 0]:.4f}")

# Batch prediction
batch_pred = model.predict(X[:5], verbose=0)
```

`predict()` returns a NumPy array. Shape `(n_samples, output_dim)`.

---

## Loss Functions for Regression

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$
> **Readable form:** MAE = average absolute dollar (or unit) error - robust to outliers

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$
> **Readable form:** MSE = average squared error - punishes large mistakes harder

```python
model_mse = Sequential([Dense(3, activation='relu', input_shape=(2,)), Dense(1)])
model_mse.compile(optimizer='adam', loss='mse', metrics=['mae'])
# loss drives gradients; metrics are for human-readable logging
```

---

## Activation Functions Reference

| Layer | Common activation | Why |
|-------|-------------------|-----|
| Hidden | `relu` | Fast, avoids vanishing gradients (mostly) |
| Hidden (rare) | `tanh` | Zero-centered; older architectures |
| Binary output | `sigmoid` | Probability in $(0, 1)$ - [Section 9.5](./section-05-binary-classification-credit-card-fraud-detection.md) |
| Multi-class output | `softmax` | Probabilities sum to 1 - [Section 9.6](./section-06-multi-class-classification-face-recognition.md) |
| Regression output | **none** | Unbounded real values |

---

## Mini-Batch Gradient Descent

Prosise connects `batch_size` to **mini-batch gradient descent (MBGD)**:

$$
\theta \leftarrow \theta - \eta \cdot \frac{1}{B}\sum_{i \in \text{batch}} \nabla_\theta L(y_i, f_\theta(x_i))
$$
> **Readable form:** after each batch of B samples, nudge all weights by a small step opposite to the average gradient

Full-batch (entire dataset) is accurate but slow. Stochastic (batch=1) is noisy. **Mini-batch** (30-256 typical) balances speed and stability.

---

## Complete Runnable Example

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

tf.random.set_seed(42)
np.random.seed(42)

# Data
X = np.random.randn(500, 2).astype(np.float32)
y = (2*X[:,0] + X[:,1]).astype(np.float32)

# Model
model = Sequential([
    Dense(8, activation='relu', input_shape=(2,)),
    Dense(4, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mae', metrics=['mae'])

# Train
hist = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate
val_mae = hist.history['val_mae'][-1]
print(f"Final validation MAE: {val_mae:.4f}")

# Predict
print(model.predict(np.array([[1.0, 1.0]]), verbose=0))
```

---

## Common Errors

| Error | Fix |
|-------|-----|
| `Input 0 of layer dense is incompatible` | Feature count ≠ `input_dim` / `input_shape` |
| `Unknown loss` | Typo - use `'mae'`, `'mse'`, `'binary_crossentropy'` |
| NaN loss | Learning rate too high, unnormalized inputs, bad data |
| Val loss = 0 instantly | Label leakage in features |

---

## Self-Check

1. What three arguments does `compile()` require?
2. What does `validation_split=0.2` do differently from `train_test_split`?
3. Why is ReLU preferred in hidden layers over sigmoid?
4. What object does `fit()` return, and what is inside it?

---

## What's Next

[Section 9.3](./section-03-network-sizing-and-architecture.md) answers the design question Prosise poses next: how many layers, how many neurons, and why width often beats depth for tabular data.

---

**Previous:** [Section 9.1](./section-01-keras-and-tensorflow-setup.md)  
**Next:** [Section 9.3 - Network Sizing & Architecture](./section-03-network-sizing-and-architecture.md)
