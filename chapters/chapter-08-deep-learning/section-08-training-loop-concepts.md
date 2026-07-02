# Section 8.8: Training Loop Concepts

> **Source:** Prosise, Ch. 8 - "Training the Network"  
> **Prerequisites:** [Sections 8.1-8.7](./section-07-activation-functions-deep-dive.md) | [Chapter 07](../chapter-07-operationalizing-models/README.md)  
> **Glossary:** [overfitting](../../GLOSSARY.md#overfitting) | [underfitting](../../GLOSSARY.md#underfitting) | [hyperparameter](../../GLOSSARY.md#hyperparameter)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Complete Training Loop

Prosise diagrams the cycle that every deep learning system repeats - from XOR in NumPy to face recognition in [Chapter 11](../chapter-11-face-detection-recognition/README.md):

```
┌─────────────────────────────────────────────────────────┐
│  1. Sample mini-batch (X_batch, y_batch)                │
│  2. Forward pass → predictions ŷ                        │
│  3. Compute loss L(ŷ, y)                              │
│  4. Backward pass → gradients ∂L/∂θ                     │
│  5. Update θ ← θ − η ∇L                                 │
│  6. Repeat until convergence or max epochs              │
└─────────────────────────────────────────────────────────┘
```

> **Readable form:** sample data → predict → measure error → compute gradients → nudge weights → repeat

[Chapter 09](../chapter-09-neural-networks/README.md) wraps this in `model.fit()`. This section explains what `fit()` actually does.

---

## Epochs

An **epoch** = one complete pass through the entire training set.

| Training set size | Batch size | Steps per epoch |
|-------------------|------------|-----------------|
| 10,000 | 100 | 100 |
| 60,000 (MNIST) | 128 | 469 |
| 500 | 32 | 16 |

More epochs = more learning opportunities - but also more [overfitting](../../GLOSSARY.md#overfitting) risk.

```python
num_epochs = 50
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for X_batch, y_batch in minibatch_iterator(X_train, y_train, batch_size=32):
        loss = train_step(X_batch, y_batch)
        epoch_loss += loss
    print(f"Epoch {epoch+1}: avg loss = {epoch_loss / n_batches:.4f}")
```

Same discipline as running multiple [cross-validation](../../GLOSSARY.md#cross-validation) folds in Part I - but here each epoch revisits the same data.

---

## Batches and Mini-Batches

| Term | Meaning |
|------|---------|
| **Batch GD** | Full dataset per gradient step |
| **Mini-batch** | Subset (typical 32-256) |
| **SGD** | Batch size 1 (rare in modern DL) |

**Why mini-batches?**

1. **Memory** - cannot fit ImageNet on one GPU forward pass
2. **Speed** - vectorized hardware utilization
3. **Generalization** - noisy gradients act as regularization

Batch size is a [hyperparameter](../../GLOSSARY.md#hyperparameter): small batches = noisier updates; large batches = smoother but may need learning rate tuning.

---

## Train / Validation / Test Split

[Chapter 07](../chapter-07-operationalizing-models/README.md) operationalized train/test discipline. Neural training adds **validation** during training:

| Set | Purpose | When used |
|-----|---------|-----------|
| **Train** | Update weights | Every mini-batch |
| **Validation** | Tune hyperparameters, early stopping | End of each epoch |
| **Test** | Final unbiased estimate | Once, after all choices |

```
Train loss   ↘
Val loss     ↘  then ↗ ← overfitting begins
              epoch →
```

Never tune epochs or learning rate on the **test** set - same leakage rules as [Section 2.7](../chapter-02-regression-models/section-07-model-comparison-and-cross-validation.md).

---

## Monitoring Curves

Healthy training:

- Train loss decreases steadily
- Val loss decreases, then may plateau
- Gap between train and val small

Unhealthy patterns:

| Pattern | Likely cause | Fix |
|---------|--------------|-----|
| Train ↓, val ↑ | Overfitting | Dropout, L2, more data ([Chapter 09](../chapter-09-neural-networks/README.md)) |
| Both flat high | Underfitting | Bigger network, train longer |
| Loss spikes / NaN | Learning rate too high | Reduce $\eta$, gradient clipping |
| Train loss noisy | Small batch | Increase batch size |

Log metrics the way [Chapter 07](../chapter-07-operationalizing-models/README.md) logs API latency - trends matter more than single points.

---

## Overfitting Preview

Neural networks with millions of parameters can **memorize** small datasets:

| Dataset | Risk |
|---------|------|
| [Titanic](../chapter-03-classification-models/section-06-case-study-titanic-survival.md) (~900 rows) | High if network is large |
| [MNIST](../chapter-03-classification-models/section-08-case-study-mnist-digit-recognition.md) (60k) | Moderate with small MLP |
| [Taxi](../chapter-02-regression-models/section-08-taxi-fare-prediction-end-to-end-project.md) (5k) | Compare to random forest baseline |

**Regularization toolkit** (Chapter 09 details):

- **Dropout** - randomly zero activations during training
- **L2 weight decay** - penalize large weights ([Ridge](../chapter-02-regression-models/section-02-linear-regression.md) analog)
- **Early stopping** - halt when val loss stops improving
- **Data augmentation** - CNNs, Chapter 10

---

## Hyperparameters vs Parameters

| Type | Examples | How set |
|------|----------|---------|
| **Parameters** | $\mathbf{W}, \mathbf{b}$ | Learned via gradient descent |
| **[Hyperparameters](../../GLOSSARY.md#hyperparameter)** | Learning rate, epochs, batch size, layer sizes | You choose (grid search, intuition) |

Same distinction as $k$ in [k-NN](../chapter-01-machine-learning/section-05-supervised-learning-k-nearest-neighbors.md) vs coefficients in [linear regression](../chapter-02-regression-models/section-02-linear-regression.md).

---

## Initialization Matters

Bad starting weights break training before epoch 1:

| Init | Rule of thumb |
|------|---------------|
| **Zeros** | All neurons identical - never use for weights |
| **Too large** | Saturated sigmoids, exploding activations |
| **Xavier / Glorot** | For sigmoid/tanh |
| **He** | For ReLU - default mental model |

```python
W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)  # He for ReLU
```

Keras handles this automatically in `Dense` layers.

---

## NumPy Training Loop: XOR (Complete Sketch)

```python
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

n_in, n_hid = 2, 4
rng = np.random.default_rng(8)
W1 = rng.normal(0, 0.5, (n_hid, n_in))
b1 = np.zeros(n_hid)
W2 = rng.normal(0, 0.5, (1, n_hid))
b2 = np.zeros(1)

def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z): return np.maximum(0, z)

lr, epochs = 0.5, 5000

for epoch in range(epochs):
    # Forward
    Z1 = X @ W1.T + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2.T + b2
    A2 = sigmoid(Z2)

    # BCE loss
    eps = 1e-9
    loss = -np.mean(y*np.log(A2+eps) + (1-y)*np.log(1-A2+eps))

    # Backward (simplified manual)
    dZ2 = A2 - y
    dW2 = dZ2.T @ A1 / len(X)
    db2 = dZ2.mean(axis=0)
    dA1 = dZ2 @ W2
    dZ1 = dA1 * (Z1 > 0)
    dW1 = dZ1.T @ X / len(X)
    db1 = dZ1.mean(axis=0)

    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1

    if epoch % 1000 == 0:
        preds = (A2 > 0.5).astype(int)
        acc = (preds == y).mean()
        print(f"epoch {epoch}: loss={loss:.4f}, acc={acc:.2%}")
```

Full commented version in [Lab 08](./section-lab-08-neural-network-intuition-workshop.md).

---

## Preparing for Keras (Chapter 09)

| NumPy concept | Keras equivalent |
|---------------|------------------|
| `forward()` | Model call / `predict()` |
| `compute_loss()` | `compile(loss=...)` |
| `backward()` + update | `compile(optimizer=...)` + `fit()` |
| Epoch loop | `epochs=` argument |
| Mini-batch | `batch_size=` argument |
| Validation split | `validation_split=` or `validation_data=` |

```python
# Preview only - full treatment in Chapter 09
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(12,)),
    layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)
```

You will rebuild [taxi](../chapter-02-regression-models/section-08-taxi-fare-prediction-end-to-end-project.md), [fraud](../chapter-03-classification-models/section-07-case-study-credit-card-fraud-detection.md), and face pipelines with this API.

---

## Reproducibility

Set seeds for debugging - same as Part I labs:

```python
import numpy as np
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)
```

Production models in [Chapter 07](../chapter-07-operationalizing-models/README.md) should log hyperparameters and data versions alongside weights.

---

## Scaling Inputs (Non-Negotiable)

Neural nets expect roughly standardized inputs:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
```

Same rationale as [SVM Chapter 05](../chapter-05-support-vector-machines/README.md) and [PCA Chapter 06](../chapter-06-principal-component-analysis/README.md).

---

## Decision Checklist Before `model.fit()`

1. ✓ Forward pass shapes verified ([Section 8.4](./section-04-forward-propagation.md))
2. ✓ Loss matches task ([Section 8.5](./section-05-loss-functions.md))
3. ✓ Output activation paired correctly ([Section 8.7](./section-07-activation-functions-deep-dive.md))
4. ✓ Data scaled; train/val/test split defined
5. ✓ Baseline from Part I documented ([Chapter 02](../chapter-02-regression-models/README.md)-[03](../chapter-03-classification-models/README.md))
6. ✓ Monitoring plan for val curves and deployment ([Chapter 07](../chapter-07-operationalizing-models/README.md))

---

## Common Mistakes (Section 8.8)

| Mistake | Why it hurts | Fix |
|---------|--------------|-----|
| **Training without validation** | Overfit silently | `validation_split=0.2` |
| **Tuning on test set** | Optimistic metrics | Hold test until final evaluation |
| **Too many epochs, no early stop** | Wasted compute + overfit | EarlyStopping callback |
| **Unscaled features** | Slow or failed training | StandardScaler in pipeline |
| **Skipping Part I baseline** | Cannot justify neural net complexity | Compare RMSE/AUC first |

---

## Self-Check

1. Difference between epoch and batch?
2. When does overfitting appear on learning curves?
3. Name three hyperparameters and three parameters.
4. What Keras function replaces your NumPy epoch loop?

---

## Further Reading

- Prosise, Ch. 8 - training loop and convergence
- [Chapter 09 - Neural Networks with Keras](../chapter-09-neural-networks/README.md)
- [Lab 08 - Intuition Workshop](./section-lab-08-neural-network-intuition-workshop.md)
- [Chapter 07 - Operationalizing Models](../chapter-07-operationalizing-models/README.md)

---

**Previous:** [Section 8.7 - Activation Functions](./section-07-activation-functions-deep-dive.md) | **Next:** [Chapter 09 - Neural Networks](../chapter-09-neural-networks/README.md)