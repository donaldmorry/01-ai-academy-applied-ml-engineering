# Section 9.3: Network Sizing & Architecture

> **Source:** Prosise, Ch. 9 - "Sizing a Neural Network"  
> **Prerequisites:** [Section 9.2](./section-02-your-first-keras-model.md)  
> **Glossary:** [neural-network](../../GLOSSARY.md#neural-network) | [overfitting](../../GLOSSARY.md#overfitting) | [hyperparameter](../../GLOSSARY.md#hyperparameter)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Design Question

Prosise asks the question every practitioner faces:

> *How do you pick the right number of layers and neurons?*

**Short answer:** minimum width and depth that achieve acceptable **validation** performance - found by intuition plus experimentation.

**Long answer:** this section.

---

## Anatomy of a Feedforward Network

A multilayer perceptron (MLP) is characterized by:

| Property | Symbol / term | Example (taxi fare) |
|----------|---------------|---------------------|
| Depth | Number of hidden layers | 2 |
| Width | Neurons per layer | 512, 512 |
| Layer type | `Dense`, `Dropout`, `Conv2D` | `Dense` for tabular |
| Activations | `relu`, `sigmoid`, `softmax` | `relu` hidden, linear output |

```
Input (3 features) → Dense(512, relu) → Dense(512, relu) → Dense(1) → fare $
```

---

## Counting Trainable Parameters

For a `Dense` layer connecting $n_{\text{in}}$ inputs to $n_{\text{out}}$ neurons:

$$
\text{params} = n_{\text{in}} \times n_{\text{out}} + n_{\text{out}}
$$
> **Readable form:** weights = every input-to-neuron connection; plus one bias per output neuron

**Prosise's taxi network** (3 → 512 → 512 → 1):

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(512, activation='relu', input_shape=(3,)),
    Dense(512, activation='relu'),
    Dense(1)
])
model.summary()
# Layer 1: 3×512 + 512 = 2,048
# Layer 2: 512×512 + 512 = 262,656
# Layer 3: 512×1 + 1 = 513
# Total: ~265,217 trainable parameters
```

> **Humorous analogy:** A quarter-million knobs for predicting taxi fares. Gradient boosting would use a few hundred trees with far fewer effective degrees of freedom - which is why NN ≠ automatic win on tabular data.

---

## Prosise's Sizing Guidelines

### 1. Prefer width over depth (tabular data)

| Guideline | Rationale |
|-----------|-----------|
| Wider layers > deeper stacks | Fewer vanishing-gradient issues; faster training |
| ReLU helps depth but isn't magic | Still prefer 1-2 hidden layers for MLPs |
| 1 hidden layer often enough | Universal approximator for smooth functions |

**Connection count example (Prosise):** 5 layers × 20 neurons = many inter-layer connections ($20^2 \times 4 = 1{,}600$) vs one layer of 100 neurons (no inter-layer connections within the stack).

### 2. Start big, shrink until validation drops

Prosise's workflow:

1. Start with **1-2 hidden layers of 512 neurons**
2. Train and record validation metric
3. **Halve width** (512 → 256 → 128) or remove a layer
4. Stop when validation accuracy/MAE crosses your threshold

```python
def build_mlp(input_dim, hidden_units, n_hidden_layers=2):
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu', input_shape=(input_dim,)))
    for _ in range(n_hidden_layers - 1):
        model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model

for units in [512, 256, 128, 64, 16]:
    m = build_mlp(3, units, n_hidden_layers=2)
    print(f"\n=== {units} neurons × 2 layers ===")
    m.summary()
```

### 3. Fewer neurons → faster training

State-of-the-art models train for days on GPUs. For your tabular problems, training time still matters in hyperparameter sweeps.

---

## The 16-Neuron Experiment

Prosise challenges readers: collapse to **one hidden layer of 16 neurons** and check $R^2$.

```python
tiny = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),
    Dense(1)
])
tiny.compile(optimizer='adam', loss='mae', metrics=['mae'])
tiny.summary()  # Only ~65 parameters
```

| Architecture | Params | Typical val MAE (taxi) |
|--------------|--------|------------------------|
| 512-512 | ~265K | ~2.25 |
| 128-128 | ~17K | Often similar |
| 16 (one layer) | ~65 | Much worse - underfitting |

**Section:** capacity must match problem complexity. Taxi fares are nonlinear but low-dimensional - you don't need a million parameters, but 16 is too few.

---

## Input and Output Shape Rules

| Problem type | Input shape | Output layer | Activation |
|--------------|-------------|--------------|------------|
| Regression | `(n_features,)` | 1 neuron | None |
| Binary classification | `(n_features,)` | 1 neuron | `sigmoid` |
| Multi-class (K classes) | `(n_features,)` | K neurons | `softmax` |
| Flattened images | `(H*W,)` or `(H, W, 1)` | K neurons | `softmax` |

```python
n_features = 29   # fraud dataset
n_classes = 5     # LFW subset

# Regression
reg_out = Dense(1)

# Binary
bin_out = Dense(1, activation='sigmoid')

# Multi-class
mc_out = Dense(n_classes, activation='softmax')
```

---

## Matching Loss to Output

| Task | Loss | Metrics |
|------|------|---------|
| Regression | `mae`, `mse` | `mae`, `mse` |
| Binary | `binary_crossentropy` | `accuracy`, `AUC` (custom) |
| Multi-class (integer labels) | `sparse_categorical_crossentropy` | `accuracy` |
| Multi-class (one-hot) | `categorical_crossentropy` | `accuracy` |

Mismatching output activation and loss is a top beginner bug.

---

## Vanishing Gradients (Why Depth Hurts)

For sigmoid/tanh-heavy deep networks, gradients shrink as they backprop through layers:

$$
\frac{\partial L}{\partial w^{(1)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \prod_{l=2}^{L} \sigma'(z^{(l)}) \cdot \ldots
$$
> **Readable form:** early layers receive tiny gradient signals when many saturated sigmoids multiply together

**ReLU** mitigates this ($\text{ReLU}'(z) = 1$ for $z > 0$), which is why it's the default hidden activation. Still, for tabular MLPs, **1-2 hidden layers** is Prosise's practical ceiling.

---

## Architecture Search Template

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_and_score(hidden_units, n_layers, X, y, epochs=30, seed=42):
    tf.random.set_seed(seed)
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu', input_shape=(X.shape[1],)))
    for _ in range(n_layers - 1):
        model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    hist = model.fit(X, y, validation_split=0.2, epochs=epochs,
                     batch_size=100, verbose=0)
    return hist.history['val_mae'][-1]

# Run multiple seeds - Prosise: embrace randomness, average results
configs = [(512, 2), (256, 2), (128, 1), (64, 1)]
for units, layers in configs:
    scores = [train_and_score(units, layers, X, y, seed=s) for s in range(3)]
    print(f"{units}×{layers}: val_mae = {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

---

## When Neural Networks Lose to Classical ML

| Situation | Better alternative |
|-----------|-------------------|
| Small tabular dataset (< 10K rows) | Random forest, XGBoost |
| Interpretability required | Linear/logistic regression |
| Heavy feature engineering already done | Gradient boosting |
| High signal-to-noise, few features | Linear models |

Neural networks shine when you have **lots of data**, **raw high-dimensional inputs** (images, text, audio), or need **end-to-end learning**. Chapter 09 revisits Part I problems partly to show **when NNs match or beat** baselines - not always.

---

## Decision Checklist

Before training, document:

1. **Input dimension** after feature engineering
2. **Output type** - regression / binary / multi-class
3. **Starting architecture** - e.g., 2×512 Dense
4. **Shrink strategy** - halve width until val metric degrades
5. **Regularization plan** - [dropout](./section-07-dropout-and-regularization.md), early stopping ([Section 9.8](./section-08-callbacks-training-control-and-model-persistence.md))
6. **Baseline to beat** - Chapter 02 RF for taxi, Chapter 03 logistic for fraud

---

## Self-Check

1. How many trainable parameters in `Dense(128, input_shape=(50,))` followed by `Dense(10)`?
2. Why does Prosise prefer 512×1 over 20×5 for the same neuron count?
3. What happens when you use 16 neurons on taxi fare data?
4. How many output neurons for 5-class face recognition?

---

## What's Next

[Section 9.4](./section-04-regression-with-neural-networks-taxi-fare-prediction.md) applies these rules to the NYC taxi fare dataset from [Chapter 02](../chapter-02-regression-models/section-08-taxi-fare-prediction-end-to-end-project.md) - full pipeline, MAE curves, and $R^2$ comparison.

---

**Previous:** [Section 9.2](./section-02-your-first-keras-model.md)  
**Next:** [Section 9.4 - Regression with Neural Networks](./section-04-regression-with-neural-networks-taxi-fare-prediction.md)
