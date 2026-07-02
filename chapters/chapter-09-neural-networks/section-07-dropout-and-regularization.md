# Section 9.7: Dropout & Regularization

> **Source:** Prosise, Ch. 9 - "Dropout"  
> **Prerequisites:** [Section 9.6](./section-06-multi-class-classification-face-recognition.md)  
> **Glossary:** [dropout](../../GLOSSARY.md#dropout) | [overfitting](../../GLOSSARY.md#overfitting) | [regularization](../../GLOSSARY.md#regularization)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Generalization Gap

After training the LFW face network ([Section 9.6](./section-06-multi-class-classification-face-recognition.md)), Prosise observes a familiar pattern:

| Metric | Typical value |
|--------|---------------|
| Training accuracy | → ~100% |
| Validation accuracy | peaks ~80-85% |

**Gap = overfitting.** The network memorized training faces but struggles on unseen ones.

> **Prosise's rule:** Training accuracy is vanity; validation accuracy is sanity.

---

## Two Ways to Reduce Overfitting

### 1. Reduce capacity

Fewer layers, fewer neurons → fewer trainable parameters → harder to memorize.

```python
# Smaller network from Section 9.3
model_small = Sequential([
    Dense(128, activation='relu', input_shape=(2914,)),
    Dense(5, activation='softmax')
])
```

### 2. Add [dropout](../../GLOSSARY.md#dropout)

**Dropout** randomly disables a fraction of connections during each training step, forcing the network to learn redundant representations that generalize.

> **Prosise's analogy:** Read a book but skip every other page - you learn high-level concepts without memorizing every sentence.

Introduced in the 2014 paper *"Dropout: A Simple Way to Prevent Neural Networks from Overfitting"* (Srivastava et al.).

---

## Dropout in Keras

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

input_dim = 2914
class_count = 5

model = Sequential([
    Dense(512, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),  # drop 20% of connections each training step
    Dense(class_count, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
```

**`Dropout(0.2)`** - during training, each neuron in the preceding layer has a 20% chance of being temporarily ignored on each forward/backward pass.

### Train and compare

```python
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

# ... load LFW, balance, split (see Section 9.6) ...

def build_face_model(dropout_rate=0.0):
    layers = [Dense(512, activation='relu', input_shape=(input_dim,))]
    if dropout_rate > 0:
        layers.append(Dropout(dropout_rate))
    layers.append(Dense(class_count, activation='softmax'))
    m = Sequential(layers)
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

for rate in [0.0, 0.2, 0.4, 0.5]:
    tf.random.set_seed(0)
    m = build_face_model(rate)
    hist = m.fit(X_train, y_train, validation_data=(X_test, y_test),
                 epochs=100, batch_size=20, verbose=0)
    print(f"dropout={rate}: train={hist.history['accuracy'][-1]:.3f}, "
          f"val={hist.history['val_accuracy'][-1]:.3f}")
```

**Prosise's result:** gap may not fully close, but validation accuracy sometimes **improves**. Only experimentation tells.

---

## Dropout Rate Guidelines

| Rate | Effect |
|------|--------|
| 0.1-0.2 | Mild regularization - start here |
| 0.3-0.4 | Moderate - common for large Dense layers |
| 0.5 | Aggressive - may need **more epochs** to converge |
| > 0.5 | Diminishing returns; training slows without gain |

> **Humorous analogy:** Dropout 0.5 is studying for the exam by reading every other word - you'll need more study time (epochs) to pass.

**Placement options:**
- After each hidden layer (common)
- Only before output layer (Prosise's face example)
- Between multiple hidden layers

```python
model = Sequential([
    Dense(512, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(class_count, activation='softmax')
])
```

---

## Dropout Is OFF During Inference

At **training time:** random masks applied.  
At **inference** (`predict`, `evaluate`): **all neurons active**, outputs scaled to compensate for training-time dropout.

Keras handles this automatically - no code change at prediction time.

```python
# Same model.predict() - dropout disabled internally
y_pred = model.predict(X_test, verbose=0)
```

---

## L2 Regularization (Weight Decay)

Complement dropout with **L2 penalty** on weights - pushes weights toward zero:

$$
L_{\text{total}} = L_{\text{data}} + \lambda \sum_{l} \|\mathbf{W}^{(l)}\|_2^2
$$
> **Readable form:** total loss = prediction error plus a penalty for large weights. λ controls strength.

```python
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

model = Sequential([
    Dense(512, activation='relu', input_shape=(input_dim,),
          kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.2),
    Dense(class_count, activation='softmax',
          kernel_regularizer=regularizers.l2(0.001))
])
```

| Technique | What it does |
|-----------|--------------|
| Dropout | Randomly drops activations - ensemble effect |
| L2 | Shrinks weights - smoother decision boundaries |
| Smaller network | Less capacity to memorize |
| Early stopping | Stop before overfitting worsens - [Section 9.8](./section-08-callbacks-training-control-and-model-persistence.md) |

---

## Dropout on Tabular Models (Taxi & Fraud)

```python
# Taxi fare regression
taxi_model = Sequential([
    Dense(256, activation='relu', input_shape=(3,)),
    Dropout(0.1),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(1)
])
taxi_model.compile(optimizer='adam', loss='mae', metrics=['mae'])

# Fraud binary
fraud_model = Sequential([
    Dense(128, activation='relu', input_shape=(29,)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
fraud_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

Tabular datasets often need **less** dropout than image models - start low (0.1-0.2).

---

## Diagnosing with Learning Curves

```python
import matplotlib.pyplot as plt

def plot_train_val(history, metric='accuracy'):
    epochs = range(1, len(history.history[metric]) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history.history[metric], '-', label=f'Train {metric}')
    plt.plot(epochs, history.history[f'val_{metric}'], ':', label=f'Val {metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.title(f'With dropout - {metric}')
    plt.show()

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                 epochs=100, batch_size=20, verbose=0)
plot_train_val(hist, 'accuracy')
```

**Healthy sign after dropout:** validation curve rises or holds while train accuracy grows more slowly.

---

## When NOT to Use Dropout

| Situation | Reason |
|-----------|--------|
| Already underfitting | Dropout reduces capacity further |
| BatchNorm + very small batches | Interaction can hurt - advanced topic |
| Final layer before regression output | Dropout before scalar output rarely helps |
| Tiny datasets (< 500 samples) | May need simpler model, not more regularization |

---

## Experiment Protocol

1. Train **baseline** without dropout - record val metric
2. Add `Dropout(0.2)` - retrain same epochs
3. Try rates `[0.1, 0.2, 0.3, 0.4]` - 3 random seeds each
4. Add L2 if gap persists
5. Combine best dropout + [EarlyStopping](./section-08-callbacks-training-control-and-model-persistence.md)

Document results in a table - [Lab 09](./section-lab-09-neural-network-triathlon.md) requires this.

---

## Self-Check

1. Why is dropout disabled during `predict()`?
2. Why might dropout 0.5 require more training epochs?
3. How does L2 regularization differ from dropout mechanistically?
4. When would reducing neurons beat adding dropout?

---

## What's Next

[Section 9.8](./section-08-callbacks-training-control-and-model-persistence.md) automates training decisions with **callbacks**, and covers **saving/loading** models for deployment ([Chapter 07](../chapter-07-operationalizing-models/README.md)).

---

## References

- Prosise, *Applied Machine Learning and AI for Engineers*, Ch. 9 - Dropout
- [Srivastava et al., Dropout, 2014](https://jmlr.org/papers/v15/srivastava14a.html)
- [Keras Dropout layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)
- [Section 9.6](./section-06-multi-class-classification-face-recognition.md) | [Section 9.8](./section-08-callbacks-training-control-and-model-persistence.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 9.6](./section-06-multi-class-classification-face-recognition.md)  
**Next:** [Section 9.8 - Callbacks & Training Control](./section-08-callbacks-training-control-and-model-persistence.md)



