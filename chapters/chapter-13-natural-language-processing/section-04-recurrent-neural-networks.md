# Section 13.4: Recurrent Neural Networks

> **Source:** Prosise, Ch. 13 - RNN, LSTM, GRU for sequence classification  
> **Prerequisites:** [Sections 13.1-13.3](./section-01-beyond-bag-of-words.md)  
> **Glossary:** [RNN](../../GLOSSARY.md#recurrent-neural-network-rnn) | [LSTM](../../GLOSSARY.md#lstm) | [GRU](../../GLOSSARY.md#gru) | [vanishing gradient](../../GLOSSARY.md#vanishing-gradient)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Processing Sequences Token by Token

**Recurrent Neural Networks (RNNs)** maintain a **hidden state** updated at each timestep - carrying summary information forward through the sentence.

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b})
$$
> **Readable form:** hidden state at step t = tanh of (recurrent weight times previous hidden state + input weight times current input + bias)

Output at step $t$ can depend on $\mathbf{h}_t$; for classification, use final $\mathbf{h}_T$ or pooled states.

> **In plain English:** The RNN reads the sentence left-to-right, updating its "memory" after each word.

---

## Simple RNN in Keras

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Embedding(max_features, 128, mask_zero=True),
    layers.SimpleRNN(64),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

**SimpleRNN** is pedagogically clear but rarely used in production - vanishing gradients limit long-range learning.

---

## The Vanishing Gradient Problem

Backpropagation through time multiplies gradients across many timesteps:

$$
\frac{\partial L}{\partial \mathbf{h}_1} = \frac{\partial L}{\partial \mathbf{h}_T} \prod_{t=2}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}
$$
> **Readable form:** gradient w.r.t. early hidden state = product of many Jacobian matrices - often shrinks toward zero

Early tokens in long reviews receive **negligible learning signal** - the network forgets "not" at the start by the time it reaches "good" at the end.

See [vanishing gradient](../../GLOSSARY.md#vanishing-gradient).

---

## LSTM: Long Short-Term Memory

**LSTM** (Hochreiter & Schmidhuber, 1997) adds **gates** controlling information flow:

$$
\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(forget)} \\
\mathbf{i}_t &= \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(input)} \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(output)} \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)
\end{aligned}
$$
> **Readable form:** forget gate decides what to erase from cell state; input gate adds new content; output gate decides what to expose as hidden state

Cell state $\mathbf{c}_t$ provides a **highway** for gradients - enabling longer dependencies.

```python
model = models.Sequential([
    layers.Embedding(max_features, 128, mask_zero=True),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation='sigmoid'),
])
```

Prosise uses LSTM for IMDB sentiment - expect to beat Chapter 04 TF-IDF baseline in [Lab 13](./section-lab-13-nlp-progression-sentiment-to-translation.md).

See [LSTM in glossary](../../GLOSSARY.md#lstm).

---

## GRU: Gated Recurrent Unit

**GRU** simplifies LSTM - two gates (reset, update), fewer parameters:

```python
model = models.Sequential([
    layers.Embedding(max_features, 128, mask_zero=True),
    layers.GRU(64, dropout=0.2),
    layers.Dense(1, activation='sigmoid'),
])
```

| | LSTM | GRU |
|--|------|-----|
| Parameters | More | Fewer |
| Long sequences | Slightly stronger | Often comparable |
| Training speed | Slower | Faster |

Try both on validation set - no universal winner.

See [GRU in glossary](../../GLOSSARY.md#gru).

---

## Bidirectional RNNs

Process sequence **forward and backward**, concatenate hidden states:

```python
layers.Bidirectional(layers.LSTM(64))
```

Captures context from both directions - "not good" benefits from right-to-left pass. Doubles compute; standard for sentence classification.

---

## Stacked RNNs

```python
model = models.Sequential([
    layers.Embedding(max_features, 128, mask_zero=True),
    layers.LSTM(64, return_sequences=True),  # pass full sequence to next layer
    layers.LSTM(32),
    layers.Dense(1, activation='sigmoid'),
])
```

`return_sequences=True` required for all but final recurrent layer.

---

## IMDB Sentiment Example (Prosise Pattern)

```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_features = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

model = tf.keras.Sequential([
    layers.Embedding(max_features, 128),
    layers.LSTM(64, dropout=0.2),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
    x_train, y_train, epochs=5, batch_size=128,
    validation_split=0.2,
)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.3f}")
```

Target: **>88%** test accuracy - compare to TF-IDF + logistic regression from Chapter 04.

---

## Training Tips

| Technique | Purpose |
|-----------|---------|
| `EarlyStopping` | Halt when val_loss plateaus |
| `dropout` / `recurrent_dropout` | Regularize RNN |
| Gradient clipping | `optimizer=Adam(clipnorm=1.0)` stabilizes |
| Smaller batch on GPU memory | Reduce if OOM |

```python
from tensorflow.keras.callbacks import EarlyStopping

callbacks = [EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
```

---

## RNN Limitations

1. **Sequential computation** - cannot parallelize timesteps during training
2. **Long sequences** still strain even LSTM (1000+ tokens)
3. **Fixed context window** in practice via truncation (`max_len=200`)

Transformers ([Section 13.6](./section-06-the-transformer-architecture.md)) address parallelism and long-range attention - at higher compute cost.

---

## Key Takeaways

1. **RNNs** maintain hidden state $\mathbf{h}_t$ across timesteps
2. **Vanishing gradients** limit SimpleRNN on long sequences
3. **LSTM/GRU gates** preserve gradient flow for longer context
4. **Bidirectional** layers capture left and right context
5. LSTM sentiment on IMDB should **beat Chapter 04 baselines** - verify in lab

---

## Check Your Understanding

1. What does hidden state $\mathbf{h}_t$ represent?
2. Why does the vanishing gradient hurt negation in long sentences?
3. What role does the LSTM forget gate play?
4. When use Bidirectional LSTM vs unidirectional?
5. Why are RNNs slower to train than transformers on long sequences?

---

## References

- Prosise, Ch. 13 - LSTM sentiment
- Hochreiter & Schmidhuber, LSTM - [https://www.bioinf.jku.at/publications/older/2604.pdf](https://www.bioinf.jku.at/publications/older/2604.pdf)
- Keras RNN guide - [https://www.tensorflow.org/guide/keras/rnn](https://www.tensorflow.org/guide/keras/rnn)

---

**Next:** [Section 13.5 - Sequence-to-Sequence Models](./section-05-sequence-to-sequence-models.md)



