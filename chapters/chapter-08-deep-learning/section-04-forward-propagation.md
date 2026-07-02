# Section 8.4: Forward Propagation

> **Source:** Prosise, Ch. 8 - "Forward Propagation"  
> **Prerequisites:** [Section 8.3](./section-03-multi-layer-networks.md) | [Section 2.2](../chapter-02-regression-models/section-02-linear-regression.md)  
> **Glossary:** [neural network](../../GLOSSARY.md#neural-network) | [feature](../../GLOSSARY.md#feature)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## What Forward Propagation Computes

**Forward propagation** (forward pass) evaluates the network from input to prediction:

$$
\mathbf{a}^{[0]} = \mathbf{x} \quad \rightarrow \quad \mathbf{a}^{[1]} \quad \rightarrow \quad \cdots \quad \rightarrow \quad \hat{\mathbf{y}} = \mathbf{a}^{[L]}
$$
> **Readable form:** start with input features, apply each layer in sequence, end with prediction

During **inference** in [Chapter 07](../chapter-07-operationalizing-models/README.md) deployment, only the forward pass runs - no loss, no gradients. Training repeats forward + backward ([Section 8.6](./section-06-backpropagation-and-gradient-descent.md)).

---

## Single Layer, Batch of Examples

Given $\mathbf{X} \in \mathbb{R}^{m \times n}$ ($m$ examples, $n$ features), weights $\mathbf{W} \in \mathbb{R}^{h \times n}$, bias $\mathbf{b} \in \mathbb{R}^{h}$:

$$
\mathbf{Z} = \mathbf{X}\mathbf{W}^T + \mathbf{b}
$$

$$
\mathbf{A} = \phi(\mathbf{Z})
$$
> **Readable form:**  
> pre-activation matrix = (batch of inputs) × (weights transposed) + bias  
> activation matrix = activation function applied row-wise / element-wise

**Shape cheat sheet:**

| Tensor | Shape | Meaning |
|--------|-------|---------|
| $\mathbf{X}$ | $(m, n)$ | Batch of inputs |
| $\mathbf{W}$ | $(h, n)$ | $h$ neurons, each with $n$ weights |
| $\mathbf{Z}$ | $(m, h)$ | Pre-activations for all examples |
| $\mathbf{A}$ | $(m, h)$ | Hidden activations |

Note $\mathbf{X}\mathbf{W}^T$ - or equivalently $(\mathbf{W}\mathbf{X}^T)^T$ - matches NumPy row-example convention.

---

## Stacking Layers

For a 3-layer network (2 hidden + output conceptual; counting hidden + output):

$$
\mathbf{Z}^{[1]} = \mathbf{X}\mathbf{W}^{[1]T} + \mathbf{b}^{[1]}, \quad \mathbf{A}^{[1]} = \text{ReLU}(\mathbf{Z}^{[1]})
$$

$$
\mathbf{Z}^{[2]} = \mathbf{A}^{[1]}\mathbf{W}^{[2]T} + \mathbf{b}^{[2]}, \quad \hat{\mathbf{Y}} = \sigma(\mathbf{Z}^{[2]})
$$
> **Readable form:** layer 1 transforms inputs; layer 2 transforms layer-1 activations into predictions

Each layer consumes the **activations** of the previous layer, not raw $\mathbf{x}$ (except layer 1).

---

## NumPy: Complete Forward Pass (2 → 3 → 1)

Tiny network for regression-style scalar output (identity on output):

```python
import numpy as np

np.random.seed(8)

# Architecture: 2 inputs, 3 hidden (ReLU), 1 output (linear)
n_in, n_hid, n_out = 2, 3, 1

W1 = np.random.randn(n_hid, n_in) * 0.5
b1 = np.zeros(n_hid)
W2 = np.random.randn(n_out, n_hid) * 0.5
b2 = np.zeros(n_out)

def relu(Z):
    return np.maximum(0, Z)

def forward(X, W1, b1, W2, b2):
    """X shape (m, n_in)"""
    Z1 = X @ W1.T + b1      # (m, n_hid)
    A1 = relu(Z1)
    Z2 = A1 @ W2.T + b2     # (m, n_out)
    Y_hat = Z2              # linear output
    cache = (X, Z1, A1, Z2)
    return Y_hat, cache

X = np.array([[1.0, 2.0],
              [0.5, -1.0],
              [-2.0, 0.3]])
Y_hat, _ = forward(X, W1, b1, W2, b2)
print("Predictions shape:", Y_hat.shape)  # (3, 1)
print(Y_hat)
```

---

## Hand Trace: One Example Through 2→3→1

Use $x = [1, 2]^T$, simplified integer weights for pencil-and-paper:

$$
\mathbf{W}^{[1]} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad \mathbf{b}^{[1]} = \begin{bmatrix} 0 \\ 0 \\ -1 \end{bmatrix}
$$
> **Readable form:** The first-layer weights copy the two inputs and combine them once, while biases set each hidden unit's threshold.

**Layer 1:**

$$
\mathbf{z}^{[1]} = \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}, \quad \mathbf{a}^{[1]} = \text{ReLU}\big(\begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}\big) = \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}
$$
> **Readable form:** hidden pre-activations [1, 2, 2]; ReLU leaves them unchanged (all positive)

$$
\mathbf{W}^{[2]} = \begin{bmatrix} 1 & -1 & 0.5 \end{bmatrix}, \quad b^{[2]} = 0
$$
> **Readable form:** The output layer adds, subtracts, and partially weights the hidden activations to form one final score.

**Layer 2:**

$$
z^{[2]} = 1(1) + (-1)(2) + 0.5(2) = 1 - 2 + 1 = 0
$$

$$
\hat{y} = 0
$$
> **Readable form:** final prediction = **0**

Verify with NumPy before moving on - hand traces build debugging reflexes when Keras shapes fail in [Chapter 09](../chapter-09-neural-networks/README.md).

---

## Shape Tracking Workflow (Prosise's Rule)

Before writing code, write a **shape table**:

| Step | Operation | Output shape |
|------|-----------|--------------|
| Input | $\mathbf{x}$ | $(2,)$ or $(1, 2)$ batch |
| Layer 1 | $\mathbf{z}^{[1]} = \mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}$ | $(3,)$ |
| Layer 1 | $\mathbf{a}^{[1]} = \text{ReLU}(\mathbf{z}^{[1]})$ | $(3,)$ |
| Layer 2 | $\hat{y} = \mathbf{W}^{[2]}\mathbf{a}^{[1]} + b^{[2]}$ | $(1,)$ |

When MNIST uses $m = 128$ mini-batch:

| Tensor | Shape |
|--------|-------|
| $\mathbf{X}$ | $(128, 784)$ |
| $\mathbf{W}^{[1]}$ | $(256, 784)$ |
| $\mathbf{A}^{[1]}$ | $(128, 256)$ |

This is the same discipline as tracking DataFrame columns through a [sklearn Pipeline](../chapter-07-operationalizing-models/README.md).

---

## Matrix Form Connection to Chapter 02

[Linear regression](../chapter-02-regression-models/section-02-linear-regression.md) in one shot:

$$
\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}
$$
> **Readable form:** With only linear operations, predictions are a matrix-vector product of inputs and effective weights.

A neural network is **chained** linear maps with nonlinearities sandwiched between:

$$
\hat{\mathbf{y}} = \mathbf{W}^{[2]} \, \phi\big(\mathbf{W}^{[1]} \mathbf{X}^T\big)^T
$$
> **Readable form:** prediction = second weight matrix × nonlinear(first weight matrix × inputs)

Without $\phi$, a two-layer network collapses:

$$
\mathbf{W}^{[2]}\mathbf{W}^{[1]}\mathbf{x} = \mathbf{W}'\mathbf{x}
$$
> **Readable form:** Stacked linear layers collapse into one equivalent linear transform if no nonlinear activation sits between them.

One matrix - why activations matter ([Section 8.7](./section-07-activation-functions-deep-dive.md)).

---

## Broadcasting Bias

Bias $\mathbf{b} \in \mathbb{R}^{h}$ adds to each of $m$ rows:

```python
Z = X @ W.T          # (m, h)
Z = Z + b            # b shape (h,) broadcasts across rows
```

Equivalent to appending a feature column of ones (the intercept trick from linear regression).

---

## Forward Pass for Classification Outputs

| Task | Output activation | $\hat{\mathbf{y}}$ interpretation |
|------|-------------------|-----------------------------------|
| Regression | Identity / linear | Real number ([Chapter 02](../chapter-02-regression-models/README.md)) |
| Binary | Sigmoid | Probability in $[0,1]$ ([Chapter 03](../chapter-03-classification-models/section-02-logistic-regression.md)) |
| Multi-class | Softmax | Class probabilities summing to 1 |

Softmax for logits $\mathbf{z} \in \mathbb{R}^K$:

$$
\text{softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$
> **Readable form:** class-k probability = e^(score_k) divided by sum of e^(all scores)

Forward pass computes logits first; softmax last ([Section 8.7](./section-07-activation-functions-deep-dive.md)).

---

## Caching Intermediate Values

Training stores $(\mathbf{Z}^{[\ell]}, \mathbf{A}^{[\ell]})$ during forward pass because [backpropagation](./section-06-backpropagation-and-gradient-descent.md) needs them:

```python
cache = {
    'A0': X,
    'Z1': Z1, 'A1': A1,
    'Z2': Z2, 'A2': Y_hat,
}
```

Memory trade-off: larger batch = more cache = more GPU RAM - relevant when operationalizing large models ([Chapter 07](../chapter-07-operationalizing-models/README.md)).

---

## NumPy: MNIST-Sized Shape Drill (No Training)

```python
m, n_in, n_hid, n_out = 64, 784, 128, 10

X = np.random.randn(m, n_in)
W1 = np.random.randn(n_hid, n_in) * 0.01
b1 = np.zeros(n_hid)
W2 = np.random.randn(n_out, n_hid) * 0.01
b2 = np.zeros(n_out)

Z1 = X @ W1.T + b1
A1 = np.maximum(0, Z1)
Z2 = A1 @ W2.T + b2

def softmax(Z, axis=1):
    expZ = np.exp(Z - Z.max(axis=axis, keepdims=True))
    return expZ / expZ.sum(axis=axis, keepdims=True)

probs = softmax(Z2)
assert probs.shape == (64, 10)
assert np.allclose(probs.sum(axis=1), 1.0)
print("Forward pass OK - rows sum to 1")
```

Connects to [Section 3.8](../chapter-03-classification-models/section-08-case-study-mnist-digit-recognition.md) - same 784→10 geometry, learned features instead of raw pixels + logistic regression.

---

## Performance: Vectorization

Python loops over examples are slow. Always batch:

| Approach | Speed |
|----------|-------|
| `for x in X: forward(x)` | Slow - Python overhead |
| `forward(X_batch)` with matmul | Fast - BLAS/GPU optimized |

This mirrors vectorized [OLS](../chapter-02-regression-models/section-02-linear-regression.md) vs row-by-row loops.

---

## Common Mistakes (Section 8.4)

| Mistake | Why it hurts | Fix |
|---------|--------------|-----|
| **Wrong transpose** | $(m,n) @ (m,n)$ mismatch | Draw shapes; use `X @ W.T` consistently |
| **Applying softmax before loss in code** | Numerical instability | Frameworks fuse softmax + cross-entropy |
| **Forgetting bias broadcast** | Wrong offsets | `assert b.shape == (n_hid,)` |
| **Mixing row/column conventions** | Silent wrong answers | Stick to examples-as-rows $(m, n)$ |
| **Not saving caches for backprop** | Recompute or fail | Return intermediates from `forward()` |

---

## Self-Check

1. If $\mathbf{X}$ is $(32, 50)$ and hidden layer has 100 neurons, what is $\mathbf{A}^{[1]}$ shape?
2. Why does $\mathbf{W}^{[2]}\mathbf{W}^{[1]}\mathbf{x}$ without activations reduce to one linear layer?
3. Write the forward pass for a 2-layer network in two equations.
4. What shape is $\mathbf{W}^{[1]}$ if layer 1 has 64 units and input has 12 features?

---

## Further Reading

- Prosise, Ch. 8 - forward propagation diagrams
- [Section 8.5 - Loss Functions](./section-05-loss-functions.md)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 8.3 - Multi-Layer Networks](./section-03-multi-layer-networks.md) | **Next:** [Section 8.5 - Loss Functions](./section-05-loss-functions.md)
