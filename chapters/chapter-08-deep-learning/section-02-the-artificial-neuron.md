# Section 8.2: The Artificial Neuron

> **Source:** Prosise, Ch. 8 - "The Artificial Neuron"  
> **Prerequisites:** [Section 8.1](./section-01-why-deep-learning.md) | [Section 2.2](../chapter-02-regression-models/section-02-linear-regression.md) | [Section 3.2](../chapter-03-classification-models/section-02-logistic-regression.md)  
> **Glossary:** [neural network](../../GLOSSARY.md#neural-network) | [feature](../../GLOSSARY.md#feature) | [classification](../../GLOSSARY.md#classification)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Biological Inspiration (Briefly)

Real neurons receive signals through **dendrites**, integrate them in the **cell body**, and fire an action potential down the **axon** if the signal exceeds a threshold. Artificial neurons are a loose cartoon of this process - useful for intuition, not neuroscience.

| Biological | Artificial analog |
|------------|-------------------|
| Synaptic strength | Weight $w_j$ |
| Resting potential offset | Bias $b$ |
| Firing rate | Activation output $a$ |
| Many inputs | Feature vector $\mathbf{x}$ |

Prosise warns against over-literal metaphors: we optimize with [gradient descent](../../GLOSSARY.md#gradient-descent), not spike-timing plasticity. The engineering abstraction is what matters.

---

## The Computational Unit

An artificial **neuron** (or **node**, **unit**) performs two steps:

### Step 1: Weighted sum (affine transform)

$$
z = \sum_{j=1}^{n} w_j x_j + b = \mathbf{w}^T \mathbf{x} + b
$$
> **Readable form:** pre-activation z = (weight₁ × feature₁) + (weight₂ × feature₂) + … + bias

This is **identical in structure** to [linear regression](../chapter-02-regression-models/section-02-linear-regression.md) and the linear score in [logistic regression](../chapter-03-classification-models/section-02-logistic-regression.md).

### Step 2: Activation function

$$
a = \phi(z)
$$
> **Readable form:** neuron output = activation function applied to z

Without $\phi$, stacking layers would collapse to one linear map - depth would buy nothing. Nonlinear $\phi$ is what makes [neural networks](../../GLOSSARY.md#neural-network) expressive ([Section 8.3](./section-03-multi-layer-networks.md)).

---

## The Perceptron: Binary Classifier

The **perceptron** (Rosenblatt, 1957) uses a step or sigmoid activation for yes/no decisions:

$$
\hat{y} = \mathbb{1}[\mathbf{w}^T \mathbf{x} + b > 0]
$$
> **Readable form:** predict class 1 if weighted sum plus bias is positive

For sigmoid output (smooth perceptron):

$$
\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$
> **Readable form:** predicted probability = 1 / (1 + e^(−z))

You already plotted this in [Section 3.2](../chapter-03-classification-models/section-02-logistic-regression.md). Logistic regression **is** a single neuron with sigmoid output.

---

## Logic Gates: What One Neuron Can Learn

With clever choice of weights, one neuron learns simple Boolean functions (inputs $x_1, x_2 \in \{0, 1\}$):

### AND gate

| $x_1$ | $x_2$ | Target |
|-------|-------|--------|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

Weights $w_1 = 1, w_2 = 1, b = -1.5$ give $z > 0$ only when both inputs are 1.

### OR gate

Weights $w_1 = 1, w_2 = 1, b = -0.5$ fire when at least one input is 1.

```python
import numpy as np

def neuron_forward(x, w, b, activation='step'):
    z = np.dot(w, x) + b
    if activation == 'step':
        return 1.0 if z > 0 else 0.0
    if activation == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(activation)

# AND neuron
w_and = np.array([1.0, 1.0])
b_and = -1.5
for x in [(0,0), (0,1), (1,0), (1,1)]:
    print(f"x={x} → {neuron_forward(np.array(x), w_and, b_and)}")
# Output: 0, 0, 0, 1
```

---

## Linear Separability

A dataset is **linearly separable** if there exists one hyperplane dividing classes:

$$
\mathbf{w}^T \mathbf{x} + b = 0
$$
> **Readable form:** decision boundary = points where weighted sum plus bias equals zero

In 2D, that is a straight line - the same limitation as linear [SVM](../chapter-05-support-vector-machines/README.md) without kernels.

```
     Class 1  •  •
              •  •
    ───────────────  ← single neuron boundary
     Class 0  ○  ○
              ○  ○
```

When classes interleave (moons, circles), **no** single neuron solves the problem - no matter how long you train.

---

## The XOR Crisis

**XOR** (exclusive OR) outputs 1 when inputs differ:

| $x_1$ | $x_2$ | XOR |
|-------|-------|-----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

No single line separates the 1s from the 0s in $(x_1, x_2)$ space:

```
  x₂
  1 |  0     1
    |
  0 |  0     1
    +───────── x₁
      0     1
```

Minsky & Papert (1969) highlighted this limitation of perceptrons, contributing to the first "AI winter." The fix - **hidden layers** - arrives in [Section 8.3](./section-03-multi-layer-networks.md). Prosise uses XOR as the canonical "why depth matters" example.

### NumPy proof: perceptron fails XOR

```python
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

# Brute-force search: no single w, b separates XOR
found = False
for w1 in np.linspace(-5, 5, 41):
    for w2 in np.linspace(-5, 5, 41):
        for b in np.linspace(-5, 5, 41):
            w = np.array([w1, w2])
            preds = (X_xor @ w + b > 0).astype(int)
            if np.array_equal(preds, y_xor):
                found = True
                break
print(f"Single perceptron solves XOR? {found}")  # False
```

---

## Hand-Traced Forward Pass: One Neuron

**Problem:** Predict spam (1) vs ham (0) from two features: keyword count $x_1 = 3$, exclamation count $x_2 = 5$.

**Parameters:** $w_1 = 0.4$, $w_2 = 0.8$, $b = -2.0$, sigmoid activation.

**Step 1 - weighted sum:**

$$
z = 0.4 \times 3 + 0.8 \times 5 + (-2.0) = 1.2 + 4.0 - 2.0 = 3.2
$$
> **Readable form:** z = 0.4×3 + 0.8×5 − 2.0 = **3.2**

**Step 2 - sigmoid:**

$$
\sigma(3.2) = \frac{1}{1 + e^{-3.2}} \approx 0.961
$$
> **Readable form:** probability of spam ≈ **96.1%**

**Step 3 - decision:** $\hat{p} > 0.5$ → predict **spam**.

```python
x = np.array([3.0, 5.0])
w = np.array([0.4, 0.8])
b = -2.0
z = x @ w + b
p = 1 / (1 + np.exp(-z))
print(f"z = {z:.2f}, p = {p:.4f}")  # z = 3.20, p = 0.9608
```

This is exactly `predict_proba` from a single-feature logistic model - the neuron is not exotic.

---

## Vectorized: Many Examples at Once

For batch matrix $\mathbf{X}$ with shape $(m, n)$ - $m$ examples, $n$ features:

$$
\mathbf{z} = \mathbf{X}\mathbf{w} + b
$$
> **Readable form:** vector of all pre-activations = (feature matrix) × (weight vector) + bias

Broadcasting adds the same $b$ to every row. This pattern scales to millions of rows - the same operation as [Section 2.2](../chapter-02-regression-models/section-02-linear-regression.md) matrix form.

```python
X = np.array([[3, 5], [0, 0], [1, 10]])  # 3 emails
w = np.array([0.4, 0.8])
b = -2.0
z = X @ w + b
p = 1 / (1 + np.exp(-z))
print(np.column_stack([z, p]))
```

---

## Weights as Feature Importance (Caution)

Large $|w_j|$ suggests feature $x_j$ strongly influences the output **when other features are held fixed** - similar to logistic coefficients in [Section 3.2](../chapter-03-classification-models/section-02-logistic-regression.md). But:

- Features on different scales mislead (always [standardize](../chapter-05-support-vector-machines/README.md) for fair comparison)
- Correlated features split credit arbitrarily
- Deep networks do not give global coefficients - use interpretability tools later

---

## Bias: The Threshold Knob

Bias $b$ shifts the decision boundary without changing its orientation:

| Effect of increasing $b$ | Geometric meaning |
|--------------------------|-------------------|
| Sigmoid output rises | Boundary moves away from origin |
| Step neuron fires more often | "Easier" to activate |

In Scikit-Learn's `LinearRegression`, the intercept is $w_0$ with a column of ones - same trick.

---

## Connection to Chapter 02 and 03

| Model | Neuron view |
|-------|-------------|
| [Linear regression](../chapter-02-regression-models/section-02-linear-regression.md) | $\phi(z) = z$ (identity activation) |
| [Logistic regression](../chapter-03-classification-models/section-02-logistic-regression.md) | $\phi(z) = \sigma(z)$, binary cross-entropy loss |
| [Ridge/Lasso](../chapter-02-regression-models/section-02-linear-regression.md) | Same neuron + weight penalty |

Neural networks **generalize** this template: multiple neurons per layer, multiple layers, different $\phi$ per layer ([Section 8.7](./section-07-activation-functions-deep-dive.md)).

---

## Common Mistakes (Section 8.2)

| Mistake | Why it hurts | Fix |
|---------|--------------|-----|
| **Forgetting bias** | Boundary forced through origin | Always include $b$ unless you have a reason |
| **Expecting one neuron to solve XOR** | Impossible - need hidden layer | Add depth ([Section 8.3](./section-03-multi-layer-networks.md)) |
| **Mixing up $z$ and $a$** | Wrong gradients later | $z$ = pre-activation, $a$ = post-activation |
| **Unscaled inputs** | One feature dominates $z$ | `StandardScaler` before training |
| **Treating perceptron = deep learning** | Underestimates composition power | Depth + backprop is the real story |

---

## Key Vocabulary

| Term | Definition |
|------|-----------|
| **Neuron / unit / node** | Computes $a = \phi(\mathbf{w}^T\mathbf{x} + b)$ |
| **Weight $w_j$** | Strength of connection from feature $j$ |
| **Bias $b$** | Offset before activation |
| **Pre-activation $z$** | Weighted sum before $\phi$ |
| **Activation $a$** | Neuron output after $\phi$ |
| **Perceptron** | Binary classifier neuron; historically a single-layer model |
| **Linear separability** | Classes dividable by one hyperplane |

---

## Self-Check

1. Write the two steps of a neuron in words (no symbols).
2. Why is XOR not linearly separable? Sketch the four points.
3. What activation does logistic regression use?
4. If $w_1 = 0$, what happens to feature $x_1$?

---

## Further Reading

- Prosise, Ch. 8 - artificial neuron and perceptron
- [Section 3.2 - Logistic Regression](../chapter-03-classification-models/section-02-logistic-regression.md)
- [GLOSSARY.md](../../GLOSSARY.md) - neural network, gradient descent

---

**Previous:** [Section 8.1 - Why Deep Learning?](./section-01-why-deep-learning.md) | **Next:** [Section 8.3 - Multi-Layer Networks](./section-03-multi-layer-networks.md)
