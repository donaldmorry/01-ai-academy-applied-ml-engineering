# Section 8.7: Activation Functions Deep Dive

> **Source:** Prosise, Ch. 8 - "Activation Functions"  
> **Prerequisites:** [Section 8.2](./section-02-the-artificial-neuron.md) | [Section 3.2](../chapter-03-classification-models/section-02-logistic-regression.md)  
> **Glossary:** [neural network](../../GLOSSARY.md#neural-network) | [classification](../../GLOSSARY.md#classification)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Why Activations Exist

Without nonlinear $\phi$, a stack of layers collapses to one linear map ([Section 8.4](./section-04-forward-propagation.md)). **Activation functions** inject nonlinearity so networks can approximate curved boundaries - like [decision trees](../chapter-02-regression-models/section-03-decision-trees-for-regression.md) but with smooth, differentiable surfaces.

Choice of $\phi$ affects:

- **Expressiveness** - what shapes the network can carve
- **Gradient flow** - whether [backprop](./section-06-backpropagation-and-gradient-descent.md) reaches early layers
- **Output interpretation** - probability vs unbounded regression

---

## Sigmoid (Logistic)

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
> **Readable form:** sigmoid(z) = 1 / (1 + e^(−z))

| Property | Value |
|----------|-------|
| Range | $(0, 1)$ |
| Derivative | $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ |
| Max derivative | $0.25$ at $z = 0$ |

**Use cases:**

- **Binary output layer** - probability interpretation ([logistic regression](../chapter-03-classification-models/section-02-logistic-regression.md))
- **Historical hidden layers** - largely replaced by ReLU

**Problems in hidden layers:**

- **Vanishing gradients** - multiplying many factors $< 0.25$ → near-zero updates in deep nets
- **Not zero-centered** - slows convergence

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-8, 8, 300)
sig = 1 / (1 + np.exp(-z))
sig_prime = sig * (1 - sig)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(z, sig); ax[0].set_title('Sigmoid'); ax[0].axhline(0.5, ls='--', alpha=0.5)
ax[1].plot(z, sig_prime); ax[1].set_title("Sigmoid derivative (max 0.25)")
plt.tight_layout()
```

---

## Hyperbolic Tangent (tanh)

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$
> **Readable form:** tanh(z) = (e^z − e^(−z)) / (e^z + e^(−z))

| Property | vs Sigmoid |
|----------|------------|
| Range | $(-1, 1)$ |
| Zero-centered | Yes - often converges faster |
| Vanishing gradient | Still saturates at $\pm 1$ |

Common in **RNN/LSTM** gates (Chapter 13 preview). Hidden layers in feedforward nets: prefer ReLU.

```python
tanh = np.tanh(z)
tanh_prime = 1 - tanh ** 2
```

---

## ReLU (Rectified Linear Unit)

$$
\text{ReLU}(z) = \max(0, z)
$$
> **Readable form:** ReLU(z) = z if z > 0, else 0

| Property | Implication |
|----------|-------------|
| Range | $[0, \infty)$ |
| Derivative | $1$ if $z > 0$, else $0$ |
| Computation | Trivial - no exponentials |

**Default choice for hidden layers** in MLPs, CNNs (Chapter 10), most modern architectures.

**Dying ReLU problem:** Neurons with $z \leq 0$ always get zero gradient - permanently "dead." **Leaky ReLU** mitigates:

$$
\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}
$$
> **Readable form:** Leaky ReLU = z when positive; small slope α×z when negative (α ≈ 0.01)

```python
relu = np.maximum(0, z)
leaky = np.where(z > 0, z, 0.01 * z)

plt.figure(figsize=(6, 4))
plt.plot(z, relu, label='ReLU')
plt.plot(z, leaky, label='Leaky ReLU')
plt.legend(); plt.title('ReLU family')
```

---

## Softmax (Multi-Class Output)

For logit vector $\mathbf{z} \in \mathbb{R}^K$:

$$
\text{softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$
> **Readable form:** class-k probability = e^(score_k) divided by sum of e^(all scores)

Properties:

- Outputs in $(0, 1)$ summing to 1
- Amplifies largest logit
- Used **only at output** for multi-class ([MNIST](../chapter-03-classification-models/section-08-case-study-mnist-digit-recognition.md))

```python
def softmax(z):
    exp_z = np.exp(z - z.max())  # stability
    return exp_z / exp_z.sum()

logits = np.array([2.0, 1.0, 0.1])
print(softmax(logits))  # sums to 1.0
```

Pair with **categorical cross-entropy** ([Section 8.5](./section-05-loss-functions.md)).

---

## Linear (Identity)

$$
\phi(z) = z
$$
> **Readable form:** output = input unchanged

**Use:** Regression output layer ([taxi fare](../chapter-02-regression-models/section-08-taxi-fare-prediction-end-to-end-project.md) in Chapter 09). Sometimes hidden layers in edge cases - rare.

---

## Comparison Table

| Activation | Range | Typical layer | Vanishing grad? |
|------------|-------|---------------|-----------------|
| **ReLU** | $[0, \infty)$ | Hidden | Mild (dead neurons) |
| **Leaky ReLU** | $(-\infty, \infty)$ | Hidden | Mild |
| **Sigmoid** | $(0, 1)$ | Binary output | Severe in hidden |
| **tanh** | $(-1, 1)$ | RNN gates | Severe in deep hidden |
| **Softmax** | $(0,1)^K$, sum=1 | Multi-class output | N/A at output |
| **Linear** | $(-\infty, \infty)$ | Regression output | N/A |

---

## Vanishing Gradients: The Mechanism

Consider 5 sigmoid layers. Backprop multiplies derivatives:

$$
\frac{\partial \mathcal{L}}{\partial z^{[1]}} \propto \prod_{\ell=2}^{5} \sigma'(z^{[\ell]})
$$
> **Readable form:** early-layer gradient ≈ product of up to 5 terms each ≤ 0.25

$0.25^5 \approx 0.001$ - early layers learn glacially. ReLU derivatives of 0 or 1 avoid this shrinkage for active units.

Prosise uses this to justify ReLU before introducing Keras defaults.

---

## Hand Trace: ReLU Derivative in Backprop

If $z^{[1]} = -2$, then $a^{[1]} = 0$ and $\partial a^{[1]}/\partial z^{[1]} = 0$ - gradient **stops** at this neuron for this example.

If $z^{[1]} = 3$, derivative is 1 - gradient flows through unchanged.

This sparsity can speed computation and create specialized detectors.

---

## Choosing Activations: Recipe

```
Hidden layers     → ReLU or Leaky ReLU
Binary output     → Sigmoid (+ BCE loss)
Multi-class output → Softmax (+ categorical CE)
Regression output → Linear (+ MSE loss)
```

Do **not** mix sigmoid hidden layers with deep networks unless you know why (legacy RBM stacks, etc.).

---

## Connection to Chapter 03 and 05

| Classical model | Activation echo |
|-----------------|-----------------|
| [Logistic regression](../chapter-03-classification-models/section-02-logistic-regression.md) | Sigmoid on output |
| [Softmax regression](../chapter-03-classification-models/section-08-case-study-mnist-digit-recognition.md) (multinomial) | Softmax on output |
| [SVM with RBF](../chapter-05-support-vector-machines/section-02-kernels-and-the-kernel-trick.md) | Implicit feature map - no explicit $\phi$ per neuron |

---

## NumPy: Activation Chapter

```python
ACTIVATIONS = {
    'relu': (lambda z: np.maximum(0, z),
             lambda a, z: (z > 0).astype(float)),
    'sigmoid': (lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),
                lambda a, z: a * (1 - a)),
    'tanh': (np.tanh,
             lambda a, z: 1 - a**2),
    'linear': (lambda z: z,
               lambda a, z: np.ones_like(z)),
}

def forward_activation(z, name):
    f, _ = ACTIVATIONS[name]
    return f(z)

def backward_activation(a, z, name):
    _, f_prime = ACTIVATIONS[name]
    return f_prime(a, z)
```

Use in [Lab 08](./section-lab-08-neural-network-intuition-workshop.md) activation comparison plots.

---

## Swish / GELU (Brief Modern Note)

Advanced architectures sometimes use smooth ReLU variants (Swish, GELU) in transformers. For Part II tabular MLPs, ReLU remains Prosise's recommendation - do not overcomplicate before [Chapter 09](../chapter-09-neural-networks/README.md).

---

## Common Mistakes (Section 8.7)

| Mistake | Why it hurts | Fix |
|---------|--------------|-----|
| **Sigmoid in all hidden layers** | Vanishing gradients | Use ReLU |
| **Softmax on hidden layer** | Breaks representation learning | Softmax at output only |
| **ReLU on regression output** | Clips negative predictions | Linear output |
| **Forgetting softmax + CE pairing** | Wrong loss landscape | Match activation to loss |
| **Not plotting activations when debugging** | Miss saturation | Histogram $z$ values each epoch |

---

## Key Vocabulary

| Term | Definition |
|------|-----------|
| **ReLU** | max(0, z); default hidden activation |
| **Leaky ReLU** | Small negative slope for z ≤ 0 |
| **Sigmoid** | S-shaped map to (0, 1) |
| **Softmax** | Normalizes logits to class probabilities |
| **Vanishing gradient** | Early layers receive near-zero updates |
| **Dying ReLU** | Neuron stuck at zero output and gradient |

---

## Self-Check

1. Why is ReLU's derivative 0 for negative inputs?
2. Maximum value of sigmoid derivative?
3. Which activation for 10-digit MNIST output?
4. Why does tanh converge faster than sigmoid for hidden units (historically)?

---

## Lab Preview

[Lab 08, Part B](./section-lab-08-neural-network-intuition-workshop.md) plots sigmoid, tanh, and ReLU side-by-side and shades regions where sigmoid gradient $< 0.05$.

---

## Further Reading

- Prosise, Ch. 8 - activation function survey
- [Section 8.8 - Training Loop](./section-08-training-loop-concepts.md)
- [Section 3.2 - Sigmoid in logistic regression](../chapter-03-classification-models/section-02-logistic-regression.md)

---

**Previous:** [Section 8.6 - Backpropagation & Gradient Descent](./section-06-backpropagation-and-gradient-descent.md) | **Next:** [Section 8.8 - Training Loop Concepts](./section-08-training-loop-concepts.md)



