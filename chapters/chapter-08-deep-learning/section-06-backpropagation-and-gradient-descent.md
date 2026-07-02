# Section 8.6: Backpropagation & Gradient Descent

> **Source:** Prosise, Ch. 8 - "Backpropagation and Gradient Descent"  
> **Prerequisites:** [Section 8.5](./section-05-loss-functions.md) | [Section 2.2](../chapter-02-regression-models/section-02-linear-regression.md)  
> **Glossary:** [backpropagation](../../GLOSSARY.md#backpropagation) | [gradient descent](../../GLOSSARY.md#gradient-descent) | [gradient boosting](../../GLOSSARY.md#gradient-boosting)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Learning Question

Forward propagation computes $\hat{y}$ from weights. **Learning** asks: how should each weight change to reduce [loss](./section-05-loss-functions.md)?

$$
\theta \leftarrow \theta - \eta \frac{\partial \mathcal{L}}{\partial \theta}
$$
> **Readable form:** new parameters = old parameters − (learning rate × gradient of loss)

This is **[gradient descent](../../GLOSSARY.md#gradient-descent)** - the same idea as iterative weight updates in advanced regression, and spiritually related to [gradient boosting](../chapter-02-regression-models/section-04-ensemble-methods-random-forests-and-gradient-boosting.md) (which uses gradients of a loss to build trees, not neural weights).

$\eta$ is the **learning rate** - a critical [hyperparameter](../../GLOSSARY.md#hyperparameter) ([Section 8.8](./section-08-training-loop-concepts.md)).

---

## Intuition: Walking Downhill

Plot loss vs one weight $w$ (others fixed):

```
Loss
  |\
  | \
  |  \___
  |      \____
  +──────────── w
```

- **Gradient $> 0$:** loss increases as $w$ increases → decrease $w$
- **Gradient $< 0$:** loss decreases as $w$ increases → increase $w$
- **Gradient $= 0$:** local minimum (hopefully global-ish)

Too-large $\eta$ overshoots the valley; too-small $\eta$ crawls ([Lab 08](./section-lab-08-neural-network-intuition-workshop.md) visualizes this).

---

## The Chain Rule (Core of Backprop)

Neural networks compose functions:

$$
\mathcal{L} = \ell\big(f^{[L]}(\cdots f^{[1]}(\mathbf{x})\cdots)\big)
$$
> **Readable form:** loss = outer function of layer L of … layer 1 of input

If $y = f(g(x))$, then:

$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}
$$
> **Readable form:** derivative of composition = (derivative of outer) × (derivative of inner)

**[Backpropagation](../../GLOSSARY.md#backpropagation)** applies the chain rule layer-by-layer **backward** from loss to every weight - efficiently reusing intermediate terms.

---

## Why Backprop Beats Naive Approaches

| Method | Cost for $P$ parameters |
|--------|-------------------------|
| **Finite differences** | $O(P)$ forward passes per gradient |
| **Backpropagation** | $O(1)$ forward + $O(1)$ backward |

For millions of parameters, finite differences are impossible. Backprop is the engineering breakthrough that made deep learning practical (1980s-present).

Prosise gives intuition in Chapter 8; Course 3 derives the full matrix calculus.

---

## Tiny Example: One Weight, MSE Loss

Model $\hat{y} = w x$, one example $(x=2, y=5)$:

$$
\mathcal{L} = (y - wx)^2 = (5 - 2w)^2
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

**Gradient:**

$$
\frac{\partial \mathcal{L}}{\partial w} = 2(5 - 2w)(-x) = -2x(5 - 2w)
$$
> **Readable form:** d(loss)/d(w) = −2 × input × (target − prediction)

At $w=1$:

$$
\frac{\partial \mathcal{L}}{\partial w} = -2(2)(5-2) = -12
$$
> **Readable form:** This derivative tells how the loss changes when the parameter changes; backprop uses it to decide the update direction.

**Update** with $\eta = 0.1$:

$$
w_{\text{new}} = 1 - 0.1(-12) = 2.2
$$
> **Readable form:** Gradient descent moves weight $w$ from 1 to 2.2 because the negative gradient says increasing the weight reduces loss.

Closer to optimal $w = 2.5$.

```python
w, x, y, eta = 1.0, 2.0, 5.0, 0.1
for step in range(5):
    y_hat = w * x
    loss = (y - y_hat) ** 2
    grad = -2 * x * (y - w * x)
    w = w - eta * grad
    print(f"step {step}: w={w:.3f}, loss={loss:.3f}")
```

---

## Two-Layer Network: Backprop Sketch

Network: $x \rightarrow z^{[1]} = w^{[1]} x + b^{[1]} \rightarrow a^{[1]} = \text{ReLU}(z^{[1]}) \rightarrow \hat{y} = w^{[2]} a^{[1]} + b^{[2]}$

MSE loss $\mathcal{L} = \frac{1}{2}(y - \hat{y})^2$ (the $\frac{1}{2}$ cancels cleanly in derivatives).

**Backward pass (scalar case):**

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}} = -(y - \hat{y})
$$
> **Readable form:** This derivative tells how the loss changes when the parameter changes; backprop uses it to decide the update direction.

$$
\frac{\partial \mathcal{L}}{\partial w^{[2]}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot a^{[1]}
$$
> **Readable form:** This derivative tells how the loss changes when the parameter changes; backprop uses it to decide the update direction.

$$
\frac{\partial \mathcal{L}}{\partial a^{[1]}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot w^{[2]}
$$
> **Readable form:** This derivative tells how the loss changes when the parameter changes; backprop uses it to decide the update direction.

$$
\frac{\partial \mathcal{L}}{\partial z^{[1]}} = \frac{\partial \mathcal{L}}{\partial a^{[1]}} \cdot \mathbb{1}[z^{[1]} > 0] \quad \text{(ReLU derivative)}
$$

$$
\frac{\partial \mathcal{L}}{\partial w^{[1]}} = \frac{\partial \mathcal{L}}{\partial z^{[1]}} \cdot x
$$
> **Readable form:** gradients flow backward, multiplying local derivatives at each gate

---

## NumPy: Manual Gradient for Linear Layer

```python
import numpy as np

# Batch size m=2, one feature
X = np.array([[1.0], [2.0]])
y = np.array([[3.0], [5.0]])
w = np.array([[1.0]])
b = np.array([0.0])

# Forward
y_hat = X @ w + b
loss = np.mean((y - y_hat) ** 2)

# Backward (MSE + linear)
m = X.shape[0]
dL_dyhat = (2 / m) * (y_hat - y)           # (m, 1)
dL_dw = X.T @ dL_dyhat                      # (1, 1)
dL_db = np.sum(dL_dyhat, axis=0)            # (1,)

print("dL/dw:", dL_dw.flatten(), "dL/db:", dL_db)
```

Matches `LinearRegression` OLS direction - neural training is generalized gradient descent on arbitrary graphs.

---

## Learning Rate $\eta$

| $\eta$ too small | $\eta$ too large |
|------------------|------------------|
| Slow convergence | Loss oscillates or diverges |
| Safe but expensive | May skip good minima |
| Needs many epochs | NaN weights (exploding) |

Prosise recommends starting around $10^{-3}$ for Adam in Keras ([Chapter 09](../chapter-09-neural-networks/README.md)); manual SGD on toy problems often uses $0.01$-$0.1$.

```python
# Same problem, eta too high
w = 1.0
for _ in range(10):
    grad = -2 * 2 * (5 - 2*w)
    w = w - 1.0 * grad  # eta=1.0 - unstable
# w bounces wildly
```

[Chapter 07](../chapter-07-operationalizing-models/README.md) monitoring applies: if production model quality degrades after retraining with new hyperparameters, suspect learning rate or data drift.

---

## Stochastic and Mini-Batch Gradient Descent

| Variant | Gradient computed on | Noise | Speed |
|---------|---------------------|-------|-------|
| **Batch GD** | Full training set | None | Slow per epoch |
| **SGD** | One example | High | Fast updates |
| **Mini-batch** | $B$ examples (32-256) | Moderate | GPU-friendly default |

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{\text{batch}}
$$
> **Readable form:** update using average gradient over current mini-batch

Noise helps escape shallow local minima - often improves generalization.

---

## Backprop on Computational Graph

From [Section 8.4](./section-04-forward-propagation.md):

```
x ──→ z¹ ──→ a¹ ──→ z² ──→ L
```

Backward messages multiply **local gradients**:

| Gate | Forward | Local gradient |
|------|---------|----------------|
| Add bias | $z = Wx + b$ | $\partial z/\partial W = x$ |
| ReLU | $a = \max(0,z)$ | $1$ if $z>0$ else $0$ |
| MSE | $\frac{1}{2}(y-\hat{y})^2$ | $\hat{y} - y$ |

TensorFlow/PyTorch automate this; understanding gates debugs `None` gradients and frozen layers.

---

## Vanishing Gradients (Preview)

Sigmoid saturates - derivative near 0 when $|z|$ is large. Deep sigmoid stacks multiply many small numbers → gradient vanishes. [Section 8.7](./section-07-activation-functions-deep-dive.md) explains why ReLU became default.

---

## Connection to Part I

| Part I concept | Neural network parallel |
|----------------|------------------------|
| [OLS closed form](../chapter-02-regression-models/section-02-linear-regression.md) | One-step if linear; iterative if nonlinear |
| [Ridge penalty](../chapter-02-regression-models/section-02-linear-regression.md) | L2 regularization on weights |
| [GBM residuals](../chapter-02-regression-models/section-04-ensemble-methods-random-forests-and-gradient-boosting.md) | Both use gradients of loss - different parameterization |
| [SVM dual optimization](../chapter-05-support-vector-machines/README.md) | Different objective; same "optimize convex-ish loss" mindset |

---

## Full Training Step (NumPy Pseudocode)

```python
for epoch in range(num_epochs):
    for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size):
        # Forward
        y_hat, cache = forward(X_batch, params)
        loss = compute_loss(y_hat, y_batch)

        # Backward
        grads = backward(cache, y_batch, params)

        # Update
        for key in params:
            params[key] -= learning_rate * grads[key]
```

[Lab 08](./section-lab-08-neural-network-intuition-workshop.md) implements a minimal version for XOR.

---

## Common Mistakes (Section 8.6)

| Mistake | Why it hurts | Fix |
|---------|--------------|-----|
| **Not zeroing gradients** | Accumulate stale grads | `grad = 0` before backward |
| **Learning rate untuned** | #1 training failure mode | Grid search; ReduceLROnPlateau in Chapter 09 |
| **Confusing chain rule order** | Wrong sign on update | Loss should decrease each step (usually) |
| **Backprop through detached tensors** | No learning | Ensure graph connected in frameworks |
| **Ignoring gradient clipping** | RNN/exploding issues | `clipnorm` in Keras for unstable runs |

---

## Self-Check

1. State the weight update rule in words.
2. Why is backprop $O(1)$ per parameter vs finite differences $O(P)$?
3. For $\hat{y} = wx$ and MSE, what is $\partial \mathcal{L}/\partial w$?
4. What happens if learning rate is too large?

---

## Further Reading

- Prosise, Ch. 8 - backpropagation intuition
- [Section 8.7 - Activation Functions](./section-07-activation-functions-deep-dive.md)
- Course 3 - full backprop derivation
- [GLOSSARY.md](../../GLOSSARY.md)

---

**Previous:** [Section 8.5 - Loss Functions](./section-05-loss-functions.md) | **Next:** [Section 8.7 - Activation Functions Deep Dive](./section-07-activation-functions-deep-dive.md)


