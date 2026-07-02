# Section 8.3: Multi-Layer Networks

> **Source:** Prosise, Ch. 8 - "Multi-Layer Networks"  
> **Prerequisites:** [Section 8.2](./section-02-the-artificial-neuron.md) | [Section 2.3](../chapter-02-regression-models/section-03-decision-trees-for-regression.md)  
> **Glossary:** [neural network](../../GLOSSARY.md#neural-network) | [underfitting](../../GLOSSARY.md#underfitting) | [overfitting](../../GLOSSARY.md#overfitting)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## From One Neuron to a Network

A **multi-layer perceptron (MLP)** chains layers of neurons:

```
Input layer  →  Hidden layer(s)  →  Output layer
  (features)     (learned reps)      (prediction)
```

| Layer type | Role | Example size (XOR network) |
|------------|------|---------------------------|
| **Input** | Pass features through (no learned weights on nodes) | 2 nodes ($x_1, x_2$) |
| **Hidden** | Nonlinear feature transformation | 2 nodes with ReLU |
| **Output** | Task-specific prediction | 1 node with sigmoid |

Prosise stresses: **hidden** means "not directly observed in input or final label" - they are internal representations discovered during training.

---

## Why Hidden Layers Fix XOR

Recall from [Section 8.2](./section-02-the-artificial-neuron.md): no single line separates XOR classes. A hidden layer **re-embeds** inputs into a space where a line *does* work.

Informal picture:

1. Hidden neurons carve the plane into regions
2. Each region maps to a new coordinate (hidden activation)
3. Output neuron draws a line in hidden-space

This is the same *spirit* as the [kernel trick](../chapter-05-support-vector-machines/section-02-kernels-and-the-kernel-trick.md) - lift data to a space where a linear classifier suffices - but the lift is **learned**, not hand-designed.

---

## Network Notation

For layer $\ell$ with $n^{[\ell]}$ neurons:

$$
\mathbf{z}^{[\ell]} = \mathbf{W}^{[\ell]} \mathbf{a}^{[\ell-1]} + \mathbf{b}^{[\ell]}
$$

$$
\mathbf{a}^{[\ell]} = \phi^{[\ell]}\big(\mathbf{z}^{[\ell]}\big)
$$
> **Readable form:**  
> pre-activations at layer ℓ = (weight matrix) × (previous activations) + (bias vector)  
> activations at layer ℓ = activation function applied element-wise to pre-activations

Convention: $\mathbf{a}^{[0]} = \mathbf{x}$ (input features).

### Weight matrix shapes

If layer $\ell-1$ has $n^{[\ell-1]}$ units and layer $\ell$ has $n^{[\ell]}$ units:

$$
\mathbf{W}^{[\ell]} \in \mathbb{R}^{n^{[\ell]} \times n^{[\ell-1]}}
$$
> **Readable form:** weight matrix rows = neurons in current layer, columns = neurons in previous layer

**Shape tracking is non-negotiable** - [Section 8.4](./section-04-forward-propagation.md) drills this until it is reflex.

---

## Universal Approximation (Intuition, Not Proof)

The **universal approximation theorem** (Cybenko, 1989; Hornik) states that a feedforward network with one sufficiently wide hidden layer can approximate any continuous function on a compact domain - arbitrarily well.

Engineering translation:

| Statement | Practical implication |
|-----------|----------------------|
| One wide hidden layer *can* work | Often inefficient |
| Deeper, narrower networks *often* need fewer parameters | Depth composes features hierarchically |
| Theory guarantees existence | Does **not** guarantee your SGD run finds the right weights |

[Decision trees](../chapter-02-regression-models/section-03-decision-trees-for-regression.md) also approximate complex functions but partition axis-aligned. Neural networks learn **oblique** boundaries and smooth surfaces - different inductive bias.

---

## Depth vs Width

| Dimension | More depth (layers) | More width (neurons/layer) |
|-----------|---------------------|---------------------------|
| **Compositional structure** | Builds features in stages | Parallel feature detectors |
| **Parameter efficiency** | Often better for images, text | May need exponentially more units shallow |
| **Training difficulty** | Vanishing gradients ([Section 8.7](./section-07-activation-functions-deep-dive.md)) | Usually easier optimization |
| **Overfitting risk** | High with small data | High with small data |

Prosise's practical advice for tabular data (previewing [Chapter 09](../chapter-09-neural-networks/README.md)): start with 1-2 hidden layers of 32-128 units - not hundreds of layers.

---

## XOR Solved: Architecture Sketch

Minimal XOR network:

- Input: 2
- Hidden: 2 (ReLU activation)
- Output: 1 (sigmoid)

```
        h₁ ──┐
    x₁ ──┤    ├──→ ŷ (sigmoid)
        h₂ ──┘
    x₂ ──┘
```

Hidden units might learn "is $x_1$ on?" and "is $x_2$ on?" - nonlinear combinations the output layer XORs via weights.

---

## NumPy: XOR Forward Pass (Hand-Initialized Weights)

Below, weights are chosen to demonstrate the mechanism - training learns them automatically in [Lab 08](./section-lab-08-neural-network-intuition-workshop.md).

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Shapes: W1 (2 hidden, 2 input), W2 (1 output, 2 hidden)
W1 = np.array([[1.0, 1.0],
               [1.0, 1.0]])
b1 = np.array([-0.5, -1.5])
W2 = np.array([[1.0, -2.0]])
b2 = np.array([-0.5])

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

def forward_xor(x):
    a0 = x
    z1 = W1 @ a0 + b1
    a1 = relu(z1)
    z2 = W2 @ a1 + b2
    a2 = sigmoid(z2)
    return a2, (z1, a1, z2)

for x in X_xor:
    y_hat, cache = forward_xor(x)
    print(f"x={x.astype(int)} → ŷ={y_hat[0]:.3f} → class {int(y_hat[0] > 0.5)}")
```

Expected: predictions align with XOR labels (0, 1, 1, 0) after reasonable training - hand-picked weights may need tuning; the lab trains them with gradient descent.

### Hand trace for input $x = [1, 0]$

**Hidden layer:**

$$
\mathbf{z}^{[1]} = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} + \begin{bmatrix} -0.5 \\ -1.5 \end{bmatrix} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}
$$

$$
\mathbf{a}^{[1]} = \text{ReLU}\big(\mathbf{z}^{[1]}\big) = \begin{bmatrix} 0.5 \\ 0 \end{bmatrix}
$$
> **Readable form:** hidden activations = [0.5, 0] after ReLU zeros the negative entry

**Output layer:**

$$
z^{[2]} = [1, -2] \cdot [0.5, 0]^T - 0.5 = 0.5 - 0.5 = 0
$$
> **Readable form:** The output preactivation combines the two hidden activations with weights, subtracts the bias, and lands at zero.

$$
\hat{y} = \sigma(0) = 0.5
$$
> **Readable form:** A zero logit maps to probability 0.5 under the sigmoid, exactly on the decision boundary.

Borderline - trained weights sharpen this to confident 1 for XOR label.

---

## Fully Connected (Dense) Layers

When every neuron in layer $\ell$ connects to every neuron in layer $\ell-1$, the layer is **fully connected** - Keras calls this `Dense`. 

Parameters in one Dense layer:

$$
\text{\# weights} = n^{[\ell]} \times n^{[\ell-1]} + n^{[\ell]} \text{ (biases)}
$$
> **Readable form:** parameter count = (outputs × inputs) + biases

Example: 784 inputs → 128 hidden = $784 \times 128 + 128 = 100{,}480$ weights - why [PCA](../chapter-06-principal-component-analysis/README.md) or convolution (Chapter 10) helps on images.

---

## Computational Graph View

Prosise diagrams networks as **graphs**:

- **Nodes** = operations (matmul, ReLU, loss)
- **Edges** = tensors flowing between ops

This graph is the backbone of automatic differentiation in TensorFlow and PyTorch. [Backpropagation](./section-06-backpropagation-and-gradient-descent.md) walks the graph backward.

```
x → matmul → +b → ReLU → matmul → +b → sigmoid → ŷ → loss
```

---

## Capacity, Underfitting, Overfitting

More layers/neurons = higher **capacity** (can fit more complex patterns):

| Symptom | Diagnosis | Part I parallel |
|---------|-----------|-----------------|
| High train & val loss | [Underfitting](../../GLOSSARY.md#underfitting) | Linear model on curved data ([Chapter 02](../chapter-02-regression-models/README.md)) |
| Low train, high val loss | [Overfitting](../../GLOSSARY.md#overfitting) | Memorizing noise |
| Low train & val loss | Good fit | Target state |

[Chapter 07](../chapter-07-operationalizing-models/README.md) taught monitoring drift in production - the same val curve that looks healthy on launch can degrade when data shifts.

---

## Comparison to Part I Nonlinear Models

| Approach | How nonlinearity enters | Training |
|----------|------------------------|----------|
| [Polynomial features](../chapter-02-regression-models/section-02-linear-regression.md) + linear | Explicit $x^2$ terms | Closed-form or convex |
| [Decision tree](../chapter-02-regression-models/section-03-decision-trees-for-regression.md) | Axis-aligned splits | Greedy search |
| [Random forest](../chapter-02-regression-models/section-04-ensemble-methods-random-forests-and-gradient-boosting.md) | Many trees averaged | Bagging |
| [RBF SVM](../chapter-05-support-vector-machines/section-02-kernels-and-the-kernel-trick.md) | Kernel feature space | Quadratic programming |
| **MLP** | Hidden layers + activations | Gradient descent |

No free lunch - MLPs shine when data is abundant and raw inputs are high-dimensional.

---

## Designing a Small MLP: Checklist

1. **Input size** = number of features (or flattened pixels)
2. **Hidden layers** = 1-2 for tabular; more for images (later chapters)
3. **Hidden width** = scale with problem difficulty; start 32-64
4. **Output size** = 1 (regression), 1 + sigmoid (binary), $K$ + softmax ($K$ classes)
5. **Activation** = ReLU hidden, task-specific output ([Section 8.7](./section-07-activation-functions-deep-dive.md))

---

## Common Mistakes (Section 8.3)

| Mistake | Why it hurts | Fix |
|---------|--------------|-----|
| **Zero hidden layers on nonlinear data** | Stuck at linear boundary | Add at least one hidden layer |
| **Mismatched weight matrix shapes** | Silent bugs or crashes | Draw $(n^{[\ell]}, n^{[\ell-1]})$ on paper |
| **Thousands of layers on 500 rows** | Guaranteed overfit | Match capacity to data ([Chapter 07](../chapter-07-operationalizing-models/README.md)) |
| **Confusing parameters with hyperparameters** | $W, b$ learned; layer count chosen | See [GLOSSARY](../../GLOSSARY.md#hyperparameter) |
| **Expecting universal approximation = easy training** | SGD may stall in bad minima | Use validation curves, learning rate tuning |

---

## Key Vocabulary

| Term | Definition |
|------|-----------|
| **MLP** | Multi-layer perceptron; feedforward fully connected network |
| **Hidden layer** | Layer between input and output; learns internal features |
| **Dense / fully connected** | Every input neuron connects to every output neuron |
| **Capacity** | Model's ability to fit complex functions |
| **Universal approximation** | One wide hidden layer can approximate continuous functions |
| **Feedforward** | Information flows input → output without cycles |

---

## Self-Check

1. How many parameters in Dense(128) after input of 50 features?
2. Why does XOR require a hidden layer?
3. What is $\mathbf{a}^{[0]}$?
4. Name two Part I models that also create nonlinear decision boundaries.

---

## Reflection

How is a hidden layer similar to and different from [PCA](../chapter-06-principal-component-analysis/README.md) components used as features?

---

## Further Reading

- Prosise, Ch. 8 - multi-layer networks and XOR
- [Section 8.4 - Forward Propagation](./section-04-forward-propagation.md)
- [Lab 08 - XOR workshop](./section-lab-08-neural-network-intuition-workshop.md)

---

**Previous:** [Section 8.2 - The Artificial Neuron](./section-02-the-artificial-neuron.md) | **Next:** [Section 8.4 - Forward Propagation](./section-04-forward-propagation.md)
