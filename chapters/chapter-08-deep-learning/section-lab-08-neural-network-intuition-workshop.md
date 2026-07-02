# Lab 08: Neural Network Intuition Workshop

> **Prerequisites:** Sections [8.1](./section-01-why-deep-learning.md)-[8.8](./section-08-training-loop-concepts.md)  
> **Estimated time:** 3-4 hours  
> **Tools:** Python 3.10+, Jupyter, NumPy, Matplotlib, scikit-learn  
> **Glossary:** [../../GLOSSARY.md](../../GLOSSARY.md) | **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Lab Objectives

By the end of this lab you will:

1. Implement a **manual forward pass** and verify that a single perceptron fails on XOR while a 2-layer network succeeds
2. **Plot and compare** sigmoid, tanh, and ReLU - highlighting vanishing-gradient regions
3. Train a **minimal NumPy network** on `make_moons` and visualize the decision boundary
4. Explore **loss vs weight** curves to connect learning rate to convergence
5. Write short answers tying results to [Chapter 02](../chapter-02-regression-models/README.md)-[07](../chapter-07-operationalizing-models/README.md) concepts

**No Keras in this lab** - that is [Lab 09](../chapter-09-neural-networks/README.md). Every exercise builds Chapter 8 intuition Prosise emphasizes before framework syntax.

---

## Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

np.random.seed(42)
plt.rcParams['figure.figsize'] = (8, 6)
```

Create notebook: `lab-08-neural-intuition.ipynb`.

---

## Part A: Manual Forward Pass & XOR (60 min)

### A.1 Single perceptron failure

From [Section 8.2](./section-02-the-artificial-neuron.md):

```python
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

def perceptron_predict(X, w, b):
    return (X @ w + b > 0).astype(int)

# Students: search or prove no (w, b) works
best_acc = 0
for _ in range(1000):
    w = np.random.randn(2)
    b = np.random.randn()
    acc = (perceptron_predict(X_xor, w, b) == y_xor).mean()
    best_acc = max(best_acc, acc)
print(f"Best random perceptron accuracy: {best_acc:.0%}")  # 75% typical
```

**Written (markdown):** Why is 75% the common ceiling? Reference linear separability.

### A.2 Two-layer forward pass (hand trace + code)

Implement forward for architecture $2 \rightarrow 4 \rightarrow 1$ with ReLU hidden, sigmoid output:

```python
def relu(z): return np.maximum(0, z)
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def init_params(n_in=2, n_hid=4, n_out=1, scale=0.5):
    return {
        'W1': np.random.randn(n_hid, n_in) * scale,
        'b1': np.zeros(n_hid),
        'W2': np.random.randn(n_out, n_hid) * scale,
        'b2': np.zeros(n_out),
    }

def forward(params, X):
    Z1 = X @ params['W1'].T + params['b1']
    A1 = relu(Z1)
    Z2 = A1 @ params['W2'].T + params['b2']
    A2 = sigmoid(Z2)
    return A2, {'Z1': Z1, 'A1': A1, 'Z2': Z2}

params = init_params()
for i, x in enumerate(X_xor):
    y_hat, cache = forward(params, x.reshape(1, -1))
    print(f"x={x.astype(int)} ŷ={y_hat[0,0]:.3f}")
```

**Task:** Pick one input (e.g. `[1, 0]`) and hand-trace $z^{[1]}$, $a^{[1]}$, $z^{[2]}$, $\hat{y}$ in a markdown cell - mirror [Section 8.3](./section-03-multi-layer-networks.md).

### A.3 Train XOR with backprop

```python
def train_xor(params, X, y, lr=0.5, epochs=8000):
    losses = []
    for epoch in range(epochs):
        A2, cache = forward(params, X)
        m = len(X)
        loss = -np.mean(y*np.log(A2+1e-9) + (1-y)*np.log(1-A2+1e-9))
        losses.append(loss)

        dZ2 = (A2 - y) / m
        params['W2'] -= lr * (dZ2.T @ cache['A1'])
        params['b2'] -= lr * dZ2.sum(axis=0)
        dA1 = dZ2 @ params['W2']
        dZ1 = dA1 * (cache['Z1'] > 0)
        params['W1'] -= lr * (dZ1.T @ X)
        params['b1'] -= lr * dZ1.sum(axis=0)

        if epoch % 2000 == 0:
            acc = ((A2 > 0.5).astype(int) == y).mean()
            print(f"epoch {epoch}: loss={loss:.4f}, acc={acc:.0%}")
    return losses

params = init_params(scale=0.8)
losses = train_xor(params, X_xor, y_xor.reshape(-1, 1))
```

Plot loss curve. Confirm **100%** training accuracy.

**Written:** Explain in 3-5 sentences why depth was necessary. Link to [kernel trick](../chapter-05-support-vector-machines/section-02-kernels-and-the-kernel-trick.md) optional comparison.

---

## Part B: Activation Comparison (45 min)

From [Section 8.7](./section-07-activation-functions-deep-dive.md):

```python
z = np.linspace(-6, 6, 400)
activations = {
    'sigmoid': 1 / (1 + np.exp(-z)),
    'tanh': np.tanh(z),
    'relu': np.maximum(0, z),
}

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for i, (name, a) in enumerate(activations.items()):
    axes[0, i].plot(z, a)
    axes[0, i].set_title(f'{name}')
    axes[0, i].axhline(0, color='k', lw=0.5)
    axes[0, i].axvline(0, color='k', lw=0.5)

    if name == 'sigmoid':
        ap = a * (1 - a)
    elif name == 'tanh':
        ap = 1 - a**2
    else:
        ap = (z > 0).astype(float)
    axes[1, i].plot(z, ap)
    axes[1, i].set_title(f"{name} derivative")
    if name == 'sigmoid':
        axes[1, i].axhspan(0, 0.05, alpha=0.3, color='red', label='vanishing zone')
        axes[1, i].legend()
plt.tight_layout()
```

**Written questions:**

1. What is the maximum sigmoid derivative? Where does it occur?
2. Why does ReLU help [backpropagation](../../GLOSSARY.md#backpropagation) in deep nets?
3. Which activation would you use for [Titanic](../chapter-03-classification-models/section-06-case-study-titanic-survival.md) binary output?

---

## Part C: Decision Boundary on Moons (75 min)

Nonlinear 2D data - linear [logistic regression](../chapter-03-classification-models/section-02-logistic-regression.md) baseline vs tiny MLP.

```python
X, y = make_moons(n_samples=400, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Baseline - Chapter 03
lr_clf = LogisticRegression().fit(X_train_s, y_train)
print(f"Linear logistic train acc: {lr_clf.score(X_train_s, y_train):.2%}")
print(f"Linear logistic test acc:  {lr_clf.score(X_test_s, y_test):.2%}")
```

### C.1 NumPy MLP trainer (reuse Part A functions)

Extend `init_params` / `forward` / backward for binary CE. Architecture: $2 \rightarrow 8 \rightarrow 4 \rightarrow 1$.

```python
def train_mlp(X, y, n_hid=(8, 4), lr=0.1, epochs=3000, batch_size=32):
    # Students: implement or use provided template
    # Return trained params and loss history
    pass  # replace with your implementation
```

**Minimum deliverable:** Test accuracy **> 90%** on moons (typically achievable).

### C.2 Decision boundary plot

```python
def plot_decision_boundary(predict_fn, X, y, title):
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200),
        np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_s = scaler.transform(grid)
    Z = predict_fn(grid_s).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k')
    plt.title(title)

# Plot side-by-side: logistic vs MLP
```

**Written:** Compare boundary shapes to [decision trees](../chapter-02-regression-models/section-03-decision-trees-for-regression.md) and [RBF SVM](../chapter-05-support-vector-machines/README.md).

---

## Part D: Loss Landscape & Learning Rate (45 min)

Single weight $w$, model $\hat{y} = \sigma(w \cdot x)$ on one point $(x=1, y=1)$:

$$
\mathcal{L}(w) = -\log(\sigma(w))
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

```python
w_range = np.linspace(-6, 6, 300)
losses = [-np.log(1 / (1 + np.exp(-w))) for w in w_range]

plt.figure()
plt.plot(w_range, losses)
plt.xlabel('w'); plt.ylabel('BCE loss'); plt.title('Loss vs single weight')
opt_w = w_range[np.argmin(losses)]
plt.axvline(opt_w, color='g', ls='--', label=f'min near w={opt_w:.1f}')
plt.legend()
```

Simulate gradient descent from $w_0 = -4$ with $\eta \in \{0.1, 0.5, 2.0\}$:

```python
def gd_single_weight(w0, lr, steps=20):
    w = w0
    trajectory = [w]
    for _ in range(steps):
        sig = 1 / (1 + np.exp(-w))
        grad = sig - 1  # dBCE/dw for y=1, x=1
        w = w - lr * grad
        trajectory.append(w)
    return trajectory

for eta in [0.1, 0.5, 2.0]:
    traj = gd_single_weight(-4.0, eta)
    print(f"eta={eta}: final w={traj[-1]:.3f}")
```

**Written:** Describe behavior for each learning rate. Relate to [Section 8.6](./section-06-backpropagation-and-gradient-descent.md).

---

## Part E: Reflection & Cross-Chapter Synthesis (30 min)

Answer in markdown (2-4 sentences each):

1. **Chapter 02:** How is MSE in a linear output neuron related to [taxi fare RMSE](../chapter-02-regression-models/section-06-regression-accuracy-measures.md)?
2. **Chapter 03:** How is sigmoid + BCE identical to [logistic regression](../chapter-03-classification-models/section-02-logistic-regression.md)?
3. **Chapter 04:** Why might TF-IDF + Naive Bayes still beat a tiny MLP on small text data?
4. **Chapter 06:** How is a hidden layer different from [PCA](../chapter-06-principal-component-analysis/README.md) features fed to logistic regression?
5. **Chapter 07:** What validation discipline from deployment applies to `val_loss` during training?

---

## Deliverables Checklist

| Item | Required |
|------|----------|
| `lab-08-neural-intuition.ipynb` with all parts runnable | ✓ |
| XOR: perceptron fail + 2-layer success + loss plot | ✓ |
| Activation comparison figure (6 subplots) | ✓ |
| Moons: logistic vs MLP boundaries side-by-side | ✓ |
| Loss vs $w$ plot + learning rate discussion | ✓ |
| Part E written answers | ✓ |

---

## Grading Rubric (Self-Assessment)

| Criterion | Meets expectations |
|-----------|-------------------|
| **Forward pass** | Hand trace matches code |
| **XOR training** | 100% train accuracy |
| **Activations** | Vanishing zone shaded on sigmoid |
| **Moons MLP** | Test accuracy > 90% |
| **Learning rate** | Identifies stable vs divergent η |

---

## Common Pitfalls

| Issue | Fix |
|-------|-----|
| XOR stuck at 75% | Increase hidden units; try lr=0.5-1.0; train longer |
| `nan` loss | Clip sigmoid inputs; reduce learning rate |
| Moons accuracy low | Standardize; add hidden units; train more epochs |
| Boundary plot wrong | Apply **same** scaler to grid as training data |
| Shapes break in backprop | Print `.shape` after every matmul ([Section 8.4](./section-04-forward-propagation.md)) |

---

## What's Next

[Chapter 09](../chapter-09-neural-networks/README.md): same problems with `model.compile()` and `model.fit()`.

---

**Previous:** [Section 8.8](./section-08-training-loop-concepts.md) | **Chapter README:** [Chapter 08](./README.md)



