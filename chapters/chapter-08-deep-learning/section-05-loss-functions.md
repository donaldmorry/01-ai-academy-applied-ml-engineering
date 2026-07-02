# Section 8.5: Loss Functions

> **Source:** Prosise, Ch. 8 - "Loss Functions"  
> **Prerequisites:** [Section 8.4](./section-04-forward-propagation.md) | [Section 2.6](../chapter-02-regression-models/section-06-regression-accuracy-measures.md) | [Section 3.3](../chapter-03-classification-models/section-03-classification-accuracy-measures.md)  
> **Glossary:** [MSE](../../GLOSSARY.md#mse) | [regression](../../GLOSSARY.md#regression) | [classification](../../GLOSSARY.md#classification)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Loss: The Training Objective

A **loss function** (cost function) measures how wrong predictions are:

$$
\mathcal{L} = \ell(\hat{y}, y)
$$
> **Readable form:** loss = penalty comparing prediction to true target

Training searches parameters $\theta = \{\mathbf{W}^{[\ell]}, \mathbf{b}^{[\ell]}\}$ to minimize **average loss** over the training set:

$$
\mathcal{L}_{\text{avg}} = \frac{1}{m}\sum_{i=1}^{m} \ell(\hat{y}_i, y_i)
$$
> **Readable form:** average loss = (1/m) × sum of per-example penalties

Prosise emphasizes: **pick the loss that matches your task** - wrong loss = right code, wrong problem ([Chapter 03](../chapter-03-classification-models/README.md) fraud vs [Chapter 02](../chapter-02-regression-models/README.md) taxi).

---

## Regression: Mean Squared Error (MSE)

Identical to [Section 2.6](../chapter-02-regression-models/section-06-regression-accuracy-measures.md):

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2
$$
> **Readable form:** MSE = average of (actual − predicted)²

| Property | Implication |
|----------|-------------|
| Penalizes large errors heavily | Outliers dominate gradients |
| Smooth | Good for gradient descent |
| Same units as target squared | [RMSE](../../GLOSSARY.md#rmse) = $\sqrt{\text{MSE}}$ for reporting in USD |

**Output activation:** linear (identity) - $\hat{y}$ can be any real number.

### NumPy hand trace

```python
import numpy as np

y_true = np.array([3.0, 5.0, 2.0])
y_hat  = np.array([2.5, 6.0, 2.0])

mse = np.mean((y_true - y_hat) ** 2)
print(f"MSE = {mse:.4f}")  # ((0.5)² + (1)² + 0) / 3 = 0.4167
```

For taxi fares in [Chapter 02](../chapter-02-regression-models/section-08-taxi-fare-prediction-end-to-end-project.md), Keras uses `loss='mse'` in [Chapter 09](../chapter-09-neural-networks/README.md).

---

## Binary Classification: Binary Cross-Entropy

For $y \in \{0, 1\}$ and predicted probability $\hat{p}$:

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{m}\sum_{i=1}^{m}\Big[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\Big]
$$
> **Readable form:** BCE = negative average of [ y×log(p̂) + (1−y)×log(1−p̂) ]

| True $y$ | What loss punishes |
|----------|-------------------|
| 1 | Low $\hat{p}$ → large $-\log(\hat{p})$ |
| 0 | High $\hat{p}$ → large $-\log(1-\hat{p})$ |

This is the loss [logistic regression](../chapter-03-classification-models/section-02-logistic-regression.md) minimizes - not [accuracy](../../GLOSSARY.md#accuracy), which is non-differentiable.

**Output activation:** sigmoid producing $\hat{p}$.

### Hand trace: one example

$y = 1$, $\hat{p} = 0.7$:

$$
\ell = -\log(0.7) \approx 0.357
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

$y = 1$, $\hat{p} = 0.01$ (confidently wrong):

$$
\ell = -\log(0.01) \approx 4.605
$$
> **Readable form:** confident mistakes hurt exponentially more

```python
def binary_cross_entropy(y, p_hat, eps=1e-15):
    p_hat = np.clip(p_hat, eps, 1 - eps)
    return -np.mean(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))

y = np.array([1, 0, 1, 1])
p = np.array([0.9, 0.1, 0.6, 0.8])
print(f"BCE = {binary_cross_entropy(y, p):.4f}")
```

Connects to [credit card fraud](../chapter-03-classification-models/section-07-case-study-credit-card-fraud-detection.md) - class imbalance handled with weights, but loss remains BCE.

---

## Multi-Class: Categorical Cross-Entropy

For one-hot label $\mathbf{y}$ (K classes) and softmax probabilities $\hat{\mathbf{p}}$:

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_{i,k} \log(\hat{p}_{i,k})
$$
> **Readable form:** CE = negative average of sum over classes of (true indicator × log predicted probability)

Only the **true class** contributes (others are 0 in one-hot).

**Output activation:** softmax.

### Integer labels + sparse encoding

When labels are integers $y \in \{0,\ldots,K-1\}$ instead of one-hot:

$$
\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m} \log(\hat{p}_{i, y_i})
$$
> **Readable form:** CE = negative average of log-probability assigned to the correct class

```python
def sparse_ce(y_int, probs):
  m = len(y_int)
  return -np.mean(np.log(probs[np.arange(m), y_int] + 1e-15))

# 3 examples, 4 classes
probs = np.array([[0.7, 0.1, 0.1, 0.1],
                  [0.2, 0.5, 0.2, 0.1],
                  [0.1, 0.1, 0.7, 0.1]])
labels = np.array([0, 1, 2])
print(f"Sparse CE = {sparse_ce(labels, probs):.4f}")
```

[MNIST digit recognition](../chapter-03-classification-models/section-08-case-study-mnist-digit-recognition.md) uses 10-way CE - neural version in Chapter 09.

---

## Why Not Use Accuracy as Loss?

| Metric | Differentiable? | Use in training? |
|--------|-----------------|------------------|
| MSE / cross-entropy | Yes | Yes |
| [Accuracy](../../GLOSSARY.md#accuracy) | No (step function) | No - monitor only |
| [F1](../chapter-03-classification-models/section-03-classification-accuracy-measures.md) | No | No - threshold tuning |

Gradients need smooth loss landscapes. Accuracy flatlines between weight updates - useless for [gradient descent](../../GLOSSARY.md#gradient-descent).

---

## Loss vs Metric (Keras Preview)

In [Chapter 09](../chapter-09-neural-networks/README.md):

```python
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

- **loss** - what optimizers minimize
- **metrics** - human-readable dashboards ([Chapter 07](../chapter-07-operationalizing-models/README.md))

---

## Per-Example vs Batch Loss

```python
def mse_per_example(y_true, y_hat):
    return (y_true - y_hat) ** 2  # shape (m,)

batch_loss = np.mean(mse_per_example(y_true, y_hat))
```

Mini-batch gradient descent uses batch loss on subsets - introduces noise that often helps generalization ([Section 8.8](./section-08-training-loop-concepts.md)).

---

## Choosing the Right Loss: Decision Tree

```
Is target continuous?
├── Yes → MSE (or MAE / Huber for outlier robustness)
└── No → How many classes?
    ├── 2 → Binary cross-entropy + sigmoid
    └── K > 2 → Categorical cross-entropy + softmax
```

| Part I project | Loss |
|----------------|------|
| [Taxi fare](../chapter-02-regression-models/section-08-taxi-fare-prediction-end-to-end-project.md) | MSE |
| [Titanic](../chapter-03-classification-models/section-06-case-study-titanic-survival.md) | BCE |
| [Fraud](../chapter-03-classification-models/section-07-case-study-credit-card-fraud-detection.md) | BCE + class weights |
| [MNIST](../chapter-03-classification-models/section-08-case-study-mnist-digit-recognition.md) | Categorical CE |

---

## Regularization Terms (Preview)

[Section 2.2](../chapter-02-regression-models/section-02-linear-regression.md) Ridge adds $\lambda \sum w_j^2$ to MSE. Neural networks add similar **L2 weight decay** in [Chapter 09](../chapter-09-neural-networks/README.md):

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \sum_{\ell} \|\mathbf{W}^{[\ell]}\|_F^2
$$
> **Readable form:** total loss = prediction loss + lambda × sum of squared weights

Fights [overfitting](../../GLOSSARY.md#overfitting) alongside dropout.

---

## Numerical Stability

$\log(0)$ explodes. Production code clips probabilities:

```python
p_hat = np.clip(p_hat, 1e-15, 1 - 1e-15)
```

Frameworks use **log-softmax** fused with CE for the same reason - you will not implement this manually in Keras.

---

## Connecting Loss to Forward Pass

Full training step skeleton:

```python
# 1. Forward
y_hat, cache = forward(X, params)

# 2. Loss
if task == 'regression':
    loss = np.mean((y - y_hat) ** 2)
elif task == 'binary':
    p = sigmoid(y_hat)
    loss = binary_cross_entropy(y, p)

# 3. Backward (Section 8.6)
# 4. Update weights
```

Loss is the **scalar** that backprop differentiates.

---

## Common Mistakes (Section 8.5)

| Mistake | Why it hurts | Fix |
|---------|--------------|-----|
| **MSE for classification** | Wrong gradients, nonsense predictions | Use cross-entropy |
| **Sigmoid + CE on multi-class** | Competing classes not normalized | Softmax + categorical CE |
| **Forgetting to match output activation** | BCE needs probabilities in (0,1) | Pair loss with correct $\phi$ |
| **Reporting loss as accuracy** | Stakeholder confusion | Track both separately |
| **Ignoring class imbalance in loss** | Model predicts majority class | Class weights ([fraud section](../chapter-03-classification-models/section-07-case-study-credit-card-fraud-detection.md)) |

---

## Key Vocabulary

| Term | Definition |
|------|-----------|
| **Loss / cost** | Scalar measuring prediction error for training |
| **[MSE](../../GLOSSARY.md#mse)** | Mean squared error for regression |
| **Binary cross-entropy** | Loss for two-class probability outputs |
| **Categorical cross-entropy** | Loss for multi-class softmax outputs |
| **One-hot encoding** | Label vector with 1 at true class |
| **Sparse labels** | Integer class index instead of one-hot |

---

## Self-Check

1. Which loss for house price prediction? For spam vs ham? For digits 0-9?
2. Why is $-\log(0.01)$ larger than $-\log(0.5)$?
3. If softmax gives $\hat{p} = [0.9, 0.05, 0.05]$ and true class is 0, what is CE?
4. Why can't we train directly on accuracy?

---

## Further Reading

- Prosise, Ch. 8 - loss functions
- [Section 2.6 - Regression Accuracy](../chapter-02-regression-models/section-06-regression-accuracy-measures.md)
- [Section 3.3 - Classification Accuracy](../chapter-03-classification-models/section-03-classification-accuracy-measures.md)
- [Section 8.6 - Backpropagation](./section-06-backpropagation-and-gradient-descent.md)

---

**Previous:** [Section 8.4 - Forward Propagation](./section-04-forward-propagation.md) | **Next:** [Section 8.6 - Backpropagation & Gradient Descent](./section-06-backpropagation-and-gradient-descent.md)



