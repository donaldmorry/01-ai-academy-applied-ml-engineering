# Section 3.2: Logistic Regression

> **Source:** Prosise, Ch. 3 - "Logistic Regression"  
> **Prerequisites:** [Section 3.1](./section-01-classification-fundamentals.md) | [Section 2.2](../chapter-02-regression-models/section-02-linear-regression.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [linear regression](../../GLOSSARY.md#linear-regression) | [regularization](../../GLOSSARY.md#regularization)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Name Is a Trap

**Logistic regression** is **classification**, not [regression](../../GLOSSARY.md#regression). The "regression" part refers to the linear score inside the model - the output is a **probability**, then a class.

Prosise uses it as the **interpretable workhorse** for Titanic survival and as the baseline for every tabular binary problem in industry.

> **Humorous analogy:** Logistic regression is a regression model wearing a classification costume. It computes a straight-line score, then squashes it through a sigmoid so it looks like a probability.

---

## From Linear Score to Probability: The Sigmoid

[Linear regression](../../GLOSSARY.md#linear-regression) from [Section 2.2](../chapter-02-regression-models/section-02-linear-regression.md) gives unbounded outputs:

$$
z = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n = \mathbf{w}^T \mathbf{x}
$$
> **Readable form:** linear score z = intercept + (weight₁ × feature₁) + (weight₂ × feature₂) + …

For [classification](../../GLOSSARY.md#classification), we need $p \in [0, 1]$. Enter the **sigmoid** (logistic function):

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
> **Readable form:** probability = 1 divided by (1 plus e raised to the negative linear score)

| $z$ | $\sigma(z)$ | Interpretation |
|-----|-------------|----------------|
| $-\infty$ | 0 | Impossible |
| 0 | 0.5 | Coin flip |
| $+\infty$ | 1 | Certain |

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-6, 6, 200)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(7, 4))
plt.plot(z, sigmoid, 'b-', linewidth=2)
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Linear score z')
plt.ylabel('Probability σ(z)')
plt.title('Sigmoid: S-Curve from Score to Probability')
plt.grid(True, alpha=0.3)
plt.show()
```

**Real-life analogy:** The sigmoid is a dimmer switch. The linear score is how hard you twist; the output brightness is probability - never below 0 or above 1.

---

## The Full Model

For binary [classification](../../GLOSSARY.md#classification) with [label](../../GLOSSARY.md#label) $y \in \{0, 1\}$:

$$
P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}}
$$
> **Readable form:** probability of class 1 = sigmoid applied to (dot product of weights and features)

**Hard prediction** (default threshold 0.5):

$$
\hat{y} = \begin{cases} 1 & \text{if } P(y=1 \mid \mathbf{x}) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}
$$
> **Readable form:** predict class 1 if probability is at least 0.5, else class 0

---

## Training: Log-Loss (Cross-Entropy), Not MSE

You might be tempted to use [MSE](../../GLOSSARY.md#mse) from regression. Don't - MSE on 0/1 labels gives a non-convex mess. Logistic regression minimizes **log-loss** (binary cross-entropy):

$$
\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\left[ y_i \log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i) \right]
$$
> **Readable form:** average loss = negative average of [ yᵢ × log(predicted probability of class 1) + (1−yᵢ) × log(1 − that probability) ]

| True $y$ | Predicted $p$ | Penalty |
|----------|---------------|---------|
| 1 | 0.99 | Tiny |
| 1 | 0.01 | **Huge** |
| 0 | 0.01 | Tiny |
| 0 | 0.99 | **Huge** |

> **In plain English:** Log-loss screams when you're confidently wrong. Predicting 99% survived for someone who died hurts far more than predicting 55%.

Scikit-Learn's `LogisticRegression` solves this with L-BFGS or other optimizers - you rarely code the gradient yourself.

---

## Interpreting Coefficients: Odds and Log-Odds

Each weight $w_j$ affects **log-odds**:

$$
\log\left(\frac{p}{1-p}\right) = w_0 + w_1 x_1 + \cdots + w_n x_n
$$
> **Readable form:** log of (probability ÷ (1 − probability)) = linear combination of features

| Coefficient sign | Meaning |
|------------------|---------|
| $w_j > 0$ | Increasing $x_j$ **increases** odds of class 1 |
| $w_j < 0$ | Increasing $x_j$ **decreases** odds of class 1 |

**Titanic intuition (Prosise):** Negative coefficient on `pclass` (3rd class) → lower survival odds. Positive on `sex=female` → higher odds. Executives and historians can audit these - unlike a black-box forest.

**Odds ratio:** $e^{w_j}$ = multiplicative change in odds per +1 unit of $x_j$ (for continuous features).

---

## Code: Titanic-Style Binary Classification

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Minimal synthetic Titanic-like data
np.random.seed(42)
n = 800
df = pd.DataFrame({
    'pclass': np.random.choice([1, 2, 3], n, p=[0.2, 0.3, 0.5]),
    'sex': np.random.choice([0, 1], n, p=[0.65, 0.35]),  # 0=male, 1=female
    'age': np.random.normal(30, 12, n).clip(1, 70),
})
# Survival probability model (simplified Prosise logic)
log_odds = -1.5 - 0.8 * (df['pclass'] - 1) + 1.2 * df['sex'] - 0.02 * (df['age'] - 30)
df['survived'] = (np.random.rand(n) < 1 / (1 + np.exp(-log_odds))).astype(int)

X = df[['pclass', 'sex', 'age']]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = make_pipeline(
    StandardScaler(),  # important for age scale vs one-hot later
    LogisticRegression(max_iter=1000, random_state=42)
)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, target_names=['Perished', 'Survived']))

# Coefficients (after scaling)
lr = pipe.named_steps['logisticregression']
for feat, coef in zip(X.columns, lr.coef_[0]):
    print(f"  {feat:10s}: {coef:+.3f}")
```

---

## Multi-Class: One-vs-Rest (OvR)

MNIST has 10 digits. Scikit-Learn's default `multi_class='ovr'` trains **one binary logistic model per class**:

- "Is it a 0?" vs "not 0"
- "Is it a 1?" vs "not 1"
- … ten times

Pick the class with highest probability. Alternative: `multi_class='multinomial'` with softmax - better calibrated for mutually exclusive classes. Details in [Section 3.8](./section-08-case-study-mnist-digit-recognition.md).

---

## [Regularization](../../GLOSSARY.md#regularization): Don't Let Weights Run Wild

Like Ridge in [Section 2.2](../chapter-02-regression-models/section-02-linear-regression.md), logistic regression supports L2 (default `penalty='l2'`) and L1 (`penalty='l1'`, needs `solver='saga'`):

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{log-loss}} + \lambda \sum_j w_j^2
$$
> **Readable form:** total loss = log-loss + lambda × (sum of squared weights)

| Setting | Effect |
|---------|--------|
| Large $\lambda$ (small `C` in sklearn - `C = 1/λ`) | Simpler model, less [overfitting](../../GLOSSARY.md#overfitting) |
| Small $\lambda$ (large `C`) | Fits training data tighter |

```python
from sklearn.model_selection import cross_val_score

for C in [0.01, 0.1, 1.0, 10.0]:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"C={C:5.2f}  CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## Decision Boundary (2D Visualization)

With two features, logistic regression's boundary is a **line** where $P(y=1) = 0.5$, i.e. $\mathbf{w}^T \mathbf{x} = 0$:

```python
from sklearn.datasets import make_classification

X2, y2 = make_classification(n_samples=200, n_features=2, n_redundant=0,
                              n_clusters_per_class=1, random_state=42)
model2d = LogisticRegression().fit(X2, y2)

xx, yy = np.meshgrid(np.linspace(X2[:,0].min()-1, X2[:,0].max()+1, 200),
                     np.linspace(X2[:,1].min()-1, X2[:,1].max()+1, 200))
Z = model2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=50, alpha=0.4, cmap='RdBu')
plt.scatter(X2[:,0], X2[:,1], c=y2, edgecolors='k', s=40)
plt.title('Logistic Regression Decision Boundary (linear)')
plt.show()
```

Trees in [Section 3.5](./section-05-tree-based-classifiers.md) draw staircase boundaries instead.

---

## Logistic vs Linear Regression on 0/1 Labels

| Approach | Output | Problem |
|----------|--------|---------|
| `LinearRegression` | 0.73, -0.12, 1.4 | Not valid probabilities; heteroscedastic errors |
| `LogisticRegression` | 0.73 → class 1 | Bounded, probabilistic, proper loss |

---

## When Logistic Regression Wins - and When It Doesn't

### Wins

- Need **interpretable** coefficients (healthcare, finance, policy)
- Linearly separable or roughly linear log-odds
- Fast baseline on tabular data
- Calibrated probabilities for ranking

### Struggles

| Issue | Symptom | Try instead |
|-------|---------|-------------|
| Complex interactions | Poor accuracy vs trees | [Random forest](./section-05-tree-based-classifiers.md), boosting |
| Unscaled mixed features | Slow convergence | `StandardScaler` + pipeline |
| Extreme imbalance | Predicts majority only | Class weights, metrics in [3.3](./section-03-classification-accuracy-measures.md) |
| Nonlinear boundaries | High train error | Trees, k-NN, SVM with kernels |

---

## Prosise's Titanic Preview

Full pipeline in [Section 3.6](./section-06-case-study-titanic-survival.md). Logistic regression there answers: *"Holding other features fixed, how does being female change survival odds?"* - a question random forests answer only indirectly.

---

## Self-Check

1. Why can't we use MSE as the primary loss for logistic regression?
2. If $w_{\text{sex}} = 1.2$, does being female increase or decrease survival probability?
3. What does `predict_proba` return that `predict` does not?
4. What is the decision boundary when $P(y=1) = 0.5$?

---

## References

- [Scikit-Learn LogisticRegression](https://scikit-learn.org/stable/chapters/generated/sklearn.linear_model.LogisticRegression.html)
- [Scikit-Learn: Linear Models - Logistic Regression](https://scikit-learn.org/stable/chapters/linear_model.html#logistic-regression)
- Prosise, *Applied ML and AI for Engineers*, Ch. 3 - Logistic Regression
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 3.1](./section-01-classification-fundamentals.md) | **Next:** [Section 3.3 - Accuracy Measures](./section-03-classification-accuracy-measures.md)



