# Section 3.1: Classification Fundamentals

> **Source:** Prosise, Ch. 3 - Opening & classification framing  
> **Prerequisites:** [Chapter 01](../chapter-01-machine-learning/README.md) | [Chapter 02](../chapter-02-regression-models/README.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [supervised learning](../../GLOSSARY.md#supervised-learning)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## From Numbers to Categories

In [Chapter 02](../chapter-02-regression-models/README.md), you predicted **continuous numbers** - taxi fares, salaries, temperatures. [Classification](../../GLOSSARY.md#classification) flips the output type: instead of "how much?" you answer **"which bucket?"**

| Task | Output | Example question |
|------|--------|------------------|
| [Regression](../../GLOSSARY.md#regression) | Real number | How much will this fare cost? |
| **Classification** | Category / class | Did this passenger survive? Is this transaction fraud? Which digit is this? |

**Real-life analogy:** Regression is a thermometer ("72°F"). Classification is a bouncer at a club ("You're on the list" vs "Not tonight").

Prosise opens Chapter 3 with exactly this shift - the same [supervised learning](../../GLOSSARY.md#supervised-learning) workflow from [Section 1.6](../chapter-01-machine-learning/section-06-the-ml-workflow-and-data-hygiene.md), but the [label](../../GLOSSARY.md#label) is discrete.

---

## Types of Classification Problems

### Binary classification (two classes)

Exactly two outcomes:

| Domain | Class 0 | Class 1 |
|--------|---------|---------|
| Email | Ham | Spam |
| Titanic | Perished | Survived |
| Credit card | Legitimate | Fraud |

Mathematically, we often encode as $y \in \{0, 1\}$.

### Multi-class classification (three or more)

One sample belongs to **exactly one** class:

- MNIST digits: $y \in \{0, 1, 2, \ldots, 9\}$
- Iris species: setosa, versicolor, virginica (you saw this in [Section 1.5](../chapter-01-machine-learning/section-05-supervised-learning-k-nearest-neighbors.md))

### Multi-label classification (advanced preview)

One sample can have **multiple** labels simultaneously - "this photo contains both cat and couch." Not covered deeply in Prosise Ch. 3, but good to name so you don't confuse it with multi-class.

> **Humorous analogy:** Binary = yes/no on a survey. Multi-class = picking your one favorite pizza topping. Multi-label = checking every topping you like - pineapple **and** jalapeños, for the brave.

---

## The Classification Contract

Given [features](../../GLOSSARY.md#feature) $\mathbf{x} = (x_1, x_2, \ldots, x_n)$, learn a function:

$$
\hat{y} = f(\mathbf{x}) \in \mathcal{C}
$$
> **Readable form:** predicted class = some function of the input features, where the answer must be one of the allowed class names in set C

For binary problems, many algorithms first learn a **score** or **probability** $p \in [0, 1]$, then threshold:

$$
\hat{y} = \begin{cases} 1 & \text{if } p \geq 0.5 \\ 0 & \text{otherwise} \end{cases}
$$
> **Readable form:** if predicted probability is at least 0.5, predict class 1; otherwise predict class 0

The threshold $0.5$ is a default, not a law - fraud detection often uses $0.01$ or lower. More in [Section 3.7](./section-07-case-study-credit-card-fraud-detection.md).

---

## Decision Boundaries: Where the Model Draws the Line

In 2D with two features, a classifier partitions the plane into regions - one per class. The **decision boundary** is the border between regions.

```
        Feature x₂
           ↑
     Class B |  Class A
    ·  ·  · | ·  ·  ·
    ·  ·  · | ·  ·  ·
  ----------+----------  ← decision boundary
    ·  ·  · | ·  ·  ·
           → Feature x₁
```

| Algorithm | Boundary shape | Analogy |
|-----------|------------------|---------|
| [Logistic regression](./section-02-logistic-regression.md) | Straight line (linear) | Folding a piece of paper to separate two piles |
| [Decision tree](./section-05-tree-based-classifiers.md) | Axis-aligned rectangles | Cutting a sheet with only vertical/horizontal scissors |
| [k-NN](../chapter-01-machine-learning/section-05-supervised-learning-k-nearest-neighbors.md) | Wiggly, local | "Ask your neighbors" - boundary follows data density |
| SVM ([Chapter 05](../chapter-05-support-vector-machines/README.md)) | Can be linear or curved (kernel) | Finding the widest street between classes |

> **In plain English:** If you plot passenger age vs ticket price, a logistic model might draw one straight line: "above this line → more likely survived." A tree might say "if sex=female AND class=1 → survived" - a box-shaped region, not a diagonal line.

---

## Baselines: The "Always Guess the Majority" Model

Before any fancy algorithm, establish a **baseline** - the simplest strategy that still counts as prediction.

**Strategy:** Always predict the most common class in the training set.

$$
\hat{y}_{\text{baseline}} = \arg\max_{c \in \mathcal{C}} \; \text{count}(y = c \text{ in training})
$$
> **Readable form:** baseline prediction = whichever class appears most often in the training labels

```python
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Titanic-style: ~62% perished
y_train = np.array([0]*623 + [1]*377)  # 0=died, 1=survived
y_test  = np.array([0, 0, 1, 0, 0, 0, 1, 0])

baseline_class = pd.Series(y_train).mode()[0]  # 0
y_pred_baseline = np.full_like(y_test, baseline_class)

print(f"Baseline accuracy: {accuracy_score(y_test, y_pred_baseline):.2%}")
# On a realistic test set matching 62/38 split, baseline ≈ 62% - not 50%!
```

> **In plain English:** On the Titanic, always guessing "died" is depressing - but it's right about 62% of the time. Your model must beat that or you're shipping sophisticated wrongness.

**Every classification project in this chapter starts here.** [Accuracy](../../GLOSSARY.md#accuracy) alone is dangerous on imbalanced data - see [Section 3.3](./section-03-classification-accuracy-measures.md).

---

## Algorithms You'll Master in This Chapter

Prosise's Chapter 3 progression:

```
Majority baseline  →  Logistic Regression  →  Decision Trees
        →  Random Forest / Gradient Boosting  →  k-NN / SVM (MNIST)
```

| Algorithm | Prosise use case | Section |
|-----------|------------------|--------|
| Logistic regression | Titanic, interpretable odds | [3.2](./section-02-logistic-regression.md) |
| Decision tree / forest | Nonlinear tabular data | [3.5](./section-05-tree-based-classifiers.md) |
| k-NN | MNIST baseline | [3.8](./section-08-case-study-mnist-digit-recognition.md) |
| SVM | MNIST with kernels | [3.8](./section-08-case-study-mnist-digit-recognition.md) |

You already know [k-NN](../../GLOSSARY.md#k-nearest-neighbors) from [Chapter 01](../chapter-01-machine-learning/section-05-supervised-learning-k-nearest-neighbors.md). Here it classifies by **majority vote** among neighbors instead of averaging numbers.

---

## Classification vs Regression: Don't Mix the Tools

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# WRONG: LinearRegression on survived (0/1) - outputs 0.73 "survived units"
# RIGHT: LogisticRegression - outputs probability, then class

# WRONG: RandomForestRegressor on MNIST digits 0-9 as floats
# RIGHT: RandomForestClassifier
```

| Mistake | Symptom |
|---------|---------|
| Regression model on class labels | Predictions like 0.847 - not a valid class |
| Classification metrics on continuous targets | Nonsense precision/recall |
| Treating ordinal codes as regression | Model thinks class "3" is 3× class "1" |

> **Humorous analogy:** Using `LinearRegression` for spam detection is like hiring a accountant to sort mail into "inbox" and "trash" - they'll give you a spreadsheet with 0.62 instead of a decision.

---

## The Three Iconic Datasets (Chapter Arc)

Prosise structures Chapter 3 around three problems that span the classification spectrum:

| Dataset | Classes | Challenge | Section |
|---------|---------|-----------|--------|
| **Titanic** | 2 (survived/perished) | Missing data, categoricals | [3.6](./section-06-case-study-titanic-survival.md) |
| **Credit card fraud** | 2 (legit/fraud) | Extreme imbalance (492 / 284,807) | [3.7](./section-07-case-study-credit-card-fraud-detection.md) |
| **MNIST** | 10 (digits 0-9) | High-dimensional pixels | [3.8](./section-08-case-study-mnist-digit-recognition.md) |

Together they teach: encoding, metrics, imbalance, and multi-class - the full toolkit before [Chapter 04 - Text Classification](../chapter-04-text-classification/README.md).

---

## Train / Test / Metrics (Inherited from Chapter 02)

The workflow is identical to regression - only the metrics change:

```
Load data → EDA → encode features → split → fit → predict → evaluate
```

| Regression ([Chapter 02](../chapter-02-regression-models/section-06-regression-accuracy-measures.md)) | Classification (this chapter) |
|-------------------------------------------------------------------------------------|------------------------------|
| [RMSE](../../GLOSSARY.md#rmse), [MAE](../../GLOSSARY.md#mae), [R²](../../GLOSSARY.md#r-squared) | Accuracy, precision, recall, F1, ROC-AUC |
| Residual plots | [Confusion matrix](./section-03-classification-accuracy-measures.md) |
| Mean baseline | Majority-class baseline |

Use [stratified splits](../../GLOSSARY.md#cross-validation) when classes are imbalanced - `train_test_split(..., stratify=y)`.

---

## Parametric vs Nonparametric (Classification Edition)

From [Section 2.2](../chapter-02-regression-models/section-02-linear-regression.md):

- **Parametric** (logistic regression): Fixed equation shape, learned weights - fast, interpretable
- **Nonparametric** (k-NN, trees): Complexity grows with data - flexible, can [overfit](../../GLOSSARY.md#overfitting)

**Practical rule:** Logistic regression and SVM often need **scaled** numeric features. Tree ensembles do not. [Section 3.4](./section-04-categorical-data-and-feature-encoding.md) handles mixed types via pipelines.

---

## When to Use Classification

| ✅ Good fit | ❌ Poor fit |
|------------|------------|
| Target is a category (yes/no, species, digit) | Target is continuous → use regression |
| Enough labeled examples per class | Thousands of classes with 5 examples each |
| Misclassification costs are understood | You need a ranking score only (consider regression on probability) |

---

## Self-Check

1. Is "predict which star rating (1-5) a user will give" classification or regression? (Trick question - discuss ordinal vs regression.)
2. Why does the majority-class baseline beat 50% on Titanic?
3. What's the difference between multi-class and multi-label?
4. Name one parametric and one nonparametric classifier from this chapter.

---

## References

- [Scikit-Learn Classification Guide](https://scikit-learn.org/stable/supervised_learning.html#classification)
- [Kaggle: Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- Prosise, *Applied ML and AI for Engineers*, Ch. 3 - Introduction
- [StatQuest: Classification](https://www.youtube.com/watch?v=eyb_Rz_Okc4)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Chapter 02 - Regression Models](../chapter-02-regression-models/README.md) | **Next:** [Section 3.2 - Logistic Regression](./section-02-logistic-regression.md)



