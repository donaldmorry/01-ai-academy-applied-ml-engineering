# Section 2.3: Decision Trees for Regression

> **Source:** Prosise, Ch. 2 - "Decision Trees"  
> **Prerequisites:** [Section 2.2](./section-02-linear-regression.md) | [Glossary: regression](../../GLOSSARY.md#regression)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Twenty Questions - But for Numbers

**Real-life analogy:** You're guessing someone's age. You ask:
1. "Are you over 30?" → Yes → go left
2. "Do you have gray hair?" → No → go left  
3. "Final guess: **35 years old**"

That's a **decision tree**. Each internal node is a yes/no question on a [feature](../../GLOSSARY.md#feature). Each leaf holds a **prediction** (average target of training samples that land there).

For [regression](../../GLOSSARY.md#regression), leaves predict the **mean** (or median) of $y$ values in that region - not a category.

> **In plain English:** A regression tree doesn't output "cat" or "dog." It outputs "people in this branch earn about 87,000 USD on average."

---

## **Milestone: Trees Capture Nonlinear Patterns**

[Linear regression](../../GLOSSARY.md#linear-regression) draws straight lines. Real life is messy:

- Taxi fares jump at airport zones
- Salaries plateau after 15 years
- Energy use spikes on weekends

Decision trees draw **axis-aligned rectangles** in feature space - piecewise constant predictions.

```
        distance < 5 mi?
       /              \
     YES               NO
  fare = 12 USD    passengers > 2?
                 /            \
            fare = 22 USD   fare = 35 USD
```

> **In plain English:** Instead of one formula for everything, the tree says "short trips cost about 12 dollars, long trips with crowds cost about 35 dollars."

---

## How Splits Are Chosen: Minimize Variance

At each node, the algorithm tries every feature and every threshold. It picks the split that **most reduces variance** of $y$ in the child nodes.

For a node with targets $\{y_1, \ldots, y_m\}$, variance is:

$$
\text{Var}(y) = \frac{1}{m}\sum_{j=1}^{m}(y_j - \bar{y})^2
$$
> **Readable form:** variance = average squared distance of targets from their mean in this node

After a candidate split into left ($L$) and right ($R$) children:

$$
\text{Cost} = \frac{|L|}{m}\text{Var}(y_L) + \frac{|R|}{m}\text{Var}(y_R)
$$
> **Readable form:** split cost = weighted average of left-child variance and right-child variance

The winning split minimizes this cost. **Greedy algorithm** - each split is locally optimal, not globally perfect.

> **Humorous analogy:** The tree is a micromanager who fixes the biggest mess on their desk first, not the mess that matters for the whole company. It works surprisingly well anyway.

---

## Training: Recursive Partitioning

1. Start with all training samples at the root
2. Try every [feature](../../GLOSSARY.md#feature), every split point
3. Pick split that most reduces variance of $y$ in child nodes
4. Repeat on each child until stopping rule (max depth, min samples)
5. At each leaf, store $\hat{y} = \text{mean}(y_{\text{leaf}})$

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

# Synthetic: fare depends on distance with a step at 10 miles
np.random.seed(0)
distance = np.random.uniform(0, 30, 500)
fare = np.where(distance < 10,
                8 + 1.2 * distance,
                20 + 0.8 * distance) + np.random.normal(0, 2, 500)

X = distance.reshape(-1, 1)
y = fare

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeRegressor(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

print(f"Train MAE: {mean_absolute_error(y_train, tree.predict(X_train)):.2f} USD")
print(f"Test MAE:  {mean_absolute_error(y_test, tree.predict(X_test)):.2f} USD")
```

---

## The Overfitting Trap

A tree with **unlimited depth** memorizes every training point → [overfitting](../../GLOSSARY.md#overfitting).

| `max_depth` | Train error | Test error | Diagnosis |
|-------------|-------------|------------|-----------|
| 2 | High | High | [Underfitting](../../GLOSSARY.md#underfitting) |
| 4 | Moderate | Moderate | ✅ Sweet spot |
| 20 | ~0 | High | Overfitting |

```python
for depth in [2, 4, 8, 20, None]:
    t = DecisionTreeRegressor(max_depth=depth, random_state=42)
    t.fit(X_train, y_train)
    print(f"depth={depth}: train={mean_absolute_error(y_train, t.predict(X_train)):.1f}, "
          f"test={mean_absolute_error(y_test, t.predict(X_test)):.1f}")
```

> **Humorous analogy:** A tree with `max_depth=None` is that friend who memorizes the textbook but panics on any new question wording. Depth limits force **generalization**.

### Key Hyperparameters

| Parameter | Controls | Typical starting point |
|-----------|----------|------------------------|
| `max_depth` | Maximum questions deep | 4-12 for tabular |
| `min_samples_split` | Minimum samples to split a node | 10-50 |
| `min_samples_leaf` | Minimum samples per leaf | 5-20 |
| `max_features` | Features considered per split | `'sqrt'` or `None` |

Prosise uses shallow-to-medium trees as building blocks for [random forests](../../GLOSSARY.md#random-forest) in [Section 2.4](./section-04-ensemble-methods-random-forests-and-gradient-boosting.md) - single trees are rarely the final production model.

---

## Visualize the Tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8))
plot_tree(tree, feature_names=['distance_mi'], filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree for Taxi Fare (depth=4)')
plt.show()
```

**Prosise's insight:** Trees are **interpretable** - show the diagram to a product manager and they understand the logic. Try that with a 200-tree random forest.

### Export rules as text

```python
from sklearn.tree import export_text
print(export_text(tree, feature_names=['distance_mi']))
```

---

## Extrapolation: Trees Don't Dream Beyond Data

[Linear regression](../../GLOSSARY.md#linear-regression) extends a line forever. A tree **clamps** predictions to the nearest leaf mean.

| Trip distance (mi) | Linear might predict | Tree predicts |
|--------------------|---------------------|---------------|
| 5 | ~14 USD | ~12 USD (short-trip leaf) |
| 50 | ~60 USD | ~35 USD (same leaf as 30 mi trip!) |

> **In plain English:** Trees have no opinion about trips longer than anything they saw in training. That's safer than a runaway line - but dangerous if you need forecasts outside historical range.

---

## Trees vs Linear Regression

| | Linear Regression | Decision Tree |
|---|------------------|---------------|
| Shape | Straight line / plane | Steps and rectangles |
| Outliers | Sensitive (unless Ridge) | Somewhat robust |
| Extrapolation | Extends line beyond data | Clamps to nearest leaf mean |
| Interpretability | Coefficients | Visual rules |
| Single tree accuracy | Baseline | Often better on nonlinear data |
| Feature scaling | Often needed | Not needed |
| Interactions | Manual (cross-terms) | Automatic |

> **In plain English:** Linear regression is a ruler. Decision trees are a flowchart drawn with a highlighter.

---

## Feature Importance (Single Tree)

```python
import pandas as pd

importance = pd.Series(tree.feature_importances_, index=['distance_mi'])
print(importance.sort_values(ascending=False))
```

Importance sums to 1.0 across features - higher means more splits used that feature. Less reliable on single trees than on forests ([Section 2.4](./section-04-ensemble-methods-random-forests-and-gradient-boosting.md)).

---

## When Prosise Reaches for Trees

Prosise uses decision trees when:

- Scatter plots show **curves, steps, or plateaus**
- You need a **human-readable** model for stakeholders
- You're building toward **ensembles** (the real accuracy play)

He does **not** deploy a lone deep tree to production - variance is too high.

---

## Common Mistakes

1. **`max_depth=None` on noisy data** - perfect train, awful test
2. **Expecting smooth predictions** - trees are stairsteps, not curves
3. **Trusting extrapolation** - 100-mile trip gets same prediction as 30-mile leaf
4. **Forgetting ensembles** - one tree is a stepping stone, not the finish line
5. **One-hot encoding tree-unfriendly categoricals** - works, but high cardinality hurts

---

## Self-Check

1. What value does a regression tree leaf predict?  
   *Mean (default) or median of training targets in that leaf.*
2. Why does unlimited depth hurt test performance?  
   *[Overfitting](../../GLOSSARY.md#overfitting) - memorizes noise.*
3. When would a single tree beat linear regression on taxi fares?  
   *When fare vs distance has jumps (tolls, zones) or interactions trees capture automatically.*
4. Must you scale features for `DecisionTreeRegressor`?  
   *No - splits are order-based, not distance-based.*

---

## References

- [Scikit-Learn DecisionTreeRegressor](https://scikit-learn.org/stable/chapters/generated/sklearn.tree.DecisionTreeRegressor.html)
- [Scikit-Learn: Decision Trees](https://scikit-learn.org/stable/chapters/tree.html)
- Prosise, *Applied ML and AI for Engineers*, Ch. 2 - Decision Trees
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFG8uY)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 2.2](./section-02-linear-regression.md) | **Next:** [Section 2.4 - Ensemble Methods](./section-04-ensemble-methods-random-forests-and-gradient-boosting.md)
