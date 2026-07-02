# Section 1.5: Supervised Learning - k-Nearest Neighbors

> **Source inheritance:** Prosise, Ch. 1 - "k-Nearest Neighbors" / "Using k-Nearest Neighbors to Classify Flowers"  
> **Enhanced with:** Geometric intuition, bias-variance via k, full Iris walkthrough  
> **Glossary:** [../../GLOSSARY.md](../../GLOSSARY.md) | **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Algorithm

[k-Nearest Neighbors (k-NN)](../../GLOSSARY.md#k-nearest-neighbors) is the simplest [supervised learning](../../GLOSSARY.md#supervised-learning) algorithm:

> To classify a new point, find the $k$ closest training examples and let them vote.

No training phase. The entire dataset IS the [model](../../GLOSSARY.md#model). This makes k-NN a **lazy learner** - all computation happens at prediction time.

### Steps

```
1. Store all training data
2. For a new point x:
   a. Compute distance to every training point
   b. Select the k nearest neighbors
   c. Classification: majority vote of neighbors' labels
      Regression: average of neighbors' values
```

### The Math: Distance and Voting

For a query point $\mathbf{x}$, compute distance to every training point. With Euclidean distance:

$$
d(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j=1}^{p} (x_j - x_{i,j})^2}
$$
> **Readable form:** distance = square root of (sum of squared differences across all features)

Select the $k$ training points with smallest $d$. For [classification](../../GLOSSARY.md#classification), predict the majority [label](../../GLOSSARY.md#label):

$$
\hat{y} = \arg\max_{c} \sum_{i \in N_k(\mathbf{x})} \mathbb{1}(y_i = c)
$$
> **Readable form:** predicted class = the label that appears most often among the k nearest neighbors

For [regression](../../GLOSSARY.md#regression), predict the mean of neighbor targets: $\hat{y} = \frac{1}{k}\sum_{i \in N_k} y_i$.

---

## The Iris Flower Dataset

The classic ML benchmark - 150 flowers, 4 [features](../../GLOSSARY.md#feature), 3 species - Prosise's flagship [supervised classification](../../GLOSSARY.md#classification) example from Ch. 1:

| Feature | Description | Range |
|---------|-------------|-------|
| Sepal length | cm | 4.3 - 7.9 |
| Sepal width | cm | 2.0 - 4.4 |
| Petal length | cm | 1.0 - 6.9 |
| Petal width | cm | 0.1 - 2.5 |

| Species | Count | Separability |
|---------|-------|-------------|
| Setosa | 50 | Easily separable |
| Versicolor | 50 | Overlaps with virginica |
| Virginica | 50 | Overlaps with versicolor |

---

## Full Implementation

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Evaluate
print(f"Accuracy: {knn.score(X_test, y_test):.2%}")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('k-NN Confusion Matrix (k=5)')
plt.show()
```

### Reading the Confusion Matrix

```
              Predicted
              setosa  versicolor  virginica
Actual setosa    10       0          0      ← Perfect
       versicolor  0       8          2      ← 2 misclassified
       virginica   0       1          9      ← 1 misclassified
```

Versicolor and virginica overlap in feature space - even humans struggle with some samples.

---

## The Effect of k

$k$ controls the bias-variance tradeoff ([overfitting](../../GLOSSARY.md#overfitting) vs [underfitting](../../GLOSSARY.md#underfitting)):

| k value | Behavior | Bias | Variance |
|---------|----------|------|----------|
| **k=1** | Nearest single neighbor decides | Low | Very high (overfitting) |
| **k=5** | 5 neighbors vote | Moderate | Moderate |
| **k=50** | Half the dataset votes | High | Low (underfitting) |
| **k=n** | All points vote → always predicts majority class | Maximum | Minimum |

```python
k_values = range(1, 31)
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

plt.plot(k_values, train_scores, label='Training')
plt.plot(k_values, test_scores, label='Test')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()
plt.title('k-NN: Effect of k on Accuracy')
plt.show()
```

**Pattern:** Training [accuracy](../../GLOSSARY.md#accuracy) decreases as $k$ increases. Test accuracy rises then falls. The optimal $k$ balances both - typically found via [cross-validation](../../GLOSSARY.md#cross-validation) (Chapter 02).

---

## Distance Metrics

k-NN depends entirely on how "near" is defined:

| Metric | Formula | Best For |
|--------|---------|----------|
| **Euclidean** | $\sqrt{\sum(x_i - y_i)^2}$ | Default, continuous features |
| **Manhattan** | $\sum\|x_i - y_i\|$ | High-dimensional, grid-like data |
| **Minkowski** | Generalization of above | Configurable |
| **Cosine** | Angle between vectors | Text, high-dimensional sparse data |

```python
# Different metrics
KNeighborsClassifier(n_neighbors=5, metric='euclidean')
KNeighborsClassifier(n_neighbors=5, metric='manhattan')
KNeighborsClassifier(n_neighbors=5, metric='cosine')
```

---

## Feature Scaling - Again

[k-NN](../../GLOSSARY.md#k-nearest-neighbors) is distance-based. Unscaled [features](../../GLOSSARY.md#feature) distort distances:

```python
# WITHOUT scaling: petal length (1-7) dominates sepal width (2-4)
# WITH scaling: all features contribute equally
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)
pipeline.fit(X_train, y_train)
print(f"Scaled accuracy: {pipeline.score(X_test, y_test):.2%}")
```

> **Best practice:** Always use `make_pipeline` to chain preprocessing + model. This prevents data leakage from scaling test data with training statistics.

---

## k-NN Strengths and Weaknesses

### Strengths
- Dead simple to understand and implement
- No training time (lazy learning)
- Naturally handles multiclass
- Non-parametric - no assumptions about data distribution
- Can learn complex decision boundaries with low k

### Weaknesses
- Slow prediction - must compute distance to ALL training points
- Curse of dimensionality - distances become meaningless in high dimensions
- Sensitive to irrelevant features and scaling
- Requires storing entire training set in memory
- Poor with imbalanced classes (majority class dominates votes)

---

## When to Use k-NN

| Use k-NN when... | Use something else when... |
|-----------------|--------------------------|
| Dataset is small (<10K samples) | Dataset is large (use trees, SVMs) |
| Decision boundary is complex | You need fast predictions |
| You want a quick baseline | Features are high-dimensional |
| Interpretability via neighbors matters | Classes are imbalanced |

---

## Beyond Classification: k-NN for Regression

```python
from sklearn.neighbors import KNeighborsRegressor

# Predict continuous values (e.g., taxi fares - Chapter 02)
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)
predictions = knn_reg.predict(X_test)  # Average of 5 nearest neighbors' values
```

---

## Common Mistakes with k-NN

| Mistake | Symptom | Fix |
|---------|---------|-----|
| **Forgetting to scale** | One [feature](../../GLOSSARY.md#feature) dominates distance | `StandardScaler` in a pipeline |
| **Using $k=1$ on noisy data** | [Overfitting](../../GLOSSARY.md#overfitting) - jagged boundaries | Increase $k$ or use [cross-validation](../../GLOSSARY.md#cross-validation) |
| **Scaling before train/test split** | Data leakage - test stats influence train | Split first; fit scaler on train only |
| **Ignoring [hyperparameter](../../GLOSSARY.md#hyperparameter) $k$** | Suboptimal [accuracy](../../GLOSSARY.md#accuracy) | Sweep $k$ on validation set |
| **Using k-NN on 1M+ rows** | Prediction latency unacceptable | Switch to trees, linear models, or approximate NN |
| **Evaluating on training set** | 100% train [accuracy](../../GLOSSARY.md#accuracy) with $k=1$ | Always report test metrics |

### Prosise Ch. 1: Why k-NN Comes First

Prosise places [k-NN](../../GLOSSARY.md#k-nearest-neighbors) immediately after [k-means](../../GLOSSARY.md#k-means) because both algorithms share the same geometric intuition - **distance in feature space** - but differ in supervision:

| | [k-Means](../../GLOSSARY.md#k-means) | [k-NN](../../GLOSSARY.md#k-nearest-neighbors) |
|---|-----------|-----|
| Labels needed? | No | Yes |
| What is "k"? | Number of clusters | Number of neighbors |
| Output | Cluster ID | Class or numeric prediction |
| [Model](../../GLOSSARY.md#model) stored | Centroids | Entire training set |

His Iris walkthrough emphasizes that versicolor and virginica overlap - so even a simple algorithm misclassifies a few flowers. That honesty sets realistic expectations before Chapter 03's more powerful classifiers.

---

## Exercises

1. Implement k-NN from scratch (no sklearn) for 2D data. Plot decision boundaries for k=1, 5, 15.
2. What accuracy do you get on Iris with only petal length and petal width? With only sepal features?
3. Use [cross-validation](../../GLOSSARY.md#cross-validation) to find the optimal $k$ for Iris. Does it match your intuition?

---

## Further Reading

- [GLOSSARY.md](../../GLOSSARY.md) - [k-nearest neighbors](../../GLOSSARY.md#k-nearest-neighbors), [supervised learning](../../GLOSSARY.md#supervised-learning)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Next:** [Section 1.6 - The ML Workflow & Data Hygiene](./section-06-the-ml-workflow-and-data-hygiene.md)
