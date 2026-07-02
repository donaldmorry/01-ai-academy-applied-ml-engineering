# Lab 01: Customer Segmentation & Flower Classification

> **Prerequisites:** Sections 1.1-1.6  
> **Estimated time:** 2-3 hours  
> **Tools:** Python 3.10+, Jupyter, scikit-learn, pandas, matplotlib  
> **Glossary:** [../../GLOSSARY.md](../../GLOSSARY.md) | **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Lab Objectives

1. Perform end-to-end [k-means](../../GLOSSARY.md#k-means) [clustering](../../GLOSSARY.md#clustering) on synthetic customer data
2. Perform end-to-end [k-NN](../../GLOSSARY.md#k-nearest-neighbors) [classification](../../GLOSSARY.md#classification) on the Iris dataset
3. Practice the full ML workflow: load → explore → prepare → train → evaluate
4. Compare the effect of scaling, $k$ values, and train/test splits

### Math You'll Use

**Inertia** (what [k-means](../../GLOSSARY.md#k-means) minimizes):

$$
\text{Inertia} = \sum_{i=1}^{n} \min_{\mu_k} \|\mathbf{x}_i - \mu_k\|^2
$$
> **Readable form:** inertia = sum of squared distances from each point to its nearest cluster center

**Silhouette** (cluster quality - higher is better, range $[-1, 1]$):

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$
> **Readable form:** silhouette for point i = (distance to other cluster minus distance to own cluster) divided by the larger distance

**k-NN [accuracy](../../GLOSSARY.md#accuracy)** on test set:

$$
\text{Accuracy} = \frac{\text{correct predictions}}{\text{total test samples}}
$$
> **Readable form:** accuracy = number correct divided by total predictions

---

## Part A: Customer Segmentation with k-Means

Prosise Ch. 1 uses a retail scenario - you replicate it with synthetic blobs that mimic customer [features](../../GLOSSARY.md#feature).

### A.1 Generate Synthetic Customer Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

np.random.seed(42)

# Generate 500 customers in 4 natural groups
X, y_true = make_blobs(n_samples=500, centers=4, n_features=4,
                        cluster_std=1.5, random_state=42)

customers = pd.DataFrame(X, columns=[
    'annual_spending', 'purchase_frequency',
    'avg_order_value', 'days_since_last_purchase'
])

print(customers.describe())
```

### A.2 Explore the Data

```python
# Pairplot-style exploration
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
pairs = [('annual_spending', 'purchase_frequency'),
         ('avg_order_value', 'days_since_last_purchase'),
         ('annual_spending', 'avg_order_value')]

for ax, (x, y) in zip(axes, pairs):
    ax.scatter(customers[x], customers[y], alpha=0.5, s=10)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
plt.tight_layout()
plt.savefig('customer_exploration.png', dpi=150)
plt.show()
```

**Question:** Do you see natural groupings? How many?

### A.3 Find Optimal k

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customers)

inertias = []
silhouettes = []

for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(2, 11), inertias, 'bo-')
ax1.set_xlabel('k'); ax1.set_ylabel('Inertia'); ax1.set_title('Elbow Method')
ax2.plot(range(2, 11), silhouettes, 'ro-')
ax2.set_xlabel('k'); ax2.set_ylabel('Silhouette'); ax2.set_title('Silhouette Score')
plt.show()
```

**Task:** Choose $k$ based on both plots. Justify your choice in writing. Note: the elbow is often subjective - combine with silhouette and business interpretability.

### A.4 Cluster and Profile

```python
OPTIMAL_K = 4  # Adjust based on your analysis

kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
customers['cluster'] = kmeans.fit_predict(X_scaled)

# Profile each cluster
profile = customers.groupby('cluster').mean()
print(profile)

# Name your personas
personas = {
    0: 'YOUR_NAME_HERE',
    1: 'YOUR_NAME_HERE',
    2: 'YOUR_NAME_HERE',
    3: 'YOUR_NAME_HERE',
}
customers['persona'] = customers['cluster'].map(personas)
```

**Deliverable:** A table mapping each cluster to a business persona with recommended marketing action.

---

## Part B: Iris Classification with k-NN

Prosise's flagship [supervised](../../GLOSSARY.md#supervised-learning) example - classify flower species from measurements.

### B.1 Load and Split

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
import seaborn as sns

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)
```

### B.2 Find Optimal k

```python
k_range = range(1, 31)
test_scores = []

for k in k_range:
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    pipe.fit(X_train, y_train)
    test_scores.append(pipe.score(X_test, y_test))

best_k = k_range[np.argmax(test_scores)]
print(f"Best k: {best_k} (accuracy: {max(test_scores):.2%})")

plt.plot(k_range, test_scores)
plt.xlabel('k'); plt.ylabel('Test Accuracy')
plt.axvline(best_k, color='r', linestyle='--', label=f'Best k={best_k}')
plt.legend()
plt.show()
```

### B.3 Final Model & Error Analysis

```python
final_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=best_k)
)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title(f'k-NN Confusion Matrix (k={best_k})')
plt.show()
```

**Task:** Which species pair is most confused? Why? (Hint: plot petal length vs petal width colored by species.)

```python
# Optional: visualize the overlap Prosise discusses
plt.scatter(X_train[:, 2], X_train[:, 3], c=y_train, cmap='viridis', alpha=0.6)
plt.xlabel('Petal length'); plt.ylabel('Petal width')
plt.title('Why versicolor and virginica confuse k-NN')
plt.show()
```

### B.4 Scaling Experiment

```python
# Compare scaled vs unscaled
for scaled in [True, False]:
    if scaled:
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k))
    else:
        model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"{'Scaled' if scaled else 'Unscaled'}: {acc:.2%}")
```

**Expected insight:** On Iris, scaling often helps modestly because petal length has a wider range than sepal width. The section: always test, never assume.

---

## Part C: Reflection (Written)

Answer in a markdown cell or separate document:

1. What $k$ did you choose for customer segmentation? What business actions would you recommend for each cluster?
2. How did scaling affect [k-NN](../../GLOSSARY.md#k-nearest-neighbors) [accuracy](../../GLOSSARY.md#accuracy) on Iris? Why?
3. What's the difference between the "[model](../../GLOSSARY.md#model)" in [k-means](../../GLOSSARY.md#k-means) vs [k-NN](../../GLOSSARY.md#k-nearest-neighbors)?
4. If you had 1 million customers, would [k-NN](../../GLOSSARY.md#k-nearest-neighbors) still be practical for [classification](../../GLOSSARY.md#classification)? Why or why not?

---

## Common Lab Mistakes

| Mistake | What Goes Wrong | Fix |
|---------|-----------------|-----|
| **Forgetting `n_init=10`** | Unstable [k-means](../../GLOSSARY.md#k-means) clusters | Pass `n_init=10` to `KMeans` |
| **Scaling before split (Part B)** | Optimistic [accuracy](../../GLOSSARY.md#accuracy) | Use `make_pipeline(StandardScaler(), KNN)` |
| **Choosing $k$ from test set** | [Overfitting](../../GLOSSARY.md#overfitting) to test data | Pick $k$ from elbow/silhouette or validation only |
| **Not stratifying Iris split** | Test set may miss a class | Use `stratify=iris.target` |
| **Skipping written reflections** | Miss the learning objective | Answer all Part C questions |

---

## Troubleshooting

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| `ImportError: sklearn` | Environment not set up | `pip install scikit-learn pandas matplotlib seaborn` |
| Elbow plot is smooth, no clear elbow | Synthetic data may have gradual drop | Trust silhouette; try $k=4$ (true generative clusters) |
| 100% test [accuracy](../../GLOSSARY.md#accuracy) | Data leakage or $k=1$ on tiny test set | Check pipeline; verify split sizes |
| All customers in one cluster | Forgot to scale | Apply `StandardScaler` before [k-means](../../GLOSSARY.md#k-means) |

---

## Submission Checklist

- [ ] Customer segmentation with justified k choice
- [ ] Persona profiles with marketing recommendations
- [ ] Iris k-NN with optimal k via experimentation
- [ ] Confusion matrix and error analysis
- [ ] Scaling comparison results
- [ ] Written reflections answered

---

## Further Reading

- [Section 1.4 - k-Means](./section-04-unsupervised-learning-k-means-clustering.md) | [Section 1.5 - k-NN](./section-05-supervised-learning-k-nearest-neighbors.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Next Chapter:** [02 - Regression Models](../chapter-02-regression-models/README.md)



