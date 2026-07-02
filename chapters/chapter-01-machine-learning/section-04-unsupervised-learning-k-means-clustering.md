# Section 1.4: Unsupervised Learning - k-Means Clustering

> **Source inheritance:** Prosise, Ch. 1 - "Unsupervised Learning with k-Means Clustering"  
> **Enhanced with:** Algorithm internals, business applications, mathematical intuition  
> **Glossary:** [../../GLOSSARY.md](../../GLOSSARY.md) | **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Business Problem

A retail company has 50,000 customers and wants to tailor marketing campaigns. They have data on:
- Annual spending
- Purchase frequency
- Average order value
- Days since last purchase
- Product category preferences

**But they have no predefined customer categories.** Marketing doesn't know if there are 3 segments or 10. This is a perfect [unsupervised learning](../../GLOSSARY.md#unsupervised-learning) problem.

[k-Means clustering](../../GLOSSARY.md#k-means) will discover natural groupings in the data.

---

## How k-Means Works

### The Algorithm (Lloyd's Algorithm)

```
1. Choose k (number of clusters)
2. Initialize k centroids randomly
3. REPEAT until convergence:
   a. ASSIGN: Each point → nearest centroid
   b. UPDATE: Each centroid → mean of assigned points
4. RETURN cluster assignments and centroids
```

### Visual Intuition

```
Iteration 0 (random centroids):     Iteration 3 (converged):
    ×  ·  ·  ·  ○  ○                  ·  ·  ·|·  ·  ·
    ·  ·  ×  ·  ○  ○                  ·  ·  ·|·  ·  ·
    ·  ·  ·  ·  ○  ×                  ·  ·  ·|·  ·  ·
    ·  ·  ·  ·  ○  ○                  ───────────────
    × = centroid                      Cluster 1 | Cluster 2
```

k-Means minimizes **inertia** - the sum of squared distances from each point to its cluster centroid:

$$
\text{Inertia} = \sum_{i=1}^{n} \min_{\mu_k} \|x_i - \mu_k\|^2
$$
> **Readable form:** inertia = sum over all points of (squared distance to nearest cluster center)

Each cluster center $\mu_k$ is updated to the mean of its assigned points:

$$
\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i
$$
> **Readable form:** cluster center = average of all points assigned to that cluster

---

## Implementation: Customer Segmentation

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load customer data
customers = pd.read_csv('customer_data.csv')
features = ['annual_spending', 'purchase_frequency', 
            'avg_order_value', 'days_since_last_purchase']

X = customers[features]

# CRITICAL: Scale features before k-means
# k-means uses Euclidean distance - unscaled features dominate
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose k clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
customers['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
print(customers.groupby('cluster')[features].mean())
```

### Why Scale Features?

Consider two features: `annual_spending` (range: $0-$100,000) and `purchase_frequency` (range: 1-52).

Without scaling, spending dominates distance calculations. A **1 USD** difference in spending outweighs a 10-purchase difference in frequency. `StandardScaler` transforms each [feature](../../GLOSSARY.md#feature) to mean $= 0$, std $= 1$.

> **Rule:** Always scale before k-means, k-NN, SVMs, and neural networks. Tree-based models (decision trees, random forests) do NOT require scaling.

---

## Choosing k: The Elbow Method

How many clusters? Try multiple values and plot inertia:

```python
inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

Look for the **elbow** - where adding more clusters yields diminishing returns. If inertia drops sharply from k=2 to k=3 but barely from k=3 to k=4, choose k=3.

### Silhouette Score (Alternative)

```python
from sklearn.metrics import silhouette_score

for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k={k}: silhouette={score:.3f}")
```

Higher silhouette (closer to 1.0) = better-defined clusters. The silhouette for point $i$ compares within-cluster distance $a(i)$ to nearest other-cluster distance $b(i)$:

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$
> **Readable form:** silhouette = (distance to other cluster minus distance to own cluster) divided by the larger of those two distances

Average $s(i)$ across all points gives the score printed by sklearn.

---

## Interpreting Results: Customer Personas

After clustering with k=4:

| Cluster | Spending | Frequency | Avg Order | Recency | Persona |
|---------|----------|-----------|-----------|---------|---------|
| 0 | Low | Low | Low | High | **Dormant** - win-back campaign |
| 1 | High | High | High | Low | **VIP** - loyalty rewards |
| 2 | Medium | Medium | Medium | Medium | **Regular** - upsell opportunities |
| 3 | Low | High | Low | Low | **Bargain Hunters** - promotions |

This transforms raw numbers into **actionable business strategy**.

---

## Segmenting in Higher Dimensions

Prosise demonstrates [clustering](../../GLOSSARY.md#clustering) with more than two [features](../../GLOSSARY.md#feature) - the algorithm works identically in any number of dimensions. Visualization requires dimensionality reduction (PCA, Chapter 06) to plot in 2D.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=customers['cluster'], cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Customer Segments (PCA projection)')
plt.colorbar(label='Cluster')
plt.show()
```

---

## k-Means Limitations

| Limitation | Explanation | Alternative |
|-----------|-------------|-------------|
| Must specify k | Algorithm doesn't choose cluster count | Elbow method, DBSCAN, GMM |
| Assumes spherical clusters | Struggles with elongated or irregular shapes | DBSCAN, spectral clustering |
| Sensitive to initialization | Bad random starts → suboptimal clusters | `n_init=10` (run 10 times, keep best) |
| Sensitive to outliers | Outliers pull centroids | Remove outliers, use k-Medoids |
| Only numerical features | Cannot handle categorical directly | Encode categories, use k-Modes |

---

## k-Means vs Other Clustering (Preview)

| Algorithm | Strengths | Weaknesses |
|-----------|-----------|-----------|
| **k-Means** | Fast, simple, scalable | Needs k, spherical clusters |
| **DBSCAN** | Finds arbitrary shapes, auto k | Sensitive to density parameters |
| **Hierarchical** | Dendrogram visualization | Slow on large datasets |
| **GMM** | Soft assignments (probabilities) | Assumes Gaussian clusters |

---

## Connection to the Ecosystem

- **Course 2 (AIMA, Ch. 19):** k-Means as an unsupervised learning algorithm within the broader learning framework; connection to EM algorithm (Ch. 20)
- **Course 3 (Goodfellow, Ch. 5.8):** Clustering as unsupervised learning; k-Means as a simple baseline
- **Course 4 (Foster, Ch. 1):** Generative models learn data distributions - [clustering](../../GLOSSARY.md#clustering) is a primitive form of distribution discovery

---

## Common Mistakes with k-Means

| Mistake | Symptom | Fix |
|---------|---------|-----|
| **Skipping feature scaling** | One [feature](../../GLOSSARY.md#feature) dominates distance | `StandardScaler` before [k-means](../../GLOSSARY.md#k-means) |
| **Using `n_init=1`** | Different runs → different clusters | Set `n_init=10` (sklearn default in recent versions) |
| **Choosing k by gut alone** | Too many/few segments | Elbow + silhouette + business sense |
| **Treating clusters as permanent truth** | Campaigns misfire when data shifts | Re-cluster periodically |
| **Clustering on test data with train** | Information leaks across splits | Fit scaler and [k-means](../../GLOSSARY.md#k-means) on train only in pipelines |
| **Expecting non-spherical groups** | Elongated clusters split wrongly | Try DBSCAN or GMM |

### Prosise Ch. 1: From Clusters to Campaigns

Prosise's retail narrative is not just algorithmic - it is **operational**. After [k-means](../../GLOSSARY.md#k-means) assigns cluster IDs, analysts compute per-cluster means and translate numbers into personas:

```
Cluster 0: low spend, high recency  → "Win-back" email with 15% coupon
Cluster 1: high spend, low recency  → "VIP early access" to new products
```

Without [unsupervised learning](../../GLOSSARY.md#unsupervised-learning), marketing would either guess segments or run expensive surveys. [Clustering](../../GLOSSARY.md#clustering) discovers structure the business did not know to look for.

---

## Exercises

1. Run k-Means on the Iris dataset (without species labels). Does k=3 recover the three species?
2. What happens when you don't scale features? Quantify the difference.
3. Try k=2 vs k=10 on customer data. Which is more useful for marketing?

---

## Further Reading

- [GLOSSARY.md](../../GLOSSARY.md) - [k-means](../../GLOSSARY.md#k-means), [clustering](../../GLOSSARY.md#clustering), [unsupervised learning](../../GLOSSARY.md#unsupervised-learning)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Next:** [Section 1.5 - k-Nearest Neighbors](./section-05-supervised-learning-k-nearest-neighbors.md)



