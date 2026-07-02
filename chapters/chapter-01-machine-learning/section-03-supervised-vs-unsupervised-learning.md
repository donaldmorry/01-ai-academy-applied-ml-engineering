# Section 1.3: Supervised vs Unsupervised Learning

> **Source inheritance:** Prosise, Ch. 1 - "Supervised Versus Unsupervised Learning"  
> **Enhanced with:** Decision framework, semi-supervised preview, connection to Course 2 learning theory  
> **Glossary:** [../../GLOSSARY.md](../../GLOSSARY.md) | **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Fundamental Divide

All of [machine learning](../../GLOSSARY.md#machine-learning) splits along one axis: **Do you have [labels](../../GLOSSARY.md#label)?**

| | [Supervised](../../GLOSSARY.md#supervised-learning) | [Unsupervised](../../GLOSSARY.md#unsupervised-learning) |
|---|-----------|-------------|
| **Training data** | Features + correct answers | Features only |
| **Goal** | Predict labels for new data | Discover hidden structure |
| **Question answered** | "What is this?" / "How much?" | "What groups exist?" |
| **Evaluation** | Compare predictions to known labels | Harder - no ground truth |
| **This chapter** | [k-NN](../../GLOSSARY.md#k-nearest-neighbors) (Section 1.5) | [k-Means](../../GLOSSARY.md#k-means) (Section 1.4) |

---

## Supervised Learning in Depth

[Supervised learning](../../GLOSSARY.md#supervised-learning) is the workhorse of applied ML. You have inputs $\mathbf{x}_i$ and known outputs $y_i$ for $i = 1, \ldots, n$, and you learn a function that minimizes prediction error:

$$
\min_{f} \sum_{i=1}^{n} L\bigl(y_i, f(\mathbf{x}_i)\bigr)
$$
> **Readable form:** find the function f that minimizes the total loss between true labels and predictions across all training examples

The loss function $L$ depends on the task. [Classification](../../GLOSSARY.md#classification) often uses 0-1 loss (wrong = 1, correct = 0). [Regression](../../GLOSSARY.md#regression) uses squared error:

$$
L_{\text{reg}}(y, \hat{y}) = (y - \hat{y})^2
$$
> **Readable form:** regression loss = squared difference between true value and prediction

### Classification vs Regression

Supervised learning splits into two subtypes based on the output:

**Classification** - predict a **category**
- Binary: spam/ham, fraud/legitimate, sick/healthy
- Multiclass: iris species (setosa, versicolor, virginica), digit (0-9)
- Output: discrete class label

**Regression** - predict a **number**
- House price, taxi fare, temperature, stock price
- Output: continuous value

The same algorithm family can often do both (e.g., neural networks), but evaluation metrics differ:
- [Classification](../../GLOSSARY.md#classification): [accuracy](../../GLOSSARY.md#accuracy), precision, recall, F1
- [Regression](../../GLOSSARY.md#regression): [MAE](../../GLOSSARY.md#mae), [MSE](../../GLOSSARY.md#mse), [RMSE](../../GLOSSARY.md#rmse), [R²](../../GLOSSARY.md#r-squared)

> Chapter 02 covers regression in depth. Chapter 03 covers classification.

### The Supervised Learning Pipeline

```python
# 1. Load labeled data
X = features  # shape: (n_samples, n_features)
y = labels    # shape: (n_samples,)

# 2. Split data - CRITICAL
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 4. Evaluate on UNSEEN test data
accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.2%}")
```

**The golden rule:** Never evaluate on data the [model](../../GLOSSARY.md#model) trained on. That measures memorization, not generalization.

### Prosise's Supervised Example: Classifying Flowers

Prosise Ch. 1 walks through the Iris dataset - 150 flowers with four measurements and a species [label](../../GLOSSARY.md#label). This is textbook [supervised classification](../../GLOSSARY.md#classification):

| Step | Iris Example | General Pattern |
|------|--------------|-----------------|
| Collect | 50 samples × 3 species | Gather labeled examples |
| Split | Train on 80%, test on 20% | Hold out unseen data |
| Train | [k-NN](../../GLOSSARY.md#k-nearest-neighbors) stores training points | Fit [model](../../GLOSSARY.md#model) parameters (or store data) |
| Evaluate | Compare predictions to true species | Measure [accuracy](../../GLOSSARY.md#accuracy) on test set |

The flower example is deliberately small so you can **see** every step. Real projects scale the same pipeline to millions of rows.

---

## Unsupervised Learning in Depth

Without [labels](../../GLOSSARY.md#label), the algorithm must find structure on its own. [Clustering](../../GLOSSARY.md#clustering) algorithms like [k-means](../../GLOSSARY.md#k-means) minimize within-group scatter:

$$
\min_{\text{assignments}} \sum_{k=1}^{K} \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2
$$
> **Readable form:** minimize the sum of squared distances from each point to its cluster center

### Common Unsupervised Tasks

| Task | Algorithm | Use Case |
|------|-----------|----------|
| **Clustering** | [k-Means](../../GLOSSARY.md#k-means), DBSCAN | Customer segmentation, document grouping |
| **Dimensionality reduction** | PCA, t-SNE | Visualization, noise filtering, compression |
| **Anomaly detection** | Isolation Forest, PCA | Fraud detection, equipment failure |
| **Association rules** | Apriori | Market basket analysis |
| **Density estimation** | GMM | Understanding data distribution |

### Why Unsupervised Learning Matters

1. **Labels are expensive.** Labeling 1 million images might cost **50,000 USD** or more. [Clustering](../../GLOSSARY.md#clustering) costs compute only.
2. **Exploration.** Before building a classifier, cluster to understand what groups exist.
3. **Preprocessing.** PCA reduces dimensions before supervised learning (Chapter 06).
4. **Foundation for generative AI.** Course 4's VAEs and GANs are unsupervised generative models.

### The Challenge: No Ground Truth

How do you know if clustering is "good"? Options:
- **Domain knowledge:** Do the clusters make business sense?
- **Internal metrics:** Silhouette score, inertia (within-cluster variance)
- **Downstream task:** Do clusters improve a supervised model?
- **Visualization:** Plot in 2D (after PCA/t-SNE) - do groups separate?

### Prosise's Unsupervised Example: Customer Segments

Prosise pairs the Iris [supervised](../../GLOSSARY.md#supervised-learning) walkthrough with a retail [clustering](../../GLOSSARY.md#clustering) scenario: thousands of customers, numeric purchase behavior, **no predefined segments**. Marketing wants personas - "VIP," "dormant," "bargain hunter" - but nobody labeled the data.

| Business Question | Supervised? | Unsupervised? |
|-------------------|-------------|---------------|
| "Will this customer churn next month?" | Yes - historical churn [labels](../../GLOSSARY.md#label) | No |
| "What natural groups exist in our customer base?" | No - no predefined groups | Yes - [k-means](../../GLOSSARY.md#k-means) |
| "Which segment should get a discount email?" | Only if segments were pre-labeled | Cluster first, then strategize |

The workflow: cluster → profile each cluster's mean [features](../../GLOSSARY.md#feature) → name personas → design campaigns. [Clustering](../../GLOSSARY.md#clustering) is exploratory; the business value comes from **interpretation**.

---

## Semi-Supervised and Self-Supervised Learning (Preview)

Modern ML often blends paradigms:

### Semi-Supervised Learning
- Small amount of labeled data + large amount of unlabeled data
- Example: 1,000 labeled medical images + 100,000 unlabeled
- Covered in Course 3 (Goodfellow, Ch. 7.6)

### Self-Supervised Learning
- Create labels from the data itself
- Example: Mask random words → predict them (BERT pretraining)
- Example: Predict next frame in video
- Powers modern LLMs and is central to Course 4

### Transfer Learning
- Train on large dataset (source task), fine-tune on small dataset (target task)
- Example: ImageNet-pretrained ResNet → your custom image classifier
- Covered in Chapter 10

---

## Decision Framework: Which Paradigm?

```
START: Do you have a prediction target?
│
├─ YES: What type of target?
│   ├─ Category → Supervised Classification (Chapter 03)
│   └─ Number → Supervised Regression (Chapter 02)
│
└─ NO: What do you want to discover?
    ├─ Groups/clusters → Unsupervised Clustering (Section 1.4)
    ├─ Reduce dimensions → PCA (Chapter 06)
    ├─ Find anomalies → Anomaly Detection (Chapter 06)
    └─ Generate new data → Generative Models (Course 4)
```

---

## Real-World Examples

| Problem | Paradigm | Why |
|---------|----------|-----|
| Predict customer churn | Supervised classification | Historical churn labels exist |
| Segment customers for marketing | Unsupervised clustering | No predefined segments |
| Forecast quarterly revenue | Supervised regression | Historical revenue numbers |
| Detect network intrusions | Unsupervised anomaly | Attacks are rare/unknown |
| Recommend movies | Hybrid | Collaborative filtering + content features |
| Generate product descriptions | Generative (Course 4) | Create new text from patterns |

---

## Common Mistakes

| Mistake | Why It Fails | What To Do |
|---------|--------------|------------|
| **Using [supervised](../../GLOSSARY.md#supervised-learning) learning without labels** | No ground truth to learn from | Collect labels, or switch to [unsupervised](../../GLOSSARY.md#unsupervised-learning) |
| **Treating [clustering](../../GLOSSARY.md#clustering) clusters as ground truth** | Clusters are hypotheses, not facts | Validate with domain experts |
| **Evaluating on training data** | Inflated [accuracy](../../GLOSSARY.md#accuracy) | Always use held-out test set |
| **Mixing paradigms without a plan** | Cluster then classify the same test rows you clustered on | Keep splits clean; cluster on train only if labels follow |
| **Ignoring class imbalance** | 99% [accuracy](../../GLOSSARY.md#accuracy) on a 99% negative class | Use precision, recall, F1 (Chapter 03) |

### Supervised vs Unsupervised: Quick Decision Table

| You have... | You want... | Paradigm |
|-------------|-------------|----------|
| Labeled emails (spam/ham) | Filter new mail | [Supervised classification](../../GLOSSARY.md#classification) |
| House features + past sale prices | Price new listings | [Supervised regression](../../GLOSSARY.md#regression) |
| Customer transactions, no segments | Marketing personas | [Unsupervised clustering](../../GLOSSARY.md#clustering) |
| Sensor readings, rare failures | Flag anomalies | Unsupervised anomaly detection |
| 1K labels + 100K unlabeled images | Better classifier | Semi-supervised (Course 3) |

---

## Key Vocabulary

| Term | Definition |
|------|-----------|
| **Labeled data** | Examples with known correct outputs |
| **Unlabeled data** | Examples with [features](../../GLOSSARY.md#feature) only |
| **Training set** | Data used to fit the [model](../../GLOSSARY.md#model) |
| **Test set** | Held-out data for final evaluation |
| **Validation set** | Held-out data for [hyperparameter](../../GLOSSARY.md#hyperparameter) tuning |
| **Cluster** | A group of similar data points ([unsupervised](../../GLOSSARY.md#unsupervised-learning)) |
| **Class** | A category [label](../../GLOSSARY.md#label) ([supervised classification](../../GLOSSARY.md#classification)) |

---

## Reflection Questions

1. Your startup has 10,000 product reviews with star ratings. Supervised or unsupervised to find "complaint themes"?
2. Why is evaluating [clustering](../../GLOSSARY.md#clustering) harder than evaluating [classification](../../GLOSSARY.md#classification)?
3. When would semi-supervised learning beat pure [supervised](../../GLOSSARY.md#supervised-learning) learning?

---

## Further Reading

- [Section 1.4 - k-Means](./section-04-unsupervised-learning-k-means-clustering.md) | [Section 1.5 - k-NN](./section-05-supervised-learning-k-nearest-neighbors.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Next:** [Section 1.4 - k-Means Clustering](./section-04-unsupervised-learning-k-means-clustering.md)



