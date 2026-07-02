# Section 6.8: Case Study - Bearing Failure & Credit Card Fraud Detection

> **Source:** Prosise, Ch. 6 - "Anomaly Detection"; NASA bearings; credit card fraud PCA  
> **Glossary:** [pca](../../GLOSSARY.md#pca) | [explained-variance](../../GLOSSARY.md#explained-variance)  
> **Math:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Anomaly Detection via Reconstruction Error

**Core hypothesis (Prosise):** Normal samples live near a low-dimensional subspace learned from healthy data. **[PCA](../../GLOSSARY.md#pca)** compresses and reconstructs them with small error. **Anomalies** - fraud, failing bearings, sensor faults - violate that subspace and reconstruct poorly.

$$
\text{score}(\mathbf{x}) = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 = \sum_{j=1}^{m}(x_j - \hat{x}_j)^2
$$
> **Readable form:** anomaly score = sum of squared differences between original sample and PCA reconstruction.

Train PCA on **normal only**. Score everything. Flag high scores.

> **Analogy:** PCA learns the "factory hum." Anything that doesn't hum the same tune after compression is shouting for attention.

---

## Part A: Credit Card Fraud (Prosise)

Unsupervised alternative to Chapter 3's supervised random forest. Dataset: 284k transactions, 29 anonymized features (`V1`-`V28`, `Amount`).

### Load and split by class

```python
import pandas as pd
import numpy as np

df = pd.read_csv('Data/creditcard.csv')
legit = df[df['Class'] == 0].drop(['Time', 'Class'], axis=1)
fraud = df[df['Class'] == 1].drop(['Time', 'Class'], axis=1)
print(f"Legit: {len(legit)}, Fraud: {len(fraud)}")
```

### Fit PCA on legitimate transactions only

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=26, random_state=0)  # 29 → 26
legit_pca = pca.fit_transform(legit)
fraud_pca = pca.transform(fraud)

legit_restored = pca.inverse_transform(legit_pca)
fraud_restored = pca.inverse_transform(fraud_pca)
```

### Anomaly scores

```python
def get_anomaly_scores(df_original, df_restored):
    loss = np.sum((np.array(df_original) - np.array(df_restored)) ** 2, axis=1)
    return pd.Series(loss, index=df_original.index)

legit_scores = get_anomaly_scores(legit, legit_restored)
fraud_scores = get_anomaly_scores(fraud, fraud_restored)
```

### Visual separation

```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

legit_scores.plot(figsize=(12, 4), title='Legitimate - reconstruction loss')
fraud_scores.plot(figsize=(12, 4), title='Fraudulent - reconstruction loss')
```

Prosise: most legit scores **< 200**; many fraud scores **> 200**.

### Threshold classifier

```python
threshold = 200

true_neg = (legit_scores < threshold).sum()
false_pos = (legit_scores >= threshold).sum()
true_pos = (fraud_scores >= threshold).sum()
false_neg = (fraud_scores < threshold).sum()

mat = [[true_neg, false_pos], [false_neg, true_pos]]
sns.heatmap(mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Legit', 'Pred Fraud'],
            yticklabels=['True Legit', 'True Fraud'])
plt.xlabel('Predicted'); plt.ylabel('True')
```

**Prosise's results:** ~50% fraud caught; 76 false positives out of 284,315 legit (**0.03%** false positive rate). Supervised RF did better on recall but PCA needs **no fraud labels** for training.

### Tuning tradeoff

| Lower threshold | Higher threshold |
|-----------------|------------------|
| Catch more fraud | Fewer false alarms |
| Angry customers (declined legit charges) | Miss more fraud |

Parameters: `n_components` (26) and `threshold` (200) - grid-search on validation fraud if labels exist for evaluation only.

---

## Part B: NASA Bearing Failure (Prosise)

Four bearings, vibration samples every 10 minutes, 984 rows. Bearing 1 fails catastrophically after rising vibration (~day 4+).

```python
import pandas as pd

df = pd.read_csv('Data/bearings.csv', index_col=0, parse_dates=True)
df.plot(figsize=(12, 6), title='Raw vibration - 4 bearings')
```

### Train on normal period only

```python
from sklearn.decomposition import PCA

x_train = df['2004-02-12 10:32:39':'2004-02-13 23:42:39']
x_test = df['2004-02-13 23:52:39':]

pca = PCA(n_components=1, random_state=0)  # multivariate → 1 systemic axis
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

import pandas as pd
df_pca = pd.concat([
    pd.DataFrame(x_train_pca, index=x_train.index),
    pd.DataFrame(x_test_pca, index=x_test.index)
])
df_pca.plot(figsize=(12, 6), title='PCA projection (1D)')
```

### Reconstruct and score

```python
df_restored = pd.DataFrame(
    pca.inverse_transform(df_pca), index=df_pca.index
)

scores = get_anomaly_scores(df, df_restored)
scores.plot(figsize=(12, 6), title='Reconstruction loss over time')
```

Loss stays tiny during normal operation; rises before failure. Threshold ~**0.002** flags impending failure **~2 days** early (Prosise). Threshold **0.0002** → ~3 days early, more false alarms.

### Row-level anomaly function

```python
def is_anomaly(row, pca, threshold):
    row_arr = np.array(row).reshape(1, -1)
    restored = pca.inverse_transform(pca.transform(row_arr))
    loss = np.sum((row_arr - restored) ** 2)
    return loss > threshold

# Normal row → False
# x = df.loc[['2004-02-16 22:52:39']]
# is_anomaly(x.iloc[0], pca, 0.002)

# Pre-failure row → True
# x = df.loc[['2004-02-18 22:52:39']]
# is_anomaly(x.iloc[0], pca, 0.002)
```

### Visual alert timeline

```python
df.plot(figsize=(12, 6))
for index, row in df.iterrows():
    if is_anomaly(row, pca, threshold=0.002):
        plt.axvline(index, color='r', alpha=0.2)
plt.title('Red lines = PCA anomaly flags')
```

---

## Multivariate vs Univariate Monitoring

Prosise: failure may show as **slight elevation across two bearings**, not one sensor screaming. PCA to 1D captures **covariance structure** - [multivariate anomaly detection](https://learn.microsoft.com/en-us/azure/ai-services/anomaly-detector/overview).

**Limitation:** PCA is **linear**. Nonlinear coupling needs autoencoders, graph attention networks (Microsoft Azure Multivariate Anomaly Detector - Prosise cites 2020 GAN paper).

---

## Production Checklist

| Step | Detail |
|------|--------|
| Baseline | Fit PCA on known-healthy window |
| Retrain | Periodic refresh as equipment ages |
| Threshold | Set from percentile of healthy scores (99th) |
| Alert | Score > threshold → ticket, not instant shutdown |
| Features | Align scaling with training |
| Drift | Monitor score distribution shift |

---

## Comparison: Supervised vs PCA Anomaly

| | Supervised (Ch. 3 RF) | PCA reconstruction |
|---|----------------------|-------------------|
| Labels needed | Fraud labels | Healthy baseline only |
| New fraud patterns | Needs retrain | May detect as anomalous |
| False positives | Very low (0.007%) | Low (0.03%) |
| Fraud recall | Higher | ~50% in Prosise demo |

Use both: PCA for screening, supervised for confirmed fraud patterns.

---

## Humor

Bearing 1: *"I'm fine."* PCA reconstruction error: *"That's what you said at 0.0018."* Bearing 1 at 0.0021: *"...okay fine."*

---

## Common Mistakes

| Mistake | Consequence |
|---------|-------------|
| Fit PCA on train + anomalies | Subspace includes failure modes |
| Same threshold across machines | Different vibration baselines |
| `n_components` too high | Anomalies reconstruct well - no signal |
| Ignoring `Amount` scale in fraud data | Standardize before PCA |

---

## Self-Check

1. Why fit PCA on legitimate transactions only for fraud?  
2. What do rising bearing scores before failure indicate geometrically?  
3. How does lowering threshold affect precision/recall tradeoff?  
4. Why is PCA(1) meaningful for 4 correlated sensors?

---

## Exercises

1. Sweep `n_components` ∈ {5, 10, 20, 26} on fraud data. Plot fraud capture vs false positive rate.
2. Replicate bearing red-line plot with thresholds 0.0002 and 0.002 side by side.
3. Add `StandardScaler` to fraud pipeline. Does separation improve?

---

## References

- [NASA IMS Bearing Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [Kaggle - Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Azure Multivariate Anomaly Detector](https://learn.microsoft.com/en-us/azure/ai-services/anomaly-detector/overview)
- Prosise, *Applied ML and AI for Engineers*, Ch. 6
- [Scikit-Learn: Outlier detection](https://scikit-learn.org/stable/chapters/outlier_detection.html)
- [GLOSSARY.md](../../GLOSSARY.md) - [pca](../../GLOSSARY.md#pca), [explained-variance](../../GLOSSARY.md#explained-variance)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 6.7 - Visualization & Exploration](./section-07-visualization-and-exploration-with-pca.md)  
**Next:** [Lab 06 - PCA in Practice](./section-lab-06-pca-in-practice.md)



