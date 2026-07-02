# Lab 06: PCA in Practice

> **Prerequisites:** Sections [6.1](./section-01-the-curse-of-dimensionality.md)-[6.8](./section-08-case-study-bearing-failure-and-credit-card-fraud-detection.md)  
> **Estimated time:** 4-6 hours  
> **Glossary:** [pca](../../GLOSSARY.md#pca) | [eigenvector](../../GLOSSARY.md#eigenvector) | [explained-variance](../../GLOSSARY.md#explained-variance)  
> **Math:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Objectives

By the end of this lab you will:

1. **Compress grayscale images** - varying $k$, quality vs compression ratio
2. **Filter noise** via PCA reconstruct on LFW faces (Prosise demo)
3. **Detect credit card fraud** with reconstruction error (unsupervised)
4. **Monitor bearing failure** on multivariate vibration data (NASA subset)
5. Produce scree plots, reconstruction galleries, and precision-recall analysis

> **Humorous briefing:** You are the dimensionality reduction department. Throw away 90% of the numbers and still recognize faces, catch fraudsters, and alert before Bearing 1 goes kinetic.

---

## Setup

```python
# pip install numpy pandas matplotlib seaborn scikit-learn joblib
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_lfw_people, load_digits
from sklearn.metrics import (precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score)

RANDOM_STATE = 42; np.random.seed(RANDOM_STATE)
RESULTS = Path('lab06_results'); RESULTS.mkdir(exist_ok=True)
sns.set()

def recon_mse(X, pca):
    X_hat = pca.inverse_transform(pca.transform(X))
    return np.sum((X - X_hat) ** 2, axis=1)
```

---

## Part A: Image Compression (60 min)

*Follow [Section 6.5](./section-05-noise-filtering-and-compression.md).*

### A1. Load faces and scree plot

```python
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = faces.data
img_shape = faces.images[0].shape
n_features = X.shape[1]

pca_full = PCA(random_state=RANDOM_STATE).fit(X)
cev = np.cumsum(pca_full.explained_variance_ratio_)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(pca_full.explained_variance_ratio_[:100], marker='.')
axes[0].set(xlabel='Component', ylabel='EVR', title='Scree')
axes[1].plot(cev); axes[1].axhline(0.95, color='r', linestyle='--')
axes[1].set(xlabel='# Components', ylabel='CEV')
plt.savefig(RESULTS / 'a_scree.png', dpi=120)
```

### A2. Compression sweep + gallery

```python
ks = [10, 30, 75, 150, 300]
rows = []
for k in ks:
    pca = PCA(n_components=k, random_state=RANDOM_STATE).fit(X)
    X_hat = pca.inverse_transform(pca.transform(X))
    rows.append({'k': k, 'CEV': pca.explained_variance_ratio_.sum(),
                 'MSE': np.mean((X - X_hat)**2), 'CR': 1 - k/n_features})
print(pd.DataFrame(rows))

fig, axes = plt.subplots(1, len(ks)+1, figsize=(3*(len(ks)+1), 3))
axes[0].imshow(X[0].reshape(img_shape), cmap='gist_gray'); axes[0].axis('off')
for ax, k in zip(axes[1:], ks):
    p = PCA(n_components=k, random_state=RANDOM_STATE).fit(X)
    img = p.inverse_transform(p.transform(X))[0].reshape(img_shape)
    ax.imshow(img, cmap='gist_gray')
    ax.set_title(f'k={k}\n{p.explained_variance_ratio_.sum():.0%}')
    ax.axis('off')
plt.savefig(RESULTS / 'a_gallery.png', dpi=120)
```

**Deliverable:** Table of $k$ vs CEV vs MSE + gallery figure.

---

## Part B: Noise Filtering (45 min)

```python
np.random.seed(0)
noisy = np.random.normal(X, 20)

pca_dn = PCA(0.8, random_state=0)
pca_dn.fit(X)  # fit on CLEAN - transform noisy
denoised = pca_dn.inverse_transform(pca_dn.transform(noisy))
print(f"Components kept: {pca_dn.n_components_}")

mse_noisy = np.mean((X - noisy)**2)
mse_clean = np.mean((X - denoised)**2)
print(f"MSE noisy={mse_noisy:.1f}, denoised={mse_clean:.1f}")
```

Show 3×3 grid: original | noisy | denoised for 3 faces.

**Written (2-3 sentences):** Why fit on clean but transform noisy?

---

## Part C: Credit Card Fraud Anomaly (90 min)

*Prosise [Section 6.8](./section-08-case-study-bearing-failure-and-credit-card-fraud-detection.md) - fraud section.*

```python
df = pd.read_csv('Data/creditcard.csv')
feat = [c for c in df.columns if c not in ('Time', 'Class')]
legit = df[df['Class']==0][feat]
fraud = df[df['Class']==1][feat]

scaler = StandardScaler()
X_legit = scaler.fit_transform(legit)
X_fraud = scaler.transform(fraud)

pca = PCA(n_components=26, random_state=RANDOM_STATE)
pca.fit(X_legit)

legit_scores = recon_mse(X_legit, pca)
fraud_scores = recon_mse(X_fraud, pca)

plt.figure(figsize=(12,4))
plt.hist(legit_scores, bins=100, alpha=0.6, label='Legit', density=True)
plt.hist(fraud_scores, bins=50, alpha=0.8, label='Fraud', density=True)
plt.axvline(200, color='r', linestyle='--', label='thr=200')
plt.legend(); plt.savefig(RESULTS / 'c_fraud_hist.png', dpi=120)
```

### Confusion matrix @ threshold=200

```python
threshold = 200
y_true = np.concatenate([np.zeros(len(legit_scores)), np.ones(len(fraud_scores))])
y_scores = np.concatenate([legit_scores, fraud_scores])
y_pred = (y_scores >= threshold).astype(int)
ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred),
    display_labels=['Legit','Fraud']).plot(cmap='Blues')
plt.savefig(RESULTS / 'c_confusion.png', dpi=120)
```

### Precision-recall curve

```python
prec, rec, _ = precision_recall_curve(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)
plt.figure(figsize=(8,5))
plt.plot(rec, prec, label=f'AP={ap:.3f}')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend()
plt.savefig(RESULTS / 'c_pr.png', dpi=120)
```

**Tasks:** Sweep `N_COMP` ∈ {10, 20, 26, 28}. Plot AP vs components. Find F1-maximizing threshold.

---

## Part D: Bearing Failure (90 min)

*NASA IMS subset - Prosise `bearings.csv`.*

```python
bearings = pd.read_csv('Data/bearings.csv', index_col=0, parse_dates=True)
bearings.plot(figsize=(12,5), title='4-bearing vibration')
plt.savefig(RESULTS / 'd_raw.png', dpi=120)

train_end = '2004-02-13 23:42:39'
x_train = bearings.loc[:train_end]

pca_b = PCA(n_components=1, random_state=RANDOM_STATE)
pca_b.fit(x_train)

Z = pca_b.transform(bearings)
X_hat = pca_b.inverse_transform(Z)
scores_b = pd.Series(recon_mse(bearings.values, pca_b), index=bearings.index)
scores_b.plot(figsize=(12,4), title='Reconstruction loss')
plt.savefig(RESULTS / 'd_scores.png', dpi=120)
```

### Timeline with alert lines

```python
for thr, tag in [(0.002, '2day'), (0.0002, '3day')]:
    fig, ax = plt.subplots(figsize=(12,5))
    bearings.plot(ax=ax, alpha=0.7)
    for ts in scores_b.index[scores_b > thr]:
        ax.axvline(ts, color='r', alpha=0.15, linewidth=1)
    ax.set_title(f'Alerts threshold={thr}')
    plt.savefig(RESULTS / f'd_alerts_{tag}.png', dpi=120)
```

### 2D trajectory (time-colored)

```python
mid = len(bearings) // 2
pca2 = PCA(2, random_state=RANDOM_STATE).fit(bearings.iloc[:mid])
coords = pca2.transform(bearings)
plt.scatter(coords[:,0], coords[:,1], c=np.arange(len(bearings)), cmap='viridis', s=15)
plt.colorbar(label='time'); plt.savefig(RESULTS / 'd_pca2d.png', dpi=120)
```

---

## Part E: Written Interpretation (30 min)

Answer in notebook (150-300 words):

1. First 3 EVR values for faces - which components capture most signal?
2. Why does PCA fraud detection need no fraud labels in training?
3. When did bearing alerts fire vs visible failure in raw plots?
4. One scenario where PCA anomaly detection fails (linearity).

---

## Submission Checklist

| Item | File |
|------|------|
| Scree + CEV | `a_scree.png` |
| Compression gallery | `a_gallery.png` |
| Fraud histogram + confusion + PR | `c_*.png` |
| Bearing timeline + 2D | `d_*.png` |
| Interpretation | notebook markdown |

---

## Extension Challenges

1. **Anonymization:** Full-rank PCA + scaler on breast cancer ([Section 6.6](./section-06-anonymization-and-privacy-with-pca.md)).
2. **Digits 2D:** `load_digits()` scatter per [Section 6.7](./section-07-visualization-and-exploration-with-pca.md).
3. **Real bearings:** NASA IMS full dataset - compare to synthetic Part C fallback.

---

## References

- [Scikit-Learn PCA](https://scikit-learn.org/stable/chapters/decomposition.html#pca)
- [Kaggle - Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [NASA IMS Bearing Data](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [LFW Dataset](http://vis-www.cs.umass.edu/lfw/)
- Prosise, *Applied ML and AI for Engineers*, Ch. 6
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 6.8 - Bearing Failure Detection](./section-08-case-study-bearing-failure-and-credit-card-fraud-detection.md)  
**Next:** [Chapter 07 - Operationalizing ML Models](../chapter-07-operationalizing-models/README.md)