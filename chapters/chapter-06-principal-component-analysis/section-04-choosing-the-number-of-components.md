# Section 6.4: Choosing the Number of Components

> **Source:** Prosise, Ch. 6 - scree plots, cumulative variance, component trade-offs  
> **Prerequisites:** [Section 6.3 - PCA with Scikit-Learn](./section-03-pca-with-scikit-learn.md)  
> **Glossary:** [pca](../../GLOSSARY.md#pca) | [overfitting](../../GLOSSARY.md#overfitting) | [bias-variance tradeoff](../../GLOSSARY.md#bias-variance-tradeoff)  
> **Math:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Question Every Engineer Asks

You can keep anywhere from 1 to $p$ principal components. Keep too few and reconstructions blur; downstream models miss signal. Keep too many and you save little dimensionality, retain noise, and risk [overfitting](../../GLOSSARY.md#overfitting) when samples are scarce.

Prosise's LFW experiment poses it directly: **150 components retained 94.8% variance - but would 50 suffice?** There is no universal answer; there *is* a disciplined process.

> **Analogy:** Choosing components is like picking JPEG quality. Quality 10 is tiny but blocky; quality 95 is huge with imperceptible gain. You want the knee of the curve - where extra storage buys almost nothing visible.

---

## Three Decision Strategies

### 1. Variance Threshold (Engineering Default)

Keep the smallest $k$ such that cumulative explained variance ≥ target (often 90-99%):

$$
\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{p} \lambda_j} \geq 0.95
$$
> **Readable form:** sum of top-$k$ eigenvalues divided by total variance reaches 95%.

Scikit-Learn one-liner:

```python
PCA(n_components=0.95)
```

**Pros:** Objective, reproducible, matches "retain information" goal.  
**Cons:** Ignores downstream task - 95% global variance might drop class-separating directions.

### 2. Scree Plot (Elbow Hunting)

Plot `explained_variance_ratio_` vs component index:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
pca = PCA().fit(X_scaled)

plt.figure(figsize=(8, 4))
plt.plot(pca.explained_variance_ratio_, marker="o", markersize=3)
plt.xlabel("Principal Component Index")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.grid(True, alpha=0.3)
plt.show()
```

Look for the **elbow** where the curve flattens - additional components add diminishing returns. Prosise notes for LFW: bulk of information sits in the first 50-100 components.

### 3. Downstream Validation (Task-Aware)

Sweep $k$, measure what you actually care about:

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

ks = [10, 25, 50, 100, 150, 200]
scores = []
for k in ks:
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=k)),
        ("clf", SVC(kernel="rbf")),
    ])
    scores.append(cross_val_score(pipe, X, y, cv=5).mean())
```

Plot $k$ vs CV accuracy or RMSE. The best $k$ balances compression and predictive power - not always the same as 95% variance.

---

## Cumulative Variance Plot

Prosise's second visualization - often more actionable than raw scree:

```python
cumvar = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 4))
plt.plot(cumvar, linewidth=2)
plt.axhline(0.95, color="r", linestyle="--", label="95% threshold")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

k_95 = np.searchsorted(cumvar, 0.95) + 1
print(f"Components needed for 95%: {k_95}")
```

Draw horizontal lines at 90%, 95%, 99% and read off $k$. Document your choice in experiment notes - future you will thank present you.

---

## Bias-Variance in Reconstruction

Treat PCA compression as a bias-variance knob:

| Few components | Many components |
|----------------|-----------------|
| High bias (blur, lost detail) | Low bias |
| Low variance (stable, denoised) | High variance (noise retained) |
| Fast models, small storage | Heavy compute, near-copy of data |

For **anomaly detection** via reconstruction error, too many components memorize noise - anomalies hide. Too few components make everything look anomalous. Tune $k$ on held-out *normal* data reconstruction error distribution.

---

## Kaiser Rule (Optional Classical Heuristic)

From factor analysis literature: keep components with eigenvalue $> 1$ **when data were standardized** (each feature unit variance). Equivalently, keep PCs explaining more than average single-feature variance.

Useful as a sanity check, not a law - high-dimensional sensor data often needs more or fewer components than Kaiser suggests.

---

## Sample-to-Feature Ratio Revisited

From [Section 6.1](./section-01-the-curse-of-dimensionality.md): aim for rows ≥ 5 × columns after reduction.

If you have 500 rows and 200 features, reducing to 50 components gives ratio 10:1 - healthier for [SVM](../chapter-05-support-vector-machines/README.md) or logistic regression. The variance threshold is necessary but not sufficient; always check $n/k$.

---

## Worked Example: Bearing Features

Suppose vibration features $p = 80$, $n = 400$ healthy readings.

```python
pca = PCA().fit(StandardScaler().fit_transform(X_healthy))
cumvar = np.cumsum(pca.explained_variance_ratio_)

for target in [0.90, 0.95, 0.99]:
    k = np.searchsorted(cumvar, target) + 1
    print(f"{target*100:.0f}% variance → k={k}, n/k={400/k:.1f}")
```

Typical output might show 90% at $k=12$, 95% at $k=22$, 99% at $k=38$. For live monitoring, engineers often pick 95% - aggressive enough to denoise, mild enough to preserve fault signatures.

---

## Comparing 50 vs 150 on Faces

Prosise invites experimentation. Quick loop:

```python
for n_comp in [50, 100, 150]:
    pca = PCA(n_components=n_comp, random_state=0)
    reduced = pca.fit_transform(faces.data)
    back = pca.inverse_transform(reduced).reshape(-1, 62, 47)
    var = pca.explained_variance_ratio_.sum()
    print(f"k={n_comp}, variance={var:.3f}")
```

At $k=50$, faces may look noticeably softer; at 150, nearly indistinguishable from originals. Your deployment budget (latency, RAM) picks the point on that curve.

---

## Documenting the Decision

Production models need metadata:

```python
metadata = {
    "pca_n_components": 22,
    "cumulative_variance": float(cumvar[21]),
    "scaler": "StandardScaler",
    "fit_samples": len(X_train),
    "rationale": "95% variance + CV plateau on bearing classifier",
}
```

Store alongside the serialized [pipeline](../chapter-07-operationalizing-models/section-02-model-serialization-with-pickle-and-joblib.md).

---

## When Not to Use Variance Alone

| Scenario | Better approach |
|----------|-----------------|
| Classification with rare classes | Task-aware CV sweep |
| Anomaly detection | Reconstruction error on normal validation set |
| Visualization only | $k = 2$ or $k = 3$ regardless of variance |
| Regulatory "no information loss" | Higher threshold (99%+) or skip PCA |

---

## Milestone

**You can justify your choice of $k$ with plots and numbers** - not gut feel. Scree and cumulative curves make the trade-off visible; downstream metrics make it accountable.

---

## Self-Check

1. What does the elbow in a scree plot represent intuitively?
2. If 95% variance requires 200 of 250 features, is PCA worthwhile?
3. Why might the best $k$ for classification differ from the 95% variance $k$?
4. How does too-large $k$ hurt PCA-based anomaly detection?
5. What sample-to-feature ratio do you target after reduction?

---

## References

- Prosise, Ch. 6 - scree and cumulative variance plots (LFW)
- Cattell, scree test - classic elbow motivation
- Scikit-Learn: `PCA(n_components=float)` - https://scikit-learn.org/stable/chapters/decomposition.html#pca
- James et al., *ISLR* - bias-variance tradeoff - https://www.statlearning.com/

---

**Previous:** [Section 6.3 - PCA with Scikit-Learn](./section-03-pca-with-scikit-learn.md)  
**Next:** [Section 6.5 - Noise Filtering & Compression](./section-05-noise-filtering-and-compression.md)



