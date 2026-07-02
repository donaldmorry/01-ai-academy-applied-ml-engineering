# Section 6.6: Anonymization & Privacy with PCA

> **Source:** Prosise, Ch. 6 - anonymizing datasets via dimensionality reduction  
> **Prerequisites:** [Section 6.5 - Noise Filtering](./section-05-noise-filtering-and-compression.md)  
> **Glossary:** [pca](../../GLOSSARY.md#pca) | [feature](../../GLOSSARY.md#feature)  
> **Math:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Why PCA Shows Up in Privacy Discussions

Engineers share datasets with vendors, researchers, and other teams. Raw rows may contain **identifiable structure**: faces, voiceprints, exact GPS traces, or unique combinations of quasi-identifiers (ZIP code + birth date + gender).

Prosise lists **anonymization** as a PCA use case: reduce dimensions, share transformed data, hope identities are harder to recover. This section explains what PCA actually removes, what it **does not** guarantee, and safer patterns for real compliance work.

> **Analogy:** PCA anonymization is like sending a sketch instead of a photograph. Recognizable if the person is famous and the sketch is good - useless as a legal shield by itself.

> **In plain English:** PCA can obscure some identifying detail, but it is **not** a substitute for formal privacy engineering (k-anonymity, differential privacy, access controls).

---

## What PCA Removes (and What It Keeps)

When you publish $\mathbf{z} = \mathbf{W}_k^T \mathbf{x}$ with only $k$ components:

- **Removed:** Projection onto the $(p-k)$-dimensional orthogonal complement - often fine-grained detail.
- **Kept:** Everything in the span of top-$k$ eigenvectors - often the dominant patterns that **identify** faces, voices, or writing style.

For LFW faces, the first 50 components still look like *someone*. Identity lives in high-variance directions - exactly where PCA focuses.

**Implication:** Aggressive truncation can make humans squint harder, but a motivated attacker with side information or a secondary model may still re-identify records.

---

## Prosise's Sharing Scenario

Typical workflow described in the chapter:

1. Collect sensitive matrix $X$ (e.g. employee health metrics, customer usage vectors).
2. Fit PCA on $X$; keep $k$ components capturing 90-95% variance.
3. Release $Z = \text{PCA}(X)$ instead of $X$.
4. Optionally discard `pca.components_` if recipients should not reconstruct.

**Reconstruction risk:** If you release **both** $Z$ and the PCA model (components + mean), recipients can approximate originals:

$$
\hat{\mathbf{x}} = \mathbf{W}_k \mathbf{z} + \boldsymbol{\mu}
$$
> **Readable form:** approximate original = principal subspace reconstruction plus mean vector.

For true obscuration, never ship the inverse map - but then downstream partners cannot interpret features in original units.

---

## Prosise Demo: Breast Cancer Full-Rank Rotation

Prosise's anonymization recipe keeps **all** components ($m \rightarrow m$), then standardizes - obscuring column meaning while preserving total variance:

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.head())  # interpretable: mean radius, texture, ...

pca = PCA(n_components=30, random_state=0)
pca_data = pca.fit_transform(df)

scaler = StandardScaler()
anon_df = pd.DataFrame(scaler.fit_transform(pca_data))
print(anon_df.head())  # unrecognizable floats

assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0)
```

**Utility check** - models should still work:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

y = data.target
raw_acc = cross_val_score(LogisticRegression(max_iter=5000), df, y, cv=5).mean()
anon_acc = cross_val_score(LogisticRegression(max_iter=5000), anon_df, y, cv=5).mean()
print(f"Raw CV acc: {raw_acc:.3f}, Anonymized: {anon_acc:.3f}")
```

> **Humor:** The columns now look like lottery numbers. The model does not care - it only cares that rows still point in the same directions in 30-D space.

---

## Credit Card Fraud Dataset (Chapter 3 Link)

Prosise's `creditcard.csv` features `V1`-`V28` are **already** PCA-style transforms of confidential inputs - only `Time` and `Amount` remain human-readable. That is industry-grade anonymization before ML ever runs. Your DIY PCA rotation on other tables mimics that pattern at smaller scale. Fraud detection with reconstruction error is covered in [Section 6.8](./section-08-case-study-bearing-failure-and-credit-card-fraud-detection.md).

---

## Identity vs Utility Trade-off

| More components | Fewer components |
|-----------------|------------------|
| Higher utility for ML partners | More obscuration |
| Easier inversion | More bias in reconstructions |
| Stronger re-identification risk | May break legitimate analysis |

Plot cumulative variance vs $k$ ([Section 6.4](./section-04-choosing-the-number-of-components.md)) **and** measure re-identification risk on a hold-out:

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def reid_score(X, y, k):
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=k)),
        ("clf", KNeighborsClassifier(n_neighbors=1)),
    ])
    return cross_val_score(pipe, X, y, cv=5).mean()

for k in [5, 10, 25, 50, 100]:
    print(f"k={k:3d}, 1-NN identity accuracy: {reid_score(faces.data, faces.target, k):.3f}")
```

If 1-NN on PCA features still gets 90% person-ID accuracy, you have **not** anonymized - you have compressed.

---

## Removing "Identity" Components Explicitly

Some practitioners **drop the first** few PCs (high variance identity) and keep middle components for analysis - a heuristic, not cryptography:

```python
pca = PCA(n_components=100)
Z = pca.fit_transform(X_scaled)
Z_privacy = Z[:, 5:]  # discard first 5 PCs
```

Works only if identity aligns with top eigenvectors **and** your threat model stops at linear recovery. Faces violate the second assumption regularly.

---

## Linkage Attacks Still Work

Even "anonymous" $Z$ can be joined with external tables on overlapping statistics. Classic section from the Netflix Prize: "anonymized" ratings were linked to IMDb public profiles. PCA does not prevent **linkage** if auxiliary data exists.

**Engineering checklist for sharing:**

- [ ] Legal review (GDPR, HIPAA, CCPA as applicable)
- [ ] Minimum necessary fields - drop columns before PCA
- [ ] Aggregate or bin quasi-identifiers
- [ ] Access controls, contracts, audit logs
- [ ] Consider **differential privacy** noise on released statistics
- [ ] Red-team re-identification attempt

---

## Differential Privacy Contrast (Conceptual)

**Differential privacy** adds calibrated noise so any single row's presence barely changes outputs. PCA alone has **no** such guarantee.

$$
\Pr[\mathcal{A}(D) \in S] \leq e^{\epsilon} \Pr[\mathcal{A}(D') \in S]
$$
> **Readable form:** algorithm output on neighboring databases $D$ and $D'$ (differ by one row) should be almost equally likely - PCA does not enforce this.

For public releases, explore DP-SGD training or DP releases of aggregates - beyond Prosise's scope but critical for production privacy.

---

## When PCA Anonymization Is Reasonable

Legitimate narrow uses:

- **Internal sandboxes** where reconstruction is blocked and IDs were removed upstream
- **Feature sharing** when original columns are proprietary scalings but partners only need relative geometry in reduced space
- **Quick pilot** before investing in formal privacy tech

Not sufficient for:

- Public open-data portals with sensitive attributes
- HIPAA "de-identified" claims without expert determination
- Biometric datasets

---

## Better Patterns Alongside PCA

1. **Remove direct identifiers** - name, email, device ID - before any ML.
2. **Generalize** - age bands, coarse geography.
3. **k-anonymity / l-diversity** on quasi-identifiers.
4. **Synthetic data** generators (Course 4) trained with privacy budgets.
5. **Federated learning** - models move, data stays.

PCA can be a **preprocessing** step inside a broader policy, not the policy itself.

---

## Documentation for Compliance Auditors

If you cite PCA in a data-sharing memo, document:

```yaml
transformation: PCA
n_components: 50
variance_retained: 0.92
identifiers_removed: [user_id, email]
re_identification_test: "1-NN accuracy 12% on held-out (baseline 99% raw)"
residual_risk: "moderate - not for public release"
```

Honest metrics beat hand-waving about "dimensions reduced."

---

## Milestone

**You understand PCA's limited role in privacy** - useful compression that may obscure detail, not a guarantee of anonymity. Pair it with identifier removal, access control, and proper privacy methods when stakes are high.

---

## Self-Check

1. Why do top principal components often encode identity for faces?
2. What must you withhold to prevent exact linear reconstruction?
3. Why can 1-NN on PCA features measure anonymization failure?
4. Name two attack vectors PCA does not stop.
5. When is PCA-only sharing defensible internally?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 3 & 6 - anonymization; credit card `V1`-`V28`
- [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Scikit-Learn PCA](https://scikit-learn.org/stable/chapters/decomposition.html#pca)
- Narayanan & Shmatikov, Netflix re-identification - https://doi.org/10.1109/SP.2008.33
- GDPR Recital 26 - https://gdpr-info.eu/
- Dwork & Roth, *Algorithmic Foundations of Differential Privacy* - https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
- [GLOSSARY.md](../../GLOSSARY.md) - [pca](../../GLOSSARY.md#pca), [feature](../../GLOSSARY.md#feature)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 6.5 - Noise Filtering & Compression](./section-05-noise-filtering-and-compression.md)  
**Next:** [Section 6.7 - Visualization & Exploration](./section-07-visualization-and-exploration-with-pca.md)
