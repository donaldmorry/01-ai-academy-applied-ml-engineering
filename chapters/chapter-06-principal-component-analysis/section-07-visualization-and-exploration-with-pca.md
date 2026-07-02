# Section 6.7: Visualization & Exploration with PCA

> **Source:** Prosise, Ch. 6 - 2D/3D projections, exploratory data analysis  
> **Prerequisites:** [Section 6.4 - Choosing Components](./section-04-choosing-the-number-of-components.md)  
> **Glossary:** [pca](../../GLOSSARY.md#pca) | [classification](../../GLOSSARY.md#classification)  
> **Math:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## You Cannot Plot 2,914 Dimensions

Humans see in two (maybe three) dimensions. When EDA stalls because `pairplot` would need $\binom{p}{2}$ charts, **PCA projection** is the standard first move: keep the top two or three components, scatter-plot, color by label, and look for clusters, outliers, and separability.

Prosise motivates PCA partly through **exploration** - seeing structure that spreadsheets hide.

> **Analogy:** PCA visualization is a drone shot of a mountain range. You lose boulders and trees, but ridgelines and valleys - the geography that matters - pop out.

---

## 2D Scatter: The Workhorse

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
y = data.target

pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
for label, name, color in [(0, "malignant", "tab:red"), (1, "benign", "tab:blue")]:
    mask = y == label
    plt.scatter(X2[mask, 0], X2[mask, 1], alpha=0.6, label=name, c=color, s=25)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.legend()
plt.title("Breast cancer: first two principal components")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Read the axes labels** with variance percentages - if PC1+PC2 explain only 40%, distance in the plot distorts reality. You are viewing a **shadow**, not the full object.

---

## 3D for Extra Depth

```python
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

pca3 = PCA(n_components=3).fit_transform(X)
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(pca3[y==0, 0], pca3[y==0, 1], pca3[y==0, 2], c="tab:red", alpha=0.5, s=15)
ax.scatter(pca3[y==1, 0], pca3[y==1, 1], pca3[y==1, 2], c="tab:blue", alpha=0.5, s=15)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
plt.title("3D PCA projection")
plt.show()
```

Rotate interactively in Jupyter with `%matplotlib notebook` or export Plotly for stakeholders who want to spin the cloud.

---

## Coloring by Metadata

PCA is unsupervised; **color is your superpower**:

- Class labels (supervised sanity check)
- Cluster IDs from [k-means](../chapter-01-machine-learning/section-04-unsupervised-learning-k-means-clustering.md)
- Time (sensor drift)
- Categorical cohort (factory line, device firmware)

```python
import seaborn as sns

df = {"PC1": X2[:, 0], "PC2": X2[:, 1], "diagnosis": data.target_names[y]}
sns.scatterplot(data=df, x="PC1", y="PC2", hue="diagnosis", alpha=0.7)
plt.title("Seaborn PCA scatter")
plt.show()
```

Separated hues suggest linear separability in full space *might* be achievable - not proof, but a hint for [classification](../chapter-03-classification-models/README.md) feasibility.

---

## Handwritten Digits in 2D (Prosise)

Prosise's canonical exploration example - 64 pixel dimensions → 2 for plotting:

```python
from sklearn.datasets import load_digits

digits = load_digits()
pca_d = PCA(n_components=2, random_state=0)
Xd = pca_d.fit_transform(digits.data)

plt.figure(figsize=(12, 8))
plt.scatter(Xd[:, 0], Xd[:, 1], c=digits.target, cmap="tab10", alpha=0.7, s=30)
plt.colorbar(ticks=range(10), label="Digit")
plt.xlabel(f"PC1 ({pca_d.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca_d.explained_variance_ratio_[1]:.1%})")
plt.title("Digits - PCA 2D (Prosise Fig. 6-5 spirit)")
plt.tight_layout()
plt.show()
```

**Prosise's read:** 0 vs 1 separate well (top/bottom); 4 vs 6 overlap (expect confusion); 3 vs 4 spread left/right. A classifier would likely succeed on some pairs and struggle on others - the plot tells you *where* to inspect the confusion matrix before training.

---

## Biplots (Features + Samples)

A **biplot** overlays feature loadings as arrows on the sample scatter - which original columns drive PC1 vs PC2?

```python
def biplot(X2, pca, feature_names, n_arrows=10):
    loadings = pca.components_.T[:, :2]
    magnitudes = (loadings ** 2).sum(axis=1)
    top_idx = magnitudes.argsort()[-n_arrows:]
    plt.figure(figsize=(9, 7))
    plt.scatter(X2[:, 0], X2[:, 1], alpha=0.3, s=20)
    for i in top_idx:
        plt.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3,
                  head_width=0.05, color="darkgreen", alpha=0.8)
        plt.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, feature_names[i], fontsize=8)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("PCA biplot (top feature loadings)")
    plt.grid(True, alpha=0.3)
    plt.show()

biplot(X2, pca, data.feature_names)
```

Long arrows = features strongly correlated with that axis direction.

---

## Faces in 2D: What You See

LFW flattened to 2D often forms fuzzy islands per person - but overlap is common because PCA is linear:

```python
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
Xf = faces.data
yf = faces.target
Xf2 = PCA(n_components=2, random_state=0).fit_transform(Xf)

plt.figure(figsize=(10, 8))
sc = plt.scatter(Xf2[:, 0], Xf2[:, 1], c=yf, cmap="tab20", alpha=0.6, s=15)
plt.colorbar(sc, label="person id")
plt.title("LFW in PCA 2D - partial clustering by identity")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()
```

Good for spotting outliers and mislabeled images; **not** a replacement for [SVM face pipeline](../chapter-05-support-vector-machines/section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md).

---

## PCA vs t-SNE vs UMAP

| Method | Linear? | Preserves | Cost | Best use |
|--------|---------|-----------|------|----------|
| PCA | Yes | Global variance | Fast | First look, preprocessing |
| t-SNE | No | Local neighborhoods | Slower | Presentation clusters |
| UMAP | No | Local + some global | Medium | Large datasets, ML prep |

```python
from sklearn.manifold import TSNE

X_tsne = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(X[:500])
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y[:500], alpha=0.6, s=20)
plt.title("t-SNE (500 samples)")
plt.show()
```

**Rule:** Present t-SNE to executives for pretty clusters; document that distances between clusters are not meaningful. Use PCA when you need **reproducible, invertible** axes tied to variance.

---

## Interactive Plotly (Optional)

```python
import plotly.express as px
import pandas as pd

df = pd.DataFrame(X2, columns=["PC1", "PC2"])
df["label"] = data.target_names[y]
fig = px.scatter(df, x="PC1", y="PC2", color="label", opacity=0.7,
                 title="Interactive PCA - hover for exploration")
fig.show()
```

Export HTML to share with product managers who do not run notebooks.

---

## Exploration Workflow

1. **Scale** features.
2. **PCA** to 2-3 for plotting; PCA to 95% for modeling.
3. **Color** by label, time, cluster.
4. **Note** variance explained on axes.
5. **Flag** outliers far from main mass - candidates for [anomaly](./section-08-case-study-bearing-failure-and-credit-card-fraud-detection.md) review.
6. **Follow up** with task-aware models - plots inspire, they do not prove.

---

## Exercises

1. Replicate digits PCA 2D and t-SNE side by side. Which digit pairs separate better in each?
2. For Iris (4 features), build a biplot. Which features drive PC1?
3. Annotate a PCA scatter with reconstruction error from `PCA(10)` - do outliers have higher error?

---

## Milestone

**You can turn any high-dimensional table into an interpretable picture** - with honest axis labels and awareness that 2D is a projection, not the territory.

---

## Self-Check

1. Why must axis labels include explained variance percentages?
2. When would separated colors in a PCA plot mislead you?
3. What does a biplot arrow direction mean?
4. Why are t-SNE cluster distances not comparable?
5. Name one case where PCA beats t-SNE for production EDA.

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 6 - 2D/3D visualization; t-SNE comparison
- [Scikit-Learn: Manifold learning](https://scikit-learn.org/stable/chapters/manifold.html)
- van der Maaten & Hinton, t-SNE - https://www.jmlr.org/papers/volume9/vandermaaten08a/
- McInnes et al., UMAP - https://arxiv.org/abs/1802.03426
- [Plotly Express scatter](https://plotly.com/python-api-reference/generated/plotly.express.scatter)
- [GLOSSARY.md](../../GLOSSARY.md) - [pca](../../GLOSSARY.md#pca), [classification](../../GLOSSARY.md#classification)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 6.6 - Anonymization & Privacy](./section-06-anonymization-and-privacy-with-pca.md)  
**Next:** [Section 6.8 - Bearing Failure Detection](./section-08-case-study-bearing-failure-and-credit-card-fraud-detection.md)