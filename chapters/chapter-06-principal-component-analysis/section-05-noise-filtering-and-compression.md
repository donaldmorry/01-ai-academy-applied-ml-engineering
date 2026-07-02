# Section 6.5: Noise Filtering & Compression

> **Source:** Prosise, Ch. 6 - PCA denoising, LFW noise demo, image compression  
> **Prerequisites:** [Section 6.4 - Choosing Components](./section-04-choosing-the-number-of-components.md)  
> **Glossary:** [pca](../../GLOSSARY.md#pca) | [feature](../../GLOSSARY.md#feature)  
> **Math:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Signal, Noise, and the PCA Assumption

Physical sensors - accelerometers on bearings, pressure transducers, microphones - rarely return pure truth. Thermal drift, electrical interference, and quantization add **noise**: variance that does not help prediction.

PCA's bet: **meaningful structure lives in a low-dimensional subspace**; noise spreads across many directions with small eigenvalues. Truncate the small eigenvalues, reconstruct, and noise averages out while signal survives.

> **Analogy:** A choir in a stadium. The melody (signal) is correlated across microphones near the stage. Random crowd chatter (noise) is uncorrelated. PCA keeps the harmonized directions and drops the static.

> **Humor break:** PCA is the ML equivalent of telling your friend to talk louder and pretend the construction site next door doesn't exist - surprisingly effective when the noise isn't structured like the signal.

---

## The Denoising Recipe

Prosise's pattern (works for faces, sensors, any row-vectors):

1. **Fit PCA** on clean or mostly-clean training data (or all data if noise is mild).
2. **Transform** to component space.
3. **Inverse transform** back to original dimensionality.

Discarded components cannot return - their energy is gone. If noise occupied those directions, it is attenuated.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.90)  # keep 90% variance - drop noisy tail
X_pca = pca.fit_transform(X_scaled)
X_denoised = scaler.inverse_transform(pca.inverse_transform(X_pca))
```

For images stored as flat vectors, reshape for display:

```python
img_clean = X_denoised[i].reshape(height, width)
```

---

## LFW Noise Experiment (Prosise)

Add Gaussian noise, then PCA-filter:

```python
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import numpy as np

faces = fetch_lfw_people(min_faces_per_person=100)
rng = np.random.RandomState(0)
noisy = faces.data + rng.normal(0, 20, faces.data.shape)

pca = PCA(n_components=0.95, svd_solver="full", random_state=0)
pca.fit(faces.data)  # fit on clean distribution
denoised = pca.inverse_transform(pca.transform(noisy)).reshape(-1, 62, 47)

fig, axes = plt.subplots(3, 6, figsize=(12, 7))
for i in range(6):
    axes[0, i].imshow(faces.images[i], cmap="gray")
    axes[1, i].imshow(noisy[i].reshape(62, 47), cmap="gray")
    axes[2, i].imshow(denoised[i], cmap="gray")
    axes[0, i].axis("off"); axes[1, i].axis("off"); axes[2, i].axis("off")
axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Noisy")
axes[2, 0].set_ylabel("PCA denoised")
plt.tight_layout()
plt.show()
```

Middle row: staticky TV. Bottom row: recognizable faces - PCA assumed the face manifold is low-rank and noise is not.

**Caveat:** If noise correlates with failure modes you need to detect, aggressive truncation can erase the very signal you care about. Tune on domain validation, not aesthetics alone.

---

## Compression as the Same Math

Image compression with PCA treats each image as a vector $\mathbf{x} \in \mathbb{R}^p$. Store only coefficients $\mathbf{z} = \mathbf{W}^T \mathbf{x}$ in $k \ll p$ dimensions.

**Compression ratio** (rough):

$$
\text{ratio} \approx \frac{p}{k + \frac{k \cdot p}{N_{\text{images}}}}
$$
> **Readable form:** you pay $k$ coefficients per image plus one-time cost of storing the $k$ basis vectors of length $p$ shared across images.

For a dataset of $N$ face images, $p = 2914$, $k = 150$:

- Raw storage: $N \times 2914$ floats
- PCA storage: $N \times 150 + 150 \times 2914$ floats

At $N = 1140$, savings are substantial.

**Quality knob:** Lower $k$ → smaller files, blurrier images. Plot MSE vs $k$:

```python
ks = [10, 25, 50, 100, 150, 2914]
mses = []
for k in ks:
    pca = PCA(n_components=k, random_state=0)
    Z = pca.fit_transform(faces.data)
    recon = pca.inverse_transform(Z)
    mses.append(np.mean((faces.data - recon) ** 2))

plt.plot(ks, mses, marker="o")
plt.xlabel("Components kept")
plt.ylabel("Mean squared reconstruction error")
plt.title("PCA compression quality")
plt.show()
```

---

## Eigenfaces Interpretation

Each row of `pca.components_` reshaped to image size is an **eigenface** - a basis image. Reconstruction is a weighted sum:

$$
\hat{\mathbf{x}} \approx \bar{\mathbf{x}} + \sum_{i=1}^{k} z_i \mathbf{v}_i
$$
> **Readable form:** reconstructed image ≈ mean face plus weighted combination of the first $k$ eigenface patterns.

Visualizing top eigenfaces explains *what* information you keep:

```python
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62, 47), cmap="seismic")
    ax.set_title(f"PC {i+1}")
    ax.axis("off")
plt.suptitle("Top 10 eigenfaces (LFW)")
plt.show()
```

---

## Sensor Noise: Practical Workflow

For multivariate vibration readings (preview of [Section 6.8](./section-08-case-study-bearing-failure-and-credit-card-fraud-detection.md)):

1. Collect **healthy-only** baseline windows.
2. Standardize per channel.
3. Fit PCA with variance threshold 90-95%.
4. On streaming data, reconstruct; flag windows where:

$$
\|\mathbf{x} - \hat{\mathbf{x}}\|_2 > \tau
$$
> **Readable form:** reconstruction error exceeds threshold $\tau$ learned from healthy validation quantile (e.g. 99th percentile).

$\tau$ too low → false alarms; too high → missed faults.

---

## PCA vs JPEG vs Autoencoders

| Method | Linear? | Best for |
|--------|---------|----------|
| PCA | Yes | Quick baseline, theory, linear manifolds |
| JPEG (DCT) | Fixed basis | Natural images, standards |
| Autoencoder (Course 3) | No | Nonlinear manifolds, complex denoising |

PCA wins on **simplicity and speed** - no GPU, no epochs, closed-form solution. When linear subspace is wrong, upgrade to nonlinear methods - but try PCA first; Prosise's bearing and face examples show how far it goes.

---

## Information Loss You Should Expect

PCA compression is **lossy**. You lose:

- Fine texture and high-frequency detail
- Rare events aligned with discarded eigenvectors
- Out-of-distribution samples poorly represented by training subspace

Always show side-by-side originals and reconstructions to stakeholders. Numbers (`explained_variance_ratio_`) plus pictures build trust.

---

## Milestone

**You can denoise and compress high-dimensional data with the same transform** - `fit_transform` + `inverse_transform` - and quantify quality with variance retained and reconstruction MSE.

---

## Self-Check

1. Why does noise often occupy trailing principal components?
2. Why fit PCA on clean data when denoising noisy test images?
3. What happens if you keep too many components when denoising?
4. How do eigenfaces relate to `pca.components_`?
5. When would you prefer an autoencoder over PCA for compression?

---

## References

- Prosise, Ch. 6 - noise filtering with LFW
- Turk & Pentland, "Eigenfaces for Recognition" - https://doi.org/10.1109/34.164876
- Scikit-Learn `inverse_transform` - https://scikit-learn.org/stable/chapters/decomposition.html
- Hinton & Salakhutdinov, reducing dimensionality with neural networks - Course 3 preview

---

**Previous:** [Section 6.4 - Choosing Components](./section-04-choosing-the-number-of-components.md)  
**Next:** [Section 6.6 - Anonymization & Privacy](./section-06-anonymization-and-privacy-with-pca.md)



