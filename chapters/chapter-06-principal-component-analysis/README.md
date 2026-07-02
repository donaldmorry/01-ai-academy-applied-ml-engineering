# Chapter 06: Principal Component Analysis

> **Source:** *Applied Machine Learning and AI for Engineers*, Chapter 6  
> **Part:** I — Machine Learning with Scikit-Learn  
> **Estimated time:** 6–8 hours  
> **Prerequisites:** [Chapter 05 — Support Vector Machines](../chapter-05-support-vector-machines/README.md); basic linear algebra (eigenvalues, matrix multiplication)

---

## Chapter Overview

High-dimensional data is everywhere — images with thousands of pixels, sensors with hundreds of channels, datasets with dozens of correlated features. **Principal Component Analysis (PCA)** finds a lower-dimensional representation that preserves as much variance as possible, enabling visualization, noise reduction, compression, and faster downstream modeling.

This chapter goes beyond "reduce to 2D and plot." You will learn the theory behind PCA (covariance, eigenvectors, explained variance), apply it to **noise filtering** and **data anonymization**, use it for **anomaly detection**, and visualize complex datasets in 2D and 3D. The engineering case study is **bearing failure detection** — using PCA on vibration sensor data to identify equipment anomalies before catastrophic failure.

PCA bridges classical ML and deep learning: the same dimensionality-reduction intuition underlies autoencoders (Course 3) and embedding layers (Chapter 13).

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain PCA geometrically as finding orthogonal directions of maximum variance
2. Compute and interpret explained variance ratios
3. Choose the number of components using scree plots and cumulative variance thresholds
4. Apply PCA for noise filtering and signal reconstruction
5. Use PCA for data anonymization by removing principal components that encode identity
6. Visualize high-dimensional data in 2D and 3D with PCA projections
7. Detect anomalies using reconstruction error from PCA
8. Apply PCA to bearing failure detection on multivariate sensor time series

---

## Sections

| # | Section | File | Topics |
|---|--------|------|--------|
| 6.1 | The Curse of Dimensionality | [01-curse-of-dimensionality.md](./section-01-the-curse-of-dimensionality.md) | Why high dimensions hurt; redundancy and correlation |
| 6.2 | PCA Theory | [02-pca-theory.md](./section-02-pca-theory-covariance-eigenvectors-and-principal-components.md) | Covariance matrix; eigenvectors/eigenvalues; principal components |
| 6.3 | PCA with Scikit-Learn | [03-pca-scikit-learn.md](./section-03-pca-with-scikit-learn.md) | `PCA`, `fit_transform`, `explained_variance_ratio_`; whitening |
| 6.4 | Choosing Components | [04-choosing-components.md](./section-04-choosing-the-number-of-components.md) | Scree plots; 95% variance rule; bias-variance in reconstruction |
| 6.5 | Noise Filtering & Compression | [05-noise-filtering-compression.md](./section-05-noise-filtering-and-compression.md) | Reconstructing signals; image compression demo; information loss |
| 6.6 | Anonymization & Privacy | [06-anonymization-privacy.md](./section-06-anonymization-and-privacy-with-pca.md) | Removing identity-encoding components; limitations of PCA anonymization |
| 6.7 | Visualization & Exploration | [07-visualization-exploration.md](./section-07-visualization-and-exploration-with-pca.md) | 2D/3D scatter plots; coloring by class; cluster structure discovery |
| 6.8 | Case Study: Bearing Failure Detection | [08-bearing-failure-detection.md](./section-08-case-study-bearing-failure-and-credit-card-fraud-detection.md) | Sensor PCA; reconstruction error thresholds; early anomaly alerts |

---

## Lab

**[Lab 06: PCA in Practice](./section-lab-06-pca-in-practice.md)**

Complete three PCA applications:

1. **Image Compression** — Load a grayscale image, apply PCA with varying component counts, reconstruct and compare quality vs compression ratio.
2. **Anomaly Detection** — Train PCA on "normal" sensor or network traffic data; flag anomalies where reconstruction error exceeds a threshold.
3. **Bearing Failure** — Work with multivariate vibration features from bearing sensors. Use PCA to visualize healthy vs failing bearings and build a simple anomaly detector.

*Deliverable:* Notebook with scree plots, reconstruction examples, anomaly detection ROC or precision-recall, and written interpretation of which components capture the most signal.

See **[Lab 06: PCA in Practice](./section-lab-06-pca-in-practice.md)** for step-by-step instructions.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| PCA & eigendecomposition | Singular value decomposition, matrix factorization (Course 2) |
| Dimensionality reduction | t-SNE, UMAP, manifold learning (Course 2) |
| Anomaly detection via reconstruction | Autoencoders, variational methods (Course 3) |
| PCA before SVM (Chapter 05) | Kernel PCA, nonlinear reduction (Course 2) |
| Sensor anomaly detection | Time-series models, LSTMs (Chapter 13) |

---

## Prerequisites

- [Chapter 05 — Support Vector Machines](../chapter-05-support-vector-machines/README.md) (pipelines, scaling)
- Basic linear algebra: vectors, matrices, dot product
- Familiarity with Matplotlib for scatter plots
- Scikit-Learn with NumPy and Matplotlib

---

## Key Takeaways

- PCA finds uncorrelated directions ordered by variance — the first components capture the most structure
- Always standardize features before PCA when variables have different scales
- Explained variance ratio tells you how much information each component retains
- Reconstruction error from PCA is a simple but effective anomaly detection signal
- PCA is linear — it cannot unwrap nonlinear manifolds (use kernel PCA or t-SNE for that)

---

## Self-Assessment

1. What does the first principal component represent geometrically?
2. Why should you standardize features before applying PCA?
3. How do you decide how many components to keep?
4. How can PCA reduce noise in a dataset? What information might you lose?
5. Why is PCA alone insufficient for strong data anonymization?
6. How does reconstruction error indicate an anomaly?
7. When would t-SNE be preferred over PCA for visualization?

---

**Previous:** [Chapter 05 — Support Vector Machines](../chapter-05-support-vector-machines/README.md)  
**Next:** [Chapter 07 — Operationalizing ML Models](../chapter-07-operationalizing-models/README.md)
