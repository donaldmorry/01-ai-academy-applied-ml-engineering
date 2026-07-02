# Chapter 05: Support Vector Machines

> **Source:** *Applied Machine Learning and AI for Engineers*, Chapter 5  
> **Part:** I — Machine Learning with Scikit-Learn  
> **Estimated time:** 8–10 hours  
> **Prerequisites:** [Chapter 03 — Classification Models](../chapter-03-classification-models/README.md); [Chapter 02 Section 2.5](../chapter-02-regression-models/section-05-support-vector-regression-svr.md) (SVR preview)

---

## Chapter Overview

Support Vector Machines (SVMs) are among the most principled and powerful classical ML algorithms. Instead of fitting a probability distribution or growing a tree, SVMs find the **optimal decision boundary** — the hyperplane that maximizes the margin between classes. With [kernel functions](../../GLOSSARY.md#kernel-trick), they learn complex nonlinear boundaries without explicitly transforming the feature space.

This chapter covers SVM theory at the engineer's level: margins, support vectors, soft margins for noisy data, and the kernel trick (linear, polynomial, RBF). You will learn why **feature normalization is mandatory** for SVMs, how to tune hyperparameters with grid search, and how to build end-to-end **Scikit-Learn pipelines** that prevent data leakage.

The capstone application is **facial recognition** on the Olivetti Faces dataset — HOG features, tuned RBF SVM, and confusion analysis, following Prosise Ch. 5.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain the maximum-margin principle and the role of support vectors
2. Apply kernel functions (linear, polynomial, RBF) via the kernel trick
3. Tune SVM hyperparameters ($C$, $\gamma$, kernel) with `GridSearchCV`
4. Normalize features correctly inside pipelines before training SVMs
5. Train binary and multiclass classifiers with `SVC`
6. Connect SVR regression to the SVM framework ([Chapter 02](../chapter-02-regression-models/section-05-support-vector-regression-svr.md))
7. Build a face recognition system with HOG features and Olivetti faces
8. Decide when SVMs belong in production and how ONNX fits in

---

## Sections

| # | Section | File | Topics |
|---|--------|------|--------|
| 5.1 | SVM Intuition | [01-svm-intuition.md](./section-01-svm-intuition-maximum-margin-and-support-vectors.md) | Hyperplanes, margins, support vectors; hard vs soft margin |
| 5.2 | Kernels & the Kernel Trick | [02-kernels-and-kernel-trick.md](./section-02-kernels-and-the-kernel-trick.md) | Linear, RBF, polynomial kernels; implicit feature mapping |
| 5.3 | Hyperparameter Tuning | [03-hyperparameter-tuning.md](./section-03-hyperparameter-tuning-c-gamma-and-gridsearchcv.md) | $C$, $\gamma$, `GridSearchCV`, coarse-to-fine search |
| 5.4 | Normalization & Pipelining | [04-normalization-and-pipelining.md](./section-04-normalization-and-pipelining-for-svms.md) | `StandardScaler`, `make_pipeline`, leakage prevention |
| 5.5 | SVM Classification | [05-svm-classification.md](./section-05-svm-classification-svc-for-binary-and-multiclass.md) | `SVC` binary/multiclass, OvR/OvO, `LinearSVC`, MNIST |
| 5.6 | SVR Regression | [06-svr-regression.md](./section-06-svr-regression-from-tube-to-production.md) | ε-tube, link to [Section 2.5](../chapter-02-regression-models/section-05-support-vector-regression-svr.md) |
| 5.7 | Facial Recognition | [07-facial-recognition-svm.md](./section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md) | HOG features, Olivetti faces, PCA preview |
| 5.8 | Production Notes | [08-svm-production-notes.md](./section-08-svm-production-notes-when-to-deploy-and-onnx-bridge.md) | When SVM wins/loses, ONNX bridge, deployment checklist |

---

## Lab

**[Lab 05: Facial Recognition with SVMs](./section-lab-05-facial-recognition-with-support-vector-machines.md)**

Build a face classification system on the Olivetti Faces dataset (Prosise Ch. 5):

1. Load and explore Olivetti faces (40 subjects, 400 images)
2. Extract HOG features and compare to raw pixels
3. Normalize with `StandardScaler` inside a `Pipeline`
4. Train an RBF-kernel SVM with `GridSearchCV` over $C$ and $\gamma$
5. Evaluate with classification report and confusion matrix
6. Analyze misclassified faces and confused person pairs
7. Compare linear vs RBF kernel performance and training time
8. Persist pipeline with `joblib` + metadata; optional ONNX export

*Deliverable:* Saved pipeline, grid-search heatmap, confusion analysis, and written comparison of HOG vs pixels.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Maximum-margin classification | Statistical learning theory, VC bounds (Course 2, Ch 19) |
| [Kernel trick](../../GLOSSARY.md#kernel-trick) | Kernel PCA, Gaussian processes (Course 2) |
| Facial recognition with SVMs | CNN-based face recognition ([Chapter 10](../chapter-10-convolutional-neural-networks/README.md)) |
| PCA + SVM pipeline | Full PCA theory ([Chapter 06](../chapter-06-principal-component-analysis/README.md)) |
| Hyperparameter search | Bayesian optimization, AutoML (Course 2) |
| SVR regression | [Chapter 02, Section 2.5](../chapter-02-regression-models/section-05-support-vector-regression-svr.md) |

---

## Prerequisites

- [Chapter 03 — Classification Models](../chapter-03-classification-models/README.md) (metrics, cross-validation)
- [Chapter 02 Section 2.5](../chapter-02-regression-models/section-05-support-vector-regression-svr.md) (SVR ε-tube — optional preview)
- Comfort with NumPy arrays and basic linear algebra (dot product, vectors)
- [Chapter 04](../chapter-04-text-classification/README.md) pipeline patterns helpful but not required
- Scikit-Learn, scikit-image (HOG), standard scientific Python stack

---

## Key Takeaways

- [SVMs](../../GLOSSARY.md#svm) find the widest margin between classes — support vectors alone define the boundary
- Always scale features before SVM training; unscaled data produces poor margins
- The RBF kernel is the default for nonlinear problems but requires careful $\gamma$ tuning
- Pipelines ensure preprocessing is fit only on training data — critical for valid evaluation
- SVMs excel on small-to-medium datasets with high-dimensional features but scale poorly to millions of samples

---

## Self-Assessment

1. What is a support vector, and why does it matter for prediction?
2. What happens when $C$ is very large? When $C$ is very small?
3. Why does the RBF kernel have a $\gamma$ parameter, and what does increasing $\gamma$ do?
4. Why would an SVM fail on unscaled features even if the data is linearly separable?
5. How does a Pipeline prevent data leakage during cross-validation?
6. When would you choose a linear kernel over RBF?
7. Why might HOG features outperform raw pixels for face recognition?

---

**Glossary:** [../../GLOSSARY.md](../../GLOSSARY.md) — [svm](../../GLOSSARY.md#svm), [kernel-trick](../../GLOSSARY.md#kernel-trick)  
**Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

**Previous:** [Chapter 04 — Text Classification](../chapter-04-text-classification/README.md)  
**Next:** [Chapter 06 — Principal Component Analysis](../chapter-06-principal-component-analysis/README.md)
