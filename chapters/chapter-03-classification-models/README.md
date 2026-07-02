# Chapter 03: Classification Models

> **Source:** *Applied Machine Learning and AI for Engineers*, Chapter 3  
> **Part:** I — Machine Learning with Scikit-Learn  
> **Estimated time:** 10–12 hours  
> **Prerequisites:** [Chapter 02 — Regression Models](../chapter-02-regression-models/README.md); understanding of train/test evaluation

---

## Chapter Overview

Classification predicts **discrete categories** — survived or perished, fraudulent or legitimate, digit 0 through 9. This chapter covers the algorithms and evaluation practices that define modern tabular ML: logistic regression as the interpretable baseline, decision trees and ensembles for complex boundaries, and the accuracy measures that prevent you from deploying a model that looks good on paper but fails in production.

You will work through three iconic datasets that span the classification spectrum:

- **Titanic** — structured tabular data with missing values and categorical features
- **Credit card fraud** — extreme class imbalance where accuracy is misleading
- **MNIST digits** — high-dimensional image-like inputs with multi-class output

By the end, you will know how to encode categorical variables, choose metrics beyond accuracy, and build pipelines that generalize to unseen data.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Distinguish binary, multi-class, and multi-label classification problems
2. Train logistic regression and interpret coefficients and decision boundaries
3. Encode categorical features with one-hot encoding and ordinal encoding
4. Build decision tree, random forest, and gradient boosting classifiers
5. Evaluate classifiers with accuracy, precision, recall, F1, and confusion matrices
6. Handle class imbalance with stratified sampling and appropriate metrics
7. Apply k-NN and SVM classifiers to digit recognition
8. Complete end-to-end classification projects on Titanic, fraud, and MNIST data

---

## Sections

| # | Section | Topics |
|---|--------|--------|
| 3.1 | [Classification Fundamentals](./section-01-classification-fundamentals.md) | Binary vs multi-class; decision boundaries; baseline classifiers |
| 3.2 | [Logistic Regression](./section-02-logistic-regression.md) | Sigmoid function; log-loss; `LogisticRegression`; interpretability |
| 3.3 | [Classification Accuracy Measures](./section-03-classification-accuracy-measures.md) | Confusion matrix; precision, recall, F1; ROC-AUC; when accuracy lies |
| 3.4 | [Categorical Data & Feature Encoding](./section-04-categorical-data-and-feature-encoding.md) | One-hot, label, ordinal encoding; `ColumnTransformer`; missing values |
| 3.5 | [Tree-Based Classifiers](./section-05-tree-based-classifiers.md) | `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier` |
| 3.6 | [Case Study: Titanic Survival](./section-06-case-study-titanic-survival.md) | EDA, feature engineering, model comparison, leaderboard-style evaluation |
| 3.7 | [Case Study: Credit Card Fraud Detection](./section-07-case-study-credit-card-fraud-detection.md) | Class imbalance, precision-recall tradeoff, anomaly framing |
| 3.8 | [Case Study: Digit Recognition (MNIST)](./section-08-case-study-mnist-digit-recognition.md) | Multi-class classification; k-NN and SVM on flattened pixels |

---

## Lab

**[Lab 03: Three-Class Classification Sprint](./section-lab-03-three-class-classification-sprint.md)**

Work through all three case studies in a single structured lab:

1. **Titanic** — Predict survival using passenger class, age, sex, and embarkation port. Compare logistic regression and random forest. Report precision and recall for the minority class (survived).
2. **Fraud Detection** — Train on a highly imbalanced credit card dataset. Optimize for recall at a fixed false-positive rate. Explain why 99.9% accuracy can still be worthless.
3. **Digit Recognition** — Classify handwritten digits 0–9. Achieve >95% test accuracy with at least two different algorithms.

*Deliverable:* Three notebook sections with confusion matrices, metric tables, and a written comparison of which algorithm suits each problem type.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Logistic regression & log-loss | Maximum likelihood, convex optimization (Course 2, Ch 19) |
| Class imbalance & cost-sensitive learning | Decision theory, utility functions (Course 2) |
| MNIST as classification | CNN architectures that replace flattened pixels (Chapter 10) |
| Fraud detection patterns | Anomaly detection, autoencoders (Course 3, generative models) |
| Categorical encoding | Entity embeddings for tabular deep learning (Course 3) |

---

## Prerequisites

- [Chapter 02 — Regression Models](../chapter-02-regression-models/README.md) (train/test splits, cross-validation, Scikit-Learn API)
- Familiarity with Pandas, NumPy, and Matplotlib
- Basic probability concepts (conditional probability helps with logistic regression)
- Jupyter environment with Scikit-Learn and imbalanced-learn (optional, for SMOTE)

---

## Key Takeaways

- Logistic regression is the interpretable workhorse for binary and multi-class tabular problems
- Accuracy is misleading when classes are imbalanced — always inspect the confusion matrix
- Categorical features must be encoded before most algorithms can use them
- Tree ensembles handle mixed feature types and interactions without manual engineering
- The right metric depends on the cost of false positives vs false negatives

---

## Self-Assessment

1. Why does logistic regression output probabilities rather than hard class labels?
2. A fraud model catches 80% of fraud but generates 500 false alarms per day. Is it good? How do you decide?
3. What is the difference between one-hot encoding and label encoding? When is each appropriate?
4. How does a confusion matrix reveal errors that accuracy hides?
5. Why might a random forest outperform logistic regression on Titanic but not on a linearly separable dataset?
6. What changes when you go from binary to multi-class classification (MNIST)?
7. When would you choose recall over precision as your primary optimization target?

---

**Previous:** [Chapter 02 — Regression Models](../chapter-02-regression-models/README.md)  
**Next:** [Chapter 04 — Text Classification](../chapter-04-text-classification/README.md)
