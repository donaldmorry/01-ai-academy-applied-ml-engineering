# Chapter 02: Regression Models

> **Source:** *Applied Machine Learning and AI for Engineers*, Chapter 2  
> **Part:** I — Machine Learning with Scikit-Learn  
> **Estimated time:** 10–12 hours  
> **Prerequisites:** [Chapter 01 — Machine Learning Fundamentals](../chapter-01-machine-learning/README.md); Python, Pandas, basic statistics

---

## Chapter Overview

Regression is the supervised learning task of predicting a **continuous numeric target** — taxi fares, house prices, sensor readings, or any quantity you can measure on a scale. This chapter moves beyond the clustering and k-NN foundations of Chapter 01 into the algorithms that power most real-world forecasting systems.

You will learn how linear models establish a baseline, how tree-based ensembles capture nonlinear relationships, and how support vector regression handles high-dimensional feature spaces. Along the way you will master the accuracy measures that tell you whether a model is actually useful — not just whether it runs without errors.

The capstone application is **taxi fare prediction**: a classic regression problem with mixed feature types, outliers, and business-relevant error metrics.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Formulate a regression problem and choose appropriate target and feature variables
2. Train and interpret linear regression models with Scikit-Learn
3. Build decision trees, random forests, and gradient boosting regressors
4. Apply support vector regression (SVR) with kernel functions
5. Evaluate models using MSE, RMSE, MAE, R², and explained variance
6. Diagnose overfitting and underfitting with train/test splits and cross-validation
7. Compare model performance and select the best approach for a business constraint
8. Complete an end-to-end taxi fare prediction pipeline

---

## Sections

| # | Section | File |
|---|--------|------|
| 2.1 | Introduction to Regression | [01-introduction-to-regression.md](./section-01-introduction-to-regression.md) |
| 2.2 | Linear Regression | [02-linear-regression.md](./section-02-linear-regression.md) |
| 2.3 | Decision Trees for Regression | [03-decision-trees-regression.md](./section-03-decision-trees-for-regression.md) |
| 2.4 | Ensemble Methods | [04-ensemble-methods.md](./section-04-ensemble-methods-random-forests-and-gradient-boosting.md) |
| 2.5 | Support Vector Regression | [05-support-vector-regression.md](./section-05-support-vector-regression-svr.md) |
| 2.6 | Regression Accuracy Measures | [06-regression-accuracy-measures.md](./section-06-regression-accuracy-measures.md) |
| 2.7 | Model Comparison & Selection | [07-model-comparison-selection.md](./section-07-model-comparison-and-cross-validation.md) |
| 2.8 | Taxi Fare Prediction | [08-taxi-fare-prediction.md](./section-08-taxi-fare-prediction-end-to-end-project.md) |

**Lab:** [lab-02.md](./section-lab-02-taxi-fare-prediction.md) | **Glossary:** [../../GLOSSARY.md](../../GLOSSARY.md)

---

## Lab

**Lab 02: Taxi Fare Prediction**

Build a complete regression pipeline on a real taxi trip dataset. You will:

- Explore trip distance, duration, pickup/dropoff zones, and passenger count
- Engineer features (time of day, day of week, haversine distance)
- Train linear regression, random forest, and gradient boosting models
- Compare models with RMSE and MAE on a held-out test set
- Analyze residual plots to identify systematic errors
- Document which model you would deploy and why

*Deliverable:* Jupyter notebook with trained models, metric comparison table, and a short recommendation memo.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Linear regression & least squares | Statistical learning theory, regularization (Course 2, Ch 19) |
| Gradient boosting | XGBoost, LightGBM, hyperparameter search at scale (Course 2) |
| Bias-variance tradeoff | PAC learning, VC dimension (Course 2, Ch 19) |
| Feature engineering for tabular data | Deep tabular models, embeddings (Course 3) |
| Taxi fare as regression baseline | Neural network regression revisit (Chapter 09) |

---

## Prerequisites

- Completion of [Chapter 01](../chapter-01-machine-learning/README.md) or equivalent familiarity with Scikit-Learn workflows
- Comfortable with Pandas DataFrames, train/test splits, and Matplotlib
- Basic understanding of mean, variance, and correlation
- Jupyter notebook environment with Scikit-Learn, Pandas, NumPy, Matplotlib installed

---

## Key Takeaways

- Always establish a simple baseline (mean predictor, linear model) before trying complex algorithms
- Tree ensembles often outperform linear models on tabular data with nonlinear interactions
- The right error metric depends on business cost — RMSE penalizes large errors more than MAE
- Cross-validation gives a more reliable performance estimate than a single train/test split
- Residual analysis reveals whether your model is missing systematic patterns

---

## Self-Assessment

1. When is linear regression the right choice despite its simplicity?
2. How does a random forest reduce variance compared to a single decision tree?
3. Why might gradient boosting outperform random forests on some datasets but take longer to tune?
4. What does an R² of 0.85 tell you — and what does it *not* tell you?
5. When would you prefer MAE over RMSE as your primary metric?
6. How do you detect overfitting in a regression model without looking at the code?
7. What features would you engineer for taxi fare prediction beyond raw GPS coordinates?

---

**Previous:** [Chapter 01 — Machine Learning Fundamentals](../chapter-01-machine-learning/README.md)  
**Next:** [Chapter 03 — Classification Models](../chapter-03-classification-models/README.md)
