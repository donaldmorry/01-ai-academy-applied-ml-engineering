# Section 1.6: The ML Workflow & Data Hygiene

> **Source inheritance:** Prosise, Ch. 1 - Summary + data preparation themes  
> **Enhanced with:** Production-grade practices, common pitfalls, checklist  
> **Glossary:** [../../GLOSSARY.md](../../GLOSSARY.md) | **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Complete ML Workflow (Revisited)

This section consolidates Chapter 01 into an actionable workflow you'll use in every subsequent chapter and course.

```
┌──────────────────────────────────────────────────────────────────┐
│                    THE ML WORKFLOW                                │
├──────────────────────────────────────────────────────────────────┤
│  1. PROBLEM DEFINITION                                           │
│     └─ What are we predicting? What metric defines success?      │
│  2. DATA COLLECTION                                              │
│     └─ Is data representative? Sufficient volume? Labeled?       │
│  3. EXPLORATORY DATA ANALYSIS (EDA)                              │
│     └─ Distributions, correlations, missing values, outliers     │
│  4. DATA PREPARATION                                             │
│     └─ Clean, encode, scale, split (train/val/test)             │
│  5. MODEL SELECTION & TRAINING                                   │
│     └─ Choose algorithm, tune hyperparameters                    │
│  6. EVALUATION                                                   │
│     └─ Test set metrics, error analysis, fairness checks         │
│  7. DEPLOYMENT & MONITORING                                      │
│     └─ Production serving, drift detection, retraining           │
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Hygiene: The Silent Killer

> "Garbage in, garbage out" - the oldest rule in computing, and the #1 cause of ML project failure.

### Missing Values

| Strategy | When to Use |
|----------|------------|
| **Drop rows** | Few missing values (<5%), missing is random |
| **Mean/median imputation** | Numerical features, missing at random |
| **Mode imputation** | Categorical features |
| **Model-based imputation** | Complex patterns in missingness |
| **Flag + impute** | Missingness itself is informative |

```python
# Check missing values
print(df.isnull().sum())

# Simple imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_train)
```

### Outliers

Outliers are data points that don't conform to the general pattern.

**Detection methods:**
- Z-score: |z| > 3
- IQR: below Q1 - 1.5×IQR or above Q3 + 1.5×IQR
- Domain knowledge: negative age, $1M taxi fare for 2-mile trip

**Handling:**
- Remove (if clearly errors)
- Cap/winsorize (replace with threshold values)
- Transform (log scale reduces outlier impact)
- Keep (if legitimate - fraud detection NEEDS outliers)

### Feature Scaling

| Scaler | Formula | Use When |
|--------|---------|----------|
| `StandardScaler` | $(x - \mu) / \sigma$ | Default for most algorithms |
| `MinMaxScaler` | $(x - x_{\min}) / (x_{\max} - x_{\min})$ | Need bounded [0,1] range |
| `RobustScaler` | $(x - \text{median}) / \text{IQR}$ | Data has outliers |

> **Readable form (StandardScaler):** scaled value = (original minus mean) divided by standard deviation

**Always fit scaler on training data only, then transform test data.**

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform train
X_test_scaled = scaler.transform(X_test)           # transform test ONLY
```

---

## Train / Validation / Test Split

```
Full Dataset (100%)
├── Training Set (60-70%)     → Fit model parameters
├── Validation Set (15-20%)   → Tune hyperparameters
└── Test Set (15-20%)         → Final unbiased evaluation
```

**Why three sets?**
- Training on test data → overfitting to test set → inflated metrics
- Tuning on test data → same problem
- Validation set is the "practice exam"; test set is the "final exam"

```python
# Two-step split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
# Result: 60% train, 20% val, 20% test
```

### Stratified Splitting

For classification with imbalanced classes, preserve class proportions:

```python
train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

If your full dataset has $n$ samples, a typical 60/20/20 split yields:

$$
n_{\text{train}} \approx 0.6n, \quad n_{\text{val}} \approx 0.2n, \quad n_{\text{test}} \approx 0.2n
$$
> **Readable form:** train size ≈ 60% of n; validation and test each ≈ 20% of n

---

## Evaluation Metrics Preview

### Classification (Chapter 03 goes deeper)

| Metric | Formula Intuition | When to Use |
|--------|------------------|-------------|
| **[Accuracy](../../GLOSSARY.md#accuracy)** | Correct / Total | Balanced classes |
| **Precision** | TP / (TP + FP) | Cost of false positives is high |
| **Recall** | TP / (TP + FN) | Cost of false negatives is high |
| **F1** | Harmonic mean of P & R | Balance precision and recall |

[Accuracy](../../GLOSSARY.md#accuracy) as an equation:

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$
> **Readable form:** accuracy = (true positives + true negatives) divided by all predictions

### Regression (Chapter 02 goes deeper)

| Metric | Intuition |
|--------|-----------|
| **[MAE](../../GLOSSARY.md#mae)** | Average absolute error (robust to outliers) |
| **[MSE](../../GLOSSARY.md#mse)** | Average squared error (penalizes large errors) |
| **[RMSE](../../GLOSSARY.md#rmse)** | Square root of [MSE](../../GLOSSARY.md#mse) - same units as target |
| **[R²](../../GLOSSARY.md#r-squared)** | Fraction of variance explained (0-1, higher is better) |

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$
> **Readable form:** RMSE = square root of (average squared prediction error)

---

## Common Pitfalls Checklist

- [ ] **Data leakage:** Test information sneaking into training (e.g., scaling before split)
- [ ] **Target leakage:** Features that contain future information (predicting churn using "account_closed_date")
- [ ] **Sampling bias:** Training data doesn't represent production data
- [ ] **Class imbalance ignored:** 99% accuracy on 99% negative class is useless
- [ ] **Overfitting to validation:** Trying too many hyperparameter combinations
- [ ] **Not setting random seeds:** Results aren't reproducible
- [ ] **Ignoring domain experts:** ML finds correlations; humans know causation

### Expanded Pitfall Guide

| Pitfall | Real Example | Prevention |
|---------|--------------|------------|
| **Data leakage** | Fitting `StandardScaler` on full dataset before split | Use `make_pipeline`; split first |
| **Target leakage** | Predicting churn using `account_closed_date` | Audit [features](../../GLOSSARY.md#feature) for future information |
| **Sampling bias** | Training on urban users, deploying nationally | Match production distribution |
| **Class imbalance ignored** | 99% [accuracy](../../GLOSSARY.md#accuracy) on fraud detection with 0.1% fraud | Use precision/recall, not [accuracy](../../GLOSSARY.md#accuracy) alone |
| **No random seed** | Results change every run | `random_state=42` everywhere |
| **Tuning on test set** | Trying 50 models, picking best test score | Use validation set; touch test once |

---

## Prosise Ch. 1 Summary: The Practitioner Checklist

Prosise closes Chapter 1 by emphasizing that algorithms are the easy part - **data preparation** separates successful projects from failed ones. His checklist maps directly to this section:

| Prosise Step | This Section | Key Tool |
|--------------|-------------|----------|
| Understand the problem | Problem definition | Task type, metric |
| Prepare the data | Data hygiene | Imputation, scaling, splits |
| Choose an algorithm | Sections 1.4-1.5 | [k-means](../../GLOSSARY.md#k-means), [k-NN](../../GLOSSARY.md#k-nearest-neighbors) |
| Train and evaluate | Train/val/test | `train_test_split`, metrics |
| Iterate | Chapter 02+ | [Cross-validation](../../GLOSSARY.md#cross-validation), new algorithms |

The chapter's two running examples - customer [clustering](../../GLOSSARY.md#clustering) and Iris [classification](../../GLOSSARY.md#classification) - are intentionally **small** so you practice the full workflow without cloud GPUs or big-data infrastructure.

---

## Chapter 01 Summary

You now understand:

1. **What ML is** - learning rules from data instead of programming them ([Section 1.1](./section-01-what-is-machine-learning.md))
2. **Where ML fits in AI** - a powerful subset, not the whole field ([Section 1.2](./section-02-machine-learning-vs-ai-vs-deep-learning.md))
3. **[Supervised vs unsupervised](../../GLOSSARY.md#supervised-learning)** - [labels](../../GLOSSARY.md#label) vs no labels ([Section 1.3](./section-03-supervised-vs-unsupervised-learning.md))
4. **[k-Means](../../GLOSSARY.md#k-means)** - discover groups in unlabeled data ([Section 1.4](./section-04-unsupervised-learning-k-means-clustering.md))
5. **[k-NN](../../GLOSSARY.md#k-nearest-neighbors)** - classify by neighbor voting ([Section 1.5](./section-05-supervised-learning-k-nearest-neighbors.md))
6. **The workflow** - from problem definition to deployment (this section)

### What's Next

| Chapter | You'll Learn |
|--------|-------------|
| **02: Regression** | Predict continuous values - taxi fares, house prices |
| **03: Classification** | Logistic regression, Titanic survival, digit recognition |
| **04: Text** | Sentiment analysis, spam filtering, recommendations |
| **05: SVMs** | Maximum-margin classifiers, kernels, face recognition |
| **06: PCA** | Dimensionality reduction, anomaly detection |
| **07: Deployment** | ONNX, Docker, ML.NET, production systems |
| **08-14: Deep Learning** | Neural networks, CNNs, NLP, Azure |

---

## Self-Assessment Answers

1. **Unsupervised vs supervised?** → [Unsupervised](../../GLOSSARY.md#unsupervised-learning) when you don't have [labels](../../GLOSSARY.md#label) and want to discover structure (segmentation, exploration). [Supervised](../../GLOSSARY.md#supervised-learning) when you have labels and want to predict.
2. **k-NN with $k=1$?** → Memorizes training data ([overfitting](../../GLOSSARY.md#overfitting)). $k=n$? → Always predicts majority class ([underfitting](../../GLOSSARY.md#underfitting)).
3. **Scaling for [k-means](../../GLOSSARY.md#k-means) but not trees?** → [k-means](../../GLOSSARY.md#k-means) uses Euclidean distance (scale matters). Trees split on individual [features](../../GLOSSARY.md#feature) (scale irrelevant).
4. **ML ⊂ AI, DL ⊂ ML?** → Yes. [AI](../../GLOSSARY.md#artificial-intelligence) includes search, logic, planning beyond ML. [Deep learning](../../GLOSSARY.md#deep-learning) is ML with deep neural networks.

---

## Further Reading

- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)
- [Lab 01](./section-lab-01-customer-segmentation-and-flower-classification.md) - hands-on practice

---

**Next:** [Lab 01 - Hands-On Practice](./section-lab-01-customer-segmentation-and-flower-classification.md) → then [Chapter 02: Regression Models](../chapter-02-regression-models/README.md)



