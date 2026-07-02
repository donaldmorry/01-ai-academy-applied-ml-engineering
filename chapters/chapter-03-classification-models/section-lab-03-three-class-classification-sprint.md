# Lab 03: Three-Class Classification Sprint

> **Prerequisites:** Sections [3.1](./section-01-classification-fundamentals.md)-[3.8](./section-08-case-study-mnist-digit-recognition.md)  
> **Estimated time:** 4-6 hours  
> **Glossary:** [../../GLOSSARY.md](../../GLOSSARY.md) | **Math:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Objectives

By the end of this lab you will:

1. Complete three end-to-end [classification](../../GLOSSARY.md#classification) projects (Titanic, fraud, MNIST)
2. Beat appropriate baselines on each dataset
3. Choose metrics matched to business context ([Section 3.3](./section-03-classification-accuracy-measures.md))
4. Compare at least two algorithms per problem
5. Write a short comparison memo explaining **which algorithm suits which problem type**

> **Humorous briefing:** You're running a classification triathlon - tabular history, financial crime, and robot handwriting. Pace yourself; fraud is the soul-crushing leg where accuracy goes to die.

---

## Setup

```python
# Recommended environment
# pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

**Data sources:**

| Dataset | URL |
|---------|-----|
| Titanic | [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data) |
| Credit card fraud | [Kaggle creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| MNIST | `sklearn.datasets.fetch_openml('mnist_784')` |

Place CSVs in `data/titanic/` and `data/creditcard.csv`, or use the synthetic fallbacks in each part.

---

## Part A: Titanic Survival (90 min)

*Follow [Section 3.6](./section-06-case-study-titanic-survival.md) - full Prosise pipeline.*

### A1. Load & EDA (20 min)

```python
# df = pd.read_csv('data/titanic/train.csv')
# Or synthetic fallback if Kaggle file unavailable:
n = 891
df = pd.DataFrame({
    'PassengerId': range(1, n+1),
    'Survived': np.random.binomial(1, 0.38, n),
    'Pclass': np.random.choice([1,2,3], n, p=[0.24, 0.21, 0.55]),
    'Sex': np.random.choice(['male','female'], n, p=[0.65, 0.35]),
    'Age': np.random.normal(30, 14, n).clip(0.5, 80),
    'SibSp': np.random.poisson(0.5, n),
    'Parch': np.random.poisson(0.4, n),
    'Fare': np.random.exponential(30, n),
    'Embarked': np.random.choice(['S','C','Q'], n, p=[0.72, 0.19, 0.09]),
    'Name': [f'Passenger_{i}' for i in range(n)],
    'Ticket': [f'T{i}' for i in range(n)],
    'Cabin': np.nan,
})
```

**Tasks:**
- Plot survival rate by `Sex` and `Pclass`
- Report class balance (% survived)
- List missing values per column

### A2. Feature engineering (15 min)

Implement from [Section 3.4](./section-04-categorical-data-and-feature-encoding.md):
- `Title` from `Name`
- `FamilySize`, `IsAlone`, `FarePerPerson`

### A3. Pipeline + models (35 min)

Build `ColumnTransformer` + compare:

| Model | Required |
|-------|----------|
| `DummyClassifier` (majority) | ✅ |
| `LogisticRegression` | ✅ |
| `RandomForestClassifier` | ✅ |

```python
# Template - fill in your feature lists
numeric_features = ['Age', 'Fare', 'FamilySize', 'FarePerPerson', 'SibSp', 'Parch']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone']

# ... build preprocessor, split stratified, fit models
```

### A4. Metrics (20 min)

**Required deliverables:**
- Confusion matrix for best model
- `classification_report` with **precision/recall for Survived (class 1)**
- 5-fold CV accuracy for logistic vs RF
- 2-3 sentences: which features matter most?

**Success criteria:** Test accuracy **> 75%** (real data) or **> 70%** (synthetic); beat baseline by **≥ 10 points**.

---

## Part B: Credit Card Fraud (90 min)

*Follow [Section 3.7](./section-07-case-study-credit-card-fraud-detection.md) - 492 / 284,807 fraud ratio.*

### B1. Load & sanity check (15 min)

```python
# df = pd.read_csv('data/creditcard.csv')
# Synthetic fallback (structure only - not for publication):
n_legit, n_fraud = 10000, 17  # ~0.17% fraud
X_synth = np.random.randn(n_legit + n_fraud, 30)
y_synth = np.array([0]*n_legit + [1]*n_fraud)
```

**Tasks:**
- Print fraud count and rate: should be **~492 / 284807 ≈ 0.173%** on real data
- Compute accuracy of always predicting legitimate - verify **~99.83%**

### B2. Stratified split + models (40 min)

| Model | Settings |
|-------|----------|
| `LogisticRegression` | `class_weight='balanced'` |
| `RandomForestClassifier` | `class_weight='balanced'`, `n_estimators=200` |

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

fraud_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
])
```

### B3. Metrics that matter (35 min)

**Do NOT lead with accuracy.** Required:

| Deliverable | Tool |
|-------------|------|
| PR curve | `PrecisionRecallDisplay` |
| ROC-AUC | `roc_auc_score` |
| Average Precision | `average_precision_score` |
| Recall at FPR ≤ 0.1% | Custom function from [Section 3.7](./section-07-case-study-credit-card-fraud-detection.md) |
| Written answer | "Why is 99.9% accuracy worthless?" (3 sentences) |

```python
def recall_at_fpr(y_true, y_scores, max_fpr=0.001):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.where(fpr <= max_fpr)[0]
    if len(idx) == 0:
        return 0.0, 1.0
    best = idx[-1]
    return tpr[best], thresholds[best]
```

**Success criteria:** ROC-AUC **> 0.90** on real data; fraud recall **> 0.50** at FPR ≤ 0.1% (tune threshold).

---

## Part C: MNIST Digit Recognition (90 min)

*Follow [Section 3.8](./section-08-case-study-mnist-digit-recognition.md).*

### C1. Load data (10 min)

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data, mnist.target.astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, train_size=60000, stratify=y, random_state=RANDOM_STATE
)
X_train_s = X_train / 255.0
X_test_s = X_test / 255.0
```

### C2. Two algorithms (50 min)

| Algorithm | Required |
|-----------|----------|
| `KNeighborsClassifier` (k=3) | ✅ |
| `SVC` (RBF kernel) **or** `LogisticRegression` multi_class='ovr' | ✅ |

```python
knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn.fit(X_train_s, y_train)
print(f"k-NN: {accuracy_score(y_test, knn.predict(X_test_s)):.4f}")

# SVM - subsample for tuning if needed
svm = SVC(kernel='rbf', C=5, gamma='scale', random_state=RANDOM_STATE)
svm.fit(X_train_s, y_train)
print(f"SVM: {accuracy_score(y_test, svm.predict(X_test_s)):.4f}")
```

### C3. Multi-class evaluation (30 min)

**Required:**
- 10×10 confusion matrix heatmap
- `classification_report` (macro F1)
- Plot 8 misclassified digits with true vs predicted labels
- Timing: train time and predict time for k-NN vs SVM

**Success criteria:** Test accuracy **> 95%** with at least one algorithm (k-NN usually qualifies; SVM often **> 97%**).

---

## Part D: Comparison Memo (30 min)

Write **half a page** (200-300 words) for a technical lead:

### Template

```markdown
## Classification Sprint Summary

### Titanic (tabular, mild imbalance)
- Best model: ___
- Key metrics: accuracy ___, survivor recall ___
- Why it worked: ___

### Fraud (extreme imbalance)
- Best model: ___
- Key metrics: PR-AUC ___, recall @ 0.1% FPR ___
- Why accuracy misled: ___

### MNIST (multi-class, high-dimensional)
- Best model: ___
- Key metrics: accuracy ___, macro F1 ___
- Why: ___

### Algorithm selection guide
| Problem type | Recommended starting algorithm | Primary metric |
|--------------|-------------------------------|----------------|
| Interpretable tabular | | |
| Imbalanced rare event | | |
| Multi-class vision (classical) | | |
```

---

## Part E: Self-Assessment (15 min)

Answer [Chapter 03 README](./README.md) self-assessment questions:

1. Why does logistic regression output probabilities?
2. 80% fraud recall, 500 false alarms/day - good or bad? How do you decide?
3. One-hot vs label encoding?
4. How does confusion matrix reveal errors accuracy hides?
5. Random forest on Titanic vs linearly separable data?
6. Binary → multi-class changes?
7. When choose recall over precision?

---

## Deliverables Checklist

- [ ] Jupyter notebook with three clearly labeled sections (A, B, C)
- [ ] Titanic: confusion matrix + CV table (logistic vs RF)
- [ ] Fraud: PR curve + recall@FPR + "why accuracy lies" paragraph
- [ ] MNIST: confusion matrix + >95% accuracy + misclassification plot
- [ ] Comparison memo (Part D)
- [ ] All metrics tables in one summary cell at the end

### Summary table template

```python
results = pd.DataFrame([
    {'dataset': 'Titanic', 'model': '...', 'metric': 'accuracy', 'value': 0.0},
    {'dataset': 'Fraud', 'model': '...', 'metric': 'PR-AUC', 'value': 0.0},
    {'dataset': 'Fraud', 'model': '...', 'metric': 'recall@0.1%FPR', 'value': 0.0},
    {'dataset': 'MNIST', 'model': '...', 'metric': 'accuracy', 'value': 0.0},
])
print(results.to_markdown(index=False))
```

---

## Grading Rubric (Self-Check)

| Criterion | Points |
|-----------|--------|
| Pipelines prevent leakage (fit on train only) | /20 |
| Correct metrics per problem | /25 |
| Beats baselines | /20 |
| Code runs + documented | /15 |
| Comparison memo quality | /20 |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| k-NN too slow | Subsample train to 10k; or use `algorithm='ball_tree'` |
| SVM hours to train | Subsample 10k for search; `LinearSVC` for speed |
| Titanic accuracy low | Check Title engineering; stratified split |
| Fraud recall = 0 | Add `class_weight='balanced'`; lower threshold |
| MNIST download fails | `fetch_openml(..., data_home='./data')` |

---

## References

- [Kaggle: Titanic](https://www.kaggle.com/competitions/titanic)
- [Kaggle: Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [OpenML MNIST](https://www.openml.org/d/554)
- [Scikit-Learn Classification Metrics](https://scikit-learn.org/stable/chapters/model_evaluation.html#classification-metrics)
- Prosise, Ch. 3 - all three case studies
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 3.8](./section-08-case-study-mnist-digit-recognition.md) | **Next Chapter:** [04 - Text Classification](../chapter-04-text-classification/README.md)