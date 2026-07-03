# Section 3.7: Case Study - Credit Card Fraud Detection

> **Source:** Prosise, Ch. 3 - "Detecting Credit Card Fraud"  
> **Prerequisites:** [Section 3.3](./section-03-classification-accuracy-measures.md) | [Section 3.5](./section-05-tree-based-classifiers.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [accuracy](../../GLOSSARY.md#accuracy)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Dataset Where Accuracy Is a Lie

Prosise introduces the [ULB MLG credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) - **284,807** transactions from European cardholders, **492** fraudulent.

$$
\text{Fraud rate} = \frac{492}{284{,}807} \approx 0.173\%
$$
> **Readable form:** fraud rate = 492 divided by 284,807 ≈ 0.17% - fewer than 2 frauds per 1,000 transactions

A model that predicts **"legitimate" for everything** achieves:

$$
\text{Accuracy} = \frac{284{,}315}{284{,}807} \approx 99.83\%
$$
> **Readable form:** accuracy ≈ 99.83% while catching **zero** fraud - the most expensive "accurate" model in the room

This section is why [Section 3.3](./section-03-classification-accuracy-measures.md) exists. Welcome to **class imbalance**.

> **Humorous analogy:** Measuring fraud detection with accuracy is like measuring a smoke detector by "percent of quiet minutes." A brick on the ceiling scores 99.99%.

---

## The Data: PCA-Transformed Features

For privacy, raw merchant/category fields are gone. You get:

| Column | Meaning |
|--------|---------|
| `Time` | Seconds since first transaction |
| `Amount` | Transaction amount |
| `V1`-`V28` | PCA components of original features |
| `Class` | **0** = legitimate, **1** = fraud (target) |

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/creditcard.csv')
print(df.shape)  # (284807, 31)
print(df['Class'].value_counts())
print(df['Class'].value_counts(normalize=True))
```

**No categorical encoding needed** - all numeric. But **scaling** `Amount` and `Time` matters for [logistic regression](./section-02-logistic-regression.md) and SVM.

---

## Step 1: Exploratory Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['Class'].value_counts().plot.bar(ax=axes[0], title='Class Counts')
df[df['Class']==1]['Amount'].hist(bins=50, ax=axes[1], color='red', alpha=0.7)
df[df['Class']==0]['Amount'].hist(bins=50, ax=axes[1], color='blue', alpha=0.3)
axes[1].set_title('Amount Distribution')
plt.tight_layout()
plt.show()
```

Fraud isn't uniformly random in time or amount - patterns hide in `V1`-`V28` even when humans can't interpret them.

---

## Step 2: Stratified Split (Non-Negotiable)

```python
from sklearn.model_selection import train_test_split

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train fraud rate: {y_train.mean():.4f}")
print(f"Test fraud rate:  {y_test.mean():.4f}")
```

Without `stratify`, a lucky split might contain 0 fraud cases in test - metrics become meaningless.

---

## Step 3: Baselines

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, roc_auc_score

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)

print("=== Always Legitimate ===")
print(classification_report(y_test, y_pred_dummy, digits=4))
# Recall for fraud = 0.000
```

| Model | Accuracy | Fraud recall |
|-------|----------|--------------|
| Always legitimate | 99.83% | **0%** |
| Your goal | Lower accuracy OK | **Maximize** |

---

## Step 4: Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

log_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # upweight rare fraud class
        random_state=42,
    )),
])
log_pipe.fit(X_train, y_train)
y_proba = log_pipe.predict_proba(X_test)[:, 1]
y_pred = log_pipe.predict(X_test)

print(classification_report(y_test, y_pred, digits=4))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

`class_weight='balanced'` sets weights inversely proportional to class frequency:

$$
w_k = \frac{n_{\text{samples}}}{n_{\text{classes}} \cdot n_k}
$$
> **Readable form:** weight for class k = total samples divided by (number of classes × count of class k)

Fraud gets ~289× higher weight than legitimate - the optimizer cares about missing fraud.

---

## Step 5: Random Forest with Class Weights

```python
from sklearn.ensemble import RandomForestClassifier

rf_pipe = Pipeline([
    ('scaler', StandardScaler()),  # optional for RF but harmless
    ('clf', RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )),
])
rf_pipe.fit(X_train, y_train)
y_proba_rf = rf_pipe.predict_proba(X_test)[:, 1]
print(f"RF ROC-AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")
```

Prosise compares algorithms - forests often strong on tabular fraud; logistic fast and interpretable on linear separations in PCA space.

---

## Step 6: Precision-Recall Curve (The Right View)

```python
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay

prec, rec, thresholds = precision_recall_curve(y_test, y_proba_rf)
ap = average_precision_score(y_test, y_proba_rf)
print(f"Average Precision: {ap:.4f}")

PrecisionRecallDisplay.from_predictions(y_test, y_proba_rf)
plt.title(f'Precision-Recall (AP = {ap:.3f})')
plt.show()
```

On 0.17% positives, **PR-AUC** tells the truth; ROC-AUC can look deceptively high.

---

## Step 7: Choosing an Operating Threshold

Default 0.5 is rarely correct for fraud. Business picks a threshold:

| Constraint | Example |
|------------|---------|
| Max false positive rate | "Block at most 0.1% of legit transactions" |
| Min fraud recall | "Catch at least 80% of fraud" |
| Cost ratio | FP costs 5 USD support call; FN costs 500 USD |

```python
def recall_at_fpr(y_true, y_scores, max_fpr=0.001):
    """Find threshold achieving target FPR, return recall at that point."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.where(fpr <= max_fpr)[0]
    if len(idx) == 0:
        return 0.0, 1.0
    best = idx[-1]
    return tpr[best], thresholds[best]

rec, thresh = recall_at_fpr(y_test, y_proba_rf, max_fpr=0.001)
print(f"At FPR ≤ 0.1%: recall = {rec:.3f}, threshold = {thresh:.4f}")
```

> **In plain English:** "We allow 1 false alarm per 1,000 legit charges - how much fraud do we catch?" That's the question CFOs ask.

```python
# Apply custom threshold
custom_thresh = 0.02  # tune from PR curve
y_pred_custom = (y_proba_rf >= custom_thresh).astype(int)
print(classification_report(y_test, y_pred_custom, digits=4))
```

---

## Step 8: Confusion Matrix at Operating Point

```python
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(y_test, y_pred_custom)
disp = ConfusionMatrixDisplay(cm, display_labels=['Legit', 'Fraud'])
disp.plot(cmap='Reds')
plt.title(f'Fraud CM (threshold={custom_thresh})')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"Legit transactions flagged: {fp:,} ({fp/(fp+tn):.4%} of legit)")
print(f"Fraud missed: {fn} of {tp+fn}")
```

Even a good model generates **thousands** of false positives - operational cost matters.

---

## Step 9: Anomaly Detection Framing (Prosise's Angle)

Fraud is **rare** → sometimes framed as **anomaly detection**: learn "normal" transaction manifold, flag outliers.

| Supervised [classification](../../GLOSSARY.md#classification) | Anomaly approach |
|------------------------------------------------------------------|------------------|
| Uses labeled fraud | Can use mostly legit data |
| Optimizes precision/recall on `Class` | Scores "weirdness" |
| Prosise's Ch. 3 path | `IsolationForest`, autoencoders ([Course 3](https://github.com/donaldmorry/03-ai-academy-deep-learning-foundations/blob/main/README.md)) |

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.00173, random_state=42, n_jobs=-1)
iso.fit(X_train[y_train == 0])  # train on legit only
y_pred_iso = (iso.predict(X_test) == -1).astype(int)  # -1 = anomaly
print(classification_report(y_test, y_pred_iso, digits=4))
```

Supervised methods usually win when labeled fraud exists - you have 492 examples. Use anomaly as complement, not replacement.

---

## Step 10: Optional - SMOTE (Synthetic Oversampling)

```python
# pip install imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

smote_pipe = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', LogisticRegression(max_iter=1000, random_state=42)),
])
smote_pipe.fit(X_train, y_train)
```

**Caution:** SMOTE inside CV pipeline only - naive SMOTE before split leaks synthetic fraud into validation.

---

## Metric Summary for This Problem

| ❌ Don't optimize alone | ✅ Do optimize / report |
|------------------------|-------------------------|
| [Accuracy](../../GLOSSARY.md#accuracy) | Recall at business FPR |
| Raw accuracy on train | PR-AUC, Average Precision |
| Single threshold 0.5 | Threshold from cost analysis |
| "99.9% accurate!" | Confusion matrix counts |

---

## Prosise's Typical Results (Ballpark)

| Model | ROC-AUC | Notes |
|-------|---------|-------|
| Logistic + balanced weights | ~0.95-0.97 | Fast baseline |
| Random forest | ~0.95-0.98 | Strong default |
| Always legitimate | undefined / 0.5 | Fraud recall = 0 |

Exact numbers vary by split and seed - **compare models on same split**.

---

## Production Considerations (Beyond the Notebook)

1. **Concept drift** - fraud patterns change; retrain monthly
2. **Latency** - score in milliseconds at checkout
3. **Human review queue** - route borderline scores to analysts
4. **Feedback loop** - confirmed fraud → new labels
5. **Fairness** - false blocks disproportionately affect certain merchants?

Deployment path: [Chapter 07](../chapter-07-operationalizing-models/README.md).

---

## Contrast with Titanic and MNIST

| | Titanic | **Fraud** | MNIST |
|--|---------|-----------|-------|
| Imbalance | Mild (62/38) | **Extreme (99.83/0.17)** | Balanced (10 classes) |
| Primary metric | Accuracy + F1 | **Recall @ FPR, PR-AUC** | Accuracy |
| Features | Categorical + numeric | All numeric PCA | 784 pixels |

---

## Self-Check

1. Compute accuracy of "always legitimate" by hand: 284,315 / 284,807.
2. Why is PR-AUC preferred over ROC-AUC here?
3. What does `class_weight='balanced'` do?
4. If blocking a legit transaction costs 5 USD and missing fraud costs 800 USD, which metric axis matters more?

---

## References

- [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [ULB MLG - Original dataset paper context](https://www.researchgate.net/publication/319867396_Calibrating_Probability_with_Under-sampling_for_Unbalanced_Classification)
- [Scikit-Learn: class_weight](https://scikit-learn.org/stable/chapters/generated/sklearn.linear_model.LogisticRegression.html)
- [imbalanced-learn: SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- Prosise, *Applied ML and AI for Engineers*, Ch. 3 - Fraud
- [StatQuest: Precision and Recall](https://www.youtube.com/watch?v=j-4uzh44IQg)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 3.6](./section-06-case-study-titanic-survival.md) | **Next:** [Section 3.8 - MNIST Digit Recognition](./section-08-case-study-mnist-digit-recognition.md)



