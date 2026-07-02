# Section 3.3: Classification Accuracy Measures

> **Source:** Prosise, Ch. 3 - "Accuracy Measures for Classification Models"  
> **Prerequisites:** [Section 3.2](./section-02-logistic-regression.md) | [Section 2.6](../chapter-02-regression-models/section-06-regression-accuracy-measures.md)  
> **Glossary:** [accuracy](../../GLOSSARY.md#accuracy) | [classification](../../GLOSSARY.md#classification)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## You Can't Improve What You Don't Measure - Classification Edition

In [Chapter 02](../chapter-02-regression-models/section-06-regression-accuracy-measures.md), you used [RMSE](../../GLOSSARY.md#rmse) and [MAE](../../GLOSSARY.md#mae). For [classification](../../GLOSSARY.md#classification), the story is richer because **not all errors cost the same**.

Missing fraud costs thousands. A false fraud alarm blocks a legitimate purchase. [Accuracy](../../GLOSSARY.md#accuracy) alone hides these tradeoffs - especially on imbalanced data like Prosise's credit card set (492 fraud / 284,807 total).

> **Humorous analogy:** Accuracy is like grading a spam filter by "percent of emails handled" - if 99% of your inbox is ham, a filter that says "not spam" to everything gets 99% accuracy and zero usefulness.

---

## The Confusion Matrix: Four Kinds of Truth

For binary classification with positive class **1** (e.g., fraud, survived):

|  | Predicted **1** | Predicted **0** |
|--|-----------------|-----------------|
| **Actual 1** | True Positive (TP) | False Negative (FN) |
| **Actual 0** | False Positive (FP) | True Negative (TN) |

```
                    Predicted
                 Pos      Neg
              ┌────────┬────────┐
    Actual Pos│   TP   │   FN   │  ← missed fraud / missed survivors
              ├────────┼────────┤
    Actual Neg│   FP   │   TN   │  ← false alarms
              └────────┴────────┘
```

> **In plain English:** TP = caught it. TN = correctly ignored. FP = cried wolf. FN = slept through the fire alarm.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_true  = [1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
y_pred  = [1, 0, 0, 1, 0, 1, 1, 0, 0, 0]

cm = confusion_matrix(y_true, y_pred)
print(cm)
# [[TN FP]
#  [FN TP]]  for sklearn's layout when labels=[0,1]

disp = ConfusionMatrixDisplay(cm, display_labels=['Neg', 'Pos'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```

---

## Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$
> **Readable form:** accuracy = (correct predictions) divided by (all predictions)

**When it's fine:** Balanced classes, symmetric error costs (rare in production).

**When it lies:** Fraud data - a model predicting "legitimate" always scores:

$$
\text{Accuracy} = \frac{284{,}315}{284{,}807} \approx 99.83\%
$$
> **Readable form:** accuracy ≈ 284,315 correct out of 284,807 ≈ 99.83% - while catching **zero** fraud

Full treatment in [Section 3.7](./section-07-case-study-credit-card-fraud-detection.md).

---

## Precision

Of everything you **predicted positive**, how many were actually positive?

$$
\text{Precision} = \frac{TP}{TP + FP}
$$
> **Readable form:** precision = true positives divided by (true positives + false positives)

| High precision means | Low precision means |
|---------------------|---------------------|
| Few false alarms | Many false positives |

**Use when FP is expensive:** Email spam filter marking boss's email as spam. Medical test falsely saying you're sick.

---

## Recall (Sensitivity, True Positive Rate)

Of all **actual positives**, how many did you catch?

$$
\text{Recall} = \frac{TP}{TP + FN}
$$
> **Readable form:** recall = true positives divided by (true positives + false negatives)

| High recall means | Low recall means |
|-------------------|------------------|
| Few missed positives | Many false negatives |

**Use when FN is expensive:** Cancer screening, fraud detection, safety-critical systems.

---

## The Precision-Recall Tradeoff

You cannot usually maximize both. Lowering the probability threshold (from [Section 3.2](./section-02-logistic-regression.md)) catches more positives → **recall up**, **precision down**.

```
Threshold ↑  →  fewer positives predicted  →  precision ↑, recall ↓
Threshold ↓  →  more positives predicted   →  precision ↓, recall ↑
```

**Real-life analogy:** Airport security. Loose screening (low threshold): high recall for threats, long lines and many false alarms (low precision). Tight screening: short lines, but things slip through.

```python
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

# Simulated fraud scores
y_true = np.array([0]*1000 + [1]*10)
y_scores = np.concatenate([
    np.random.beta(2, 8, 1000),   # legit: low scores
    np.random.beta(8, 2, 10),     # fraud: high scores
])

precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
print(f"At threshold 0.5: precision≈{precisions[len(thresholds)//2]:.2f}, "
      f"recall≈{recalls[len(thresholds)//2]:.2f}")
print(f"Average Precision: {average_precision_score(y_true, y_scores):.3f}")
```

---

## F1 Score: Harmonic Mean of Precision and Recall

$$
F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}
$$
> **Readable form:** F1 = 2 × (precision × recall) divided by (precision + recall)

| F1 | Interpretation |
|----|----------------|
| 1.0 | Perfect precision and recall |
| 0.0 | Terrible on at least one axis |

**When to use:** Single number for imbalanced binary problems when you care about the **positive class** equally on both axes. Not a substitute for domain-specific cost analysis.

```python
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

print(classification_report(y_true[:10], y_pred, target_names=['Neg', 'Pos']))
# Includes precision, recall, F1 per class
```

---

## Multi-Class Metrics

For MNIST ([Section 3.8](./section-08-case-study-mnist-digit-recognition.md)), extend via averaging:

| Average | Meaning |
|---------|---------|
| `macro` | Unweighted mean per class - treats "1" and "9" equally |
| `weighted` | Weighted by class support |
| `micro` | Global TP/FP/FN - equals accuracy for single-label multi-class |

---

## ROC Curve and AUC

**ROC** = Receiver Operating Characteristic. Plots **True Positive Rate** (recall) vs **False Positive Rate**:

$$
\text{TPR} = \frac{TP}{TP + FN} = \text{Recall}
$$

$$
\text{FPR} = \frac{FP}{FP + TN}
$$
> **Readable form:** true positive rate = TP divided by (TP + FN); false positive rate = FP divided by (FP + TN)

Vary the threshold from 1 → 0; trace the curve. **Perfect classifier:** hugs top-left. **Random guess:** diagonal line.

**AUC** (Area Under ROC Curve): probability a random positive scores higher than a random negative.

$$
\text{AUC} \in [0.5, 1.0]
$$
> **Readable form:** AUC ranges from 0.5 (no better than random) to 1.0 (perfect ranking)

| AUC | Quality |
|-----|---------|
| 0.5 | Coin flip |
| 0.7-0.8 | Acceptable |
| 0.8-0.9 | Good |
| 0.9+ | Excellent (verify not leakage!) |

```python
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
auc = roc_auc_score(y_true, y_scores)
print(f"ROC-AUC: {auc:.3f}")

RocCurveDisplay.from_predictions(y_true, y_scores)
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.title(f'ROC Curve (AUC = {auc:.3f})')
plt.legend()
plt.show()
```

### ROC-AUC vs Precision-Recall

| Situation | Prefer |
|-----------|--------|
| Balanced classes | ROC-AUC |
| **Heavily imbalanced** (fraud) | **Precision-Recall AUC** |
| Need interpretable threshold | PR curve at operating point |

> **In plain English:** On fraud data with 0.17% positives, ROC-AUC can look great while precision at useful recall is awful. Always plot **both** for imbalanced problems.

---

## Metric Selection Cheat Sheet (Prosise + Industry)

| Business question | Primary metric |
|-------------------|----------------|
| "Are we right most of the time?" (balanced) | Accuracy |
| "Minimize false accusations" | Precision |
| "Catch every case" | Recall |
| "Balance both on minority class" | F1 |
| "Rank/score quality" (balanced) | ROC-AUC |
| "Rank on rare class" (fraud) | PR-AUC, recall at fixed FPR |

---

## Full Evaluation Pipeline

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, f1_score
)

X, y = make_classification(n_samples=2000, weights=[0.9, 0.1],
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

clf = LogisticRegression(class_weight='balanced', max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"F1:        {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")
print("\n", classification_report(y_test, y_pred))
```

---

## Regression vs Classification Metrics (Cross-Link)

| [Regression (2.6)](../chapter-02-regression-models/section-06-regression-accuracy-measures.md) | Classification (this section) |
|--------------------------------------------------------------------------------|------------------------------|
| Residual $e_i = y_i - \hat{y}_i$ | TP, FP, TN, FN |
| [RMSE](../../GLOSSARY.md#rmse) | Accuracy (with caution) |
| [MAE](../../GLOSSARY.md#mae) | MAE on probabilities (uncommon) |
| [R²](../../GLOSSARY.md#r-squared) | Pseudo-R² for logistic (advanced) |

---

## Common Mistakes

1. **Reporting accuracy only on fraud** - deploys a brick
2. **Optimizing ROC-AUC, deploying on precision** - metric mismatch
3. **Ignoring class labels in confusion matrix** - which class is "positive"?
4. **Tuning on test set** - same sin as regression; use validation / [CV](../../GLOSSARY.md#cross-validation)
5. **Forgetting baseline** - beat majority class from [Section 3.1](./section-01-classification-fundamentals.md)

---

## Self-Check

1. A model has TP=40, FP=10, FN=20, TN=930. Compute precision, recall, accuracy, F1.
2. Why can 99.9% accuracy be worthless for fraud?
3. When is PR-AUC better than ROC-AUC?
4. If you need at most 1% false positive rate, which curve do you read?

---

## References

- [Scikit-Learn: Model Evaluation - Classification](https://scikit-learn.org/stable/chapters/model_evaluation.html#classification-metrics)
- [Scikit-Learn: Confusion Matrix](https://scikit-learn.org/stable/chapters/generated/sklearn.metrics.confusion_matrix.html)
- [Scikit-Learn: ROC](https://scikit-learn.org/stable/chapters/generated/sklearn.metrics.roc_auc_score.html)
- Prosise, *Applied ML and AI for Engineers*, Ch. 3 - Accuracy Measures
- [StatQuest: Sensitivity and Specificity](https://www.youtube.com/watch?v=vP06aMoz4v8)
- [StatQuest: ROC and AUC](https://www.youtube.com/watch?v=4jRBRDbJedM)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 3.2](./section-02-logistic-regression.md) | **Next:** [Section 3.4 - Categorical Encoding](./section-04-categorical-data-and-feature-encoding.md)
