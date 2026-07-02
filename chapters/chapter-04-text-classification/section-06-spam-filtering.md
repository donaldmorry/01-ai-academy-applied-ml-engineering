# Section 4.6: Spam Filtering

> **Source:** Prosise, Ch. 4 - email/SMS spam classification, precision focus  
> **Prerequisites:** [Section 4.5](./section-05-sentiment-analysis.md) | [Section 3.7](../chapter-03-classification-models/section-07-case-study-credit-card-fraud-detection.md)  
> **Glossary:** [naive-bayes](../../GLOSSARY.md#naive-bayes) | [classification](../../GLOSSARY.md#classification)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Asymmetric Cost of Being Wrong

A spam filter has two failure modes with **wildly different** consequences:

| Error | Name | User experience |
|-------|------|-----------------|
| Spam → inbox | False Negative (FN) | Annoying - delete junk |
| Ham → spam folder | False Positive (FP) | **Catastrophic** - miss job offer, invoice, security alert |

Prosise stresses: optimize **precision** for the spam class - when the model says "spam," it should almost always be right. This mirrors the fraud detection mindset in [Section 3.7](../chapter-03-classification-models/section-07-case-study-credit-card-fraud-detection.md), but with FP as the expensive error instead of FN.

> **Humorous analogy:** A spam filter with low precision is a overzealous bouncer who throws your grandma out of the club because she said "free hugs." Technically suspicious. Socially unemployed.

> **In plain English:** Missing some spam is acceptable. Marking one important email as spam is not. Design metrics and thresholds accordingly.

---

## Prosise Dataset Characteristics

Chapter 4 uses a classic imbalanced corpus:

- **~500 spam** messages (minority)
- **Thousands of ham** messages (majority)

```python
import pandas as pd
import numpy as np

# Typical SMS spam dataset columns: 'label', 'message'
# label ∈ {'ham', 'spam'}

np.random.seed(42)
ham_templates = [
    "Hey are we still meeting for lunch tomorrow",
    "Your package has been delivered to the front door",
    "Mom called please call her back when free",
    "Meeting moved to 3pm conference room B",
    "Thanks for the report I will review tonight",
]
spam_templates = [
    "Congratulations you won a free prize click here now",
    "URGENT claim your refund call this number immediately",
    "Free entry to weekly lottery text WIN to 12345",
    "You have been selected for a cash reward limited time",
    "Lowest rate guaranteed act now exclusive offer",
]

messages = []
labels = []
for _ in range(4500):
    messages.append(np.random.choice(ham_templates))
    labels.append('ham')
for _ in range(500):
    messages.append(np.random.choice(spam_templates))
    labels.append('spam')

df = pd.DataFrame({'message': messages, 'label': labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(df['label'].value_counts())
print(f"Spam rate: {(df['label']=='spam').mean():.2%}")
```

Real data: [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection), Kaggle spam datasets.

---

## Why Accuracy Lies (Again)

A dummy classifier predicting **always ham**:

$$
\text{Accuracy} = \frac{4500}{5000} = 90\%
$$
> **Readable form:** accuracy = 4500 correct out of 5000 = 90% while catching zero spam

Sound familiar? Same trap as 99.83% accuracy on fraud in [Section 3.3](../chapter-03-classification-models/section-03-classification-accuracy-measures.md). For spam, report **precision**, **recall**, and **F1** on the spam class.

---

## Precision: The North Star Metric

$$
\text{Precision}_{\text{spam}} = \frac{TP}{TP + FP}
$$
> **Readable form:** spam precision = true spam caught divided by (all messages flagged as spam)

**High precision** → few ham messages in the spam folder.

$$
\text{Recall}_{\text{spam}} = \frac{TP}{TP + FN}
$$
> **Readable form:** spam recall = true spam caught divided by (all actual spam messages)

**High recall** → most spam removed from inbox.

For Prosise's use case: **prioritize precision**, accept moderate recall. Tune via threshold and class weights.

---

## Building the Baseline Pipeline

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

X = df['message'].astype(str)
y = (df['label'] == 'spam').astype(int)  # 1=spam, 0=ham

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

spam_pipe = Pipeline([
    ('vec', CountVectorizer(stop_words='english', min_df=2, ngram_range=(1, 2))),
    ('clf', MultinomialNB(alpha=1.0)),
])

spam_pipe.fit(X_train, y_train)
y_pred = spam_pipe.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
```

[Naive Bayes](../../GLOSSARY.md#naive-bayes) is historically the spam filter workhorse - Paul Graham's 2002 essay popularized it.

---

## Inspecting Spam Vocabulary

```python
vec = spam_pipe.named_steps['vec']
clf = spam_pipe.named_steps['clf']

feature_names = vec.get_feature_names_out()
log_probs = clf.feature_log_prob_

spam_idx = 1
top_spam = log_probs[spam_idx].argsort()[-20:][::-1]
print("Top spam indicators:")
for i in top_spam:
    print(f"  {feature_names[i]}")
```

Expect: `free`, `win`, `urgent`, `prize`, `click`, `claim`. If `meeting` ranks high, investigate data leakage or mislabeled ham.

---

## Threshold Tuning for Precision

Default `predict()` uses 0.5 posterior - not optimal for asymmetric costs.

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

proba_spam = spam_pipe.predict_proba(X_test)[:, 1]

print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
for thresh in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    y_hat = (proba_spam >= thresh).astype(int)
    prec = precision_score(y_test, y_hat)
    rec  = recall_score(y_test, y_hat)
    f1   = f1_score(y_test, y_hat)
    print(f"{thresh:10.2f} {prec:10.3f} {rec:10.3f} {f1:10.3f}")
```

**Pattern:** Higher threshold → precision ↑, recall ↓. Pick the lowest threshold meeting your precision SLA (e.g., 99% precision).

---

## Precision-Recall Curve

```python
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt

PrecisionRecallDisplay.from_predictions(y_test, proba_spam, name='Spam filter')
plt.title('Precision-Recall Curve (Spam Class)')
plt.show()
```

Same visualization family as fraud in [Section 3.7](../chapter-03-classification-models/section-07-case-study-credit-card-fraud-detection.md). The PR curve is more informative than ROC on imbalanced text.

---

## Cross-Validation on Text

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    spam_pipe, X, y,
    cv=5,
    scoring='precision',  # optimize what matters
)
print(f"CV precision (spam): {scores.mean():.3f} (+/- {scores.std():.3f})")
```

Use `scoring='precision'`, `'recall'`, or `'f1'` - not `'accuracy'`.

---

## Evaluating a Custom Message

```python
def classify_email(text, pipeline, threshold=0.85):
    p_spam = pipeline.predict_proba([text])[0, 1]
    if p_spam >= threshold:
        return 'SPAM', p_spam
    return 'HAM', 1 - p_spam

tests = [
    "Quarterly budget review attached please confirm",
    "FREE VIAGRA CLICK NOW LIMITED OFFER!!!",
    "Your verification code is 847291",
]

for msg in tests:
    label, conf = classify_email(msg, spam_pipe, threshold=0.85)
    print(f"[{conf:.2f}] {label:4s} - {msg}")
```

High threshold (0.85+) protects precision in production.

---

## Confusion Matrix Interpretation

```
                 Predicted
              Ham      Spam
Actual Ham   [TN]     [FP]  ← FP = boss's email deleted
Actual Spam  [FN]     [TP]  ← FN = junk in inbox
```

From [Section 3.3](../chapter-03-classification-models/section-03-classification-accuracy-measures.md):

- **Minimize FP** → high precision
- Accept some **FN** → moderate recall is OK

Quantify user pain with confusion matrix counts from [Section 3.3](../chapter-03-classification-models/section-03-classification-accuracy-measures.md).

---

## Key Takeaways

1. **Spam filtering** is imbalanced binary [classification](../../GLOSSARY.md#classification) (~500 spam, many ham)
2. **False positives** (ham marked spam) cost more than false negatives
3. Optimize **precision** first; tune threshold on `predict_proba`
4. **[Naive Bayes](../../GLOSSARY.md#naive-bayes)** + `CountVectorizer` is a strong, fast baseline
5. **Accuracy is misleading** - same section as fraud and [Section 3.3](../chapter-03-classification-models/section-03-classification-accuracy-measures.md)
6. **Monitor false positives** in production and retrain continuously

---

## Check Your Understanding

1. Why is a false positive worse than a false negative for email?
2. What accuracy does an "always ham" classifier achieve on 500 spam / 4500 ham?
3. How does raising the spam probability threshold affect precision and recall?
4. Why does Prosise use n-grams for spam detection?
5. What would you log in production to improve the filter over time?

---

## References

- Prosise, Ch. 4 - spam filtering
- Paul Graham - "A Plan for Spam" (2002): [http://www.paulgraham.com/spam.html](http://www.paulgraham.com/spam.html)
- UCI SMS Spam Collection: [https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- Scikit-Learn - precision_recall_curve: [https://scikit-learn.org/stable/chapters/generated/sklearn.metrics.precision_recall_curve.html](https://scikit-learn.org/stable/chapters/generated/sklearn.metrics.precision_recall_curve.html)

---

**Previous:** [Section 4.5 - Sentiment Analysis](./section-05-sentiment-analysis.md)  
**Next:** [Section 4.7 - Cosine Similarity & Recommenders](./section-07-cosine-similarity-and-recommender-systems.md)



