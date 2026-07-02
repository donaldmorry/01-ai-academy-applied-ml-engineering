# Lab 04: Sentiment, Spam, and Recommendations

> **Prerequisites:** Sections [4.1](./section-01-text-as-data.md)-[4.8](./section-08-pipelines-and-production-text-ml.md)  
> **Estimated time:** 4-6 hours  
> **Glossary:** [../../GLOSSARY.md](../../GLOSSARY.md) | **Math:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Objectives

By the end of this lab you will:

1. Build a **sentiment analyzer** on movie reviews with TF-IDF + classifier (>85% accuracy target)
2. Build a **spam filter** optimized for **precision** on imbalanced SMS/email data
3. Build a **content-based movie recommender** using [cosine similarity](../../GLOSSARY.md#cosine-similarity)
4. Inspect top weighted terms and misclassified examples
5. Persist all three systems with `joblib` pipelines
6. Write a brief limitations note on [bag-of-words](../../GLOSSARY.md#bag-of-words) approaches

> **Humorous briefing:** Three apps, one vectorizer family, zero GPUs. You are about to deploy more NLP than most startups in 2008 - using math from the 1960s. It still works.

---

## Setup

```python
# pip install pandas numpy matplotlib seaborn scikit-learn nltk joblib
import warnings; warnings.filterwarnings('ignore')
import joblib, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_score, recall_score, f1_score, PrecisionRecallDisplay)
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

RANDOM_STATE = 42; np.random.seed(RANDOM_STATE)
MODEL_DIR = Path('lab04_models'); MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR = Path('data/lab04'); DATA_DIR.mkdir(parents=True, exist_ok=True)
```

---

## Data Sources

| Dataset | URL | Use |
|---------|-----|-----|
| IMDB / sentiment CSV | [Stanford Sentiment](http://ai.stanford.edu/~amaas/data/sentiment/) | Part A |
| SMS Spam Collection | [UCI](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) | Part B |
| Movie plots (custom) | Synthetic below or Kaggle TMDB | Part C |

Place files in `data/lab04/`. Synthetic fallbacks included if downloads unavailable.

---

## Part A: Sentiment Analyzer (90 min)

*Follow [Section 4.5](./section-05-sentiment-analysis.md) - Prosise movie review pattern with `ngram_range=(1,2)`.*

### A1. Load data (15 min)

```python
# df = pd.read_csv(DATA_DIR / 'reviews.csv')  # columns: review, sentiment
# Or use synthetic fallback from Section 4.5 - 2500 pos + 2500 neg templates

df_sent = load_sentiment_data()  # see Section 4.5 for full loader
print(df_sent['sentiment'].value_counts())
```

**Tasks:** Plot review length by sentiment; report class balance.

### A2. Train pipeline (30 min)

```python
X = df_sent['review'].astype(str)
y = df_sent['sentiment'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

sentiment_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        min_df=5,
        ngram_range=(1, 2),
        max_features=20000,
    )),
    ('clf', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
])

sentiment_pipe.fit(X_train, y_train)
y_pred = sentiment_pipe.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))
```

**Also train Naive Bayes baseline** ([Section 4.4](./section-04-naive-bayes-for-text.md)):

```python
nb_pipe = Pipeline([
    ('counts', CountVectorizer(stop_words='english', min_df=5, ngram_range=(1, 2))),
    ('clf', MultinomialNB(alpha=1.0)),
])
nb_pipe.fit(X_train, y_train)
print("NB accuracy:", nb_pipe.score(X_test, y_test))
```

**Target:** ≥ 85% test accuracy on real IMDB subset; synthetic may exceed 95%.

### A3. Feature inspection (20 min)

```python
vec = sentiment_pipe.named_steps['tfidf']
clf = sentiment_pipe.named_steps['clf']
features = vec.get_feature_names_out()
coefs = clf.coef_[0]

top_pos_idx = np.argsort(coefs)[-10:][::-1]
top_neg_idx = np.argsort(coefs)[:10]

print("=== 10 most POSITIVE terms ===")
for i in top_pos_idx:
    print(f"  {features[i]:25s} {coefs[i]:+.4f}")

print("\n=== 10 most NEGATIVE terms ===")
for i in top_neg_idx:
    print(f"  {features[i]:25s} {coefs[i]:+.4f}")
```

**Deliverable:** Table of top 10 positive and negative terms.

### A4. Error analysis (15 min)

```python
proba = sentiment_pipe.predict_proba(X_test)[:, 1]
errors = pd.DataFrame({
    'review': X_test.values,
    'true': y_test.values,
    'pred': y_pred,
    'prob_pos': proba,
})
errors = errors[errors['true'] != errors['pred']]
print(errors.head(8).to_string())

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['neg', 'pos']).plot(cmap='Blues')
plt.title('Sentiment Confusion Matrix')
plt.show()
```

**Write 2-3 sentences:** What patterns cause misclassification?

### A5. Custom predictions (10 min)

Test three custom review strings with `predict` and `predict_proba`. Include one with negation (e.g., `"Not the worst film I have seen"`).

---

## Part B: Spam Filter (90 min)

*Follow [Section 4.6](./section-06-spam-filtering.md) - precision focus, ~500 spam / many ham.*

### B1. Load data (15 min)

Load UCI `SMSSpamCollection` or use the 4500 ham / 500 spam synthetic generator from [Section 4.6](./section-06-spam-filtering.md).

### B2. Train & evaluate (30 min)

```python
X = df_spam['message'].astype(str)
y = (df_spam['label'] == 'spam').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

spam_pipe = Pipeline([
    ('vec', CountVectorizer(stop_words='english', min_df=2, ngram_range=(1, 2))),
    ('clf', MultinomialNB(alpha=1.0)),
])
spam_pipe.fit(X_train, y_train)

proba_spam = spam_pipe.predict_proba(X_test)[:, 1]
y_pred_default = spam_pipe.predict(X_test)

print("=== Default threshold 0.5 ===")
print(classification_report(y_test, y_pred_default, target_names=['ham', 'spam']))
```

### B3. Precision-focused threshold (25 min)

```python
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
best_thresh = 0.5
for thresh in np.arange(0.5, 0.99, 0.05):
    y_hat = (proba_spam >= thresh).astype(int)
    prec = precision_score(y_test, y_hat)
    rec = recall_score(y_test, y_hat)
    f1 = f1_score(y_test, y_hat)
    print(f"{thresh:10.2f} {prec:10.3f} {rec:10.3f} {f1:10.3f}")
    if prec >= 0.95 and rec >= 0.3:
        best_thresh = thresh

THRESHOLD = 0.90  # adjust per your results
y_pred_tuned = (proba_spam >= THRESHOLD).astype(int)
print(f"\n=== Tuned threshold {THRESHOLD} ===")
print(classification_report(y_test, y_pred_tuned, target_names=['ham', 'spam']))

PrecisionRecallDisplay.from_predictions(y_test, proba_spam)
plt.title('Spam Precision-Recall Curve')
plt.show()
```

**Target:** Precision ≥ 0.95 on spam class at chosen threshold (adjust if synthetic data differs).

### B4. Confusion matrix & FP audit (20 min)

Report FP (ham→spam) count from `confusion_matrix`. Plot with `ConfusionMatrixDisplay`. List any false-positive message texts.

---

## Part C: Movie Recommender (60 min)

*Follow [Section 4.7](./section-07-cosine-similarity-and-recommender-systems.md).*

### C1. Build catalog (15 min)

Use the 12-movie catalog from [Section 4.7](./section-07-cosine-similarity-and-recommender-systems.md) (`title`, `genre`, `description`). Set `movies['text'] = genre + ' ' + description`.

### C2. TF-IDF + similarity (20 min)

```python
tfidf = TfidfVectorizer(stop_words='english', min_df=1, ngram_range=(1, 2))
matrix = tfidf.fit_transform(movies['text'])
sim = cosine_similarity(matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(sim, xticklabels=movies['title'], yticklabels=movies['title'],
            cmap='YlOrRd', annot=True, fmt='.2f')
plt.title('Movie Cosine Similarity')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

### C3. Recommendation function (25 min)

Implement `recommend(title, top_k=5)` using `cosine_similarity` matrix - skip self, return top-K titles. Test seeds: `Space Odyssey`, `Love in Paris`, `Robot Uprising`. Add `recommend_from_text(query)` for free-text queries.

---

## Part D: Persist & Load (30 min)

`joblib.dump` sentiment and spam pipelines; save recommender artifact (vectorizer, matrix, similarity, titles). Write `metadata.json`. Reload and smoke-test with `assert loaded.predict(["great film"])[0] == 1`.

---

## Part E: Limitations Memo (20 min)

Write **150-250 words** covering:

1. What [bag-of-words](../../GLOSSARY.md#bag-of-words) loses (word order, semantics, negation scope)
2. Why spam precision mattered more than recall ([Section 3.3](../chapter-03-classification-models/section-03-classification-accuracy-measures.md))
3. How content-based recommenders differ from collaborative filtering
4. One scenario where you would upgrade to embeddings/transformers ([Chapter 13](../chapter-13-natural-language-processing/README.md))

---

## Deliverables Checklist

| # | Item | Done? |
|---|------|-------|
| 1 | Sentiment model ≥ 85% accuracy (real data) | ☐ |
| 2 | Top 10 positive + negative weighted terms | ☐ |
| 3 | Spam filter with precision ≥ 0.95 at chosen threshold | ☐ |
| 4 | PR curve plot for spam | ☐ |
| 5 | Top-5 recommendations for 3 seed movies | ☐ |
| 6 | Similarity heatmap | ☐ |
| 7 | Three `joblib` artifacts + metadata.json | ☐ |
| 8 | Custom prediction examples (≥ 3 per system) | ☐ |
| 9 | Limitations memo | ☐ |

---

## References

- Prosise, Ch. 4 - all three applications
- [Section 4.5](./section-05-sentiment-analysis.md) | [4.6](./section-06-spam-filtering.md) | [4.7](./section-07-cosine-similarity-and-recommender-systems.md)
- Scikit-Learn text tutorial: [https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

---

**Previous:** [Section 4.8 - Pipelines & Production Text ML](./section-08-pipelines-and-production-text-ml.md)  
**Next:** [Chapter 05 - Support Vector Machines](../chapter-05-support-vector-machines/README.md)