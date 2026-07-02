# Section 4.5: Sentiment Analysis

> **Source:** Prosise, Ch. 4 - movie review sentiment classification  
> **Prerequisites:** [Section 4.4](./section-04-naive-bayes-for-text.md) | [Section 3.3](../chapter-03-classification-models/section-03-classification-accuracy-measures.md)  
> **Glossary:** [tf-idf](../../GLOSSARY.md#tf-idf) | [naive-bayes](../../GLOSSARY.md#naive-bayes) | [classification](../../GLOSSARY.md#classification)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Business Problem

Movie studios, streaming platforms, and product teams want to know: **does this review praise or pan the experience?** Manual reading does not scale to millions of posts. **Sentiment analysis** - binary [classification](../../GLOSSARY.md#classification) of opinion polarity - is the textbook introduction to NLP in production.

Prosise trains on thousands of labeled movie reviews: positive vs negative. The pipeline combines everything from Sections 4.1-4.4:

- [TF-IDF](../../GLOSSARY.md#tf-idf) vectorization with `ngram_range=(1,2)`
- [Naive Bayes](../../GLOSSARY.md#naive-bayes) or logistic regression
- Metrics beyond raw [accuracy](../../GLOSSARY.md#accuracy) from [Section 3.3](../chapter-03-classification-models/section-03-classification-accuracy-measures.md)

> **Humorous analogy:** Sentiment analysis is a film critic who never sleeps, never asks for popcorn, and has read every IMDb review twice - but still thinks "not bad" means "bad" until you teach it bigrams.

> **In plain English:** Learn which words and phrases correlate with positive and negative labels, then score new reviews by which side's vocabulary they resemble more.

---

## Dataset Shape (Prosise Pattern)

Typical CSV layout:

| review | sentiment |
|--------|-----------|
| "A masterpiece of modern cinema…" | positive |
| "I walked out after twenty minutes…" | negative |

```python
import pandas as pd
import numpy as np

# df = pd.read_csv('movie_reviews.csv')
# Prosise-style: thousands of reviews, roughly balanced classes

# Synthetic stand-in for teaching (replace with real CSV)
np.random.seed(42)
positive_templates = [
    "brilliant acting wonderful story loved every minute",
    "fantastic film best movie of the year highly recommend",
    "outstanding performances emotional and powerful",
    "great cinematography amazing soundtrack a true gem",
]
negative_templates = [
    "terrible waste of time boring predictable plot",
    "awful acting worst film I have ever seen",
    "disappointing dull and poorly written",
    "hate this movie complete garbage avoid it",
]

reviews = []
labels = []
for _ in range(1250):
    t = np.random.choice(positive_templates)
    reviews.append(t + " " + "great " * np.random.randint(0, 3))
    labels.append(1)
for _ in range(1250):
    t = np.random.choice(negative_templates)
    reviews.append(t + " " + "bad " * np.random.randint(0, 3))
    labels.append(0)

df = pd.DataFrame({'review': reviews, 'sentiment': labels})
print(df['sentiment'].value_counts())
print(df['review'].str.len().describe())
```

Real datasets: [Stanford IMDB](http://ai.stanford.edu/~amaas/data/sentiment/), Kaggle sentiment competitions.

---

## Train / Test Split with Stratification

```python
from sklearn.model_selection import train_test_split

X = df['review'].astype(str)
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
```

`stratify=y` preserves 50/50 balance in both splits - same discipline as [Section 3.6](../chapter-03-classification-models/section-06-case-study-titanic-survival.md).

---

## Vectorization: The Prosise Recipe

Prosise's key hyperparameter choice for movie reviews:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words='english',
    min_df=5,
    ngram_range=(1, 2),
    max_features=20000,
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print(f"Matrix shape: {X_train_vec.shape}")
print(f"Sample features: {vectorizer.get_feature_names_out()[:15]}")
```

### Why `ngram_range=(1, 2)`?

| Unigram only | Unigrams + bigrams |
|--------------|-------------------|
| `"not"`, `"good"` separate | `"not good"` as one feature |
| Misses `"not bad"` irony partially | Captures common negation patterns |
| Smaller vocabulary | Larger vocabulary - watch memory |

From [Section 4.3](./section-03-bag-of-words-and-tf-idf.md): bigrams sneak limited word order back into [bag-of-words](../../GLOSSARY.md#bag-of-words).

### Why `min_df=5`?

Terms appearing in fewer than 5 documents are usually typos or ultra-rare names - they [overfit](../../GLOSSARY.md#overfitting) without generalizing.

---

## Model A: Multinomial Naive Bayes

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

count_vec = CountVectorizer(
    stop_words='english',
    min_df=5,
    ngram_range=(1, 2),
    max_features=20000,
)

X_train_counts = count_vec.fit_transform(X_train)
X_test_counts  = count_vec.transform(X_test)

nb = MultinomialNB(alpha=1.0)
nb.fit(X_train_counts, y_train)
y_pred_nb = nb.predict(X_test_counts)

print("Naive Bayes accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb, target_names=['neg', 'pos']))
```

Prosise often reports **85%+ accuracy** on held-out reviews with this setup.

---

## Model B: Logistic Regression on TF-IDF

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

lr = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
lr.fit(X_train_vec, y_train)
y_pred_lr = lr.predict(X_test_vec)

print("Logistic Regression accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, target_names=['neg', 'pos']))
```

Often edges NB by 1-3 points on clean review corpora. Interpretable coefficients tie back to [Section 3.2](../chapter-03-classification-models/section-02-logistic-regression.md).

---

## Inspecting the Most Predictive Terms

Prosise emphasizes **feature inspection** - what did the model learn?

```python
import numpy as np

def top_terms(vectorizer, model, class_idx, n=15):
  feature_names = vectorizer.get_feature_names_out()
  if hasattr(model, 'coef_'):
      weights = model.coef_[class_idx]
  else:
      weights = model.feature_log_prob_[class_idx]
  top_idx = np.argsort(weights)[-n:][::-1]
  return [(feature_names[i], weights[i]) for i in top_idx]

print("=== Logistic: most POSITIVE ===")
for term, w in top_terms(vectorizer, lr, 0, 15):
    print(f"  {term:25s} {w:+.3f}")

print("\n=== Logistic: most NEGATIVE ===")
neg_weights = lr.coef_[0]
feature_names = vectorizer.get_feature_names_out()
bottom_idx = np.argsort(neg_weights)[:15]
for i in bottom_idx:
    print(f"  {feature_names[i]:25s} {neg_weights[i]:+.3f}")
```

Expected positive cues: `loved`, `brilliant`, `fantastic`, `wonderful`.  
Expected negative cues: `terrible`, `awful`, `boring`, `waste`.

> **In plain English:** If your top positive term is `the`, something went wrong with stop words or preprocessing.

---

## Confusion Matrix and Error Types

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred_lr)
disp = ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title('Sentiment Confusion Matrix')
plt.show()
```

| Error | Business impact |
|-------|-----------------|
| False positive (predict pos, actually neg) | Over-hype bad films |
| False negative (predict neg, actually pos) | Miss marketing gold |

For symmetric costs, **accuracy** and **F1** are reasonable. For reputation-sensitive dashboards, tune threshold on `predict_proba` ([Section 3.3](../chapter-03-classification-models/section-03-classification-accuracy-measures.md)).

---

## Error Analysis on Misclassified Reviews

```python
import pandas as pd

errors = pd.DataFrame({
    'review': X_test.values,
    'true': y_test.values,
    'pred': y_pred_lr,
    'prob_pos': proba,
})
errors = errors[errors['true'] != errors['pred']].sort_values('prob_pos')

print("=== False Negatives (true pos, predicted neg) ===")
print(errors[errors['true'] == 1].head(5)[['review', 'prob_pos']])

print("\n=== False Positives (true neg, predicted pos) ===")
print(errors[errors['true'] == 0].head(5)[['review', 'prob_pos']])
```

Look for sarcasm, mixed sentiment, and domain-specific vocabulary.

---

## Limitations of Bag-of-Words Sentiment

| Limitation | Example |
|------------|---------|
| Negation scope | `"didn't not like it"` |
| Comparative | `"better than the first one"` (needs context) |
| Aspect sentiment | `"great visuals, awful dialogue"` |
| Emoji / slang | `"this slaps 🔥"` |
| Language mix | Code-switching breaks English stop lists |

Deep models in [Chapter 13](../chapter-13-natural-language-processing/README.md) address these; TF-IDF + logistic remains a mandatory baseline.

---

## Key Takeaways

1. **Sentiment analysis** is binary text [classification](../../GLOSSARY.md#classification)
2. Prosise uses **movie reviews** with TF-IDF, `min_df=5`, **`ngram_range=(1,2)`**
3. **Naive Bayes** on counts and **logistic regression** on TF-IDF are strong baselines (>85% on clean data)
4. **Inspect top weighted terms** to validate model sanity
5. **Error analysis** on misclassified text beats staring at accuracy alone
6. **Pipelines** prevent leaking vocabulary across train/test

---

## Check Your Understanding

1. Why does Prosise use `ngram_range=(1,2)` for reviews?
2. What is the difference between NB (counts) and logistic (TF-IDF) inputs?
3. How would you investigate a model that confuses sarcastic positives?
4. When would you lower the classification threshold below 0.5?
5. What does `min_df=5` remove from the vocabulary?

---

## References

- Prosise, Ch. 4 - sentiment analysis case study
- Maas et al. (2011) - Learning Word Vectors for Sentiment Analysis: [http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)
- IMDB dataset: [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)
- Scikit-Learn text classification tutorial: [https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

---

**Previous:** [Section 4.4 - Naive Bayes for Text](./section-04-naive-bayes-for-text.md)  
**Next:** [Section 4.6 - Spam Filtering](./section-06-spam-filtering.md)