# Section 4.4: Naive Bayes for Text

> **Source:** Prosise, Ch. 4 - Multinomial Naive Bayes, Bayes' theorem  
> **Prerequisites:** [Section 4.3](./section-03-bag-of-words-and-tf-idf.md) | [Section 3.2](../chapter-03-classification-models/section-02-logistic-regression.md)  
> **Glossary:** [naive-bayes](../../GLOSSARY.md#naive-bayes) | [bag-of-words](../../GLOSSARY.md#bag-of-words)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Why Naive Bayes Dominates Text Baselines

Ask a room of NLP engineers for a first spam or sentiment model - most will say **[Naive Bayes](../../GLOSSARY.md#naive-bayes)** before transformers, before BERT, before the coffee finishes brewing. It trains in seconds on sparse [bag-of-words](../../GLOSSARY.md#bag-of-words) matrices, handles thousands of features, and often reaches 85-90% accuracy on Prosise's movie reviews with minimal tuning.

The name sounds sophisticated. The idea is Bayes' theorem plus one outrageous simplifying assumption that somehow works.

> **Humorous analogy:** Naive Bayes is the chef who assumes every ingredient tastes independently - tomatoes don't affect basil, salt doesn't interact with sugar. The dish should be inedible. Yet at text classification potlucks, it keeps winning "good enough" ribbons.

> **In plain English:** Compute the probability the document belongs to each class given its word counts. Pick the highest. Train by counting words in labeled examples.

---

## Bayes' Theorem Refresher

We want the probability of class $C$ given document $D$:

$$
P(C \mid D) = \frac{P(D \mid C)\, P(C)}{P(D)}
$$
> **Readable form:** P(class given document) = (P(document given class) × P(class)) divided by P(document)

- $P(C)$ - **prior** - how common is spam in training data?
- $P(D \mid C)$ - **likelihood** - how probable is this word mixture if class is $C$?
- $P(D)$ - **evidence** - normalizing constant (same for all classes when comparing)
- $P(C \mid D)$ - **posterior** - what we want for [classification](../../GLOSSARY.md#classification)

For prediction, we only need:

$$
\hat{C} = \arg\max_C \, P(C \mid D) = \arg\max_C \, P(D \mid C)\, P(C)
$$
> **Readable form:** predicted class = the class C that maximizes P(document given C) times P(class C)

$P(D)$ cancels when comparing classes.

Connection to [GLOSSARY Bayes' Rule](../../GLOSSARY.md#bayes-rule) in Course 2 - same identity, applied engineering.

---

## The "Naive" Independence Assumption

A document $D$ is a sequence of words $w_1, w_2, \ldots, w_n$. The true joint likelihood is:

$$
P(D \mid C) = P(w_1, w_2, \ldots, w_n \mid C)
$$
> **Readable form:** The expression assigns probability to the event or value using the stated model assumptions.

Naive Bayes **assumes words are conditionally independent** given the class:

$$
P(w_1, \ldots, w_n \mid C) \approx \prod_{i=1}^{n} P(w_i \mid C)
$$
> **Readable form:** P(all words given class) ≈ product of P(each word given class) - each word's probability does not depend on other words

**This is false.** `"not"` and `"good"` are not independent. Yet the approximation works because:

1. We only need the **ranking** of classes, not exact probabilities
2. Word counts in high dimensions provide redundant evidence
3. Errors in joint probability estimates partially cancel in $\arg\max$

Same spirit as linear models assuming additive feature effects in [Section 3.2](../chapter-03-classification-models/section-02-logistic-regression.md) - wrong generative story, useful discriminative boundary.

---

## Multinomial Naive Bayes

For **count** data (bag-of-words), use `MultinomialNB`. It models word counts per class with a multinomial distribution.

Training estimates:

$$
\hat{P}(w_j \mid C) = \frac{\text{count of } w_j \text{ in class } C + \alpha}{\text{total words in class } C + \alpha |V|}
$$
> **Readable form:** estimated P(word j given class C) = (count of word j in class C plus smoothing alpha) divided by (total words in class C plus alpha times vocabulary size)

$\alpha$ = **Laplace smoothing** (default `alpha=1.0` in sklearn) - prevents zero probability for unseen words in a class.

Prior:

$$
\hat{P}(C) = \frac{\text{number of docs in class } C}{N}
$$
> **Readable form:** estimated P(class C) = (documents in class C) divided by (total documents)

---

## sklearn Implementation

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

reviews = [
    "loved the film brilliant acting",
    "worst movie ever waste of money",
    "fantastic story great performances",
    "boring predictable plot",
    "amazing cinematography loved it",
    "terrible acting awful script",
] * 200

labels = [1, 0, 1, 0, 1, 0] * 200  # 1=positive

X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.25, random_state=42, stratify=labels
)

vectorizer = CountVectorizer(stop_words='english', min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

model = MultinomialNB(alpha=1.0)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=['neg', 'pos']))
```

**Use `CountVectorizer`, not `TfidfVectorizer`** - MultinomialNB expects non-negative counts.

---

## Predictions, Probabilities, and Log Space

```python
test_doc = ["brilliant acting wonderful film"]
X_new = vectorizer.transform(test_doc)

print("Class:", model.predict(X_new))
print("Probabilities:", model.predict_proba(X_new))
print("Log proba:", model.predict_log_proba(X_new))
```

Internally sklearn works in **log space** to avoid numerical underflow when multiplying hundreds of tiny probabilities:

$$
\log P(C \mid D) \propto \log P(C) + \sum_i \log P(w_i \mid C)
$$
> **Readable form:** log P(class given document) is proportional to log P(class) plus the sum of log P(each word given class)

Products of probabilities underflow to zero on long documents; sums of logs do not.

---

## Inspecting Learned Word Likelihoods

```python
import numpy as np

feature_names = vectorizer.get_feature_names_out()
log_probs = model.feature_log_prob_   # shape: (n_classes, n_features)

for class_idx, class_name in enumerate(['negative', 'positive']):
    top_indices = np.argsort(log_probs[class_idx])[-8:]
    words = [feature_names[i] for i in top_indices]
    print(f"{class_name}: {words}")
```

Words with high log-probability in class "positive" are your learned sentiment lexicon - no manual dictionary required.

---

## Alpha Smoothing in Action

```python
from sklearn.naive_bayes import MultinomialNB

for alpha in [0.01, 1.0, 10.0]:
    m = MultinomialNB(alpha=alpha)
    m.fit(X_train_vec, y_train)
    acc = m.score(X_test_vec, y_test)
    print(f"alpha={alpha}: accuracy={acc:.3f}")
```

| $\alpha$ | Effect |
|----------|--------|
| Small | Sharp estimates; risk zeros without smoothing |
| 1.0 (default) | Laplace smoothing - good default |
| Large | Uniform pull; underfits word distinctions |

---

## Naive Bayes vs Logistic Regression

| Criterion | MultinomialNB | LogisticRegression |
|-----------|---------------|-------------------|
| Training speed | Very fast | Fast |
| Input | Word counts | TF-IDF or counts |
| Calibration | Often poor | Better probabilities |
| Interpretability | Per-class word rates | Feature weights |
| Non-linear boundaries | Limited | Linear in features |
| Small data | Excellent | Good |

Prosise often trains both on movie reviews - NB as quick baseline, logistic for slightly higher accuracy. Compare with [Section 3.2](../chapter-03-classification-models/section-02-logistic-regression.md) metrics from [Section 3.3](../chapter-03-classification-models/section-03-classification-accuracy-measures.md).

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english', min_df=2)
X_tr = tfidf.fit_transform(X_train)
X_te = tfidf.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_tr, y_train)
print("LR accuracy:", lr.score(X_te, y_test))
print("NB accuracy:", model.score(X_test_vec, y_test))
```

---

## Binary Text Classification Workflow (Prosise Pattern)

```
1. Load labeled text (reviews, emails)
2. train_test_split with stratify
3. CountVectorizer on train → sparse X
4. MultinomialNB.fit
5. classification_report on test
6. Inspect top words per class
7. Error analysis on misclassified docs
```

Error analysis - read false positives and false negatives:

```python
import pandas as pd

results = pd.DataFrame({
    'text': X_test,
    'true': y_test,
    'pred': y_pred,
})
errors = results[results['true'] != results['pred']]
print(errors.head(10))
```

Gold for finding sarcasm, negation failures, and OOV slang.

---

## When Naive Bayes Struggles

| Failure mode | Why |
|--------------|-----|
| `"not good"` | Independence misses negation without bigrams |
| Highly correlated features | Duplicate evidence overweighted |
| Very short texts | Few words → weak evidence |
| Severe domain shift | Word priors from 2005 email ≠ 2025 SMS |

Mitigations: `ngram_range=(1,2)` in vectorizer ([Section 4.3](./section-03-bag-of-words-and-tf-idf.md)), more training data, or upgrade to logistic/SVM.

---

## Spam Prior Sensitivity

If training data is 90% ham, the learned prior biases toward ham. Adjust with `class_prior` when deployment base rates differ ([Section 4.6](./section-06-spam-filtering.md)).

---

## Connection to Chapter 03

- [Section 3.1](../chapter-03-classification-models/section-01-classification-fundamentals.md) - NB is a generative classifier
- [Section 3.3](../chapter-03-classification-models/section-03-classification-accuracy-measures.md) - evaluate with precision/recall, not accuracy alone on spam
- [Section 3.7](../chapter-03-classification-models/section-07-case-study-credit-card-fraud-detection.md) - imbalance strategies parallel spam

---

## Key Takeaways

1. **[Naive Bayes](../../GLOSSARY.md#naive-bayes)** applies Bayes' theorem with word independence assumption
2. **MultinomialNB** pairs with `CountVectorizer` for text counts
3. **Training** = count words per class + smoothing ($\alpha$)
4. **Prediction** = class with highest posterior (computed in log space)
5. **Fast, strong baseline** - often within a few points of logistic regression
6. **Independence is false** but rankings are often correct enough

---

## Check Your Understanding

1. Write Bayes' theorem and identify prior, likelihood, and posterior.
2. What does the "naive" assumption claim?
3. Why use counts instead of TF-IDF with MultinomialNB?
4. What does `alpha=1.0` smoothing prevent?
5. Why does sklearn use log probabilities internally?

---

## References

- Prosise, Ch. 4 - Naive Bayes classifier
- Scikit-Learn `MultinomialNB`: [https://scikit-learn.org/stable/chapters/generated/sklearn.naive_bayes.MultinomialNB.html](https://scikit-learn.org/stable/chapters/generated/sklearn.naive_bayes.MultinomialNB.html)
- Manning et al. - *Introduction to Information Retrieval*, Ch. 13 (Naive Bayes text): [https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf](https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf)
- Rennie et al. (2003) - Tackling the poor assumptions of Naive Bayes text classifiers: [https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)

---

**Previous:** [Section 4.3 - Bag-of-Words & TF-IDF](./section-03-bag-of-words-and-tf-idf.md)  
**Next:** [Section 4.5 - Sentiment Analysis](./section-05-sentiment-analysis.md)



