# Section 4.1: Text as Data

> **Source:** Prosise, Ch. 4 - "Text Classification" (opening sections)  
> **Prerequisites:** [Chapter 03 - Classification Models](../chapter-03-classification-models/README.md), especially [Section 3.1](../chapter-03-classification-models/section-01-classification-fundamentals.md)  
> **Glossary:** [bag-of-words](../../GLOSSARY.md#bag-of-words) | [tokenization](../../GLOSSARY.md#tokenization) | [classification](../../GLOSSARY.md#classification)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Why Text Is Different from Tabular Data

In [Chapter 03](../chapter-03-classification-models/README.md), you classified passengers, credit cards, and handwritten digits - each row had a fixed number of numeric or encoded columns. **Text breaks that contract.** A movie review might be three words or three thousand. The same sentiment can be expressed a hundred different ways:

```
"This film was absolutely brilliant."
"Worst two hours of my life."
"meh"
```

All three are valid training examples for [classification](../../GLOSSARY.md#classification), but none fit neatly into a spreadsheet column without transformation.

> **Humorous analogy:** Tabular data is a hotel with numbered rooms - every guest gets room 42. Text is a music festival where people camp wherever they want. Machine learning algorithms are the hotel managers; they need you to assign everyone a room number first.

> **In plain English:** Before any classifier from [Section 3.2](../chapter-03-classification-models/section-02-logistic-regression.md) or [Section 3.5](../chapter-03-classification-models/section-05-tree-based-classifiers.md) can touch text, you must convert variable-length strings into fixed-length numerical [features](../../GLOSSARY.md#feature). That conversion - and everything that leads up to it - is the subject of this chapter.

---

## Unstructured vs Structured Data

| Property | Tabular (Titanic, fraud) | Text (reviews, email) |
|----------|--------------------------|------------------------|
| Feature count | Fixed per row | Grows with vocabulary |
| Missing values | Explicit NaN | Implicit (word not present) |
| Order | Column order fixed | Word order matters semantically |
| Scale | Millions of rows, dozens of columns | Millions of columns (words), thousands of rows |
| Noise | Outliers, typos in numbers | Slang, typos, emojis, sarcasm |

Prosise's Chapter 4 walks through three production-shaped problems built on the same foundation:

1. **Sentiment analysis** - positive vs negative movie reviews
2. **Spam filtering** - ham vs spam email/SMS
3. **Content-based recommendation** - find similar movies by plot text

Each problem is [supervised learning](../../GLOSSARY.md#supervised-learning) except the recommender, which uses similarity rather than labels. All three share the same preprocessing and vectorization pipeline you will build in Sections 4.2-4.3.

---

## Core Vocabulary

Understanding text ML starts with four terms every paper and API reuses:

### Corpus

The **entire collection** of documents you are working with - every review, email, or plot summary in your dataset.

```python
corpus = [
    "The movie was great.",
    "Terrible acting and plot.",
    "An okay evening at the cinema.",
]
print(len(corpus))  # 3 documents
```

### Document

One **instance** in the corpus - a single review, one email, one movie description. In Scikit-Learn, documents are usually strings in a Python list or a Pandas `Series`.

### Token

A **unit of text** after splitting - typically a word or punctuation chunk. The sentence `"Don't stop!"` might tokenize to `["do", "n't", "stop", "!"]` depending on your tokenizer. See [tokenization](../../GLOSSARY.md#tokenization) in [Section 4.2](./section-02-text-preprocessing.md).

### Vocabulary

The **set of unique tokens** the model knows about. If your corpus contains only `"cat"`, `"dog"`, and `"fish"`, the vocabulary has size 3. Real corpora have vocabularies from thousands to millions of tokens.

> **In plain English:** Think of the vocabulary as the dictionary your model is allowed to use. Words not in the dictionary are invisible at prediction time - a classic source of "my model missed obvious spam" bugs.

---

## The Text Classification Pipeline (Preview)

Prosise's Chapter 4 follows a repeatable pipeline. You will implement every stage in this chapter:

```
Raw text  →  Preprocess  →  Vectorize  →  Classify  →  Evaluate
   │              │              │             │            │
   │         tokenize       bag-of-words    Naive Bayes   precision
   │         stop words     or TF-IDF       or logistic   recall
   │         stem           n-grams         regression    F1
   ▼              ▼              ▼             ▼            ▼
 strings      clean tokens   sparse matrix   labels      metrics
```

This mirrors the `Pipeline` pattern from [Section 3.4](../chapter-03-classification-models/section-04-categorical-data-and-feature-encoding.md), but the transformer is a **vectorizer** instead of `ColumnTransformer`. [Section 4.8](./section-08-pipelines-and-production-text-ml.md) wires it for production.

---

## Why Not Feed Raw Strings to sklearn?

Scikit-Learn estimators expect a 2D array of shape $(n_{\text{samples}}, n_{\text{features}})$ - see [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md). A string is not a number. Even if you hashed each character to ASCII, you would destroy word-level meaning and produce brittle features.

The standard approach is the **[bag-of-words](../../GLOSSARY.md#bag-of-words)** model: represent each document as a vector counting how often each vocabulary word appears. Word order is discarded - controversial, but remarkably effective for spam and sentiment.

$$
\mathbf{x}_d = (c_{d,1}, c_{d,2}, \ldots, c_{d,V})
$$
> **Readable form:** document vector x = (count of word 1 in doc, count of word 2, …, count of word V in vocabulary)

Document $d$, vocabulary size $V$, count $c_{d,i}$ = times word $i$ appears in document $d$.

> **In plain English:** Each review becomes a row of numbers - one column per word in the dictionary. Most entries are zero because most words do not appear in any given review. That sparsity is normal and handled efficiently by sparse matrices.

---

## A Minimal End-to-End Sketch

Here is the entire Chapter 4 idea in twenty lines - you will unpack each piece in later sections:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

reviews = [
    "Loved it, fantastic film",
    "Awful waste of time",
    "Brilliant performances throughout",
    "Boring and predictable",
]
labels = [1, 0, 1, 0]  # 1 = positive, 0 = negative

X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.25, random_state=42
)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

print(classification_report(y_test, y_pred))
print("Vocabulary:", vectorizer.get_feature_names_out())
```

**Observe:**
- `fit_transform` on training data **learns** the vocabulary
- `transform` on test data uses the **same** vocabulary - critical for honest evaluation, just like fitting encoders only on train in [Section 3.4](../chapter-03-classification-models/section-04-categorical-data-and-feature-encoding.md)
- `MultinomialNB` expects non-negative counts - perfect for bag-of-words

---

## Challenges Specific to Text

### Variable length

Documents have different lengths. Vectorization fixes the dimensionality to vocabulary size $|V|$, not document length. Longer documents tend to have larger counts - [TF-IDF](./section-03-bag-of-words-and-tf-idf.md) partially corrects this.

### Out-of-vocabulary (OOV) words

At prediction time, words never seen during `fit` are dropped silently:

```python
vectorizer = CountVectorizer()
vectorizer.fit(["hello world"])
X = vectorizer.transform(["hello zzyzx"])
print(X.toarray())  # [[1 0]] - "zzyzx" ignored
```

Production systems need monitoring for OOV rate spikes (new slang, new product names).

### Class imbalance

Spam corpora often have far more ham than spam - sound familiar from [Section 3.7](../chapter-03-classification-models/section-07-case-study-credit-card-fraud-detection.md)? The same [precision and recall](../../GLOSSARY.md#accuracy) tradeoffs apply; [Section 4.6](./section-06-spam-filtering.md) optimizes precision.

### Domain shift

A model trained on 1990s movie reviews may fail on tweet-length modern slang. Retraining and vocabulary refresh are operational necessities, not optional polish.

### Semantics and negation

Bag-of-words treats `"not good"` as presence of `"not"` and `"good"` - potentially positive-ish counts. N-grams (pairs of adjacent words) help; [Section 4.3](./section-03-bag-of-words-and-tf-idf.md) covers `ngram_range`.

---

## Text as a High-Dimensional Sparse Problem

Suppose $|V| = 50{,}000$ words and you have 5,000 documents. Your feature matrix is $5000 \times 50000$ - **250 million** cells - but each short review might activate only 50-200 of them.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["the cat sat on the mat", "the dog sat on the log"]
vec = CountVectorizer()
X = vec.fit_transform(corpus)

print(X.shape)           # (2, 8) - 2 docs, 8 unique tokens
print(X.nnz)             # number of non-zero entries
print(X.toarray())
# [[1 1 1 1 1 0 0 0]
#  [1 0 1 1 0 1 1 1]]
```

Sparse storage (`scipy.sparse`) keeps memory manageable. This is why Naive Bayes and linear models dominate text - they scale well on sparse inputs, unlike dense k-NN on MNIST from [Section 3.8](../chapter-03-classification-models/section-08-case-study-mnist-digit-recognition.md).

---

## Supervised Text Classification Tasks

| Task | Labels | Prosise example |
|------|--------|-----------------|
| Binary sentiment | positive / negative | Movie reviews |
| Binary spam | spam / ham | Email, SMS |
| Multi-class topic | politics / sports / tech | News headlines |
| Multi-label tagging | multiple genres per film | Metadata (advanced) |

Chapter 4 focuses on **binary** problems - the same metrics from [Section 3.3](../chapter-03-classification-models/section-03-classification-accuracy-measures.md) apply unchanged.

---

## Loading Real Text Data

Prosise uses CSV files with a text column and a label column. Pattern:

```python
import pandas as pd

# Typical layout after download
# df = pd.read_csv('reviews.csv')
# df.columns → ['review', 'sentiment']

df = pd.DataFrame({
    'review': [
        'A wonderful story with heart',
        'Plot holes everywhere',
        'Best film this year',
    ],
    'sentiment': ['positive', 'negative', 'positive'],
})

X = df['review'].astype(str)   # ensure strings, handle NaN
y = (df['sentiment'] == 'positive').astype(int)

print(X.head())
print(y.value_counts())
```

**Always cast to string** - mixed types break vectorizers. **Inspect class balance** before celebrating accuracy.

---

## Content-Based Recommendation (Not Classification, Same Vectors)

The third Chapter 4 application finds movies with similar plot descriptions using **cosine similarity** on TF-IDF vectors - no labels required. [Section 4.7](./section-07-cosine-similarity-and-recommender-systems.md) treats this as retrieval in vector space rather than [classification](../../GLOSSARY.md#classification).

> **Humorous analogy:** Classification is a bouncer at the door ("positive or negative?"). Cosine similarity is a librarian who hands you books shelved near the one you liked.

---

## What Classical Text ML Cannot Do

Bag-of-words + Naive Bayes is a strong baseline but blind to:

- **Long-range dependencies:** "The movie, after a slow first hour, eventually …" 
- **Coreference:** "She convinced him the ending was fake"
- **Sarcasm:** "Oh sure, *another* brilliant sequel"
- **Multilingual mixing** without language-specific tokenizers

[Chapter 13](../chapter-13-natural-language-processing/README.md) addresses these with embeddings and transformers. Master Chapter 4 first - TF-IDF + linear models remain production workhorses for spam, routing, and intent detection.

---

## Connection to Prior Chapters

| Prior section | Connection to text |
|--------------|------------------|
| [3.1 Classification fundamentals](../chapter-03-classification-models/section-01-classification-fundamentals.md) | Text labels are still discrete classes |
| [3.3 Accuracy measures](../chapter-03-classification-models/section-03-classification-accuracy-measures.md) | Precision/recall for spam and sentiment |
| [3.4 Categorical encoding](../chapter-03-classification-models/section-04-categorical-data-and-feature-encoding.md) | Vectorizer is a learned encoder for text |
| [3.7 Credit card fraud](../chapter-03-classification-models/section-07-case-study-credit-card-fraud-detection.md) | Imbalance playbook repeats for spam |

---

## Key Takeaways

1. **Text is unstructured** - variable length, huge implicit feature spaces, heavy sparsity
2. **Corpus, document, token, vocabulary** are the core nouns
3. **Vectorization** converts strings to numeric sparse matrices algorithms can consume
4. **Bag-of-words** discards order but enables fast, interpretable baselines
5. **Train/test discipline** applies to vocabulary learning exactly as to scalers and encoders
6. **Chapter 4's three apps** share one pipeline with different objectives at the end

---

## Check Your Understanding

1. What is the difference between a corpus and a document?
2. Why does `CountVectorizer` need separate `fit_transform` (train) and `transform` (test)?
3. What happens to a word that appears in test data but not training data?
4. Why is accuracy alone dangerous for spam detection? (Hint: [Section 3.3](../chapter-03-classification-models/section-03-classification-accuracy-measures.md))
5. What information does bag-of-words discard?

---

## References

- Prosise, J. *Applied Machine Learning and AI for Engineers* (O'Reilly, 2023), Chapter 4
- Scikit-Learn - Text feature extraction: [https://scikit-learn.org/stable/chapters/feature_extraction.html#text-feature-extraction](https://scikit-learn.org/stable/chapters/feature_extraction.html#text-feature-extraction)
- Jurafsky & Martin - Speech and Language Processing (3rd ed. draft), Ch. 2-4: [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)
- NLTK Book, Ch. 1 - Language Processing and Python: [https://www.nltk.org/book/ch01.html](https://www.nltk.org/book/ch01.html)

---

**Next:** [Section 4.2 - Text Preprocessing](./section-02-text-preprocessing.md)



