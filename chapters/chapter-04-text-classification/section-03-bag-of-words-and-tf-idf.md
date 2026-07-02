# Section 4.3: Bag-of-Words & TF-IDF

> **Source:** Prosise, Ch. 4 - vector space model, `CountVectorizer`, `TfidfVectorizer`  
> **Prerequisites:** [Section 4.2](./section-02-text-preprocessing.md) | [Section 3.1](../chapter-03-classification-models/section-01-classification-fundamentals.md)  
> **Glossary:** [bag-of-words](../../GLOSSARY.md#bag-of-words) | [tf-idf](../../GLOSSARY.md#tf-idf)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## From Words to Numbers

[Section 4.1](./section-01-text-as-data.md) established that classifiers need fixed-length numeric [features](../../GLOSSARY.md#feature). The **[bag-of-words](../../GLOSSARY.md#bag-of-words)** (BoW) model is the workhorse: each document becomes a vector of word counts, ignoring grammar and order.

Prosise builds every Chapter 4 application on BoW or its weighted cousin **[TF-IDF](../../GLOSSARY.md#tf-idf)**. Scikit-Learn implements both via `CountVectorizer` and `TfidfVectorizer`.

> **Humorous analogy:** Bag-of-words is a recipe review where you list ingredient counts but throw away the instructions. `"2 eggs, 1 cup flour"` - you know what's in the dish, not whether it's cake or pasta.

> **In plain English:** You count how many times each word in your vocabulary appears in each document. The result is a sparse matrix ready for [Naive Bayes](./section-04-naive-bayes-for-text.md), logistic regression, or SVM.

---

## Bag-of-Words Step by Step

**Corpus:**

```
Doc 1: "the cat sat"
Doc 2: "the dog sat"
Doc 3: "cat dog cat"
```

**Vocabulary:** `{the, cat, sat, dog}` - 4 features

**Count matrix** $X$ where $X_{d,w}$ = count of word $w$ in document $d$:

|       | the | cat | sat | dog |
|-------|-----|-----|-----|-----|
| Doc 1 |  1  |  1  |  1  |  0  |
| Doc 2 |  1  |  0  |  1  |  1  |
| Doc 3 |  0  |  2  |  0  |  1  |

$$
X \in \mathbb{R}^{n \times |V|}
$$
> **Readable form:** X is a matrix with n rows (documents) and |V| columns (vocabulary size)

---

## CountVectorizer in Practice

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are great",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print("Shape:", X.shape)                    # (3, 11)
print("Feature names:", vectorizer.get_feature_names_out())
print("Dense view:\n", X.toarray())
```

**Key methods:**

| Method | When |
|--------|------|
| `fit(corpus)` | Learn vocabulary from training docs |
| `transform(corpus)` | Convert to counts using learned vocab |
| `fit_transform(corpus)` | Both - training only |
| `get_feature_names_out()` | Map column index → word |

Never `fit_transform` on test data - same leakage rule as [Section 3.4](../chapter-03-classification-models/section-04-categorical-data-and-feature-encoding.md).

---

## Important CountVectorizer Parameters

### `min_df` and `max_df`

Filter terms by **document frequency** - fraction of docs containing the term:

```python
vec = CountVectorizer(min_df=2, max_df=0.8)
```

| Parameter | Effect |
|-----------|--------|
| `min_df=2` | Drop words appearing in fewer than 2 documents (removes rare typos) |
| `max_df=0.9` | Drop words in more than 90% of docs (near-universal stop words) |
| `min_df=0.01` | Keep words in at least 1% of docs (large corpora) |

Prosise uses `min_df` to prune huge vocabularies on movie reviews - critical for speed and generalization.

### `max_features`

Cap vocabulary at top-N most frequent terms:

```python
vec = CountVectorizer(max_features=5000)
```

Useful when memory is tight; risks dropping rare but discriminative spam phrases.

### `ngram_range`

Capture multi-word phrases:

```python
vec = CountVectorizer(ngram_range=(1, 2))  # unigrams + bigrams
```

`"not good"` becomes a single feature. Essential for sentiment in [Section 4.5](./section-05-sentiment-analysis.md).

### `stop_words`

```python
vec = CountVectorizer(stop_words='english')
```

See [Section 4.2](./section-02-text-preprocessing.md).

---

## N-Grams: Sneaking Order Back In

An **n-gram** is a contiguous sequence of $n$ tokens.

| n | Name | Example from `"the cat sat"` |
|---|------|------------------------------|
| 1 | unigram | `the`, `cat`, `sat` |
| 2 | bigram | `the cat`, `cat sat` |
| 3 | trigram | `the cat sat` |

```python
vec = CountVectorizer(ngram_range=(1, 2))
vec.fit(["the movie was not good"])  # includes 'not good' as one feature
```

**Tradeoff:** `ngram_range=(1,3)` on large corpora explodes $|V|$.

---

## The Problem with Raw Counts

Long documents have more words → larger counts → classifiers may bias toward document length, not content.

Word `"the"` appears everywhere → huge counts but zero discrimination.

**TF-IDF** fixes both intuitions.

---

## Term Frequency (TF)

**Term frequency** measures how often term $t$ appears in document $d$. Common variant with normalization:

$$
\text{tf}(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total terms in } d}
$$
> **Readable form:** tf(t, d) = (count of term t in document d) divided by (total terms in document d)

Or simply raw count - sklearn's default before IDF weighting.

---

## Inverse Document Frequency (IDF)

**Inverse document frequency** downweights terms that appear in many documents:

$$
\text{idf}(t) = \log \frac{N}{1 + \text{df}(t)} + 1
$$
> **Readable form:** idf(t) = log of (N divided by (1 + df(t))) plus 1, where N = total documents and df(t) = number of documents containing term t

The $+1$ in denominator and numerator smoothing prevents division by zero - sklearn's smoothed variant.

Rare discriminative words get high IDF. Ubiquitous words (`"the"`) get low IDF.

---

## TF-IDF Combined

$$
\text{tf-idf}(t, d) = \text{tf}(t, d) \times \text{idf}(t)
$$
> **Readable form:** tf-idf(term, document) = term frequency in that document times inverse document frequency of that term

Document vector in TF-IDF space:

$$
\mathbf{v}_d = (\text{tf-idf}(t_1, d), \ldots, \text{tf-idf}(t_{|V|}, d))
$$
> **Readable form:** document vector = (tf-idf for word 1, tf-idf for word 2, …, tf-idf for each word in vocabulary)

---

## TfidfVectorizer

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "great movie great acting",
    "terrible movie terrible plot",
    "great plot great story",
]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus)
print(tfidf.get_feature_names_out())
print(X.toarray())
```

`TfidfVectorizer` = `CountVectorizer` + TF-IDF normalization in one step.

---

## Why TF-IDF for Classification?

| Aspect | Raw counts | TF-IDF |
|--------|-----------|--------|
| Long document bias | Strong | Reduced via TF normalization |
| Common words | Dominate | Downweighted by IDF |
| Rare spam cues | May be rare counts | Boosted by high IDF |
| Naive Bayes | Works (counts) | Use counts, not TF-IDF* |
| Logistic / SVM | Works | Often better |

\*Multinomial Naive Bayes expects integer counts - use `CountVectorizer` for NB, `TfidfVectorizer` for linear models. Prosise demonstrates both.

---

## Prosise Movie Review Vectorization Pattern

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# df: columns 'review', 'sentiment'
# X_text = df['review'].fillna('')
# y = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    stop_words='english',
    min_df=5,
    ngram_range=(1, 2),
    max_features=20000,
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print(f"Training matrix: {X_train_vec.shape}")
print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
```

`stratify=y` preserves class balance per [Section 3.3](../chapter-03-classification-models/section-03-classification-accuracy-measures.md).

---

## Common Pitfalls

1. **Fitting on full dataset** before split → inflated metrics
2. **`min_df` too high** → loses rare spam indicators
3. **`max_features` too low** → drops signal
4. **TF-IDF with MultinomialNB** → technically suboptimal; use counts for NB
5. **Forgetting `str`** → `NaN` breaks vectorizer

---

## Key Takeaways

1. **[Bag-of-words](../../GLOSSARY.md#bag-of-words)** = count vectors; order discarded
2. **`CountVectorizer`** learns vocabulary and produces sparse count matrices
3. **[TF-IDF](../../GLOSSARY.md#tf-idf)** weights terms by importance in document vs corpus
4. **`min_df`, `max_df`, `max_features`** control vocabulary size and noise
5. **`ngram_range=(1,2)`** captures phrases like `"not good"` for sentiment
6. **Fit on train only** - non-negotiable for honest evaluation

---

## Check Your Understanding

1. What is the shape of the matrix produced by vectorizing $n$ documents with vocabulary size $|V|$?
2. Why does `min_df=5` help on a 50,000-review corpus?
3. How does TF-IDF downweight the word `"the"`?
4. When should you prefer counts over TF-IDF?
5. What does `ngram_range=(1,2)` add that unigrams miss?

---

## References

- Prosise, Ch. 4 - bag of words and TF-IDF
- Scikit-Learn - TF-IDF: [https://scikit-learn.org/stable/chapters/feature_extraction.html#tfidf-term-weighting](https://scikit-learn.org/stable/chapters/feature_extraction.html#tfidf-term-weighting)
- `TfidfVectorizer` API: [https://scikit-learn.org/stable/chapters/generated/sklearn.feature_extraction.text.TfidfVectorizer.html](https://scikit-learn.org/stable/chapters/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- Manning, Raghavan & Schütze - *Introduction to Information Retrieval*, Ch. 6: [https://nlp.stanford.edu/IR-book/](https://nlp.stanford.edu/IR-book/)

---

**Previous:** [Section 4.2 - Text Preprocessing](./section-02-text-preprocessing.md)  
**Next:** [Section 4.4 - Naive Bayes for Text](./section-04-naive-bayes-for-text.md)



