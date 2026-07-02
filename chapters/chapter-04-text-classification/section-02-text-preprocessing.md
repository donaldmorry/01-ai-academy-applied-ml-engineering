# Section 4.2: Text Preprocessing

> **Source:** Prosise, Ch. 4 - tokenization, normalization, stop words, stemming  
> **Prerequisites:** [Section 4.1](./section-01-text-as-data.md) | [Section 3.4](../chapter-03-classification-models/section-04-categorical-data-and-feature-encoding.md)  
> **Glossary:** [tokenization](../../GLOSSARY.md#tokenization) | [bag-of-words](../../GLOSSARY.md#bag-of-words)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Garbage In, Garbage Out - Text Edition

Raw web text arrives messy: `"BEST MOVIE EVER!!!"`, `"u r gonna luv it"`, HTML tags, URLs, and emoji. Feeding that directly into a vectorizer creates duplicate features (`"Movie"`, `"movie"`, `"MOVIE"`) and noise words (`"the"`, `"a"`) that swamp signal.

**Preprocessing** is the cleaning step between raw strings and [bag-of-words](../../GLOSSARY.md#bag-of-words) vectors. Prosise demonstrates a practical pipeline: lowercase, [tokenization](../../GLOSSARY.md#tokenization), stop-word removal, and stemming.

> **Humorous analogy:** Preprocessing is like prepping vegetables before cooking. You would not throw a whole cabbage into soup - unless you enjoy crunching through outer leaves and rubber bands.

> **In plain English:** The goal is not literary perfection. The goal is to make `"running"` and `"runs"` count as related signals instead of unrelated columns, while dropping words that appear in every document and carry no class information.

---

## The Preprocessing Checklist

| Step | Purpose | Example |
|------|---------|---------|
| Lowercasing | Unify case variants | `Movie` → `movie` |
| Tokenization | Split into units | `"don't stop"` → tokens |
| Stop-word removal | Drop ultra-common words | remove `the`, `is`, `at` |
| Stemming / lemmatization | Reduce inflections | `running` → `run` |
| Optional: regex cleanup | Remove URLs, digits | strip `http://...` |

Not every task needs every step. Sentiment on social media might **keep** emoticons; legal document classification might **keep** numbers. Always validate on held-out data - same discipline as feature choices in [Section 3.6](../chapter-03-classification-models/section-06-case-study-titanic-survival.md).

---

## Tokenization

[Tokenization](../../GLOSSARY.md#tokenization) splits a character string into a list of tokens. The naive approach:

```python
text = "Machine learning is fun!"
tokens = text.lower().split()
print(tokens)
# ['machine', 'learning', 'is', 'fun!']
```

Problems: `"fun!"` still has punctuation attached; `"don't"` is one token; hyphenated words stay glued.

### NLTK word tokenizer

```python
import nltk
# nltk.download('punkt')       # run once
# nltk.download('punkt_tab')   # newer NLTK versions

from nltk.tokenize import word_tokenize

text = "Machine learning isn't easy - but it's rewarding."
tokens = word_tokenize(text.lower())
print(tokens)
# ['machine', 'learning', 'is', "n't", 'easy', '-', 'but', 'it', "'s", 'rewarding', '.']
```

NLTK handles contractions and punctuation more carefully than `.split()`.

### Regular expressions

```python
import re
tokens = re.findall(r'\b[a-z]{2,}\b', text.lower())  # word-boundary regex
```

### Scikit-Learn's built-in tokenizer

`CountVectorizer` tokenizes internally with `token_pattern`:

```python
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(token_pattern=r'(?u)\b\w\w+\b')  # default: 2+ char words
vec.fit(["hello world"])
print(vec.get_feature_names_out())  # ['hello' 'world']
```

You can pass a custom callable:

```python
def my_analyzer(text):
    return [t for t in text.lower().split() if len(t) > 2]

vec = CountVectorizer(analyzer=my_analyzer)
```

> **In plain English:** Tokenization defines what counts as a "word" for your model. Change the tokenizer, change the vocabulary, change the model.

---

## Normalization: Lowercasing

Almost always applied for English classification:

```python
corpus = ["The Film", "the film", "THE FILM"]
normalized = [doc.lower() for doc in corpus]
```

**When to skip:** Case-sensitive tasks (product codes, tickers), German nouns (capitalization carries meaning - rare in Prosise's examples).

---

## Stop Words

**Stop words** are extremely common tokens (`the`, `and`, `is`, `of`) that rarely help discrimination. They inflate vocabulary size and dilute TF-IDF scores.

```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

print(len(ENGLISH_STOP_WORDS))  # 318 built into sklearn
print(list(ENGLISH_STOP_WORDS)[:10])
```

### NLTK stop words

```python
import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
tokens = ['the', 'movie', 'was', 'great']
filtered = [t for t in tokens if t not in stop_words]
print(filtered)  # ['movie', 'great']
```

### With CountVectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(stop_words='english')
X = vec.fit_transform(["the movie was great", "the film was bad"])
print(vec.get_feature_names_out())
# ['bad' 'film' 'great' 'movie']  - 'the' and 'was' removed
```

Custom list: `CountVectorizer(stop_words=['the', 'a', 'movie'])`.

> **Humorous analogy:** Stop words are podcast filler - "um," "you know." Removing them helps you hear the argument.

**Caution:** For negation-heavy sentiment (`"not good"`), removing `"not"` is catastrophic. Prosise's sentiment model in [Section 4.5](./section-05-sentiment-analysis.md) often keeps stop words or uses bigrams (`ngram_range=(1,2)`) so `"not good"` survives as a phrase.

---

## Stemming

**Stemming** chops suffixes heuristically to map inflected forms to a common stem:

| Word | Porter stem |
|------|-------------|
| running | run |
| runs | run |
| runner | runner |
| happily | happili |

Not always pretty - linguists cringe; engineers celebrate smaller vocabularies.

### Porter stemmer (NLTK)

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ['running', 'runs', 'runner', 'easily', 'fairly']
print([stemmer.stem(w) for w in words])
# ['run', 'run', 'runner', 'easili', 'fairli']
```

### Plugging stemming into sklearn

```python
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def stem_tokenizer(text):
    return [stemmer.stem(t) for t in text.lower().split()]

vec = CountVectorizer(tokenizer=stem_tokenizer)
corpus = ["The runners are running quickly", "A runner won the race"]
X = vec.fit_transform(corpus)
print(vec.get_feature_names_out())
```

> **Note:** In sklearn ≥ 1.2, use `analyzer` instead of deprecated `tokenizer` parameter - same idea.

---

## A Complete Prosise-Style Preprocessor

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """Prosise Ch.4 pattern: lower, tokenize, drop stops, stem."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)          # letters only
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

samples = [
    "The acting was absolutely brilliant!",
    "Don't waste your money on this garbage film.",
]
for s in samples:
    print(f"IN:  {s}")
    print(f"OUT: {preprocess(s)}\n")
```

Output collapses inflections and strips noise - ready for [Section 4.3](./section-03-bag-of-words-and-tf-idf.md).

---

## Preprocess Before or Inside the Vectorizer?

Two valid patterns:

### Pattern A: Preprocess strings, then default vectorizer

```python
corpus_clean = [preprocess(doc) for doc in raw_corpus]
vec = CountVectorizer()
X = vec.fit_transform(corpus_clean)
```

Clear debugging - you can print cleaned text.

### Pattern B: Custom analyzer in vectorizer (Prosise production style)

```python
vec = CountVectorizer(analyzer=preprocess)
X = vec.fit_transform(raw_corpus)
```

Fits inside a `Pipeline` without a separate step - see [Section 4.8](./section-08-pipelines-and-production-text-ml.md).

**Critical rule:** Fit preprocessing statistics (if any) on training data only. Stop word lists are usually fixed; learned vocabularies are not.

---

## When Preprocessing Hurts

| Scenario | Risk |
|----------|------|
| Bigrams for negation | Stemming splits `"not"` handling |
| Named entities | Stemming mangles brand names |
| Short texts (tweets) | Aggressive stop removal leaves nothing |
| Multilingual corpora | English stemmer on French = chaos |

Always compare **with vs without** preprocessing in cross-validation - the winning recipe is empirical, not doctrinal.

---

## Connection to Chapter 03

[Section 3.4](../chapter-03-classification-models/section-04-categorical-data-and-feature-encoding.md) taught that encoders must `fit` on train only. Text preprocessing is mostly deterministic (fixed stop lists, fixed stemmer), but the **vectorizer vocabulary** is learned and must follow the same split discipline.

[Section 3.3](../chapter-03-classification-models/section-03-classification-accuracy-measures.md) metrics apply unchanged after preprocessing - cleaning does not fix imbalanced spam corpora.

---

## Key Takeaways

1. **[Tokenization](../../GLOSSARY.md#tokenization)** defines the atomic units of your model
2. **Lowercasing** merges case variants - default for English classification
3. **Stop words** reduce noise; use carefully when negation matters
4. **Stemming** shrinks vocabulary by normalizing inflections
5. **Custom analyzers** integrate preprocessing into sklearn vectorizers and pipelines
6. **Validate preprocessing choices** on dev data - no free lunch

---

## Check Your Understanding

1. What is the difference between tokenization and stemming?
2. Why might removing stop words hurt sentiment analysis?
3. How do you plug a custom preprocessor into `CountVectorizer`?
4. What happens if preprocessing removes every token from a document?
5. Why does sklearn provide `ENGLISH_STOP_WORDS` built in?

---

## References

- Prosise, Ch. 4 - text preprocessing sections
- NLTK documentation: [https://www.nltk.org/](https://www.nltk.org/)
- NLTK Book, Ch. 3 - Processing Raw Text: [https://www.nltk.org/book/ch03.html](https://www.nltk.org/book/ch03.html)
- Scikit-Learn `CountVectorizer`: [https://scikit-learn.org/stable/chapters/generated/sklearn.feature_extraction.text.CountVectorizer.html](https://scikit-learn.org/stable/chapters/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- Porter stemmer original paper: [https://tartarus.org/martin/PorterStemmer/](https://tartarus.org/martin/PorterStemmer/)

---

**Previous:** [Section 4.1 - Text as Data](./section-01-text-as-data.md)  
**Next:** [Section 4.3 - Bag-of-Words & TF-IDF](./section-03-bag-of-words-and-tf-idf.md)