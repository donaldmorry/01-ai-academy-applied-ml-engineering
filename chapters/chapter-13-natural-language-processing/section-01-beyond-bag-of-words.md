# Section 13.1: Beyond Bag-of-Words

> **Source:** Prosise, Ch. 13 - limitations of TF-IDF, distributional semantics, word embeddings  
> **Prerequisites:** [Chapter 04 - Text Classification](../chapter-04-text-classification/README.md) | [Chapter 09 - Neural Networks](../chapter-09-neural-networks/README.md)  
> **Glossary:** [feature](../../GLOSSARY.md#feature) | [representation-learning](../../GLOSSARY.md#representation-learning) | [deep-learning](../../GLOSSARY.md#deep-learning)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## What Chapter 04 Gave You

[Chapter 04](../chapter-04-text-classification/README.md) treated text as **bag-of-words (BoW)** or **TF-IDF** vectors - word counts with optional weighting. That pipeline powered spam filters, sentiment classifiers, and cosine-similarity recommenders. It works surprisingly well for short documents and clear keyword signals.

Prosise Chapter 13 asks: what do we lose, and what do neural methods gain?

| Representation | Captures word order? | Captures semantics? | Sparse? |
|----------------|---------------------|-------------------|---------|
| BoW / TF-IDF | No | Weak (co-occurrence only) | Yes - huge vocab |
| Word embeddings | Partially (in sequence models) | Yes - dense vectors | No - fixed dim |
| Transformers | Yes - full context | Yes - contextual | Dense activations |

> **Humorous analogy:** BoW is alphabet soup - you know which letters are in the bowl, not whether they spell "excellent service" or "service excellent."

> **In plain English:** Neural NLP learns dense word vectors and (later) attention over full sentences - capturing meaning BoW cannot.

---

## Limitations of Bag-of-Words

Consider two reviews:

1. *"Not bad at all"*
2. *"Not good at all"*

BoW sees similar tokens (`not`, `bad`/`good`, `at`, `all`) - polarity can flip with negation **order**, which BoW discards.

Other BoW blind spots:

| Phenomenon | Example | BoW behavior |
|------------|---------|--------------|
| **Negation** | "not happy" vs "happy" | Partial overlap |
| **Synonyms** | "great" vs "excellent" | Unrelated dimensions |
| **Typos / morphology** | "running" vs "run" | Separate tokens unless stemmed |
| **Long-range dependency** | "The movie, after three hours, was boring" | All words bagged equally |
| **Vocabulary explosion** | 100k+ unique tokens | Sparse high-dimensional vectors |

---

## Distributional Semantics

**You shall know a word by the company it keeps** (Firth, 1957). Words appearing in similar contexts have similar meanings.

Neural **word embeddings** map each word $w$ to a dense vector $\mathbf{v}_w \in \mathbb{R}^m$ where $m \ll |V|$ (vocabulary size).

Classic result (Word2Vec):

$$
\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}
$$
> **Readable form:** vector arithmetic in embedding space captures analogies - king minus man plus woman approximates queen

---

## From Counts to Learned Vectors

**TF-IDF** weight for term $t$ in document $d$:

$$
\text{tfidf}(t, d) = \text{tf}(t, d) \times \log\frac{N}{\text{df}(t)}
$$
> **Readable form:** TF-IDF = term frequency in document times log of (corpus size / documents containing term)

**Embedding lookup** - token index $i$ maps to row $\mathbf{E}_i$ of matrix $\mathbf{E} \in \mathbb{R}^{|V| \times m}$:

$$
\mathbf{v}_w = \mathbf{E}_{\text{index}(w)}
$$
> **Readable form:** word vector = row of embedding matrix selected by word index - learned during training

The embedding matrix is a [representation learning](../../GLOSSARY.md#representation-learning) layer - weights updated by backpropagation on your task.

---

## Why Neural NLP Won State of the Art

| Era | Method | Strength |
|-----|--------|----------|
| 2000s | BoW + SVM/NB | Fast, interpretable baseline |
| 2013+ | Word2Vec/GloVe + RNN | Semantic vectors + sequence |
| 2017+ | Transformers | Parallel training, full context |
| 2018+ | BERT/GPT | Pretrained language understanding |

Prosise's Chapter 13 walks the ladder: embeddings → RNN/LSTM → seq2seq → transformers → BERT.

---

## When BoW Still Wins

Not every problem needs BERT:

| Scenario | Prefer BoW + linear model |
|----------|---------------------------|
| <1000 labeled documents | Yes - avoid overfitting |
| Keyword-driven task (spam) | Often sufficient |
| Sub-millisecond latency | TF-IDF + logistic regression |
| Interpretability required | Feature weights readable |
| No GPU, tight budget | Classical pipeline |

**Always benchmark** Chapter 04 baseline before deploying a transformer ([Section 13.8](./section-08-nlp-model-selection-guide.md)).

---

## Semantic Similarity Example

```python
# Conceptual - pretrained GloVe vectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Pretend 50-dim vectors for illustration
v_excellent = np.random.randn(50)
v_amazing = v_excellent + 0.1 * np.random.randn(50)
v_terrible = -v_excellent + 0.1 * np.random.randn(50)

def sim(a, b):
    return cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0,0]

print(f'excellent-amazing: {sim(v_excellent, v_amazing):.3f}')
print(f'excellent-terrible: {sim(v_excellent, v_terrible):.3f}')
```

Trained embeddings place synonyms closer than antonyms - TF-IDF treats them as orthogonal axes.

---

## Chapter Roadmap

| Section | Topic |
|--------|-------|
| 13.2 | Keras `TextVectorization` |
| 13.3 | `Embedding` layer |
| 13.4 | RNN / LSTM / GRU |
| 13.5 | Encoder-decoder NMT |
| 13.6 | Transformer architecture |
| 13.7 | BERT fine-tuning |
| 13.8 | Model selection guide |

---

## Connection to Chapter 14

Cloud **Azure Language** APIs ([Chapter 14](../chapter-14-azure-cognitive-services/README.md)) ship pretrained models - same build-vs-buy tradeoff as Custom Vision. Understanding embeddings and transformers helps you decide when to call an API vs fine-tune locally.

---

## Sparsity and Memory

A vocabulary of $|V| = 50{,}000$ words with BoW produces vectors of length 50,000 - mostly zeros. Storage for $N$ documents:

$$
\text{BoW dense storage} \approx N \times |V| \times 4 \text{ bytes (float32)}
$$
> **Readable form:** dense bag-of-words storage grows with number of documents times vocabulary size - impractical at web scale without sparse matrices

Embeddings compress each token to $m$ floats (e.g., 128-768), reused across documents via lookup - parameter count is $|V| \times m$, independent of corpus document count.

---

## Comparing Pipelines Side-by-Side

| Step | Chapter 04 (sklearn) | Chapter 13 (Keras) |
|------|---------------------|-------------------|
| Clean text | Regex, lowercase | `TextVectorization` |
| Vectorize | `TfidfVectorizer` | `Embedding` + sequence |
| Classify | `LogisticRegression` | LSTM / BERT head |
| Serialize | `joblib` pipeline | `SavedModel` |

Prosise's IMDB spam/sentiment examples appear in both chapters - rerun with identical train/test split when benchmarking.

---

## Subword Tokenization Preview

Word-level tokenizers struggle with rare words and typos. Modern systems (BERT WordPiece, GPT BPE) split into **subwords**:

- `"unhappiness"` → `["un", "##happiness"]`
- `"covid19"` → handled without OOV drop

[Section 13.7](./section-07-bert-and-transfer-learning-for-nlp.md) uses Hugging Face tokenizers with subword vocabularies - no manual `Tokenizer` needed.

---

## Historical Context

| Year | Milestone |
|------|-----------|
| 2013 | Word2Vec - static embeddings |
| 2014 | GloVe - global co-occurrence vectors |
| 2017 | Transformer - contextual attention |
| 2018 | BERT - bidirectional pretraining |

Prosise Chapter 13 spans this arc in one chapter - you are not behind if BoW still wins your baseline.

---

## Self-Check

1. Why does BoW fail on "not bad" vs "not good"?
2. What is distributional semantics in one sentence?
3. How does embedding dimension $m$ compare to vocabulary size $|V|$?
4. Name two cases where TF-IDF is still appropriate.
5. What famous vector analogy demonstrates semantic structure?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 13 - introduction
- [Chapter 04 - TF-IDF](../chapter-04-text-classification/section-03-bag-of-words-and-tf-idf.md)
- [Mikolov et al., Word2Vec, 2013](https://arxiv.org/abs/1301.3781)
- [Pennington et al., GloVe, 2014](https://nlp.stanford.edu/pubs/glove.pdf)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Chapter 12](../chapter-12-object-detection/README.md) | **Next:** [Section 13.2](./section-02-text-preprocessing-with-keras.md)
