# Section 13.3: Embedding Layers

> **Source:** Prosise, Ch. 13 - Keras Embedding and pretrained vectors  
> **Prerequisites:** [Sections 13.1-13.2](./section-01-beyond-bag-of-words.md)  
> **Glossary:** [tokenization](../../GLOSSARY.md#tokenization) | [neural-network](../../GLOSSARY.md#neural-network) | [deep-learning](../../GLOSSARY.md#deep-learning)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Lookup Tables That Learn Meaning

The Keras **`Embedding`** layer is a trainable lookup table: integer token index $i$ maps to vector $\mathbf{e}_i \in \mathbb{R}^{d}$.

$$
\mathbf{E} \in \mathbb{R}^{V \times d}, \quad \mathbf{e}_i = \mathbf{E}[i,:]
$$
> **Readable form:** embedding matrix E has shape vocabulary-size by embedding-dimension; row i is the vector for token i

Input shape `(batch, sequence_length)` → output `(batch, sequence_length, embedding_dim)`.

> **In plain English:** Each word ID becomes a vector of learned numbers - similar words drift together during training.

---

## Basic Embedding Layer

```python
import tensorflow as tf
from tensorflow.keras import layers, models

max_features = 10000
embedding_dim = 128
sequence_length = 200

model = models.Sequential([
    layers.Embedding(
        input_dim=max_features,      # vocabulary size
        output_dim=embedding_dim,    # vector dimension d
        input_length=sequence_length,  # optional; deprecated in TF 2.x for dynamic
    ),
    layers.GlobalAveragePooling1D(),  # mean over sequence → fixed vector
    layers.Dense(1, activation='sigmoid'),
])
```

**GlobalAveragePooling1D** aggregates sequence embeddings for classification - simpler than LSTM, weaker on order-sensitive tasks.

---

## Training Embeddings End-to-End

Embeddings learn from **downstream task loss** (sentiment, translation):

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
    x_train_seq, y_train,
    validation_data=(x_val_seq, y_val),
    epochs=10,
    batch_size=32,
)
```

Gradients backpropagate through embedding rows - frequent sentiment words ("great", "terrible") move apart in vector space.

---

## Embedding Matrix Inspection

```python
embedding_layer = model.layers[0]
weights = embedding_layer.get_weights()[0]  # shape (max_features, embedding_dim)
print(weights.shape)

# Nearest neighbors in embedding space (cosine similarity)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

word_idx = vectorize.get_vocabulary().index('great')
sims = cosine_similarity(weights[word_idx:word_idx+1], weights)[0]
top = sims.argsort()[-6:][::-1]
vocab = vectorize.get_vocabulary()
print([vocab[i] for i in top])
```

Visualize with t-SNE in [Lab 13](./section-lab-13-nlp-progression-sentiment-to-translation.md) (optional).

---

## Pretrained Embeddings: GloVe and Word2Vec

**Transfer learning for words** - initialize $\mathbf{E}$ from GloVe/Word2Vec trained on billions of tokens:

```python
import numpy as np

def load_glove_matrix(glove_path, vectorize_layer):
    vocab = vectorize_layer.get_vocabulary()
    embedding_dim = 100  # GloVe 100d
    matrix = np.zeros((len(vocab), embedding_dim))
    
    embeddings_index = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    for i, word in enumerate(vocab):
        vec = embeddings_index.get(word)
        if vec is not None:
            matrix[i] = vec
    return matrix

embedding_matrix = load_glove_matrix('glove.6B.100d.txt', vectorize_layer)

embedding_layer = layers.Embedding(
    input_dim=len(vectorize_layer.get_vocabulary()),
    output_dim=100,
    weights=[embedding_matrix],
    trainable=False,  # freeze or True for fine-tune
)
```

| Strategy | When |
|----------|------|
| `trainable=False` | Small labeled data - keep GloVe semantics |
| `trainable=True` | Domain-specific jargon adapts vectors |
| Random init | Large in-domain corpus, no OOV issues |

See [GloVe in glossary](../../GLOSSARY.md#glove).

---

## Embedding Dimension Tradeoffs

| $d$ | Effect |
|-----|--------|
| 50-100 | Fast, risk underfitting complex semantics |
| 128-256 | Common default for LSTM classifiers |
| 300 | GloVe standard |
| 768+ | BERT hidden size - [Section 13.7](./section-07-bert-and-transfer-learning-for-nlp.md) |

Rule of thumb: $d \approx \sqrt[4]{V}$ is a heuristic starting point; tune on validation loss.

---

## Padding Index and Masking

Padding token (index 0) should not contribute to pooling:

```python
model = models.Sequential([
    layers.Embedding(max_features, 128, mask_zero=True),
    layers.LSTM(64),  # respects mask - skips padded timesteps
    layers.Dense(1, activation='sigmoid'),
])
```

Without `mask_zero=True`, LSTM processes padding zeros as meaningful input.

---

## Embedding + LSTM Stack (Preview)

Full sentiment architecture ([Section 13.4](./section-04-recurrent-neural-networks.md)):

```python
model = models.Sequential([
    layers.Embedding(max_features, 128, mask_zero=True),
    layers.LSTM(64, dropout=0.2),
    layers.Dense(1, activation='sigmoid'),
])
```

Embedding provides continuous input; LSTM models **order** across the sequence.

---

## Word Analogies (Word2Vec Property)

If embeddings capture linear structure:

$$
\mathbf{e}_{\text{king}} - \mathbf{e}_{\text{man}} + \mathbf{e}_{\text{woman}} \approx \mathbf{e}_{\text{queen}}
$$
> **Readable form:** king vector minus man vector plus woman vector ≈ queen vector

Learned embeddings on small IMDB may show weaker analogies than GloVe - scale of pretraining matters.

---

## Connection to Transformers

Transformer models embed tokens similarly but add **positional encoding** and **contextual** updates - token vector depends on full sentence, not fixed lookup row. Static `Embedding` layer is the foundation BERT builds upon.

---

## Key Takeaways

1. **`Embedding`** maps token indices to dense vectors $\mathbf{e}_i \in \mathbb{R}^d$
2. Embeddings **train end-to-end** from task loss or initialize from GloVe/Word2Vec
3. **`mask_zero=True`** prevents padding from polluting sequence models
4. Dimension $d$ trades capacity vs overfitting and compute
5. Pretrained vectors help **small labeled datasets**; random init needs more data

---

## Check Your Understanding

1. What shape does Embedding output for input (32, 200)?
2. Why freeze vs fine-tune pretrained GloVe weights?
3. What does `mask_zero=True` accomplish?
4. How does GlobalAveragePooling1D differ from LSTM for sequence aggregation?
5. Why might Word2Vec analogies fail on small-domain trained embeddings?

---

## References

- Prosise, Ch. 13
- Pennington et al., GloVe - [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
- Keras Embedding - [https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding)

---

**Next:** [Section 13.4 - Recurrent Neural Networks](./section-04-recurrent-neural-networks.md)
