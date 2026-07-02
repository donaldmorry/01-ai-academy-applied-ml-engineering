# Lab 13: NLP Progression - Sentiment to Translation

> **Prerequisites:** Sections [13.1](./section-01-beyond-bag-of-words.md)-[13.8](./section-08-nlp-model-selection-guide.md)  
> **Estimated time:** 6-8 hours  
> **Glossary:** [../../GLOSSARY.md](../../GLOSSARY.md) | **Math:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Objectives

By the end of this lab you will:

1. Train **Embedding + LSTM** sentiment on IMDB - beat Chapter 04 TF-IDF baseline
2. Build a **mini encoder-decoder NMT** model; evaluate with **BLEU**
3. **Fine-tune DistilBERT** on classification; compare accuracy and training time
4. Produce a **metric comparison table** and written architecture analysis

> **Humorous briefing:** Three eras of NLP walk into a notebook. Only one gets deployed - your analysis decides which.

---

## Setup

```python
# pip install tensorflow transformers datasets torch sacrebleu scikit-learn pandas matplotlib
import numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path

RANDOM_STATE = 42; tf.random.set_seed(RANDOM_STATE)
OUT_DIR = Path('lab13_output'); OUT_DIR.mkdir(exist_ok=True)
```

GPU recommended for Part C; Parts A-B runnable on CPU with patience.

---

## Part A: Embedding + LSTM Sentiment (120 min)

*Follow [Sections 13.2-13.4](./section-02-text-preprocessing-with-keras.md).*

### A1. Load IMDB (15 min)

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

max_features = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)
```

### A2. Chapter 04 baseline (20 min)

Reconstruct word indices as strings for TF-IDF (or use preprocessed CSV):

```python
# Quick baseline on subset for speed - expand for full comparison
word_index = imdb.get_word_index()
reverse_index = {v+3: k for k, v in word_index.items()}
reverse_index[0], reverse_index[1], reverse_index[2] = '<pad>', '<start>', '<unk>'

def decode(seq):
    return ' '.join(reverse_index.get(i, '?') for i in seq if i > 2)

X_text = [decode(s) for s in x_train[:5000]]
# ... train_test_split, TfidfVectorizer, LogisticRegression
# Record baseline_accuracy
```

**Target:** Record baseline accuracy for comparison table.

### A3. LSTM model (60 min)

```python
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

model = models.Sequential([
    layers.Embedding(max_features, 128, mask_zero=True),
    layers.LSTM(64, dropout=0.2),
    layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    x_train, y_train, validation_split=0.2, epochs=5, batch_size=128,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
)
lstm_acc = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"LSTM test accuracy: {lstm_acc:.3f}")
```

**Optional:** t-SNE on embedding weights for frequent words.

### A4. Error analysis (25 min)

Find 5 reviews LSTM misclassified but baseline got right (and vice versa). Hypothesize why.

---

## Part B: Mini Neural Machine Translation (150 min)

*Follow [Section 13.5](./section-05-sequence-to-sequence-models.md).*

### B1. Data (30 min)

Use English-French parallel corpus - [manythings.org](https://www.manythings.org/anki/) `fra-eng.zip` or Hugging Face:

```python
# pip install datasets
from datasets import load_dataset
ds = load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train[:10000]")
pairs = [(ex['translation']['en'], ex['translation']['fr']) for ex in ds]
```

Limit to **10K pairs** for lab runtime.

### B2. Vectorize source/target (30 min)

Two `TextVectorization` layers - `max_tokens=10000`, `output_sequence_length=30`.

### B3. Train encoder-decoder (60 min)

Build model from [Section 13.5](./section-05-sequence-to-sequence-models.md). Train 10-15 epochs with early stopping.

### B4. BLEU evaluation (30 min)

```python
import sacrebleu

test_pairs = pairs[-500:]  # held-out
hyps, refs = [], []
for en, fr in test_pairs[:50]:
    hyp = translate_greedy(...)  # your decode function
    hyps.append(detokenize(hyp))
    refs.append([fr])
bleu = sacrebleu.corpus_bleu(hyps, refs)
print(f"BLEU: {bleu.score:.1f}")
```

Show 5 example translations qualitatively.

---

## Part C: DistilBERT Fine-Tuning (120 min)

*Follow [Section 13.7](./section-07-bert-and-transfer-learning-for-nlp.md).*

### C1. Prepare data (20 min)

Use IMDB subset or AG News - binary/multiclass classification.

### C2. Fine-tune (60 min)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import time

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir='lab13_distilbert',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
)

t0 = time.perf_counter()
trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
bert_train_sec = time.perf_counter() - t0
bert_acc = ...  # evaluate on test
```

Use Hugging Face `Dataset` objects for `train_ds` and `val_ds`; if you start from Keras/TensorFlow tensors in Part A, convert the split before passing it to `Trainer`.

Record **training time** and **test accuracy**.

### C3. Compare (40 min)

Fill comparison table (Part D).

---

## Part D: Comparison Table & Analysis (60 min)

| Model | Test accuracy | Train time | Inference latency (ms) | Best when? |
|-------|---------------|------------|------------------------|------------|
| TF-IDF + LR | | | | |
| Embedding + LSTM | | | | |
| Encoder-decoder (BLEU) | N/A (BLEU= ) | | | |
| DistilBERT | | | | |

**Written analysis (300+ words):**
- Which model for **mobile spam filter** (latency-critical)?
- Which for **legal document sentiment** (accuracy-critical)?
- Which for **English→French product descriptions** (100K pairs)?

Reference [Section 13.8](./section-08-nlp-model-selection-guide.md).

---

## Deliverables Checklist

- [ ] Notebook with Parts A-C runnable end-to-end
- [ ] Comparison table with measured metrics
- [ ] Written architecture analysis
- [ ] 5 example translations + BLEU score
- [ ] Error analysis on LSTM vs baseline

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM on BERT | Reduce batch size; use DistilBERT; subset data |
| LSTM below baseline | Increase epochs; try Bidirectional LSTM |
| BLEU near 0 | Check tokenization; verify decoder teacher forcing setup |
| NMT gibberish | Reduce vocab; train longer; use shorter max length |

---

## References

- Prosise, Ch. 13
- [Chapter 04](../chapter-04-text-classification/README.md) - TF-IDF baseline
- Hugging Face NLP course - [https://huggingface.co/learn/nlp-course](https://huggingface.co/learn/nlp-course)

---

**Next chapter:** [Chapter 14 - Azure Cognitive Services](../chapter-14-azure-cognitive-services/README.md)
