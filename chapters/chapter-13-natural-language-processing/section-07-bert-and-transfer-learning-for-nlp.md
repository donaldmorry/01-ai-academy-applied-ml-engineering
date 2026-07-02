# Section 13.7: BERT & Transfer Learning for NLP

> **Source:** Prosise, Ch. 13 - BERT fine-tuning and Hugging Face  
> **Prerequisites:** [Sections 13.1-13.6](./section-01-beyond-bag-of-words.md)  
> **Glossary:** [deep-learning](../../GLOSSARY.md#deep-learning) | [neural-network](../../GLOSSARY.md#neural-network) | [representation-learning](../../GLOSSARY.md#representation-learning) | [backpropagation](../../GLOSSARY.md#backpropagation)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Pretrained Language Understanding

**BERT** (Bidirectional Encoder Representations from Transformers) pretrains a deep transformer on massive text, then **fine-tunes** on downstream tasks with modest labeled data - sentiment, NER, question answering.

> **In plain English:** BERT learned general language patterns from large curated text corpora; you teach it your specific labeling task with hundreds or thousands of examples instead of millions.

Prosise Chapter 13 uses **Hugging Face Transformers** - the de facto library for BERT and descendants (DistilBERT, RoBERTa).

---

## BERT Pretraining Objectives

**Masked Language Modeling (MLM):**

$$
P(x_m \mid x_{\backslash m})
$$
> **Readable form:** predict masked token given all other tokens (bidirectional context)

Randomly mask 15% of tokens; model predicts original words. **Bidirectional** - unlike left-to-right language models.

**Fine-tuning** adds a task head on top of pretrained weights - classification, token labeling, etc.

See [representation-learning](../../GLOSSARY.md#representation-learning) for the transfer-learning pattern.

---

## Tokenization: WordPiece

BERT uses **WordPiece** subword tokenization:

```
"playing" → ["play", "##ing"]
"unhappiness" → ["un", "##happiness"]
```

Handles OOV via subwords - rare words decompose into known pieces.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print(tokenizer.tokenize("Unhappiness is playing out"))
# ['un', '##happ', '##iness', 'is', 'playing', 'out']
```

---

## Special Tokens

| Token | Role |
|-------|------|
| `[CLS]` | Classification summary token (first position) |
| `[SEP]` | Separator between sentence pairs |
| `[MASK]` | MLM pretraining mask |
| `[PAD]` | Batch padding for BERT/DistilBERT tokenizers |

In Hugging Face tokenizers, the padding token is exposed as `tokenizer.pad_token` and its integer index as `tokenizer.pad_token_id`. For `bert-base-uncased` and `distilbert-base-uncased`, the literal token string is `[PAD]`; this is correct, not a placeholder.

```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
assert tokenizer.pad_token == "[PAD]"
assert tokenizer.pad_token_id == 0
```

Use `padding=True` during batch tokenization so shorter examples align without accidentally treating padding as real text. The common fallback `tokenizer.pad_token = tokenizer.eos_token` is for GPT-style tokenizers that have no pad token by default, not for BERT or DistilBERT.

For **single-sentence classification**, `[CLS]` hidden state feeds the classifier:

$$
\hat{y} = \text{softmax}(W \mathbf{h}_{[CLS]} + b)
$$
> **Readable form:** prediction = softmax of linear layer applied to [CLS] token representation

See [representation-learning](../../GLOSSARY.md#representation-learning).

---

## Fine-Tuning vs Feature Extraction

| Approach | Trainable | Data needed | Speed |
|----------|-----------|-------------|-------|
| **Fine-tune** | All or top layers | Moderate | Slower |
| **Feature extraction** | Classifier head only | Small | Faster |

**Fine-tuning** updates BERT weights for domain language - recommended when you have 500+ labels and GPU time.

**Feature extraction** freezes BERT, trains logistic regression on `[CLS]` vectors - quick baseline.

See [representation-learning](../../GLOSSARY.md#representation-learning) for the fine-tuning vs feature-extraction tradeoff.

---

## Hugging Face Fine-Tuning (PyTorch-First)

```python
# pip install transformers datasets torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "distilbert-base-uncased"
num_labels = 2

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels
)

texts = ["Great film", "Awful waste"]
labels = torch.tensor([1, 0])

encodings = tokenizer(
    texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
)

batch = dict(encodings)
batch["labels"] = labels

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()
outputs = model(**batch)
loss = outputs.loss
loss.backward()
optimizer.step()
```

**Learning rate:** $2 \times 10^{-5}$ to $5 \times 10^{-5}$ typical for BERT fine-tune - much smaller than from-scratch training.

---

## Trainer API for Full Runs

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./bert_out',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    evaluation_strategy='epoch',
)

# trainer = Trainer(model=model, args=training_args, train_dataset=..., eval_dataset=...)
# trainer.train()
```

Use the low-level loop for understanding and the `Trainer` API for reproducible full fine-tuning runs. TensorFlow/Keras examples still exist, but PyTorch examples are the safer default for current Hugging Face documentation and community support.

---

## DistilBERT for Course Labs

**DistilBERT** - 40% smaller, 60% faster, ~97% of BERT performance:

```python
model_name = "distilbert-base-uncased"  # preferred for Lab 13 on limited GPU
```

Full BERT-base (110M params) needs more memory; DistilBERT fits consumer GPUs.

---

## Comparison to LSTM Baseline

[Lab 13](./section-lab-13-nlp-progression-sentiment-to-translation.md) requires metric table:

| Model | Expected IMDB acc | Train time (GPU) |
|-------|-------------------|------------------|
| TF-IDF + logistic | ~85-90% | seconds |
| Embedding + LSTM | ~88-90% | minutes |
| DistilBERT fine-tune | ~91-93% | 10-30 min |

Gains diminish on **tiny** datasets - BERT can overfit 200 examples without regularization.

Prosise's practitioner caveat: transformer fine-tuning usually starts to justify its overhead once you have roughly **1,500 labeled examples** or a strong domain reason to use transfer learning. Below that, TF-IDF or an LSTM baseline may be cheaper, easier to debug, and nearly as accurate.

---

## Saving and Loading

```python
model.save_pretrained('./my_bert_sentiment')
tokenizer.save_pretrained('./my_bert_sentiment')

# Reload
from transformers import AutoModelForSequenceClassification, AutoTokenizer
loaded_model = AutoModelForSequenceClassification.from_pretrained('./my_bert_sentiment')
loaded_tokenizer = AutoTokenizer.from_pretrained('./my_bert_sentiment')
```

Package model + tokenizer together for serving ([Chapter 07](../chapter-07-operationalizing-models/README.md)).

---

## Inference Pipeline

```python
def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = int(torch.argmax(logits, dim=1).item())
    return 'positive' if pred == 1 else 'negative'

predict_sentiment("Not bad at all", loaded_model, loaded_tokenizer)
```

Contextual embeddings handle negation better than bag-of-words - compare on adversarial examples.

---

## Responsible Deployment Notes

- **Bias** in pretraining corpora propagates to fine-tuned models
- **Latency:** BERT inference 10-100× slower than linear models - batch or distill for production
- **Privacy:** sending text to cloud BERT APIs vs on-prem - see [Chapter 14](../chapter-14-azure-cognitive-services/section-04-azure-language-services.md)

---

## Key Takeaways

1. **BERT** pretrains bidirectional transformer with MLM; fine-tune for downstream tasks
2. **`[CLS]` token** representation drives sentence classification
3. **DistilBERT** balances accuracy and lab-friendly compute
4. Fine-tune with **low learning rate** ($\sim 2 \times 10^{-5}$)
5. Compare BERT gains against **Chapter 04 and LSTM** - not every task needs transformers

---

## Check Your Understanding

1. What is masked language modeling?
2. What role does the [CLS] token play in classification?
3. When freeze BERT vs fine-tune all weights?
4. Why use DistilBERT over BERT-base in labs?
5. What is WordPiece tokenization solving?

---

## References

- Devlin et al., BERT - [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
- Hugging Face course - [https://huggingface.co/learn/nlp-course](https://huggingface.co/learn/nlp-course)
- Prosise, Ch. 13

---

**Next:** [Section 13.8 - NLP Model Selection Guide](./section-08-nlp-model-selection-guide.md)
