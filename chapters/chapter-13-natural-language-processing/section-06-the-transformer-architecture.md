# Section 13.6: The Transformer Architecture

> **Source:** Prosise, Ch. 13 - transformers and attention  
> **Prerequisites:** [Sections 13.4-13.5](./section-04-recurrent-neural-networks.md)  
> **Glossary:** [neural-network](../../GLOSSARY.md#neural-network) | [deep-learning](../../GLOSSARY.md#deep-learning) | [backpropagation](../../GLOSSARY.md#backpropagation) | [representation-learning](../../GLOSSARY.md#representation-learning)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Attention Is All You Need

The **Transformer** (Vaswani et al., 2017) removed recurrence entirely. Every token attends to every other token in **parallel** - enabling GPU-friendly training on massive corpora and state-of-the-art results on translation, classification, and generation.

> **In plain English:** Instead of reading left-to-right like an RNN, every word in the sentence looks at every other word at once and decides what's relevant.

> **Humorous analogy:** RNNs are a conga line - each person only sees the person ahead. Transformers are a roundtable where everyone hears everyone simultaneously.

---

## Scaled Dot-Product Attention

Given queries $Q$, keys $K$, values $V$ (matrices of token representations):

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$
> **Readable form:** attention output = softmax of (query-key dot products scaled by sqrt(d_k)), then multiply by values

Scaling by $\sqrt{d_k}$ prevents dot products from growing too large (softmax saturation).

**Self-attention:** $Q$, $K$, $V$ all come from the **same** sequence - each token attends to all tokens in the sentence.

See [neural-network](../../GLOSSARY.md#neural-network) for foundational context.

---

## Multi-Head Attention

Run $h$ attention heads in parallel with different learned projections:

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
$$
> **Readable form:** multi-head = run several attention operations in parallel with different weight matrices, concatenate, project

Each head can focus on different relation types - syntax, coreference, long-distance dependencies.

---

## Positional Encoding

Without recurrence, the model has **no inherent word order**. **Positional encodings** inject position information:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$
> **Readable form:** even dimensions use sine, odd dimensions use cosine of position-dependent frequencies

Sinusoidal encodings are added to token embeddings before encoder/decoder layers. Modern Hugging Face models often use learned position embeddings or relative-position variants internally, so you usually configure a pretrained model rather than writing this formula by hand.

See [neural-network](../../GLOSSARY.md#neural-network) for how positional information supplements token embeddings.

---

## Transformer Block

Each encoder layer contains:

```
Input embeddings + positional encoding
    ↓
Multi-Head Self-Attention
    ↓
Add & Norm (residual + layer norm)
    ↓
Feed-Forward Network (two linear layers)
    ↓
Add & Norm
```

Decoder adds **masked** self-attention (cannot peek at future tokens) and **cross-attention** to encoder outputs.

$$
\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x} W_1 + b_1) W_2 + b_2
$$
> **Readable form:** feed-forward = expand dimension, ReLU, project back - applied per token independently

Stack $N$ layers (BERT-base: $N=12$).

---

## Why Transformers Beat RNNs for Scale

| Property | RNN/LSTM | Transformer |
|----------|----------|-------------|
| Parallelism | Sequential timesteps | All tokens parallel |
| Long-range deps | Via hidden state (limited) | Direct attention paths |
| Training speed | Slower on long seq | Faster on GPU/TPU |
| Parameters | Often fewer | Often more (needs data) |
| Inference | Linear in length | Quadratic memory in length |

Quadratic attention cost $O(n^2)$ in sequence length - research on efficient attention continues.

Transformers enable parallel sequence modeling - see [deep-learning](../../GLOSSARY.md#deep-learning) and [neural-network](../../GLOSSARY.md#neural-network).

---

## Encoder-Only vs Decoder-Only vs Encoder-Decoder

| Variant | Architecture | Examples |
|---------|--------------|----------|
| Encoder-only | BERT | Classification, NER |
| Decoder-only | GPT | Text generation |
| Encoder-decoder | Original Transformer, T5 | Translation, summarization |

Prosise Chapter 13 emphasizes **encoder-only BERT** for classification ([Section 13.7](./section-07-bert-and-transfer-learning-for-nlp.md)).

---

## Minimal Transformer Classifier (PyTorch-First)

Full implementation is verbose; most current Hugging Face examples use PyTorch-first APIs. Prosise's chapter is Keras-oriented, but the maintained production path for fine-tuning examples is typically `AutoModelForSequenceClassification`:

```python
# pip install transformers torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

texts = ["Great movie!", "Terrible plot."]
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
# outputs = model(encodings)  # after fine-tuning head
```

DistilBERT is a smaller transformer - good for Course 1 labs without massive GPU.

---

## Attention Visualization (Interpretability)

Attention weights show which tokens influenced a prediction:

```python
# After loading model with output_attentions=True
# attentions[layer][head][batch, query_idx, key_idx]
```

High weight from "not" → "good" suggests negation handling - debugging tool for production NLP.

---

## From Transformer to BERT

**BERT** (Devlin et al., 2018) pretrains **encoder-only** transformer with:

- **Masked Language Modeling (MLM)** - predict masked tokens
- **Next Sentence Prediction (NSP)** - sentence relationship (later often dropped)

Pretraining learns general language; **fine-tuning** adds task-specific head ([Section 13.7](./section-07-bert-and-transfer-learning-for-nlp.md)).

---

## Connection to Chapter 14

Azure Language services run transformer-class models server-side - you call REST APIs instead of training attention layers yourself. Architectural literacy from this section explains **latency**, **token limits**, and **why context windows matter**.

---

## Key Takeaways

1. **Self-attention** relates all token pairs: $\text{softmax}(QK^\top/\sqrt{d_k})V$
2. **Multi-head attention** captures diverse relation types in parallel
3. **Positional encoding** injects order without recurrence
4. Transformers **parallelize** training; cost scales $O(n^2)$ with sequence length
5. **Encoder-only** (BERT) dominates classification fine-tuning in Prosise Ch. 13

---

## Complexity Summary

| Component | Time complexity | Space (activations) |
|-----------|-----------------|---------------------|
| Self-attention | $O(n^2 \cdot d)$ | $O(n^2)$ attention matrix |
| Feed-forward per layer | $O(n \cdot d^2)$ | $O(n \cdot d)$ |
| Full encoder stack | $L$ layers of above | Dominated by $n^2$ for long $n$ |

Practical limit: documents beyond a few thousand tokens need chunking or models with linear attention variants (outside Course 1 scope).

---

## Check Your Understanding

1. What problem does positional encoding solve?
2. Write scaled dot-product attention in words.
3. Why divide by $\sqrt{d_k}$ in attention?
4. How does masked decoder self-attention differ from encoder self-attention?
5. Why are transformers faster to *train* than RNNs despite more parameters?

---

## References

- Vaswani et al., "Attention Is All You Need" - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Prosise, Ch. 13
- Illustrated Transformer - [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

---

**Next:** [Section 13.7 - BERT & Transfer Learning for NLP](./section-07-bert-and-transfer-learning-for-nlp.md)


---

## Assessment Practice

Use the shared [Assessment Appendix](../../ASSESSMENT_APPENDIX.md) for concept audits, worked examples, implementation checks, experiment logs, oral-exam prompts, and deliverable checklists.
