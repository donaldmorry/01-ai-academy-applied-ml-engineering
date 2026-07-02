# Chapter 13: Natural Language Processing

> **Source:** *Applied Machine Learning and AI for Engineers*, Chapter 13  
> **Part:** II — Deep Learning with Keras & TensorFlow  
> **Estimated time:** 12–14 hours  
> **Prerequisites:** [Chapter 04 — Text Classification](../chapter-04-text-classification/README.md); [Chapter 09 — Neural Networks](../chapter-09-neural-networks/README.md)

---

## Chapter Overview

Chapter 04 taught text classification with bag-of-words and Naive Bayes. This chapter goes deeper — into **word embeddings**, **recurrent neural networks**, **transformers**, and **BERT** — the techniques behind modern language AI.

You will build the same preprocessing foundations at a neural scale: tokenization with Keras `TextVectorization`, trainable embedding layers, sequence models (RNN, LSTM, GRU) for sentiment and classification, and encoder-decoder architectures for **neural machine translation (NMT)**. The chapter culminates with **transformer** attention mechanisms and using **BERT** for downstream tasks — connecting classical text ML to the large language model era.

NLP is the fastest-moving area in applied AI. This chapter gives you the architectural literacy to understand, fine-tune, and responsibly deploy language models — whether in Keras or via cloud APIs (Chapter 14).

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Preprocess text with Keras `TextVectorization` (tokenization, sequencing, padding)
2. Build and train embedding layers that learn word representations from data
3. Implement RNN, LSTM, and GRU models for sequence classification
4. Explain the vanishing gradient problem and why LSTMs help
5. Build an encoder-decoder model for neural machine translation
6. Explain the transformer architecture: self-attention, multi-head attention, positional encoding
7. Use pretrained BERT for text classification via fine-tuning or feature extraction
8. Compare classical (Chapter 04), RNN, and transformer approaches on the same task

---

## Sections

| # | Section | File | Topics |
|---|--------|------|--------|
| 13.1 | Beyond Bag-of-Words | [01-beyond-bag-of-words.md](./section-01-beyond-bag-of-words.md) | Limitations of TF-IDF; distributional semantics; word embeddings |
| 13.2 | Text Preprocessing with Keras | [02-text-preprocessing-keras.md](./section-02-text-preprocessing-with-keras.md) | `TextVectorization`; vocabulary; OOV token; sequence padding |
| 13.3 | Embedding Layers | [03-embedding-layers.md](./section-03-embedding-layers.md) | `Embedding` layer; learned vs pretrained (GloVe, Word2Vec) |
| 13.4 | Recurrent Neural Networks | [04-recurrent-neural-networks.md](./section-04-recurrent-neural-networks.md) | RNN, LSTM, GRU; sequence classification; sentiment with LSTM |
| 13.5 | Sequence-to-Sequence Models | [05-sequence-to-sequence-models.md](./section-05-sequence-to-sequence-models.md) | Encoder-decoder; teacher forcing; neural machine translation |
| 13.6 | The Transformer Architecture | [06-transformer-architecture.md](./section-06-the-transformer-architecture.md) | Self-attention; multi-head attention; positional encoding; parallelization |
| 13.7 | BERT & Transfer Learning for NLP | [07-bert-and-transfer-learning.md](./section-07-bert-and-transfer-learning-for-nlp.md) | Pretraining; fine-tuning; `[CLS]` token; Hugging Face Transformers |
| 13.8 | NLP Model Selection Guide | [08-nlp-model-selection-guide.md](./section-08-nlp-model-selection-guide.md) | When RNNs suffice; when transformers are worth the cost; latency vs accuracy |

---

## Lab

**[Lab 13: NLP Progression — Sentiment to Translation](./section-lab-13-nlp-progression-sentiment-to-translation.md)**

Three-tier lab showing NLP evolution:

1. **Embedding + LSTM Sentiment** — IMDB reviews with `TextVectorization` + `Embedding` + `LSTM`. Beat Chapter 04 TF-IDF baseline. Visualize learned embeddings with t-SNE (optional).
2. **Neural Machine Translation (Mini)** — English-French (or similar) sentence pairs with encoder-decoder LSTM. Evaluate with BLEU score on held-out sentences.
3. **BERT Fine-Tuning** — Fine-tune DistilBERT on a classification task (sentiment or news category). Compare accuracy and training time to LSTM.

*Deliverable:* Notebook with all three models, metric comparison table, and written analysis of when each architecture is appropriate.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Word embeddings | Word2Vec math, GloVe, fastText (Course 2, Ch 23) |
| RNNs & LSTMs | Formal RNN theory, bidirectional models (Course 3) |
| Transformers & attention | Full transformer derivation, GPT architecture (Course 3) |
| BERT & pretraining | LLMs, RLHF, prompt engineering (Course 3) |
| Neural machine translation | Attention paper, seq2seq theory (Course 2, Ch 23–24) |

---

## Prerequisites

- [Chapter 04 — Text Classification](../chapter-04-text-classification/README.md) (text preprocessing, classification metrics)
- [Chapter 09 — Neural Networks](../chapter-09-neural-networks/README.md) (Keras training, callbacks)
- TensorFlow 2.x with Keras
- Hugging Face `transformers` library for BERT section (`pip install transformers`)
- GPU recommended for BERT fine-tuning; CPU workable for small models

---

## Key Takeaways

- Embeddings capture semantic relationships that bag-of-words cannot — "king" - "man" + "woman" ≈ "queen"
- LSTMs solve vanishing gradients for longer sequences but are slow to train sequentially
- Transformers replace recurrence with attention — enabling parallel training and state-of-the-art results
- BERT provides pretrained language understanding; fine-tuning adapts it to specific tasks with modest data
- Always compare neural NLP to your Chapter 04 baseline — simpler methods are sometimes sufficient

---

## Self-Assessment

1. What information does an embedding layer capture that TF-IDF cannot?
2. Why do RNNs struggle with long sequences, and how do LSTMs address this?
3. What is teacher forcing in sequence-to-sequence training?
4. How does self-attention allow a model to relate distant words in a sentence?
5. What is the difference between fine-tuning BERT and using it as a frozen feature extractor?
6. Why are transformers faster to train than RNNs despite having more parameters?
7. When would you still choose TF-IDF + logistic regression over BERT for a text classification task?

---

**Previous:** [Chapter 12 — Object Detection](../chapter-12-object-detection/README.md)  
**Next:** [Chapter 14 — Azure Cognitive Services](../chapter-14-azure-cognitive-services/README.md)
