# Section 13.8: NLP Model Selection Guide

> **Source:** Prosise, Ch. 13 - closing guidance and comparisons  
> **Prerequisites:** [Sections 13.1-13.7](./section-01-beyond-bag-of-words.md)  
> **Glossary:** [bag-of-words](../../GLOSSARY.md#bag-of-words) | [TF-IDF](../../GLOSSARY.md#tf-idf) | [neural-network](../../GLOSSARY.md#neural-network) | [deep-learning](../../GLOSSARY.md#deep-learning)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Right Model, Not the Newest

Chapter 13 presents a progression: TF-IDF → embeddings → LSTM → transformer → BERT. **Production engineers** pick the **simplest model that meets requirements** - not the architecture with the most arXiv citations.

> **In plain English:** BERT is a sledgehammer. Sometimes a TF-IDF screwdriver finishes the job faster.

---

## Decision Flowchart

```
Start: text classification task
    ↓
Labeled data < 500 AND keywords clear?
    YES → TF-IDF + logistic regression (Chapter 04)
    NO ↓
Need interpretable term weights?
    YES → TF-IDF + linear model
    NO ↓
Latency budget < 10ms on CPU?
    YES → TF-IDF or small linear; avoid BERT
    NO ↓
Complex semantics, negation, long context?
    YES → LSTM or BERT (GPU)
    NO ↓
SOTA accuracy, GPU available, API cost OK?
    YES → Fine-tune DistilBERT/BERT
```

Document your decision in design reviews - stakeholders care about **cost and latency**, not layer counts.

---

## Comparison Table

| Approach | Data | Latency | Accuracy ceiling | Interpretability |
|----------|------|---------|------------------|------------------|
| TF-IDF + NB/LR | Small+ | Very low | Moderate | High (term weights) |
| Embedding + LSTM | Medium | Medium | Good | Low |
| Embedding + GRU | Medium | Medium | Good | Low |
| Fine-tuned BERT | Medium+ | High | Excellent | Low |
| Azure Language API | Any | Network | Excellent | None (black box) |

[Chapter 14](../chapter-14-azure-cognitive-services/section-04-azure-language-services.md) adds the **API column** when build-vs-buy favors cloud.

---

## Latency and Cost

Rough order-of-magnitude (single sentence, CPU unless noted):

| Model | Inference |
|-------|-----------|
| TF-IDF + LR | <1 ms |
| LSTM (128 units) | 5-20 ms |
| DistilBERT | 50-200 ms CPU; 10-30 ms GPU |
| BERT-base | 100-500 ms CPU |
| GPT-class generation | seconds |

At **1M requests/day**, BERT self-hosting vs API vs TF-IDF differs by thousands of dollars monthly - model selection is a **business** decision.

---

## Data Size Guidelines

| Labeled examples | Starting point |
|------------------|----------------|
| < 200 | TF-IDF; risk overfitting neural |
| 200-2,000 | LSTM or frozen BERT features |
| 2,000-20,000 | Fine-tune DistilBERT |
| 20,000+ | Full BERT fine-tune; consider domain-adaptive pretrain |

Prosise's practical **golden constant**: treat roughly **1,500 labeled examples** of typical review-length text as the point where transformer fine-tuning becomes worth testing against TF-IDF, bag-of-words, or an LSTM baseline. This is a heuristic, not a law: very short texts may need more examples, highly specialized language may justify BERT earlier, and every project should keep the Chapter 04 baseline in the comparison.

Unlabeled data enables **continued pretraining** (advanced) - beyond Prosise scope but explains why LLMs scale.

---

## Task-Specific Notes

### Sentiment / intent
- LSTM often matches BERT within 1-2% on clean IMDB-style data
- BERT wins on sarcasm, mixed sentiment, long reviews

### Neural machine translation
- Seq2seq LSTM teaches fundamentals; production uses transformer MT (Azure Translator in Chapter 14)

### Named entity recognition
- Token-level BERT labels - not covered deeply in Ch. 13; Azure Language NER in Chapter 14

### Search / similarity
- TF-IDF + cosine ([Section 4.7](../chapter-04-text-classification/section-07-cosine-similarity-and-recommender-systems.md)) or embedding retrieval

---

## Hybrid Architectures

Enterprise pattern:

1. **Fast filter:** TF-IDF routes obvious cases (90% traffic)
2. **Neural escalation:** BERT handles ambiguous 10%
3. **Human review:** low-confidence queue

Log confidence scores; tune thresholds on validation **cost** (false negative vs false positive).

---

## Chapter 13 Summary

| Section | Tool | When |
|--------|------|------|
| 13.1 | Concepts | Motivation |
| 13.2 | TextVectorization | Keras preprocessing |
| 13.3 | Embedding | Dense semantics |
| 13.4 | LSTM/GRU | Sequence classification |
| 13.5 | Seq2seq | Translation |
| 13.6 | Transformer | Attention foundation |
| 13.7 | BERT | Transfer learning |
| 13.8 | Selection | Engineering judgment |

[Lab 13](./section-lab-13-nlp-progression-sentiment-to-translation.md) implements three tiers and writes the comparison analysis required here.

---

## Connection to Course Capstone

The [Course Capstone](../../projects/capstone/README.md) may combine custom NLP with Azure APIs - use this guide to justify each component in your architecture document.

---

## Key Takeaways

1. **Start with Chapter 04 baseline** - prove neural complexity is needed
2. **Latency and data size** constrain architecture more than leaderboard scores
3. **DistilBERT** is the practical fine-tuning default for Course 1
4. **Hybrid cascades** reduce cost in production
5. **Azure Language** ([Chapter 14](../chapter-14-azure-cognitive-services/README.md)) when build time exceeds budget

---

## Check Your Understanding

1. When would TF-IDF beat BERT on a sentiment task?
2. What three non-accuracy factors drive model selection?
3. Describe a hybrid TF-IDF + BERT cascade.
4. Why fine-tune DistilBERT instead of training LSTM from scratch on 50K labels?
5. When choose Azure Language over fine-tuned BERT?

---

## Production Monitoring Checklist

After deployment, track metrics **per architecture tier**:

| Signal | TF-IDF | LSTM | BERT |
|--------|--------|------|------|
| Latency p95 | <10ms | <100ms | <500ms |
| Accuracy drift | Weekly sample | Weekly | Monthly eval |
| OOV rate | Rare word % | UNK token % | Subword coverage |
| Cost per 1k inferences | CPU only | GPU optional | GPU typical |

Set alerts when production accuracy drops **>2%** from validation - trigger retrain or fallback to simpler tier.

---

## Exercises

1. Complete the decision flowchart for a support-ticket routing product (real or hypothetical).
2. Fill the comparison table with numbers from [Lab 13](./section-lab-13-nlp-progression-sentiment-to-translation.md).
3. Design a hybrid cascade: what fraction of traffic goes to TF-IDF vs BERT?
4. Write one paragraph justifying Azure Language API vs fine-tuned DistilBERT for your use case.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Defaulting to BERT | Slow, costly MVP | Start with Chapter 04 baseline |
| Ignoring latency SLA | Timeouts in prod | Benchmark p95 before launch |
| No API cost estimate | Budget overrun | Model monthly volume × price |
| Skipping error analysis | Repeated failure mode | Inspect misclassified examples |
| Single metric (accuracy only) | Bad precision/recall tradeoff | Report F1, confusion matrix |

---

## Self-Check

1. When would TF-IDF beat BERT on a sentiment task?  
   *Small data, strict latency, or when keywords dominate and baseline accuracy suffices.*
2. What three non-accuracy factors drive model selection?  
   *Latency, cost, data size (and interpretability).*
3. Describe a hybrid TF-IDF + BERT cascade.  
   *Route easy cases to TF-IDF; escalate low-confidence to BERT.*
4. Why fine-tune DistilBERT instead of training LSTM from scratch on 50K labels?  
   *Better accuracy ceiling with pretrained language knowledge.*
5. When choose Azure Language over fine-tuned BERT?  
   *Fast MVP, no GPU, standard sentiment/NER, acceptable API cost.*

---

## References

- Prosise, Ch. 13 - closing sections
- Sculley et al., "Hidden Technical Debt in Machine Learning Systems" - production complexity
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Course 1 Completion Note

This guide closes Chapter 13 - you now have a framework for choosing classical, recurrent, and transformer NLP stacks before integrating cloud APIs in [Chapter 14](../chapter-14-azure-cognitive-services/README.md).

---

**Next:** [Lab 13 - NLP Progression](./section-lab-13-nlp-progression-sentiment-to-translation.md)
