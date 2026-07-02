# Section 14.4: Azure Language Services

> **Source:** Prosise, Ch. 14 - "The Language Service"  
> **Prerequisites:** [Section 14.2](./section-02-azure-setup-and-authentication.md) | [Chapter 04](../chapter-04-text-classification/README.md) | [Chapter 13](../chapter-13-natural-language-processing/README.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [tokenization](../../GLOSSARY.md#tokenization)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## NLP Models You Don't Train

[Chapter 13](../chapter-13-natural-language-processing/README.md) built LSTM and BERT classifiers. **Azure Language** exposes sentiment, NER, key phrases, PII detection, and QA via API - transformer backends you don't host.

> **Humorous analogy:** Fine-tuning BERT is cooking from scratch. Language service is catering - same category, you didn't chop onions.

> **In plain English:** POST text JSON; get sentiment scores, entities, or key phrases back.

---

## Unified Service

Legacy Text Analytics, LUIS, and QnA Maker merged into **Language** - target Language for new projects.

```python
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import os

client = TextAnalyticsClient(
    os.environ['AZURE_LANGUAGE_ENDPOINT'],
    AzureKeyCredential(os.environ['AZURE_LANGUAGE_KEY']),
)
```

---

## Sentiment Analysis

```python
docs = [{'id': '1', 'text': 'Programming is fun, but the hours are long'}]
for r in client.analyze_sentiment(docs):
    print(r.confidence_scores)  # positive=0.94, neutral=0.05, negative=0.01
```

**Batch tweets** - one round trip:

```python
input_docs = [
    {'id': '1000', 'text': 'Programming is fun, but the hours are long'},
    {'id': '1001', 'text': 'Great food and excellent service'},
    {'id': '1002', 'text': 'The product worked but is overpriced'},
]
for r in client.analyze_sentiment(input_docs):
    print(f'{r.id} => pos={r.confidence_scores.positive:.2f}')
```

Label from scores:

$$
\text{label} = \arg\max(p_{\text{pos}}, p_{\text{neu}}, p_{\text{neg}})
$$
> **Readable form:** predicted sentiment = category with highest confidence score

Compare [Section 4.5](../chapter-04-text-classification/section-05-sentiment-analysis.md) - same task, data leaves your network.

---

## Named Entity Recognition

```python
docs = ["My printer isn't working. Can someone from IT come to my office?"]
for r in client.recognize_entities(docs):
    for e in r.entities:
        print(f'{e.text} ({e.category})')
# printer (Product), IT (Skill), office (Location)
```

Route support tickets, enrich CRM - API wins on generic entities; custom wins on proprietary catalogs.

---

## PII Detection

```python
docs = ['Contact John at john@email.com. SSN: 123-45-6789']
for r in client.recognize_pii_entities(docs):
    for e in r.entities:
        print(f'{e.text} - {e.category}')
```

Redact before logging - complements access control, not a replacement.

---

## Key Phrase Extraction

```python
docs = ['NLP encompasses classification, NER, QA, and translation.']
for r in client.extract_key_phrases(docs):
    for phrase in r.key_phrases:
        print(phrase)
```

Batch hundreds of docs for topic clustering.

---

## vs Chapters 04 and 13

| Aspect | TF-IDF + NB | Fine-tuned BERT | Language API |
|--------|-------------|-----------------|--------------|
| Setup | Minutes | Hours + GPU | Minutes |
| Domain slang | Retrain | Fine-tune | May miss |
| 100 tweets | In-process | GPU batch | 1 API call |
| Privacy | On-prem | On-prem | Cloud |

Measure on your holdout set before choosing.

---

## Language Detection

```python
for r in client.detect_language(['Bonjour le monde', 'Hello world']):
    print(r.primary_language.name, r.primary_language.confidence_score)
```

Route to translation ([Section 14.5](./section-05-azure-translator.md)) when needed.

---

## Brand Monitoring Architecture

```
Cron → fetch tweets @Contoso
     → analyze_sentiment(batch)  # one call
     → store time series
     → alert if 7-day negative mean > threshold
```

Prosise: batching beats per-tweet calls for cost and latency.

---

## Error Handling

```python
from azure.core.exceptions import HttpResponseError
try:
    client.analyze_sentiment([{'id': '1', 'text': ''}])
except HttpResponseError as e:
    print(e.message)
```

Validate non-empty strings client-side.

---

## Key Takeaways

1. **Language** unifies sentiment, NER, PII, key phrases
2. **Batch documents** per request
3. **Sentiment** - three probabilities, argmax for label
4. **NER** for routing and enrichment
5. **PII** for redaction workflows
6. **Benchmark** against Chapters 04/13 on your data

---

## Check Your Understanding

1. What replaced Text Analytics?
2. Why batch 100 tweets in one call?
3. How do confidence scores become a label?
4. When train BERT instead of using API?
5. What is PII detection used for?

---

## Sentiment as a Distribution

Azure returns $(p_+, p_0, p_-)$ with $p_+ + p_0 + p_- = 1$. The reported label is $\arg\max$:

$$
\hat{y} = \underset{y \in \{+1,0,-1\}}{\arg\max}\; P(y \mid \text{text})
$$
> **Readable form:** pick the sentiment class with highest predicted probability

Compare to [Chapter 04](../chapter-04-text-classification/section-05-sentiment-analysis.md) softmax outputs from your own [classification](../../GLOSSARY.md#classification) model.

---

## Glossary Links

- [supervised-learning](../../GLOSSARY.md#supervised-learning) - training custom sentiment on your labels
- [neural-network](../../GLOSSARY.md#neural-network) - BERT fine-tuning in Chapter 13
- [overfitting](../../GLOSSARY.md#overfitting) - risk when fine-tuning on tiny in-domain sets

---

## Exercises

1. Batch 50 reviews; plot histogram of $p_+$ scores.
2. Run NER on travel queries; map entities to Contoso slots.
3. Redact PII from a synthetic paragraph before logging.

---

## References

- Prosise, J. *Applied Machine Learning and AI for Engineers* (O'Reilly, 2023), Ch. 14 - Language
- [Language service docs](https://learn.microsoft.com/en-us/azure/ai-services/language-service/)
- [Section 4.5](../chapter-04-text-classification/section-05-sentiment-analysis.md)
- [Section 13.7](../chapter-13-natural-language-processing/section-07-bert-and-transfer-learning-for-nlp.md)

---

**Next:** [Section 14.5 - Azure Translator](./section-05-azure-translator.md)

---

## Assessment Practice

Use the shared [Assessment Appendix](../../ASSESSMENT_APPENDIX.md) for concept audits, worked examples, implementation checks, experiment logs, oral-exam prompts, and deliverable checklists.
