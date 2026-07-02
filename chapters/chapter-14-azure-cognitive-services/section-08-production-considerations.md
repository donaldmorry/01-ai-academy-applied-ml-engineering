# Section 14.8: Production Considerations

> **Source:** Prosise, Ch. 14 - containers, reliability, responsible AI  
> **Prerequisites:** [Sections 14.1-14.7](./section-01-build-vs-buy-in-ai.md) | [Chapter 07](../chapter-07-operationalizing-models/README.md)  
> **Glossary:** [overfitting](../../GLOSSARY.md#overfitting) | [model](../../GLOSSARY.md#model) | [cross-validation](../../GLOSSARY.md#cross-validation)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## From Notebook to Production API

Chapters 01-13 trained models you own. Chapter 14 calls models Microsoft owns. Production concerns overlap - and diverge:

| Concern | Custom model (Chapter 07) | Cognitive Services |
|---------|--------------------------|-------------------|
| Version drift | You control retraining | Backend models update silently |
| Latency | Local inference | Network round trip |
| Cost | GPU/hosting | Per-call metering |
| Privacy | Data stays on-prem | Data crosses network boundary |
| Rate limits | Your hardware | Service quotas (429) |

> **Humorous analogy:** Custom models are pets - you feed and walk them. Cognitive Services are rideshare - convenient until surge pricing and the driver takes a new route without asking.

> **In plain English:** Ship APIs with retries, monitoring, and a plan for when the cloud says "slow down" or "no."

---

## Rate Limits and HTTP 429

Azure throttles requests per subscription tier. **HTTP 429** means "too many requests."

**Response strategy - exponential backoff:**

$$
t_{\text{wait}} = \min\left(t_{\max},\; t_0 \cdot 2^{k}\right)
$$
> **Readable form:** wait time = minimum of max cap and base delay times 2 to the power of retry attempt number

```python
import time
import random

def call_with_backoff(fn, max_retries=5, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if '429' not in str(e) and attempt == max_retries - 1:
                raise
            delay = min(60, base_delay * (2 ** attempt)) + random.uniform(0, 0.5)
            time.sleep(delay)
    raise RuntimeError('Max retries exceeded')
```

**Prevention:**
- Batch documents ([Section 14.4](./section-04-azure-language-services.md))
- Cache idempotent reads (same image caption twice)
- Choose tier matching peak QPS

---

## Caching Strategy

| Operation | Cacheable? | Key |
|-----------|------------|-----|
| Image caption | Yes | Image hash |
| Sentiment of static review | Yes | Text hash |
| Live speech STT | No | Real-time |
| Translation | Yes (short TTL) | `(text, target_lang)` |

```python
import hashlib, functools

@functools.lru_cache(maxsize=1024)
def cached_sentiment(text: str) -> float:
    result = client.analyze_sentiment([{'id': '1', 'text': text}])
    return result[0].confidence_scores.positive
```

Invalidate cache when you upgrade API version or need fresh model scores.

---

## Azure Cognitive Services Containers

Prosise highlights **Docker containers** for on-prem / edge deployment:

**Benefits:**
- No internet required for inference (metering still needs connectivity except disconnected EA scenarios)
- **Locked API version** and model weights - no surprise score changes
- Lower latency vs cloud round trip

**Drawbacks:**
- Still billed via encrypted metering on port 443
- You manage container updates
- Not a free bypass - disconnected containers require Enterprise Agreement approval

```
Cloud API:     App → HTTPS → Azure region → model
Container:     App → localhost:5000 → same API surface, local process
```

See Microsoft docs: "What Are Azure Cognitive Services Containers?"

---

## SDK vs REST in Production

| Factor | SDK (`azure-ai-*`) | REST (`requests`) |
|--------|-------------------|-------------------|
| API path changes | SDK update shields you | You update URLs |
| Error types | Typed exceptions (`AzureError`) | Parse status codes |
| Batching | Built-in helpers | Manual JSON |
| Translator | Often REST-only | Standard pattern |

Prosise recommends SDKs for Language and Vision; REST remains common for Translator. Wrap both in your own service layer so swapping is painless.

---

## Authentication Best Practices

1. **Never** embed keys in source - use env vars, Azure Key Vault, managed identity
2. **Rotate** keys - Portal provides Key1/Key2 for zero-downtime rotation
3. **Multiservice key** - one key for multiple services (simpler ops)
4. **AAD + RBAC** - preferred for apps hosted in Azure

```python
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

client = TextAnalyticsClient(
    os.environ['AZURE_LANGUAGE_ENDPOINT'],
    AzureKeyCredential(os.environ['AZURE_LANGUAGE_KEY']),
)
```

---

## Data Residency and Privacy

Cognitive Services process data in the **region you select** when creating the resource. For GDPR/HIPAA:

- Choose EU region for EU users
- Review [data processing terms](https://www.microsoft.com/licensing/terms/product/ForOnlineServices/all)
- Microsoft states it does not store customer content for model training by default - verify current policy
- **PII:** run `recognize_pii_entities` before logging ([Section 14.4](./section-04-azure-language-services.md))
- **Containers** when data must not leave your network

Voice biometrics and face data carry extra regulatory weight - document retention policies.

---

## Observability

Log structured fields per API call:

```python
log.info('cognitive_call', extra={
    'service': 'vision',
    'operation': 'ocr',
    'latency_ms': elapsed_ms,
    'status': response.status_code,
    'bytes_in': len(image_bytes),
})
```

**Dashboards:** track p95 latency, error rate, 429 count, daily spend. Alert when sentiment monitoring pipeline stops ingesting tweets.

---

## Responsible AI

Azure provides **Responsible AI** tooling:

| Tool | Purpose |
|------|---------|
| Content Safety | Flag harmful text/images |
| Content Moderator | User-upload filtering |
| Transparency docs | Model limitations per service |
| Face service restrictions | Limited access per responsible AI policy |

Engineering checklist:
- Disclose AI-generated captions/translations to users
- Human review for high-stakes decisions (medical, legal)
- Test across languages and demographics for bias
- Provide appeal path when automated moderation blocks content

---

## Cost Management

Free tiers (F0) suit development; production needs budgeting:

$$
\text{monthly\_cost} \approx \sum_{\text{services}} (\text{calls} \times \text{price\_per\_call})
$$
> **Readable form:** monthly cost ≈ sum over services of (number of calls times price per call)

**Tactics:**
- Batch API calls
- Cache repeated queries
- Downsample video frames before Vision calls
- Use custom models at scale when API unit economics invert ([Section 14.1](./section-01-build-vs-buy-in-ai.md))

---

## Hybrid Architecture Pattern

Most enterprise apps combine approaches from this course:

```
User request
  → Custom classifier (Chapter 03) for domain SKU
  → Language API for sentiment on reviews
  → Custom Vision (Chapter 12) for warehouse defects
  → ONNX container (Chapter 07) for edge inference
```

Document **which service handles which task** in your architecture decision record (ADR).

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| No retry on 429 | Random user failures | Exponential backoff |
| Logging raw PII | Compliance violation | PII detection pre-log |
| Single key in prod | Outage on rotation | Key1/Key2 swap pattern |
| Ignoring model drift | Scores change over time | Pin container version or monitor distributions |
| F0 tier in production | Hard throttling | Upgrade tier before launch |

---

## Self-Check

1. What does HTTP 429 indicate, and how should you respond?
2. Why might enterprise customers prefer containerized Cognitive Services?
3. What is the tradeoff between cloud APIs and local ONNX models?
4. Why rotate subscription keys with two-key pattern?
5. When is batching preferable to per-item API calls?

---

## Exercises

1. Implement a wrapper class that retries `analyze_sentiment` on 429 with backoff.
2. Estimate API cost for Contoso Travel at 50k uploads/month (3 lines each).
3. Write an ADR comparing F0 vs S1 tier for a tweet monitoring service.
4. List three Responsible AI checks before launching a user photo upload feature.

---

## Course Completion

Completing this section finishes **Course 1: Applied Machine Learning & AI for Engineers**. You have earned the **ML Practitioner** milestone - regression to deployment, CNNs to object detection, NLP to cloud AI orchestration.

**Capstone:** [Course Capstone Project](../../projects/capstone/README.md)

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 14 - Summary & containers
- [Azure Cognitive Services containers](https://learn.microsoft.com/en-us/azure/ai-services/cognitive-services-container-support)
- [Authenticate requests](https://learn.microsoft.com/en-us/azure/ai-services/authentication)
- [Azure Responsible AI](https://www.microsoft.com/ai/responsible-ai)
- [GLOSSARY.md](../../GLOSSARY.md) - [overfitting](../../GLOSSARY.md#overfitting), [model](../../GLOSSARY.md#model)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 14.7 - Contoso Travel](./section-07-contoso-travel-multi-service-app.md)  
**Next:** [Lab 14 - Contoso Travel Assistant](./section-lab-14-contoso-travel-assistant.md)
