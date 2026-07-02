# Section 14.1: Build vs Buy in AI

> **Source:** Prosise, Ch. 14 - opening; AI as a service  
> **Prerequisites:** Chapters [01](../chapter-01-machine-learning/README.md)-[13](../chapter-13-natural-language-processing/README.md)  
> **Glossary:** [deep-learning](../../GLOSSARY.md#deep-learning) | [machine-learning](../../GLOSSARY.md#machine-learning) | [classification](../../GLOSSARY.md#classification)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Mission Control Problem

Prosise opens Chapter 14 with Apollo-era engineers who put humans on the moon using computers weaker than today's phones. Modern **deep learning** can recognize objects, translate speech, and caption images - feats unimaginable in the 1960s. But training ResNet for ImageNet reportedly cost **hundreds of thousands of dollars** in GPU time, expertise, and labeled data.

Most engineering teams cannot - and should not - replicate that investment for every feature.

> **Humorous analogy:** Building your own foundation model is like brewing your own jet fuel because you don't trust the gas station. Possible in theory. Your sprint review will not survive it.

> **In plain English:** Cloud AI APIs let you call pretrained models over HTTPS. You trade control for speed - often the right trade.

---

## AI as a Service

Microsoft (**Azure Cognitive Services**), Amazon (**AWS AI Services**), and Google (**Cloud AI**) employ dedicated research teams who train models on massive datasets, operate GPU clusters, expose capabilities via **REST APIs**, and improve models behind the same endpoint over time.

If you can send an HTTP request, you can add sentiment analysis, OCR, or speech synthesis - without a PhD or a data center. Prosise focuses on Azure; patterns transfer to AWS and Google.

---

## When to Build (Custom Models - Chapters 02-13)

| Factor | Custom model wins |
|--------|-------------------|
| **Domain specificity** | Medical jargon, legal clauses, proprietary SKUs |
| **Privacy / compliance** | Data cannot leave VPC; HIPAA, defense, air-gapped |
| **Cost at scale** | Millions of inferences/day - API per-call pricing exceeds hosting |
| **Latency** | Sub-10 ms on edge hardware; no network round trip |
| **Differentiation** | Model quality IS the product moat |
| **Offline** | Ships, factories without reliable internet |

Examples: fraud detection ([Chapter 03](../chapter-03-classification-models/README.md)), YOLO on factory lines ([Chapter 12](../chapter-12-object-detection/README.md)), fine-tuned DistilBERT on tickets ([Chapter 13](../chapter-13-natural-language-processing/README.md)).

---

## When to Buy (Cognitive Services APIs)

| Factor | API wins |
|--------|----------|
| **Commodity task** | Generic sentiment, OCR, translation, speech |
| **Time to market** | Ship in days, not sprints |
| **Small/medium volume** | Free F0 tiers cover prototypes |
| **No ML team** | Backend developers only |
| **Breadth** | Vision + language + speech in one app |
| **Maintenance** | Vendor handles model updates |

---

## Hybrid Architectures

```
User upload
    ├─► Computer Vision OCR ──► extract text
    ├─► TF-IDF router (Chapter 04) ──► classify intent
    ├─► Fine-tuned BERT if confidence < 0.7
    └─► Translator ──► user's language
```

[Section 14.7](./section-07-contoso-travel-multi-service-app.md) chains OCR + translation. [Chapter 12, Section 7](../chapter-12-object-detection/section-07-azure-custom-vision.md) uses Custom Vision when generic detection fails.

---

## Cost Model Comparison

$$
\text{TCO}_{\text{build}} = C_{\text{data}} + C_{\text{train}} + C_{\text{host}} + C_{\text{mlops}}
$$
> **Readable form:** custom total cost = labeling + training compute + inference hosting + ML engineering time

$$
\text{TCO}_{\text{API}} = n_{\text{calls}} \times p_{\text{call}} + C_{\text{integration}}
$$
> **Readable form:** API total cost = number of calls times price per call plus integration effort

At low volume, API is cheaper. At millions of calls/month with stable models, self-hosting may win - but requires [Chapter 07](../chapter-07-operationalizing-models/README.md) ops maturity.

---

## Capability Overlap Map

| Task | Course 1 custom | Azure service |
|------|-----------------|---------------|
| Sentiment | Chapters 04, 13 | Language |
| Image classification | Chapter 10 CNN | Computer Vision tags |
| Object detection | Chapter 12 YOLO | Vision detect / Custom Vision |
| Translation | Chapter 13 NMT | Translator |
| Speech-to-text | - | Speech |
| Digit OCR | Chapter 03 MNIST | Vision OCR |

Overlap ≠ equivalence - benchmark on **your** data.

---

## Control vs Convenience

```
Full control ◄────────────────────────────────► Convenience
 Custom CNN    Fine-tuned BERT    Cognitive APIs    No-code SaaS
```

Move right when speed matters; move left when privacy or unit economics demand it.

---

## Decision Worksheet

1. Volume forecast - calls/day for 12 months?
2. Data residency - can images/text leave your network?
3. Accuracy - does generic API pass on a labeled sample?
4. Latency SLA - p99 milliseconds?
5. Team - ML engineers for maintenance?
6. Regulatory - explainability, audit trail?

Document in [Lab 14](./section-lab-14-contoso-travel-assistant.md)'s decision matrix.

---

## Responsible AI

Cloud APIs process user data on vendor infrastructure. Read data processing terms, use PII detection ([Section 14.4](./section-04-azure-language-services.md)), and review Responsible AI tools ([Section 14.8](./section-08-production-considerations.md)) before production.

---

## Key Takeaways

1. **State-of-the-art DL is expensive to build** - APIs democratize access
2. **Custom models** win on domain, privacy, and scale economics
3. **Cognitive Services** win on speed, breadth, and low initial cost
4. **Hybrid architectures** are the enterprise norm
5. **TCO** depends on volume, ops headcount, and data constraints
6. **Evaluate on your data** before choosing

---

## Check Your Understanding

1. Why did Prosise compare modern AI to Apollo-era computing?
2. Name three conditions favoring a custom model over an API.
3. What is a hybrid NLP routing architecture?
4. When does API pricing typically lose to self-hosting?
5. Which Chapter 04 task has a direct Azure Language equivalent?

---

## Extended Decision Framework

Use a weighted scorecard when stakeholders disagree:

| Criterion | Weight | Custom (1-5) | API (1-5) |
|-----------|--------|--------------|-----------|
| Time to market | 0.25 | | |
| Domain fit | 0.25 | | |
| Privacy / compliance | 0.20 | | |
| 3-year TCO | 0.20 | | |
| Team ML depth | 0.10 | | |

Document the ADR in your repo - future you will thank present you.

---

## Math: Break-Even Volume

Let $V$ be monthly inferences, $c_b$ build amortized cost per call, $c_a$ API cost per call:

$$
V^* = \frac{C_{\text{build, fixed}}}{c_a - c_b}
$$
> **Readable form:** break-even volume = fixed build cost divided by per-call savings of self-hosting over the API

When $c_a \approx c_b$, break-even never arrives - prefer APIs unless non-cost factors dominate.

---

## Glossary Connections

- [supervised-learning](../../GLOSSARY.md#supervised-learning) - custom models from Chapters 02-13
- [deep-learning](../../GLOSSARY.md#deep-learning) - what Cognitive Services pre-train for you
- [classification](../../GLOSSARY.md#classification) - sentiment, vision tagging, speech intent
- [overfitting](../../GLOSSARY.md#overfitting) - risk when you *do* train on tiny domain data instead of using a general API

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Building sentiment from scratch for MVP | Months delay | Azure Language or HF pipeline first |
| Using APIs for core IP | No competitive moat | Custom model on proprietary data |
| Ignoring volume economics | Surprise cloud bill | Forecast calls × price at 12 months |
| No hybrid fallback | Total outage when API down | Degrade gracefully |
| Skipping domain benchmark | Wrong buy/build choice | Label 200 samples, compare API vs custom |

---

## Self-Check

1. What did Microsoft reportedly spend training ResNet (per Prosise)?  
   *Hundreds of thousands of dollars.*
2. Name two reasons to buy Cognitive Services APIs.  
   *Fast time-to-market and commodity tasks.*
3. What is a hybrid architecture?  
   *Mix of custom models and cloud APIs in one product.*
4. When does self-hosting often beat APIs?  
   *High volume, privacy requirements, or strict latency.*
5. Which chapter covers Custom Vision as a middle ground?  
   *[Chapter 12](../chapter-12-object-detection/section-07-azure-custom-vision.md).*

---

## Exercises

1. Complete the decision worksheet for a real or hypothetical product.
2. Estimate monthly API cost at 100k vs 10M calls (pick one service pricing page).
3. Map three Course 1 chapters to their Azure API equivalents in a table.
4. Write three sentences defending build vs buy for your current employer.

---

## References

- Prosise, J. *Applied Machine Learning and AI for Engineers* (O'Reilly, 2023), Chapter 14 (opening)
- Azure Cognitive Services: [https://azure.microsoft.com/en-us/products/ai-services](https://azure.microsoft.com/en-us/products/ai-services)
- [Section 13.8](../chapter-13-natural-language-processing/section-08-nlp-model-selection-guide.md)
- [Chapter 07](../chapter-07-operationalizing-models/README.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Exercises

1. Fill the decision scorecard for a fraud-detection product (build vs API).
2. Estimate $V^*$ given hypothetical $C_{\text{build,fixed}}$, $c_a$, and $c_b$.
3. List three hybrid architectures from your own work or hobbies.

---

**Next:** [Section 14.2 - Azure Setup & Authentication](./section-02-azure-setup-and-authentication.md)



