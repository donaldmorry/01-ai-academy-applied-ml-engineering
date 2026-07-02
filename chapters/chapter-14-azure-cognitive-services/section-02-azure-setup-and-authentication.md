# Section 14.2: Azure Setup & Authentication

> **Source:** Prosise, Ch. 14 - "Keys and Endpoints" / "Calling Azure Cognitive Services APIs"  
> **Prerequisites:** [Section 14.1](./section-01-build-vs-buy-in-ai.md)  
> **Glossary:** [machine-learning](../../GLOSSARY.md#machine-learning)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Before Your First API Call

Every Azure Cognitive Service call needs:

1. **Endpoint** - base URL for the service region
2. **Subscription key** - authenticates billing to your Azure account

> **Humorous analogy:** Endpoint is the restaurant address; the subscription key is your credit card on file. Lose the key and someone else orders steak on your tab.

---

## Creating a Resource (Azure Portal)

Prosise walks through creating a **Language** resource (Fig. 14-3):

1. Sign in to [Azure Portal](https://portal.azure.com)
2. **Create a resource** → AI + Machine Learning → **Language** (or Computer Vision, Speech, etc.)
3. Select subscription, resource group, **region**, **pricing tier** (F0 free tier when available)
4. After deployment → **Keys and Endpoint** (Fig. 14-4)

You receive **Key 1** and **Key 2** - so you can rotate one while the other remains valid.

---

## Single-Service vs Multi-Service Keys

| Key type | Scope |
|----------|-------|
| **Single-service** | One resource (e.g., Language only) |
| **Multi-service (Cognitive Services)** | One key for multiple APIs |

Create a **Cognitive Services** multi-service resource to simplify key management across Vision, Language, and Speech.

---

## REST Call with Requests (Prosise)

```python
import requests

KEY = 'YOUR_SUBSCRIPTION_KEY'
ENDPOINT = 'https://YOUR_REGION.api.cognitive.microsoft.com/'

input_data = {'documents': [
    {'id': '1000', 'text': 'Programming is fun, but the hours are long'}
]}

headers = {
    'Ocp-Apim-Subscription-Key': KEY,
    'Content-type': 'application/json'
}

uri = ENDPOINT + 'text/analytics/v3.0/sentiment'
response = requests.post(uri, headers=headers, json=input_data)
results = response.json()

for doc in results['documents']:
    print(doc['confidenceScores'])
# {'positive': 0.94, 'neutral': 0.05, 'negative': 0.01}
```

JSON in, JSON out - language-agnostic (Python, C#, JavaScript, curl).

---

## Python SDK (Recommended)

```python
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

client = TextAnalyticsClient(ENDPOINT, AzureKeyCredential(KEY))
documents = [{'id': '1000', 'text': 'Programming is fun, but the hours are long'}]

for result in client.analyze_sentiment(documents):
    print(result.confidence_scores)
```

SDK hides URL paths, handles serialization, provides typed errors.

---

## Error Handling

```python
from azure.core.exceptions import AzureError

try:
    response = client.analyze_sentiment(documents)
except AzureError as e:
    print(e.message)
```

| Exception | Typical cause |
|-----------|---------------|
| `ClientAuthenticationError` | Invalid/expired key |
| HTTP 429 | Rate limit exceeded |
| HTTP 400 | Malformed request body |

`AzureError` is the shared base in `azure-core`.

---

## C# SDK Example (Prosise)

Prosise shows identical sentiment output from C# using `Azure.AI.TextAnalytics` NuGet package - same endpoint and key, different language SDK. Pattern transfers to Java, JavaScript, and other official SDKs.

---

## Environment Variables (Production)

Never hardcode keys in source:

```python
import os
KEY = os.environ['AZURE_LANGUAGE_KEY']
ENDPOINT = os.environ['AZURE_LANGUAGE_ENDPOINT']
```

Use Azure Key Vault, GitHub Secrets, or CI/CD secret stores.

---

## Azure AD Authentication (Brief)

Enterprise apps can use **Azure Active Directory** + RBAC instead of raw keys - better for apps hosted in Azure. See Microsoft docs: "Authenticate Requests to Azure Cognitive Services."

---

## Free Tiers and Billing

Prosise: e.g., **5,000 text samples/month** free for Language sentiment - then per-call charges apply. Monitor usage in Azure Portal → Cost Management.

Custom Vision: train/export to avoid ongoing prediction costs when possible ([Chapter 12](../chapter-12-object-detection/section-07-azure-custom-vision.md)).

---

## Key Rotation Workflow

1. Regenerate Key 2 in portal
2. Update apps to use Key 2
3. Regenerate Key 1
4. Verify no traffic on old key via logs

Two keys prevent downtime during rotation (Prosise explains why portal shows two).

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Key in Git repo | Security incident | Rotate key; use env vars |
| Wrong endpoint region | 404 or auth errors | Match endpoint to resource region |
| Training key for prediction | Auth failure | Custom Vision uses separate keys |
| No 429 retry logic | Flaky production app | Exponential backoff |
| Deprecated Text Analytics API path | 404 | Use unified Language service |

---

## Self-Check

1. What two values do you need for every Cognitive Services call?  
   *Endpoint and subscription key.*
2. Why does Azure provide two keys?  
   *Rotate one without downtime.*
3. What's the advantage of SDK over raw REST?  
   *Simpler API, built-in error handling, versioned methods.*
4. What header carries the subscription key in REST calls?  
   *`Ocp-Apim-Subscription-Key`.*
5. Where should production keys live?  
   *Environment variables or Key Vault - not source code.*

---

## Exercises

1. Create a free Azure Language resource. Call sentiment API from Python (REST or SDK).
2. Intentionally use a wrong key - catch and log `ClientAuthenticationError`.
3. Store keys in `.env` (gitignored) and load with `python-dotenv`.
4. Compare Prosise's REST snippet to SDK - line count and readability.

---

## Testing with curl and Postman

Prosise notes APIs are language-agnostic - verify setup before writing app code:

```bash
curl -X POST "$ENDPOINT/text/analytics/v3.0/sentiment" \
  -H "Ocp-Apim-Subscription-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{"documents":[{"id":"1","text":"Programming is fun"}]}'
```

Postman collections for Cognitive Services are available from Microsoft - useful for sharing repro steps with teammates.

---

## Regional Endpoints

Create resources in the **region closest to your users** for lowest latency. Endpoint URLs embed the region (e.g., `eastus.api.cognitive.microsoft.com`). Keys are **not** portable across regions - one resource, one region pair.

---

## References

- Prosise, Ch. 14 - Keys and Endpoints
- [Authenticate Cognitive Services](https://learn.microsoft.com/azure/ai-services/authentication)
- [Azure pricing calculator](https://azure.microsoft.com/pricing/calculator/)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Next:** [Section 14.3 - Azure Computer Vision](./section-03-azure-computer-vision.md)