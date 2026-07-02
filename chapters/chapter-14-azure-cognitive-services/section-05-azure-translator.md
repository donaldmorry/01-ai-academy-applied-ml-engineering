# Section 14.5: Azure Translator

> **Source:** Prosise, Ch. 14 - "The Translator Service"  
> **Prerequisites:** [Section 14.2](./section-02-azure-setup-and-authentication.md) | [Section 13.5](../chapter-13-natural-language-processing/section-05-sequence-to-sequence-models.md)  
> **Glossary:** [deep-learning](../../GLOSSARY.md#deep-learning)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## NMT as an API

[Section 13.5](../chapter-13-natural-language-processing/section-05-sequence-to-sequence-models.md) trained English→French LSTM on 50k pairs. **Azure Translator** runs production NMT on 100+ languages via REST.

Contoso Travel translates OCR text from road signs ([Section 14.7](./section-07-contoso-travel-multi-service-app.md)).

> **Humorous analogy:** Your seq2seq model is phrasebook French. Translator is the diplomat who lived in Paris - and bills per paragraph.

> **In plain English:** POST text; receive translation. Auto-detect source language optional.

---

## Setup

Create **Translator** resource. Need KEY, ENDPOINT (Text Translation), and REGION.

```python
import os, requests

KEY = os.environ['AZURE_TRANSLATOR_KEY']
ENDPOINT = os.environ['AZURE_TRANSLATOR_ENDPOINT']
REGION = os.environ['AZURE_TRANSLATOR_REGION']

headers = {
    'Ocp-Apim-Subscription-Key': KEY,
    'Ocp-Apim-Subscription-Region': REGION,
    'Content-type': 'application/json',
}
```

Global resources: omit region header or set `global`.

---

## Language Detection

```python
input_text = [{'text': 'Quand votre nouveau livre sera-t-il disponible?'}]
uri = ENDPOINT + 'detect?api-version=3.0'
r = requests.post(uri, headers=headers, json=input_text)
print(r.json()[0]['language'])  # fr
```

---

## Translation

```python
uri = ENDPOINT + 'translate?api-version=3.0&from=fr&to=en'
r = requests.post(uri, headers=headers, json=input_text)
print(r.json()[0]['translations'][0]['text'])
# When will your new book be available?
```

**Auto-detect** - omit `from`:

```python
uri = ENDPOINT + 'translate?api-version=3.0&to=en'
result = requests.post(uri, headers=headers, json=input_text).json()[0]
print(result['detectedLanguage'])  # fr, score 1.0
print(result['translations'][0]['text'])
```

---

## Batch Translation

```python
batch = [
    {'text': 'Hello, how can I help you?'},
    {'text': 'Your flight has been delayed.'},
]
r = requests.post(ENDPOINT + 'translate?api-version=3.0&to=de',
                  headers=headers, json=batch)
for item in r.json():
    print(item['translations'][0]['text'])
```

---

## Transliteration

Handles script conversion (Thai, Hindi → Latin) - useful for display and search normalization.

---

## Custom Glossaries

Force consistent translation of brand/technical terms via custom models - upload glossary in Portal. Critical for legal and medical content where generic NMT mistranslates.

---

## Document Translation

For **PDFs and DOCX** at scale:
- Azure Blob Storage for source/target
- Async batch API + `azure-ai-translation-document` SDK

Sign photos use OCR + text translation (Contoso), not document API.

---

## vs Chapter 13

| Aspect | Lab LSTM | Translator API |
|--------|----------|----------------|
| Data | 50k pairs | Billions |
| BLEU | Low | Production |
| Privacy | Local | Cloud |
| Custom | Retrain | Glossaries |

---

## Error Handling

```python
try:
    r = requests.post(uri, headers=headers, json=input_text)
    r.raise_for_status()
except requests.exceptions.HTTPError:
    if r.status_code == 429:
        print('Rate limited - backoff')
    elif r.status_code == 401:
        print('Key/region mismatch')
```

---

## Contoso Helper

```python
def translate_lines(lines, target_lang='en'):
    payload = [{'text': ln} for ln in lines if ln.strip()]
    uri = ENDPOINT + f'translate?api-version=3.0&to={target_lang}'
    r = requests.post(uri, headers=headers, json=payload)
    r.raise_for_status()
    return [t['translations'][0]['text'] for t in r.json()]
```

Chains after Vision OCR ([Section 14.3](./section-03-azure-computer-vision.md)).

---

## Key Takeaways

1. **Translator** - NMT for 100+ languages via REST
2. **Region header** for regional resources
3. **Omit `from`** for auto-detect + translate
4. **Batch** strings per request
5. **Document Translation** for PDFs at scale
6. **Chapter 13** teaches mechanics; API delivers quality

---

## Check Your Understanding

1. What three credentials beyond Language key?
2. Detect + translate in one call?
3. Document vs text translation?
4. Common cause of 401?
5. Contoso pipeline role?

---

## Translation Confidence

When chaining detect → translate, only proceed if detection confidence exceeds $\tau$:

$$
\text{translate} \Leftrightarrow \max_{\ell} P(\text{lang}=\ell \mid x) > \tau
$$
> **Readable form:** translate only when the top language hypothesis is sufficiently confident

---

## Glossary and Connections

- [machine-learning](../../GLOSSARY.md#machine-learning) - seq2seq in [Chapter 13](../chapter-13-natural-language-processing/section-05-sequence-to-sequence-models.md)
- [deep-learning](../../GLOSSARY.md#deep-learning) - NMT behind the API

| Custom (Chapter 13) | Translator API |
|--------------------|----------------|
| You train/fine-tune | Microsoft maintains |
| Domain phrases | Custom glossary optional |
| GPU cost | Per-character billing |

---

## Exercises

1. Translate the same paragraph across five languages; log latency.
2. Use auto-detect on mixed English/Spanish input.
3. Document when you would fine-tune MarianMT vs call Translator.

---

## Further Reading

- [GLOSSARY.md](../../GLOSSARY.md) - seq2seq and [machine-learning](../../GLOSSARY.md#machine-learning) terms
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md) - reading probability notation above

---

## References

- Prosise, J. *Applied Machine Learning and AI for Engineers* (O'Reilly, 2023), Ch. 14 - Translator
- [Translator docs](https://learn.microsoft.com/en-us/azure/ai-services/translator/)
- [Section 13.5](../chapter-13-natural-language-processing/section-05-sequence-to-sequence-models.md)

---

**Next:** [Section 14.6 - Azure Speech Services](./section-06-azure-speech-services.md)



