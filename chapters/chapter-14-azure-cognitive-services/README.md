# Chapter 14: Azure Cognitive Services

> **Source:** *Applied Machine Learning and AI for Engineers*, Chapter 14  
> **Part:** II — Deep Learning with Keras & TensorFlow  
> **Estimated time:** 6–8 hours  
> **Prerequisites:** Chapters [01](../chapter-01-machine-learning/README.md)–[13](../chapter-13-natural-language-processing/README.md); Azure account (free tier sufficient)

---

## Chapter Overview

Not every problem requires training a custom model. **Azure Cognitive Services** provides production-ready AI APIs for vision, language, speech, and decision tasks — letting you integrate world-class models into applications with REST calls instead of GPUs and training pipelines.

This capstone chapter shows when to **build vs buy**: you will use Azure Computer Vision for image analysis, Azure Language for sentiment and entity extraction, Azure Translator for multilingual support, Azure Speech for transcription and synthesis, and assemble them into **Contoso Travel** — a conversational travel assistant that combines multiple cognitive services into one cohesive application.

After 13 chapters of hands-on model building, this chapter teaches the equally important skill of **architecting AI solutions** that mix custom models (Chapters 02–13) with managed cloud services — the pattern most enterprise applications follow.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Navigate the Azure Cognitive Services portfolio and choose the right service for a task
2. Authenticate and call Cognitive Services REST APIs from Python
3. Analyze images with Computer Vision (tags, captions, OCR, adult content detection)
4. Perform sentiment analysis, key phrase extraction, and entity recognition with Azure Language
5. Translate text between languages with Azure Translator
6. Transcribe speech to text and synthesize text to speech with Azure Speech
7. Design a multi-service application architecture (Contoso Travel assistant)
8. Compare cloud API approach vs custom models from earlier chapters — cost, latency, control, privacy

---

## Sections

| # | Section | File | Topics |
|---|--------|------|--------|
| 14.1 | Build vs Buy in AI | [01-build-vs-buy-in-ai.md](./section-01-build-vs-buy-in-ai.md) | When custom models win; when APIs win; hybrid architectures |
| 14.2 | Azure Setup & Authentication | [02-azure-setup-authentication.md](./section-02-azure-setup-and-authentication.md) | Resource groups; keys and endpoints; SDK vs REST; pricing tiers |
| 14.3 | Azure Computer Vision | [03-azure-computer-vision.md](./section-03-azure-computer-vision.md) | Image tagging, description, OCR; spatial analysis; Custom Vision connection |
| 14.4 | Azure Language Services | [04-azure-language-services.md](./section-04-azure-language-services.md) | Sentiment analysis; key phrases; named entity recognition (NER); PII detection |
| 14.5 | Azure Translator | [05-azure-translator.md](./section-05-azure-translator.md) | Text translation; language detection; custom glossaries |
| 14.6 | Azure Speech Services | [06-azure-speech-services.md](./section-06-azure-speech-services.md) | Speech-to-text; text-to-speech; neural voices; real-time transcription |
| 14.7 | Contoso Travel: Multi-Service App | [07-contoso-travel-multi-service-app.md](./section-07-contoso-travel-multi-service-app.md) | Architecture design; chaining services; error handling; user experience |
| 14.8 | Production Considerations | [08-production-considerations.md](./section-08-production-considerations.md) | Rate limits, retries, caching; data residency; responsible AI dashboard |

---

## Lab

**[Lab 14: Contoso Travel Assistant](./section-lab-14-contoso-travel-assistant.md)**

Build an end-to-end travel assistant that combines multiple Azure Cognitive Services:

1. **User speaks** a travel question → Azure Speech transcribes to text
2. **Language service** extracts intent: destination, dates, entities (cities, landmarks)
3. **Translator** handles multilingual input/output if needed
4. **Computer Vision** analyzes uploaded travel photos — tags landmarks, reads signs (OCR)
5. **Language service** generates a sentiment-aware response summary
6. **Speech service** reads the response aloud

Implement as a Python CLI or simple web app. Handle API errors gracefully. Log which services were called for each request.

*Deliverable:* Working application code, architecture diagram, and a decision matrix documenting when you would replace each API call with a custom model from earlier chapters.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Cloud AI APIs | Full Azure ML, model registry, MLOps (Course 2) |
| Computer Vision API | Custom vision training at scale (Chapter 12) |
| Language & NER | Information extraction, knowledge graphs (Course 2, Ch 23) |
| Speech services | End-to-end speech models, Whisper (Course 3) |
| Multi-service architecture | Microservices, event-driven AI (Course 2) |

---

## Prerequisites

- Completion of Chapters 01–13 (or equivalent ML/DL background)
- Azure account with Cognitive Services access (free F0 tiers available)
- Python with `azure-ai-*` SDKs or `requests` for REST calls
- Microphone optional for speech input demo
- [Chapter 07](../chapter-07-operationalizing-models/README.md) deployment concepts helpful

---

## Key Takeaways

- Cognitive Services let you ship AI features in hours, not weeks — ideal for standard tasks
- Custom models (Chapters 02–13) win when you need domain specificity, privacy, or cost at scale
- Production AI apps often combine both: APIs for commodity tasks, custom models for differentiation
- Always handle API failures, rate limits, and latency in multi-service architectures
- Review Azure's Responsible AI tools and data processing terms before handling user data

---

## Self-Assessment

1. When would you train a custom sentiment model (Chapter 04/13) instead of using Azure Language?
2. What authentication information do you need for every Cognitive Services API call?
3. How does Azure Computer Vision OCR differ from training a custom digit recognizer (Chapter 03)?
4. What are the tradeoffs of sending user speech to a cloud transcription service?
5. How would you structure Contoso Travel to minimize API calls and cost?
6. What happens when a Cognitive Services API returns a 429 status code?
7. Which Chapter 12 capability overlaps with Azure Custom Vision, and when would you use each?

---

**Previous:** [Chapter 13 — Natural Language Processing](../chapter-13-natural-language-processing/README.md)  
**Next:** [Course Capstone Project](../../projects/capstone/README.md)

---

## Course Completion

Congratulations — completing this chapter finishes **Course 1: Applied Machine Learning & AI for Engineers**. You have earned the **ML Practitioner** milestone.

You can now build regression and classification systems, deploy models to production, train neural networks and CNNs, detect objects and faces, process natural language, and integrate cloud AI services — the full applied ML engineering toolkit.
