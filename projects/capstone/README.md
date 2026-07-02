# Capstone Project: End-to-End ML System

> **Course:** 1 — Applied ML & AI for Engineers  
> **When:** After completing Chapters 01–14  
> **Estimated time:** 15–25 hours

---

## Project Overview

Demonstrate mastery of Course 1 by building a **complete machine learning system** — from raw data to deployed model. This mirrors real engineering work: messy data, algorithm selection, evaluation, and production deployment.

---

## Requirements

### 1. Problem Selection
Choose one of these domains or propose your own (approved):
- **Healthcare:** Predict patient readmission risk
- **Finance:** Credit default or fraud detection
- **Retail:** Customer churn prediction
- **Manufacturing:** Equipment failure prediction
- **Custom:** Your domain with instructor approval

### 2. Technical Requirements

| Component | Requirement |
|-----------|------------|
| Data exploration | EDA with visualizations, missing value analysis |
| Classical ML | ≥2 algorithms from Part I (Chapters 01–07) |
| Deep learning | ≥1 neural network or CNN from Part II |
| Text or vision | At least one unstructured data modality |
| Deployment | ONNX export OR Docker container OR API endpoint |
| Evaluation | Multiple metrics with error analysis |
| Documentation | README explaining decisions and tradeoffs |

### 3. Deliverables

```
capstone/
├── README.md              # Problem, approach, results, sections learned
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_classical_ml.ipynb
│   ├── 03_deep_learning.ipynb
│   └── 04_evaluation.ipynb
├── src/                   # Production code (if applicable)
├── models/                # Saved models (ONNX/pickle)
├── Dockerfile             # If containerized
└── requirements.txt
```

---

## Evaluation Rubric

| Criterion | Weight | Excellent |
|-----------|--------|-----------|
| Problem formulation | 15% | Clear metric, justified approach |
| Data preparation | 15% | Thorough EDA, proper splits, no leakage |
| Model selection | 20% | Compared multiple approaches with justification |
| Deep learning | 15% | Appropriate architecture, proper training |
| Deployment | 15% | Working production artifact |
| Documentation | 10% | Clear, reproducible, honest about limitations |
| Presentation | 10% | Can explain every decision to a non-technical stakeholder |

---

## Milestone: 🟢 ML Practitioner

Completing this capstone + all 14 chapters earns the **ML Practitioner** badge and qualifies you to begin Course 2.
