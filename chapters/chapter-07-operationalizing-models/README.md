# Chapter 07: Operationalizing ML Models

> **Source:** *Applied Machine Learning and AI for Engineers*, Chapter 7  
> **Part:** I — Machine Learning with Scikit-Learn  
> **Estimated time:** 8–10 hours  
> **Prerequisites:** Chapters [02](../chapter-02-regression-models/README.md)–[06](../chapter-06-principal-component-analysis/README.md); at least one trained Scikit-Learn model

---

## Chapter Overview

A model that only runs in a Jupyter notebook is a prototype, not a product. This chapter bridges the gap between experimentation and deployment — teaching you to **serialize models**, serve them from **other languages and platforms**, containerize with **Docker**, export to **ONNX** for cross-framework inference, integrate with **ML.NET** for .NET applications, and even embed predictions in **Microsoft Excel**.

You have spent six chapters building models in Python. Now you will learn the engineering practices that make those models usable by web services, desktop apps, mobile clients, and business analysts who will never open a notebook.

This chapter completes Part I of the course. Everything you deploy here — taxi fare regressors, fraud classifiers, text pipelines — can be productionized with the techniques below.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Serialize and deserialize Scikit-Learn models with Pickle and joblib
2. Consume pickled Python models from a C# application
3. Containerize a model inference service with Docker
4. Export models to ONNX format for cross-platform deployment
5. Load and run ONNX models with ONNX Runtime (Python and C#)
6. Integrate ML.NET for training and inference in .NET ecosystems
7. Embed ML predictions in Excel workbooks for business users
8. Design a minimal production inference API with input validation and versioning

---

## Sections

| # | Section | File | Topics |
|---|--------|------|--------|
| 7.1 | From Notebook to Production | [01-from-notebook-to-production.md](./section-01-from-notebook-to-production.md) | Model artifacts; inference vs training; deployment patterns |
| 7.2 | Model Serialization with Pickle & Joblib | [02-model-serialization-pickle-joblib.md](./section-02-model-serialization-with-pickle-and-joblib.md) | Saving/loading pipelines; security; versioning; [MLOps](../../GLOSSARY.md#mlops) |
| 7.3 | Consuming Models from C# | [03-consuming-models-from-csharp.md](./section-03-consuming-models-from-c-sharp.md) | REST API (Flask/FastAPI); C# HttpClient; latency trade-offs |
| 7.4 | Docker for ML Services | [04-docker-for-ml-services.md](./section-04-docker-for-ml-services.md) | Dockerfile; packaging model + deps; health checks; compose |
| 7.5 | ONNX Export & Inference | [05-onnx-export-inference.md](./section-05-onnx-export-and-inference.md) | `skl2onnx`; ONNX Runtime; C# scoring; parity tests |
| 7.6 | ML.NET Overview | [06-mlnet-overview.md](./section-06-ml-net-overview.md) | Training/scoring in C#; IDataView; Model Builder; ONNX import |
| 7.7 | Excel Integration | [07-excel-integration.md](./section-07-excel-integration.md) | xlwings UDFs; Power Query REST; taxi/sentiment functions |
| 7.8 | Production Checklist | [08-production-checklist.md](./section-08-production-checklist.md) | Validation, monitoring, versioning, rollback, security |

---

## Lab

**[Lab 07: Deploy a Taxi Fare Predictor](./section-lab-07-deploy-a-taxi-fare-predictor.md)**

Take the regression model from Chapter 02 and operationalize it end-to-end:

1. **Serialize** the trained pipeline (preprocessor + model) with joblib
2. **Export to ONNX** and verify predictions match the Python model within tolerance
3. **Build a Docker container** running a FastAPI or Flask inference endpoint
4. **Test the API** with curl or Postman — send JSON features, receive fare prediction
5. **(Optional)** Load the ONNX model in a simple C# console app with ONNX Runtime
6. **(Optional)** Create an Excel workbook that calls the API for batch predictions

*Deliverable:* Dockerfile, API source code, ONNX model file, README with deployment instructions, and screenshot or log of successful inference.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Model serialization & APIs | MLOps, CI/CD for ML, feature stores (Course 2) |
| Docker containerization | Kubernetes, cloud deployment (Course 2) |
| ONNX interoperability | TensorFlow Lite, edge deployment (Chapter 10–12) |
| ML.NET | Enterprise .NET ML ecosystems |
| Production monitoring | Drift detection, A/B testing (Course 2) |

---

## Prerequisites

- Completed at least one modeling chapter (02–06) with a saved trained pipeline
- Basic command-line familiarity (cd, pip, docker build)
- Docker installed (or access to a container runtime)
- For C#/ML.NET sections: .NET SDK (optional track)
- For Excel section: Microsoft Excel (optional track)

---

## Glossary Terms Introduced

| Term | Definition |
|------|------------|
| [ONNX](../../GLOSSARY.md#onnx) | Open Neural Network Exchange — cross-platform model format |
| [MLOps](../../GLOSSARY.md#mlops) | ML operations — production lifecycle for models |

---

## Key Takeaways

- Always serialize the full pipeline (preprocessing + model), not just the estimator
- Pickle is convenient but not secure for untrusted sources — treat model files like code
- [ONNX](../../GLOSSARY.md#onnx) enables one training pipeline and many deployment targets
- Docker standardizes the "it works on my machine" problem for ML services
- Production ML requires versioning, input validation, and monitoring — not just accuracy
- [MLOps](../../GLOSSARY.md#mlops) extends serialization into registry, CI/CD, and drift detection (Course 2)

---

## Self-Assessment

1. Why should you save the entire Scikit-Learn Pipeline rather than only the final estimator?
2. What are the security risks of loading a pickle file from an untrusted source?
3. When would you choose ONNX over pickle for deployment?
4. What belongs in a Docker image for an ML inference service?
5. How do you verify that an ONNX export produces identical predictions to the original model?
6. What is the advantage of ML.NET for a team that builds primarily in C#?
7. What three things would you monitor in production that you ignore during notebook development?

---

**Previous:** [Chapter 06 — Principal Component Analysis](../chapter-06-principal-component-analysis/README.md)  
**Next:** [Chapter 08 — Deep Learning Foundations](../chapter-08-deep-learning/README.md)
