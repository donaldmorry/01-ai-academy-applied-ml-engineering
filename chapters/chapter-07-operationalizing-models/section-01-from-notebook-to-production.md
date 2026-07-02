# Section 7.1: From Notebook to Production

> **Source:** Prosise, Ch. 7 - "Operationalizing Machine Learning Models"  
> **Prerequisites:** At least one trained model from [Chapters 02-06](../../README.md)  
> **Glossary:** [model](../../GLOSSARY.md#model) | [MLOps](../../GLOSSARY.md#mlops)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Gap Between Experiment and Product

You have spent six chapters training models in Jupyter notebooks. You split data, tuned hyperparameters, and celebrated when test [MAE](../../GLOSSARY.md#mae) dropped below \$5.00. That is real progress - but a model that only runs inside a notebook is a **prototype**, not a product.

**Operationalizing** a model means making its predictions available to applications, services, and people who will never open Python:

| Consumer | Example |
|----------|---------|
| Mobile app | Rider sees estimated taxi fare before booking |
| C# desktop app | Fraud analyst scores transactions in a WinForms UI |
| Excel workbook | Communications team monitors social sentiment |
| Batch pipeline | Nightly job flags suspicious accounts |

Prosise frames the central question of Chapter 7 this way: *How do you invoke models written in Python from apps written in other languages, on any platform?*

---

## Training vs Inference

Two distinct phases - conflating them causes production bugs.

| Phase | When | What happens | Compute profile |
|-------|------|--------------|-----------------|
| **Training** | Offline, periodic | Fit weights on historical data | Heavy (minutes-hours) |
| **Inference** | Online, per request | Apply learned weights to new inputs | Light (milliseconds) |

$$
\hat{y} = f_{\theta^*}(\mathbf{x})
$$
> **Readable form:** prediction equals the model function $f$ evaluated at new input $\mathbf{x}$, using parameters $\theta^*$ learned during training.

In a notebook you often retrain on every run. In production you **train once**, serialize the artifact, and **load** it thousands of times per day.

```python
# NOT production - retrains every startup
model = LinearRegression()
model.fit(X_train, y_train)
fare = model.predict([[3.2, 2, 14, 4, 0, 1]])[0]

# Production pattern - load once, predict many
import joblib
model = joblib.load('models/taxi_pipeline.joblib')
fare = model.predict([[3.2, 2, 14, 4, 0, 1]])[0]
```

---

## What Is a Model Artifact?

A **model artifact** is everything needed to reproduce predictions without retraining:

1. **Preprocessor state** - fitted `StandardScaler` means, `CountVectorizer` vocabulary, one-hot category maps
2. **Estimator parameters** - regression coefficients, tree structures, SVM support vectors
3. **Metadata** (best practice) - sklearn version, training date, feature schema, metrics

> **In plain English:** Save the whole kitchen (oven + recipe + spices), not just the finished cake. If you save only the final estimator, production inputs arrive raw and predictions are nonsense.

From [Section 4.8](../chapter-04-text-classification/section-08-pipelines-and-production-text-ml.md):

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=20)),
    ('clf', LogisticRegression(max_iter=1000)),
])
pipe.fit(X_train, y_train)
# Serialize `pipe`, not `pipe.named_steps['clf']` alone
```

---

## Deployment Architecture Patterns

Prosise illustrates two strategies (Figure 7-1 in the book). Both solve the same problem - Python-trained model, non-Python consumer - with different trade-offs.

### Pattern A: REST API (Web Service)

Wrap the model in a Python web service (Flask, FastAPI). Any client that speaks HTTP can call it.

```
[Mobile App] ──HTTP POST──▶ [Flask/FastAPI + pickled model] ──▶ JSON prediction
[C# Client]  ──HTTP GET───▶
[Excel]      ──HTTP───────▶  (via Power Query or VBA)
```

**Pros:** Language-agnostic; central updates; easy auth and logging at the gateway  
**Cons:** Network latency; requires always-on server; serialization overhead per request

Typical round-trip for a local Flask sentiment service: **~2+ seconds** (Prosise's benchmark). Remote hosting adds more.

### Pattern B: ONNX + Native Runtime

Export the sklearn model to [ONNX](../../GLOSSARY.md#onnx) format. Load with **ONNX Runtime** in C#, Java, C++, JavaScript, etc. - no Python process required.

```
[Python training] ──skl2onnx──▶ taxi.onnx ──▶ [C# InferenceSession] ──▶ prediction
```

**Pros:** Sub-millisecond inference locally (~0.001 s in Prosise's test); no Python on client; edge/mobile friendly  
**Cons:** Not every sklearn estimator converts cleanly; export must be validated; ops tooling differs from pickle

### Pattern C: Same-Language Stack

Train and score in the client's language - e.g., [ML.NET](./section-06-ml-net-overview.md) for C# teams. No bridge required.

**Pros:** Type safety, single toolchain, compiled performance  
**Cons:** Retrain in a second framework; feature parity gaps vs sklearn

### Pattern D: Embedded Python (Advanced)

Libraries like Python.NET embed a Python interpreter inside a .NET process. Rare in greenfield projects; useful for legacy bridges.

---

## Choosing a Pattern

| Situation | Recommended pattern |
|-----------|---------------------|
| Many clients, mixed languages | REST API + Docker ([Section 7.4](./section-04-docker-for-ml-services.md)) |
| Low-latency C# desktop/mobile | ONNX ([Section 7.5](./section-05-onnx-export-and-inference.md)) |
| .NET-only team, greenfield | ML.NET ([Section 7.6](./section-06-ml-net-overview.md)) |
| Business analysts in Excel | xlwings UDF or REST-backed Excel ([Section 7.7](./section-07-excel-integration.md)) |
| Prototype / internal Python only | joblib + script ([Section 7.2](./section-02-model-serialization-with-pickle-and-joblib.md)) |

---

## The Taxi Fare Story (Thread Through Chapter 07)

[Section 2.8](../chapter-02-regression-models/section-08-taxi-fare-prediction-end-to-end-project.md) built a fare regressor. Chapter 07 deploys it end-to-end:

| Step | Section | Deliverable |
|------|--------|-------------|
| Serialize pipeline | 7.2 | `taxi_pipeline.joblib` |
| REST API | 7.3-7.4 | Flask/FastAPI + Docker |
| ONNX export | 7.5 | `taxi.onnx` |
| C# scoring | 7.5-7.6 | Console app with ORT or ML.NET |
| Excel batch scoring | 7.7 | Workbook with `=predict_fare(...)` |
| Production hardening | 7.8 | Validation, versioning, monitoring |

Prosise's taxi ONNX example expects three floats: day of week (0-6), hour (0-23), distance (miles). Your Chapter 02 model may have more features - the **principle** is identical; adjust `initial_types` accordingly.

---

## Inference API Design (Preview)

Even a minimal service needs a contract. Sketch this before writing code ([Section 7.8](./section-08-production-checklist.md) expands):

```json
// POST /v1/predict
// Request
{
  "trip_distance": 3.2,
  "passenger_count": 2,
  "hour": 14,
  "day_of_week": 4
}

// Response
{
  "predicted_fare_usd": 18.47,
  "model_version": "taxi_v3",
  "latency_ms": 12
}
```

**Version the URL or header** (`/v1/`, `X-Model-Version`) so you can roll out `taxi_v4` without breaking existing clients.

---

## MLOps: The Bigger Picture

Prosise introduces [MLOps](../../GLOSSARY.md#mlops) when discussing pickle versioning: *How do you store models in a central repository? Deploy to devices? Version models alongside datasets?*

This chapter teaches **tactics** (serialize, containerize, export). MLOps teaches **strategy**:

- Model registry (MLflow, Azure ML, Weights & Biases)
- CI/CD pipelines that retrain on schedule or data drift
- Feature stores for consistent online/offline features
- A/B testing and shadow deployments

You will deepen these in Course 2. For now, adopt three habits:

1. **Pin dependency versions** in `requirements.txt` / Docker image
2. **Tag every artifact** with version string + git commit + metrics
3. **Never deploy without a parity test** - same input, notebook vs production, within tolerance

---

## Common Anti-Patterns

| Anti-pattern | Why it fails |
|--------------|--------------|
| Saving only `model`, not `Pipeline` | Raw features don't match training format |
| Hardcoding file paths | Breaks in Docker / cloud |
| No input validation | Garbage in → garbage out; security risk |
| Retraining on every API call | Unbounded latency and cost |
| Skipping ONNX parity check | Silent numerical divergence |
| `debug=True` Flask in production | Code execution risk, poor performance |

---

## Mental Model Summary

```
Experimentation (Chapters 01-06)          Operationalization (Chapter 07)
─────────────────────────────────        ─────────────────────────────────
Notebook, EDA, CV, metrics        →      Artifact, API, container, monitor
"What algorithm wins?"            →      "How do clients call it safely?"
Python-only                       →      Python + C# + Excel + HTTP
```

---

## Check Your Understanding

1. What is the difference between training and inference?
2. Why must you serialize the full `Pipeline`, not just the final estimator?
3. When would you choose a REST API over ONNX for a C# client?
4. What three items belong in model metadata alongside the `.joblib` file?
5. What problem does MLOps solve that pickle alone does not?

---

## Next Steps

- **Section 7.2:** Serialize models with pickle and joblib - security, size, versioning  
- **Lab 07:** Deploy the taxi fare predictor end-to-end

**Previous:** [Chapter 06 - PCA](../chapter-06-principal-component-analysis/README.md)  
**Next:** [Section 7.2 - Model Serialization](./section-02-model-serialization-with-pickle-and-joblib.md)



