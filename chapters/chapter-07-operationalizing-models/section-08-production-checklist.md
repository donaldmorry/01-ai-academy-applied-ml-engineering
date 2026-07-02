# Section 7.8: Production Checklist

> **Source:** Prosise, Ch. 7 - Summary + engineering best practices  
> **Prerequisites:** [Sections 7.1-7.7](./section-01-from-notebook-to-production.md)  
> **Glossary:** [MLOps](../../GLOSSARY.md#mlops) | [ONNX](../../GLOSSARY.md#onnx) | [model](../../GLOSSARY.md#model)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Notebook vs Production

In a notebook, you care about:

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$
> **Readable form:** mean absolute error - average prediction mistake in dollars.

In production, you also care about:

- What if `trip_distance = -5`?
- What if the model file is from last year?
- What if latency spikes at 9 AM Monday?
- What if sklearn 1.5 can't load a 1.3 pickle?

This section is the **checklist** Prosise implies but does not formalize - the bridge from Chapter 7 tactics to full [MLOps](../../GLOSSARY.md#mlops).

---

## The Production Inference Contract

Every deployed model needs an explicit contract:

| Field | Example |
|-------|---------|
| Input schema | 6 floats, documented ranges |
| Output schema | `predicted_fare_usd: float` |
| Model version | `taxi_v3` |
| Training data window | `2023-01-01` to `2024-06-30` |
| Metrics at deploy | MAE = \$3.42 on holdout |
| Owner / on-call | `ml-platform@company.com` |

Store as JSON beside the artifact:

```json
{
  "model_name": "taxi_fare_regressor",
  "version": "taxi_v3",
  "sklearn_version": "1.4.2",
  "trained_at": "2024-06-15T10:00:00Z",
  "features": [
    {"name": "trip_distance", "type": "float", "min": 0.1, "max": 100},
    {"name": "passenger_count", "type": "int", "min": 1, "max": 6}
  ],
  "metrics": {"test_mae_usd": 3.42}
}
```

---

## 1. Input Validation

Never trust client input - validate before `predict()`:

```python
from pydantic import BaseModel, Field, field_validator

class FareRequest(BaseModel):
    trip_distance: float = Field(..., gt=0, le=100)
    passenger_count: int = Field(..., ge=1, le=6)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)
    is_rush: int = Field(..., ge=0, le=1)

    @field_validator('trip_distance')
    @classmethod
    def reasonable_distance(cls, v):
        if v > 50:
            # log outlier - may be GPS error
            pass
        return v
```

| Failure mode | Response |
|--------------|----------|
| Missing field | HTTP 422 + JSON detail |
| Out of range | HTTP 422 |
| Wrong type | HTTP 422 |
| Valid but extreme | Predict + log warning |

**Why it matters:** Out-of-distribution inputs produce confident nonsense. A negative distance might yield a negative fare.

---

## 2. Model Versioning

Prosise: pickles trained on sklearn 1.3 may fail on 1.5. Version **everything**:

| Artifact | Versioning strategy |
|----------|---------------------|
| `.joblib` / `.onnx` | Semantic: `taxi_v3.joblib` |
| Docker image | Git SHA tag: `taxi-api:abc123` |
| API | URL path: `/v1/predict`, `/v2/predict` |
| Metadata | JSON sidecar with metrics + deps |

### Blue/green deployment

```
Traffic ──▶ v2 (new model)  90%
       └──▶ v1 (old model)  10%   ← shadow or canary
```

Compare error rates and latency before full cutover.

### Rollback

Keep `taxi_v2.joblib` in object storage. If v3 MAE degrades in production monitoring, redeploy v2 image - **no retrain required**.

---

## 3. Parity Testing (Pre-Deploy Gate)

Before any promotion:

```python
def assert_parity(sklearn_pipe, onnx_session, X_test, tol=1e-4):
    input_name = onnx_session.get_inputs()[0].name
    for i in range(min(100, len(X_test))):
        row = X_test.iloc[[i]].values.astype('float32')
        sk = sklearn_pipe.predict(row)[0]
        onnx = onnx_session.run(None, {input_name: row})[0].ravel()[0]
        assert abs(sk - onnx) < tol, f'Row {i}: {sk} vs {onnx}'
```

Also test API vs notebook on fixed fixture:

```bash
# golden_test.json - known inputs and expected outputs
pytest tests/test_api_parity.py
```

---

## 4. Monitoring: What to Watch

Notebook metrics are offline. Production needs **online** signals:

| Metric | Why |
|--------|-----|
| **Latency** p50/p95/p99 | SLA breaches, capacity planning |
| **Request rate** | Autoscaling triggers |
| **Error rate** | 4xx validation vs 5xx server |
| **Prediction distribution** | Mean fare drifting? |
| **Input distribution** | Feature drift - new data unlike training |
| **Null / default rate** | Clients sending bad payloads |

### Data drift example

Training: mean `trip_distance` = 3.2 mi.  
Production (this week): mean = 8.1 mi → surge pricing region? GPS bug? Retrain candidate.

$$
\text{drift score} \propto |\mu_{\text{prod}} - \mu_{\text{train}}|
$$
> **Readable form:** if production feature averages move far from training, model accuracy may degrade even though code is unchanged.

Log inputs and outputs (respecting privacy/GDPR) to a warehouse for weekly drift reports.

---

## 5. Observability Stack (Minimal)

```python
import time
import logging

logger = logging.getLogger(__name__)

@app.post('/v1/predict')
def predict_fare(req: FareRequest):
    t0 = time.perf_counter()
    try:
        fare = float(pipe.predict(pd.DataFrame([req.model_dump()]))[0])
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info('predict_ok', extra={
            'model_version': 'taxi_v3',
            'latency_ms': latency_ms,
            'fare': fare,
        })
        return FareResponse(predicted_fare_usd=round(fare, 2))
    except Exception:
        logger.exception('predict_fail')
        raise
```

Ship logs to Datadog, CloudWatch, or ELK. Alert on error rate > 1% or p95 latency > 500 ms.

---

## 6. Security Checklist

| Item | Action |
|------|--------|
| Pickle provenance | Only load from trusted CI builds |
| `debug=False` | Always in Flask/FastAPI production |
| HTTPS | Terminate TLS at load balancer |
| Auth | API keys or OAuth for external clients |
| Rate limiting | Prevent abuse / DoS |
| Non-root container | [Section 7.4](./section-04-docker-for-ml-services.md) |
| Secret management | Vault / env vars, not git |

---

## 7. Performance Checklist

| Item | Guideline |
|------|-----------|
| Load model once | At process startup, not per request |
| Workers | `uvicorn --workers N` - N ≈ CPU cores for CPU models |
| ONNX for edge | Sub-ms vs seconds REST ([Section 7.5](./section-05-onnx-export-and-inference.md)) |
| Batch endpoint | `/v1/predict_batch` for bulk scoring |
| Model size | HashingVectorizer if pickle too large |

---

## 8. Documentation Deliverables

Every deployment ships with:

```markdown
# Taxi Fare API - Deployment README

## Endpoints
- POST /v1/predict - single fare
- GET /health - liveness

## Build
docker build -t taxi-fare-api:1.0.0 .

## Run
docker run -p 8000:8000 taxi-fare-api:1.0.0

## Verify
curl -X POST http://localhost:8000/v1/predict -H 'Content-Type: application/json' \
  -d '{"trip_distance":3.2,...}'

## Model
- Version: taxi_v3
- Test MAE: $3.42
- Trained: 2024-06-15
```

Lab 07 requires this README.

---

## 9. Failure Modes & Playbooks

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| All predictions ↑ 20% | Feature bug in client | Compare raw JSON to training schema |
| 500 on load | sklearn version mismatch | Re-export joblib or pin deps |
| ONNX mismatch | Wrong `initial_types` | Re-export with correct shape |
| Latency spike | Cold start / no workers | Pre-warm containers |
| Excel #ERROR! | Missing pickle path | Fix xlwings working directory |

---

## 10. Chapter 07 Summary Table

| Section | Technique | Primary consumer |
|--------|-----------|------------------|
| 7.1 | Architecture patterns | Engineering leads |
| 7.2 | pickle / joblib | Python services |
| 7.3 | REST + C# HttpClient | Polyglot clients |
| 7.4 | Docker | DevOps / cloud |
| 7.5 | ONNX + ORT | C#, mobile, edge |
| 7.6 | ML.NET | .NET teams |
| 7.7 | Excel xlwings / REST | Business analysts |
| 7.8 | This checklist | Everyone shipping models |

Prosise's chapter summary:

> *Serialize for Python clients. Wrap in REST for other languages. Containerize for portability. Use ONNX to bridge without a web service. Use ML.NET when the stack is .NET. Excel when users live in spreadsheets.*

---

## Graduation: Part I Complete

Chapter 07 completes **Part I - Machine Learning with Scikit-Learn**. You can now:

1. Train models (Chapters 01-06)
2. Serialize and serve them (Chapter 07)
3. Hand them to C#, Excel, and cloud runtimes

Part II ([Chapter 08 - Deep Learning](../chapter-08-deep-learning/README.md)) adds neural networks - exportable via the same ONNX path for deployment.

---

## Check Your Understanding

1. Name three things you monitor in production but ignore in notebooks.
2. What is a parity test and when must it run?
3. How do blue/green deployments reduce rollback risk?
4. Why is input validation a security concern, not just a data quality concern?
5. What metadata should every model artifact carry?

---

## Next Steps

- **Lab 07:** End-to-end taxi deployment capstone  
- **Chapter 08:** Deep learning foundations

**Previous:** [Section 7.7](./section-07-excel-integration.md)  
**Next:** [Lab 07 - Deploy a Taxi Fare Predictor](./section-lab-07-deploy-a-taxi-fare-predictor.md)
