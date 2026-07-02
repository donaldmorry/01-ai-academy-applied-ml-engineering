# Lab 07: Deploy a Taxi Fare Predictor

> **Prerequisites:** [Chapter 02 Lab](../chapter-02-regression-models/section-lab-02-taxi-fare-prediction.md) (trained taxi model) | [Sections 7.1-7.8](./section-01-from-notebook-to-production.md)  
> **Estimated time:** 4-5 hours  
> **Glossary:** [ONNX](../../GLOSSARY.md#onnx) | [MLOps](../../GLOSSARY.md#mlops) | [MAE](../../GLOSSARY.md#mae)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Objectives

By the end of this lab you will:

1. **Serialize** the Chapter 02 taxi pipeline with joblib
2. **Export to ONNX** and verify predictions match Python within tolerance
3. **Build a Docker container** with a FastAPI inference endpoint
4. **Test the API** with curl - JSON features in, fare (USD) out
5. **(Optional)** Score with ONNX Runtime in C#
6. **(Optional)** Batch-score rows from Excel via REST

This lab operationalizes Prosise Chapter 7's taxi fare example - the same three-feature ONNX demo extended to your full Chapter 02 pipeline.

**Deliverable:** `Dockerfile`, API source, `taxi_pipeline.onnx`, `DEPLOY.md`, screenshot or log of successful inference.

---

## Setup

```bash
mkdir -p lab-07-deploy/{models,src,tests}
cd lab-07-deploy

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install pandas numpy scikit-learn joblib \
    fastapi uvicorn pydantic \
    skl2onnx onnxruntime onnx \
    pytest httpx
```

Reuse your [Lab 02](../chapter-02-regression-models/section-lab-02-taxi-fare-prediction.md) trained pipeline, or run the synthetic trainer from that lab. Minimum artifacts:

```python
import joblib
from sklearn.metrics import mean_absolute_error

# After training pipe on Chapter 02 features:
mae = mean_absolute_error(y_test, pipe.predict(X_test))
print(f'Test MAE: ${mae:.2f}')
joblib.dump(pipe, 'models/taxi_pipeline.joblib')
X_test.to_csv('models/X_test_sample.csv', index=False)
```

Features: `trip_distance`, `passenger_count`, `hour`, `day_of_week`, `is_weekend`, `is_rush`.

---

## Part A: Serialize & Metadata (30 min)

### Task A1 - Save pipeline

Confirm `models/taxi_pipeline.joblib` loads and predicts:

```python
import joblib
loaded = joblib.load('models/taxi_pipeline.joblib')
print(loaded.predict(X_test.iloc[[0]]))
```

### Task A2 - Write metadata sidecar

```python
import json, sklearn
from datetime import datetime, timezone

meta = {
    'model_name': 'taxi_fare_regressor',
    'version': 'taxi_v1',
    'sklearn_version': sklearn.__version__,
    'trained_at': datetime.now(timezone.utc).isoformat(),
    'features': features,
    'test_mae_usd': float(mae),
}
with open('models/taxi_pipeline.json', 'w') as f:
    json.dump(meta, f, indent=2)
```

> **In plain English:** Future you (and [MLOps](../../GLOSSARY.md#mlops) tooling) needs to know which sklearn version trained this file.

---

## Part B: ONNX Export & Parity (45 min)

### Task B1 - Export

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

n_features = pipe.named_steps['scaler'].n_features_in_
initial_type = [('float_input', FloatTensorType([None, n_features]))]

onnx_model = convert_sklearn(
    pipe,
    initial_types=initial_type,
    target_opset=17,
)

with open('models/taxi_pipeline.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
print('Exported models/taxi_pipeline.onnx')
```

### Task B2 - Parity test

Create `tests/test_onnx_parity.py`:

```python
import numpy as np
import joblib
import onnxruntime as rt
import pandas as pd

pipe = joblib.load('models/taxi_pipeline.joblib')
session = rt.InferenceSession('models/taxi_pipeline.onnx')
input_name = session.get_inputs()[0].name

X = pd.read_csv('models/X_test_sample.csv').head(50).values.astype(np.float32)

for i, row in enumerate(X):
    sk = pipe.predict(row.reshape(1, -1))[0]
    onnx = session.run(None, {input_name: row.reshape(1, -1)})[0].ravel()[0]
    assert abs(sk - onnx) < 1e-3, f'Row {i}: sklearn={sk}, onnx={onnx}'

print('ONNX parity: PASS (50 rows)')
```

```bash
pytest tests/test_onnx_parity.py -v
```

If assertions fail, check feature order and `target_opset`.

---

## Part C: FastAPI Service (60 min)

### Task C1 - Create `src/taxi_api.py`

```python
"""Taxi fare inference API."""
from pathlib import Path
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv('MODEL_PATH', 'models/taxi_pipeline.joblib'))
pipe = joblib.load(MODEL_PATH)
MODEL_VERSION = 'taxi_v1'

app = FastAPI(title='Taxi Fare API', version=MODEL_VERSION)


class FareRequest(BaseModel):
    trip_distance: float = Field(..., gt=0, le=100)
    passenger_count: int = Field(..., ge=1, le=6)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)
    is_rush: int = Field(..., ge=0, le=1)


class FareResponse(BaseModel):
    predicted_fare_usd: float
    model_version: str


@app.get('/health')
def health():
    return {'status': 'ok', 'model_version': MODEL_VERSION}


@app.post('/v1/predict', response_model=FareResponse)
def predict(req: FareRequest):
    row = pd.DataFrame([req.model_dump()])
    fare = float(pipe.predict(row)[0])
    if fare < 0:
        raise HTTPException(422, 'Model returned negative fare - check input')
    return FareResponse(
        predicted_fare_usd=round(fare, 2),
        model_version=MODEL_VERSION,
    )
```

### Task C2 - Local smoke test

```bash
uvicorn src.taxi_api:app --reload --port 8000
```

```bash
curl -s -X POST http://localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"trip_distance":4.0,"passenger_count":1,"hour":17,
       "day_of_week":4,"is_weekend":0,"is_rush":1}'
```

Save output; then `docker stop taxi-lab && docker rm taxi-lab`.

### Task C3 - API tests

Add `tests/test_api.py` using `TestClient` from FastAPI - test `/health`, valid POST (200 + positive fare), and invalid `trip_distance` (422). See [Section 7.3](./section-03-consuming-models-from-c-sharp.md) for the full pattern.

```bash
pytest tests/ -v
```

---

## Part D: Docker (60 min)

### Task D1 - `requirements.txt`

```text
fastapi==0.110.0
uvicorn[standard]==0.27.1
joblib==1.3.2
pandas==2.2.0
scikit-learn==1.4.2
pydantic==2.6.0
```

Pin the same sklearn version used during training.

### Task D2 - `Dockerfile`

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/taxi_api.py src/
COPY models/taxi_pipeline.joblib models/
COPY models/taxi_pipeline.json models/

ENV MODEL_PATH=/app/models/taxi_pipeline.joblib
ENV PYTHONPATH=/app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "src.taxi_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Task D3 - Build and run

```bash
docker build -t taxi-fare-api:lab07 .
docker run -d --name taxi-lab -p 8000:8000 taxi-fare-api:lab07

curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"trip_distance":4.0,"passenger_count":1,"hour":17,
       "day_of_week":4,"is_weekend":0,"is_rush":1}'
```

```bash
docker logs taxi-lab
docker stop taxi-lab && docker rm taxi-lab
```

---

## Part E: Optional Extensions

| Track | Task | Reference |
|-------|------|-----------|
| **C# ONNX** | `dotnet add package Microsoft.ML.OnnxRuntime`; score `taxi_pipeline.onnx` | [Section 7.5](./section-05-onnx-export-and-inference.md) |
| **Excel** | Power Query POST to `/v1/predict` or xlwings UDF | [Section 7.7](./section-07-excel-integration.md) |

---

## Part F: `DEPLOY.md` + Submission

Document build/run/verify steps (model version, test MAE, `docker build`, `curl` command, `pytest`). Include a screenshot or log of successful Docker inference.

### Rubric (100 + 10 bonus)

joblib/metadata 15 | ONNX parity 20 | FastAPI 20 | Docker+curl 25 | DEPLOY.md 10 | pytest 10 | optional C#/Excel +10

### Troubleshooting

Pickle error → pin sklearn. ONNX mismatch → feature order. `ModuleNotFoundError: src` → `PYTHONPATH=/app`. 422 → check `is_weekend`/`is_rush` are 0/1.

### Checklist

- [ ] `models/taxi_pipeline.{joblib,onnx,json}` + `src/taxi_api.py`
- [ ] `Dockerfile`, `requirements.txt`, tests, `DEPLOY.md`, inference log

**Previous:** [Section 7.8](./section-08-production-checklist.md)  
**Next:** [Chapter 08 - Deep Learning Foundations](../chapter-08-deep-learning/README.md)