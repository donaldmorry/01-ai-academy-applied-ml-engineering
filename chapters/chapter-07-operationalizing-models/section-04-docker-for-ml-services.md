# Section 7.4: Docker for ML Services

> **Source:** Prosise, Ch. 7 - "Containerizing a Machine Learning Model"  
> **Prerequisites:** [Section 7.3](./section-03-consuming-models-from-c-sharp.md)  
> **Glossary:** [MLOps](../../GLOSSARY.md#mlops)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The "Works on My Machine" Problem

A Flask sentiment service needs:

- Python 3.x
- Flask, numpy, scipy, scikit-learn
- `app.py` + `sentiment.pkl`
- Compatible sklearn version for the pickle

Installing that stack on every developer laptop, CI agent, and production VM is fragile. **Containers** bundle the app and all dependencies into an immutable image.

> **In plain English:** A Docker image is a snapshot of a mini-computer - OS libraries, Python, packages, your code, your model - that runs identically everywhere Docker runs.

Prosise: *"Think of containers as lightweight VMs that start quickly and consume far less memory."*

---

## Core Concepts

| Term | Definition |
|------|------------|
| **Image** | Read-only blueprint (like a class in OOP) |
| **Container** | Running instance of an image (like an object) |
| **Dockerfile** | Text recipe to build an image |
| **Registry** | Storage for images (Docker Hub, ACR, ECR) |

```
Dockerfile  ──docker build──▶  Image  ──docker run──▶  Container (listening :8000)
```

---

## Prosise's Minimal Dockerfile (Sentiment + Flask)

```dockerfile
FROM python:3.11-slim

RUN pip install --no-cache-dir flask numpy scipy scikit-learn==1.4.2 && \
    mkdir /app

COPY app.py /app
COPY models/sentiment.pkl /app/models/sentiment.pkl

WORKDIR /app
EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["app.py"]
```

Build and run:

```bash
docker build -t sentiment-server .
docker run -p 5000:5000 sentiment-server
```

Test:

```bash
curl -G "http://localhost:5000/analyze" \
  --data-urlencode "text=Great food and excellent service!"
```

Cloud deployment (Prosise Azure example):

```text
http://your-service.region.azurecontainer.io:5000/analyze?text=...
```

Clients need only HTTP - no Python on their machines.

---

## Production Dockerfile: Taxi Fare FastAPI

### Project layout

```text
taxi-deploy/
├── Dockerfile
├── requirements.txt
├── taxi_api.py
└── models/
    └── taxi_pipeline.joblib
```

### `requirements.txt`

```text
fastapi==0.110.0
uvicorn[standard]==0.27.1
joblib==1.3.2
pandas==2.2.0
scikit-learn==1.4.2
pydantic==2.6.0
```

### `Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Non-root user for security
RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY taxi_api.py .
COPY models/taxi_pipeline.joblib models/

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "taxi_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### Build and run

```bash
cd taxi-deploy
docker build -t taxi-fare-api:1.0.0 .
docker run -d --name taxi-api -p 8000:8000 taxi-fare-api:1.0.0

curl -X POST http://localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"trip_distance":4.0,"passenger_count":1,"hour":17,
       "day_of_week":4,"is_weekend":0,"is_rush":1}'
```

---

## What Belongs in the Image?

| Include | Exclude |
|---------|---------|
| Inference code (`taxi_api.py`) | Training notebooks |
| Serialized model (`.joblib`) | Raw training CSVs (unless needed) |
| `requirements.txt` with pinned versions | Dev tools (jupyter, pytest) unless test image |
| Minimal Python base image | Full CUDA stack (unless GPU inference) |

**Multi-stage builds** (advanced): train in a fat image; copy only `models/` + API into a slim runtime image.

```dockerfile
# Stage 1: train (optional CI step)
FROM python:3.11 AS trainer
COPY train.py data/ ./
RUN pip install pandas scikit-learn && python train.py

# Stage 2: serve
FROM python:3.11-slim
COPY --from=trainer /app/models/taxi_pipeline.joblib models/
COPY taxi_api.py requirements.txt ./
RUN pip install -r requirements.txt
CMD ["uvicorn", "taxi_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Image Size Optimization

| Technique | Effect |
|-----------|--------|
| `python:3.11-slim` vs `python:3.11` | −500 MB+ |
| `--no-cache-dir` on pip | Smaller layers |
| Multi-stage build | No training deps in prod |
| `HashingVectorizer` vs `CountVectorizer` | Smaller model file (Prosise: 50 MB → 8 MB) |

```bash
docker images taxi-fare-api
# REPOSITORY      TAG    SIZE
# taxi-fare-api   1.0.0  ~450MB
```

---

## Environment Variables & Configuration

Never bake secrets into images. Inject at runtime:

```bash
docker run -p 8000:8000 \
  -e MODEL_PATH=/app/models/taxi_pipeline.joblib \
  -e LOG_LEVEL=info \
  taxi-fare-api:1.0.0
```

```python
# taxi_api.py
import os
MODEL_PATH = os.getenv('MODEL_PATH', 'models/taxi_pipeline.joblib')
```

---

## Docker Compose (Local Dev Stack)

```yaml
# docker-compose.yml
services:
  taxi-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro   # hot-swap model without rebuild
    environment:
      - MODEL_PATH=/app/models/taxi_pipeline.joblib
    restart: unless-stopped
```

```bash
docker compose up --build
```

Read-only volume mount lets you test a new `.joblib` without rebuilding - useful during [MLOps](../../GLOSSARY.md#mlops) iteration.

---

## Kubernetes Preview (Course 2)

Docker solves *"run this image anywhere."* **Kubernetes** scales replicas, rolls out updates, and heals failures - same image from `docker build`.

---

## Security Hardening

| Practice | Why |
|----------|-----|
| Non-root `USER` | Limits container escape impact |
| Pin base image digest | Reproducible, patchable builds |
| `HEALTHCHECK` | Orchestrator restarts unhealthy pods |
| No `debug=True` | Prevents code execution exploits |
| Scan images (`docker scout`, Trivy) | Catch CVEs in dependencies |

---

## CI/CD Sketch

Tag images with **git commit SHA**. Push to a container registry; deploy with your orchestrator or `docker run` on a VM.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `ModuleNotFoundError: sklearn` | Missing from `requirements.txt` |
| Pickle load error | sklearn version mismatch vs training |
| Connection refused | Wrong port mapping (`-p 8000:8000`) |
| Permission denied on model | File owned by root; use `USER app` + correct COPY |
| Slow cold start | Large model; consider ONNX or model server |

```bash
docker logs taxi-api
docker exec -it taxi-api bash
```

---

## Cloud Build Without Local Docker

Prosise notes: upload `Dockerfile` to Azure (or AWS CodeBuild, Google Cloud Build). Cloud builds the image and stores it in a **container registry** - no local Docker required. Deploy container instances from the registry URL.

---

## Check Your Understanding

1. What is the difference between a Docker image and a container?
2. Why pin `scikit-learn==1.4.2` in both training and the Dockerfile?
3. What should `HEALTHCHECK` verify for an ML API?
4. Why run the container as a non-root user?
5. What files belong in the inference image vs the training environment?

---

## Next Steps

- **Section 7.5:** Export the same taxi model to ONNX for clients without HTTP  
- **Lab 07:** Build and test the taxi Docker deployment

**Previous:** [Section 7.3](./section-03-consuming-models-from-c-sharp.md)  
**Next:** [Section 7.5 - ONNX Export & Inference](./section-05-onnx-export-and-inference.md)