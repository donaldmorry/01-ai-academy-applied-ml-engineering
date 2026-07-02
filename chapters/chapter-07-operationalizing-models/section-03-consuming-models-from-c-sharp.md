# Section 7.3: Consuming Models from C#

> **Source:** Prosise, Ch. 7 - "Consuming a Python Model from a C# Client"  
> **Prerequisites:** [Section 7.2](./section-02-model-serialization-with-pickle-and-joblib.md)  
> **Glossary:** [model](../../GLOSSARY.md#model) | [ONNX](../../GLOSSARY.md#onnx)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Language Gap

Python dominates ML training - Pandas, scikit-learn, Jupyter. Enterprise apps often run in **C#** - WinForms, WPF, ASP.NET Core, Xamarin. You cannot call `model.predict()` from C# as if it were a native method.

Prosise offers three bridges:

| Approach | How it works | Python on client? |
|----------|--------------|-------------------|
| **REST API** | Flask/FastAPI wraps model; C# sends HTTP | No |
| **ONNX Runtime** | Export `.onnx`; C# loads natively | No |
| **Embedded Python** | Python.NET hosts interpreter | Yes |

This section focuses on **REST** - the most universal pattern. [Section 7.5](./section-05-onnx-export-and-inference.md) covers ONNX for in-process C# inference.

---

## Flask REST Service (Prosise Sentiment Example)

### `app.py` - minimal sentiment API

```python
import pickle
from flask import Flask, request

app = Flask(__name__)

with open('models/sentiment.pkl', 'rb') as f:
    pipe = pickle.load(f)


@app.route('/analyze', methods=['GET'])
def analyze():
    text = request.args.get('text')
    if not text:
        return 'No string to analyze', 400

    score = pipe.predict_proba([text])[0][1]
    return str(score)


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
```

Start the service:

```bash
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000
```

Test with curl:

```bash
curl -G -w "\n" "http://localhost:5000/analyze" \
  --data-urlencode "text=Great food and excellent service!"
# 0.94...
```

> **In plain English:** Flask loads the pickle once at startup. Every HTTP request reuses the same in-memory model - fast after the first load.

---

## Production-Grade FastAPI (Taxi Fare)

GET with query strings breaks on special characters and complex payloads. Use **POST + JSON** for real APIs:

```python
"""taxi_api.py - fare prediction REST service."""
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title='Taxi Fare API', version='1.0.0')

MODEL_PATH = Path('models/taxi_pipeline.joblib')
pipe = joblib.load(MODEL_PATH)


class FareRequest(BaseModel):
    trip_distance: float = Field(..., gt=0, le=100, description='Miles')
    passenger_count: int = Field(..., ge=1, le=6)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)
    is_rush: int = Field(..., ge=0, le=1)


class FareResponse(BaseModel):
    predicted_fare_usd: float
    model_version: str = 'taxi_v1'


@app.post('/v1/predict', response_model=FareResponse)
def predict_fare(req: FareRequest):
  try:
      import pandas as pd
      row = pd.DataFrame([req.model_dump()])
      fare = float(pipe.predict(row)[0])
      if fare < 0:
          raise HTTPException(422, 'Model returned negative fare')
      return FareResponse(predicted_fare_usd=round(fare, 2))
  except Exception as e:
      raise HTTPException(500, str(e)) from e


@app.get('/health')
def health():
    return {'status': 'ok', 'model': str(MODEL_PATH)}
```

Run:

```bash
pip install fastapi uvicorn joblib pandas scikit-learn
uvicorn taxi_api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"trip_distance":3.2,"passenger_count":2,"hour":14,
       "day_of_week":4,"is_weekend":0,"is_rush":1}'
```

---

## C# Client - HttpClient (Prosise)

```csharp
using System;
using System.Net.Http;
using System.Threading.Tasks;
using System.Web;  // HttpUtility for .NET Framework; use Uri.EscapeDataString in .NET Core

class Program
{
    static async Task Main(string[] args)
    {
        string text = args.Length > 0
            ? args[0]
            : Console.ReadLine() ?? "";

        var client = new HttpClient();
        var encoded = Uri.EscapeDataString(text);
        var url = $"http://localhost:5000/analyze?text={encoded}";

        var response = await client.GetAsync(url);
        response.EnsureSuccessStatusCode();
        var score = await response.Content.ReadAsStringAsync();

        Console.WriteLine($"Sentiment score: {score}");
    }
}
```

### C# client - JSON POST for taxi API

```csharp
using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;

record FareRequest(
    double trip_distance,
    int passenger_count,
    int hour,
    int day_of_week,
    int is_weekend,
    int is_rush);

record FareResponse(double predicted_fare_usd, string model_version);

class Program
{
    static async Task Main()
    {
        var client = new HttpClient { BaseAddress = new Uri("http://localhost:8000") };

        var request = new FareRequest(3.2, 2, 14, 4, 0, 1);
        var response = await client.PostAsJsonAsync("/v1/predict", request);
        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<FareResponse>();
        Console.WriteLine($"Predicted fare: ${result!.predicted_fare_usd:F2}");
    }
}
```

Any language with HTTP works - Java `HttpClient`, Python `requests`, JavaScript `fetch`, PowerShell `Invoke-RestMethod`.

---

## API Design Checklist

| Concern | Bad | Good |
|---------|-----|------|
| Input | Query string only | JSON body with schema validation |
| Errors | Plain text "error" | HTTP status + structured JSON |
| Versioning | `/predict` | `/v1/predict` |
| Health | None | `GET /health` for load balancers |
| Docs | None | OpenAPI at `/docs` (FastAPI auto-generates) |

### Error response shape

```json
{
  "detail": "trip_distance must be greater than 0"
}
```

---

## Latency: REST vs In-Process

Prosise benchmarked sentiment analysis:

| Method | Latency (local) |
|--------|-----------------|
| Flask REST round-trip | ~2+ seconds |
| Python ONNX Runtime | ~0.001 seconds |

REST pays for HTTP serialization, network stack, and Python WSGI overhead. For **high-throughput, low-latency** C# apps scoring millions of rows, prefer [ONNX](./section-05-onnx-export-and-inference.md). For **multi-language clients, centralized updates, and auth**, prefer REST.

Remote hosting adds WAN latency on top.

---

## Security Considerations

| Risk | Mitigation |
|------|------------|
| Unauthenticated API | API keys, OAuth2, mTLS |
| Injection via text input | Validate length; rate limit |
| `debug=True` in Flask | Never in production |
| Pickle loaded from shared volume | Restrict file permissions |
| HTTPS | Terminate TLS at gateway |

```python
# Production: debug=False, use gunicorn/uvicorn workers
# app.run(debug=True)  # REMOVE
```

---

## Architecture Diagram

```
┌─────────────┐     HTTP/JSON      ┌──────────────────────────┐
│  C# Client  │ ─────────────────▶ │  FastAPI / Flask         │
│  (ASP.NET)  │ ◀───────────────── │  + joblib pipeline       │
└─────────────┘   predicted_fare   │  (Python process)        │
                                   └──────────────────────────┘
```

Containerize this stack in [Section 7.4](./section-04-docker-for-ml-services.md) so the C# client needs no Python installation - only a URL.

---

## Alternative: gRPC (Brief)

For internal microservices, **gRPC** offers binary payloads and HTTP/2 multiplexing - lower overhead than REST JSON. Same model loading pattern; different wire protocol. REST remains the default for public and polyglot clients.

---

## When REST Is the Right Choice

- Multiple client platforms (web, mobile, C#, Excel)
- Model updates without redeploying clients (point to new API version)
- Centralized logging, auth, rate limiting
- Team already operates API gateways and Kubernetes ingress

When REST is **not** ideal:

- Sub-10 ms latency requirements on edge devices
- Offline/air-gapped scoring
- Batch scoring millions of rows inside a single C# process

→ Use ONNX ([Section 7.5](./section-05-onnx-export-and-inference.md)) or ML.NET ([Section 7.6](./section-06-ml-net-overview.md)).

---

## Check Your Understanding

1. Why does the Flask app load `sentiment.pkl` at startup instead of per request?
2. Why is POST + JSON preferable to GET + query string for taxi features?
3. What HTTP status code should invalid input return?
4. When would you choose ONNX over REST for a C# desktop app?
5. What does `GET /health` enable in production?

---

## Next Steps

- **Section 7.4:** Package the API in Docker  
- **Section 7.5:** Eliminate the HTTP hop with ONNX Runtime in C#

**Previous:** [Section 7.2](./section-02-model-serialization-with-pickle-and-joblib.md)  
**Next:** [Section 7.4 - Docker for ML Services](./section-04-docker-for-ml-services.md)