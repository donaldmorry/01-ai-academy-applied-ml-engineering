# Section 7.7: Excel Integration

> **Source:** Prosise, Ch. 7 - "Adding Machine Learning Capabilities to Excel"  
> **Prerequisites:** [Section 7.2](./section-02-model-serialization-with-pickle-and-joblib.md) | [Section 7.3](./section-03-consuming-models-from-c-sharp.md)  
> **Glossary:** [model](../../GLOSSARY.md#model) | [MLOps](../../GLOSSARY.md#mlops)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## ML for Business Users

Data scientists live in notebooks. Business analysts live in **Excel**. Prosise's scenario: a vacation-rentals communications team needs to score social media text for sentiment - without learning Python.

**Goal:** Type text in cell A1, see a sentiment score in B1:

```excel
=analyze_text(A1)
```

Three integration paths:

| Approach | Python on analyst PC? | Best for |
|----------|---------------------|----------|
| **xlwings UDF** | Yes (background) | Small teams, Windows Excel |
| **REST API + Power Query** | No | Centralized model, many users |
| **REST + VBA HttpClient** | No | Custom macros, legacy workbooks |

This section covers **xlwings** (Prosise's path) and **REST-backed** alternatives.

---

## Path A: xlwings Python UDFs (Prosise)

**xlwings** bridges Excel and Python. Write Python functions decorated with `@xw.func`; call them like built-in Excel functions.

### Step 1: Trust VBA add-ins

Excel → **File → Options → Trust Center → Trust Center Settings → Macro Settings** → check **"Trust access to the VBA project object model"**.

Required for xlwings add-in to communicate with Python.

### Step 2: Install xlwings

```bash
pip install xlwings
xlwings addin install
```

Installs the xlwings ribbon tab in Excel (Windows).

### Step 3: Quickstart project

```bash
mkdir sentiment-excel && cd sentiment-excel
xlwings quickstart sentiment
```

Creates:

```text
sentiment/
├── sentiment.xlsm    # Excel macro workbook
└── sentiment.py      # Python UDF source
```

Copy `models/sentiment.pkl` (from [Section 7.2](./section-02-model-serialization-with-pickle-and-joblib.md)) into `sentiment/`.

### Step 4: UDF code (Prosise)

```python
"""sentiment/sentiment.py - Excel UDF for sentiment scoring."""
import os
import pickle
import xlwings as xw

model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'sentiment.pkl')
)

with open(model_path, 'rb') as f:
    pipe = pickle.load(f)


@xw.func
def analyze_text(text):
    """Return P(positive) from 0.0 to 1.0."""
    if text is None or str(text).strip() == '':
        return ''
    score = pipe.predict_proba([str(text)])[0][1]
    return float(score)
```

### Step 5: Import and use

1. Open `sentiment.xlsm`
2. **xlwings** tab → **Import Functions**
3. Cell A1: `Great food and excellent service`
4. Cell B1: `=analyze_text(A1)`
5. Confirm B1 shows a value between 0.0 and 1.0

> **In plain English:** Excel calls Python behind the scenes. The full sklearn pipeline - vectorizer + classifier - runs on each formula evaluation.

---

## Taxi Fare UDF (Chapter 02 Extension)

```python
import os
import joblib
import xlwings as xw
import pandas as pd

model_path = os.path.join(os.path.dirname(__file__), 'taxi_pipeline.joblib')
pipe = joblib.load(model_path)


@xw.func
@xw.arg('trip_distance', numbers=float)
@xw.arg('passenger_count', numbers=int)
def predict_fare(trip_distance, passenger_count, hour, day_of_week, is_weekend, is_rush):
    """Predict taxi fare in USD."""
    row = pd.DataFrame([{
        'trip_distance': trip_distance,
        'passenger_count': passenger_count,
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'is_rush': is_rush,
    }])
    return round(float(pipe.predict(row)[0]), 2)
```

Excel:

```excel
=predict_fare(3.2, 2, 14, 4, 0, 1)
```

---

## xlwings Limitations

| Constraint | Implication |
|------------|-------------|
| Windows-focused for UDFs | Mac Excel support limited |
| Python must be installed | IT deployment overhead |
| Recalc calls Python each time | Slow for thousands of rows |
| Pickle on analyst machine | Version sync with training |
| Security policies | Macros/add-ins often blocked |

For enterprise rollouts, prefer centralized REST ([Section 7.3](./section-03-consuming-models-from-c-sharp.md)).

---

## Path B: REST API + Excel Power Query

No Python on analyst PCs. Model runs in Docker ([Section 7.4](./section-04-docker-for-ml-services.md)).

### Setup

1. Deploy taxi API at `https://ml-api.company.com`
2. Excel → **Data → Get Data → From Other Sources → From Web**
3. Advanced: POST URL with JSON body (Power Query M)

### M code sketch (batch scoring)

```powerquery
let
    Source = Excel.CurrentWorkbook(){[Name="Trips"]}[Content],
    AddPrediction = Table.AddColumn(Source, "predicted_fare", each
        let
            body = Text.ToBinary(
                "{""trip_distance"":" & Text.From([trip_distance])
                & ",""passenger_count"":" & Text.From([passenger_count])
                & ",""hour"":" & Text.From([hour])
                & ",""day_of_week"":" & Text.From([day_of_week])
                & ",""is_weekend"":" & Text.From([is_weekend])
                & ",""is_rush"":" & Text.From([is_rush]) & "}"
            ),
            response = Web.Contents(
                "https://ml-api.company.com/v1/predict",
                [Content = body, Headers = [#"Content-Type"="application/json"]]
            ),
            json = Json.Document(response)
        in
            json[predicted_fare_usd]
    )
in
    AddPrediction
```

Refresh pulls latest model from server - no pickle distribution.

---

## Path C: VBA + REST (Sentiment)

For analysts comfortable with macros, VBA can call the Flask service:

```vba
Function AnalyzeSentiment(text As String) As Double
    Dim http As Object
    Dim url As String
    Set http = CreateObject("MSXML2.XMLHTTP")
    url = "http://localhost:5000/analyze?text=" & EncodeURIComponent(text)
    http.Open "GET", url, False
    http.send
    AnalyzeSentiment = CDbl(http.responseText)
End Function
```

Use `EncodeURIComponent` or similar - special characters break URLs.

---

## Comparison Matrix

| Factor | xlwings UDF | Power Query REST | VBA REST |
|--------|-------------|------------------|----------|
| Setup complexity | Medium | Medium | Low-Medium |
| Offline use | Yes (if Python local) | No | No |
| Model updates | Redistribute pickle | Server-side only | Server-side only |
| Row performance | Slow at scale | Better for batches | Per-cell HTTP (slow) |
| IT approval | Python + macros | HTTPS only | Macros + network |

---

## Production Hygiene for Excel ML

1. **Document the function** - what score means, valid input range
2. **Freeze model version** - cell comment: `model_version=taxi_v3`
3. **Avoid full-column live UDF** on 100k rows - precompute via API batch job
4. **Handle errors** - return `#N/A` or message, not Python traceback
5. **Audit trail** - log API calls server-side for compliance

```python
@xw.func
def analyze_text(text):
    try:
        if not text:
            return ''
        return float(pipe.predict_proba([str(text)])[0][1])
    except Exception:
        return '#ERROR!'
```

---

## Security Notes

- xlwings executes arbitrary Python - same trust model as pickle
- REST path: use HTTPS, API keys, rate limits
- Do not embed API secrets in workbook cells - use Excel's credential store or enterprise gateway

---

## Prosise's Key Insight

> *"UDFs written in Python present Excel users with a new whole world of possibilities… and provide a valuable opportunity for Excel users to operationalize machine learning models written in Python."*

The pattern generalizes: meet users where they work. Not every stakeholder needs a Jupyter login.

---

## Check Your Understanding

1. What does `@xw.func` do?
2. Why is xlwings slow for scoring 50,000 rows in column B?
3. How does a REST-backed Power Query workbook avoid distributing pickle files?
4. What Excel trust setting is required for xlwings?
5. When would you choose REST over xlwings for a finance team?

---

## Next Steps

- **Section 7.8:** Production checklist - validation, monitoring, rollback  
- **Lab 07 (optional):** Excel workbook calling the taxi API

**Previous:** [Section 7.6](./section-06-ml-net-overview.md)  
**Next:** [Section 7.8 - Production Checklist](./section-08-production-checklist.md)