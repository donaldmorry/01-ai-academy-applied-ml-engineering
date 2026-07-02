# Section 7.2: Model Serialization with Pickle & Joblib

> **Source:** Prosise, Ch. 7 - "Consuming a Python Model from a Python Client"  
> **Prerequisites:** [Section 7.1](./section-01-from-notebook-to-production.md) | [Section 4.8](../chapter-04-text-classification/section-08-pipelines-and-production-text-ml.md)  
> **Glossary:** [model](../../GLOSSARY.md#model) | [MLOps](../../GLOSSARY.md#mlops)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Why Serialize?

Training is expensive. Inference should be cheap. **Serialization** freezes a trained object - preprocessor + estimator - to disk so any process can **deserialize** it later without refitting.

Prosise's Titanic example: train `LogisticRegression` once, `pickle.dump` to `titanic.pkl`, load anywhere with `pickle.load`, call `predict_proba`.

$$
P(\text{survived} \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)
$$
> **Readable form:** survival probability equals the sigmoid of the linear score - but only if $\mathbf{x}$ is encoded the same way as during training (one-hot sex, class dummies, etc.).

---

## Pickle Basics

Python's built-in `pickle` chapter converts objects to byte streams and back.

### Titanic classifier (Prosise)

```python
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data/titanic.csv')
df = df[['Survived', 'Age', 'Sex', 'Pclass']]
df = pd.get_dummies(df, columns=['Sex', 'Pclass'])
df.dropna(inplace=True)

X = df.drop('Survived', axis=1)
y = df['Survived']

model = LogisticRegression(random_state=0)
model.fit(X, y)

with open('models/titanic.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### Loading and predicting

```python
import pickle
import pandas as pd

with open('models/titanic.pkl', 'rb') as f:
    model = pickle.load(f)

# Feature columns MUST match training exactly
female = pd.DataFrame({
    'Age': [30],
    'Sex_female': [1], 'Sex_male': [0],
    'Pclass_1': [1], 'Pclass_2': [0], 'Pclass_3': [0],
})

prob_survive = model.predict_proba(female)[0][1]
print(f'Probability of survival: {prob_survive:.1%}')
```

> **In plain English:** Pickle is a snapshot of Python objects in memory. Load it, and the model wakes up exactly as you left it - coefficients, classes, everything.

---

## Serializing Pipelines (Required for Production)

From [Section 4.8](../chapter-04-text-classification/section-08-pipelines-and-production-text-ml.md), the sentiment pipeline combines `CountVectorizer` + `LogisticRegression`:

```python
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

df = pd.read_csv('data/reviews.csv', encoding='ISO-8859-1')
df = df.drop_duplicates()

X = df['Text']
y = df['Sentiment']

vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    stop_words='english',
    min_df=20,
)
model = LogisticRegression(max_iter=1000, random_state=0)
pipe = make_pipeline(vectorizer, model)
pipe.fit(X, y)

with open('models/sentiment.pkl', 'wb') as f:
    pickle.dump(pipe, f)
```

Client code - raw string in, score out:

```python
import pickle

with open('models/sentiment.pkl', 'rb') as f:
    pipe = pickle.load(f)

score = pipe.predict_proba(['Great food and excellent service!'])[0][1]
print(f'Sentiment score: {score:.3f}')
```

Works with `StandardScaler`, `ColumnTransformer`, and any sklearn transformer + estimator chain.

---

## Standalone CLI Client (Prosise)

```python
#!/usr/bin/env python3
"""sentiment.py - score text from command line."""
import pickle
import sys

with open('models/sentiment.pkl', 'rb') as f:
    pipe = pickle.load(f)

text = sys.argv[1] if len(sys.argv) > 1 else input('Text to analyze: ')
score = pipe.predict_proba([text])[0][1]
print(score)
```

```bash
python sentiment.py "Great food and excellent service!"
# 0.94
```

Copy `sentiment.pkl` beside the script. No notebook, no retraining.

---

## joblib: sklearn's Recommended Serializer

For models with large NumPy arrays or sparse matrices, **joblib** is faster and more compact:

```python
import joblib
from pathlib import Path

MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

# Save
joblib.dump(pipe, MODEL_DIR / 'sentiment_pipeline.joblib')

# Load
loaded = joblib.load(MODEL_DIR / 'sentiment_pipeline.joblib')
print(loaded.predict_proba(['Terrible service and cold food'])[0])
```

| Serializer | Best for | sklearn recommendation |
|------------|----------|------------------------|
| `pickle` | Small objects, stdlib only | Acceptable |
| `joblib` | Large numpy/scipy data | **Preferred** |

### Taxi fare pipeline (Chapter 02 → Chapter 07)

```python
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Assume X_train, y_train from Lab 02
taxi_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0)),
])
taxi_pipe.fit(X_train, y_train)

joblib.dump(taxi_pipe, 'models/taxi_pipeline.joblib')

# Verify round-trip
loaded = joblib.load('models/taxi_pipeline.joblib')
sample = X_test.iloc[[0]]
assert abs(
    loaded.predict(sample)[0] - taxi_pipe.predict(sample)[0]
) < 1e-6
print('Round-trip OK')
```

---

## File Size: CountVectorizer vs HashingVectorizer

Prosise notes: `sentiment.pkl` with `CountVectorizer(min_df=20)` ≈ **50 MB** - the entire vocabulary is embedded. Remove `min_df` → ~90 MB.

**HashingVectorizer** uses feature hashing - no stored vocabulary:

```python
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(
    HashingVectorizer(n_features=2**18, ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000),
)
pipe.fit(X, y)
joblib.dump(pipe, 'models/sentiment_hashed.joblib')  # ~8 MB
```

| Vectorizer | Vocabulary stored | `min_df` support | Typical size |
|------------|-------------------|------------------|--------------|
| CountVectorizer | Yes | Yes | Large |
| TfidfVectorizer | Yes | Yes | Large |
| HashingVectorizer | No | No | Small |

Trade-off: hashed models may score slightly differently; no `inverse_transform` for interpretability.

---

## Security: Never Load Untrusted Pickles

Pickle is **not secure**. Malicious pickle files can execute arbitrary code on `load()`.

```python
# DANGER - equivalent to running untrusted code
model = pickle.load(open('from_the_internet.pkl', 'rb'))  # Don't do this
```

**Rules:**
- Treat `.pkl` / `.joblib` like source code - only load from trusted builds
- Sign artifacts in CI; verify signatures before deploy
- For user-uploaded models, use sandboxed conversion or ONNX from a trusted pipeline
- Prefer ONNX or REST for cross-team handoffs when provenance is unclear

---

## Versioning Pickle Files

Prosise warns: **a model pickled with sklearn 1.3 may not unpickle cleanly on 1.5.** Sometimes warnings; sometimes hard failure.

```python
import sklearn
import json
from datetime import datetime, timezone

metadata = {
    'model_name': 'sentiment_v2',
    'sklearn_version': sklearn.__version__,
    'python_version': '3.11',
    'trained_at': datetime.now(timezone.utc).isoformat(),
    'metrics': {'auc': 0.91},
    'feature_schema': {'input': 'raw_text', 'output': 'probability_positive'},
}

joblib.dump(pipe, 'models/sentiment_v2.joblib')
with open('models/sentiment_v2.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

Pin versions in deployment:

```text
# requirements.txt
scikit-learn==1.4.2
joblib==1.3.2
pandas==2.2.0
```

When upgrading sklearn org-wide, plan a **re-export or retrain** pass for all stored artifacts. This is core [MLOps](../../GLOSSARY.md#mlops) - model registry tracks which artifact matches which runtime.

---

## What Gets Serialized?

| Component | Serialized? | Notes |
|-----------|-------------|-------|
| Fitted `Pipeline` | Yes | Full preprocess + predict |
| `GridSearchCV.best_estimator_` | Yes | Best pipeline inside |
| Training DataFrame | **No** | Store separately if needed |
| Custom Python in Pipeline | Yes, if importable | Define in `.py` chapter, not notebook cell |

Custom transformers must live in importable chapters - otherwise `joblib.load` fails in production.

---

## Comparison: Pickle vs ONNX (Preview)

| Aspect | pickle / joblib | ONNX |
|--------|-----------------|------|
| Languages | Python only | C#, Java, C++, JS, … |
| Security | Unsafe if untrusted | Data format, no arbitrary code |
| sklearn version lock | Strict | Converter + opset version |
| Pipeline support | Full sklearn | Subset of estimators |

[Section 7.5](./section-05-onnx-export-and-inference.md) covers ONNX export. Many teams use **both**: joblib for Python services, ONNX for edge and .NET clients.

---

## Check Your Understanding

1. Why does `CountVectorizer` inflate pickle file size?
2. What is the security risk of `pickle.load` on an untrusted file?
3. Why does sklearn recommend joblib over pickle for large models?
4. What metadata should you store alongside every `.joblib` artifact?
5. Why must custom pipeline steps be defined in importable `.py` files?

**Previous:** [Section 7.1](./section-01-from-notebook-to-production.md)  
**Next:** [Section 7.3 - Consuming Models from C#](./section-03-consuming-models-from-c-sharp.md)



