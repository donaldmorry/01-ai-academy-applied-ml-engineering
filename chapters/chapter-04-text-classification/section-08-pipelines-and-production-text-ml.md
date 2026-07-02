# Section 4.8: Pipelines & Production Text ML

> **Source:** Prosise, Ch. 4 - end-to-end pipelines, model persistence  
> **Prerequisites:** [Sections 4.1-4.7](./section-01-text-as-data.md) | [Section 3.4](../chapter-03-classification-models/section-04-categorical-data-and-feature-encoding.md)  
> **Glossary:** [tf-idf](../../GLOSSARY.md#tf-idf) | [naive-bayes](../../GLOSSARY.md#naive-bayes)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Leakage Trap in Text ML

The most common text ML bug: call `TfidfVectorizer.fit_transform()` on **all** data before `train_test_split`. The vectorizer learns vocabulary from test reviews - future information leaks into training. Metrics look great; production fails.

The fix is the same as [Section 3.4](../chapter-03-classification-models/section-04-categorical-data-and-feature-encoding.md): wrap vectorizer + classifier in a sklearn **`Pipeline`**, `fit` on train only, persist the **entire** pipeline with **`joblib`**.

> **Humorous analogy:** Fitting on all data before splitting is like giving students the final exam questions during practice - stellar grades, zero learning, angry employer.

> **In plain English:** A pipeline chains preprocessing and model into one object. Save once, load in production, call `predict()` on raw strings.

---

## Pipeline Anatomy for Text

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        min_df=5,
        ngram_range=(1, 2),
        max_features=20000,
    )),
    ('clf', LogisticRegression(max_iter=2000, random_state=42)),
])
```

**Execution on `fit(X_train, y_train)`:**
1. `TfidfVectorizer.fit_transform(X_train)` - learn vocabulary + IDF
2. `LogisticRegression.fit(X_tfidf, y_train)` - learn weights

**Execution on `predict(X_test)`:**
1. `TfidfVectorizer.transform(X_test)` - apply learned vocabulary
2. `LogisticRegression.predict(...)` - return labels

One API. No manual steps. No leakage.

---

## End-to-End Training Example

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

np.random.seed(42)
reviews = (["great film"] * 500 + ["terrible movie"] * 500)
labels = [1] * 500 + [0] * 500

X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.2, random_state=42, stratify=labels
)

# Sentiment pipeline (TF-IDF + logistic)
sentiment_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=2, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, random_state=42)),
])

sentiment_pipe.fit(X_train, y_train)
print(classification_report(y_test, sentiment_pipe.predict(X_test)))

# Spam-style pipeline (counts + NB)
spam_pipe = Pipeline([
    ('counts', CountVectorizer(stop_words='english', min_df=2, ngram_range=(1, 2))),
    ('clf', MultinomialNB(alpha=1.0)),
])
```

---

## Accessing Pipeline Steps

```python
# Hyperparameter tuning syntax: stepname__parameter
sentiment_pipe.set_params(clf__C=0.1)

# Extract fitted components
vectorizer = sentiment_pipe.named_steps['tfidf']
classifier = sentiment_pipe.named_steps['clf']

print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
print(f"Classes: {classifier.classes_}")
```

Double underscore `__` routes parameters to nested estimators - essential for `GridSearchCV`.

---

## GridSearchCV on Text Pipelines

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'tfidf__min_df': [2, 5, 10],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1.0, 10.0],
}

grid = GridSearchCV(
    sentiment_pipe,
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV F1:", grid.best_score_)
print(classification_report(y_test, grid.predict(X_test)))
```

Cross-validation refits the vectorizer inside each fold on training folds only - correct nested validation per [Section 2.7](../chapter-02-regression-models/section-07-model-comparison-and-cross-validation.md).

---

## Persisting Models with joblib

```python
import joblib
from pathlib import Path

MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

# Save entire pipeline
joblib.dump(sentiment_pipe, MODEL_DIR / 'sentiment_pipeline.joblib')
joblib.dump(spam_pipe, MODEL_DIR / 'spam_pipeline.joblib')

print("Saved models to", MODEL_DIR)
```

**Why joblib over pickle?** Efficient for large numpy/scipy sparse arrays - sklearn's recommended serializer.

### Loading in production

```python
import joblib

loaded_pipe = joblib.load('models/sentiment_pipeline.joblib')

def score_review(text: str) -> dict:
    label = int(loaded_pipe.predict([text])[0])
    proba = loaded_pipe.predict_proba([text])[0]
    return {
        'sentiment': 'positive' if label == 1 else 'negative',
        'confidence': float(proba[label]),
    }

print(score_review("A wonderful evening at the cinema"))
```

Raw string in → sentiment out. Vectorizer travels with the model.

---

## Versioning and Metadata

Production teams bundle metadata alongside `.joblib`:

```python
import json
from datetime import datetime

metadata = {
    'model_name': 'sentiment_v1',
    'trained_at': datetime.utcnow().isoformat(),
    'sklearn_version': '1.4.0',
    'training_size': len(X_train),
    'metrics': {'accuracy': float(sentiment_pipe.score(X_test, y_test))},
}

joblib.dump(sentiment_pipe, 'models/sentiment_v1.joblib')
with open('models/sentiment_v1.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

Track sklearn version - breaking changes across major releases can invalidate pickles.

---

## Custom Preprocessing in Pipelines

From [Section 4.2](./section-02-text-preprocessing.md), plug a preprocessor as `analyzer`:

```python
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stem_analyzer(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [stemmer.stem(t) for t in text.split()
              if t not in stop_words and len(t) > 2]
    return tokens

custom_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer=stem_analyzer, min_df=3)),
    ('clf', MultinomialNB()),
])
```

Custom analyzers must be **importable** at load time - define in a `.py` chapter, not notebook cell, for production.

---

## Inference Latency Considerations

| Stage | Typical cost |
|-------|--------------|
| Tokenization + TF-IDF transform | 1-10 ms per doc |
| Linear model predict | < 1 ms |
| Large vocabulary / n-grams | Memory + CPU ↑ |

```python
import time

texts = ["sample review"] * 1000

start = time.perf_counter()
_ = sentiment_pipe.predict(texts)
elapsed = time.perf_counter() - start
print(f"1000 predictions in {elapsed:.3f}s ({elapsed/1000*1000:.2f} ms/doc)")
```

**Optimizations:**
- `max_features` cap
- `min_df` pruning
- Batch `predict` calls
- Pre-compile regex in analyzer

---

## REST API Sketch (Flask)

```python
# app.py - illustrative production wrapper
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('models/sentiment_pipeline.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'empty text'}), 400
    label = int(model.predict([text])[0])
    proba = model.predict_proba([text])[0]
    return jsonify({
        'sentiment': 'positive' if label == 1 else 'negative',
        'confidence': float(proba[label]),
    })

# Run: flask run --port 5000
```

Same pattern deploys spam filter and routes to [Chapter 07](../chapter-07-operationalizing-models/README.md) for containers and ONNX.

---

## Key Takeaways

1. **`Pipeline`** chains vectorizer + model - prevents vocabulary leakage
2. **`stepname__param`** syntax enables grid search over text hyperparameters
3. **`joblib.dump/load`** persists the full artifact for production
4. **Load once, predict many** - raw strings in, labels out
5. **Custom analyzers** must live in importable chapters for reload
6. **Text ML deploys easily** - small models, fast inference, no GPU

---

## Check Your Understanding

1. Why must `fit_transform` never run on the full dataset before splitting?
2. What does `clf__C=0.1` mean in a pipeline parameter dict?
3. Why save the pipeline instead of only the classifier?
4. What breaks if you train an analyzer in a notebook and load in production?
5. Which Chapter 4 apps need threshold tuning at inference time?

---

## References

- Prosise, Ch. 4 - production patterns; Ch. 7 for deployment depth
- Scikit-Learn Pipelines: [https://scikit-learn.org/stable/chapters/compose.html](https://scikit-learn.org/stable/chapters/compose.html)
- Model persistence: [https://scikit-learn.org/stable/model_persistence.html](https://scikit-learn.org/stable/model_persistence.html)
- joblib documentation: [https://joblib.readthedocs.io/](https://joblib.readthedocs.io/)

---

**Previous:** [Section 4.7 - Cosine Similarity & Recommenders](./section-07-cosine-similarity-and-recommender-systems.md)  
**Next:** [Lab 04 - Sentiment, Spam, and Recommendations](./section-lab-04-sentiment-spam-and-recommendations.md)