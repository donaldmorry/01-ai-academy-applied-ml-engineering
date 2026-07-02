# Section 5.8: SVM Production Notes - When to Deploy & ONNX Bridge

> **Source:** Prosise, Ch. 5 - engineering tradeoffs; deployment patterns from Part I  
> **Prerequisites:** [Sections 5.1-5.7](./section-01-svm-intuition-maximum-margin-and-support-vectors.md) | [Section 4.8](../chapter-04-text-classification/section-08-pipelines-and-production-text-ml.md)  
> **Glossary:** [svm](../../GLOSSARY.md#svm) | [model](../../GLOSSARY.md#model) | [hyperparameter](../../GLOSSARY.md#hyperparameter)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Honest Engineering Question

You've learned maximum margins, [kernels](../../GLOSSARY.md#kernel-trick), grid search, pipelines, and face recognition. Before shipping an SVM to production, ask:

> *Will an SVM beat simpler/faster alternatives on **my** data at **my** scale with **my** latency budget?*

Prosise teaches SVMs because they **illuminate kernel methods** and excel on specific problem shapes - not because they belong in every microservice.

> **Humorous analogy:** An SVM is a precision Swiss watch - beautiful on a wedding day, impractical when you need to time 10,000 potato sack races.

> **In plain English:** Deploy SVMs when data is small-to-medium, high-dimensional, and needs strong nonlinear boundaries. Skip them when you have millions of rows or need sub-millisecond inference at scale.

---

## When SVMs Win

| Scenario | Why SVM works | Prosise example |
|----------|---------------|-----------------|
| **High-dim, low-$n$** | RBF kernel, few support vectors | Olivetti faces ([Section 5.7](./section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md)) |
| **Text (sparse linear)** | `LinearSVC` on TF-IDF | [Chapter 04](../chapter-04-text-classification/README.md) |
| **Clean tabular, < 50k rows** | Tuned RBF beats untuned baselines | Ch. 5 benchmarks |
| **Need smooth boundary** | Kernel SVM vs tree stairsteps | SVR ([Section 5.6](./section-06-svr-regression-from-tube-to-production.md)) |
| **Binary with strong margin** | Interpretable support vectors | Fraud/spam after feature engineering |

**Complexity intuition:** Training is roughly $O(n^2 \cdot p)$ to $O(n^3)$ depending on solver - acceptable for $n = 5{,}000$, painful for $n = 5{,}000{,}000$.

---

## When SVMs Lose

| Scenario | Better alternative | Why |
|----------|-------------------|-----|
| **Millions of rows** | [Gradient boosting](../../GLOSSARY.md#gradient-boosting), linear models, SGD | Cubic training time |
| **Need `predict_proba` fast** | Logistic regression, calibrated trees | Platt scaling is expensive |
| **Feature importance required** | Random forest, XGBoost | SVM weights opaque in kernel space |
| **Streaming / online learning** | SGDClassifier, passive-aggressive | SVM retrains from scratch |
| **Images at scale** | CNNs ([Chapter 10](../chapter-10-convolutional-neural-networks/README.md)) | SVM+HOG is legacy baseline |
| **Default tabular regression** | GBM | [Section 2.5](../chapter-02-regression-models/section-05-support-vector-regression-svr.md) taxi fares |

---

## Decision Flowchart

```
                    Start
                      │
          n > 100,000 samples?
                 /         \
              Yes            No
               │              │
         GBM / linear    Need nonlinear?
         / SGD               /        \
                          Yes         No
                           │           │
                      p > 10,000?   LinearSVC or
                       /      \     logistic regression
                     Yes       No
                      │         │
                 LinearSVC   RBF SVC
                 or PCA+SVM  + GridSearchCV
```

---

## Production Checklist

### 1. Persist the full pipeline

```python
import joblib

# best_model from GridSearchCV - includes scaler + SVC
joblib.dump(best_model, 'models/face_svm_v1.joblib')

# Load in API
model = joblib.load('models/face_svm_v1.joblib')
prediction = model.predict(hog_features.reshape(1, -1))
```

Never save bare `SVC` without the fitted scaler ([Section 5.4](./section-04-normalization-and-pipelining-for-svms.md)).

### 2. Version hyperparameters in metadata

```python
import json

metadata = {
    'model_type': 'SVC',
    'kernel': 'rbf',
    'C': 1000,
    'gamma': 0.01,
    'feature_type': 'hog',
    'training_samples': 300,
    'cv_accuracy': 0.94,
    'sklearn_version': '1.4.0',
}
with open('models/face_svm_v1.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

### 3. Monitor support vector count

```python
svc = model.named_steps['svc']
sv_ratio = svc.support_vectors_.shape[0] / n_train
print(f"Support vector ratio: {sv_ratio:.2%}")
```

High ratio (>50%) → model may be overfitting or data is very noisy. Retune $C$.

### 4. Latency profiling

```python
import time

X_sample = X_test[:100]
start = time.perf_counter()
model.predict(X_sample)
elapsed = time.perf_counter() - start
print(f"100 predictions: {elapsed*1000:.1f} ms ({elapsed*10:.2f} ms/sample)")
```

Prediction time scales with **number of support vectors**, not training set size.

---

## ONNX Bridge: Beyond sklearn

Many production environments (mobile, edge, .NET, C++) don't run Python. **ONNX** (Open Neural Network Exchange) exports models to a portable format.

**Reality check:** sklearn's `SVC` with RBF kernel has **limited** ONNX support compared to tree ensembles and neural nets. Options:

### Option A: skl2onnx (linear SVM)

```python
# pip install skl2onnx onnxruntime
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

linear_pipe = make_pipeline(StandardScaler(), LinearSVC(C=1.0))
linear_pipe.fit(X_train, y_train)

initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(linear_pipe, initial_types=initial_type)

with open('linear_svm.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
```

### Option B: ONNX Runtime inference

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('linear_svm.onnx')
input_name = session.get_inputs()[0].name
result = session.run(None, {input_name: X_test.astype(np.float32)})
predictions = result[0]
```

### Option C: RBF SVM workarounds

| Approach | Tradeoff |
|----------|----------|
| `LinearSVC` + feature expansion | Approximate nonlinear, ONNX-friendly |
| PCA + `LinearSVC` | Loses kernel flexibility |
| Train SVM in Python, export support vectors + weights manually | Custom ONNX graph |
| Switch to `SGDClassifier(loss='hinge')` | Linear only, fast, portable |
| Use [ONNX Runtime](https://onnxruntime.ai/) with custom kernel op | Engineering heavy |

**Prosise's pragmatic advice:** For edge deployment of nonlinear SVMs, many teams **retrain** with gradient boosted trees (ONNX via `onnxmltools`) or small neural nets rather than fighting RBF export.

---

## Alternatives at Inference Time

If ONNX export fails, consider:

```python
from sklearn.linear_model import SGDClassifier

# Hinge loss ≈ linear SVM, supports partial_fit for streaming
sgd = make_pipeline(
    StandardScaler(),
    SGDClassifier(loss='hinge', alpha=0.001, max_iter=1000, random_state=42)
)
sgd.fit(X_train, y_train)
```

| Method | Portable | Nonlinear | Online |
|--------|----------|-----------|--------|
| `SVC` RBF | Hard | Yes | No |
| `LinearSVC` | Medium | No | No |
| `SGDClassifier` | Easy | No | Yes |
| XGBoost + ONNX | Good | Yes | No |

---

## Memory Footprint

Stored [model](../../GLOSSARY.md#model) size ≈ support vectors × feature dimension:

$$
\text{memory} \approx n_{SV} \times p \times 8 \text{ bytes (float64)}
$$
> **Readable form:** memory ≈ number of support vectors times features times 8 bytes

Example: 150 support vectors × 1,764 HOG features × 8 B ≈ **2.1 MB** - trivial.  
50,000 support vectors × 4,096 pixels × 8 B ≈ **1.6 GB** - problematic.

---

## A/B Testing SVM vs Baseline

Before replacing logistic regression in production:

| Metric | Measure |
|--------|---------|
| Offline | CV [accuracy](../../GLOSSARY.md#accuracy), F1, calibration |
| Latency | p50/p99 prediction time |
| Memory | Model file size, RSS at load |
| Ops | Retrain time, dependency footprint |

Ship SVM only if offline gain exceeds **2× latency cost** (rule of thumb - adjust per domain).

---

## Security & Fairness Notes

- **Adversarial faces:** HOG + SVM is spoofable with printed photos - not liveness detection
- **Bias across demographics:** Olivetti is 40 subjects, not representative - never deploy fairness claims from this dataset alone
- **Model extraction:** Support vectors leak training exemplars - consider privacy in sensitive domains

---

## Chapter 05 Retrospective

| Section | You learned |
|--------|-------------|
| [5.1](./section-01-svm-intuition-maximum-margin-and-support-vectors.md) | Margin, support vectors |
| [5.2](./section-02-kernels-and-the-kernel-trick.md) | Linear, RBF, poly kernels |
| [5.3](./section-03-hyperparameter-tuning-c-gamma-and-gridsearchcv.md) | $C$, $\gamma$, GridSearchCV |
| [5.4](./section-04-normalization-and-pipelining-for-svms.md) | StandardScaler, pipelines |
| [5.5](./section-05-svm-classification-svc-for-binary-and-multiclass.md) | SVC multiclass |
| [5.6](./section-06-svr-regression-from-tube-to-production.md) | SVR link to Chapter 02 |
| [5.7](./section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md) | HOG + Olivetti |
| **5.8 (this)** | Production tradeoffs, ONNX |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| SVM on 2M rows "because Ch. 5" | Week-long training | Subsample or switch algorithm |
| Export RBF SVC to ONNX blindly | Conversion failure | LinearSVC or trees |
| Ignoring inference latency | SLA violations | Profile SV count |
| No model metadata | Undebuggable prod | JSON sidecar with params |
| Skipping baseline comparison | Over-engineered SVM | Logistic regression first |

---

## Self-Check

1. What is SVM training complexity w.r.t. $n$?  
   *Roughly $O(n^2)$ to $O(n^3)$ - prohibitive for very large $n$.*
2. Why does prediction depend on support vectors, not $n$?  
   *Only support vectors have nonzero dual coefficients in the decision function.*
3. When is ONNX export straightforward for SVMs?  
   *Linear kernels via skl2onnx; RBF is difficult.*
4. Name one Prosise scenario where GBM beats SVR.  
   *Taxi fare tabular regression.*

---

## Exercises

1. Profile train vs predict time for RBF SVC on $n \in \{500, 2000, 8000\}$ samples (same $p$).
2. Export a `LinearSVC` pipeline to ONNX; run inference with `onnxruntime` and verify accuracy matches sklearn.
3. Write a one-page "build vs skip" memo for SVM on your current work dataset using the decision flowchart.

---

## References

- [skl2onnx Documentation](https://onnx.ai/sklearn-onnx/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Scikit-Learn Model Persistence](https://scikit-learn.org/stable/model_persistence.html)
- [Section 4.8](../chapter-04-text-classification/section-08-pipelines-and-production-text-ml.md) - production pipelines
- Prosise, Ch. 5 - when to choose SVM
- [GLOSSARY.md](../../GLOSSARY.md) - [svm](../../GLOSSARY.md#svm)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 5.7](./section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md) | **Lab:** [Lab 05 - Facial Recognition](./section-lab-05-facial-recognition-with-support-vector-machines.md) | **Next Chapter:** [Chapter 06 - PCA](../chapter-06-principal-component-analysis/README.md)



