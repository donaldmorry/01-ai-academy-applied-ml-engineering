# Section 7.5: ONNX Export & Inference

> **Source:** Prosise, Ch. 7 - "Using ONNX to Bridge the Language Gap"  
> **Prerequisites:** [Section 7.2](./section-02-model-serialization-with-pickle-and-joblib.md) | [Section 7.4](./section-04-docker-for-ml-services.md)  
> **Glossary:** [ONNX](../../GLOSSARY.md#onnx) | [model](../../GLOSSARY.md#model)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## What Is ONNX?

**ONNX** (Open Neural Network Exchange) is a platform-agnostic format for representing machine learning models. Originally designed to port deep-learning graphs between TensorFlow and PyTorch, it now supports scikit-learn via **skl2onnx**.

$$
\text{Python sklearn pipeline} \xrightarrow{\text{skl2onnx}} \text{model.onnx} \xrightarrow{\text{ORT}} \text{C\#, Java, C++, JS, …}
$$
> **Readable form:** train in Python, export to a standard file, run anywhere with ONNX Runtime - no Python interpreter on the client.

Prosise: ONNX is a "game changer" for projecting Python models to other platforms. Runtimes exist for Python, C, C++, C#, Java, JavaScript, Objective-C on Windows, Linux, macOS, Android, and iOS.

---

## When to Choose ONNX vs REST vs Pickle

| Criterion | pickle/joblib | REST API | ONNX |
|-----------|---------------|----------|------|
| Languages | Python only | Any (HTTP) | Many native |
| Latency (local) | ~ms | ~seconds (Prosise Flask) | ~0.001 s |
| Network required | No | Yes | No |
| Centralized updates | Redeploy file | Redeploy server | Redeploy file |
| sklearn coverage | Full | Full (via Python) | Subset |
| Security | Unsafe if untrusted | API auth | Data format |

Use ONNX when you need **in-process, low-latency** inference in C#, mobile, or edge without hosting Python.

---

## Install Export Tools

```bash
pip install skl2onnx onnxruntime onnx
# For some pipelines:
pip install onnxmltools
```

---

## Export: Sentiment Pipeline (Text Input)

Prosise Example 7-1 pipeline - `CountVectorizer` + `LogisticRegression`:

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

# `pipe` - fitted make_pipeline(vectorizer, LogisticRegression)
initial_type = [('string_input', StringTensorType([None, 1]))]

onnx_model = convert_sklearn(
    pipe,
    initial_types=initial_type,
    target_opset=17,  # pin opset for reproducibility
)

with open('models/sentiment.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print('Exported models/sentiment.onnx')
```

`initial_types` declares the **input schema** ONNX expects - here one string column per row.

---

## Export: Taxi Fare Regressor (Numeric Input)

Prosise's Chapter 2 taxi model - three floats: day of week, hour, distance:

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Simple Ridge on 3 features (Prosise minimal example)
initial_type = [('float_input', FloatTensorType([None, 3]))]

onnx_model = convert_sklearn(model, initial_types=initial_type)

with open('models/taxi.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
```

### Chapter 02 pipeline (6 features + StandardScaler)

```python
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

pipe = joblib.load('models/taxi_pipeline.joblib')

n_features = pipe.named_steps['scaler'].n_features_in_
initial_type = [('float_input', FloatTensorType([None, n_features]))]

onnx_model = convert_sklearn(pipe, initial_types=initial_type, target_opset=17)

with open('models/taxi_pipeline.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
```

Feature order must match training columns exactly.

---

## Inference: Python ONNX Runtime

```python
import numpy as np
import onnxruntime as rt

session = rt.InferenceSession(
    'models/sentiment.onnx',
    providers=['CPUExecutionProvider'],
)

input_name = session.get_inputs()[0].name
# Outputs: [0]=label, [1]=probabilities for classifiers
label_name = session.get_outputs()[1].name

text = np.array(['Great food and excellent service!'], dtype=object).reshape(1, 1)
result = session.run([label_name], {input_name: text})
score = result[0][0][1]  # P(positive)
print(f'Sentiment: {score:.3f}')
```

> **In plain English:** Load the session **once** at process startup. Call `session.run()` per prediction - don't reload the file each time.

### Taxi fare (float input)

```python
import numpy as np
import onnxruntime as rt

session = rt.InferenceSession('models/taxi.onnx')
input_name = session.get_inputs()[0].name

# [day_of_week, hour, distance_miles]
features = np.array([[4.0, 17.0, 2.0]], dtype=np.float32)
output = session.run(None, {input_name: features})
fare = output[0][0][0]
print(f'Predicted fare: ${fare:.2f}')
```

---

## Parity Verification (Critical)

Never deploy ONNX without comparing to sklearn:

```python
import numpy as np
import joblib
import onnxruntime as rt

pipe = joblib.load('models/taxi_pipeline.joblib')
session = rt.InferenceSession('models/taxi_pipeline.onnx')

X_sample = np.array([[3.2, 2, 14, 4, 0, 1]], dtype=np.float32)

sklearn_pred = pipe.predict(X_sample)[0]
input_name = session.get_inputs()[0].name
onnx_pred = session.run(None, {input_name: X_sample})[0].ravel()[0]

tol = 1e-4
assert abs(sklearn_pred - onnx_pred) < tol, f'Mismatch: {sklearn_pred} vs {onnx_pred}'
print(f'Parity OK: ${sklearn_pred:.2f} ≈ ${onnx_pred:.2f}')
```

Tolerance depends on model - tree ensembles may differ slightly more than linear models.

---

## Inference: C# with Microsoft.ML.OnnxRuntime

Install NuGet package:

```bash
dotnet add package Microsoft.ML.OnnxRuntime
```

### Sentiment (string input) - Prosise

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

class Program
{
    static void Main(string[] args)
    {
        string text = args.Length > 0 ? args[0] : Console.ReadLine() ?? "";

        var tensor = new DenseTensor<string>(
            new[] { text }, new[] { 1, 1 });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("string_input", tensor)
        };

        using var session = new InferenceSession("models/sentiment.onnx");
        var results = session.Run(inputs);

        // Last output = probability map for classifiers
        var output = results.Last().AsEnumerable<NamedOnnxValue>();
        float score = output.First().AsDictionary<long, float>()[1];

        Console.WriteLine($"Sentiment score: {score:F3}");
    }
}
```

### Taxi fare (float input) - Prosise

```csharp
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Package input: day, hour, distance
var input = new float[] { 4.0f, 17.0f, 2.0f };
var tensor = new DenseTensor<float>(input, new[] { 1, 3 });

using var session = new InferenceSession("models/taxi.onnx");
var outputs = session.Run(new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("float_input", tensor)
});

float fare = outputs.First().AsTensor<float>().First();
Console.WriteLine($"Predicted fare: ${fare:F2}");
```

No Python. No HTTP. Native C# performance.

---

## Performance: REST vs ONNX (Prosise)

| Method | Avg latency (local sentiment) |
|--------|-------------------------------|
| Flask REST | > 2 seconds |
| Python ONNX Runtime | ~0.001 seconds |

Three orders of magnitude - plus WAN latency if REST is remote. Batch scoring 1M rows in C# strongly favors ONNX.

---

## Conversion Limitations

Not every sklearn estimator exports cleanly:

| Usually works | Often problematic |
|---------------|-------------------|
| Linear models, logistic regression | Custom transformers |
| Tree ensembles (with opset care) | Exotic kernels |
| StandardScaler, OneHotEncoder | NLTK inside Pipeline |
| HashingVectorizer | Some meta-estimators |

Check [skl2onnx supported ops](https://onnx.ai/sklearn-onnx/) before committing to ONNX. Fall back to REST if conversion fails.

---

## Opset Versioning

```python
onnx_model = convert_sklearn(pipe, initial_types=initial_type, target_opset=17)
```

Pin `target_opset` in export and document it in metadata. Older ONNX Runtime builds may not support the latest ops.

---

## ONNX in Docker (Optional Pattern)

Serve ONNX via a lightweight runtime container instead of full sklearn:

```dockerfile
FROM python:3.11-slim
RUN pip install onnxruntime fastapi uvicorn numpy
COPY onnx_api.py models/taxi_pipeline.onnx ./
CMD ["uvicorn", "onnx_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Smaller dependency footprint than full scikit-learn - useful at scale.

---

## Check Your Understanding

1. What does `initial_types` specify during ONNX export?
2. Why must you run a parity test before deploying ONNX?
3. When is ONNX faster than REST, and why?
4. What NuGet package loads ONNX models in C#?
5. Why load `InferenceSession` once instead of per prediction?

---

## Next Steps

- **Section 7.6:** Train natively in C# with ML.NET  
- **Lab 07:** Export taxi model to ONNX and verify parity

**Previous:** [Section 7.4](./section-04-docker-for-ml-services.md)  
**Next:** [Section 7.6 - ML.NET Overview](./section-06-ml-net-overview.md)



