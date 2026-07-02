# Section 7.6: ML.NET Overview

> **Source:** Prosise, Ch. 7 - "Building ML Models in C# with ML.NET"  
> **Prerequisites:** [Section 7.3](./section-03-consuming-models-from-c-sharp.md) | [Section 7.5](./section-05-onnx-export-and-inference.md)  
> **Glossary:** [ONNX](../../GLOSSARY.md#onnx) | [model](../../GLOSSARY.md#model)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Why ML.NET?

Scikit-learn dominates Python ML. But if your **client and team are .NET-native**, bridging Python adds friction - pickle files, Docker services, ONNX conversion.

**ML.NET** is Microsoft's free, open-source, cross-platform ML library for C#. Prosise: *"When it comes to writing ML/AI solutions in C#, there is no better tool for the job."*

| Advantage | Detail |
|-----------|--------|
| Same language | Train and score in C# - no Python runtime |
| Type safety | Strongly typed input/output schemas |
| Performance | Compiled code; IDataView streams huge datasets |
| ONNX built-in | Load `.onnx` from Python or deep-learning frameworks |
| Production pedigree | Derived from internal Microsoft ML used for 10+ years |

---

## ML.NET vs Scikit-Learn vs ONNX Bridge

| Path | When to use |
|------|-------------|
| **sklearn → REST** | Polyglot clients, quick MVP |
| **sklearn → ONNX → ML.NET** | Python training, C# inference |
| **ML.NET end-to-end** | .NET-only team, greenfield project |
| **ML.NET + ONNX import** | Deep models trained elsewhere |

Prosise's 2019 ML.NET paper: on a 9 GB Amazon review dataset, sklearn and H2O could not process in memory; ML.NET trained at 95% accuracy. On 10% subsample, ML.NET was most accurate and **6× faster than sklearn**.

---

## Core Architecture

```
MLContext
  ├── Data (LoadFromTextFile, LoadFromEnumerable, …)
  ├── Transforms (FeaturizeText, NormalizeMinMax, …)
  ├── BinaryClassification / Regression / Clustering Trainers
  └── Model (Save, Load, CreatePredictionEngine)
```

Every ML.NET app starts with `MLContext` - analogous to sklearn's chapter namespace + random seed control.

---

## Sentiment Analysis (Prosise Equivalent)

Equivalent to Python Example 7-1 (`CountVectorizer` + `LogisticRegression`):

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;

// Input schema - maps CSV columns
public class ReviewInput
{
    [LoadColumn(0)]
    public string Text { get; set; } = "";

    [LoadColumn(1), ColumnName("Label")]
    public bool Sentiment { get; set; }
}

public class ReviewOutput
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }

    public float Probability { get; set; }
}

class Program
{
    static void Main()
    {
        var context = new MLContext(seed: 0);

        // Load data
        var data = context.Data.LoadFromTextFile<ReviewInput>(
            "data/reviews.csv",
            hasHeader: true,
            separatorChar: ',',
            allowQuoting: true);

        var split = context.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);

        // Pipeline: text featurization + logistic regression
        var pipeline = context.Transforms.Text.FeaturizeText(
                outputColumnName: "Features",
                inputColumnName: nameof(ReviewInput.Text))
            .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

        var model = pipeline.Fit(split.TrainSet);

        // Evaluate
        var predictions = model.Transform(split.TestSet);
        var metrics = context.BinaryClassification.Evaluate(predictions);
        Console.WriteLine($"AUC-PR: {metrics.AreaUnderPrecisionRecallCurve:P2}");

        // Single prediction
        var engine = context.Model.CreatePredictionEngine<ReviewInput, ReviewOutput>(model);
        var result = engine.Predict(new ReviewInput
        {
            Text = "Among the best movies I have ever seen"
        });
        Console.WriteLine($"Sentiment score: {result.Probability:F3}");
    }
}
```

> **In plain English:** `FeaturizeText` ≈ `CountVectorizer`. `SdcaLogisticRegression` ≈ `LogisticRegression`. `Fit` ≈ `fit`. `CreatePredictionEngine` ≈ a thread-safe `predict` wrapper.

---

## IDataView: Unlimited-Size Data

Pandas DataFrames must fit in RAM. ML.NET's **IDataView** uses a cursor pattern - like a SQL query over data:

$$
\text{IDataView} \leftrightarrow \text{SELECT features, label FROM reviews.csv}
$$
> **Readable form:** data is streamed row-by-row; you never need the full 9 GB in memory at once.

This is why ML.NET handled Prosise's full Amazon dataset when sklearn could not.

---

## Prediction Engine vs Bulk Transform

| API | Use case |
|-----|----------|
| `CreatePredictionEngine<TIn,TOut>` | Single-row, low-latency API calls |
| `model.Transform(dataView)` | Batch scoring entire files |

For high-traffic services, create **multiple prediction engines** (pool) - ML.NET designed `PredictionEngine` for single-threaded use.

```csharp
// Batch scoring
var scored = model.Transform(testData);
```

---

## Saving and Loading Models

Analogous to pickle/joblib:

```csharp
// Save trained pipeline to zip
context.Model.Save(model, data.Schema, "models/sentiment.zip");

// Load
var loadedModel = context.Model.Load("models/sentiment.zip", out DataViewSchema schema);
var engine = context.Model.CreatePredictionEngine<ReviewInput, ReviewOutput>(loadedModel);
```

ML.NET models are **`.zip` archives** - not pickle. Safe to share within .NET ecosystem; still verify provenance.

---

## Taxi Fare Regression in ML.NET

```csharp
public class TaxiTrip
{
    [LoadColumn(0)] public float TripDistance { get; set; }
    [LoadColumn(1)] public float PassengerCount { get; set; }
    [LoadColumn(2)] public float Hour { get; set; }
    [LoadColumn(3)] public float DayOfWeek { get; set; }
    [LoadColumn(4)] public float IsWeekend { get; set; }
    [LoadColumn(5)] public float IsRush { get; set; }
    [LoadColumn(6)] public float FareAmount { get; set; }  // label
}

public class FarePrediction
{
    [ColumnName("Score")]
    public float PredictedFare { get; set; }
}

var pipeline = context.Transforms.Concatenate("Features",
        nameof(TaxiTrip.TripDistance),
        nameof(TaxiTrip.PassengerCount),
        nameof(TaxiTrip.Hour),
        nameof(TaxiTrip.DayOfWeek),
        nameof(TaxiTrip.IsWeekend),
        nameof(TaxiTrip.IsRush))
    .Append(context.Transforms.NormalizeMinMax("Features"))
    .Append(context.Regression.Trainers.Sdca());

var model = pipeline.Fit(trainData);
```

Mirrors Chapter 02 sklearn pipeline: feature concat → scale → regressor.

---

## Loading ONNX Models in ML.NET

ML.NET can score ONNX models without skl2onnx conversion logic in your app:

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;

var context = new MLContext();
var pipeline = context.Transforms.ApplyOnnxModel(
    modelFile: "models/taxi_pipeline.onnx",
    outputColumnNames: new[] { "variable" },
    inputColumnNames: new[] { "float_input" });

// Wrap in prediction pipeline with input schema...
```

Useful when training stays in Python but you want ML.NET's hosting, tooling, and .NET integration.

---

## Model Builder (AutoML UI)

**ML.NET Model Builder** (Visual Studio extension) provides a GUI:

1. Select scenario (classification, regression, recommendation)
2. Point at CSV/SQL
3. Train and compare algorithms
4. Export consumable C# code + `.zip` model

Great for .NET developers new to ML - analogous to sklearn + minimal code gen.

---

## NimbusML: sklearn ↔ ML.NET Bridge

**NimbusML** provides Python bindings to ML.NET trainers from Jupyter. Teams can prototype in Python notebooks using ML.NET algorithms, then deploy the `.zip` in C#.

```python
# Conceptual - requires nimbusml package
from nimbusml import Pipeline, LogisticRegressionBinaryClassifier
# Train with ML.NET backend from Python
```

Niche but valuable for mixed teams.

---

## ML.NET Limitations (Honest Assessment)

| sklearn has | ML.NET gap / workaround |
|-----------|-------------------------|
| Full NLP ecosystem | FeaturizeText; ONNX for transformers |
| Arbitrary custom steps | C# custom transforms |
| Deep learning from scratch | Import ONNX; use ML.NET DL extensions |
| Huge community recipes | Growing samples on GitHub |

For novel research, train in Python. For .NET product integration, ML.NET shines.

---

## Project Setup

```bash
dotnet new console -n TaxiFareMlNet
cd TaxiFareMlNet
dotnet add package Microsoft.ML
dotnet run
```

Target `net8.0` or later for current ML.NET releases.

---

## Check Your Understanding

1. What is `MLContext` and what sklearn concept does it resemble?
2. Why does ML.NET use `CreatePredictionEngine` instead of calling `Predict` on the model directly?
3. How does IDataView differ from a Pandas DataFrame?
4. What file format does ML.NET use to save models?
5. When would you import an ONNX file into ML.NET instead of training natively?

---

## Next Steps

- **Section 7.7:** Excel integration for business users  
- **Lab 07 (optional):** Score `taxi.onnx` from a C# console app

**Previous:** [Section 7.5](./section-05-onnx-export-and-inference.md)  
**Next:** [Section 7.7 - Excel Integration](./section-07-excel-integration.md)



