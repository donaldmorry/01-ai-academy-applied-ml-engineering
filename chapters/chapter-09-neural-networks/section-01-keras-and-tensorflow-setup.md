# Section 9.1: Keras & TensorFlow Setup

> **Source:** Prosise, Ch. 9 - "Neural Networks" (TensorFlow & Keras introduction)  
> **Prerequisites:** [Chapter 08 - Deep Learning Foundations](../chapter-08-deep-learning/README.md)  
> **Glossary:** [deep-learning](../../GLOSSARY.md#deep-learning) | [neural-network](../../GLOSSARY.md#neural-network) | [tensorflow](../../GLOSSARY.md#tensorflow) | [keras](../../GLOSSARY.md#keras)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## From Theory to Framework

[Chapter 08](../chapter-08-deep-learning/README.md) explained how [neural networks](../../GLOSSARY.md#neural-network) work: layers, activations, [backpropagation](../../GLOSSARY.md#backpropagation), optimizers. This chapter puts those ideas into production code.

Prosise's Chapter 9 settles on one stack the industry has largely standardized on:

| Layer | Role |
|-------|------|
| **[TensorFlow](../../GLOSSARY.md#tensorflow)** | Low-level engine - tensors, GPU ops, SavedModel export |
| **[Keras](../../GLOSSARY.md#keras)** | High-level API - `Sequential`, `Dense`, `fit()` in a few lines |

> **In plain English:** TensorFlow does the math at scale; Keras is the Python interface you actually write day to day. Google recommends Keras on top of TensorFlow 2.x.

---

## What Is a Tensor?

A **tensor** is a generalized array:

| Rank | Name | Example |
|------|------|---------|
| 0 | Scalar | Single fare amount: $12.50 |
| 1 | Vector | Three features: `[day, hour, distance]` |
| 2 | Matrix | Batch of 100 samples × 3 features |
| 3+ | Higher-order | Image batch: `(batch, height, width, channels)` |

A feedforward network is a **directed graph of tensor transformations**. The three-layer perceptron from Chapter 8 takes a 1D tensor (2 inputs), passes through a hidden layer (3 neurons), and outputs a scalar.

---

## Install & Verify

```python
# Recommended: TensorFlow 2.x bundles Keras
# pip install tensorflow

import tensorflow as tf
from tensorflow import keras

print(f"TensorFlow: {tf.__version__}")
print(f"Keras:      {keras.__version__}")
print(f"Eager mode: {tf.executing_eagerly()}")  # True in TF 2.x by default
```

### GPU vs CPU

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU available: {gpus}")
    # Optional: limit memory growth to avoid grabbing all VRAM
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU - training on CPU (fine for tabular sections in this chapter)")
```

| Workload | CPU OK? | GPU helps? |
|----------|---------|------------|
| Taxi fare (38K rows, Dense) | Yes | Marginal |
| Fraud (285K rows, Dense) | Yes | Moderate |
| LFW faces (1K images, Dense) | Yes | Moderate |
| CNNs ([Chapter 10](../chapter-10-convolutional-neural-networks/README.md)) | Slow | **Essential** |

> **Humorous analogy:** A GPU for tabular Dense networks is like a sports car in a school zone - nice to have, not required. For CNNs it's the autobahn.

---

## Eager Execution (TF 2.x)

TensorFlow 1.x built static graphs; debugging was painful. **TF 2.x runs eagerly by default** - operations execute immediately, like NumPy with autograd.

```python
import numpy as np

a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0], [6.0]])
c = tf.matmul(a, b)  # Executes right away - inspect with print(c.numpy())
print(c.numpy())
```

Behind the scenes, `@tf.function` can still compile hot paths for speed. Keras handles this when you call `model.compile()` and `model.fit()`.

---

## The Two Keras APIs

Prosise introduces both; this course focuses on **Sequential** (sufficient for feedforward networks):

### Sequential API - layer stack

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1)  # regression output - no activation
])
model.summary()
```

### Functional API - multiple inputs/outputs, shared layers

```python
from tensorflow.keras import Input
from tensorflow.keras.models import Model

inputs = Input(shape=(10,))
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
outputs = Dense(1)(x)
model = Model(inputs=inputs, outputs=outputs)
```

Use Functional API when you need branching, skip connections, or multi-head outputs (object detection in later chapters). For taxi/fraud/face tabular problems, Sequential is enough.

---

## Reproducibility & Randomness

Neural networks are **stochastic**. Same code, different runs → slightly different metrics. Prosise says: embrace it, but control seeds when comparing architectures:

```python
import os
import random
import numpy as np
import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

**Weight initialization:** `Dense` layers default to **Glorot uniform** weights and **zero** biases. Different initializers (`HeNormal`, etc.) matter more for deep CNNs than shallow tabular nets.

---

## Project Layout

```
course-01-applied-ml-ai/
├── data/
│   ├── taxi-fares.csv
│   ├── creditcard.csv
│   └── ...
├── chapters/09-neural-networks/
│   ├── 01-keras-tensorflow-setup.md   ← you are here
│   └── lab-09.md
└── models/                            # saved checkpoints
```

```python
from pathlib import Path

DATA_DIR = Path('data')
MODEL_DIR = Path('models/module09')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

assert (DATA_DIR / 'taxi-fares.csv').exists() or True  # download if missing
```

---

## TensorFlow vs Scikit-Learn Mental Model

| Concept | Scikit-Learn | Keras |
|---------|--------------|-------|
| Define model | `RandomForestRegressor()` | `Sequential([Dense(...)])` |
| Configure | hyperparameters in constructor | `model.compile(optimizer, loss, metrics)` |
| Train | `model.fit(X, y)` | `model.fit(X, y, epochs=..., batch_size=...)` |
| Predict | `model.predict(X)` | `model.predict(X)` |
| Save | `joblib.dump` | `model.save('path')` |

The **fit/predict** symmetry is intentional - Keras was designed to feel like Scikit for deep learning.

---

## Common Import Pattern

Use this header in every notebook for this chapter:

```python
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, classification_report, roc_auc_score

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
sns.set_theme()
```

---

## Version Notes

| Topic | Guidance |
|-------|----------|
| `tensorflow.keras` vs standalone `keras` | Use `tensorflow.keras` (bundled, Google-supported) |
| `input_dim` vs `input_shape` | Prefer `input_shape=(n,)` for Dense; `input_dim=n` still works |
| Apple Silicon | `pip install tensorflow-macos` + `tensorflow-metal` |
| CUDA | Match TF version to CUDA/cuDNN per [tensorflow.org/install](https://www.tensorflow.org/install) |

---

## Self-Check

1. What is the difference between a tensor and a NumPy array in practice?
2. Why does TensorFlow 2.x default to eager execution?
3. When would you choose the Functional API over Sequential?
4. Why do two training runs on the same data rarely produce identical metrics?

---

## What's Next

[Section 9.2](./section-02-your-first-keras-model.md) builds a complete network: `compile()`, `fit()`, history plots, and `predict()` - the template every section in this chapter extends.

---

**Previous:** [Chapter 08](../chapter-08-deep-learning/README.md)  
**Next:** [Section 9.2 - Your First Keras Model](./section-02-your-first-keras-model.md)