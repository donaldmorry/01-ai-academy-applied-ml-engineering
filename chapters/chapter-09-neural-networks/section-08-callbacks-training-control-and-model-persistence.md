# Section 9.8: Callbacks, Training Control & Model Persistence

> **Source:** Prosise, Ch. 9 - "Saving and Loading Models" & "Keras Callbacks"  
> **Prerequisites:** [Section 9.7](./section-07-dropout-and-regularization.md)  
> **Glossary:** [epoch](../../GLOSSARY.md#epoch) | [early-stopping](../../GLOSSARY.md#early-stopping) | [keras](../../GLOSSARY.md#keras) | [tensorflow](../../GLOSSARY.md#tensorflow)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Epoch Problem

Validation accuracy rarely rises smoothly. It **peaks**, dips, oscillates - then you train too long and [overfit](../../GLOSSARY.md#overfitting).

**Worse:** if you mark the best epoch and retrain for exactly that many epochs, randomness gives different results.

Prosise's solution: **Keras callbacks** - hooks that run during training and can stop early, save checkpoints, or adjust learning rate.

---

## Callback Basics

```python
from tensorflow.keras.callbacks import Callback

class StopCallback(Callback):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get('val_accuracy', 0) >= self.threshold:
            print(f"\nStopping: val_accuracy >= {self.threshold}")
            self.model.stop_training = True

callback = StopCallback(0.95)
model.fit(X, y, validation_split=0.2, epochs=100, batch_size=20,
          callbacks=[callback])
```

**Hook methods:** `on_train_begin`, `on_epoch_begin`, `on_epoch_end`, `on_train_end`, `on_train_batch_begin`, `on_train_batch_end`.

`fit(callbacks=[...])` accepts a **list** - combine multiple callbacks.

---

## EarlyStopping

Most-used built-in callback. Stops when a monitored metric stops improving.

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_accuracy',    # metric to watch
    patience=5,                  # epochs with no improvement before stop
    restore_best_weights=True,   # rewind to best epoch weights
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=20,
    callbacks=[early_stop]
)
```

| Parameter | Meaning |
|-----------|-------------|
| `monitor` | `'val_loss'`, `'val_accuracy'`, `'val_mae'`, etc. |
| `patience` | Tolerate N epochs without improvement |
| `restore_best_weights=True` | **Critical** - final weights = best epoch, not last |
| `mode` | `'auto'`, `'min'` (loss), `'max'` (accuracy) |

### Regression variant

```python
early_stop_mae = EarlyStopping(
    monitor='val_mae',
    patience=10,
    restore_best_weights=True,
    mode='min'
)
```

### Monitor training loss instead

```python
EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
```

---

## ModelCheckpoint

Save the best model during training - pairs naturally with early stopping:

```python
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint

MODEL_DIR = Path('models/module09')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

checkpoint = ModelCheckpoint(
    filepath=str(MODEL_DIR / 'face_best.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=20,
    callbacks=[early_stop, checkpoint]
)
```

| Option | Effect |
|--------|--------|
| `save_best_only=True` | Keep only the best checkpoint |
| `save_weights_only=True` | Architecture not saved - need to rebuild model |
| `filepath` | `.keras`, `.h5`, or SavedModel directory |

---

## ReduceLROnPlateau

When validation loss plateaus, shrink the learning rate:

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,       # multiply LR by 0.5
    patience=3,
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stop, checkpoint, reduce_lr]
```

$$
\eta_{\text{new}} = \eta_{\text{old}} \times \text{factor}
$$
> **Readable form:** if val loss stalls for 3 epochs, halve the learning rate - finer weight adjustments

---

## TensorBoard

Log metrics, histograms, and graphs for visualization:

```python
from tensorflow.keras.callbacks import TensorBoard

tb = TensorBoard(
    log_dir='logs/face_nn',
    histogram_freq=1,
    write_graph=True
)

model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=100, callbacks=[tb])
```

Launch: `%tensorboard --logdir logs` → open the local URL in your browser. See TensorFlow docs for histogram and graph tabs.

---

## Saving & Loading Models (Prosise)

Operationalizing neural networks requires persisting **architecture + weights + optimizer state**.

### Option 1: Keras native `.keras` format (recommended TF 2.x)

```python
model.save('models/module09/taxi_model.keras')
```

### Option 2: HDF5 (legacy, still common)

```python
model.save('models/module09/taxi_model.h5')  # Prosise's original format
```

### Option 3: TensorFlow SavedModel (production)

```python
model.save('models/module09/taxi_savedmodel')  # directory, not single file
```

| Format | Pros | Cons |
|--------|------|------|
| `.keras` | Single file, TF 2.x native | Python/Keras ecosystem |
| `.h5` | Single file, widely seen in tutorials | Legacy; some TF features limited |
| **SavedModel** | Google recommended; ML.NET, TF Serving, TFLite | Directory of files |

> **Prosise:** No functional difference for Python reload - SavedModel exports to other runtimes.

### Load

```python
from tensorflow.keras.models import load_model

model = load_model('models/module09/taxi_model.keras')
# or: load_model('models/module09/taxi_model.h5')
# or: load_model('models/module09/taxi_savedmodel')

# Predictions identical to original (modulo floating-point)
pred = model.predict(X_test[:5], verbose=0)
```

### Weights only

```python
model.save_weights('models/module09/face_weights.weights.h5')

# Must rebuild architecture first
new_model = build_face_model()
new_model.load_weights('models/module09/face_weights.weights.h5')
```

---

## Production Callback Stack (Template)

```python
from pathlib import Path
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)

def get_callbacks(task_name, monitor='val_loss', mode='min'):
    out = Path(f'models/module09/{task_name}')
    out.mkdir(parents=True, exist_ok=True)
    return [
        EarlyStopping(monitor=monitor, patience=15,
                      restore_best_weights=True, mode=mode, verbose=1),
        ModelCheckpoint(filepath=str(out / 'best.keras'),
                        monitor=monitor, save_best_only=True, mode=mode, verbose=1),
        ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=5,
                          min_lr=1e-6, verbose=1),
        CSVLogger(str(out / 'history.csv'))
    ]

# Taxi regression
taxi_cbs = get_callbacks('taxi', monitor='val_mae', mode='min')
taxi_model.fit(X, y, validation_split=0.2, epochs=200,
               batch_size=100, callbacks=taxi_cbs)

# Fraud classification
fraud_cbs = get_callbacks('fraud', monitor='val_loss', mode='min')
fraud_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                epochs=50, batch_size=100, callbacks=fraud_cbs)
```

---

## Serialize Preprocessing Too

```python
import joblib

bundle = {
    'model_path': 'models/module09/taxi_best.keras',
    'scaler': scaler,           # if used
    'feature_columns': ['day_of_week', 'pickup_time', 'distance'],
    'threshold': 0.35           # fraud only
}
joblib.dump(bundle, 'models/module09/taxi_bundle.joblib')
```

Neural networks don't embed feature engineering - save metadata alongside weights.

---

## Summary Table

| Callback | Purpose |
|----------|---------|
| `EarlyStopping` | Stop at best epoch; restore best weights |
| `ModelCheckpoint` | Persist best model to disk |
| `ReduceLROnPlateau` | Adaptive learning rate decay |
| `TensorBoard` | Live training visualization |
| `CSVLogger` | Epoch log file |
| `LearningRateScheduler` | Custom LR schedule per epoch |

---

## Self-Check

1. What does `restore_best_weights=True` do when early stopping triggers?
2. When should you use SavedModel over `.h5`?
3. Why can loaded Keras models continue training but joblib-loaded RandomForests cannot?
4. How do you combine three callbacks in one `fit()` call?

---

## What's Next

[Lab 09](./section-lab-09-neural-network-triathlon.md) - the **Neural Network Triathlon**: taxi, fraud, and faces with callbacks, dropout, saved models, and baseline comparison tables.

---

## References

- Prosise, *Applied Machine Learning and AI for Engineers*, Ch. 9 - Callbacks, saving models
- [Keras Callbacks API](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks)
- [TensorFlow SavedModel guide](https://www.tensorflow.org/guide/saved_model)
- [EarlyStopping documentation](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)
- [Section 9.7](./section-07-dropout-and-regularization.md) | [Lab 09](./section-lab-09-neural-network-triathlon.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 9.7](./section-07-dropout-and-regularization.md)  
**Next:** [Lab 09 - Neural Network Triathlon](./section-lab-09-neural-network-triathlon.md)



