# Section 10.3: Building a CNN in Keras

> **Source:** Prosise, Ch. 10 - Keras CNN architecture for image classification  
> **Prerequisites:** [Section 10.2](./section-02-convolution-and-pooling.md) | [Chapter 09](../chapter-09-neural-networks/README.md)  
> **Glossary:** [neural-network](../../GLOSSARY.md#neural-network) | [deep-learning](../../GLOSSARY.md#deep-learning) | [overfitting](../../GLOSSARY.md#overfitting)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Canonical Stack

Prosise's teaching CNN for image [classification](../../GLOSSARY.md#classification) follows a repeatable template:

```
Input → [Conv → Conv → Pool]×N → Flatten → Dense → Dropout → Softmax
```

Each **conv block** extracts richer features at lower resolution. The **Dense head** maps global features to class probabilities.

> **Humorous analogy:** Conv blocks are telescopes zooming from "something fuzzy and white" to "tusk + blubber shape." The Dense head is the naturalist shouting "walrus!"

> **In plain English:** Stack convolution and pooling layers, then attach the same kind of classifier head you used in Chapter 09.

---

## Minimal Working CNN

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_small_cnn(input_shape=(128, 128, 3), num_classes=4):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1: 128×128×3 → 64×64×32
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),

        # Block 2: 64×64×32 → 32×32×64
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),

        # Block 3: 32×32×64 → 16×16×128
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model

model = build_small_cnn(num_classes=4)
model.summary()
```

For Arctic wildlife (polar bear, penguin, walrus, arctic fox), `num_classes=4`.

---

## Compile: Same Rules as Chapter 09

Multi-class single-label images use **softmax** + **categorical cross-entropy**:

$$
\mathcal{L} = -\sum_{c=1}^{K} y_c \log(\hat{y}_c)
$$
> **Readable form:** loss = negative sum over classes of (true one-hot × log predicted probability)

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
```

| Problem | Output activation | Loss |
|---------|-------------------|------|
| Multi-class (wildlife) | softmax | `categorical_crossentropy` |
| Binary | sigmoid | `binary_crossentropy` |
| Integer labels 0..K-1 | softmax | `sparse_categorical_crossentropy` |

Use `sparse_categorical_crossentropy` when labels are integers - no one-hot encoding required.

---

## Training Loop

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ModelCheckpoint('arctic_cnn_best.keras', monitor='val_accuracy', save_best_only=True),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks,
)
```

`train_ds` and `val_ds` are `tf.data` or `ImageDataGenerator` flows ([Section 10.4](./section-04-arctic-wildlife-classifier.md)).

Plot curves:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy'); plt.legend()
plt.tight_layout()
plt.show()
```

Gap between train and val accuracy signals [overfitting](../../GLOSSARY.md#overfitting) - address with [augmentation](./section-06-data-augmentation.md) and Dropout.

---

## Channel Progression Rationale

| After block | Spatial | Channels | Params (order of magnitude) |
|-------------|---------|----------|----------------------------|
| Input | 128² | 3 | - |
| Pool 1 | 64² | 32 | ~9k per conv |
| Pool 2 | 32² | 64 | ~37k per conv |
| Pool 3 | 16² | 128 | ~147k per conv |
| Flatten | - | 32,768 | - |
| Dense 256 | - | - | ~8.4M |

The Dense layer dominates parameters. For small datasets, consider **GlobalAveragePooling2D** instead of Flatten:

```python
layers.GlobalAveragePooling2D(),  # (16,16,128) → (128,)
layers.Dense(128, activation='relu'),
layers.Dropout(0.5),
layers.Dense(num_classes, activation='softmax'),
```

This drops millions of weights - often better generalization on Arctic-scale data.

---

## Functional API for Branched Models

When you need multiple inputs or shared layers, use the Functional API ([Chapter 09](../chapter-09-neural-networks/README.md)):

```python
inputs = layers.Input(shape=(128, 128, 3))
x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(4, activation='softmax')(x)
model = models.Model(inputs, outputs)
```

Transfer learning ([Section 10.5](./section-05-transfer-learning-with-resnet50v2.md)) **requires** Functional API - cut a pretrained base and attach a new head.

---

## Batch Normalization (Optional Upgrade)

**BatchNorm** normalizes activations per mini-batch - often faster convergence:

```python
layers.Conv2D(64, 3, padding='same', use_bias=False),
layers.BatchNormalization(),
layers.Activation('relu'),
```

Order varies by paper; Keras `Conv2D(..., activation='relu')` without BatchNorm is fine for Prosise's first pass.

---

## Input Size Tradeoffs

| `target_size` | Pros | Cons |
|---------------|------|------|
| 64×64 | Fast on CPU | Loses fine detail |
| 128×128 | Prosise Arctic default | Balanced |
| 224×224 | Matches ImageNet backbones | Slower scratch training |

Keep scratch CNNs at 128×128; use 224×224 when feeding ResNet50V2.

---

## Regularization Checklist

From [Chapter 09](../chapter-09-neural-networks/README.md):

| Technique | Where | Arctic wildlife |
|-----------|-------|-----------------|
| Dropout(0.5) | Before final Dense | Yes |
| L2 on Dense | `kernel_regularizer` | Optional |
| EarlyStopping | Callbacks | Yes |
| Data augmentation | `ImageDataGenerator` | **Critical** ([Section 10.6](./section-06-data-augmentation.md)) |
| Fewer filters | Architecture | Prefer 32-64-128 over 512 |

---

## Prediction & Interpretation

```python
import numpy as np
from tensorflow.keras.preprocessing import image

img_path = 'test/penguin_01.jpg'
img = image.load_img(img_path, target_size=(128, 128))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

probs = model.predict(x)[0]
class_names = ['arctic_fox', 'penguin', 'polar_bear', 'walrus']
pred_idx = np.argmax(probs)

print(f'Predicted: {class_names[pred_idx]} ({probs[pred_idx]:.2%})')
for name, p in zip(class_names, probs):
    print(f'  {name}: {p:.2%}')
```

Save `class_names` alongside the model in production ([Chapter 07](../chapter-07-operationalizing-models/README.md)).

---

## Evaluate on Held-Out Test Set

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

y_true, y_pred = [], []
for batch_x, batch_y in test_ds:
    preds = model.predict(batch_x, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(batch_y, axis=1))

print(classification_report(y_true, y_pred, target_names=class_names))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True'); plt.xlabel('Predicted')
plt.title('Arctic CNN Confusion Matrix')
plt.show()
```

Walrus ↔ polar bear confusion often reflects similar ice backgrounds - augmentation helps.

---

## Save & Reload

```python
model.save('arctic_wildlife_cnn.keras')  # Keras 3 native format

loaded = tf.keras.models.load_model('arctic_wildlife_cnn.keras')
assert loaded.evaluate(test_ds)[1] == model.evaluate(test_ds)[1]
```

Also export SavedModel for TensorFlow Serving if deploying ([Chapter 07](../chapter-07-operationalizing-models/README.md)).

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Negative dimension` | Wrong input shape | Include channel dim |
| `val_accuracy` stuck at 25% | 4 classes random | Check labels, LR, data paths |
| OOM GPU | Batch too large | Reduce `batch_size` to 16 or 8 |
| NaN loss | LR too high | Try `1e-4`, check scaling |

---

## Architecture Comparison Table

| Model | Params | Arctic val acc (typical) | Training time |
|-------|--------|--------------------------|---------------|
| Small 3-block CNN | ~2-5M | 75-85% scratch | Minutes-hours |
| + augmentation | same | 80-88% | same |
| ResNet50V2 transfer | ~24M (mostly frozen) | 90-95%+ | Fewer epochs ([Section 10.5](./section-05-transfer-learning-with-resnet50v2.md)) |

---

## Self-Check

1. What layer converts spatial features to class logits?
2. Why is Dropout placed before the final Dense?
3. What loss pairs with softmax for 4 wildlife classes?
4. How does GlobalAveragePooling2D reduce overfitting vs Flatten?
5. What callbacks prevent wasted epochs?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 10 - building CNNs
- [TensorFlow CNN tutorial](https://www.tensorflow.org/tutorials/images/cnn)
- [Keras Sequential API](https://keras.io/guides/sequential_model/)
- [Keras Functional API](https://keras.io/guides/functional_api/)
- [Chapter 09 callbacks](../chapter-09-neural-networks/README.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 10.2](./section-02-convolution-and-pooling.md) | **Next:** [Section 10.4](./section-04-arctic-wildlife-classifier.md)



