# Section 10.5: Transfer Learning with ResNet50V2

> **Source:** Prosise, Ch. 10 - ImageNet pretrained models, feature extraction  
> **Prerequisites:** [Section 10.4](./section-04-arctic-wildlife-classifier.md)  
> **Glossary:** [deep-learning](../../GLOSSARY.md#deep-learning) | [neural-network](../../GLOSSARY.md#neural-network) | [overfitting](../../GLOSSARY.md#overfitting)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Why Transfer Learning?

Training a deep CNN from scratch on Arctic wildlife demands thousands of labeled images per class. **Transfer learning** reuses weights pretrained on **ImageNet** - 1.2 million images, 1000 categories - so lower layers already detect edges, textures, and object parts useful on your photos.

> **Humorous analogy:** Instead of teaching a new hire to read from the alphabet, you hire someone who already speaks fluent "vision" and only train them on the four animal names used on your cruise.

> **In plain English:** Download ResNet50V2 with ImageNet weights, freeze the body, attach a small new classifier head for your 4 wildlife classes, train only the head.

---

## ResNet50V2 Overview

**ResNet** (Residual Network) introduced **skip connections** - layers learn residuals $F(x)$ added to input $x$:

$$
\mathbf{h}_{l+1} = \mathbf{h}_l + F_l(\mathbf{h}_l)
$$
> **Readable form:** next layer output = previous output + learned correction

**ResNet50V2** has ~50 weight layers, V2 denotes improved batch norm / conv ordering. ImageNet top-1 accuracy ~76%.

| Property | Value |
|----------|-------|
| Default input | 224×224×3 |
| Pretrained weights | `imagenet` via Keras Applications |
| Output before head | 7×7×2048 feature maps |
| Parameters | ~23.6 million |

---

## Two Transfer Learning Modes

| Mode | What's trained | When to use |
|------|----------------|-------------|
| **Feature extraction** | New head only; base frozen | Small dataset (<1k images), first pass |
| **Fine-tuning** | Head + top N base layers | After head converges ([Section 10.7](./section-07-fine-tuning-pretrained-models.md)) |

Prosise starts with feature extraction on Arctic wildlife.

---

## Build the Model (Functional API)

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input

IMG_SIZE = (224, 224)
NUM_CLASSES = 4

base_model = ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3),
)
base_model.trainable = False  # freeze all layers

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = preprocess_input(inputs)  # ResNet-specific scaling
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.summary()
```

**`include_top=False`** removes ImageNet's 1000-class Dense head - you add your own.

**`preprocess_input`** maps pixels to the distribution ResNet saw during ImageNet training (not simply `/255`).

---

## Why 1000 Classes Become 4

The frozen backbone outputs a **general visual feature vector** - not ImageNet class decisions. The old 1000-neuron head is discarded. Your new Dense(4, softmax) maps features to polar bear / penguin / walrus / arctic fox.

$$
\hat{\mathbf{y}} = \text{softmax}(W \cdot \text{GAP}(\text{ResNet}(x)) + b), \quad W \in \mathbb{R}^{4 \times 2048}
$$
> **Readable form:** wildlife probabilities = softmax of (small weight matrix × pooled ResNet features + bias)

Only $4 \times 2048 + 4 \approx 8{,}000$ head weights train - vs 23M if unfrozen.

---

## Data Pipeline with Correct Preprocessing

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Option A: manual rescale then preprocess in model (shown above)
train_gen = ImageDataGenerator().flow_from_directory(
    'arctic_wildlife/train',
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode='categorical',
)

# Option B: no rescale in generator - preprocess_input in model handles it
```

Do **not** double-normalize: if using `preprocess_input` in the model, skip `rescale=1./255` in the generator.

---

## Compile & Train (Feature Extraction Phase)

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks,
)
```

Expect **faster convergence** and **higher val accuracy** than scratch CNN - often 90%+ in fewer epochs.

---

## Compare Scratch vs Transfer

| Metric | Scratch CNN (128×128) | ResNet50V2 frozen |
|--------|----------------------|-------------------|
| Trainable params | ~2-5M | ~8k (head only) |
| Epochs to converge | 30-50 | 5-15 |
| Val accuracy | 75-85% | 88-95% |
| Inference latency | Lower | Higher |
| Disk (weights) | ~10 MB | ~100 MB |

For Prosise's lab, transfer learning should **beat** scratch with less training time.

---

## Other ImageNet Backbones

Keras Applications offers interchangeable bases:

| Model | Params | Speed | Notes |
|-------|--------|-------|-------|
| MobileNetV2 | ~3.5M | Fast | Mobile/edge |
| EfficientNetB0 | ~5M | Medium | Good accuracy/size |
| ResNet50V2 | ~24M | Medium | Prosise default |
| VGG16 | ~138M | Slow | Simple, heavy |

```python
from tensorflow.keras.applications import MobileNetV2
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
```

Swap base - keep the same head pattern.

---

## Extract Features Without Training

For analysis or classical ML on top:

```python
feature_extractor = models.Model(
    inputs=model.input,
    outputs=base_model.output,
)
# Or GAP features:
gap_extractor = models.Model(inputs=model.input, outputs=model.layers[-3].output)

batch, _ = next(train_gen)
features = gap_extractor.predict(batch[:8])
print(features.shape)  # (8, 2048)
```

Useful for visualization and k-NN baselines on embeddings.

---

## Prediction with Preprocessing

```python
import numpy as np
from tensorflow.keras.preprocessing import image

img = image.load_img('probe.jpg', target_size=IMG_SIZE)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# preprocess_input applied inside model if wired as above

probs = model.predict(x, verbose=0)[0]
print(dict(zip(class_names, probs.round(3))))
```

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Wrong preprocessing | Stuck at chance accuracy | Use `resnet_v2.preprocess_input` |
| Forgot `base_model.trainable = False` | OOM, slow, unstable | Freeze before compile |
| Input 128×128 on ResNet | Shape mismatch or poor features | Resize to 224×224 |
| BatchNorm in train mode when frozen | Noisy features | `base_model(x, training=False)` |

---

## When NOT to Use ImageNet Transfer

- **Non-RGB inputs** (hyperspectral, medical) - lower layers may not transfer
- **Tiny embedded targets** - MobileNet or custom small CNN
- **Fundamentally different domain** (microscopy vs wildlife) - still try, but validate carefully

Arctic wildlife photos are close enough to ImageNet's natural images - transfer works well.

---

## Save Full Model

```python
model.save('models/arctic_resnet50v2_phase1.keras')
```

Includes frozen base + trained head. For fine-tuning, reload and unfreeze ([Section 10.7](./section-07-fine-tuning-pretrained-models.md)).

---

## Self-Check

1. What does `include_top=False` do?
2. Why use `preprocess_input` instead of dividing by 255?
3. How many parameters train in feature-extraction mode?
4. Why does ResNet50V2 expect 224×224 inputs?
5. When should you switch to fine-tuning?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 10 - transfer learning
- [Keras ResNet50V2](https://keras.io/api/applications/resnet/#resnet50v2-function)
- [He et al., Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [ImageNet dataset](https://www.image-net.org/)
- [TensorFlow transfer learning guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 10.4](./section-04-arctic-wildlife-classifier.md) | **Next:** [Section 10.6](./section-06-data-augmentation.md)
