# Section 10.6: Data Augmentation

> **Source:** Prosise, Ch. 10 - fighting overfitting with transformed training images  
> **Prerequisites:** [Section 10.4](./section-04-arctic-wildlife-classifier.md) | [Section 10.5](./section-05-transfer-learning-with-resnet50v2.md)  
> **Glossary:** [overfitting](../../GLOSSARY.md#overfitting) | [deep-learning](../../GLOSSARY.md#deep-learning)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Small-Data Problem

Arctic wildlife datasets might have **150-300 images per species**. A CNN with millions of parameters will memorize training photos - spectacular train accuracy, weak validation performance. **Data augmentation** artificially expands diversity by applying label-preserving transforms to training images only.

> **Humorous analogy:** Augmentation is wardrobe changes for the same animal - same penguin, but flipped, zoomed, and slightly rotated. The model learns "penguin-ness," not "penguin facing left at pixel (42, 17)."

> **In plain English:** Randomly rotate, flip, shift, and zoom training images each epoch so the network sees new variations without collecting new photos.

---

## Augmentation vs Real New Data

| Approach | Pros | Cons |
|----------|------|------|
| Collect more photos | Best generalization | Expensive, slow |
| Augmentation | Free, instant | Cannot invent unseen angles |
| Transfer learning | Leverages ImageNet | Heavier model |

Prosise combines **augmentation + transfer learning** for Arctic wildlife - standard industry practice.

---

## ImageDataGenerator Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
)

val_datagen = ImageDataGenerator(rescale=1./255)  # NO augmentation on val/test
test_datagen = ImageDataGenerator(rescale=1./255)
```

**Critical rule:** never augment validation or test sets - metrics must reflect real unmodified images.

---

## Parameter Guide

| Parameter | Effect | Wildlife-safe range |
|-----------|--------|---------------------|
| `rotation_range` | Random rotate ± degrees | 10-30° |
| `width_shift_range` | Horizontal translate (fraction) | 0.1-0.2 |
| `height_shift_range` | Vertical translate | 0.1-0.2 |
| `zoom_range` | Random scale | 0.1-0.2 |
| `horizontal_flip` | Mirror left-right | **Yes** for most animals |
| `vertical_flip` | Upside-down | **No** for wildlife |
| `shear_range` | Skew transform | 0-0.15 |
| `brightness_range` | Lighting change | (0.8, 1.2) if needed |

```python
# BAD for wildlife - penguins rarely appear upside-down
# vertical_flip=True

# GOOD - polar bear facing left vs right is still a polar bear
horizontal_flip=True
```

---

## Visualize Augmentations

```python
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img('arctic_wildlife/train/penguin/sample.jpg', target_size=(128, 128))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

plt.figure(figsize=(12, 4))
plt.subplot(1, 5, 1)
plt.imshow(image.load_img('arctic_wildlife/train/penguin/sample.jpg', target_size=(128,128)))
plt.title('Original'); plt.axis('off')

for i in range(2, 6):
    plt.subplot(1, 5, i)
    batch = train_datagen.flow(x, batch_size=1)[0][0].astype('uint8')
    plt.imshow(batch)
    plt.title(f'Aug {i-1}'); plt.axis('off')
plt.suptitle('Random augmentations - same label')
plt.tight_layout()
plt.show()
```

Every augmented image keeps the **same folder label** - the penguin class.

---

## Keras Preprocessing Layers (Modern API)

Augmentation as model layers - runs on GPU inside `tf.data`:

```python
import tensorflow as tf

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.08),  # fraction of 2π
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
], name='augmentation')

# Only active during training when included before base:
inputs = tf.keras.Input(shape=(128, 128, 3))
x = data_augmentation(inputs)
x = tf.keras.layers.Rescaling(1./255)(x)
# ... conv blocks ...
```

When the model is in eval mode (`model.predict`), Keras disables random augmentation automatically.

---

## Augmentation + Transfer Learning

For ResNet50V2, light augmentation avoids destroying ImageNet-aligned statistics:

```python
from tensorflow.keras.applications.resnet_v2 import preprocess_input

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
])

inputs = tf.keras.Input(shape=(224, 224, 3))
x = augment(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
# ... head ...
```

Heavy color jitter on a frozen ResNet head is usually unnecessary in Prosise's first pipeline.

---

## Effective Dataset Size

If you have 200 training images and train 50 epochs with augmentation, the model sees **200 × 50 = 10,000** varied views - but not 10,000 independent samples. Augmentation reduces [overfitting](../../GLOSSARY.md#overfitting); it does not replace real diversity.

$$
N_{\text{effective}} \ll N_{\text{epochs}} \times N_{\text{images}}
$$
> **Readable form:** effective independent sample count is much less than epochs times images

---

## Measuring Impact

Train the same architecture with and without augmentation:

| Config | Train acc | Val acc | Gap |
|--------|-----------|---------|-----|
| No aug | 98% | 72% | 26% overfit |
| With aug | 88% | 84% | 4% healthier |

Lower train accuracy with **higher** val accuracy is a win.

```python
# Log experiment
results = {
    'scratch_no_aug': {'val_acc': 0.76},
    'scratch_aug': {'val_acc': 0.84},
    'resnet_aug': {'val_acc': 0.93},
}
```

---

## Harmful Augmentations for Wildlife

| Transform | Why harmful |
|-----------|-------------|
| `vertical_flip` | Unnatural orientation |
| Extreme rotation (>45°) | Labels ambiguous (belly vs back) |
| Random erasing huge patches | Removes species-defining features |
| Wrong species paste (cutout mix) | Needs advanced methods (CutMix) |

**CutMix / Mixup** - blend two images - can help large datasets but may confuse small 4-class wildlife sets. Prosise sticks to geometric transforms.

---

## Augmentation on tf.data Pipeline

```python
def augment_fn(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.clip_by_value(image, 0, 255)
    return image, label

train_ds = train_ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
```

Compose with `Rescaling` after augment on uint8 images.

---

## Test-Time Augmentation (TTA) - Optional

Average predictions over flipped versions at inference:

```python
def predict_tta(model, img_array):
    preds = []
    preds.append(model.predict(img_array, verbose=0))
    preds.append(model.predict(tf.image.flip_left_right(img_array), verbose=0))
  return np.mean(preds, axis=0)
```

Marginal gains (~1-2% accuracy) - use in competitions, not required for lab.

---

## Integration Checklist

1. Augment **train only**
2. Match `target_size` to model (128 scratch, 224 ResNet)
3. Keep validation generator **deterministic**
4. Pair with EarlyStopping
5. Document which transforms you used in lab writeup

---

## Self-Check

1. Why must validation data stay unaugmented?
2. Is `horizontal_flip` safe for penguin photos?
3. Why is `vertical_flip` usually disabled?
4. What's the difference between ImageDataGenerator and Keras preprocessing layers?
5. Does augmentation increase the number of files on disk?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 10 - data augmentation
- [Keras ImageDataGenerator](https://keras.io/api/preprocessing/image/)
- [Keras preprocessing layers](https://keras.io/api/layers/preprocessing_layers/)
- [TensorFlow data augmentation tutorial](https://www.tensorflow.org/tutorials/images/data_augmentation)
- [Section 10.4](./section-04-arctic-wildlife-classifier.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 10.5](./section-05-transfer-learning-with-resnet50v2.md) | **Next:** [Section 10.7](./section-07-fine-tuning-pretrained-models.md)



