# Section 10.1: Why CNNs for Images?

> **Source:** Prosise, Ch. 10 - "Convolutional Neural Networks" / spatial structure  
> **Prerequisites:** [Chapter 09 - Neural Networks](../chapter-09-neural-networks/README.md) | [Section 3.8](../chapter-03-classification-models/section-08-case-study-mnist-digit-recognition.md)  
> **Glossary:** [deep-learning](../../GLOSSARY.md#deep-learning) | [neural-network](../../GLOSSARY.md#neural-network) | [feature](../../GLOSSARY.md#feature)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Problem with Flattening

In [Chapter 09](../chapter-09-neural-networks/README.md), you fed MNIST digits and Olivetti faces into **Dense** layers by flattening images into long vectors. That works - SVMs hit ~98% on MNIST without any spatial awareness. But flattening treats pixel $(5, 12)$ and pixel $(25, 3)$ as unrelated [features](../../GLOSSARY.md#feature), even when they are neighbors on the same edge.

A 224×224 RGB photo has $224 \times 224 \times 3 = 150{,}528$ inputs. A fully connected first layer with 512 neurons creates:

$$
150{,}528 \times 512 \approx 77 \text{ million weights}
$$
> **Readable form:** first dense layer weight count = pixels × neurons ≈ 77 million parameters

That is expensive, data-hungry, and blind to the fact that **nearby pixels matter together**.

> **Humorous analogy:** Flattening a photo is like shredding a map into confetti, numbering each scrap, and asking someone to find the highway. The numbers are all there - the geography is gone.

> **In plain English:** Images are 2D grids with local structure. CNNs are built to exploit that structure instead of ignoring it.

---

## Three Core Ideas

Convolutional Neural Networks (CNNs) replace giant fully connected layers with three inductive biases:

| Idea | What it means | Why it helps |
|------|---------------|--------------|
| **Local connectivity** | Each neuron sees a small patch (e.g., 3×3) | Edges and textures are local patterns |
| **Parameter sharing** | Same filter weights slide across the image | Detect "vertical edge" everywhere with few weights |
| **Translation equivariance** | Shift input → feature map shifts the same way | A polar bear in the corner activates the same filters as one in the center |

Together, these let a CNN learn **hierarchical representations**:

```
Pixels → edges → textures → parts (eye, wing) → objects (penguin, walrus)
```

This is [representation learning](../../GLOSSARY.md#representation-learning) specialized for grid data.

---

## Spatial Locality

Human vision starts with edge detectors in the retina and V1 cortex. CNNs mimic that stack:

1. **Layer 1** - oriented edges, color blobs
2. **Layer 2-3** - corners, simple textures
3. **Deeper layers** - object parts and whole objects

A 3×3 convolution at position $(i, j)$ combines only 9 neighboring pixels (per channel):

$$
z_{i,j} = \sum_{u=0}^{2}\sum_{v=0}^{2} w_{u,v} \cdot x_{i+u,\,j+v} + b
$$
> **Readable form:** output at row i, column j = sum of (filter weight × pixel value) over a 3×3 neighborhood + bias

Compare to a Dense layer where every output connects to **all** 150k inputs.

---

## Parameter Sharing in Numbers

One 3×3 filter on a 3-channel RGB image uses:

$$
3 \times 3 \times 3 + 1 = 28 \text{ parameters}
$$
> **Readable form:** one small filter = 3×3×3 weights + 1 bias = 28 learnable numbers

That single filter produces a **feature map** the same size as the input (before pooling). With 32 filters you have $32 \times 28 = 896$ parameters - not 77 million.

| Architecture | First-layer params (224×224×3 input) |
|--------------|--------------------------------------|
| Dense(512) | ~77M |
| Conv2D(32, 3×3) | ~896 |
| Conv2D(64, 3×3) | ~1,792 |

Same input. Orders of magnitude fewer weights. Better sample efficiency.

---

## Translation Invariance (Pooling Helps)

**Equivariance:** if the walrus moves left, the "tusk texture" activation moves left.

**Invariance:** after [max pooling](./section-02-convolution-and-pooling.md), small shifts may leave the pooled value unchanged - the network cares less about exact pixel location.

Prosise emphasizes this when moving from MNIST Dense nets to CNNs: MNIST digits are centered; natural photos are not.

```python
import numpy as np
import matplotlib.pyplot as plt

# Toy 8×8 image with a bright blob
img = np.zeros((8, 8))
img[2:5, 3:6] = 1.0

# Shift blob right by 2 columns
shifted = np.zeros_like(img)
shifted[2:5, 5:8] = 1.0

# Simple vertical edge filter
edge_filter = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

def conv2d_valid(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = image[i:i+kh, j:j+kw]
            out[i, j] = np.sum(patch * kernel)
    return out

feat_orig = conv2d_valid(img, edge_filter)
feat_shift = conv2d_valid(shifted, edge_filter)

fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes[0,0].imshow(img, cmap='gray'); axes[0,0].set_title('Original')
axes[0,1].imshow(feat_orig, cmap='viridis'); axes[0,1].set_title('Feature map')
axes[1,0].imshow(shifted, cmap='gray'); axes[1,0].set_title('Shifted input')
axes[1,1].imshow(feat_shift, cmap='viridis'); axes[1,1].set_title('Shifted features')
for ax in axes.ravel(): ax.axis('off')
plt.suptitle('Convolution is equivariant to translation')
plt.tight_layout()
plt.show()
```

The feature maps look like shifted versions of each other - the filter does not need retraining for each position.

---

## When Flatten + Dense Still Makes Sense

| Scenario | Flatten + Dense | CNN |
|----------|-----------------|-----|
| Tiny images (28×28 MNIST) | Competitive with tuning | Better, but margin smaller |
| Tabular data | **Correct choice** | Wrong tool |
| Large natural images | Poor sample efficiency | **Standard** |
| Pre-extracted features (HOG) | [Chapter 05 SVM](../chapter-05-support-vector-machines/section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md) pipeline | Optional CNN on crops |

Prosise's arc: classical features + SVM on faces ([Chapter 05](../chapter-05-support-vector-machines/section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md)) → flatten + Dense ([Chapter 09](../chapter-09-neural-networks/README.md)) → CNNs (this chapter) → specialized detectors ([Chapter 11](../chapter-11-face-detection-recognition/README.md), [Chapter 12](../chapter-12-object-detection/README.md)).

---

## Image Tensors: Shape Conventions

Keras expects **channels-last** layout for TensorFlow:

$$
\text{batch shape} = (\text{samples},\; \text{height},\; \text{width},\; \text{channels})
$$
> **Readable form:** a batch of images = (how many images, rows, columns, color channels)

| Image type | Shape for one sample | Channel meaning |
|------------|---------------------|-----------------|
| Grayscale | (64, 64, 1) | Single intensity |
| RGB | (224, 224, 3) | Red, green, blue |
| RGBA | (128, 128, 4) | + alpha transparency |

```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array

path = 'arctic/polar_bear/img_001.jpg'  # example
img = load_img(path, target_size=(224, 224))
arr = img_to_array(img)           # (224, 224, 3)
batch = arr.reshape(1, *arr.shape)  # (1, 224, 224, 3) for model.predict
print(arr.shape, arr.min(), arr.max())  # values often 0-255 before scaling
```

Always scale consistently: divide by 255, or use preprocessing specific to a pretrained backbone ([Section 10.5](./section-05-transfer-learning-with-resnet50v2.md)).

---

## CNN vs Human-Engineered Features

[Section 5.7](../chapter-05-support-vector-machines/section-07-facial-recognition-with-svm-hog-and-olivetti-faces.md) used **HOG** - histograms of oriented gradients hand-designed by researchers. CNNs **learn** filters from data:

| Approach | Who designs features | Typical data needs |
|----------|---------------------|-------------------|
| HOG + SVM | Human expert | Hundreds-thousands |
| CNN from scratch | Network + data | Thousands-millions |
| Transfer learning | ImageNet pretraining + your head | Hundreds ([Section 10.5](./section-05-transfer-learning-with-resnet50v2.md)) |

For Arctic wildlife with ~200 images per species, you will not train 77M dense weights - you will use convolutions and transfer learning.

---

## Receptive Field Preview

Each deeper layer "sees" a larger patch of the original image - the **receptive field**. Stacking 3×3 convolutions grows the field without huge kernels:

| Stack depth | Approximate receptive field |
|-------------|----------------------------|
| 1× Conv 3×3 | 3×3 pixels |
| 2× Conv 3×3 | 5×5 pixels |
| 3× Conv 3×3 | 7×7 pixels |

Deeper networks combine local evidence into global object understanding. [Section 10.2](./section-02-convolution-and-pooling.md) covers the arithmetic; [Section 10.3](./section-03-building-a-cnn-in-keras.md) builds the full stack.

---

## Connection to Chapter 09

Your Keras skills transfer directly:

| Chapter 09 skill | Chapter 10 usage |
|-----------------|-----------------|
| `model.compile()` | Same - choose `categorical_crossentropy` for multi-class wildlife |
| `fit()` + validation | Same - watch val_accuracy for [overfitting](../../GLOSSARY.md#overfitting) |
| Callbacks (EarlyStopping) | Essential on small Arctic dataset |
| Dropout | Used in Dense head; sometimes in CNN blocks |

The new pieces are `Conv2D`, `MaxPooling2D`, and `Flatten` - spatial layers before the familiar Dense classifier head.

---

## Decision Checklist: Do I Need a CNN?

Ask before coding:

1. **Is the input a grid** (image, spectrogram, heatmap)? If no → not a CNN.
2. **Does spatial layout matter?** If features are tabular rows → Dense/gradient boosting.
3. **How much labeled data?** If <1k images → plan augmentation + transfer learning.
4. **Latency budget?** Small CNNs on CPU are fine; huge ResNets may need GPU.

---

## Self-Check

1. Why does a Dense layer on full-resolution photos have so many parameters?
2. What are the three inductive biases of CNNs?
3. What is the difference between equivariance and invariance?
4. What tensor shape does Keras expect for a batch of RGB images?
5. When might HOG + SVM still be a reasonable choice?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 10 - CNN introduction
- [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- [TensorFlow: Convolutional layers](https://www.tensorflow.org/tutorials/images/cnn)
- [GLOSSARY.md](../../GLOSSARY.md) - [deep-learning](../../GLOSSARY.md#deep-learning), [neural-network](../../GLOSSARY.md#neural-network)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)
- [Chapter 09 - Neural Networks](../chapter-09-neural-networks/README.md)
- [Section 3.8 - MNIST](../chapter-03-classification-models/section-08-case-study-mnist-digit-recognition.md)

---

**Previous:** [Chapter 09](../chapter-09-neural-networks/README.md) | **Next:** [Section 10.2](./section-02-convolution-and-pooling.md)
