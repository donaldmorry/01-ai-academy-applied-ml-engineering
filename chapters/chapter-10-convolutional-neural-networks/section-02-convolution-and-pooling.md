# Section 10.2: Convolution & Pooling

> **Source:** Prosise, Ch. 10 - filters, stride, padding, pooling layers  
> **Prerequisites:** [Section 10.1](./section-01-why-cnns-for-images.md)  
> **Glossary:** [deep-learning](../../GLOSSARY.md#deep-learning) | [neural-network](../../GLOSSARY.md#neural-network)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Convolution as Sliding Dot Product

A **convolutional filter** (kernel) is a small weight matrix that slides across the image. At each position, compute the dot product between the filter and the underlying patch, then add a bias.

For one output channel at position $(i, j)$:

$$
y_{i,j} = \sum_{c=1}^{C}\sum_{u=0}^{k_h-1}\sum_{v=0}^{k_w-1} w_{c,u,v}\, x_{c,\, i+u,\, j+v} + b
$$
> **Readable form:** output at (i,j) = sum over channels and kernel positions of (weight × input pixel) + bias

- $C$ - input channels (1 for grayscale, 3 for RGB)
- $k_h, k_w$ - kernel height and width (often 3×3 or 5×5)
- $w$ - learnable filter weights
- $b$ - learnable bias (one per output filter)

> **Humorous analogy:** A convolution filter is a cookie cutter you stamp across the dough. Every stamp uses the same cutter (parameter sharing); you get a sheet of cookies (feature map) describing where that shape matched.

> **In plain English:** Each filter hunts one pattern - horizontal edge, bluish blob, fur texture - and writes a heatmap of where it fired.

---

## Output Shape Arithmetic

The spatial size of a convolution output is determined by input size, kernel size, **stride**, and **padding**.

$$
H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} - k_h + 2p}{s} \right\rfloor + 1
$$

$$
W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - k_w + 2p}{s} \right\rfloor + 1
$$
> **Readable form:** output height = floor((input height − kernel height + 2×padding) / stride) + 1 (same formula for width)

| Symbol | Meaning | Typical value |
|--------|---------|---------------|
| $H_{\text{in}}, W_{\text{in}}$ | Input spatial dimensions | 224×224 |
| $k_h, k_w$ | Kernel size | 3×3 |
| $s$ | Stride - step size of the slide | 1 or 2 |
| $p$ | Padding - zeros around border | 0 or "same" |

**Example:** $H_{\text{in}}=32$, $k=3$, $s=1$, $p=0$ (valid convolution):

$$
H_{\text{out}} = \frac{32 - 3 + 0}{1} + 1 = 30
$$
> **Readable form:** The operation slides a local window over the input and computes a feature value for each output location.

**"Same" padding** in Keras pads so output spatial size equals input (when stride=1):

```python
from tensorflow.keras.layers import Conv2D

# 32×32 input → 32×32 output (stride 1, same padding)
layer = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
```

---

## Multiple Filters → Feature Maps

`Conv2D(filters=64, ...)` learns **64 different filters**, producing 64 output channels stacked into a 3D tensor:

$$
\text{output shape} = (H_{\text{out}},\; W_{\text{out}},\; 64)
$$
> **Readable form:** output = height × width × number of filters

Each filter specializes during training via backpropagation - same mechanism as Dense layers in [Chapter 09](../chapter-09-neural-networks/README.md).

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu', name='conv1'),
    layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2'),
])
model.summary()
# conv1: (None, 64, 64, 32)  - 32 feature maps
# conv2: (None, 64, 64, 64)  - 64 feature maps
```

**Parameter count** for one Conv2D layer:

$$
\text{params} = k_h \cdot k_w \cdot C_{\text{in}} \cdot C_{\text{out}} + C_{\text{out}}
$$
> **Readable form:** weights = kernel height × kernel width × input channels × output filters + one bias per filter

For `Conv2D(64, 3)` on 32-channel input: $3 \times 3 \times 32 \times 64 + 64 = 18{,}496$ parameters.

---

## Stride: Downsampling Without Pooling

Stride $s=2$ moves the filter two pixels at a time, halving spatial dimensions (with appropriate padding):

| Input | Kernel | Stride | Padding | Output |
|-------|--------|--------|---------|--------|
| 32×32 | 3×3 | 1 | same | 32×32 |
| 32×32 | 3×3 | 2 | same | 16×16 |
| 224×224 | 7×7 | 2 | valid | 109×109 |

ResNet's first layers use strided convolutions for efficiency. Prosise's smaller Arctic CNN typically uses stride 1 convs + pooling blocks.

---

## Activation After Convolution

Raw convolution outputs are linear combinations. Apply a nonlinearity - almost always **ReLU**:

$$
\text{ReLU}(z) = \max(0, z)
$$
> **Readable form:** ReLU = keep positive values, zero out negatives

ReLU sparsifies feature maps (many zeros) and speeds training. Keras `activation='relu'` on `Conv2D` fuses conv + activation.

---

## Pooling: Spatial Summarization

**Pooling** reduces height and width while keeping depth (number of channels). Goals:

1. Shrink computation for deeper layers
2. Add local translation invariance
3. Widen effective receptive field

### Max Pooling

Take the maximum value in each $p \times p$ window (default $p=2$):

$$
y_{i,j} = \max_{0 \le u,v < p} x_{i \cdot s + u,\; j \cdot s + v}
$$
> **Readable form:** pooled value = largest number in each pooling window

```python
from tensorflow.keras.layers import MaxPooling2D

# 64×64×32 → 32×32×32 (2×2 pool, stride 2)
pool = MaxPooling2D(pool_size=2, strides=2)
```

| Input | Pool 2×2, stride 2 | Channels |
|-------|-------------------|----------|
| 64×64×32 | 32×32×32 | unchanged |

### Average Pooling

$$
y_{i,j} = \frac{1}{p^2}\sum_{u,v} x_{i \cdot s + u,\; j \cdot s + v}
$$
> **Readable form:** pooled value = average of pixels in the window

`AvgPool2D` smooths more than max pool. ImageNet classifiers often end with **global average pooling** - average each entire feature map to one number per channel.

| Pool type | Prosise / Keras usage |
|-----------|----------------------|
| MaxPool 2×2 | Default in custom Arctic CNN |
| GlobalAvgPool | ResNet head before Dense ([Section 10.5](./section-05-transfer-learning-with-resnet50v2.md)) |
| AvgPool | Some lightweight architectures |

---

## Classic Conv-Pool Block

Prosise's teaching pattern repeats:

```
Conv2D → ReLU → Conv2D → ReLU → MaxPool2D
```

Each block **increases channels, decreases spatial size**:

| Stage | Spatial | Channels | What it learns |
|-------|---------|----------|----------------|
| Input | 128×128 | 3 | Raw RGB |
| Block 1 | 64×64 | 32 | Edges, color |
| Block 2 | 32×32 | 64 | Textures |
| Block 3 | 16×16 | 128 | Parts |
| Head | 1×1 or flat | K classes | Species decision |

```python
def conv_block(filters):
    return tf.keras.Sequential([
        layers.Conv2D(filters, 3, padding='same', activation='relu'),
        layers.Conv2D(filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
    ])
```

---

## Manual Convolution (Intuition Builder)

```python
import numpy as np

def conv2d_single_channel(image, kernel, stride=1, padding=0):
    """Valid/same-style conv for teaching - single channel."""
    if padding > 0:
        image = np.pad(image, padding, mode='constant')
    kh, kw = kernel.shape
    out_h = (image.shape[0] - kh) // stride + 1
    out_w = (image.shape[1] - kw) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[i, j] = np.sum(patch * kernel)
    return out

# Sobel vertical edge detector
sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

rng = np.random.default_rng(42)
fake_img = rng.random((16, 16))
edges = conv2d_single_channel(fake_img, sobel_y)
print(f'Input {fake_img.shape} → Output {edges.shape}')
```

Keras does this in optimized C++/CUDA - but the math is identical.

---

## Padding Modes in Practice

| `padding` | Behavior | Use when |
|-----------|----------|----------|
| `'valid'` | No padding; shrinks spatial size | You want exact control |
| `'same'` | Pad so output size ≈ input size (stride 1) | Standard for stacked 3×3 convs |

Odd-size kernels (3×3) with "same" padding keep dimensions stable before pooling.

---

## Channel Progression Strategy

Prosise's heuristic for custom CNNs on small datasets:

- Start with **32** filters in block 1
- **Double** filters when spatial halves: 32 → 64 → 128
- Do **not** exceed ~512 filters on Arctic-scale data - [overfitting](../../GLOSSARY.md#overfitting) risk

Total path: $128 \to 64 \to 32 \to 16$ spatial with $3 \to 32 \to 64 \to 128$ channels is a solid teaching baseline ([Section 10.3](./section-03-building-a-cnn-in-keras.md)).

---

## Flatten: Bridge to Dense Head

After conv blocks, tensors are still 3D. `Flatten()` reshapes to a vector for classification:

$$
(16, 16, 128) \rightarrow (32{,}768,)
$$
> **Readable form:** flatten = multiply spatial dimensions × channels into one long vector

Modern architectures often replace Flatten + huge Dense with **GlobalAveragePooling2D** → small Dense - fewer parameters, less overfitting.

---

## Debugging Shape Errors

The #1 CNN beginner error: mismatched tensor ranks.

```python
# WRONG - missing channel dimension
layers.Input(shape=(64, 64))        # Keras expects (64, 64, 1) for grayscale

# RIGHT
layers.Input(shape=(64, 64, 1))
```

Trace shapes layer by layer:

```python
for layer in model.layers:
    print(layer.name, layer.output_shape)
```

Or call `model.summary()` after every architectural change.

---

## Self-Check

1. Compute output width for $W_{\text{in}}=28$, $k=5$, $s=1$, $p=0$.
2. How many parameters in `Conv2D(32, 3)` on RGB input?
3. What does MaxPooling2D(2) do to a 64×64×32 tensor?
4. Why apply ReLU after convolution?
5. When would global average pooling replace Flatten?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 10 - convolution and pooling
- [CS231n: Conv layers](https://cs231n.github.io/convolutional-networks/)
- [Keras Conv2D API](https://keras.io/api/layers/convolution_layers/convolution2d/)
- [Keras Pooling layers](https://keras.io/api/layers/pooling_layers/)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 10.1](./section-01-why-cnns-for-images.md) | **Next:** [Section 10.3](./section-03-building-a-cnn-in-keras.md)
