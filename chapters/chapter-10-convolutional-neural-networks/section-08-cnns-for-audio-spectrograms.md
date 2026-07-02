# Section 10.8: CNNs for Audio - Spectrograms

> **Source:** Prosise, Ch. 10 - applying Conv2D to time-frequency representations  
> **Prerequisites:** [Section 10.3](./section-03-building-a-cnn-in-keras.md) | [Section 10.2](./section-02-convolution-and-pooling.md)  
> **Glossary:** [deep-learning](../../GLOSSARY.md#deep-learning) | [feature](../../GLOSSARY.md#feature) | [classification](../../GLOSSARY.md#classification)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Grids Are Not Just Photographs

Convolutional layers operate on **2D grids**. A photograph is a spatial grid (height × width). A **spectrogram** is a time-frequency grid - time on one axis, frequency on the other, intensity as pixel brightness. To a CNN, a spectrogram looks like a grayscale image: local patterns (harmonics, transients) become learnable [features](../../GLOSSARY.md#feature).

> **Humorous analogy:** A spectrogram is sheet music written by a machine that only paints in grayscale - birds chirp in high frequencies, walrus bellows in low rumble. The CNN doesn't care that it's sound; it hunts blobs and stripes.

> **In plain English:** Convert audio clips to spectrogram images, then reuse the same Conv2D stack from Arctic wildlife training.

Prosise closes Chapter 10 with this extension to show CNNs generalize beyond RGB photos - linking forward to [Chapter 13](../chapter-13-natural-language-processing/README.md) and speech systems.

---

## From Waveform to Spectrogram

Raw audio is a 1D amplitude signal $x(t)$ sampled at rate $f_s$ (e.g., 22,050 Hz).

**Short-Time Fourier Transform (STFT)** windows the signal and computes frequency content per window:

$$
X(m, k) = \sum_{n=0}^{N-1} x[n + mH]\, w[n]\, e^{-j 2\pi kn / N}
$$
> **Readable form:** each spectrogram cell = complex Fourier coefficient for time window m and frequency bin k

The **magnitude spectrogram** (often log-scaled) becomes the 2D input:

$$
S(m, k) = \log\left(1 + |X(m, k)|\right)
$$
> **Readable form:** spectrogram value = log of (1 + magnitude of frequency content)

---

## Mel Spectrogram (Human Hearing Scale)

Humans perceive pitch logarithmically. **Mel filterbanks** warp frequency axis:

```python
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

AUDIO_PATH = 'audio/walrus_call.wav'
SAMPLE_RATE = 22050
DURATION_SEC = 3.0

y, sr = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE, duration=DURATION_SEC)
print(f'Samples: {len(y)}, duration: {len(y)/sr:.2f}s')

mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
mel_db = librosa.power_to_db(mel_spec, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram - Walrus Vocalization')
plt.tight_layout()
plt.show()
```

Shape: `(n_mels, time_frames)` - e.g., `(128, 130)` - treat as `(height, width, 1)` for Conv2D.

---

## Audio CNN Pipeline Overview

```
.wav file → load waveform → mel spectrogram → resize/normalize → Conv2D CNN → class
```

| Step | Output shape (example) |
|------|------------------------|
| Waveform | (66150,) for 3s @ 22kHz |
| Mel spec | (128, 130) |
| Resize | (128, 128, 1) |
| CNN | 4 classes: bird, wind, engine, animal_call |

---

## Build Dataset from Labeled Clips

Folder structure mirrors image [classification](../../GLOSSARY.md#classification):

```
audio_wildlife/
├── train/
│   ├── bird/
│   ├── engine/
│   ├── wind/
│   └── walrus/
└── val/
    └── ...
```

Convert each `.wav` to spectrogram on the fly or cache as `.npy` / `.png`:

```python
def wav_to_mel_fixed(path, n_mels=128, target_frames=128, sr=22050):
    y, _ = librosa.load(path, sr=sr, duration=3.0)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Pad or crop time axis to fixed width
    if mel_db.shape[1] >= target_frames:
        mel_db = mel_db[:, :target_frames]
    else:
        pad = target_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='constant')
    # Normalize to [0, 1]
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_norm[..., np.newaxis]  # (128, 128, 1)
```

---

## Custom tf.data Generator

```python
import tensorflow as tf
from pathlib import Path

def make_audio_dataset(root_dir, batch_size=32):
    paths, labels = [], []
    class_names = sorted([d.name for d in Path(root_dir).iterdir() if d.is_dir()])
    label_map = {n: i for i, n in enumerate(class_names)}

    for cls in class_names:
        for wav in (Path(root_dir) / cls).glob('*.wav'):
            paths.append(str(wav))
            labels.append(label_map[cls])

    def load(path, label):
        spec = tf.numpy_function(
            lambda p: wav_to_mel_fixed(p.decode('utf-8')),
            [path], tf.float32
        )
        spec.set_shape((128, 128, 1))
        return spec, tf.one_hot(label, len(class_names))

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(1000).map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, class_names
```

---

## Small Audio CNN Architecture

Reuse [Section 10.3](./section-03-building-a-cnn-in-keras.md) pattern with single-channel input:

```python
def build_audio_cnn(num_classes=4):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 1)),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])

audio_model = build_audio_cnn()
audio_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

No ImageNet transfer - spectrograms are not natural RGB images. Train from scratch or use audio-pretrained models (YAMNet) in advanced projects.

---

## Audio-Specific Augmentation

| Transform | Implementation | Preserves label? |
|-----------|----------------|------------------|
| Time shift | roll waveform | Usually yes |
| Pitch shift | `librosa.effects.pitch_shift` | Careful - species dependent |
| Add noise | Gaussian noise on waveform | Yes |
| Time stretch | `librosa.effects.time_stretch` | Usually yes |
| Spec augment (mask bands) | zero time/freq stripes | Common in speech |

```python
def augment_waveform(y, sr):
    if np.random.rand() < 0.5:
        y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.9, 1.1))
    if np.random.rand() < 0.3:
        noise = np.random.randn(len(y)) * 0.005
        y = y + noise
    return y
```

Apply **before** STFT - augmenting spectrograms directly also works (SpecAugment).

---

## Why Conv2D on Spectrograms Works

| Pattern in spectrogram | CNN filter detects |
|------------------------|-------------------|
| Horizontal stripes | Steady tones |
| Vertical transients | Percussive events |
| Harmonic stacks | Engine or animal calls |
| Broadband noise | Wind |

Same translation equivariance as images - a chirp earlier in time shifts feature map horizontally.

---

## Comparison: 1D CNN vs 2D on Spectrogram

| Approach | Input | Pros |
|----------|-------|------|
| 1D Conv on waveform | Raw samples | End-to-end, long receptive field |
| 2D Conv on spectrogram | Mel image | Reuse image CNN skills, fewer samples |

Prosise chooses 2D for pedagogical continuity with Chapter 10.

---

## Evaluation

```python
from sklearn.metrics import classification_report

y_true, y_pred = [], []
for batch_x, batch_y in val_ds:
    preds = audio_model.predict(batch_x, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(batch_y, axis=1))

print(classification_report(y_true, y_pred, target_names=class_names))
```

Confusion between **wind** and **engine** noise is common - similar broadband energy.

---

## Stretch Goal for Lab 10

Classify one Arctic animal vocalization clip among ambient classes. Document:

1. Waveform plot
2. Mel spectrogram image
3. CNN confusion matrix
4. Comparison to image CNN training time

---

## Dependencies

```bash
pip install librosa soundfile
```

`librosa` uses `soundfile` / `audioread` backends for `.wav` loading.

---

## Connections Forward

| Topic | Chapter |
|-------|--------|
| Speech recognition | Chapter 13 NLP |
| Azure Speech | Chapter 14 |
| Raw waveform models | Course 3 deep learning |

---

## Self-Check

1. Why is a mel spectrogram a 2D grid suitable for Conv2D?
2. What axis represents time in the spectrogram plot?
3. Why doesn't ImageNet transfer apply directly to mel spectrograms?
4. Name two audio augmentations applied before STFT.
5. What shape tensor enters `build_audio_cnn`?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 10 - audio CNN extension
- [Librosa documentation](https://librosa.org/doc/latest/index.html)
- [SpecAugment (Park et al.)](https://arxiv.org/abs/1904.08779)
- [CS231n: CNNs beyond images](https://cs231n.stanford.edu/)
- [Section 10.3](./section-03-building-a-cnn-in-keras.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 10.7](./section-07-fine-tuning-pretrained-models.md) | **Next:** [Lab 10](./section-lab-10-arctic-wildlife-and-transfer-learning.md)
