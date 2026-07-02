# Section 11.5: Face Embeddings & ArcFace

> **Source:** Prosise, Ch. 11 - ArcFace, VGGFace, face embeddings, metric learning  
> **Prerequisites:** [Section 11.4](./section-04-cnn-face-detection.md) | [Chapter 10 - Transfer Learning](../chapter-10-convolutional-neural-networks/section-05-transfer-learning-with-resnet50v2.md)  
> **Glossary:** [deep-learning](../../GLOSSARY.md#deep-learning) | [neural-network](../../GLOSSARY.md#neural-network) | [classification](../../GLOSSARY.md#classification)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## From Classification to Embeddings

[Chapter 09](../chapter-09-neural-networks/section-06-multi-class-classification-face-recognition.md) and [Chapter 10](../chapter-10-convolutional-neural-networks/section-05-transfer-learning-with-resnet50v2.md) treated face recognition as **closed-set classification**: softmax over $N$ known people, one-hot labels, cross-entropy loss. That works when every probe belongs to a training class - but production systems need **identity vectors** that compare across sessions, galleries, and new enrollments without retraining the entire softmax head.

Modern recognition replaces the final classifier with a **face embedding** - a dense vector $\mathbf{e} \in \mathbb{R}^d$ where same-person faces cluster and different-person faces separate.

> **Humorous analogy:** Softmax is a roll call with fixed names on the attendance sheet. Embeddings are fingerprint smudges - you match smudges even when someone new walks in without updating the sheet.

> **In plain English:** Map each aligned face crop to a fixed-length vector; compare vectors with cosine or Euclidean distance instead of forcing a softmax label.

---

## Why Softmax Alone Falls Short

| Limitation | Consequence |
|------------|-------------|
| Fixed class count | Cannot add a person without retraining |
| Closed-set assumption | Unknown faces forced into a known bucket |
| No natural similarity score | Probabilities sum to 1.0 - misleading for verification |
| Celebrity overlap in LFW | ImageNet/VGGFace weights may overlap training identities |

Prosise demonstrates this with **VGGFace** (ResNet50 trained on millions of faces) vs generic **ImageNet ResNet50** on the same LFW subset - task-specific weights dominate.

---

## The Embedding Pipeline

```
Aligned face (224×224×3)
        ↓
   CNN backbone (frozen or fine-tuned)
        ↓
   L2-normalized vector e ∈ R^512
        ↓
   Compare to gallery with cosine similarity
```

**Enrollment:** store $\mathbf{e}_i$ for each person (mean of several photos).  
**Probe:** compute $\mathbf{e}_q$, find nearest gallery vector, accept if similarity exceeds threshold.

---

## Cosine Similarity & Euclidean Distance

Given L2-normalized embeddings:

$$
\text{sim}(\mathbf{e}_1, \mathbf{e}_2) = \mathbf{e}_1^\top \mathbf{e}_2 = \cos\theta
$$
> **Readable form:** similarity = dot product of unit-length vectors = cosine of angle between them (1.0 = identical direction, 0 = orthogonal, -1.0 = opposite)

For unnormalized vectors, use sklearn's `cosine_similarity`. Cosine ranges from $[-1, 1]$ in general; face systems usually operate in the positive region after training, so thresholds such as 0.4-0.6 are empirical operating points, not mathematical bounds. Euclidean distance on normalized vectors relates monotonically:

$$
\|\mathbf{e}_1 - \mathbf{e}_2\|_2^2 = 2(1 - \cos\theta)
$$
> **Readable form:** squared distance between unit vectors increases as cosine similarity decreases

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compare_embeddings(e1, e2):
    e1 = e1.reshape(1, -1)
    e2 = e2.reshape(1, -1)
    return float(cosine_similarity(e1, e2)[0, 1])
```

---

## Metric Learning Intuition

**Contrastive** and **triplet** losses train embeddings directly:

- **Triplet loss:** $\max(0, \|\mathbf{e}_a - \mathbf{e}_p\|^2 - \|\mathbf{e}_a - \mathbf{e}_n\|^2 + \margin)$
- **ArcFace (Additive Angular Margin):** adds angular margin on the hypersphere before softmax during training - tighter same-person clusters, wider between-person gaps

ArcFace loss (conceptual):

$$
L = -\log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos\theta_j}}
$$
> **Readable form:** penalize the angle to the correct class by margin m on a unit sphere, then apply scaled softmax - pushes embeddings apart by identity

You rarely implement this from scratch - use pretrained ArcFace weights.

---

## ArcFace in Python

Prosise uses the `arcface` package (512-dimensional embeddings):

```python
from arcface import ArcFace
from PIL import Image
import numpy as np

af = ArcFace.ArcFace()

# PIL or numpy RGB face chip, 224×224 recommended
face_img = Image.open('enrollment/jeff_01.jpg').convert('RGB')
embedding = af.calc_emb(np.array(face_img))  # shape (512,)

print(embedding.shape, np.linalg.norm(embedding))
```

**Face verification** (1:1 - same person?):

```python
emb1 = af.calc_emb(np.array(Image.open('probe_a.jpg')))
emb2 = af.calc_emb(np.array(Image.open('probe_b.jpg')))
sim = cosine_similarity([emb1], [emb2])[0, 0]
print(f'Same person if sim > threshold (typical 0.4-0.6): {sim:.3f}')
```

---

## VGGFace vs ImageNet vs ArcFace

Prosise's LFW five-person experiment (100 images each, 80/20 split):

| Backbone | Pretraining | Typical val accuracy |
|----------|-------------|----------------------|
| Custom CNN from scratch | 400 face images | ~85% |
| ResNet50 | ImageNet | ~94% |
| VGGFace (ResNet50) | Millions of faces | ~100% on test split |
| ArcFace embeddings + threshold | Face metric learning | Strong verification; open-set friendly |

**Section:** For face tasks, **task-specific pretraining** beats generic ImageNet every time - "for a neural network, it's all about the weights."

```python
# VGGFace feature extraction head (Prosise pattern)
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Resizing, Flatten, Dense

base_model = load_model('data/vggface.h5')
base_model.trainable = False

model = Sequential([
    Resizing(224, 224),
    base_model,
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(5, activation='softmax')  # closed-set head for LFW subset
])
```

ArcFace is preferable when you need **verification**, **gallery search**, or **open-set rejection** ([Section 11.7](./section-07-open-set-classification.md)).

---

## Building a Simple Embedding Gallery

```python
from pathlib import Path

GALLERY = {}  # name -> list of embeddings

def enroll_person(name, image_paths, embedder):
    vectors = []
    for p in image_paths:
        img = np.array(Image.open(p).convert('RGB'))
        vectors.append(embedder.calc_emb(img))
    GALLERY[name] = np.mean(vectors, axis=0)  # centroid embedding

def identify(probe_path, embedder, threshold=0.45):
    probe = embedder.calc_emb(np.array(Image.open(probe_path).convert('RGB')))
    best_name, best_sim = None, -1.0
    for name, centroid in GALLERY.items():
        sim = cosine_similarity([probe], [centroid])[0, 0]
        if sim > best_sim:
            best_name, best_sim = name, sim
    if best_sim >= threshold:
        return best_name, best_sim
    return 'UNKNOWN', best_sim
```

**Centroid enrollment** (mean of 3-8 photos) is more robust than a single reference image - handles lighting and expression variation.

---

## Preprocessing Requirements

Embeddings assume **aligned, well-cropped faces** from [Section 11.3](./section-03-opencv-face-pipeline.md):

| Step | Why it matters |
|------|----------------|
| MTCNN detect + square crop | Removes background noise |
| Resize to 224×224 | Matches backbone input |
| RGB (not BGR) | Consistent color channels |
| EXIF orientation fix | Prevents sideways embeddings |

Misaligned chips collapse verification accuracy faster than backbone choice.

---

## ArcFace vs FaceNet vs InsightFace

| Library | Embedding dim | Notes |
|---------|---------------|-------|
| ArcFace (`arcface`) | 512 | Prosise's example; verification-focused |
| FaceNet | 128-512 | Classic triplet-loss embeddings |
| InsightFace | 512 | Production-grade; ONNX models available |
| `deepface` wrapper | varies | Unified API over multiple backends |

Pick one backend per project - mixing embedding spaces breaks gallery compatibility.

---

## When to Use Embeddings vs Softmax Head

| Scenario | Recommended approach |
|----------|---------------------|
| Fixed small set, batch retrain OK | Softmax on VGGFace features |
| Access control, growing roster | ArcFace gallery + threshold |
| 1:1 phone unlock | Verification on device embedding |
| 1:N airport search | Nearest-neighbor over large gallery index |

---

## Self-Check

1. Why can't softmax probabilities alone support open-set rejection?
2. What does L2 normalization do before cosine similarity?
3. Why does VGGFace outperform ImageNet ResNet50 on faces?
4. What is the difference between verification and identification in embedding space?
5. Why enroll multiple photos per person?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 11 - ArcFace, VGGFace, embeddings
- [Deng et al., ArcFace, 2019](https://arxiv.org/abs/1801.07698)
- [Cao et al., VGGFace2, 2018](https://arxiv.org/abs/1710.08092)
- [Schroff et al., FaceNet, 2015](https://arxiv.org/abs/1503.03832)
- [Section 11.4](./section-04-cnn-face-detection.md) | [Section 11.6](./section-06-building-a-face-recognition-system.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 11.4](./section-04-cnn-face-detection.md) | **Next:** [Section 11.6](./section-06-building-a-face-recognition-system.md)


