# Section 11.7: Open-Set Classification

> **Source:** Prosise, Ch. 11 - closed-set vs open-set, thresholds, unknown face rejection  
> **Prerequisites:** [Section 11.6](./section-06-building-a-face-recognition-system.md) | [Section 11.5](./section-05-face-embeddings-and-arcface.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [overfitting](../../GLOSSARY.md#overfitting) | [hyperparameter](../../GLOSSARY.md#hyperparameter)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Problem: "It Doesn't Know What It Doesn't Know"

Prosise's sobering demo: pass a photo of **yourself** to the VGGFace softmax model trained on Jeff, Lori, and Abby. It likely labels you as one of them - often with **high confidence**. The network is a **closed-set classifier**: softmax forces every input into one of $K$ trained classes.

$$
\sum_{k=1}^{K} p_k = 1.0 \quad \Rightarrow \quad p_k > 0 \text{ for all } k
$$
> **Readable form:** softmax probabilities always sum to 100% - there is no explicit "none of the above" bucket

Access control, dorm entry, and driver verification need **open-set** behavior: accept known identities, **reject unknowns**.

> **Humorous analogy:** Closed-set is a multiple-choice exam with no "skip" option - you bubble in a random letter anyway. Open-set allows "I don't know this student."

> **In plain English:** Tune thresholds so low-confidence or low-similarity probes return UNKNOWN instead of a forced name.

---

## Closed-Set vs Open-Set

| Property | Closed-set | Open-set |
|----------|------------|----------|
| Training classes | Fixed $K$ | Fixed $K$ + implicit "other" |
| Probe from new person | Misclassified as known | Rejected as unknown |
| Output | Argmax class + prob | Accept/reject + optional ID |
| Typical loss | Cross-entropy | CE + margin, or embedding distance |
| Real-world fit | Lab benchmarks | Production security |

Figure 11-8 in Prosise illustrates the conceptual split - engineering open-set behavior is still an active research area.

---

## Naive Fix: Prediction Threshold on Softmax

Prosise's first line of defense in `label_faces`:

```python
predictions = model.predict(np.expand_dims(chip, axis=0), verbose=0)
confidence = float(np.max(predictions))

if confidence >= prediction_threshold:  # default 0.9
    name = names[int(np.argmax(predictions))]
else:
    name = 'UNKNOWN'
```

| Threshold | Effect |
|-----------|--------|
| Lower (0.7) | More known accepts; more false accepts on strangers |
| Higher (0.99) | Fewer false accepts; more false rejects on enrolled people |
| Default 0.9 | Prosise's starting point on tiny dataset |

**Limitation:** softmax can be **overconfident** on out-of-distribution faces - high max prob even when wrong. Narrow head (8 neurons) and dropout help but aren't perfect.

---

## Better Fix: ArcFace Verification Loop

After argmax identification, **verify** with embedding similarity to enrolled reference:

```python
def identify_open_set(chip, model, names, gallery_embedder, gallery_refs,
                      softmax_thresh=0.9, cosine_thresh=0.45):
    preds = model.predict(np.expand_dims(chip, axis=0), verbose=0)
    conf = float(np.max(preds))
    idx = int(np.argmax(preds))
    candidate = names[idx]

    if conf < softmax_thresh:
        return 'UNKNOWN', conf

    probe_emb = gallery_embedder.calc_emb(chip_for_arcface(chip))
    ref_emb = gallery_refs[candidate]
    sim = cosine_similarity([probe_emb], [ref_emb])[0, 0]

    if sim >= cosine_thresh:
        return candidate, sim
    return 'UNKNOWN', sim
```

Two gates reduce false accepts - softmax AND cosine must pass.

---

## Embedding-Only Open-Set (Recommended)

Skip softmax entirely for production galleries:

```python
def match_gallery(probe_emb, gallery, threshold=0.45):
    best_name, best_sim = None, -1.0
    for name, centroid in gallery.items():
        sim = cosine_similarity([probe_emb], [centroid])[0, 0]
        if sim > best_sim:
            best_name, best_sim = name, sim
    if best_sim >= threshold:
        return best_name, best_sim
    return 'UNKNOWN', best_sim
```

**No forced partition of probability mass** - low similarity naturally means unknown.

---

## TAR, FAR, and Threshold Tuning

Security systems quote **True Accept Rate (TAR)** at fixed **False Accept Rate (FAR)**:

| Metric | Definition |
|--------|------------|
| **TAR** | Fraction of genuine probes accepted |
| **FAR** | Fraction of impostor probes incorrectly accepted |
| **FRR** | 1 − TAR (false reject rate) |

Sweep threshold on a **validation set** with both enrolled and stranger photos:

```python
import pandas as pd

def threshold_sweep(genuine_sims, impostor_sims, thresholds):
    rows = []
    for t in thresholds:
        tar = np.mean(genuine_sims >= t)
        far = np.mean(impostor_sims >= t)
        rows.append({'threshold': t, 'TAR': tar, 'FAR': far})
    return pd.DataFrame(rows)

# genuine_sims: cosine sims between probe and correct centroid
# impostor_sims: sims between probe and wrong person / strangers
df = threshold_sweep(genuine, impostor, np.arange(0.30, 0.70, 0.02))
```

Plot TAR vs FAR - pick threshold where FAR ≤ target (e.g., 0.1% for dorm access).

---

## Research Approaches (Awareness)

Prosise cites advanced open-set methods you may encounter in literature:

| Method | Idea |
|--------|------|
| **OpenMax** (2016) | Extra "unknown" logit via Weibull fit on activations |
| **Entropic Open-Set Loss** (2018) | Push unknown class toward uniform softmax |
| **Out-of-distribution detection** | Monitor activation statistics |

Engineers often start with **embedding threshold + calibration set** before bespoke loss functions.

---

## Anti-Overfitting for Open-Set

On small galleries, overfitting increases false accepts:

```python
# Prosise mitigations
Dense(8, activation='relu')   # not 1024
# optional:
Dropout(0.3)
EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
```

More enrollment photos per person beats a wider classifier head.

---

## Evaluation Protocol

1. **Gallery:** 3-5 photos/person → centroid embeddings  
2. **Genuine probes:** held-out photos of enrolled people  
3. **Impostor probes:** photos of people NOT in gallery  
4. **Sweep threshold** - plot TAR/FAR curve  
5. **Choose operating point** for product requirement (security vs convenience)

```python
# Example counts at threshold 0.45
# Genuine: 24 probes, 22 accepted → TAR 91.7%
# Impostor: 30 strangers, 2 false accepts → FAR 6.7%  ← too high for security
```

Document chosen threshold in deployment metadata - retune when gallery changes.

---

## Closed-Set Accuracy Is Misleading

| Reported metric | What it hides |
|-----------------|---------------|
| 100% test accuracy on 3 classes | Zero strangers in test set |
| High softmax on LFW celebrities | Overlap with VGGFace pretraining |
| Confusion matrix on enrolled only | No UNKNOWN row |

Always report **impostor rejection rate** alongside enrolled accuracy.

---

## Integration with `label_faces`

```python
def label_faces_open(path, gallery, embedder, sim_threshold=0.45, **kwargs):
    # ... detect faces ...
    for face in faces:
        chip = get_face(np_image, face)
        emb = embedder.calc_emb(preprocess_chip(chip))
        name, sim = match_gallery(emb, gallery, sim_threshold)
        label = name if name != 'UNKNOWN' else f'Unknown ({sim:.2f})'
        # draw box + label
```

---

## Self-Check

1. Why does softmax sum-to-one prevent native open-set behavior?
2. What trade-off does raising `prediction_threshold` create?
3. Define TAR and FAR in your own words.
4. Why verify ArcFace similarity after softmax argmax?
5. Why is closed-set test accuracy insufficient for dorm access control?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 11 - open-set discussion
- [Bendale & Boult, OpenMax, 2016](https://arxiv.org/abs/1511.06233)
- [Shu et al., Entropic Open-Set Loss, 2018](https://arxiv.org/abs/1804.07643)
- [Danka, "Does a Neural Network Know When It Doesn't Know?"](https://towardsdatascience.com/does-a-neural-network-know-when-it-doesnt-know-6a7476483f68)
- [Section 11.5](./section-05-face-embeddings-and-arcface.md) | [Section 11.8](./section-08-ethics-and-deployment.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 11.6](./section-06-building-a-face-recognition-system.md) | **Next:** [Section 11.8](./section-08-ethics-and-deployment.md)



