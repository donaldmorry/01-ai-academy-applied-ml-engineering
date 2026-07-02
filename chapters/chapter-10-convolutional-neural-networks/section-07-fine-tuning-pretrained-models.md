# Section 10.7: Fine-Tuning Pretrained Models

> **Source:** Prosise, Ch. 10 - unfreezing layers, differential learning rates, catastrophic forgetting  
> **Prerequisites:** [Section 10.5](./section-05-transfer-learning-with-resnet50v2.md) | [Section 10.6](./section-06-data-augmentation.md)  
> **Glossary:** [deep-learning](../../GLOSSARY.md#deep-learning) | [overfitting](../../GLOSSARY.md#overfitting) | [hyperparameter](../../GLOSSARY.md#hyperparameter)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Beyond Feature Extraction

[Section 10.5](./section-05-transfer-learning-with-resnet50v2.md) froze ResNet50V2 and trained only the 4-class head. That works well when data is scarce. **Fine-tuning** unfreezes some backbone layers so low-level and mid-level filters adapt to Arctic ice tones, fur textures, and tusk shapes - often gaining another 2-5% validation accuracy.

> **Humorous analogy:** Feature extraction is hiring a seasoned photographer and only teaching them your four animal names. Fine-tuning lets them adjust their lens settings for Arctic glare - but if you retrain them from scratch on four photos, they forget how cameras work.

> **In plain English:** After the new head converges, unfreeze the top few ResNet blocks, use a tiny learning rate, and train briefly with continued augmentation.

---

## The Two-Phase Recipe

| Phase | Base trainable? | LR | Epochs |
|-------|-----------------|-----|--------|
| 1 - Feature extraction | No | 1e-3 | Until val loss plateaus |
| 2 - Fine-tuning | Top N blocks yes | 1e-5 - 1e-4 | 5-15 |

**Never** skip phase 1 - a random head produces huge gradients that destroy pretrained weights if the base is unfrozen too early.

---

## Unfreeze Top Layers

ResNet50V2 has ~190 layers. Prosise unfreezes only the **last 20-40 layers**:

```python
base_model.trainable = True

# Freeze all but last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

trainable_count = sum(tf.keras.backend.count_params(w) for w in base_model.trainable_weights)
print(f'Trainable params: {trainable_count:,}')
```

Verify:

```python
assert not base_model.layers[0].trainable
assert base_model.layers[-1].trainable
```

---

## Recompile with Lower Learning Rate

After changing `trainable`, you **must** recompile:

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
```

Large LR on unfrozen BatchNorm layers causes **catastrophic forgetting** - ImageNet features wiped, val accuracy collapses.

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla \mathcal{L}, \quad \eta \ll 10^{-3}
$$
> **Readable form:** weight update = old weights minus (small learning rate × gradient)

---

## Differential Learning Rates (Advanced)

Lower layers change slower than the head:

```python
import tensorflow.keras.backend as K

# Conceptual pattern - separate optimizers or custom train_step
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
# Head could use 1e-4 via model splitting - optional for Prosise lab
```

TensorFlow `tfa` optimizers or manual layer-wise LR schedules exist; Prosise's text uses a single small LR for simplicity.

---

## Fine-Tune Training Loop

```python
fine_tune_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2, min_lr=1e-7),
]

history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=fine_tune_callbacks,
    # initial_epoch=history.epoch[-1] + 1,  # optional continuity
)
```

Compare `history` (phase 1) vs `history_ft` (phase 2) val_accuracy.

---

## BatchNorm During Fine-Tuning

BatchNorm layers behave differently in train vs inference. When base is frozen, use `training=False`. When fine-tuning:

```python
# Keras fit() sets training=True - BatchNorm updates running stats
# For very small batches, consider freezing BN layers:
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
```

Freezing BN while unfreezing conv weights is a common trick on tiny datasets.

---

## Catastrophic Forgetting: Symptoms & Fixes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Val acc drops after unfreeze | LR too high | 1e-5, fewer layers |
| Train acc high, val random | Overfitting unfrozen layers | More augmentation, fewer layers |
| Loss spikes epoch 1 | Head not pretrained | Complete phase 1 first |
| OOM | Too many trainable params | Unfreeze fewer layers, smaller batch |

---

## How Many Layers to Unfreeze?

| Dataset size | Suggestion |
|--------------|------------|
| <500 images | Head only or last 10 layers |
| 500-2k | Last 20-30 layers |
| >2k | Last 50+ or full model with tiny LR |

Arctic wildlife (~800-1200 total) → last **20-30** layers is Prosise's sweet spot.

---

## Full Pipeline Code Sketch

```python
# === Phase 1 ===
base_model.trainable = False
model = build_resnet_model(base_model, num_classes=4)
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=[early_stop])

# === Phase 2 ===
base_model.trainable = True
for layer in base_model.layers[:-25]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=[early_stop])

model.save('models/arctic_resnet50v2_finetuned.keras')
```

---

## Metrics to Report in Lab

| Model | Val accuracy | Train time | Trainable params |
|-------|--------------|------------|------------------|
| Scratch CNN + aug | 84% | 45 min | 3M |
| ResNet frozen + aug | 92% | 8 min | 8k |
| ResNet fine-tuned | **95%** | +5 min | 2.5M |

Fine-tuning should justify its complexity with measurable gains.

---

## When to Stop at Frozen Base

- Val accuracy already >95%
- Dataset <300 images total
- Deployment needs smallest artifact
- Fine-tuning unstable across seeds

Production teams often ship frozen-base models first, fine-tune offline if metrics demand it.

---

## Saving Checkpoints Between Phases

```python
model.save('checkpoints/after_phase1.keras')
# ... unfreeze ...
model.save('checkpoints/after_phase2.keras')
```

Rollback if phase 2 degrades performance.

---

## Link to Chapter 07 Deployment

Fine-tuned ResNet is ~100 MB - acceptable for server deployment. For mobile, distill to MobileNet ([Chapter 07](../chapter-07-operationalizing-models/README.md)). Document preprocessing (`preprocess_input`) in model metadata.

---

## Self-Check

1. Why train the head before unfreezing the base?
2. What learning rate range is typical for fine-tuning?
3. What is catastrophic forgetting?
4. Should BatchNorm always remain trainable?
5. How do you choose how many layers to unfreeze?

---

## References

- Prosise, *Applied ML and AI for Engineers*, Ch. 10 - fine-tuning
- [TensorFlow transfer learning: fine-tune](https://www.tensorflow.org/tutorials/images/transfer_learning#fine_tuning)
- [Keras trainable property](https://keras.io/api/models/model_training_apis/#trainable-property)
- [Section 10.5](./section-05-transfer-learning-with-resnet50v2.md)
- [Section 10.6](./section-06-data-augmentation.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 10.6](./section-06-data-augmentation.md) | **Next:** [Section 10.8](./section-08-cnns-for-audio-spectrograms.md)



