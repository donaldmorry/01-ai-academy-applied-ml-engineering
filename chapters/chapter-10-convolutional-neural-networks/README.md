# Chapter 10: Convolutional Neural Networks

> **Source:** *Applied Machine Learning and AI for Engineers*, Chapter 10  
> **Part:** II — Deep Learning with Keras & TensorFlow  
> **Estimated time:** 12–14 hours  
> **Prerequisites:** [Chapter 09 — Neural Networks](../chapter-09-neural-networks/README.md); image data basics (height, width, channels)

---

## Chapter Overview

Flattening images into vectors (as in Chapter 09) throws away spatial structure. **Convolutional Neural Networks (CNNs)** preserve it — learning local patterns like edges, textures, and shapes through convolutional filters, then combining them into hierarchical representations that outperform any hand-crafted feature on visual tasks.

This chapter covers CNN architecture (conv, pool, flatten, dense), trains a custom classifier on **Arctic wildlife** images, applies **transfer learning** with **ResNet50V2** pretrained on ImageNet, uses **data augmentation** to combat overfitting on small datasets, and extends CNN concepts to **audio spectrograms** — showing that convolutions are not limited to photographs.

CNNs are the backbone of modern computer vision. Everything in Chapters 11 (faces) and 12 (object detection) builds on the foundations here.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain convolution, pooling, and feature maps with spatial intuition
2. Build a CNN from scratch with Keras Conv2D, MaxPooling2D, and Flatten layers
3. Train an image classifier on a custom dataset (Arctic wildlife)
4. Apply transfer learning with ResNet50V2 and other ImageNet pretrained models
5. Use data augmentation (rotation, flip, zoom) to improve generalization
6. Fine-tune pretrained models with frozen and unfrozen base layers
7. Visualize learned filters and intermediate feature maps
8. Apply CNN principles to audio classification via spectrogram representations

---

## Sections

| # | Section | File | Topics |
|---|--------|------|--------|
| 10.1 | Why CNNs for Images? | [01-why-cnns-for-images.md](./section-01-why-cnns-for-images.md) | Spatial locality; parameter sharing; translation invariance |
| 10.2 | Convolution & Pooling | [02-convolution-and-pooling.md](./section-02-convolution-and-pooling.md) | Filters, stride, padding; max/average pooling; output shape arithmetic |
| 10.3 | Building a CNN in Keras | [03-building-a-cnn-in-keras.md](./section-03-building-a-cnn-in-keras.md) | Conv2D → Pool → Conv2D → Pool → Flatten → Dense; channel progression |
| 10.4 | Arctic Wildlife Classifier | [04-arctic-wildlife-classifier.md](./section-04-arctic-wildlife-classifier.md) | Custom dataset loading; `ImageDataGenerator`; training from scratch |
| 10.5 | Transfer Learning with ResNet50V2 | [05-transfer-learning-resnet50v2.md](./section-05-transfer-learning-with-resnet50v2.md) | ImageNet weights; feature extraction; cut-off base model |
| 10.6 | Data Augmentation | [06-data-augmentation.md](./section-06-data-augmentation.md) | Rotation, shift, flip, zoom; `ImageDataGenerator` and Keras preprocessing layers |
| 10.7 | Fine-Tuning Pretrained Models | [07-fine-tuning-pretrained-models.md](./section-07-fine-tuning-pretrained-models.md) | Unfreezing layers; differential learning rates; avoiding catastrophic forgetting |
| 10.8 | CNNs for Audio | [08-cnns-for-audio.md](./section-08-cnns-for-audio-spectrograms.md) | Spectrograms as images; applying Conv2D to sound classification |

---

## Lab

**[Lab 10: Arctic Wildlife & Transfer Learning](./section-lab-10-arctic-wildlife-and-transfer-learning.md)**

Build two image classification systems:

1. **From Scratch** — Train a custom CNN on Arctic wildlife images (polar bear, penguin, walrus, etc.). Use data augmentation. Achieve >80% validation accuracy.
2. **Transfer Learning** — Load ResNet50V2 with ImageNet weights. Add a custom classification head. Fine-tune and beat the from-scratch model with fewer epochs.
3. **(Stretch)** Convert a short audio clip to a spectrogram and classify with a small CNN.

Include: training curves, confusion matrix, misclassified image gallery, and comparison of training time and accuracy between scratch and transfer approaches.

*Deliverable:* Notebook with both models saved, side-by-side metrics, and written analysis of when transfer learning is worth the added complexity.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Convolution operations | Mathematical foundations, receptive fields (Course 3, Ch 9) |
| Transfer learning | Domain adaptation, few-shot learning (Course 3) |
| Data augmentation | Generative augmentation, GANs (Course 3) |
| Arctic wildlife CNN | Object detection, segmentation (Chapter 12) |
| Audio spectrograms | Speech recognition, signal processing (Chapter 13–14) |

---

## Prerequisites

- [Chapter 09 — Neural Networks](../chapter-09-neural-networks/README.md) (Keras training loop, callbacks, regularization)
- Understanding of image tensors (samples, height, width, channels)
- Sufficient disk space for image datasets and pretrained weights (~100 MB+)
- GPU strongly recommended for training; CPU workable for transfer learning with frozen base

---

## Key Takeaways

- CNNs learn spatial hierarchies: edges → textures → parts → objects
- Data augmentation is essential when training data is limited
- Transfer learning leverages millions of ImageNet images — start here for most real projects
- Freeze pretrained layers initially; unfreeze selectively for fine-tuning
- The same convolution principle applies to any grid-structured data, including spectrograms

---

## Self-Assessment

1. Why is parameter sharing in convolutions important compared to fully connected layers on images?
2. How do you calculate the output spatial dimensions after a Conv2D layer with given kernel, stride, and padding?
3. When should you train from scratch vs use transfer learning?
4. What is the risk of unfreezing too many pretrained layers too early?
5. Which augmentations make sense for wildlife photos? Which would be harmful?
6. Why does ResNet50V2 work with 1000 ImageNet classes but classify only 4 wildlife species in our head?
7. How is a spectrogram analogous to an image for CNN purposes?

---

**Glossary:** [../../GLOSSARY.md](../../GLOSSARY.md) — [deep-learning](../../GLOSSARY.md#deep-learning), [neural-network](../../GLOSSARY.md#neural-network), [classification](../../GLOSSARY.md#classification), [overfitting](../../GLOSSARY.md#overfitting)  
**Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

**Previous:** [Chapter 09 — Neural Networks](../chapter-09-neural-networks/README.md)  
**Next:** [Chapter 11 — Face Detection & Recognition](../chapter-11-face-detection-recognition/README.md)
