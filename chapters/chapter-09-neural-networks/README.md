# Chapter 09: Neural Networks with Keras & TensorFlow

> **Source:** *Applied Machine Learning and AI for Engineers*, Chapter 9  
> **Part:** II — Deep Learning with Keras & TensorFlow  
> **Estimated time:** 10–12 hours  
> **Prerequisites:** [Chapter 08 — Deep Learning Foundations](../chapter-08-deep-learning/README.md); TensorFlow 2.x and Keras installed

---

## Chapter Overview

Theory becomes practice. This chapter teaches you to build, train, and evaluate neural networks with **Keras** on top of **TensorFlow** — the same stack used in production at Google and thousands of companies worldwide.

You will revisit problems from Part I with neural networks: **taxi fare regression**, **credit card fraud classification**, and **facial recognition** — comparing deep learning results to the Scikit-Learn baselines you already know. Along the way you will learn network sizing heuristics, dropout regularization, callbacks for training control, and how to save and reload Keras models.

By the end, you will have a repeatable template for any tabular or vector-input deep learning problem — the same template extended with convolutional layers in Chapter 10.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Build sequential and functional Keras models for regression and classification
2. Compile models with appropriate loss functions, optimizers, and metrics
3. Train models with `fit()`, validation splits, and early stopping
4. Size hidden layers using practical heuristics for tabular data
5. Apply dropout and L2 regularization to reduce overfitting
6. Use Keras callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
7. Save, load, and export Keras models (SavedModel, HDF5)
8. Compare neural network performance to Scikit-Learn baselines on taxi, fraud, and face datasets

---

## Sections

| # | Section | File | Topics |
|---|--------|------|--------|
| 9.1 | Keras & TensorFlow Setup | [01-keras-tensorflow-setup.md](./section-01-keras-and-tensorflow-setup.md) | TensorFlow 2.x; Keras API; GPU vs CPU; eager execution |
| 9.2 | Your First Keras Model | [02-your-first-keras-model.md](./section-02-your-first-keras-model.md) | Sequential API; Dense layers; compile and fit |
| 9.3 | Network Sizing & Architecture | [03-network-sizing-architecture.md](./section-03-network-sizing-and-architecture.md) | Hidden layer count and width; input/output shape rules |
| 9.4 | Regression with Neural Networks | [04-regression-neural-networks.md](./section-04-regression-with-neural-networks-taxi-fare-prediction.md) | Taxi fare prediction; MAE/MSE loss; comparing to Chapter 02 models |
| 9.5 | Binary Classification: Fraud Detection | [05-binary-classification-fraud.md](./section-05-binary-classification-credit-card-fraud-detection.md) | Sigmoid output; binary cross-entropy; class weights for imbalance |
| 9.6 | Multi-Class Classification: Face Recognition | [06-multiclass-face-recognition.md](./section-06-multi-class-classification-face-recognition.md) | Softmax output; sparse/categorical cross-entropy; LFW faces |
| 9.7 | Dropout & Regularization | [07-dropout-and-regularization.md](./section-07-dropout-and-regularization.md) | `Dropout`, L2; when and where to apply; validation curves |
| 9.8 | Callbacks & Training Control | [08-callbacks-and-training-control.md](./section-08-callbacks-training-control-and-model-persistence.md) | EarlyStopping, ModelCheckpoint, TensorBoard; save/load (SavedModel, HDF5) |

---

## Lab

**[Lab 09: Neural Network Triathlon](./section-lab-09-neural-network-triathlon.md)**

Train Keras models on three datasets and compare to Part I baselines:

1. **Taxi Fare (Regression)** — Dense network on engineered features. Beat or match your Chapter 02 random forest RMSE.
2. **Fraud Detection (Binary)** — Handle class imbalance with class weights. Report precision, recall, and AUC. Compare to Chapter 03 logistic regression.
3. **Face Recognition (Multi-Class)** — Classify LFW faces with a Dense network on flattened pixels. Compare to Chapter 05 SVM pipeline.

For each task: plot training/validation loss curves, document architecture choices, apply early stopping, and save the best model.

*Deliverable:* Single notebook with three trained models, comparison tables, and architecture justification for each.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Dense networks & backprop | Full optimization theory (Course 3, Ch 8) |
| Dropout & regularization | Bayesian interpretation, weight decay (Course 3) |
| Tabular neural networks | Entity embeddings, TabNet (Course 3) |
| Fraud detection NN | Autoencoders for anomaly detection (Course 3) |
| Model export | TensorFlow Serving, TFLite (Chapters 10–12) |

---

## Prerequisites

- [Chapter 08 — Deep Learning Foundations](../chapter-08-deep-learning/README.md)
- TensorFlow 2.x with Keras (`pip install tensorflow`)
- Completed Chapters 02, 03, and 05 (for baseline comparisons)
- GPU recommended but not required for tabular datasets

---

## Key Takeaways

- Keras Sequential API is sufficient for most feedforward network architectures
- Match your output activation and loss to the problem type: linear+MSE for regression, sigmoid+BCE for binary, softmax+CCE for multi-class
- Dropout during training is one of the most effective regularizers for tabular networks
- Callbacks automate training decisions — early stopping prevents wasted epochs and overfitting
- Neural networks are not always better than gradient boosting on tabular data — always compare to baselines

---

## Self-Assessment

1. What three arguments does `model.compile()` require, and how do they change for regression vs classification?
2. How do you choose the number and size of hidden layers for a tabular dataset?
3. Why does dropout help prevent overfitting, and why is it disabled during inference?
4. What does EarlyStopping monitor, and what happens to the model weights when it triggers?
5. How do class weights address imbalanced fraud detection without oversampling?
6. When might a neural network underperform a random forest on taxi fare data?
7. What is the difference between `model.save()` SavedModel format and HDF5?

---

**Previous:** [Chapter 08 — Deep Learning Foundations](../chapter-08-deep-learning/README.md)  
**Next:** [Chapter 10 — Convolutional Neural Networks](../chapter-10-convolutional-neural-networks/README.md)
