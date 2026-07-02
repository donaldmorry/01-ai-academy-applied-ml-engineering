# Chapter 11: Face Detection & Recognition

> **Source:** *Applied Machine Learning and AI for Engineers*, Chapter 11  
> **Part:** II — Deep Learning with Keras & TensorFlow  
> **Estimated time:** 8–10 hours  
> **Prerequisites:** [Chapter 10 — Convolutional Neural Networks](../chapter-10-convolutional-neural-networks/README.md); OpenCV installed

---

## Chapter Overview

Face technology is two distinct problems: **detection** (where are the faces?) and **recognition** (whose face is it?). This chapter covers the full pipeline from classical to state-of-the-art — starting with the **Viola-Jones** cascade detector in **OpenCV**, progressing to **CNN-based detection**, and finishing with **ArcFace** embeddings for high-accuracy recognition.

You will also confront **open-set classification** — the real-world scenario where the model must reject faces it has never seen, not force them into a known category. This is critical for security and access control applications where a false acceptance is far costlier than a false rejection.

Face systems appear in phones, airports, photo apps, and surveillance. Understanding both the classical and deep learning approaches makes you effective at building, evaluating, and responsibly deploying these systems.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Distinguish face detection from face recognition and verification
2. Implement face detection with OpenCV's Haar cascade (Viola-Jones)
3. Detect faces with modern CNN-based detectors (MTCNN or OpenCV DNN)
4. Extract face embeddings with ArcFace or similar metric learning models
5. Build a face recognition system using embedding distance (cosine/Euclidean)
6. Handle open-set classification with confidence thresholds and rejection
7. Align and preprocess face crops for consistent recognition input
8. Evaluate face systems with TAR at fixed FAR and recognition accuracy metrics

---

## Sections

| # | Section | File | Topics |
|---|--------|------|--------|
| 11.1 | Detection vs Recognition | [01-detection-vs-recognition.md](./section-01-detection-vs-recognition.md) | Two-stage pipeline; verification vs identification; 1:N matching |
| 11.2 | Viola-Jones & Haar Cascades | [02-viola-jones-haar-cascades.md](./section-02-viola-jones-and-haar-cascades.md) | Integral images; AdaBoost cascades; `cv2.CascadeClassifier` |
| 11.3 | OpenCV Face Pipeline | [03-opencv-face-pipeline.md](./section-03-opencv-face-pipeline.md) | Detect → crop → resize → grayscale/RGB; preprocessing standards |
| 11.4 | CNN Face Detection | [04-cnn-face-detection.md](./section-04-cnn-face-detection.md) | MTCNN, RetinaFace concepts; DNN chapter in OpenCV |
| 11.5 | Face Embeddings & ArcFace | [05-face-embeddings-arcface.md](./section-05-face-embeddings-and-arcface.md) | Metric learning; embedding vectors; cosine similarity matching |
| 11.6 | Building a Face Recognition System | [06-building-face-recognition-system.md](./section-06-building-a-face-recognition-system.md) | Enrollment gallery; nearest-neighbor matching; threshold selection |
| 11.7 | Open-Set Classification | [07-open-set-classification.md](./section-07-open-set-classification.md) | Unknown person rejection; threshold tuning; false accept vs false reject |
| 11.8 | Ethics & Deployment Considerations | [08-ethics-and-deployment.md](./section-08-ethics-and-deployment.md) | Bias, consent, liveness detection; responsible use guidelines |

---

## Lab

**[Lab 11: Face Detection & Recognition Pipeline](./section-lab-11-face-detection-and-recognition-pipeline.md)**

Build a complete face system:

1. **Detection** — Detect faces in a folder of group photos using Haar cascades. Visualize bounding boxes. Compare with a CNN detector on challenging angles/lighting.
2. **Enrollment** — Build a gallery of 5+ known individuals with 3–5 photos each. Extract ArcFace (or FaceNet) embeddings.
3. **Recognition** — Given a probe image, match to gallery or reject as unknown. Tune threshold on a validation set.
4. **Open-Set Evaluation** — Test with faces not in the gallery. Report false accept rate and true accept rate.

*Deliverable:* Python script or notebook with detection visualizations, recognition demo on webcam or static images, and threshold analysis plot.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Viola-Jones cascades | Boosting theory (Course 2, Ch 19) |
| CNN face detection | RetinaFace, YOLO-face (Chapter 12) |
| ArcFace embeddings | Contrastive learning, triplet loss (Course 3) |
| Open-set recognition | Open-world recognition, out-of-distribution detection (Course 3) |
| Face ethics & bias | Fairness in ML, dataset representation (Course 2) |

---

## Prerequisites

- [Chapter 10 — Convolutional Neural Networks](../chapter-10-convolutional-neural-networks/README.md)
- OpenCV (`pip install opencv-python`)
- A face embedding library (face_recognition, insightface, or deepface) or pretrained Keras model
- Webcam optional for live demo
- Test images with multiple faces and varied lighting

---

## Key Takeaways

- Detection and recognition are separate stages — a recognition model needs well-cropped, aligned faces
- Haar cascades are fast but brittle; CNN detectors handle pose and lighting far better
- Modern recognition uses embedding vectors, not raw pixel classification
- Open-set rejection requires explicit threshold tuning — closed-set accuracy is misleading
- Face recognition systems carry significant ethical responsibilities around consent and bias

---

## Self-Assessment

1. What is the difference between face verification (1:1) and identification (1:N)?
2. Why do Haar cascades struggle with profile views and low light?
3. How does ArcFace produce embeddings that cluster same-person faces together?
4. What happens if you set the recognition threshold too low? Too high?
5. Why is open-set classification harder than closed-set multi-class classification?
6. What preprocessing steps matter most between detection and recognition?
7. What are two ethical concerns when deploying face recognition in public spaces?

---

**Glossary:** [../../GLOSSARY.md](../../GLOSSARY.md) — [deep-learning](../../GLOSSARY.md#deep-learning), [classification](../../GLOSSARY.md#classification), [feature](../../GLOSSARY.md#feature)  
**Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

**Previous:** [Chapter 10 — Convolutional Neural Networks](../chapter-10-convolutional-neural-networks/README.md)  
**Next:** [Chapter 12 — Object Detection](../chapter-12-object-detection/README.md)
