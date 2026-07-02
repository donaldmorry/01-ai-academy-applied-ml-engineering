# Chapter 12: Object Detection

> **Source:** *Applied Machine Learning and AI for Engineers*, Chapter 12  
> **Part:** II — Deep Learning with Keras & TensorFlow  
> **Estimated time:** 8–10 hours  
> **Prerequisites:** [Chapter 10 — Convolutional Neural Networks](../chapter-10-convolutional-neural-networks/README.md); [Chapter 11](../chapter-11-face-detection-recognition/README.md) detection concepts helpful

---

## Chapter Overview

Image classification answers "what is in this image?" Object detection answers **"what is in this image, and where?"** — drawing bounding boxes around every instance of every object class. This chapter traces the evolution of detection architectures from **R-CNN** through **Mask R-CNN** to single-shot detectors like **YOLO** and **YOLOv3**, then applies them in practice.

You will run pretrained detectors on real images, understand the tradeoffs between accuracy and speed (two-stage vs single-shot), and use **Azure Custom Vision** to train a detector without building the architecture from scratch — a pattern common in enterprise ML where time-to-value matters more than research novelty.

Object detection powers autonomous vehicles, retail analytics, wildlife monitoring, and industrial inspection. The skills here complete your computer vision toolkit alongside classification (Chapter 10) and face systems (Chapter 11).

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Distinguish object detection from image classification and segmentation
2. Explain the R-CNN family: region proposals, RoI pooling, Fast/Faster R-CNN
3. Describe Mask R-CNN extensions for instance segmentation
4. Understand YOLO's single-shot approach and its speed advantages
5. Run YOLOv3 (or YOLOv5/v8) inference on images and video with pretrained weights
6. Interpret detection outputs: bounding boxes, class labels, confidence scores, NMS
7. Train a custom object detector with Azure Custom Vision
8. Evaluate detectors with mAP (mean Average Precision) at a conceptual level

---

## Sections

| # | Section | File | Topics |
|---|--------|------|--------|
| 12.1 | From Classification to Detection | [01-from-classification-to-detection.md](./section-01-from-classification-to-detection.md) | Bounding boxes; multiple objects per image; localization + classification |
| 12.2 | R-CNN & Fast R-CNN | [02-r-cnn-and-fast-r-cnn.md](./section-02-r-cnn-and-fast-r-cnn.md) | Region proposals; selective search; RoI pooling; pipeline bottlenecks |
| 12.3 | Faster R-CNN & Mask R-CNN | [03-faster-r-cnn-and-mask-r-cnn.md](./section-03-faster-r-cnn-and-mask-r-cnn.md) | Region Proposal Network; instance segmentation masks |
| 12.4 | Single-Shot Detectors & YOLO | [04-single-shot-detectors-and-yolo.md](./section-04-single-shot-detectors-and-yolo.md) | Grid-based prediction; anchor boxes; speed vs accuracy tradeoff |
| 12.5 | YOLOv3 in Practice | [05-yolov3-in-practice.md](./section-05-yolov3-in-practice.md) | Loading weights; inference on images/video; confidence and IoU thresholds |
| 12.6 | Non-Maximum Suppression | [06-non-maximum-suppression.md](./section-06-non-maximum-suppression.md) | Overlapping boxes; NMS algorithm; tuning thresholds |
| 12.7 | Azure Custom Vision | [07-azure-custom-vision.md](./section-07-azure-custom-vision.md) | No-code/low-code training; uploading images; tagging objects; REST API deployment |
| 12.8 | Choosing a Detection Architecture | [08-choosing-detection-architecture.md](./section-08-choosing-a-detection-architecture.md) | mAP, FPS, edge vs cloud; when to use cloud services vs self-hosted YOLO |

---

## Lab

**[Lab 12: Object Detection in the Wild](./section-lab-12-object-detection-in-the-wild.md)**

Complete two detection tracks:

1. **YOLO Inference** — Run YOLOv3 (or modern YOLO variant) on a set of diverse images (street scenes, wildlife, indoor). Visualize bounding boxes with labels and confidence scores. Experiment with confidence threshold and NMS IoU.
2. **Azure Custom Vision** — Create a Custom Vision object detection project. Upload and tag 30+ images of a custom object class (e.g., product SKU, plant species, safety equipment). Train, evaluate, and call the prediction API from Python.

*Deliverable:* Notebook with YOLO visualizations, Custom Vision API integration code, and a comparison table of setup effort, accuracy, and inference latency for both approaches.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| R-CNN family | Detailed architecture papers, FPN (Course 3) |
| YOLO & single-shot detectors | YOLOv5–v8, anchor-free detectors (Course 3) |
| Instance segmentation | Panoptic segmentation, SAM (Course 3) |
| Azure Custom Vision | Full Azure ML, MLOps pipelines (Course 2) |
| mAP evaluation | COCO benchmark, evaluation protocols (Course 3) |

---

## Prerequisites

- [Chapter 10 — Convolutional Neural Networks](../chapter-10-convolutional-neural-networks/README.md)
- [Chapter 11](../chapter-11-face-detection-recognition/README.md) detection pipeline concepts (helpful)
- OpenCV for visualization
- YOLO weights and framework (Darknet, ultralytics, or TensorFlow port)
- Azure account for Custom Vision (free tier sufficient)

---

## Key Takeaways

- Object detection = localization (where) + classification (what) for every object instance
- Two-stage detectors (Faster R-CNN) are more accurate; single-shot (YOLO) is faster
- Non-maximum suppression is essential to remove duplicate overlapping detections
- Pretrained COCO models detect 80 common classes out of the box
- Cloud services like Azure Custom Vision accelerate custom detector training for business teams

---

## Self-Assessment

1. How does object detection differ from image classification with a CNN?
2. What problem does Non-Maximum Suppression solve?
3. Why is YOLO faster than Faster R-CNN?
4. What does Mask R-CNN add beyond bounding boxes?
5. How do confidence threshold and IoU threshold affect detection results?
6. When would you choose Azure Custom Vision over training YOLO yourself?
7. What is mAP, and why is it preferred over simple accuracy for detection?

---

**Previous:** [Chapter 11 — Face Detection & Recognition](../chapter-11-face-detection-recognition/README.md)  
**Next:** [Chapter 13 — Natural Language Processing](../chapter-13-natural-language-processing/README.md)
