# Applied Machine Learning and AI for Engineers

<p align="center">
  <img src="./assets/course-symbol.png" alt="Symbolic banner for applied machine learning engineering" width="100%">
</p>

A practical, engineering-first course in machine learning, deep learning, computer vision, natural language processing, and model deployment. This repository is designed for learners who want to build real systems, understand the tradeoffs behind those systems, and develop enough judgment to choose the simplest model that solves the problem well.

This course follows the learning arc of Jeff Prosise's *Applied Machine Learning and AI for Engineers* while expanding it into a structured academy-style curriculum with chapters, sections, labs, glossary links, capstone work, and production-oriented review prompts.

## Who This Course Is For

- Software engineers moving into applied ML or AI engineering.
- Data analysts who want to become model builders.
- Students who prefer runnable examples before formal theory.
- Product-minded builders who need enough ML depth to ship responsibly.

You should be comfortable with Python functions, classes, virtual environments, and notebooks. You do not need prior machine learning experience.

## What You Will Be Able To Do

By the end of the course, you will be able to:

1. Frame supervised, unsupervised, and deep learning problems clearly.
2. Build regression, classification, clustering, recommendation, and NLP pipelines.
3. Train and evaluate classical models with Scikit-Learn.
4. Train neural networks and convolutional networks with Keras and TensorFlow.
5. Apply transfer learning, augmentation, and embedding-based NLP.
6. Build face detection, face recognition, and object detection workflows.
7. Package models for production with ONNX, containers, cloud services, and API boundaries.
8. Explain accuracy, latency, cost, data quality, and monitoring tradeoffs to stakeholders.

## Curriculum Map

| Chapter | Focus | Sections |
|---------|-------|----------|
| [01](./chapters/chapter-01-machine-learning/README.md) | Machine Learning Fundamentals | 7 |
| [02](./chapters/chapter-02-regression-models/README.md) | Regression Models | 9 |
| [03](./chapters/chapter-03-classification-models/README.md) | Classification Models | 9 |
| [04](./chapters/chapter-04-text-classification/README.md) | Text Classification | 9 |
| [05](./chapters/chapter-05-support-vector-machines/README.md) | Support Vector Machines | 9 |
| [06](./chapters/chapter-06-principal-component-analysis/README.md) | Principal Component Analysis | 9 |
| [07](./chapters/chapter-07-operationalizing-models/README.md) | Operationalizing ML Models | 9 |
| [08](./chapters/chapter-08-deep-learning/README.md) | Deep Learning Foundations | 9 |
| [09](./chapters/chapter-09-neural-networks/README.md) | Neural Networks with Keras & TensorFlow | 9 |
| [10](./chapters/chapter-10-convolutional-neural-networks/README.md) | Convolutional Neural Networks | 9 |
| [11](./chapters/chapter-11-face-detection-recognition/README.md) | Face Detection & Recognition | 9 |
| [12](./chapters/chapter-12-object-detection/README.md) | Object Detection | 9 |
| [13](./chapters/chapter-13-natural-language-processing/README.md) | Natural Language Processing | 9 |
| [14](./chapters/chapter-14-azure-cognitive-services/README.md) | Azure Cognitive Services | 9 |

## How To Study

Work through chapters in order. Each chapter contains multiple sections and a lab or practice task. Read the section, run the code, change one parameter, and write down what changed. Do not skip the classical ML chapters: they are the baseline that keeps later neural models honest.

A productive weekly rhythm is:

- Day 1: Read two sections and reproduce the examples.
- Day 2: Complete the chapter lab or extend an example.
- Day 3: Run error analysis and document one failure case.
- Day 4: Review glossary terms and compare with earlier chapters.
- Day 5: Write a short implementation note as if you were handing the model to another engineer.

## Capstone

The capstone asks you to build an end-to-end ML system: problem framing, data preparation, model comparison, evaluation, deployment plan, and risk analysis. Start with [the capstone specification](./projects/capstone/README.md) once you finish Chapter 07, then refine it as the deep learning and AI service chapters add new options.

## Repository Guide

| Path | Purpose |
|------|---------|
| `chapters/` | Course chapters and section files |
| `projects/` | Capstone project specification and rubrics |
| `resources/` | Supplementary references and study aids |
| `assets/` | Repository banner and visual assets |
| `GLOSSARY.md` | Shared technical vocabulary |
| `MATH_CONVENTIONS.md` | Math formatting conventions |
| `ASSESSMENT_APPENDIX.md` | Reusable mastery prompts and review templates |

## Source And Attribution

This is an independent educational curriculum inspired by Jeff Prosise's *Applied Machine Learning and AI for Engineers*. It is not a replacement for the book. Learners are encouraged to read the original text and use this repository as a structured companion for implementation, review, and portfolio work.

## Start Here

Begin with [Chapter 01: Machine Learning Fundamentals](./chapters/chapter-01-machine-learning/README.md).
