# Section 1.2: Machine Learning vs AI vs Deep Learning

> **Source inheritance:** Prosise, Ch. 1 - "Machine Learning Versus Artificial Intelligence"  
> **Enhanced with:** Ecosystem taxonomy, Russell & Norvig framing (preview of Course 2)  
> **Glossary:** [../../GLOSSARY.md](../../GLOSSARY.md) | **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Confusion

Walk into any tech conference and you'll hear "AI," "machine learning," and "deep learning" used interchangeably. Marketing departments love this ambiguity. Engineers cannot afford it.

This section establishes precise definitions that the entire ecosystem depends on.

---

## The Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                  ARTIFICIAL INTELLIGENCE                 │
│  "Building machines that act intelligently"              │
│  ┌───────────────────────────────────────────────────┐  │
│  │              MACHINE LEARNING                      │  │
│  │  "Learning from data"                              │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │           DEEP LEARNING                      │  │  │
│  │  │  "Learning with neural networks (deep)"      │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
│  Also includes: search, logic, planning, robotics...     │
└─────────────────────────────────────────────────────────┘
```

**Artificial Intelligence** is the broadest term.  
**[Machine Learning](../../GLOSSARY.md#machine-learning)** is a *subset* of [AI](../../GLOSSARY.md#artificial-intelligence) - systems that improve through experience.  
**[Deep Learning](../../GLOSSARY.md#deep-learning)** is a *subset* of ML - systems using multi-layer neural networks.

---

## Artificial Intelligence - The Big Picture

Russell & Norvig (Course 2) define AI as:

> The study and construction of agents that receive percepts from the environment and perform actions.

An **intelligent agent** perceives, reasons, and acts to maximize some measure of success. AI encompasses:

| Approach | Examples | Covered In |
|----------|----------|-----------|
| **Thinking humanly** | Cognitive modeling | Course 2, Ch. 1 |
| **Thinking rationally** | Logic, inference | Course 2, Ch. 7-9 |
| **Acting humanly** | Turing Test | Course 2, Ch. 1 |
| **Acting rationally** | Rational agents | Course 2, Ch. 2 |
| **Learning from data** | ML, deep learning | Courses 1, 3, 4 |

**Key insight:** ML is the dominant approach in modern AI, but it is not the only one. Before big data, AI was primarily about search (finding paths in state spaces), logic (proving theorems), and planning (scheduling actions). These remain essential - AlphaGo combines deep learning *with* Monte Carlo tree search.

---

## Machine Learning - Learning from Experience

ML systems improve performance on a task through experience (data), without explicit reprogramming.

### Common Learning Paradigms in AI

#### 1. Supervised Learning
- **Data:** Input-output pairs ([features](../../GLOSSARY.md#feature) + [labels](../../GLOSSARY.md#label))
- **Goal:** Learn mapping from inputs to outputs
- **Examples:** Email spam detection, house price prediction, medical diagnosis
- **Algorithms:** [k-NN](../../GLOSSARY.md#k-nearest-neighbors), decision trees, SVMs, neural networks

#### 2. Unsupervised Learning
- **Data:** Inputs only (no labels)
- **Goal:** Discover hidden structure
- **Examples:** Customer segmentation, anomaly detection, dimensionality reduction
- **Algorithms:** [k-means](../../GLOSSARY.md#k-means), PCA, autoencoders

#### 3. Reinforcement Learning
- **Data:** Agent interacts with environment, receives rewards/penalties
- **Goal:** Learn policy that maximizes cumulative reward
- **Examples:** Game playing (AlphaGo), robotics, recommendation systems
- **Algorithms:** Q-learning, policy gradients, PPO
- **Covered in:** Course 2 (Ch. 22), Course 4 (Ch. 12 - World Models)

Prosise's Chapter 1 focuses on supervised and unsupervised ML because those are the applied workflows used in the first chapters. Reinforcement learning is best treated here as a neighboring AI learning framework: it can use ML models internally, but its core setup is an agent optimizing reward through interaction rather than a fixed labeled or unlabeled dataset.

---

## Deep Learning - Neural Networks at Scale

Deep learning uses **artificial neural networks** with multiple hidden layers ("deep" = many layers) to learn hierarchical representations of data.

A single neuron computes a weighted sum plus bias, passed through an activation function:

$$
z = \sum_{j=1}^{n} w_j x_j + b, \quad a = \sigma(z)
$$
> **Readable form:** pre-activation = (sum of weight times feature) + bias; output = activation applied to pre-activation

Stacking many such units into layers lets the network approximate complex functions - the universal approximation idea that makes [deep learning](../../GLOSSARY.md#deep-learning) powerful on unstructured data.

### Why "Deep" Matters

```
Input Image (pixels)
       ↓
Layer 1: Edges, gradients
       ↓
Layer 2: Corners, textures
       ↓
Layer 3: Parts (eyes, wheels)
       ↓
Layer 4: Objects (faces, cars)
       ↓
Output: Classification
```

Each layer learns increasingly abstract features. This **representation learning** is deep learning's superpower - you don't hand-engineer features; the network discovers them.

### When Deep Learning Wins

| Condition | Why DL Excels |
|-----------|--------------|
| Large datasets (millions+ examples) | More parameters need more data |
| Unstructured data (images, text, audio) | Hierarchical features are hard to hand-craft |
| Complex patterns | Deep networks have high capacity |
| Transfer learning available | Pretrained models (ResNet, BERT, GPT) accelerate development |

### When Deep Learning Is Overkill

| Condition | Better Alternative |
|-----------|-------------------|
| Small tabular dataset (<10K rows) | Gradient boosting, [random forests](../../GLOSSARY.md#random-forest) |
| Need interpretability | Decision trees, [linear regression](../../GLOSSARY.md#linear-regression) |
| Limited compute | Classical ML is faster to train |
| Simple linear relationships | Linear/logistic regression |

> **Course progression:** You learn classical ML in Chapters 01-07, introductory deep learning in Chapters 08-14, mathematical deep learning theory in Course 3, and generative deep learning in Course 4.

---

## Comparison Table

| Aspect | Classical ML | Deep Learning |
|--------|-------------|---------------|
| Feature engineering | Manual (critical) | Automatic (learned) |
| Data requirements | Moderate | Large |
| Training time | Seconds to minutes | Minutes to weeks |
| Interpretability | Often high | Often low ("black box") |
| Hardware | CPU sufficient | GPU strongly preferred |
| Best for | Tabular data, small datasets | Images, text, audio, sequences |
| Examples | Random forest, SVM, [k-NN](../../GLOSSARY.md#k-nearest-neighbors) | CNN, RNN, Transformer |

---

## Prosise Ch. 1: "AI" vs "ML" in the Book

Prosise uses **Artificial Intelligence** as the umbrella and positions [machine learning](../../GLOSSARY.md#machine-learning) as the practical engine behind most modern AI products. His key pedagogical move: show that you do not need neural networks to do useful AI work.

| Prosise Claim | Classical ML Example | Deep Learning Counterpart |
|---------------|---------------------|---------------------------|
| ML learns from data | [k-NN](../../GLOSSARY.md#k-nearest-neighbors) on Iris flowers | CNN on ImageNet |
| Feature engineering matters | Hand-pick sepal/petal measurements | Network learns filters automatically |
| Start simple, then scale | [k-means](../../GLOSSARY.md#k-means) customer segments | Embedding-based segmentation |

His Iris flower classifier (Section 1.5) is intentionally **not** deep learning - it proves that intelligent behavior can emerge from geometry (distance + voting) alone. That humility saves practitioners from reaching for a GPU when a 50-line sklearn script suffices.

### The Symbolic AI Era (Context)

Before the data-driven revolution, AI meant **symbolic** methods:

| Approach | Core idea | Modern status |
|----------|-----------|---------------|
| **Search** | Explore state spaces (chess, routing) | Still used in planning, games (AlphaGo) |
| **Logic** | Prove theorems, enforce constraints | Rules engines, verification |
| **Expert systems** | Encode human rules in if-then chains | Largely replaced by ML in perception tasks |

Russell & Norvig (Course 2) cover these rigorously. The ecosystem insight: **production AI is hybrid** - neural nets for perception, search/planning for decisions, ML for ranking and prediction.

---

## AI Beyond Machine Learning

Modern AI systems often combine multiple approaches:

### AlphaGo (2016)
- **Monte Carlo Tree Search** (classical AI - Course 2, Ch. 5)
- **Deep neural networks** for position evaluation (Course 3)
- **Reinforcement learning** from self-play (Course 2, Ch. 22)

### Self-Driving Cars
- **Computer vision** (deep learning - Course 1, Chapters 10-12)
- **Path planning** (search algorithms - Course 2, Ch. 3-4)
- **Probabilistic reasoning** for sensor fusion (Course 2, Ch. 12-14)
- **Reinforcement learning** for control policies (Course 2, Ch. 22)

### ChatGPT
- **Transformer architecture** (deep learning - Course 4, Ch. 9)
- **Pretraining on massive text** (unsupervised learning)
- **Reinforcement learning from human feedback** (RLHF)
- **Ethics and safety** considerations (Course 2, Ch. 27; Course 4, Ch. 14)

---

## The Practitioner's Mental Model

When facing a new problem, ask:

```
1. Is this an AI problem at all?
   └─ Can simple rules solve it? → Don't use ML

2. Do I have data?
   └─ Passive data, no labels → Unsupervised learning
   └─ Interactive environment with rewards → Reinforcement learning
   └─ Yes labeled data → Supervised learning

3. What type of data?
   └─ Tabular → Classical ML (Chapters 01-07)
   └─ Images/text/audio → Deep learning (Chapters 08-14+)

4. How much data?
   └─ Small → Classical ML or transfer learning
   └─ Large → Deep learning from scratch or fine-tuning

5. What constraints?
   └─ Must explain decisions → Interpretable models
   └─ Must deploy on edge → Model size matters
   └─ Must be real-time → Latency constraints
```

---

## Common Misconceptions

| Myth | Reality |
|------|---------|
| "AI = ML = Deep Learning" | AI ⊃ ML ⊃ DL; AI includes search, logic, planning |
| "More layers always = better" | Deeper networks need more data, risk overfitting |
| "AI will replace all programmers" | AI augments engineers; understanding fundamentals matters more |
| "You need a PhD for ML" | This course proves otherwise - start building today |
| "Deep learning solved AI" | No single approach solves all problems; hybrid systems win |
| "Classical ML is obsolete" | Tabular data, small datasets, and interpretability needs keep classical methods dominant in industry |
| "You need GPUs for all ML" | [k-NN](../../GLOSSARY.md#k-nearest-neighbors), trees, and [linear regression](../../GLOSSARY.md#linear-regression) run fine on CPU |

### Common Mistakes When Choosing an Approach

| Mistake | Symptom | Better Path |
|---------|---------|-------------|
| Using [deep learning](../../GLOSSARY.md#deep-learning) on 500-row spreadsheets | [Overfitting](../../GLOSSARY.md#overfitting), slow iteration | Gradient boosting or logistic regression |
| Calling any automation "AI" | Stakeholder confusion | Reserve "AI" for learning/adaptive systems |
| Skipping classical baselines | No benchmark to beat | Always try [k-NN](../../GLOSSARY.md#k-nearest-neighbors) or logistic regression first |
| Ignoring interpretability requirements | Regulators reject black-box | Trees, linear models, SHAP (later chapters) |

---

## Reflection Questions

1. Is a chess engine that searches 20 moves ahead "machine learning"? Why or why not?
2. Why did deep learning explode after 2012 but not in the 1990s?
3. Where would you draw the line between "smart automation" and "[AI](../../GLOSSARY.md#artificial-intelligence)"?

---

## Further Reading

- [GLOSSARY.md](../../GLOSSARY.md) - [machine learning](../../GLOSSARY.md#machine-learning), [deep learning](../../GLOSSARY.md#deep-learning), [artificial intelligence](../../GLOSSARY.md#artificial-intelligence)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Next:** [Section 1.3 - Supervised vs Unsupervised Learning](./section-03-supervised-vs-unsupervised-learning.md)


