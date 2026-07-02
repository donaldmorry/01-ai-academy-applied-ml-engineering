# Chapter 04: Text Classification

> **Source:** *Applied Machine Learning and AI for Engineers*, Chapter 4  
> **Part:** I — Machine Learning with Scikit-Learn  
> **Estimated time:** 8–10 hours  
> **Prerequisites:** [Chapter 03 — Classification Models](../chapter-03-classification-models/README.md); basic string manipulation in Python

---

## Chapter Overview

Most of the world's data is unstructured text — emails, reviews, support tickets, social posts. This chapter teaches you to transform raw language into numerical features that classical ML algorithms can process, then apply those features to real problems: **sentiment analysis**, **spam filtering**, and **recommender systems**.

You will learn the text preprocessing pipeline (tokenization, stop-word removal, stemming), bag-of-words and TF-IDF vectorization, and why Naive Bayes remains a surprisingly strong baseline for text. The chapter also introduces **cosine similarity** as the foundation for content-based recommendation — connecting classification to the retrieval problems you will see again in NLP and search systems.

This is your first encounter with domain-specific feature engineering. The skills here transfer directly to Chapter 13 (deep learning NLP) and Chapter 14 (Azure Language services).

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Preprocess text data: tokenization, lowercasing, stop words, stemming/lemmatization
2. Convert text to numerical features with CountVectorizer and TfidfVectorizer
3. Train Naive Bayes classifiers for text and explain the independence assumption
4. Build a sentiment analysis system for movie or product reviews
5. Implement a spam filter with high precision on held-out email data
6. Compute cosine similarity between document vectors
7. Build a content-based movie recommender from plot descriptions or genres
8. Evaluate text classifiers with appropriate metrics for imbalanced corpora

---

## Sections

| # | Section | Topics |
|---|--------|--------|
| 4.1 | [Text as Data](./section-01-text-as-data.md) | Challenges of unstructured text; corpus, document, token vocabulary |
| 4.2 | [Text Preprocessing](./section-02-text-preprocessing.md) | Tokenization, normalization, stop words, stemming; NLTK and Scikit-Learn |
| 4.3 | [Bag-of-Words & TF-IDF](./section-03-bag-of-words-and-tf-idf.md) | `CountVectorizer`, `TfidfVectorizer`; n-grams; sparsity |
| 4.4 | [Naive Bayes for Text](./section-04-naive-bayes-for-text.md) | Multinomial NB; independence assumption; `MultinomialNB` |
| 4.5 | [Sentiment Analysis](./section-05-sentiment-analysis.md) | Review classification; feature inspection; error analysis on misclassified text |
| 4.6 | [Spam Filtering](./section-06-spam-filtering.md) | Email/ham classification; precision focus; cross-validation on text |
| 4.7 | [Cosine Similarity & Recommender Systems](./section-07-cosine-similarity-and-recommender-systems.md) | Vector space model; content-based filtering; similarity matrices |
| 4.8 | [Pipelines & Production Text ML](./section-08-pipelines-and-production-text-ml.md) | `Pipeline` with vectorizer + classifier; persistence; latency considerations |

---

## Lab

**[Lab 04: Sentiment, Spam, and Recommendations](./section-lab-04-sentiment-spam-and-recommendations.md)**

Build three text-based systems in one lab:

1. **Sentiment Analyzer** — Train on a movie review dataset (e.g., IMDB). Achieve >85% accuracy with TF-IDF + Naive Bayes or logistic regression. Show the 10 most positive and 10 most negative weighted terms.
2. **Spam Filter** — Classify SMS or email messages as spam/ham. Optimize for precision — a false positive (good email marked spam) is worse than a missed spam.
3. **Movie Recommender** — Given a movie's description or genre tags, recommend the top-5 most similar movies using cosine similarity on TF-IDF vectors.

*Deliverable:* Notebook with three working systems, example predictions on custom input text, and a brief note on limitations of bag-of-words approaches.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Bag-of-words / TF-IDF | Word embeddings, Word2Vec, GloVe (Chapter 13) |
| Naive Bayes text classification | Formal probabilistic NLP, HMMs (Course 2, Ch 23) |
| Cosine similarity & retrieval | Vector databases, semantic search (Course 3) |
| Sentiment analysis | Transformer fine-tuning, BERT (Chapter 13) |
| Recommender systems | Collaborative filtering, matrix factorization (Course 2) |

---

## Prerequisites

- [Chapter 03 — Classification Models](../chapter-03-classification-models/README.md) (classification metrics, pipelines)
- Python string operations and regular expressions (basic)
- NLTK or equivalent text processing library installed
- Familiarity with sparse matrices (helpful but not required)

---

## Key Takeaways

- Text must be converted to numbers before classical ML — TF-IDF is the standard starting point
- Naive Bayes is fast, interpretable, and often competitive on text despite its simplifying assumption
- N-grams capture local word order that bag-of-words alone misses
- Cosine similarity measures document similarity independent of document length
- Bag-of-words loses word order and semantics — deep learning (Chapter 13) addresses these limits

---

## Self-Assessment

1. Why is TF-IDF preferred over raw word counts for most text classification tasks?
2. What does the Naive Bayes "independence assumption" mean, and why does it still work?
3. How would you evaluate a spam filter where false positives are 10× more costly than false negatives?
4. What information is lost when you represent a document as a bag-of-words vector?
5. How does cosine similarity differ from Euclidean distance for text vectors?
6. Why might a content-based recommender fail for a new user with no history?
7. What preprocessing steps would you add for social media text vs formal documents?

---

**Previous:** [Chapter 03 — Classification Models](../chapter-03-classification-models/README.md)  
**Next:** [Chapter 05 — Support Vector Machines](../chapter-05-support-vector-machines/README.md)
