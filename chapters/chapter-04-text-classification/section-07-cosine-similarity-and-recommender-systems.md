# Section 4.7: Cosine Similarity & Recommender Systems

> **Source:** Prosise, Ch. 4 - content-based movie recommendation via cosine similarity  
> **Prerequisites:** [Section 4.3](./section-03-bag-of-words-and-tf-idf.md) | [Section 3.1](../chapter-03-classification-models/section-01-classification-fundamentals.md)  
> **Glossary:** [cosine-similarity](../../GLOSSARY.md#cosine-similarity) | [tf-idf](../../GLOSSARY.md#tf-idf)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## From Classification to "Find Me Something Similar"

Sections 4.5-4.6 assigned **labels** - positive/negative, spam/ham. Prosise's third Chapter 4 application asks a different question: **given a movie you liked, which other movies resemble it?**

No labels required. This is **content-based filtering**: represent each movie by its plot summary (or genres), embed in vector space via [TF-IDF](../../GLOSSARY.md#tf-idf), and rank neighbors by **[cosine similarity](../../GLOSSARY.md#cosine-similarity)**.

> **Humorous analogy:** Classification is sorting mail into bins. Cosine similarity is the bookstore clerk who says "if you liked that murder mystery with a cat, try this one - it also has 'detective,' 'night,' and suspiciously many references to tuna."

> **In plain English:** Turn each movie description into a number vector. Measure the angle between vectors. Small angle = similar content. Recommend the closest matches.

---

## Content-Based vs Collaborative Filtering

| Approach | Uses | Pros | Cons |
|----------|------|------|------|
| **Content-based** | Item features (plot, genre) | Works for new items; no user history needed | Misses "people who liked X also liked Y" |
| **Collaborative** | User ratings matrix | Discovers unexpected taste links | Cold-start for new users/items |

Prosise implements **content-based** with plot text - same vectorization pipeline as sentiment, different downstream math.

---

## Movie Dataset Pattern

```python
import pandas as pd

movies = pd.DataFrame({
    'title': [
        'Space Odyssey',
        'Galaxy Raiders',
        'Love in Paris',
        'Midnight in Paris',
        'Robot Uprising',
        'Star Fighters',
        'Romantic Dinner',
        'Cyber Revolution',
    ],
    'genre': [
        'sci-fi', 'sci-fi', 'romance', 'romance',
        'sci-fi', 'sci-fi', 'romance', 'sci-fi',
    ],
    'description': [
        'astronauts explore distant planets alien discovery space station',
        'intergalactic war heroes battle alien fleet save galaxy',
        'two strangers meet in paris fall in love eiffel tower',
        'writer travels back in time paris artistic golden age',
        'artificial intelligence robots rebel against humanity future city',
        'space pilots defend earth from invading alien armada',
        'chef and food critic romance restaurant wine dinner',
        'hackers and AI fight for control of global network',
    ],
})

print(movies[['title', 'genre']])
```

Combine `genre` + `description` for richer vectors:

```python
movies['text'] = movies['genre'] + ' ' + movies['description']
```

---

## TF-IDF Matrix for All Movies

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
tfidf_matrix = vectorizer.fit_transform(movies['text'])

print(f"Shape: {tfidf_matrix.shape}")  # (n_movies, n_terms)
print(f"Features: {vectorizer.get_feature_names_out()[:12]}")
```

Each row is a movie in high-dimensional term space - sparse like [Section 4.3](./section-03-bag-of-words-and-tf-idf.md).

---

## Cosine Similarity Definition

For vectors $\mathbf{a}$ and $\mathbf{b}$:

$$
\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|} = \frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2} \cdot \sqrt{\sum_i b_i^2}}
$$
> **Readable form:** cosine similarity = dot product of a and b divided by (length of a times length of b)

**Range:** $[-1, 1]$ for general vectors; $[0, 1]$ for non-negative TF-IDF vectors.

| Value | Meaning |
|-------|---------|
| 1.0 | Identical direction (very similar) |
| 0.0 | Orthogonal (no shared terms) |
| -1.0 | Opposite (rare with TF-IDF) |

> **In plain English:** Cosine measures the **angle** between vectors, not their length. Two documents about "space aliens" match even if one is a short blurb and one is a long plot summary - unlike Euclidean distance, which punishes length.

---

## Why Not Euclidean Distance?

Document A: 1000-word sci-fi plot. Document B: 50-word sci-fi blurb. Same theme, different magnitudes.

Euclidean distance $\|\mathbf{a} - \mathbf{b}\|$ grows with vector length. Cosine normalizes by $\|\mathbf{a}\|$ and $\|\mathbf{b}\|$ - compares **topic proportions**, not document size.

---

## Computing Similarity with sklearn

```python
from sklearn.metrics.pairwise import cosine_similarity

sim_matrix = cosine_similarity(tfidf_matrix)
print(sim_matrix.shape)  # (8, 8) - movie-to-movie

# Similarity between movie 0 and all others
print(sim_matrix[0])
```

Diagonal is 1.0 (each movie identical to itself). Matrix is symmetric.

---

## Recommending Top-K Similar Movies

```python
import numpy as np

def recommend_similar(title, movies_df, sim_matrix, top_k=5):
    idx = movies_df[movies_df['title'] == title].index[0]
    scores = sim_matrix[idx]
    # Exclude self
    top_indices = np.argsort(scores)[::-1][1:top_k + 1]
    results = movies_df.iloc[top_indices][['title', 'genre']].copy()
    results['similarity'] = scores[top_indices]
    return results

print(recommend_similar('Space Odyssey', movies, sim_matrix, top_k=3))
```

Expected neighbors for `Space Odyssey`: other sci-fi titles (`Galaxy Raiders`, `Star Fighters`, `Robot Uprising`) - not romance films.

---

## Pretty Printing Recommendations

```python
def print_recommendations(query_title, top_k=5):
    recs = recommend_similar(query_title, movies, sim_matrix, top_k)
    print(f"\nBecause you liked '{query_title}':")
    for _, row in recs.iterrows():
        print(f"  • {row['title']:22s} ({row['genre']:8s}) sim={row['similarity']:.3f}")

print_recommendations('Love in Paris')
print_recommendations('Robot Uprising')
```

---

## User Query as a New Vector

"What if the user types a free-text preference?"

```python
def recommend_from_query(query_text, vectorizer, tfidf_matrix, movies_df, top_k=5):
    query_vec = vectorizer.transform([query_text])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = movies_df.iloc[top_indices][['title', 'genre']].copy()
    results['similarity'] = scores[top_indices]
    return results

query = "space battle aliens galaxy war"
print(recommend_from_query(query, vectorizer, tfidf_matrix, movies))
```

The query is just another document in the same vector space - no retraining needed.

---

## Prosise Movie Recommender - Full Script

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your movie catalog
# movies = pd.read_csv('movies.csv')  # columns: title, plot, genres

movies['combined'] = movies['genre'].fillna('') + ' ' + movies['description'].fillna('')

tfidf = TfidfVectorizer(stop_words='english', min_df=1, ngram_range=(1, 2))
matrix = tfidf.fit_transform(movies['combined'])
similarity = cosine_similarity(matrix)

def get_recommendations(title, n=5):
    if title not in movies['title'].values:
        raise ValueError(f"Unknown title: {title}")
    idx = movies[movies['title'] == title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:n + 1]  # skip self
    for i, score in scores:
        print(f"  {movies.iloc[i]['title']:30s}  similarity={score:.4f}")

print("Recommendations for 'Space Odyssey':")
get_recommendations('Space Odyssey')
```

---

## Similarity Matrix Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(
    sim_matrix,
    xticklabels=movies['title'],
    yticklabels=movies['title'],
    cmap='YlOrRd',
    annot=True,
    fmt='.2f',
)
plt.title('Movie Cosine Similarity')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

Sci-fi cluster should show high intra-genre similarity.

---

## Limitations of Content-Based TF-IDF

| Limitation | Consequence |
|------------|-------------|
| Vocabulary overlap only | `"car"` vs `"automobile"` - no match |
| No semantics | `"king"` / `"queen"` not linked without co-occurrence |
| Popularity bias | Blockbusters share generic words |
| Filter bubble | Only recommends "more of the same" |
| Cold start (new user) | OK for items; user taste unknown without ratings |

[Chapter 13](../chapter-13-natural-language-processing/README.md) embeddings fix vocabulary mismatch; collaborative filtering adds social signal.

---

## Hybrid Idea (Awareness)

Production systems often blend:

$$
\text{score} = \alpha \cdot \text{cosine\_content} + (1 - \alpha) \cdot \text{collab\_score}
$$
> **Readable form:** final score = alpha times content similarity plus (1 minus alpha) times collaborative score

Prosise stays content-only for pedagogical clarity.

---

## Key Takeaways

1. **Content-based recommenders** match item features, not user history
2. **[TF-IDF](../../GLOSSARY.md#tf-idf)** vectors from plot/genre text feed the same pipeline as classifiers
3. **[Cosine similarity](../../GLOSSARY.md#cosine-similarity)** measures angle between vectors - length-invariant
4. **Similarity matrix** enables top-K recommendations in one dot product
5. **Free-text queries** transform and compare without retraining
6. **Limitations** motivate embeddings and collaborative filtering later

---

## Check Your Understanding

1. Why is cosine similarity preferred over Euclidean distance for TF-IDF documents?
2. What is the difference between content-based and collaborative filtering?
3. How do you exclude the query movie from its own recommendation list?
4. What happens if two movies share no vocabulary words?
5. Why does combining genre + plot improve recommendations?

---

## References

- Prosise, Ch. 4 - cosine similarity and movie recommender
- Scikit-Learn `cosine_similarity`: [https://scikit-learn.org/stable/chapters/generated/sklearn.metrics.pairwise.cosine_similarity.html](https://scikit-learn.org/stable/chapters/generated/sklearn.metrics.pairwise.cosine_similarity.html)
- Manning et al. - *IR Book*, Ch. 6 (vector space model): [https://nlp.stanford.edu/IR-book/pdf/06vect.pdf](https://nlp.stanford.edu/IR-book/pdf/06vect.pdf)
- Ricci, Rokach & Shapira - Recommender Systems Handbook (Springer)

---

**Previous:** [Section 4.6 - Spam Filtering](./section-06-spam-filtering.md)  
**Next:** [Section 4.8 - Pipelines & Production Text ML](./section-08-pipelines-and-production-text-ml.md)
