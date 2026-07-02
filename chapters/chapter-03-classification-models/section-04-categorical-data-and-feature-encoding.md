# Section 3.4: Categorical Data & Feature Encoding

> **Source:** Prosise, Ch. 3 - Feature preparation for Titanic & tabular classifiers  
> **Prerequisites:** [Section 3.1](./section-01-classification-fundamentals.md) | [Section 1.3](../chapter-01-machine-learning/section-03-supervised-vs-unsupervised-learning.md)  
> **Glossary:** [feature](../../GLOSSARY.md#feature) | [classification](../../GLOSSARY.md#classification)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Algorithms Speak Numbers, Humans Speak Categories

Prosise's Titanic dataset is a section in **messy reality**: `Sex` is "male"/"female", `Embarked` is "S", "C", "Q", `Pclass` is 1/2/3. Scikit-Learn's `LogisticRegression` and SVM multiply **numeric** columns - they cannot multiply "Southampton."

**Feature encoding** converts categories into numbers without lying to the model about order that doesn't exist.

> **Humorous analogy:** Feeding raw strings to logistic regression is like asking someone to multiply "banana" by 2.7. Encoding is the translator at the UN.

---

## Types of Categorical Features

| Type | Example | Has meaningful order? |
|------|---------|----------------------|
| **Nominal** | Color, embarkation port, sex | No |
| **Ordinal** | Ticket class 1 > 2 > 3, education level | Yes (careful!) |
| **Binary** | Survived yes/no | Special case of nominal |

Wrong encoding invents fake math: label-encoding `Embarked` as S=0, C=1, Q=2 implies Q is "twice" C. Usually false.

---

## Label Encoding (Integer Codes)

Map each category to $0, 1, 2, \ldots$

$$
\text{Embarked}: \quad \text{S} \to 0,\; \text{C} \to 1,\; \text{Q} \to 2
$$
> **Readable form:** each category name maps to a unique integer

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['embarked_code'] = le.fit_transform(df['Embarked'].fillna('S'))
print(le.classes_)  # ['C', 'Q', 'S'] - alphabetical order, not geographic!
```

| ✅ OK for | ❌ Avoid for |
|----------|-------------|
| Tree models (sometimes) | [Logistic regression](./section-02-logistic-regression.md), linear models |
| Target encoding (advanced) | Nominal features with no order |

Trees can split on `embarked_code == 2` without assuming 2 > 1 in a linear sense - but one-hot is still safer for consistency.

---

## One-Hot Encoding (Dummy Variables)

Create one binary column per category. For $k$ categories, use **$k-1$** dummies or $k$ with `drop='first'` to avoid the **dummy variable trap** (perfect multicollinearity with intercept).

For `Sex` ∈ {male, female}:

| Passenger | sex_female |
|-----------|------------|
| male | 0 |
| female | 1 |

For `Embarked` with `drop='first'` (drop C):

| Embarked | embarked_Q | embarked_S |
|----------|------------|------------|
| C | 0 | 0 |
| Q | 1 | 0 |
| S | 0 | 1 |

```python
import pandas as pd

df = pd.DataFrame({
    'Sex': ['male', 'female', 'female', 'male'],
    'Embarked': ['S', 'C', 'Q', 'S'],
})

dummies = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
print(dummies)
```

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
embarked_arr = ohe.fit_transform(df[['Embarked']])
print(ohe.get_feature_names_out(['Embarked']))
```

> **In plain English:** One-hot says "is this passenger from Queenstown?" as a separate yes/no switch - no fake ordering.

---

## Ordinal Encoding (When Order Matters)

`Pclass`: 1st > 2nd > 3rd - higher class, better survival odds in Prosise's analysis.

```python
from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder(categories=[[1, 2, 3]])  # explicit order
df['pclass_ord'] = ord_enc.fit_transform(df[['Pclass']])
```

**Caution:** Only use ordinal encoding when the **spacing** between levels is roughly meaningful, or when trees handle arbitrary splits. For linear models, one-hot `Pclass` is often safer unless you truly believe "2nd vs 3rd" equals "1st vs 2nd" in effect.

---

## Handling Missing Values

Titanic's `Age` is missing for ~20% of passengers. Prosise imputes before encoding:

| Strategy | When |
|----------|------|
| **Median** (numeric) | Default, robust |
| **Mode** (categorical) | Most common category |
| **Indicator column** | `Age_missing=1` preserves "missingness" signal |
| **Model-based** | Predict age from other features (advanced) |

```python
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

df['Age'] = num_imputer.fit_transform(df[['Age']])
df['Embarked'] = cat_imputer.fit_transform(df[['Embarked']]).ravel()
```

> **Real-life analogy:** Missing age isn't "age zero" - it's "we don't know." Median imputation guesses typical; a missing flag says "passenger list was incomplete" (sometimes correlated with survival).

---

## ColumnTransformer: The Production Pattern

Different columns need different transforms - **never** fit on the full dataframe before splitting (leakage).

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Titanic-like sample
df = pd.DataFrame({
    'Pclass': [3, 1, 3, 2, 1],
    'Sex': ['male', 'female', 'female', 'male', 'female'],
    'Age': [22, 38, 26, np.nan, 35],
    'Fare': [7.25, 71.28, 7.92, 53.1, 31.0],
    'Embarked': ['S', 'C', 'S', 'Q', 'S'],
})
y = np.array([0, 1, 1, 0, 1])

numeric_features = ['Age', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore')),
        ]), categorical_features),
    ]
)

pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000)),
])

pipe.fit(df, y)
print(pipe.named_steps['prep'].get_feature_names_out())
```

**Milestone:** `Pipeline` + `ColumnTransformer` = what Prosise builds toward in Titanic ([Section 3.6](./section-06-case-study-titanic-survival.md)) and what you'll deploy in [Chapter 07](../chapter-07-operationalizing-models/README.md).

---

## Feature Engineering Beyond Encoding

Prosise extracts additional signals:

| Feature | Formula / logic | Why |
|---------|-----------------|-----|
| `FamilySize` | `SibSp + Parch + 1` | Solo travelers vs families |
| `IsAlone` | `FamilySize == 1` | Different survival dynamics |
| `Title` | Parse from `Name` | "Mr", "Mrs", "Master" encodes age/sex |
| `FarePerPerson` | `Fare / FamilySize` | Normalizes ticket cost |

```python
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
```

These become categorical (Title) or numeric (FamilySize) and flow through the same `ColumnTransformer`.

---

## High Cardinality: When One-Hot Explodes

`Name`, `Ticket`, `Cabin` on Titanic have hundreds of unique values. One-hot → sparse, wide, [overfitting](../../GLOSSARY.md#overfitting).

| Approach | Tradeoff |
|----------|----------|
| Drop column | Lose signal |
| Extract structure (Title, deck letter) | Prosise's approach |
| Target / frequency encoding | Leakage risk if done wrong |
| Hashing (`FeatureHasher`) | Collisions, fast |
| Embeddings | [Course 3](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/README.md) deep tabular |

For this chapter: **engineer**, don't one-hot 900 ticket strings.

---

## Scaling Interaction with Encoding

From [Section 2.2](../chapter-02-regression-models/section-02-linear-regression.md):

| Model | Scale numerics? | Encode categoricals? |
|-------|-----------------|----------------------|
| Logistic / SVM | Yes (`StandardScaler`) | One-hot |
| [Random forest](./section-05-tree-based-classifiers.md) | No | One-hot or label |
| [k-NN](../chapter-01-machine-learning/section-05-supervised-learning-k-nearest-neighbors.md) | **Yes** | One-hot |

Put scaling **only** on numeric branch of `ColumnTransformer` - never scale one-hot columns to 0.3, 0.7 nonsense.

---

## `handle_unknown='ignore'`

Production sees categories not in training - new embarkation code "X". `OneHotEncoder(handle_unknown='ignore')` outputs all zeros for that column group → model degrades gracefully instead of crashing.

---

## Common Mistakes

1. **Fit encoder on train+test** - leaks category frequencies
2. **Label-encode nominal for logistic regression** - invents order
3. **Forget imputation before one-hot** - NaN breaks encoder
4. **Scale entire matrix after get_dummies** - wrecks binary columns
5. **Drop intercept column incorrectly** - dummy trap confuses coefficients

---

## Self-Check

1. Why is one-hot preferred over label encoding for `Sex` in logistic regression?
2. What does `drop='first'` prevent?
3. Why impute `Age` with median instead of 0?
4. Name two Prosise Titanic features engineered from raw columns.

---

## References

- [Scikit-Learn: Preprocessing Categorical Features](https://scikit-learn.org/stable/chapters/preprocessing.html#encoding-categorical-features)
- [Scikit-Learn: ColumnTransformer](https://scikit-learn.org/stable/chapters/generated/sklearn.compose.ColumnTransformer.html)
- [Scikit-Learn: Imputation](https://scikit-learn.org/stable/chapters/impute.html)
- [Pandas: get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)
- Prosise, *Applied ML and AI for Engineers*, Ch. 3 - Titanic feature prep
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 3.3](./section-03-classification-accuracy-measures.md) | **Next:** [Section 3.5 - Tree-Based Classifiers](./section-05-tree-based-classifiers.md)



