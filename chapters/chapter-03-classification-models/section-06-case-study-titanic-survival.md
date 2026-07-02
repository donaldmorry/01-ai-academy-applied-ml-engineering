# Section 3.6: Case Study - Titanic Survival

> **Source:** Prosise, Ch. 3 - "Predicting Titanic Survival" (full pipeline)  
> **Prerequisites:** Sections [3.1](./section-01-classification-fundamentals.md)-[3.5](./section-05-tree-based-classifiers.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [supervised learning](../../GLOSSARY.md#supervised-learning) | [cross-validation](../../GLOSSARY.md#cross-validation)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Kaggle Problem That Taught a Generation

On April 15, 1912, RMS Titanic struck an iceberg. Of 2,224 passengers and crew, 1,502 died. The question that launched a million data science careers:

> **Did a passenger survive?**

Prosise walks this dataset end-to-end - the same problem as the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic). You'll replicate his pipeline with modern Scikit-Learn patterns from [Section 3.4](./section-04-categorical-data-and-feature-encoding.md) and evaluate with [Section 3.3](./section-03-classification-accuracy-measures.md).

**Target:** `Survived` ∈ {0, 1} - binary [classification](../../GLOSSARY.md#classification).

> **Humorous analogy:** Titanic ML is the "Wonderwall" of data science - everyone has played it at least once at a meetup.

---

## Step 1: Load and Explore

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Kaggle: train.csv - https://www.kaggle.com/competitions/titanic/data
df = pd.read_csv('data/titanic/train.csv')

print(df.shape)          # (891, 12)
print(df['Survived'].value_counts(normalize=True))
# ~62% perished, ~38% survived - imbalanced but not extreme
print(df.isnull().sum())
```

| Column | Description | Missing? |
|--------|-------------|----------|
| `PassengerId` | ID | No - drop for modeling |
| `Survived` | Target | No |
| `Pclass` | Ticket class 1/2/3 | No |
| `Name` | Full name | No - extract Title |
| `Sex` | male/female | No |
| `Age` | Years | ~20% |
| `SibSp` | Siblings/spouses aboard | No |
| `Parch` | Parents/children aboard | No |
| `Ticket` | Ticket number | No - high cardinality |
| `Fare` | Ticket price | Few |
| `Cabin` | Cabin ID | ~77% - mostly drop |
| `Embarked` | S/C/Q port | 2 |

### EDA: What history already tells us

```python
sns.barplot(data=df, x='Sex', y='Survived', ci=None)
plt.title('Survival Rate by Sex')
plt.ylabel('Fraction Survived')
plt.show()

sns.barplot(data=df, x='Pclass', y='Survived', ci=None)
plt.title('Survival Rate by Class')
plt.show()
```

**Prosise's narrative (confirmed by data):** "Women and children first" + class privilege → `Sex`, `Pclass`, `Age` dominate. Your model should rediscover this without being told the history.

---

## Step 2: Feature Engineering

```python
def engineer_features(data):
    out = data.copy()
    out['Title'] = out['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
    out['Title'] = out['Title'].replace({
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
        'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare',
        'Rev': 'Rare', 'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare',
    })
    out['FamilySize'] = out['SibSp'] + out['Parch'] + 1
    out['IsAlone'] = (out['FamilySize'] == 1).astype(int)
    out['FarePerPerson'] = out['Fare'] / out['FamilySize']
    out['Deck'] = out['Cabin'].str[0]  # first letter or NaN
    return out

df = engineer_features(df)
```

| Feature | Rationale |
|---------|-----------|
| `Title` | "Master" ≈ boy; "Mrs"/"Miss" ≈ female adult |
| `FamilySize` | Large families had different evacuation dynamics |
| `IsAlone` | Solo travelers vs groups |
| `FarePerPerson` | Normalizes group tickets |
| `Deck` | Cabin letter correlates with ship location |

---

## Step 3: Define Features and Target

```python
TARGET = 'Survived'
DROP = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']

numeric_features = ['Age', 'Fare', 'FamilySize', 'FarePerPerson', 'SibSp', 'Parch']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'Deck']

X = df.drop(columns=DROP)
y = df[TARGET]
```

---

## Step 4: Preprocessing Pipeline

From [Section 3.4](./section-04-categorical-data-and-feature-encoding.md):

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore')),
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## Step 5: Baseline and Models

### Majority-class baseline

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)
print(f"Baseline accuracy: {accuracy_score(y_test, y_pred_base):.3f}")
# Expect ~0.62 - always predict "died"
```

### Logistic regression (Prosise's interpretable model)

```python
from sklearn.linear_model import LogisticRegression

log_pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=2000, random_state=42)),
])
log_pipe.fit(X_train, y_train)
y_pred_log = log_pipe.predict(X_test)

print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_log, target_names=['Died', 'Survived']))
print(confusion_matrix(y_test, y_pred_log))
```

### Random forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=300, max_depth=8,
                                   min_samples_leaf=4, random_state=42, n_jobs=-1)),
])
rf_pipe.fit(X_train, y_train)
y_pred_rf = rf_pipe.predict(X_test)
print(f"RF accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
```

### Gradient boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

gb_pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                       max_depth=3, random_state=42)),
])
gb_pipe.fit(X_train, y_train)
print(f"GB accuracy: {accuracy_score(y_test, gb_pipe.predict(X_test)):.3f}")
```

---

## Step 6: Cross-Validation (Don't Trust One Split)

From [Section 2.7](../chapter-02-regression-models/section-07-model-comparison-and-cross-validation.md):

```python
models = {
    'Logistic': log_pipe,
    'RandomForest': rf_pipe,
    'GradientBoosting': gb_pipe,
}

for name, pipe in models.items():
    scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"{name:18s}  CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

Typical Prosise-range results: **~78-84%** accuracy depending on features and split - beating baseline by 16-22 points.

---

## Step 7: Metrics Beyond Accuracy

Survivors are the **minority class** (~38%). Report precision/recall for class 1:

```python
from sklearn.metrics import roc_auc_score, RocCurveDisplay

y_proba = log_pipe.predict_proba(X_test)[:, 1]
print(f"Logistic ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title('Titanic - Logistic Regression ROC')
plt.show()
```

| Metric | Why for Titanic |
|--------|-----------------|
| **Recall (Survived)** | "Of actual survivors, how many did we find?" |
| **Precision (Survived)** | "When we predict survived, how often right?" |
| **F1** | Balance for minority class |
| **ROC-AUC** | Ranking quality across thresholds |

---

## Step 8: Interpret Logistic Coefficients

```python
lr = log_pipe.named_steps['clf']
feature_names = log_pipe.named_steps['prep'].get_feature_names_out()
coefs = pd.Series(lr.coef_[0], index=feature_names).sort_values()

print("Most negative (lower survival odds):")
print(coefs.head(5))
print("\nMost positive (higher survival odds):")
print(coefs.tail(5))
```

Expect positive weights on `Sex_female`, `Pclass_1`, `Title_Mrs` - aligning with history and Prosise's discussion.

> **In plain English:** Logistic regression gives you sentences for stakeholders: "Being female increased survival odds, holding class and age constant."

---

## Step 9: Feature Importance (Random Forest)

```python
rf = rf_pipe.named_steps['clf']
importances = pd.Series(
    rf.feature_importances_,
    index=rf_pipe.named_steps['prep'].get_feature_names_out()
).sort_values(ascending=False)

importances.head(10).plot(kind='barh', figsize=(8, 5), title='Top 10 RF Importances')
plt.tight_layout()
plt.show()
```

Trees won't give signed coefficients - importance ≠ causation.

---

## Step 10: Generate Kaggle Submission (Optional)

```python
test_df = pd.read_csv('data/titanic/test.csv')
test_df = engineer_features(test_df)
X_sub = test_df.drop(columns=[c for c in DROP if c in test_df.columns])

final_model = gb_pipe  # pick your CV winner
final_model.fit(X, y)  # retrain on ALL training data
predictions = final_model.predict(X_sub)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions,
})
submission.to_csv('submission.csv', index=False)
```

**Warning:** Refit on full data only **after** you've locked your pipeline choices via CV.

---

## Prosise vs Kaggle Leaderboard Reality

| Approach | Typical accuracy |
|----------|------------------|
| Majority baseline | ~62% |
| Logistic + basic features | ~78-80% |
| Prosise pipeline + trees | ~82-84% |
| Legendary feature engineering + stacking | 100% on public LB (overfit!) |

> **Humorous analogy:** Chasing 100% on Titanic's public leaderboard is like memorizing the answer key - impressive at the party, useless on the real test (holdout).

---

## Common Pitfalls on This Dataset

1. **Leakage via `Survived`-correlated imputation** - fit imputer inside pipeline on train only ✅
2. **Including `PassengerId`** - arbitrary ID, not predictive
3. **One-hot `Name`** - 891 unique strings → disaster
4. **Ignoring stratified split** - lucky split inflates scores
5. **Accuracy only** - report survivor recall ([Section 3.3](./section-03-classification-accuracy-measures.md))

---

## Connection to Chapter Arc

| Section | Titanic application |
|--------|---------------------|
| [3.2 Logistic](./section-02-logistic-regression.md) | Interpretable odds |
| [3.4 Encoding](./section-04-categorical-data-and-feature-encoding.md) | Sex, Embarked, Title |
| [3.5 Trees](./section-05-tree-based-classifiers.md) | RF/GB beat linear if interactions matter |
| [3.7 Fraud](./section-07-case-study-credit-card-fraud-detection.md) | Contrast: extreme imbalance |
| [3.8 MNIST](./section-08-case-study-mnist-digit-recognition.md) | Contrast: 10 classes, pixels |

---

## Self-Check

1. Why is majority baseline ~62%, not 50%?
2. Which two features would you expect highest importance?
3. Why fit `SimpleImputer` inside `Pipeline`?
4. Logistic vs RF - which gives audit-friendly coefficients?

---

## References

- [Kaggle: Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- [Titanic data dictionary](https://www.kaggle.com/competitions/titanic/data)
- [Scikit-Learn: Pipelines](https://scikit-learn.org/stable/chapters/compose.html)
- Prosise, *Applied ML and AI for Engineers*, Ch. 3 - Titanic
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 3.5](./section-05-tree-based-classifiers.md) | **Next:** [Section 3.7 - Credit Card Fraud](./section-07-case-study-credit-card-fraud-detection.md)