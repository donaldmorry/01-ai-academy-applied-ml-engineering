# Section 3.5: Tree-Based Classifiers

> **Source:** Prosise, Ch. 3 - Decision trees & ensembles for classification  
> **Prerequisites:** [Section 3.4](./section-04-categorical-data-and-feature-encoding.md) | [Section 2.3](../chapter-02-regression-models/section-03-decision-trees-for-regression.md)  
> **Glossary:** [random forest](../../GLOSSARY.md#random-forest) | [gradient boosting](../../GLOSSARY.md#gradient-boosting) | [overfitting](../../GLOSSARY.md#overfitting)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Twenty Questions, But for Data

A **decision tree** classifies by asking yes/no questions about [features](../../GLOSSARY.md#feature). You met regression trees in [Section 2.3](../chapter-02-regression-models/section-03-decision-trees-for-regression.md); **classification trees** predict the **majority class** in each leaf instead of the mean.

Prosise uses trees on Titanic because they handle nonlinear rules like *"women in 1st class almost always survived"* without you hand-writing interaction terms.

> **Humorous analogy:** A decision tree is that friend who plays "20 Questions" - but cheats by memorizing the training set if you let them ask too many questions ([overfitting](../../GLOSSARY.md#overfitting)).

---

## How a Classification Tree Splits

At each node, pick the feature and threshold that best **separates classes**. Common criteria:

### Gini impurity

$$
G = 1 - \sum_{k=1}^{K} p_k^2
$$
> **Readable form:** Gini = 1 minus (sum of squared class proportions) at this node

| Node purity | $p_k$ | Gini |
|-------------|-------|------|
| Pure (all class A) | $p_A=1$ | 0 |
| 50/50 split | $p_A=p_B=0.5$ | 0.5 |

### Entropy (alternative)

$$
H = -\sum_{k=1}^{K} p_k \log_2(p_k)
$$
> **Readable form:** entropy = negative sum of (class proportion × log-base-2 of class proportion)

**Goal:** Choose split that **maximally reduces** impurity in child nodes.

```
                    [All passengers]
                    Gini = 0.48
                   /              \
         Sex=female?                Sex=male
         Gini=0.11                  Gini=0.31
        /        \                 /         \
    Pclass≤2?   ...            Age≤16?      ...
```

> **In plain English:** First split might be sex because it separates survivors best. Then class, then age - the tree discovers Prosise's historical narrative automatically.

---

## DecisionTreeClassifier in Scikit-Learn

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Minimal Titanic-like data
np.random.seed(42)
n = 600
df = pd.DataFrame({
    'pclass': np.random.choice([1, 2, 3], n),
    'sex': np.random.choice([0, 1], n),
    'age': np.random.uniform(5, 60, n),
})
log_odds = -1.0 - 0.7*(df['pclass']-1) + 1.5*df['sex'] - 0.02*(df['age']-25)
y = (np.random.rand(n) < 1/(1+np.exp(-log_odds))).astype(int)
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
print(f"Train accuracy: {tree.score(X_train, y_train):.3f}")
print(f"Test accuracy:  {tree.score(X_test, y_test):.3f}")

plt.figure(figsize=(16, 8))
plot_tree(tree, feature_names=X.columns, class_names=['Died', 'Survived'],
          filled=True, fontsize=9)
plt.show()
```

### Hyperparameters that control overfitting

| Parameter | Effect |
|-----------|--------|
| `max_depth` | Shallower → simpler, less overfit |
| `min_samples_leaf` | Require more samples per leaf |
| `min_samples_split` | Minimum samples to split a node |
| `max_leaf_nodes` | Cap total leaves |
| `ccp_alpha` | Cost-complexity pruning |

```python
# Grid search preview - full pattern in Chapter 02 Section 2.7
from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [3, 5, 8, None], 'min_samples_leaf': [1, 5, 20]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)
```

---

## Random Forest: Committee of Trees

A [random forest](../../GLOSSARY.md#random-forest) trains many trees on **bootstrap samples** and **random feature subsets**, then **votes**:

$$
\hat{y} = \text{mode}\left(\hat{y}_{\text{tree}_1}, \hat{y}_{\text{tree}_2}, \ldots, \hat{y}_{\text{tree}_B}\right)
$$
> **Readable form:** final prediction = most common vote among all B trees

For probabilities, average predicted probabilities across trees.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

# Feature importance (impurity decrease - interpret with caution)
for feat, imp in sorted(zip(X.columns, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.3f}")
```

| vs Single tree | Random forest |
|----------------|---------------|
| High variance | Averaging reduces variance |
| Interpretable plot | Feature importances only (approximate) |
| Fast to train one | Slower, parallelizable |

**Real-life analogy:** One tree is one opinionated uncle. Random forest is Thanksgiving dinner - democratic vote, harder to quote anyone specifically.

From [Section 2.4](../chapter-02-regression-models/section-04-ensemble-methods-random-forests-and-gradient-boosting.md): same bagging idea, different leaf prediction (class vote vs mean).

---

## Gradient Boosting: Fix Your Mistakes Sequentially

[Gradient boosting](../../GLOSSARY.md#gradient-boosting) builds trees **one at a time**, each focusing on **residual errors** (misclassified examples weighted higher):

$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})
$$
> **Readable form:** updated model = previous model + (learning rate × new small tree's correction)

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
)
gb.fit(X_train, y_train)
print(f"GB test accuracy: {gb.score(X_test, y_test):.3f}")
```

| Random forest | Gradient boosting |
|---------------|-------------------|
| Trees independent (parallel) | Trees sequential |
| Robust defaults | Often higher accuracy, needs tuning |
| Less overfit risk | Can overfit if `n_estimators` too high |

Prosise often finds boosting edges out on tabular [classification](../../GLOSSARY.md#classification) - same pattern as taxi fares in [Section 2.8](../chapter-02-regression-models/section-08-taxi-fare-prediction-end-to-end-project.md).

---

## Trees vs Logistic Regression on Titanic

| Criterion | [Logistic regression](./section-02-logistic-regression.md) | Tree ensembles |
|-----------|---------------------------------------------------|----------------|
| Interpretability | Coefficients, odds ratios | Partial dependence, SHAP (advanced) |
| Interactions | Manual (cross-terms) | Automatic |
| Mixed feature types | Needs encoding ([3.4](./section-04-categorical-data-and-feature-encoding.md)) | Handles well post-encoding |
| Linear boundaries | Native | Staircase boundaries |
| Probability calibration | Often better | May need `CalibratedClassifierCV` |

Full comparison in [Section 3.6](./section-06-case-study-titanic-survival.md).

---

## Class Imbalance with Trees

For fraud ([Section 3.7](./section-07-case-study-credit-card-fraud-detection.md)):

```python
rf_balanced = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',  # weight inversely proportional to frequency
    random_state=42,
    n_jobs=-1,
)
```

Boosting: `GradientBoostingClassifier` lacks `class_weight` - use `sample_weight` in `fit()` or switch to `HistGradientBoostingClassifier` / XGBoost / LightGBM in production.

Evaluate with [precision/recall/F1](./section-03-classification-accuracy-measures.md), not [accuracy](../../GLOSSARY.md#accuracy) alone.

---

## When Trees Shine - and When They Don't

### Shine

- Tabular data with mixed types and interactions (Titanic, customer churn)
- No need to scale features
- Medium-sized datasets (thousands to millions of rows)
- Quick strong baseline with random forest

### Struggle

| Issue | Why |
|-------|-----|
| High-dimensional sparse text | Use [Chapter 04](../chapter-04-text-classification/README.md) |
| Extrapolation beyond training range | Trees plateau at training extremes |
| Smooth global trends only | [Logistic](./section-02-logistic-regression.md) or linear may win |
| Individual pixel MNIST | Works but slow; k-NN/SVM/CNN better ([3.8](./section-08-case-study-mnist-digit-recognition.md)) |

---

## Pipelines with Tree Classifiers

Trees don't need scaling - simplify the preprocessor:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), ['Age', 'Fare']),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore')),
    ]), ['Sex', 'Embarked', 'Pclass']),
])

rf_pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42)),
])
```

---

## Common Mistakes

1. **Unlimited depth** - 100% train, 55% test
2. **Tuning on test accuracy** - use CV ([Section 2.7](../chapter-02-regression-models/section-07-model-comparison-and-cross-validation.md))
3. **Trusting feature importance blindly** - correlated features split importance
4. **Skipping encoding** for string columns
5. **Using accuracy on fraud** - see [3.7](./section-07-case-study-credit-card-fraud-detection.md)

---

## Self-Check

1. Gini 0 vs Gini 0.5 - which node is purer?
2. Why does random forest reduce overfitting vs one deep tree?
3. What's the difference between bagging and boosting?
4. Do random forests need `StandardScaler`?

---

## References

- [Scikit-Learn: DecisionTreeClassifier](https://scikit-learn.org/stable/chapters/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Scikit-Learn: RandomForestClassifier](https://scikit-learn.org/stable/chapters/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Scikit-Learn: GradientBoostingClassifier](https://scikit-learn.org/stable/chapters/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [Scikit-Learn: Ensembles](https://scikit-learn.org/stable/chapters/ensemble.html)
- Prosise, *Applied ML and AI for Engineers*, Ch. 3 - Tree classifiers
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFG8HA)
- [StatQuest: Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 3.4](./section-04-categorical-data-and-feature-encoding.md) | **Next:** [Section 3.6 - Titanic Survival](./section-06-case-study-titanic-survival.md)
