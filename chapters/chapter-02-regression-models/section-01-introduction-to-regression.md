# Section 2.1: Introduction to Regression

> **Source:** Prosise, Ch. 2 - Opening & regression framing  
> **Prerequisites:** [Chapter 01](../chapter-01-machine-learning/README.md) | [Glossary: regression](../../GLOSSARY.md#regression)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## What Is Regression?

In [Chapter 01](../chapter-01-machine-learning/README.md), you learned [supervised learning](../../GLOSSARY.md#supervised-learning) - predicting outputs from inputs. When that output is a **number on a continuous scale**, the task is called **[regression](../../GLOSSARY.md#regression)**.

| Task type | Output example | Question |
|-----------|---------------|----------|
| [Classification](../../GLOSSARY.md#classification) | "Spam" or "Ham" | *Which category?* |
| **Regression** | 18.47 USD | *How much? How many? How long?* |

**Real-life analogy:** A weather app doesn't just say "rain" or "no rain" ([classification](../../GLOSSARY.md#classification)). It says **"72°F at 3 PM"** - that's regression. Your fitness tracker estimating **calories burned**? Regression. Uber predicting **ETA in minutes**? Regression.

Prosise opens Chapter 2 with exactly this framing: engineers rarely build "AI" in the abstract - they build systems that answer numeric questions for a business.

---

## **Milestone Idea: Regression = Learning a Number-Line Mapping**

> **In plain English:** You show the computer thousands of examples like *"5 miles → 14 USD fare"* and *"12 miles → 28 USD fare."* It learns the pattern and guesses fares for trips it has never seen.

Formally, given [features](../../GLOSSARY.md#feature) $\mathbf{x} = (x_1, x_2, \ldots, x_n)$, regression learns a function:

$$
\hat{y} = f(\mathbf{x})
$$
> **Readable form:** predicted value = some function of all input features

where $\hat{y}$ is the **predicted** value and $y$ is the true [label](../../GLOSSARY.md#label).

The hat on $\hat{y}$ matters: it signals "this is our guess," not ground truth. Confusing $\hat{y}$ and $y$ is like reporting your weather-app forecast as if it were a thermometer reading.

---

## Regression vs Classification - Don't Mix Them Up

Imagine a hospital system:

- **Classification:** "Does this scan show cancer?" → Yes/No
- **Regression:** "How many days until this patient is discharged?" → 4.2 days

Using classification code for a regression problem (or vice versa) is like using a thermometer to weigh flour - wrong tool, confusing results.

| Signal you have regression | Signal you have classification |
|---------------------------|-------------------------------|
| Target column is `float` (fare, temperature) | Target is `str` or small integer codes |
| `sklearn` metrics: MAE, RMSE, R² | `accuracy`, `f1`, `log_loss` |
| Loss cares about distance from true number | Loss cares about wrong category |

```python
# Regression target: continuous
y_train = [14.50, 22.10, 8.75, 31.00]  # taxi fares in USD

# Classification target: discrete
y_train = ['setosa', 'versicolor', 'virginica']  # iris species
```

### Ordinal data trap

Sometimes categories look like numbers: shirt sizes 1 = S, 2 = M, 3 = L. **Don't regress on these** unless the spacing is meaningful (2 isn't "twice as large" as 1). Use classification or ordinal-specific methods instead.

> **Humorous analogy:** Predicting "size 2.7 shirts" from customer height is how you end up manufacturing clothing for fictional people.

---

## The Dumbest Useful Model: Predict the Mean

Before any fancy algorithm, establish a **baseline** - the simplest [model](../../GLOSSARY.md#model) that still counts as "trying."

**Strategy:** Always predict the average of all training targets.

$$
\hat{y}_{\text{baseline}} = \bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i
$$
> **Readable form:** baseline prediction = (sum of all training targets) divided by (number of training examples)

```python
import numpy as np
from sklearn.metrics import mean_absolute_error

y_train = np.array([12.0, 15.0, 18.0, 22.0, 25.0])
y_test  = np.array([14.0, 20.0, 30.0])

baseline_prediction = y_train.mean()  # 18.4
y_pred_baseline = np.full_like(y_test, baseline_prediction, dtype=float)

print(f"Baseline MAE: {mean_absolute_error(y_test, y_pred_baseline):.2f} USD")
# If your fancy model can't beat this, something is very wrong.
```

> **In plain English:** If every NYC taxi ride costs roughly 18 USD on average, always guessing "18 dollars" isn't brilliant - but it's your floor. Any model worse than this is actively harmful.

**Every regression project in this chapter starts here.** If [random forest](../../GLOSSARY.md#random-forest) can't beat "always guess the mean," debug your data before tuning [hyperparameters](../../GLOSSARY.md#hyperparameter).

### Why the mean baseline is special

The mean minimizes [MSE](../../GLOSSARY.md#mse) among all constant predictors - a fact we'll reuse in [Section 2.6](./section-06-regression-accuracy-measures.md) when interpreting $R^2$.

---

## What Makes a Good Regression Dataset?

Prosise's taxi fare example checks these boxes:

| Requirement | Taxi example | Bad counterexample |
|-------------|-------------|---------------------|
| Numeric target | `fare_amount` in USD | "cheap/medium/expensive" labels |
| Enough rows | Millions of TLC trips | 12 rows from a spreadsheet |
| Relevant features | distance, time, passengers | only car color |
| Clean outliers | remove 500 USD typos | train on GPS glitches |
| Train/test split | hold out unseen months | test on same rows you trained on |

```python
import pandas as pd

# Quick sanity check Prosise recommends before modeling
df = pd.read_csv('data/taxi-fares.csv')
print(df['fare_amount'].describe())
print(df[['trip_distance', 'fare_amount']].corr())
# If correlation is near zero, your features may not help - investigate first.
```

---

## Algorithms You'll Master in This Chapter

Prosise introduces a progression from simple to powerful:

```
Mean baseline  →  Linear Regression  →  Decision Trees
       →  Random Forest  →  Gradient Boosting  →  SVR
```

| Algorithm | Analogy | Best when |
|-----------|---------|-----------|
| [Linear regression](../../GLOSSARY.md#linear-regression) | Drawing a straight ramp through data | Relationships are roughly linear |
| Decision tree | Twenty Questions game | Nonlinear, interpretable splits |
| [Random forest](../../GLOSSARY.md#random-forest) | Committee of trees voting | Tabular data, robust default |
| [Gradient boosting](../../GLOSSARY.md#gradient-boosting) | Student fixing each mistake on the next test | Highest accuracy, needs tuning |
| SVR | Rigid tube around predictions | High dimensions, kernel tricks |

You already know [k-NN](../../GLOSSARY.md#k-nearest-neighbors) from Chapter 01 - it can regress too (average of neighbors' values), but Prosise shows that dedicated regression algorithms usually win on structured data.

### k-NN regression in one snippet

```python
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5)
# knn.fit(X_train, y_train)  # prediction = mean of 5 nearest training fares
```

k-NN is nonparametric and slow at scale - fine for tiny demos, rarely the production choice for millions of taxi rows.

---

## Parametric vs Nonparametric (Quick Preview)

- **Parametric** ([linear regression](../../GLOSSARY.md#linear-regression)): Fits a fixed equation with learned parameters $w_0, w_1, \ldots$
- **Nonparametric** (k-NN, trees): Shape of model grows with data

> **Humorous analogy:** Parametric is like ordering from a fixed menu ("I'll have the line equation, extra slope"). Nonparametric is like an all-you-can-eat buffet that expands every time your grandma brings a new dish.

This matters because parametric models often need **normalized** [features](../../GLOSSARY.md#feature) - we'll handle that in [Section 2.7](./section-07-model-comparison-and-cross-validation.md). Tree ensembles mostly don't care about scale.

---

## The NYC Taxi Fare Story (Chapter Capstone)

Throughout this chapter, we build toward predicting **NYC taxi fares** from TLC (Taxi & Limousine Commission) data:

- **Target:** `fare_amount` (USD) - continuous → regression
- **Features:** trip distance, passenger count, pickup time, location zones
- **Business value:** Power a fare estimator in a taxi company's mobile app

This is the same project Prosise uses in Chapter 2 and revisits when deploying via ONNX in [Chapter 07](../chapter-07-operationalizing-models/README.md).

### Prosise's narrative arc

1. Load messy real-world CSVs (not toy Iris)
2. Clean impossible values (negative fares, 200-mile glitches)
3. Engineer time features (rush hour, weekend)
4. Compare algorithms with [cross-validation](../../GLOSSARY.md#cross-validation)
5. Pick a winner and explain error in **dollars**, not just $R^2$

---

## When to Use Regression

| ✅ Good fit | ❌ Poor fit |
|-----------|-----------|
| Predicting prices, temperatures, demand | Predicting yes/no outcomes → use classification |
| Enough labeled historical data | No numeric target column |
| Errors can be measured in USD/degrees | Categories disguised as numbers (1=small, 2=medium, 3=large) |
| Stakeholders ask "how much?" | Stakeholders ask "which type?" |

---

## End-to-End Workflow Preview

You'll execute this pipeline repeatedly through Chapter 02:

```
1. Define business question (predict fare in USD)
2. Explore & clean data
3. Split train / test (never leak test into training)
4. Fit mean baseline → record MAE
5. Try candidate models
6. Compare with CV + metrics (Section 2.6-2.7)
7. Analyze residuals
8. Write recommendation memo (Lab 02)
```

> **In plain English (again):** Regression isn't one algorithm - it's a **problem type** with a standard engineering workflow. The algorithms are interchangeable tools in step 5.

---

## Common Mistakes (Catch These Early)

1. **Treating regression as classification** - using `accuracy` on continuous fares
2. **Skipping the baseline** - celebrating RMSE = 5 USD when the mean predictor already gets 4.8 USD
3. **Data leakage** - computing `fare_mean` on the full dataset including test rows
4. **Ignoring units** - mixing cents and dollars (see [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md))
5. **Fake continuity** - regressing on zip codes as if 10001 + 10002 = 20003

---

## Self-Check

1. Is "predicting tomorrow's stock price" regression or classification?  
   *Regression - price is continuous (even if noisy).*
2. Why must you beat the mean baseline before celebrating?  
   *A model worse than the mean is worse than doing nothing.*
3. What's the difference between $\hat{y}$ and $y$?  
   *$\hat{y}$ is prediction; $y$ is actual observed value.*
4. Why does Prosise start with taxi fares instead of Iris?  
   *Real business data has missing values, outliers, and messy CSVs - closer to production.*

---

## References

- [Scikit-Learn Regression Guide](https://scikit-learn.org/stable/supervised_learning.html#regression)
- [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- Prosise, *Applied ML and AI for Engineers*, Ch. 2 (O'Reilly)
- [StatQuest: Linear Regression](https://www.youtube.com/watch?v=PaFPbb66DxQ) - visual intuition
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Next:** [Section 2.2 - Linear Regression](./section-02-linear-regression.md)
