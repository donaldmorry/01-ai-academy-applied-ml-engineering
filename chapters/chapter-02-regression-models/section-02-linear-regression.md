# Section 2.2: Linear Regression

> **Source:** Prosise, Ch. 2 - "Linear Regression"  
> **Prerequisites:** [Section 2.1](./section-01-introduction-to-regression.md) | [Glossary: linear regression](../../GLOSSARY.md#linear-regression)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The High-School Equation That Powers Industry

Remember **y = m·x + b** from algebra? That is **simple linear regression** - one input feature, one output number, one straight line.

$$
\hat{y} = w_1 x + w_0
$$
> **Readable form:** predicted value = (slope × input) + intercept

| Symbol | Name | Meaning |
|--------|------|---------|
| $\hat{y}$ | y-hat | **Predicted** output (not the true value) |
| $x$ | feature | Input (years of experience, miles traveled, etc.) |
| $w_1$ | slope | How much $\hat{y}$ changes when $x$ increases by 1 |
| $w_0$ | intercept | Predicted $\hat{y}$ when $x = 0$ |

**Real-life analogy:** A consultant charges **60 USD per hour** (slope) plus a **500 USD** project setup fee (intercept). Ten hours of work:

$$
\hat{y} = 60 \times 10 + 500 = 1100 \text{ USD}
$$
> **Readable form:** 60 × 10 + 500 = **1100 dollars**

### Prosise's salary example (from the book)

For programmer income vs years of experience, Prosise fits:

$$
\hat{y} = 3984 + 60040 \times \text{years\_experience}
$$
> **Readable form:** predicted income = 3984 + 60040 × (years of experience)

If experience = 10 years: $\hat{y} = 3984 + 60040 \times 10 = 638{,}384$ - wait, Prosise's units are in the dataset's scale; his plotted example gives roughly **99,880 USD** when the feature is coded appropriately. The section: **always check units** (thousands vs dollars vs log-scaled).

---

## Simple vs Multiple Linear Regression

### Simple (one feature)

One $x$, one line. Easy to plot.

### Multiple (many features)

Real taxi fares depend on distance, passengers, time of day, and more:

$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3 + \cdots + w_n x_n
$$
> **Readable form:**  
> predicted fare = intercept  
> + (weight₁ × distance)  
> + (weight₂ × passenger count)  
> + (weight₃ × hour of day) + …

**Milestone:** This is still "linear" because the model is **linear in the weights** $w_j$, even if features are nonlinear transforms (e.g. $x_1 = \text{distance}^2$ later).

### Matrix form (used everywhere in [Course 3](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-02-linear-algebra/README.md))

Stack all examples into a design matrix $\mathbf{X}$ and all weights into $\mathbf{w}$:

$$
\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}
$$
> **Readable form:** vector of all predictions = (table of all features) × (column of weights)

For 3 training rows and 2 features (plus intercept column of ones):

```
     | 1  x₁₁  x₁₂ |   | w₀ |   | ŷ₁ |
X =  | 1  x₂₁  x₂₂ | × | w₁ | = | ŷ₂ |
     | 1  x₃₁  x₃₂ |   | w₂ |   | ŷ₃ |
```

---

## How Training Works: Ordinary Least Squares (OLS)

**Goal:** Find weights $w_0, w_1, \ldots$ that minimize **[MSE](../../GLOSSARY.md#mse)** (mean squared error).

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$
> **Readable form:**  
> MSE = (1/n) × [ (actual₁ − predicted₁)² + (actual₂ − predicted₂)² + … + (actualₙ − predictedₙ)² ]

Each term $(y_i - \hat{y}_i)$ is a **residual** - how far off prediction $i$ is.

| Residual sign | Meaning |
|---------------|---------|
| Positive | Model **under**-predicted (actual higher) |
| Negative | Model **over**-predicted |
| Near zero | Good fit for that point |

> **In plain English:** Imagine hanging string lights along a path. MSE measures the average squared vertical gap between each light and the string. OLS adjusts the string's angle and height until those gaps are as small as possible **on average**.

> **In plain English (again):** Squaring errors means a **20-dollar mistake hurts four times more than a 10-dollar mistake** (because 20² = 400 vs 10² = 100). That matters when large errors are unacceptable - see [Section 2.6](./section-06-regression-accuracy-measures.md) for [RMSE](../../GLOSSARY.md#rmse) vs [MAE](../../GLOSSARY.md#mae).

### Closed-form solution (no iteration needed for basic OLS)

For standard linear regression, calculus gives a direct formula:

$$
\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$
> **Readable form:** weights = (feature-table transpose × feature-table)⁻¹ × feature-table transpose × targets

Scikit-Learn's `LinearRegression` uses a stable numerical version of this (or SVD). You rarely code the inverse by hand - but **knowing OLS minimizes squared error** is the milestone.

### Iterative view (connects to neural networks later)

Some variants adjust $w$ step by step using gradients - same idea as [gradient descent](../../GLOSSARY.md#gradient-descent) in [Course 3](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-04-numerical-computation/README.md). OLS for vanilla linear regression is the "one-shot" special case.

---

## Code: Build, Train, Evaluate

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# --- Synthetic: years of experience → salary (thousands USD) ---
np.random.seed(42)
years = np.random.uniform(0, 20, 200)
salary = 40 + 8 * years + np.random.normal(0, 5, 200)  # true: 40 + 8*x + noise

X = years.reshape(-1, 1)  # sklearn expects 2D: (n_samples, n_features)
y = salary

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Train ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Learned parameters ---
print(f"Intercept w₀: {model.intercept_:.2f}")
print(f"Slope w₁:     {model.coef_[0]:.2f}")  # expect ~8

# --- Predict & metrics ---
y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}  (typical error in 'thousands USD' units)")
print(f"MAE:  {mae:.2f}")
print(f"R²:   {r2:.3f}")
```

### Visualize the fit

```python
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, alpha=0.6, label='Actual (test)')
x_line = np.linspace(0, 20, 100).reshape(-1, 1)
plt.plot(x_line, model.predict(x_line), 'r-', linewidth=2, label='OLS fit')
plt.xlabel('Years of Experience')
plt.ylabel('Salary (thousands USD)')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Interpreting Coefficients (Why CFOs Love Linear Models)

Each weight $w_j$ answers:

> *"If feature $x_j$ increases by 1 unit, holding all other features fixed, how much does $\hat{y}$ change?"*

| Coefficient | In our toy example | Business sentence |
|-------------|-------------------|-------------------|
| $w_0 = 40$ | Intercept | Base salary ~40k at 0 years |
| $w_1 = 8$ | Slope | Each extra year adds ~8k |

A [random forest](../../GLOSSARY.md#random-forest) might predict better but speaks in riddles. Linear regression gives **auditable sentences** for regulators and executives.

<a id="multicollinearity"></a>

**Caution:** Interpretation assumes features are **independent enough**. If `rooms` and `square_feet` move together ([multicollinearity](#multicollinearity)), individual coefficients become unstable even if predictions stay good.

---

## Parametric Models and Normalization

Prosise emphasizes: [linear regression](../../GLOSSARY.md#linear-regression) is **parametric** - it fits a fixed equation shape with learned parameters. [k-NN](../../GLOSSARY.md#k-nearest-neighbors) is **nonparametric** - the "model" is the entire dataset.

**Practical rule:** Parametric models (linear, logistic, SVM) often need **feature scaling** when features live on different scales (0-1 vs 0-1,000,000). Tree models do not. Full normalization in [Chapter 05](../chapter-05-support-vector-machines/README.md).

---

## When Linear Regression Works - and When It Doesn't

### Works well

- Scatter plot looks roughly like a line (or plane in 3D)
- Additive effects: distance + time + tolls each add roughly constant dollars
- You need speed, transparency, or a legal audit trail
- Baseline before trying complex models ([Section 2.1](./section-01-introduction-to-regression.md))

### Struggles

| Problem | Symptom | Fix |
|---------|---------|-----|
| Curved relationship | Residuals form a U-shape | Polynomial features, splines, or trees |
| Outliers | One point pulls the line | Ridge/Lasso, robust regression, or remove outlier |
| Multicollinearity | Huge coefficients, unstable signs | Drop redundant feature, Ridge, Lasso |
| Extrapolation | Wild predictions outside training range | Don't trust; trees clamp to training range |

### Polynomial regression (still linear in weights!)

$$
\hat{y} = w_0 + w_1 x + w_2 x^2
$$
> **Readable form:** predicted y = intercept + linear term + squared term

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)
poly_model.fit(X_train, y_train)
print(f"Poly R² on test: {r2_score(y_test, poly_model.predict(X_test)):.3f}")
```

---

## Ridge and Lasso: Regularized Linear Regression

When outliers or redundant features poison OLS, add a **penalty** on large weights - [regularization](../../GLOSSARY.md#regularization).

### Ridge (L2)

$$
\text{Loss} = \text{MSE} + \lambda \sum_j w_j^2
$$
> **Readable form:** error + lambda × (sum of squared weights)

Shrinks all coefficients toward zero; keeps all features.

### Lasso (L1)

$$
\text{Loss} = \text{MSE} + \lambda \sum_j |w_j|
$$
> **Readable form:** error + lambda × (sum of absolute weights)

Can drive some $w_j$ **exactly to zero** - automatic feature selection.

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=10000))

for name, m in [('Ridge', ridge), ('Lasso', lasso)]:
    m.fit(X_train, y_train)
    print(f"{name} RMSE: {np.sqrt(mean_squared_error(y_test, m.predict(X_test))):.2f}")
```

> **Humorous analogy:** OLS is an intern who chases every outlier. Ridge is a calm manager. Lasso is Marie Kondo - "this feature doesn't spark joy" *delete*.

Prosise notes **Lasso helps with multicollinearity** by zeroing redundant columns (e.g. `rooms` vs `square_feet` in housing data).

---

## Exploring High-Dimensional Data Before Fitting

You cannot plot 50 features at once. Prosise suggests:

### Pair plots (Seaborn)

```python
import seaborn as sns

# Iris example from Chapter 01
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]
sns.pairplot(df, hue='species', diag_kind='hist')
plt.show()
```

The diagonal histograms also reveal **class balance** - important for classification in [Chapter 03](../chapter-03-classification-models/README.md).

### PCA preview ([Chapter 06](../chapter-06-principal-component-analysis/README.md))

Reduce 90% of dimensions while keeping ~90% of variance - then plot 2D. Not magic; eigenmath.

---

## Common Mistakes (Read This Before the Lab)

1. **Fitting on test data** - never call `fit` on your test set
2. **Ignoring units** - mixing cents and dollars destroys coefficients
3. **Trusting $R^2$ alone** - high $R^2$ with patterned residuals = wrong model shape
4. **Extrapolating** - linear line extended to 500 miles for taxi data is nonsense
5. **Skipping baseline** - always compare to mean predictor ([Section 2.1](./section-01-introduction-to-regression.md))

---

## Self-Check

1. Write MSE in words (no symbols).
2. If $R^2 = 0$, how does your model compare to always predicting the mean?
3. Why does Lasso set some weights to exactly zero but Ridge usually does not?
4. Is polynomial regression (degree 2) still "linear regression" in the sklearn sense?

---

## References

- [Scikit-Learn LinearRegression](https://scikit-learn.org/stable/chapters/generated/sklearn.linear_model.LinearRegression.html)
- [Scikit-Learn Ordinary Least Squares](https://scikit-learn.org/stable/chapters/linear_model.html#ordinary-least-squares)
- [Scikit-Learn Ridge / Lasso](https://scikit-learn.org/stable/chapters/linear_model.html#ridge-regression)
- Prosise, *Applied ML and AI for Engineers*, Ch. 2 - Linear Regression
- [3Blue1Brown: Linear transformations](https://www.youtube.com/watch?v=kYB8IZ-a5VA)
- [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)
- [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Previous:** [Section 2.1](./section-01-introduction-to-regression.md) | **Next:** [Section 2.3 - Decision Trees](./section-03-decision-trees-for-regression.md)
