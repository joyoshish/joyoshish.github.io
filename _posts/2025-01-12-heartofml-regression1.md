---
layout: post
title: "Linear Regression Explained: From Normal Equations to Residual Diagnostics"
date: 2025-01-12
description: Heart of ML — Regression E01
tags: ml ai regression
categories: machine-learning heart-of-ml
thumbnail: assets/img/regression01_banner.png
math: true
giscus_comments: true
related_posts: true
chart:
  plotly: true
toc:
  beginning: true
---



## Introduction: Drawing the First Line in the Sand

> *“It is a capital mistake to theorize before one has data.”*
> — Sherlock Holmes (Arthur Conan Doyle)

In the late 18th century, while mapping the orbits of planets, astronomers kept running into a pesky problem: the predictions and the observations just wouldn’t match. Johannes Kepler, Carl Friedrich Gauss, and Adrien-Marie Legendre all wrestled with these deviations. But they didn’t give up. Instead, they embraced the imperfections of the data. Their solution? A straight line that minimized the total "error." And just like that, the seeds of **regression** were sown.

Regression isn’t just one of the oldest tools in data analysis—it’s a gateway. Whether you're studying the effect of study hours on GPA, predicting housing prices, or quantifying how customer age affects purchasing behavior, **linear regression is the first algorithm we turn to**. Not because it’s flashy, but because it’s interpretable, grounded, and surprisingly powerful.

This blog post begins our journey into supervised learning. In this first episode, we’ll unravel the core principles behind **Ordinary Least Squares (OLS) Linear Regression**, understand its geometry, derive its solution, examine its assumptions, and explore how we diagnose model fit.

Here’s what we’ll cover:

* The math and motivation behind the linear model
* How to derive and interpret the normal equation
* Visual and geometric intuition
* Key assumptions for reliable inference
* Residual analysis and influence diagnostics

By the end, you won’t just know how to fit a line to data—you’ll know how to trust it, critique it, and use it wisely in real-world data science projects.

---



## Problem Setup: Modeling $$ \mathbf{y = Xw + \varepsilon} $$

Before we can learn from data, we need a mathematical language to describe it. In supervised learning, this typically means we are given **inputs** and **outputs**, and our goal is to **model the relationship** between them.

Linear regression does this by assuming a **linear relationship** between the input features and the output variable.

---

### What Are We Trying to Learn?

Suppose we have a dataset with:

* $$ n $$ observations (rows)
* $$ p $$ input features or predictors (columns)

Our goal is to **predict** an output $$ y $$ given inputs $$ x_1, x_2, \ldots, x_p $$.

In matrix form, we write this relationship as:

$$
\mathbf{y} = \mathbf{Xw} + \boldsymbol{\varepsilon}
$$

Where:

* $$ \mathbf{y} \in \mathbb{R}^{n \times 1} $$ is the vector of observed outputs
* $$ \mathbf{X} \in \mathbb{R}^{n \times p} $$ is the design matrix of input features
* $$ \mathbf{w} \in \mathbb{R}^{p \times 1} $$ is the vector of model coefficients (weights)
* $$ \boldsymbol{\varepsilon} \in \mathbb{R}^{n \times 1} $$ is the error or noise term

We are trying to **learn** the weights $$ \mathbf{w} $$ such that the predicted output $$ \mathbf{Xw} $$ is as close as possible to the true observed output $$ \mathbf{y} $$.

---

### Intuition: What Does It Mean Geometrically?

Think of each row of $$ \mathbf{X} $$ as a point in $$ \mathbb{R}^p $$. What linear regression does is find the vector $$ \mathbf{w} $$ such that the **linear combination** of features best approximates the output $$ \mathbf{y} $$.

We can also visualize this in two dimensions:

* Imagine you're plotting hours studied (x-axis) vs. exam score (y-axis).
* Each point is a student.
* Linear regression draws the **best fitting straight line** through those points.

---

### A Simple Numerical Example

Suppose we have 3 data points and we want to fit a straight line:

<div style="overflow-x: auto;">
  <table class="simple-table">
    <thead>
      <tr>
        <th>Hours Studied (x)</th>
        <th>Exam Score (y)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>2</td>
      </tr>
      <tr>
        <td>2</td>
        <td>3</td>
      </tr>
      <tr>
        <td>3</td>
        <td>5</td>
      </tr>
    </tbody>
  </table>
</div>

Let’s model this as:

$$
y = w_0 + w_1 x + \varepsilon
$$

In matrix form, this becomes:

$$
\mathbf{X} = \begin{bmatrix}
1 & 1 \\
1 & 2 \\
1 & 3
\end{bmatrix}, \quad
\mathbf{y} = \begin{bmatrix}
2 \\
3 \\
5
\end{bmatrix}
$$

We prepend a column of ones to $$ \mathbf{X} $$ to represent the intercept term $$ w_0 $$. Now the linear model becomes:

$$
\mathbf{y} = \mathbf{Xw} + \boldsymbol{\varepsilon}
$$

---

### Visualizing the Data

We can visualize this dataset and the best-fit line using the following code:

```python
import numpy as np
import matplotlib.pyplot as plt

# Input features (with intercept)
X = np.array([[1, 1],
              [1, 2],
              [1, 3]])
y = np.array([2, 3, 5])

# Compute weights using the normal equation
w = np.linalg.inv(X.T @ X) @ X.T @ y

# Line for plotting
x_vals = np.linspace(0, 4, 100)
y_vals = w[0] + w[1] * x_vals

# Plot
plt.figure(figsize=(6, 4))
plt.scatter(X[:, 1], y, color='blue', label='Data Points')
plt.plot(x_vals, y_vals, color='red', label='Fitted Line')
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

This code will plot the data points and the linear regression line that minimizes the total squared error.

<iframe src="/assets/plotly/normal_equation_fit.html" width="700" height="450" frameborder="0"></iframe>


---

### What Are We Minimizing?

In linear regression, we **don’t** assume our predictions are perfect. Instead, we model the residuals as:

$$
\boldsymbol{\varepsilon} = \mathbf{y} - \mathbf{Xw}
$$

These are the errors our model makes on the training data. We choose $$ \mathbf{w} $$ such that the total **squared error** is minimized:

$$
\min_{\mathbf{w}} \left\| \mathbf{y} - \mathbf{Xw} \right\|^2
$$

This minimization problem leads us directly to the **normal equation**, which we’ll derive in the next section.

---

### Practical Implications

* Always include an intercept term (via a column of ones in $$ \mathbf{X} $$), unless you're explicitly modeling through the origin.
* Scaling your features is advisable, especially when you're applying optimization methods like gradient descent or adding regularization.
* This matrix-based formulation generalizes naturally to multivariate regression where $$ p > 1 $$.
* If $$ \mathbf{X}^\top \mathbf{X} $$ is not invertible, it typically indicates **multicollinearity** — that is, redundant or highly correlated features.

--- 



## Deriving the Normal Equation: $$ \mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y} $$



We now formalize how to compute the **best-fit weights** $$ \mathbf{w} $$ that minimize the squared error between predictions $$ \mathbf{Xw} $$ and actual outputs $$ \mathbf{y} $$.

---

### The Optimization Objective

From the previous section, recall that our objective is to minimize the sum of squared residuals:

$$
\mathcal{L}(\mathbf{w}) = \left\| \mathbf{y} - \mathbf{Xw} \right\|^2 = (\mathbf{y} - \mathbf{Xw})^\top (\mathbf{y} - \mathbf{Xw})
$$

This is a **convex quadratic function**, which means it has a unique global minimum.

---

### Step-by-Step Derivation

We expand the loss function:

$$
\mathcal{L}(\mathbf{w}) = \mathbf{y}^\top \mathbf{y} - 2\mathbf{y}^\top \mathbf{Xw} + \mathbf{w}^\top \mathbf{X}^\top \mathbf{Xw}
$$

To find the optimal $$ \mathbf{w} $$, we take the gradient of $$ \mathcal{L} $$ with respect to $$ \mathbf{w} $$ and set it to zero:

$$
\nabla_\mathbf{w} \mathcal{L} = -2\mathbf{X}^\top \mathbf{y} + 2\mathbf{X}^\top \mathbf{Xw} = 0
$$

Solving for $$ \mathbf{w} $$:

$$
\mathbf{X}^\top \mathbf{Xw} = \mathbf{X}^\top \mathbf{y}
$$

Assuming $$ \mathbf{X}^\top \mathbf{X} $$ is invertible:

$$
\mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$

This closed-form solution is known as the **normal equation**.

---

### Geometric Interpretation

The predicted vector $$ \hat{\mathbf{y}} = \mathbf{Xw} $$ is the **orthogonal projection** of $$ \mathbf{y} $$ onto the column space of $$ \mathbf{X} $$.

That is, the error vector $$ \mathbf{y} - \mathbf{Xw} $$ is orthogonal to every column of $$ \mathbf{X} $$:

$$
\mathbf{X}^\top (\mathbf{y} - \mathbf{Xw}) = 0
$$

This is why solving the normal equation yields the "closest" approximation of $$ \mathbf{y} $$ using linear combinations of $$ \mathbf{X} $$'s columns.

---

### Applying the Normal Equation (Revisit Our Example)

Let's compute $$ \mathbf{w} $$ using the 3-point dataset we saw earlier:

$$
\mathbf{X} = \begin{bmatrix}
1 & 1 \\
1 & 2 \\
1 & 3
\end{bmatrix}, \quad
\mathbf{y} = \begin{bmatrix}
2 \\
3 \\
5
\end{bmatrix}
$$

Compute:

1. $$ \mathbf{X}^\top \mathbf{X} = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix} $$
2. $$ \mathbf{X}^\top \mathbf{y} = \begin{bmatrix} 10 \\ 23 \end{bmatrix} $$
3. Invert $$ \mathbf{X}^\top \mathbf{X} $$ and multiply with $$ \mathbf{X}^\top \mathbf{y} $$:

$$
\mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
= \begin{bmatrix} 1 \\ 1.5 \end{bmatrix}
$$

So the fitted model is:

$$
\hat{y} = 1 + 1.5x
$$

---


### When the Normal Equation May Fail

The normal equation relies on $$ \mathbf{X}^\top \mathbf{X} $$ being **invertible**. However, it may not be invertible if:

* You have **perfect multicollinearity** (some columns are linear combinations of others)
* You have **more features than data points** ($$ p > n $$)

In such cases, we must resort to:

* **Pseudoinverse** (Moore-Penrose inverse)
* **Regularization** (Ridge regression)
* **Numerical optimization** (Gradient Descent)

---

## Geometric Interpretation: Projecting $$\mathbf{y}$$ onto the Column Space of $$\mathbf{X}$$

One of the most beautiful ways to understand linear regression is through **geometry**.

When we solve the linear model:

$$
\mathbf{y} = \mathbf{Xw} + \boldsymbol{\varepsilon}
$$

we are looking for a **vector** $$\mathbf{Xw}$$ that lies in the **column space** of $$\mathbf{X}$$ and is **as close as possible** to the observed response $$\mathbf{y}$$.

---

### The Column Space: A Subspace of $$\mathbb{R}^n$$

The matrix $$\mathbf{X} \in \mathbb{R}^{n \times p}$$ consists of $$p$$ column vectors in $$\mathbb{R}^n$$. Any linear combination of these columns — i.e., any $$\mathbf{Xw}$$ — lies in a subspace of $$\mathbb{R}^n$$ called the **column space of $$\mathbf{X}$$**.

We cannot reach all of $$\mathbb{R}^n$$ using $$\mathbf{Xw}$$ — only the part that spans its columns.

---

### What Linear Regression Does

Linear regression finds the **projection** of $$\mathbf{y}$$ onto the column space of $$\mathbf{X}$$. That is:

$$
\hat{\mathbf{y}} = \mathbf{Xw}
$$

is the vector in the column space of $$\mathbf{X}$$ that is closest (in Euclidean distance) to the actual target $$\mathbf{y}$$.

The residual vector $$\boldsymbol{\varepsilon} = \mathbf{y} - \hat{\mathbf{y}}$$ is **orthogonal** to the column space of $$\mathbf{X}$$:

$$
\mathbf{X}^\top (\mathbf{y} - \mathbf{Xw}) = 0
$$

---

### Visual Example in 3D

Let's say:

* $$\mathbf{y} \in \mathbb{R}^3$$
* $$\mathbf{X}$$ has 2 linearly independent columns
* The column space of $$\mathbf{X}$$ is a **plane** in 3D

Then:

* $$\hat{\mathbf{y}} = \mathbf{Xw}$$ lies in that plane
* $$\mathbf{y} - \hat{\mathbf{y}}$$ is the **perpendicular dropped from $$\mathbf{y}$$ onto the plane**

This is exactly like dropping a perpendicular from a point to a plane in geometry — the foot of that perpendicular is your fitted value $$\hat{\mathbf{y}}$$.


```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a 2D plane via two column vectors (basis)
x1 = np.array([1, 0, 0])
x2 = np.array([0, 1, 0])
X = np.column_stack((x1, x2))

# Target vector y (off the plane)
y = np.array([1, 2, 3])

# Projection of y onto the column space of X
X_pseudo_inv = np.linalg.pinv(X)
w = X_pseudo_inv @ y
y_hat = X @ w
residual = y - y_hat

# Plot
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0, 0, 0, *x1, color='blue', label='x1')
ax.quiver(0, 0, 0, *x2, color='green', label='x2')
ax.quiver(0, 0, 0, *y, color='black', label='y', linestyle='dashed')
ax.quiver(0, 0, 0, *y_hat, color='red', label='Projection $\\hat{y}$')
ax.quiver(*y_hat, *residual, color='orange', label='Residual')

ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_zlim(0, 4)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Projection of $\\mathbf{y}$ onto Column Space of $\\mathbf{X}$')
ax.legend()
plt.tight_layout()
plt.show()
```

<iframe src="/assets/plotly/geometric_projection.html" width="700" height="500" frameborder="0"></iframe>


---

## Gauss-Markov Theorem: When OLS is BLUE

One of the most celebrated results in statistics and econometrics is the **Gauss-Markov Theorem**. It justifies the use of ordinary least squares (OLS) as not just a reasonable estimator — but the **best** in a specific, well-defined sense.

But "best" here has a precise technical meaning.

---

### What the Gauss-Markov Theorem Says

Under a set of classical assumptions (which we'll spell out), the **OLS estimator**:

$$
\hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$

is the **Best Linear Unbiased Estimator**, often abbreviated as **BLUE**.

Let's break that down:

* **Best**: Has **minimum variance** among all linear unbiased estimators
* **Linear**: Can be written as a linear function of the observed outputs $$ \mathbf{y} $$
* **Unbiased**: Expected value of the estimator equals the true parameter:

  $$
  \mathbb{E}[\hat{\mathbf{w}}] = \mathbf{w}
  $$

In short: **If your assumptions hold, no other linear unbiased estimator will give you lower variance than OLS**.

---

### What Makes the Theorem Tick: The Assumptions

Let's walk through the classical assumptions required for the Gauss-Markov theorem to apply.

These aren't just arbitrary conditions — each one ensures the OLS estimator is unbiased and efficient.

---

#### 1. **Linearity of Parameters**

The model is linear **in the parameters**, not necessarily in the features:

$$
\mathbf{y} = \mathbf{Xw} + \boldsymbol{\varepsilon}
$$

This means:

* $$ \mathbf{w} $$ appears linearly (no $$ \mathbf{w}^2 $$, $$ \log \mathbf{w} $$, etc.)
* The function may include nonlinear transformations of inputs (e.g., $$ x^2 $$), as long as it's linear in $$ \mathbf{w} $$

This is **what makes linear regression "linear"** — not the shape of the data, but the linearity in coefficients.

---

#### 2. **Independence of Errors**

Each error term $$ \varepsilon_i $$ is statistically **independent** of the others:

$$
\text{Cov}(\varepsilon_i, \varepsilon_j) = 0 \quad \text{for all } i \ne j
$$

This assumption ensures that knowing one observation doesn't give information about the error in another.

In time series or spatial data, this often fails — which leads to autocorrelation and requires different modeling strategies (like ARIMA, GEE, GLS).

---

#### 3. **Homoscedasticity (Constant Variance of Errors)**

The error terms have constant variance:

$$
\text{Var}(\varepsilon_i) = \sigma^2 \quad \text{for all } i
$$

This is a key assumption — it means the **spread of the errors stays consistent** across all levels of the inputs.

If this fails (called **heteroscedasticity**), OLS is still unbiased but **no longer has minimum variance**. You lose efficiency.

We'll learn to check this using residual plots and **scale-location plots**.

---

#### 4. **No Multicollinearity**

The columns of $$ \mathbf{X} $$ must be **linearly independent**, i.e.,

$$
\mathbf{X}^\top \mathbf{X} \text{ is invertible}
$$

Why?

Because the normal equation:

$$
\hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$

requires us to invert $$ \mathbf{X}^\top \mathbf{X} $$.

If one feature is an exact linear combination of others (e.g., $$ x_3 = x_1 + x_2 $$), this matrix becomes **singular**, and we cannot uniquely solve for $$ \mathbf{w} $$.

Even if the matrix is technically invertible but nearly singular (high condition number), estimates become **unstable** and **high variance** — this is called **near-multicollinearity**.

---

#### 5. **Normality of Errors** (for inference)

This assumption is **not required** for OLS to be BLUE. It **is required** for:

* Valid hypothesis testing (t-tests, F-tests)
* Confidence intervals on coefficients
* Reliable p-values

We assume:

$$
\boldsymbol{\varepsilon} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})
$$

This makes $$ \hat{\mathbf{w}} $$ also normally distributed:

$$
\hat{\mathbf{w}} \sim \mathcal{N}(\mathbf{w}, \sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1})
$$

This is why we check residuals using **Q-Q plots** — to assess whether this assumption is reasonable.

---

### Putting It All Together

If all assumptions (1–4) hold:

* OLS is **unbiased** and **minimum variance** among linear estimators
* It's the **most efficient linear estimator** of $$ \mathbf{w} $$
* If (5) also holds, we get nice inference tools and valid statistical testing

---

### Visualization: Why OLS is “Best” Among Linear Unbiased Estimators

To understand why the OLS estimator is called **“Best Linear Unbiased Estimator (BLUE)”**, let’s simulate a simple regression problem many times over — mimicking repeated experiments or bootstrapped datasets.

We’ll compare two estimators:

* **OLS Estimator**: The true least squares solution, known to be unbiased and efficient
* **Noisy Estimator**: A linear, unbiased estimator but with artificially added noise to inflate variance

We generate 1000 simulated datasets, each with 30 observations drawn from the true model:

$$
y = 2x + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 1)
$$

For each simulated dataset:

* We compute the OLS slope estimate $$\hat{w}_{\text{OLS}}$$
* We compute an alternate unbiased estimator: $$\hat{w}_{\text{noisy}} = \hat{w}_{\text{OLS}} + \text{random noise}$$

We then plot the distributions of these estimates across the 1000 trials.

What You Should Observe here?

* Both histograms are **centered around the true slope** (2.0), confirming that both estimators are **unbiased**.
* But the **OLS histogram is narrower** — its values fluctuate less from sample to sample.
* The **noisy estimator**, despite being unbiased, has much higher variance.

This visualization makes the Gauss-Markov Theorem intuitive: **OLS minimizes variance** among all linear and unbiased estimators. Other estimators might still be unbiased, but they’ll “wiggle” more — making them less efficient.

Use the interactive chart below to explore this variance difference:


<iframe src="/assets/plotly/ols_variance_comparison.html" width="800" height="500" frameborder="0"></iframe>



---

### Summary Table of Assumptions

<div style="overflow-x: auto;">
  <table class="simple-table">
    <thead>
      <tr>
        <th>Assumption</th>
        <th>Description</th>
        <th>Effect if Violated</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Linearity of parameters</td>
        <td>Model is linear in $$\mathbf{w}$$</td>
        <td>Model is mis-specified</td>
      </tr>
      <tr>
        <td>Independence of errors</td>
        <td>No autocorrelation among residuals</td>
        <td>Estimates remain unbiased, but standard errors are incorrect</td>
      </tr>
      <tr>
        <td>Homoscedasticity</td>
        <td>Errors have constant variance $$ \sigma^2 $$</td>
        <td>OLS is unbiased, but not efficient</td>
      </tr>
      <tr>
        <td>No multicollinearity</td>
        <td>Columns of $$ \mathbf{X} $$ are linearly independent</td>
        <td>OLS is undefined or unstable</td>
      </tr>
      <tr>
        <td>Normality (optional)</td>
        <td>Errors are normally distributed</td>
        <td>Inference is invalid (no confidence intervals or p-values)</td>
      </tr>
    </tbody>
  </table>
</div>

---


## Diagnostics & Residual Analysis in Linear Regression

Fitting a linear regression model is only the beginning. To **trust** and **interpret** the results meaningfully, we must carefully examine the **residuals** — the differences between observed and predicted values:

$$
\hat{\varepsilon}_i = y_i - \hat{y}_i
$$

These residuals contain the fingerprints of assumption violations, model misspecification, influential points, and even bad data.

This section provides a rigorous framework — both mathematically and practically — to **diagnose** your linear regression model using residual analysis and influence metrics.

---

## Residual Diagnostics


### 1. Residuals vs. Fitted Values

#### Concept

This plot helps diagnose:

* **Non-linearity**: Is the true relationship not linear?
* **Heteroscedasticity**: Does the variance of residuals change with the fitted value?

#### What We Plot

* **x-axis**: Predicted values $$ \hat{y}_i $$
* **y-axis**: Residuals $$ \hat{\varepsilon}_i = y_i - \hat{y}_i $$

#### Ideal Outcome

Residuals randomly scattered around zero — no pattern, no curvature, constant spread.

#### What to Look For:

* **Random scatter around zero**: Good — assumptions likely hold
* **Curved pattern**: Indicates **non-linear relationship**
* **Funnel shape** (widening spread): Suggests **heteroscedasticity**
* **Clusters**: May indicate group structure or autocorrelation

---

#### Counterexample: Residual Pattern from a Non-Linear Relationship

Suppose the true relationship is:

$$
y = 2x + 0.5x^2 + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 3)
$$

But we fit a linear model $$ y = w_0 + w_1 x $$ instead.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n = 100
x = np.linspace(0, 10, n)
y_true = 2 * x + 0.5 * x**2
y = y_true + np.random.normal(0, 3, size=n)

# Fit linear regression: y ~ x
X = np.column_stack((np.ones(n), x))
w = np.linalg.inv(X.T @ X) @ X.T @ y
y_pred = X @ w
residuals = y - y_pred

# Plot residuals vs fitted values
plt.figure(figsize=(7, 4))
plt.scatter(y_pred, residuals, alpha=0.8)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values ($\\hat{y}$)")
plt.ylabel("Residuals ($\\hat{\\varepsilon}$)")
plt.title("Residuals vs Fitted Values: Non-linear Pattern")
plt.grid(True)
plt.tight_layout()
plt.show()
```

<iframe src="/assets/plotly/residuals_vs_fitted.html" width="700" height="450" frameborder="0"></iframe>

In the residual plot, you'll see a **parabolic curve** — clear evidence that the linear model is **underfitting** a non-linear trend. This indicates model misspecification and suggests that a **quadratic term** should be included in the model.


---

### 2. Residual Histogram

#### Purpose

The **residual histogram** is one of the simplest but most effective visual diagnostics to check whether residuals are:

* **Symmetric** around zero
* **Roughly bell-shaped**, resembling a normal distribution
* **Free of extreme outliers** or heavy skewness

This plot is used as an **informal proxy for the normality assumption** in linear regression, which underpins the validity of p-values, confidence intervals, and hypothesis testing.

---

#### Mathematical Setup

Recall the residuals:

$$
\hat{\varepsilon}_i = y_i - \hat{y}_i
$$

If the errors $$\varepsilon_i$$ in the true model are i.i.d. from a normal distribution:

$$
\varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

then the residuals $$\hat{\varepsilon}_i$$ should be approximately normal too (especially with large sample size).

---

#### Practical Questions This Helps Answer

* Are residuals **skewed** (asymmetry)?
* Do we see **outliers** (extreme residuals)?
* Are the tails **fatter or thinner** than expected under normality?

---

#### Example: Simulated Linear Model

We simulate a dataset where residuals **do follow normality**.



```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
n = 100
x = np.linspace(0, 10, n)
y = 3 * x + np.random.normal(0, 2, size=n)

# Linear regression
X = np.column_stack((np.ones(n), x))
w = np.linalg.inv(X.T @ X) @ X.T @ y
y_pred = X @ w
residuals = y - y_pred

# Plot histogram of residuals
plt.figure(figsize=(7, 4))
plt.hist(residuals, bins=20, color='steelblue', edgecolor='black', alpha=0.75)
plt.axvline(0, color='red', linestyle='--')
plt.title("Histogram of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
```

<iframe src="/assets/plotly/residual_histogram.html" width="700" height="450" frameborder="0"></iframe>

#### Interpretation

In this example, the histogram shows:

* **Centered** residuals around zero
* A **bell-like shape**
* No significant **asymmetry** or **outliers**

This supports the normality assumption.

In practice, strong skew, long tails, or multimodal distributions may suggest:

* Transforming the dependent variable (e.g., log-scale)
* Modeling with different error structures
* Investigating potential outliers or heteroscedasticity


---

### 3. Q–Q Plot (Quantile-Quantile Plot)

#### Purpose

The **Q–Q plot** is a more formal and visual way to assess the **normality of residuals** in linear regression. While histograms can suggest shape, Q–Q plots directly compare residual quantiles to theoretical quantiles from a normal distribution.

This is particularly important when validating the assumptions for:

* Confidence intervals
* Hypothesis testing (t-tests, F-tests)
* General inference under the Gauss-Markov framework with normal errors

---

#### How It Works

Let the residuals be:

$$
\hat{\varepsilon}_1, \hat{\varepsilon}_2, \dots, \hat{\varepsilon}_n
$$

Sort them:

$$
\hat{\varepsilon}_{(1)} \leq \hat{\varepsilon}_{(2)} \leq \dots \leq \hat{\varepsilon}_{(n)}
$$

Let the **theoretical quantiles** come from a normal distribution with the same mean and standard deviation as the residuals:

$$
q_i = \Phi^{-1}\left(\frac{i - 0.5}{n}\right) \cdot \hat{\sigma} + \bar{\hat{\varepsilon}}
$$

We plot:

* **x-axis**: $$q_i$$ (Theoretical quantiles)
* **y-axis**: $$\hat{\varepsilon}_{(i)}$$ (Sample quantiles)

---

#### Ideal Behavior

If the residuals are normally distributed, the points will lie approximately on the **45-degree line** through the origin.

---

#### Deviations and Interpretations

<div style="overflow-x: auto;">
  <table class="simple-table">
    <thead>
      <tr>
        <th>Shape</th>
        <th>Interpretation</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>S-shaped curve</td>
        <td>Heavy tails (leptokurtic)</td>
      </tr>
      <tr>
        <td>C-shaped curve</td>
        <td>Light tails (platykurtic)</td>
      </tr>
      <tr>
        <td>Curve above/below</td>
        <td>Skewness</td>
      </tr>
      <tr>
        <td>Zig-zag near ends</td>
        <td>Outliers or small sample size</td>
      </tr>
    </tbody>
  </table>
</div>


---

#### Example: Residuals from a Well-Specified Linear Model

We simulate a linear regression where the residuals are normal.



```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(2)
n = 100
x = np.linspace(0, 10, n)
y = 5 + 2 * x + np.random.normal(0, 1.5, size=n)

X = np.column_stack((np.ones(n), x))
w = np.linalg.inv(X.T @ X) @ X.T @ y
y_pred = X @ w
residuals = y - y_pred

# Q-Q plot
plt.figure(figsize=(6, 4))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q–Q Plot of Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()
```

<iframe src="/assets/plotly/qqplot_residuals.html" width="700" height="450" frameborder="0"></iframe>



#### Interpretation

In this example, the Q–Q plot shows points falling close to the diagonal — suggesting that the residuals are approximately normally distributed.

If your Q–Q plot curves upward or downward significantly, it implies that the tails of the residual distribution do **not** match those of a normal distribution. That’s a strong signal to investigate outliers, skewness, or transformations such as logarithmic or Box-Cox.


---

### 4. Scale-Location Plot (Spread vs. Level Plot)

#### Purpose

The **scale-location plot** (also called **spread-location plot**) is a diagnostic tool used to detect **heteroscedasticity** — when the variance of residuals changes with the fitted values. This violates the **homoscedasticity assumption** of the linear model:

$$
\text{Var}(\varepsilon_i) = \sigma^2 \quad \forall i
$$

In practical terms, if the variance of residuals increases or decreases with $$\hat{y}_i$$, it affects:

* The efficiency of the OLS estimator
* The validity of standard errors, confidence intervals, and hypothesis tests

---

#### How It Works

This plot uses **standardized residuals**:

$$
r_i = \frac{\hat{\varepsilon}_i}{\hat{\sigma} \sqrt{1 - h_{ii}}}
$$

Where:

* $$\hat{\varepsilon}_i = y_i - \hat{y}_i$$ is the residual
* $$h_{ii}$$ is the **leverage** of observation $$i$$ (from the hat matrix)

Then we plot:

* **x-axis**: Fitted values $$\hat{y}_i$$

* **y-axis**: $$\sqrt{ \mid r_i \mid }$$ — square root of the absolute standardized residual

The square root transformation helps stabilize the variance and make patterns more visually detectable.

---

#### What to Look For

* **Flat horizontal band**: Good — residual spread is consistent
* **Fanning out**: Variance increases with $$\hat{y}$$ → heteroscedasticity
* **Funneling in**: Variance decreases with $$\hat{y}$$ → also problematic



```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)
n = 100
x = np.linspace(1, 10, n)
hetero_noise = np.random.normal(0, x, size=n)  # variance increases with x
y = 2 * x + hetero_noise

X = np.column_stack((np.ones(n), x))
w = np.linalg.inv(X.T @ X) @ X.T @ y
y_pred = X @ w
residuals = y - y_pred

# Leverage values (diagonal of hat matrix)
H = X @ np.linalg.inv(X.T @ X) @ X.T
leverage = np.diag(H)
sigma_hat = np.sqrt(np.sum(residuals**2) / (n - 2))
standardized_residuals = residuals / (sigma_hat * np.sqrt(1 - leverage))

plt.figure(figsize=(7, 4))
plt.scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.8)
plt.title("Scale-Location Plot")
plt.xlabel("Fitted Values ($\\hat{y}$)")
plt.ylabel("$\\sqrt{|\\text{Standardized Residuals}|}$")
plt.grid(True)
plt.tight_layout()
plt.show()
```

<iframe src="/assets/plotly/scale_location_plot.html" width="700" height="450" frameborder="0"></iframe>


#### Interpretation

In this example, the **residual spread increases** with the fitted values — a classic signal of **heteroscedasticity**. This violates the OLS assumption of constant variance.

**Fixes and Remedies**:

* Use **log** or **Box-Cox transformations** on the dependent variable
* Apply **Weighted Least Squares (WLS)**
* Consider robust standard errors (e.g., White’s standard errors)

---

### 5. Durbin–Watson Test

#### Purpose

The **Durbin–Watson (DW) statistic** is a formal test used to detect **first-order autocorrelation** in the residuals of a regression model — that is, whether the residuals are correlated with themselves at lag 1.

This is **especially important in time series regression**, where successive observations may not be independent.

OLS assumes:

$$
\text{Cov}(\varepsilon_i, \varepsilon_{i-1}) = 0
$$

If this assumption fails, standard errors are misestimated, and inference becomes unreliable.

---

#### The Durbin–Watson Statistic

Given residuals $$\hat{\varepsilon}_1, \hat{\varepsilon}_2, \dots, \hat{\varepsilon}_n$$ from a regression model, the DW statistic is:

$$
DW = \frac{\sum_{i=2}^{n} (\hat{\varepsilon}_i - \hat{\varepsilon}_{i-1})^2}{\sum_{i=1}^{n} \hat{\varepsilon}_i^2}
$$

This statistic ranges from 0 to 4:

* $$DW \approx 2$$: No autocorrelation
* $$DW < 2$$: Positive autocorrelation
* $$DW > 2$$: Negative autocorrelation

---

#### Rule of Thumb

<div style="overflow-x: auto;">
  <table class="simple-table">
    <thead>
      <tr>
        <th>DW Value</th>
        <th>Interpretation</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>≈ 2</td>
        <td>No autocorrelation</td>
      </tr>
      <tr>
        <td>&lt; 2</td>
        <td>Positive autocorrelation (common in time series)</td>
      </tr>
      <tr>
        <td>&gt; 2</td>
        <td>Negative autocorrelation</td>
      </tr>
    </tbody>
  </table>
</div>

---

We simulate a linear model where the errors are autocorrelated, i.e., each residual depends on the previous one. We'll compute the residuals and the Durbin–Watson statistic and plot the residuals to visualize the autocorrelation pattern.

```python
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

np.random.seed(4)
n = 100
x = np.linspace(0, 10, n)

# Create autocorrelated errors: ε_t = ρ * ε_{t-1} + noise
rho = 0.8
e = np.zeros(n)
e[0] = np.random.normal()
for t in range(1, n):
    e[t] = rho * e[t-1] + np.random.normal()

y = 2 * x + e

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
residuals = model.resid

# Durbin-Watson test
dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson statistic: {dw_stat:.3f}")

# Plot residuals
plt.figure(figsize=(7, 4))
plt.plot(residuals, marker='o', linestyle='-', alpha=0.8)
plt.title("Residuals from OLS Model with Autocorrelation")
plt.xlabel("Time Index")
plt.ylabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.show()
```

<iframe src="/assets/plotly/durbin_watson_residuals.html" width="700" height="450" frameborder="0"></iframe>

#### Interpretation

In this simulation, we intentionally added **positive autocorrelation** to the residuals. The Durbin–Watson statistic will be **much less than 2** — often around **1 or lower** — clearly flagging this issue.

Such autocorrelation invalidates the assumption of independence, which in turn:

* Underestimates standard errors
* Inflates Type I error rates
* Makes OLS inference unreliable

---

**Solutions**:

* Use **ARIMA models** or **Generalized Least Squares (GLS)**
* Apply **Newey–West standard errors** to correct for autocorrelation
* Use **time-aware modeling strategies** if the data is sequential

---

## Influence & Leverage Diagnostics

OLS is sensitive to **outliers** and **influential points**. Even one anomalous observation can severely bias estimates.

To detect such points, we analyze:

### 1. Hat Matrix and Leverage

#### What is the Hat Matrix?

The **hat matrix** $$ \mathbf{H} $$ is central to understanding how linear regression generates fitted values:

$$
\hat{\mathbf{y}} = \mathbf{H} \mathbf{y}, \quad \text{where} \quad \mathbf{H} = \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top
$$

It is called the **hat matrix** because it "puts a hat" on $$ \mathbf{y} $$ — i.e., it transforms the observed response into the predicted response via linear projection onto the column space of $$ \mathbf{X} $$.

---

#### Leverage: Diagonal of the Hat Matrix

Each diagonal element of $$ \mathbf{H} $$ — denoted $$ h_{ii} $$ — represents the **leverage** of the $$ i $$-th observation. It measures how much influence the point $$ \mathbf{x}_i $$ has on its own fitted value $$ \hat{y}_i $$.

Mathematically:

$$
h_{ii} = \mathbf{x}_i^\top (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{x}_i
$$

Where $$ \mathbf{x}_i $$ is the $$ i $$-th row vector of $$ \mathbf{X} $$.

---

#### Interpretation of Leverage

* A **high leverage point** lies far away from the center of the $$ x $$-space (mean of predictors).
* A **low leverage point** lies near the centroid of the data.
* Leverage is **purely a function of the input features** — not the response variable.

#### Properties of $$ \mathbf{H} $$

* $$ \mathbf{H} $$ is symmetric and idempotent:

  $$
  \mathbf{H} = \mathbf{H}^\top, \quad \mathbf{H}^2 = \mathbf{H}
  $$
* $$ 0 < h_{ii} < 1 $$ for all $$ i $$
* The sum of leverages equals the number of parameters:

  $$
  \sum_{i=1}^{n} h_{ii} = \text{trace}(\mathbf{H}) = p
  $$

---

#### Rule of Thumb: High Leverage

A common practical threshold:

$$
h_{ii} > \frac{2p}{n}
$$

Where:

* $$ p $$ = number of model parameters (including intercept)
* $$ n $$ = number of observations

Observations exceeding this threshold are considered **high-leverage** and should be further analyzed for influence.



### 2. Studentized Residuals

#### Purpose

**Studentized residuals** are standardized versions of the raw residuals that take into account the **leverage** of each observation. Unlike plain residuals, studentized residuals adjust for the fact that points with high leverage tend to have **underestimated** residuals (due to their ability to pull the regression line toward them).

They are essential for:

* Comparing residuals across observations with different leverages
* Detecting **outliers in the response variable** $$ y_i $$
* Serving as building blocks for more advanced influence measures like **Cook's Distance** and **DFBETAs**

---

#### Formula

Given:

* $$ \hat{\varepsilon}_i $$: raw residual
* $$ h_{ii} $$: leverage (diagonal of the hat matrix)
* $$ \hat{\sigma}^2 $$: estimated variance of residuals

The **studentized residual** is:

$$
r_i = \frac{\hat{\varepsilon}_i}{\hat{\sigma} \sqrt{1 - h_{ii}}}
$$

---

#### Interpretation

<div style="margin-top: 1rem; margin-bottom: 1rem;"> <p><strong>Outlier Detection:</strong><br> <span style="display: inline-block; white-space: nowrap;"> Large values of $$|r_i|$$ indicate that the actual response $$y_i$$ is significantly different from the predicted value, even after adjusting for leverage. </span> </p> <p><strong>Statistical Reference:</strong><br> <span style="display: inline-block; white-space: nowrap;"> Studentized residuals are interpreted using a t-distribution with $$n - p - 1$$ degrees of freedom. </span> </p> <p><strong>Threshold Guidelines:</strong><br> <span style="display: inline-block; white-space: nowrap;">$$|r_i| > 2$$</span> → moderate outlier<br> <span style="display: inline-block; white-space: nowrap;">$$|r_i| > 3$$</span> → strong outlier </p> </div>


---

#### Practical Use

Studentized residuals are especially helpful when:

* A point appears to have a **small residual** but **high leverage** (underestimated raw residual)
* You want to **screen systematically for outliers** without being distorted by leverage

---

**Why studentized residuals matter**:

* They expose outliers that raw residuals might conceal
* They normalize for leverage, allowing fair comparison across all points

### 3. Cook's Distance

#### Purpose

**Cook's Distance** is a powerful diagnostic metric that quantifies the **influence** of each observation on the fitted regression model. Unlike raw residuals or leverage alone, Cook's Distance combines:

* **Residual size**: how far off the point is from the model prediction
* **Leverage**: how extreme the point is in the feature space

Together, they assess how much **removing a point** would change the regression coefficients.

---

#### Mathematical Definition

Cook's Distance for the $$ i $$-th observation is defined as:

$$
D_i = \frac{(\hat{\mathbf{w}}_{(-i)} - \hat{\mathbf{w}})^\top (\mathbf{X}^\top \mathbf{X}) (\hat{\mathbf{w}}_{(-i)} - \hat{\mathbf{w}})}{p \cdot \hat{\sigma}^2}
$$

where:

* $$ \hat{\mathbf{w}}_{(-i)} $$ is the OLS estimate after excluding observation $$ i $$
* $$ \hat{\mathbf{w}} $$ is the full model estimate
* $$ p $$ is the number of parameters (including intercept)
* $$ \hat{\sigma}^2 $$ is the estimated variance

This expression directly measures **how much the regression coefficients would change** if a single point were deleted.

---

#### Efficient Formula

To avoid re-fitting the model $$ n $$ times, Cook's Distance can be computed using:

$$
D_i = \frac{r_i^2}{p} \cdot \frac{h_{ii}}{1 - h_{ii}}
$$

where:

* $$ r_i $$ is the studentized residual
* $$ h_{ii} $$ is the leverage for observation $$ i $$
* $$ p $$ is the number of predictors (including intercept)

---

#### Interpretation

Cook's Distance is not just about one bad prediction — it tells you which points **pull the entire model** disproportionately.

Large values of $$ D_i $$ indicate observations that significantly affect the regression coefficients. As a general guideline:

$$ D_i > 1 $$ is often considered large and worth investigating, but thresholds should be interpreted relative to the data size and leverage structure.

---

### 4. DFBETAs

#### Purpose

**DFBETAs** provide a **fine-grained view** of influence in linear regression. While Cook's Distance tells us whether a point influences the overall model fit, **DFBETAs tell us which specific regression coefficient is being affected — and by how much**.

This makes DFBETAs especially valuable when you're trying to understand:

* Which coefficient(s) are sensitive to particular observations
* Whether changes in model parameters are being driven by a small number of points
* How to debug unstable or counterintuitive coefficient signs

---

#### Definition

The **DFBETA** for observation $$ i $$ and coefficient $$ j $$ is defined as:

$$
DFBETA_{ij} = \frac{\hat{w}_j - \hat{w}_{j(-i)}}{\text{SE}(\hat{w}_j)}
$$

Where:

* $$ \hat{w}_j $$ is the estimated coefficient using all observations
* $$ \hat{w}_{j(-i)} $$ is the estimate of the same coefficient after deleting observation $$ i $$
* $$ \text{SE}(\hat{w}_j) $$ is the standard error of the coefficient from the full model

---

#### Interpretation

The numerator captures **how much coefficient $$ j $$ changes** if observation $$ i $$ is removed. The denominator scales this change by the estimated standard error, making DFBETA a **standardized sensitivity score**.

Large absolute values of $$ DFBETA_{ij} $$ indicate that observation $$ i $$ has a strong influence on coefficient $$ \hat{w}_j $$.

Rule of thumb: flag any point where  
$$ |DFBETA_{ij}| > \frac{2}{\sqrt{n}} $$

This threshold is based on standard normal theory and becomes more conservative as sample size increases.


## Summary of What Each Tool Tells You

<div style="overflow-x: auto;">
  <table class="simple-table">
    <thead>
      <tr>
        <th>Tool</th>
        <th>Detects</th>
        <th>Actionable Insight</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Residual vs. Fitted Plot</td>
        <td>Non-linearity, heteroscedasticity</td>
        <td>Add non-linear terms, transform variables</td>
      </tr>
      <tr>
        <td>Residual Histogram / Q-Q Plot</td>
        <td>Normality of errors</td>
        <td>Needed for valid inference</td>
      </tr>
      <tr>
        <td>Scale-Location Plot</td>
        <td>Variance pattern</td>
        <td>Check for transformation or weighted least squares</td>
      </tr>
      <tr>
        <td>Durbin-Watson Test</td>
        <td>Autocorrelation in residuals</td>
        <td>Use time-series methods</td>
      </tr>
      <tr>
        <td>Hat values</td>
        <td>Leverage</td>
        <td>Outlying $$x$$ values</td>
      </tr>
      <tr>
        <td>Studentized residuals</td>
        <td>Outliers in $$y$$</td>
        <td>Flag unusual responses</td>
      </tr>
      <tr>
        <td>Cook’s Distance</td>
        <td>Overall influence</td>
        <td>Delete/check points that unduly affect fit</td>
      </tr>
      <tr>
        <td>DFBETAs</td>
        <td>Influence on each coefficient</td>
        <td>Fine-grained sensitivity analysis</td>
      </tr>
    </tbody>
  </table>
</div>

---


## Concluding Thoughts

Linear regression may appear elementary — a straight line through a scatter of points. But as we’ve unraveled across this blog, its depth lies not just in its simplicity but in its interpretability, mathematical grounding, and diagnostic transparency.

We began with the **problem setup**, understanding that modeling $$\mathbf{y} = \mathbf{Xw} + \varepsilon$$ is an exercise in **projection** — projecting the response vector onto the column space of the input features. This geometric view revealed that linear regression is not just an algebraic trick but a fundamental idea in vector spaces and approximation theory.

From there, we derived the **normal equation**, explored its **geometric interpretation**, and discovered under what conditions OLS is **Best Linear Unbiased Estimator (BLUE)** — all under the powerful lens of the **Gauss-Markov Theorem**. But theory, we learned, is only as good as its assumptions.

That's why we dove deep into **diagnostics**: examining **residuals**, assessing **normality**, checking for **homoscedasticity**, **autocorrelation**, **leverage**, and **influence**. We visualized how a seemingly innocent outlier or a high-leverage point can silently skew results, and how metrics like **Cook’s Distance** and **DFBETAs** help keep our models honest.

More than formulas, we encountered a philosophy: a model should not only fit data, it should also respect it — by scrutinizing its assumptions, listening to its residuals, and acknowledging its limits.

As datasets grow in size and complexity, linear regression might no longer be the final model — but it remains the **first lens**. A lens for understanding relationships, for diagnosing data, and for interpreting results. Its transparency makes it an essential starting point — not just for statisticians, but for any data scientist who values insight over blind prediction.

In the next chapter, we’ll turn our attention to where OLS falls short — especially in the presence of multicollinearity and high-dimensionality — and explore how **Ridge**, **Lasso**, and **Elastic Net** regression bring in the power of **regularization** to stabilize and refine our models.
