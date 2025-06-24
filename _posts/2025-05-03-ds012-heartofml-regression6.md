---
layout: post
title: "Capturing Non-Linearity in Regression"
date: 2025-05-03
description: "Master non-linear regression to model complex data relationships in this comprehensive guide. Discover polynomial regression, basis expansions (splines, Fourier, RBF), and non-parametric methods like kernel smoothing and LOESS, with practical Python code, visualizations, and diagnostics. Learn how to avoid overfitting, apply regularization, and engineer features for robust predictions in domains like energy forecasting, financial trends, and biomedical research. Perfect for data scientists and analysts seeking to enhance predictive modeling skills."
tags: ml ai
categories: machine-learning heart-of-ml data-science-series
thumbnail: 
math: true
giscus_comments: true
related_posts: true
chart:
  plotly: true
toc:
  beginning: true
---




> *“Linear models are like well-behaved students: simple, efficient, and obedient. But real-world data? That’s the class clown—noisy, nonlinear, and mischievous.”*

Linear regression has long been the bread-and-butter of statistical modeling. Elegant in its simplicity, fast in computation, and surprisingly powerful for linearly separable patterns—it’s often the first tool we reach for. But what happens when the world refuses to be linear?

Imagine trying to model human growth across ages, or predict energy consumption based on time of day. You may notice curvatures, thresholds, or local effects that violate the flat, global assumptions of linearity. In these scenarios, a straight line isn't just insufficient—it’s misleading.

This blog explores the next layer: **non-linear regression**. Not by abandoning linear models entirely, but by extending and adapting them—injecting curvature, locality, and flexibility—through the use of:

* **Polynomial expansions**: bending straight lines into parabolas and beyond.
* **Basis functions**: letting sine waves, splines, or Gaussian bells do the talking.
* **Non-parametric smoothers**: trusting the data itself to trace its own shape.

Our aim is to unpack not only *how* these techniques work, but also *when* and *why* they should be used. Through interactive visualizations, diagnostics, and real-world use cases, we’ll learn to recognize when the linear assumption starts to crack—and how to mend it with robust, expressive tools.

---


# Polynomial Regression: Bending the Line Without Breaking the Model

Linear regression is often the first tool in the data scientist’s toolkit. It’s interpretable, efficient, and surprisingly powerful—so long as your data obeys one crucial assumption: **linearity**.

But what happens when that assumption fails?

What if your data curves, arches, or oscillates? A straight line cannot capture the pattern. Yet discarding linear models entirely would forgo their elegance and tractability. **Polynomial regression** offers a compromise: *a flexible, yet linear-in-parameters approach to modeling non-linear relationships.*

---

## Revisiting Linearity: When a Straight Line Isn’t Enough

The core form of a simple linear model is:


$$
y = \beta_0 + \beta_1 x + \varepsilon
$$


Here, $x$ is the input (a scalar), $y$ is the output, $\beta_0$ is the intercept, $\beta_1$ is the slope, and $\varepsilon$ is noise.

In many real-world settings, this assumption—that $y$ changes proportionally with $x$—simply doesn't hold. Consider the following:

* The height of a projectile over time follows a **quadratic** curve.
* Diminishing returns in advertising spend follow a **non-linear saturation** effect.
* Temperature vs. energy consumption shows **seasonal ripples**.

These are not linear problems. But they *can* still be modeled using **linear regression**, if we transform the inputs using polynomial basis functions.

---

## Core Idea: Still Linear, Just Not in the Inputs

In **polynomial regression**, we extend the input space by adding powers of the original input:


$$
y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \dots + \beta_d x^d + \varepsilon
$$


This is a **non-linear function in the input $x$**, but still **linear in the parameters** $\beta$. That means we can use all the machinery of linear regression—ordinary least squares, closed-form solutions, interpretability—with extended capacity to fit curved relationships.

Formally, if $x \in \mathbb{R}$, then the polynomial feature map $\phi(x)$ is:


$$
\phi(x) = [1, x, x^2, x^3, \dots, x^d]^T
$$


---

## Mathematical Framing

Let $X \in \mathbb{R}^{n \times 1}$ be your input matrix with $n$ samples. After polynomial expansion to degree $d$, you obtain:


$$
\Phi = 
\begin{bmatrix}
1 & x_1 & x_1^2 & \dots & x_1^d \\
1 & x_2 & x_2^2 & \dots & x_2^d \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_n & x_n^2 & \dots & x_n^d \\
\end{bmatrix}
\in \mathbb{R}^{n \times (d+1)}
$$


Then, the optimal parameters minimize the mean squared error:


$$
\min_{\beta} \|\Phi \beta - y\|_2^2
$$


This is identical to linear regression, except your design matrix $\Phi$ now contains polynomial terms.

---

## Visual Example: Fitting with Varying Degrees

Imagine fitting models with degrees from 1 to 14 to a wiggly sine-like curve. At degree 1, you underfit. At degree 14, you perfectly trace every noise bump—overfitting catastrophically.

This trade-off is captured in the **bias–variance decomposition**:

* **Low-degree** polynomials = high bias, low variance.
* **High-degree** polynomials = low bias, high variance.
* Somewhere in the middle lies the sweet spot.

Let’s visualize this.

<iframe src="/assets/plotly/poly_mse_vs_degree.html" width="100%" height="500" frameborder="0"></iframe>

This chart plots training and test MSE as a function of degree. Observe how test error initially drops but then climbs—classic overfitting behavior.

---

## Use Cases of Polynomial Regression

Polynomial regression is often the right choice when:

* The relationship is **smooth but curved**, with no sharp discontinuities.
* You expect **symmetric curvature**, like U- or bell-shaped trends.
* You want to capture **trend acceleration/deceleration** in time-series data.

Some real-world applications include:

* **Projectile motion modeling**: height as a function of time.
* **Dose–response modeling** in pharmacology.
* **Modeling learning curves**: accuracy as a function of training epochs.

---

## Python Implementation: Step-by-Step

Let’s see how this works in code. We use scikit-learn’s `PolynomialFeatures` for the transformation and `LinearRegression` to fit the model.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Create a 2nd-degree polynomial regression model
polyreg = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=True),
    LinearRegression()
)

# Fit to data (reshape required if X_train is 1D)
polyreg.fit(X_train.reshape(-1, 1), y_train)

# Predict on test data
y_pred = polyreg.predict(X_test.reshape(-1, 1))
```

* `PolynomialFeatures` creates new features: $x, x^2, \dots$.
* `include_bias=True` adds the constant term (intercept).
* `make_pipeline` ensures the transformation and regression happen sequentially.

This pipeline is clean, reproducible, and compatible with cross-validation and hyperparameter tuning.

---

## Interaction Terms in Higher Dimensions

With multiple input features, `PolynomialFeatures` does more than powers—it introduces **interactions**.

For example, with $x_1, x_2$ and degree 2, you get:


$$
[1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]
$$


These interaction terms are critical when the effect of one feature depends on another (e.g., **region × marketing spend**).

To create **only** interaction terms without self-squared terms:

```python
PolynomialFeatures(degree=2, interaction_only=True)
```

This can help reduce the feature explosion while capturing cross-variable relationships.

---

## Overfitting: The Hidden Cost of Flexibility

Higher-degree polynomials are powerful—but with power comes risk.

They tend to:

* **Oscillate wildly** at the boundaries (Runge's phenomenon).
* Be **sensitive to small noise** in training data.
* Lead to **ill-conditioned** design matrices (multicollinearity).

### Diagnosing Overfit

* **Training error drops** as degree increases.
* **Test error decreases** initially, then **increases**.
* Coefficient magnitudes may **explode**, especially near the boundaries.

### Remedies

* Use **regularization** (e.g., Ridge, Lasso).
* Limit polynomial **degree ≤ 3** in most practical cases.
* Prefer **domain-driven features** over brute-force expansions.

---

## Summary: When to Reach for Polynomials

Polynomial regression is a powerful extension to linear models. It bends straight lines into expressive curves while preserving the analytical simplicity of least squares.

Use it when:

* You need simple curvature.
* You have small-to-medium datasets.
* You want fast, interpretable models.

Avoid it when:

* The relationship has **sharp transitions** (use splines instead).
* Your data is high-dimensional (risk of exponential feature blowup).
* You don’t regularize (polynomials can be unstable without constraints).

---

Up next, we’ll explore a more flexible, compositional approach to modeling curvature: **basis expansion**. This includes **splines**, **Fourier series**, and **radial basis functions**, which allow more **localized** and **structured** control over non-linearity—often outperforming polynomials in both stability and interpretability.

---


# Basis Expansion: Beyond Polynomials, Toward Flexibility

Polynomial regression offers a compelling way to capture curvature by extending features with powers of input variables. But it comes with a built-in fragility. As the degree of the polynomial increases, models often become **unstable**, **oscillatory**, and **sensitive to outliers**—especially near the domain boundaries.

Can we build models that are flexible like high-degree polynomials, yet stable and grounded in real-world structure?

Enter **basis expansion**.

Basis expansion techniques transform your original feature space using **non-polynomial functions**—splines, sinusoids, Gaussians—to better represent the underlying shape of the data. This approach doesn’t assume a global fit like a single high-degree polynomial does. Instead, it divides the problem into smaller, **localized approximations**.

---

## What is Basis Expansion?

The core idea behind basis expansion is deceptively simple:

> **Instead of fitting a model directly to your original features, you first map your data into a new space using a set of basis functions**.

Mathematically, suppose your input is $x \in \mathbb{R}$. You define a collection of $K$ basis functions $\{\phi_1(x), \phi_2(x), \dots, \phi_K(x)\}$, and model the output $y$ as:


$$
y = \beta_0 + \beta_1 \phi_1(x) + \beta_2 \phi_2(x) + \cdots + \beta_K \phi_K(x) + \varepsilon
$$


This is still **linear in the parameters** $\beta$, allowing you to use ordinary least squares. However, it is **nonlinear in the inputs**, making the model expressive and adaptable.

The key question becomes: what should the basis functions $\phi_k(x)$ be?

---

## Three Powerful Basis Families

We now explore three widely-used basis expansion strategies—each suited to a different class of patterns in the data.

---

### 1. Splines: Local Polynomials with Global Smoothness

**Splines** are perhaps the most versatile and intuitive basis expansion technique. A spline is a **piecewise polynomial** function: the domain is divided into intervals by **knots**, and a separate polynomial is fitted within each interval. Crucially, the individual pieces are joined smoothly, ensuring continuity and differentiability.

#### Intuition

Imagine fitting several gentle curves to segments of your data and then stitching them together seamlessly. The beauty of splines lies in their **local control**: changing one region does not affect the rest of the curve.

#### Types of Splines

* **B-splines**: Basis splines used to build smooth curves efficiently.
* **Natural splines**: Add boundary constraints to avoid instability at the edges.
* **Cubic splines**: Use degree-3 polynomials for each segment—balance between smoothness and complexity.

#### Mathematical View

Suppose you have knots at $\xi_1, \xi_2, \dots, \xi_m$. A spline of degree $d$ is a function $s(x)$ such that:

* On each interval $[\xi_i, \xi_{i+1}]$, $s(x)$ is a polynomial of degree $d$.
* The function and its derivatives up to degree $d - 1$ are continuous over the entire domain.

This smoothness is enforced using the **B-spline basis functions** $\phi_k(x)$, which have local support.

---

### Python Example: Fitting Cubic Splines

We can fit cubic splines using `patsy` for basis expansion and `statsmodels` for regression:

```python
from patsy import dmatrix
import statsmodels.api as sm

# Assume X_train is a 1D numpy array of input values
spline_basis = dmatrix("bs(x, df=6, degree=3, include_intercept=True)", 
                       {"x": X_train}, 
                       return_type="dataframe")

spline_model = sm.OLS(y_train, spline_basis).fit()
```

* `"bs(x, df=6)"` creates B-spline basis with 6 degrees of freedom.
* `degree=3` makes them cubic splines.
* The design matrix `spline_basis` is then used in an ordinary least squares fit.

#### Why Splines Work

* They **adapt to local changes** in the data.
* They **prevent overfitting** by controlling the number and placement of knots.
* Natural splines additionally impose boundary conditions, reducing oscillation at the domain edges.

#### Use Cases

* Modeling **growth curves** in biology or medicine.
* Smoothing **longitudinal measurements** (e.g., patient vitals over time).
* Capturing **non-linear economic trends** without overfitting.

---

### 2. Fourier Basis: Modeling Periodicity

The **Fourier basis** consists of sine and cosine functions of varying frequencies. It’s particularly well-suited to **periodic data**—where patterns repeat over time.

A truncated Fourier expansion of a signal $x$ looks like:


$$
y = \beta_0 + \sum_{k=1}^{K} \left[ \beta_k \cos(2\pi k x) + \gamma_k \sin(2\pi k x) \right] + \varepsilon
$$


This allows the model to represent **seasonal components**, **cyclical behaviors**, or **time-of-day effects**.

#### Use Cases

* **Electricity demand** forecasting (daily/weekly cycles).
* **Web traffic** prediction (hourly or weekly seasonality).
* **Retail sales** with holiday or seasonal patterns.

#### Why It Works

* The Fourier basis is **orthogonal**, which helps numerical stability.
* Captures **smooth, global** periodic effects.
* Truncating at small $K$ avoids overfitting.

#### In Python

Though you can hand-engineer sine and cosine terms, many time series libraries (like `statsmodels.tsa`, `prophet`, or `FourierFeaturizer` in sklearn-compatible packages) support Fourier basis expansions natively.

---

### 3. Radial Basis Functions (RBF): Local Smooth Bumps

**Radial Basis Functions** are real-valued functions whose value depends only on the distance from a center:


$$
\phi_k(x) = \exp\left( -\frac{(x - c_k)^2}{2\sigma^2} \right)
$$


Each basis function is a **bell-shaped curve** centered at $c_k$, with bandwidth $\sigma$. A linear combination of these functions can approximate arbitrarily complex functions.

#### Why It’s Powerful

* RBFs are **localized**: each basis focuses on a small region.
* You can **control smoothness** via the bandwidth parameter.
* Unlike splines, RBFs don’t require piecewise joins—they’re smooth everywhere.

#### Use Cases

* Modeling **sensor signals** or **geospatial phenomena**.
* **Function approximation** in reinforcement learning or control.
* Used heavily in **support vector machines** and **kernel ridge regression**.

---

### Choosing the Right Basis

<div style="overflow-x: auto;"> <table class="simple-table"> <thead> <tr> <th>Basis Type</th> <th>Strengths</th> <th>Best For</th> </tr> </thead> <tbody> <tr> <td><strong>Splines</strong></td> <td>Local, interpretable, fast</td> <td>Smooth curves with variable trends</td> </tr> <tr> <td><strong>Fourier</strong></td> <td>Captures periodicity, global smoothness</td> <td>Time-series with known cycles</td> </tr> <tr> <td><strong>RBF</strong></td> <td>Local bumps, flexible smoothness</td> <td>Arbitrary non-linearities, dense data</td> </tr> </tbody> </table> </div>

---

## Basis Expansion vs. Polynomial Regression

<div style="overflow-x: auto;"> <table class="simple-table"> <thead> <tr> <th>Aspect</th> <th>Polynomial Regression</th> <th>Basis Expansion</th> </tr> </thead> <tbody> <tr> <td>Functional form</td> <td>Global polynomial</td> <td>Sum of custom basis functions</td> </tr> <tr> <td>Stability near edges</td> <td>Poor (Runge phenomenon)</td> <td>Better with splines or RBFs</td> </tr> <tr> <td>Local control</td> <td>None</td> <td>High (splines, RBFs)</td> </tr> <tr> <td>Periodicity modeling</td> <td>Poor</td> <td>Excellent with Fourier</td> </tr> <tr> <td>Interpretability</td> <td>High (low-degree)</td> <td>Moderate (depends on basis)</td> </tr> </tbody> </table> </div>

---

## Practical Tip: When in Doubt, Use Splines

Splines offer the **best of both worlds**:

* They're flexible, but not wildly so.
* They’re local, yet smoothly connected.
* They’re linear-in-parameters, making them fast and interpretable.

In many real-world problems—from biomedical signals to economic forecasting—splines can outperform both raw polynomials and overly generic black-box models.

---

Next, we turn to **non-parametric smoothing methods** like **LOESS** and **kernel regression**, which completely remove the idea of a fixed basis and instead **let the data determine the shape** of the curve, one point at a time.

---

# Non-Parametric Regression: Letting the Data Shape the Model

The modeling techniques we’ve discussed so far—polynomial regression, basis expansions like splines or Fourier—rely on explicitly choosing a **set of basis functions** ahead of time. In doing so, we encode our assumptions about the structure of the data into the model.

But what if we want to make fewer assumptions? What if we let the data speak for itself?

This is the philosophy of **non-parametric regression**. Rather than fitting a global equation or assembling basis functions, these methods directly estimate the target function from data points—**locally**, **adaptively**, and often with minimal human intervention.

Non-parametric models are flexible because they don’t constrain the shape of the function to a fixed formula. Instead, they learn the function’s shape entirely from the data, using notions of **proximity** and **local averaging**.

Let’s explore two major families:

* **Kernel Smoothing**, where predictions are local weighted averages.
* **LOESS/LOWESS**, which fits polynomials locally around each query point.

---

## 1. Kernel Smoothing: Gentle Weighted Averages

One of the simplest yet most elegant non-parametric regression techniques is **kernel smoothing**, also known as **Nadaraya–Watson regression**.

### Concept

Given a query point $x$, the idea is to **predict its output by averaging nearby observations**, weighted by how close they are to $x$.

Mathematically, the predicted value $\hat{y}(x)$ is:


$$
\hat{y}(x) = \frac{\sum_{i=1}^n K\left( \frac{x - x_i}{h} \right) y_i}{\sum_{i=1}^n K\left( \frac{x - x_i}{h} \right)}
$$


Here:

* $K(\cdot)$ is a **kernel function**, which assigns higher weights to points near $x$.
* $h$ is the **bandwidth parameter**, controlling how local or global the smoothing is.

### Common Kernels

* **Gaussian**: $K(u) = \exp(-\frac{u^2}{2})$
* **Epanechnikov**: $K(u) = \max(0, 1 - u^2)$
* **Uniform**: All points within a window are weighted equally.

The choice of kernel is less important than the choice of **bandwidth** $h$. A small $h$ leads to more localized models (high variance), while a large $h$ produces smooth global models (high bias).

### Intuition

If you’re predicting stock price on day $x$, kernel smoothing tells you:

> "Look at stock prices on days near $x$, and average them—but give more weight to days that are closer."

---

### Python Example: Kernel Ridge Regression

One way to implement kernel smoothing is through **Kernel Ridge Regression** with an RBF kernel.

```python
from sklearn.kernel_ridge import KernelRidge

# RBF kernel acts like a Gaussian smoother
kernel_reg = KernelRidge(kernel="rbf", gamma=0.1)
kernel_reg.fit(X_train.reshape(-1, 1), y_train)

# Predict
y_pred = kernel_reg.predict(X_test.reshape(-1, 1))
```

* `kernel="rbf"` uses the Gaussian radial basis function.
* `gamma` controls the inverse bandwidth: smaller gamma = smoother curve.

While technically a regularized method, when the regularization is small, it behaves like a classic Nadaraya–Watson smoother.

---

### Use Cases

* **Stock price trends**: averaging nearby days to estimate tomorrow’s price.
* **Environmental monitoring**: smoothing air quality or temperature readings.
* **Denoising**: removing local noise in sensor data.

---

## 2. LOESS / LOWESS: Local Polynomial Regression

Whereas kernel smoothing fits **constant values** around each query point, **LOESS** (Locally Estimated Scatterplot Smoothing) fits **small polynomials** locally.

### Concept

LOESS works as follows:

1. For each query point $x$, select its neighboring data points.
2. Assign weights to the neighbors based on distance to $x$.
3. Fit a low-degree polynomial (typically linear or quadratic) to these points.
4. Use the fitted polynomial to predict $\hat{y}(x)$.

This process is repeated **at each query point**, meaning LOESS is **adaptive** to the local structure of the data.

### Mathematical Formulation

For each query point $x$, we solve:


$$
\min_{\beta} \sum_{i=1}^n w_i(x) \left( y_i - \beta_0 - \beta_1 x_i - \beta_2 x_i^2 \right)^2
$$


Where:

* $w_i(x)$ is the weight assigned to point $x_i$, based on distance to $x$.
* The polynomial is typically linear or quadratic.

### Why LOESS?

* It captures **local curvature** in the data.
* It naturally adjusts to heteroscedasticity (non-constant variance).
* It’s **robust to outliers** with a variant called **LOWESS** (locally weighted scatterplot smoothing).

---

### Python Example: LOWESS in statsmodels

```python
from statsmodels.nonparametric.smoothers_lowess import lowess

# frac determines the proportion of points used for each local regression
smoothed = lowess(y_train, X_train.flatten(), frac=0.3)

# smoothed is a 2D array: [ [x1, y1_smoothed], [x2, y2_smoothed], ... ]
```

* `frac=0.3` means each prediction uses 30% of the data.
* You can interpolate this for new values using `scipy.interpolate`.

---

### Use Cases

* **Biomedical signals**: modeling heart rate or growth curves.
* **Econometrics**: smoothing noisy economic indicators.
* **Exploratory data analysis**: visually revealing trends without assumptions.

---

## Comparing Methods

Let’s see how these two methods differ in their properties:

<div style="overflow-x: auto;">
<table class="simple-table">
  <thead>
    <tr>
      <th>Aspect</th>
      <th>Kernel Smoothing</th>
      <th>LOESS / LOWESS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Prediction type</td>
      <td>Local average</td>
      <td>Local polynomial fit</td>
    </tr>
    <tr>
      <td>Captures curvature</td>
      <td>No (unless via kernel shape)</td>
      <td>Yes (via polynomial fit)</td>
    </tr>
    <tr>
      <td>Outlier resistance</td>
      <td>Low</td>
      <td>High (in robust LOWESS)</td>
    </tr>
    <tr>
      <td>Computation</td>
      <td>Fast, matrix-based</td>
      <td>Slower, point-wise fitting</td>
    </tr>
    <tr>
      <td>Main parameter</td>
      <td>Bandwidth (or gamma)</td>
      <td>Fraction of data (frac)</td>
    </tr>
  </tbody>
</table>
</div>

---

## Practical Considerations

* **Model interpretability** is limited: these methods do not yield a global equation.
* **No extrapolation**: both methods fail outside the domain of training data.
* **Efficiency**: LOESS can be slow for large datasets since it fits a model per point.
* **Combining with other methods**: You can use these smoothers for residual correction, pre-processing, or visualization.

---

## Summary

Non-parametric methods offer a radically different modeling approach: they **fit the data where it is**, without imposing a rigid structure. They are ideal when you:

* Have enough data to learn local patterns.
* Don’t want to commit to a fixed global functional form.
* Need flexible, visual, or robust smoothing for exploration or presentation.

Kernel smoothing and LOESS may not generalize well far from data or scale easily to high dimensions, but within their domain, they provide an elegant and often surprisingly effective modeling tool.

---

# Feature Engineering in Nonlinear Regression: Building the Right Inputs

Non-linearity in modeling doesn’t always require changing the algorithm. Sometimes, it’s the **inputs** that need to change.

While we’ve explored powerful ways to capture non-linearities—polynomials, splines, Fourier series, kernel smoothers—these methods often rely heavily on **how you prepare your data**. That’s where **feature engineering** comes in. Thoughtful transformation of inputs can convert an inexpressive model into a predictive powerhouse.

This section outlines key feature engineering strategies specifically tuned to non-linear regression problems.

---

## Interaction Terms: Capturing Synergy

In many real-world problems, the effect of one variable depends on the value of another. These **interaction effects** are often missed by linear models unless explicitly modeled.

### Example

* Suppose your dataset includes `marketing_spend` and `region`.
* The effectiveness of spend might depend on the region—urban vs. rural.

By introducing an **interaction term** like:

$$
\text{interaction} = \text{marketing_spend} \times \text{region}
$$

you allow the model to capture this synergy.

### How to Engineer Interactions

Using `PolynomialFeatures` from scikit-learn:

```python
from sklearn.preprocessing import PolynomialFeatures

# Only generate interaction terms (no squares or higher-order powers)
interactor = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interactions = interactor.fit_transform(X)
```

This transformation automatically generates all pairwise multiplicative combinations of features.

> Use interaction terms when your domain knowledge or residual plots suggest **conditional relationships** between variables.

---

## Preprocessing: Standardization for Numerical Stability

When you include polynomial or interaction terms—especially those involving squares or cross-multiplications—numerical instability can creep in. Features with larger scales dominate the loss function, leading to biased or erratic parameter estimation.

**Solution**: Standardize the features before expansion.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge())
])
pipe.fit(X_train, y_train)
```

> Always apply scaling **before** generating polynomial or interaction terms to ensure all terms are on comparable scales.

---

## Diagnostics: Know When You’re Missing Non-Linearity

When modeling non-linear data with linear tools—or poorly chosen transformations—it’s easy to miss structure. A few key diagnostics help detect this:

### 1. **Residual Plots**

* Plot residuals vs. predicted values.
* Non-random patterns (e.g., U-shapes, funnels) indicate unmodeled non-linearity.

### 2. **Train vs. Test Error Curves**

* If test error decreases then sharply increases with model complexity, you’re likely overfitting.
* If both errors are high, you may be underfitting due to missing features or non-linear patterns.

### 3. **Partial Dependence Plots (PDP)**

* Visualize marginal effects of features in tree-based or GAM models.
* If shapes are non-linear, that’s your cue to use non-linear feature transformations.

---

## Regularization: Keeping Expansion in Check

Non-linear feature expansions—especially polynomials and interactions—can lead to **feature explosion**, with hundreds or thousands of terms. This introduces multicollinearity, overfitting, and numerical instability.

### Use Ridge or Lasso

* **Ridge regression** shrinks all coefficients but keeps them in the model.
* **Lasso regression** performs variable selection by setting some coefficients to zero.

```python
from sklearn.linear_model import RidgeCV

ridge_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
ridge_model.fit(X_poly, y)
```

> Combine `PolynomialFeatures` with `RidgeCV` or `LassoCV` for robust, scalable modeling.

---

## Real-World Perspective: Why Splines Often Win

In applied fields like **energy modeling**, high-degree polynomials are rarely the right choice. They tend to overfit, especially when usage patterns vary locally (e.g., air conditioning spikes in hot hours).

**Splines**, on the other hand:

* Model usage patterns locally and adaptively.
* Are less sensitive to outliers.
* Do not oscillate wildly outside the data range.

For example:

* Modeling electricity demand vs. temperature: splines can capture **non-linear thresholds**, like sudden increases in energy use above 30°C.

> In many business-critical models, spline features—generated via libraries like `patsy` or `pyGAM`—strike the right balance between interpretability and flexibility.

---

## Summary: Curating Your Inputs for Expressive Power

<div style="overflow-x: auto;">
<table class="simple-table">
  <thead>
    <tr>
      <th>Strategy</th>
      <th>Purpose</th>
      <th>When to Use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Interaction terms</strong></td>
      <td>Capture conditional relationships</td>
      <td>When one variable moderates another</td>
    </tr>
    <tr>
      <td><strong>Standardization</strong></td>
      <td>Prevent large-scale features from dominating</td>
      <td>Before polynomial or kernel expansion</td>
    </tr>
    <tr>
      <td><strong>Regularization</strong></td>
      <td>Control overfitting from feature explosion</td>
      <td>Always with polynomial/basis features</td>
    </tr>
    <tr>
      <td><strong>Residual diagnostics</strong></td>
      <td>Detect unmodeled structure or non-linearity</td>
      <td>In any modeling loop</td>
    </tr>
    <tr>
      <td><strong>Splines</strong></td>
      <td>Localized non-linear modeling</td>
      <td>For energy, biomed, economics, etc.</td>
    </tr>
  </tbody>
</table>
</div>


---

Feature engineering for non-linear regression is not about throwing math at a wall—it’s about transforming raw data into representations that a model can understand. Whether it’s creating the right interaction, choosing the right basis, or knowing when to stop—engineering your inputs is half the modeling battle.


---

# Practical Considerations: Choosing the Right Nonlinear Modeling Strategy

Non-linearity is everywhere—in customer behavior, biological signals, mechanical systems, and beyond. But modeling it effectively is as much about **strategy** as it is about mathematics.

You now have a toolbox full of nonlinear techniques: **polynomial regression**, **basis expansions** (splines, Fourier, RBF), **kernel smoothers**, and **local regression**. But when should you reach for which? What should guide your modeling decisions?

Let’s bring clarity to this decision space by exploring practical considerations from multiple angles.

---

## 1. Accuracy vs. Interpretability

The more expressive your model becomes, the more you risk sacrificing transparency.

<div style="overflow-x: auto;">
<table class="simple-table">
  <thead>
    <tr>
      <th>Method</th>
      <th>Accuracy Potential</th>
      <th>Interpretability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Linear Regression</td>
      <td>Low (on nonlinear data)</td>
      <td>High</td>
    </tr>
    <tr>
      <td>Polynomial Regression</td>
      <td>Moderate (at right d)</td>
      <td>Medium (low for high d)</td>
    </tr>
    <tr>
      <td>Splines</td>
      <td>High</td>
      <td>Medium–High (locally)</td>
    </tr>
    <tr>
      <td>Fourier Basis</td>
      <td>High (for periodicity)</td>
      <td>Medium</td>
    </tr>
    <tr>
      <td>Kernel Smoothing</td>
      <td>High (within domain)</td>
      <td>Low</td>
    </tr>
    <tr>
      <td>LOESS / LOWESS</td>
      <td>Very High (local)</td>
      <td>Low (no global formula)</td>
    </tr>
  </tbody>
</table>
</div>


**Takeaway**: If you need model explanations (e.g., regulated industries), favor **splines or low-degree polynomials**. If accuracy is your sole metric, **LOESS or kernel methods** might be better—but you may lose interpretability.

---

## 2. Data Size and Dimensionality

Some nonlinear models scale poorly with either **number of observations** or **number of features**.

<div style="overflow-x: auto;">
<table class="simple-table">
  <thead>
    <tr>
      <th>Method</th>
      <th>Handles Large n?</th>
      <th>Handles High d?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Polynomial Regression</td>
      <td>✓ (small–medium n)</td>
      <td>✗ (feature explosion)</td>
    </tr>
    <tr>
      <td>Splines</td>
      <td>✓ (moderate n)</td>
      <td>✓ (with care)</td>
    </tr>
    <tr>
      <td>Fourier Basis</td>
      <td>✓</td>
      <td>✗ (usually univariate)</td>
    </tr>
    <tr>
      <td>Kernel Smoothing</td>
      <td>✗ (slow at large n)</td>
      <td>✓ (with kernel tricks)</td>
    </tr>
    <tr>
      <td>LOESS / LOWESS</td>
      <td>✗ (O(n²) cost)</td>
      <td>✗</td>
    </tr>
  </tbody>
</table>
</div>


**Takeaway**: Use **splines or ridge-regularized polynomials** for structured, scalable modeling. Avoid LOESS or naïve kernel smoothing on large datasets.

---

## 3. Domain Assumptions

The more you know about your data’s structure, the more efficient your modeling.

<div style="overflow-x: auto;">
<table class="simple-table">
  <thead>
    <tr>
      <th>Scenario</th>
      <th>Suggested Model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Repeating cycles / seasonality</td>
      <td>Fourier basis</td>
    </tr>
    <tr>
      <td>Rapid changes in certain regions</td>
      <td>Splines or RBF basis</td>
    </tr>
    <tr>
      <td>Smooth global trends</td>
      <td>Polynomial regression (low degree)</td>
    </tr>
    <tr>
      <td>Localized, noisy signals</td>
      <td>LOESS or kernel smoothing</td>
    </tr>
    <tr>
      <td>Feature interactions dominate</td>
      <td>PolynomialFeatures + Regularization</td>
    </tr>
  </tbody>
</table>
</div>


**Takeaway**: Structure your modeling pipeline around domain signals. If seasonality dominates, don’t reach for polynomials—use Fourier or additive models.

---

## 4. Extrapolation Behavior

Linear and polynomial models can extrapolate beyond the training data. Most non-parametric methods **cannot**.

<div style="overflow-x: auto;">
<table class="simple-table">
  <thead>
    <tr>
      <th>Method</th>
      <th>Can Extrapolate?</th>
      <th>Behavior Outside Training Range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Linear Regression</td>
      <td>Yes</td>
      <td>Linear trend continues</td>
    </tr>
    <tr>
      <td>Polynomial Regression</td>
      <td>Yes (wildly)</td>
      <td>Often unstable, oscillates</td>
    </tr>
    <tr>
      <td>Splines (Natural)</td>
      <td>Limited</td>
      <td>Linear tails (by constraint)</td>
    </tr>
    <tr>
      <td>Fourier Basis</td>
      <td>Yes (repeats)</td>
      <td>Periodic extension</td>
    </tr>
    <tr>
      <td>Kernel Smoothing</td>
      <td>No</td>
      <td>Undefined or flattens out</td>
    </tr>
    <tr>
      <td>LOESS / LOWESS</td>
      <td>No</td>
      <td>Constant or unreliable outside domain</td>
    </tr>
  </tbody>
</table>
</div>


**Takeaway**: If extrapolation is essential (e.g., forecasting future behavior), use models that **extend smoothly**: natural splines, linear models, or hybrid systems.

---

## 5. Regularization is Your Friend

With power comes risk. Any nonlinear model that expands features (especially polynomial or RBF bases) needs **regularization** to prevent overfitting.

### Techniques:

* **Ridge regression** for shrinkage and stability.
* **Lasso regression** for sparse, interpretable solutions.
* **Elastic Net** for a balanced tradeoff.

```python
from sklearn.linear_model import ElasticNetCV

enet = ElasticNetCV(l1_ratio=0.5, alphas=[0.01, 0.1, 1.0])
enet.fit(X_poly, y)
```

**Takeaway**: The more terms you generate (via interaction or basis expansion), the more aggressively you should regularize.

---

## 6. Visualization and Diagnostics

Nonlinear modeling invites **illusion of fit**. Always validate with:

* **Residual plots**: look for systematic trends or curvature.
* **Train/test curves**: check for overfitting.
* **Cross-validation**: robustly estimate generalization error.
* **Partial dependence plots**: understand variable effects in black-box models.

```python
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Prediction vs Actual")
```

Never trust a pretty curve unless you’ve validated its stability and generalizability.

---

## 7. Additive Models (Preview)

If you want the structure of basis expansions with the flexibility of smoothness control, **Generalized Additive Models (GAMs)** provide a middle ground:

$$
y = \beta_0 + f_1(x_1) + f_2(x_2) + \dots + f_p(x_p) + \varepsilon
$$

Each $f_j$ is a smooth, non-linear function—often a spline or kernel smoother.

* They handle **non-linearity per feature** without modeling interactions.
* They remain **interpretable** due to their additive structure.

We'll cover GAMs in a dedicated future post.

---

## Summary Table: Nonlinear Modeling Cheat Sheet

<div style="overflow-x: auto;">
<table class="simple-table">
  <thead>
    <tr>
      <th>Method</th>
      <th>Strengths</th>
      <th>Weaknesses</th>
      <th>When to Use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Polynomial Regression</td>
      <td>Simple, fast, closed-form</td>
      <td>Overfits, extrapolates poorly</td>
      <td>Low-order trends, small datasets</td>
    </tr>
    <tr>
      <td>Splines</td>
      <td>Local control, interpretable</td>
      <td>Requires knot choice</td>
      <td>Growth curves, energy models</td>
    </tr>
    <tr>
      <td>Fourier Basis</td>
      <td>Perfect for periodic data</td>
      <td>Inapplicable for non-periodic domains</td>
      <td>Seasonal trends, time-series</td>
    </tr>
    <tr>
      <td>Kernel Smoothing</td>
      <td>No assumptions, adaptive</td>
      <td>Slow on large data, no extrapolation</td>
      <td>Denoising, exploratory analysis</td>
    </tr>
    <tr>
      <td>LOESS / LOWESS</td>
      <td>Flexible, locally adaptive</td>
      <td>O(n²), hard to scale</td>
      <td>Bio curves, data visualization</td>
    </tr>
  </tbody>
</table>
</div>

---


There is no universal best method for modeling non-linearity. The best tool is the one that matches your **data scale**, **domain intuition**, **interpretability needs**, and **computational constraints**.

---

# Wrapping Up

Linear regression teaches us discipline. Non-linear regression teaches us creativity.

In this post, we’ve stepped beyond the straight line—exploring how data behaves when it bends, twists, or responds in subtle gradients. We’ve seen how polynomial regression can curve modestly, how basis expansions can build expressive shapes from structured ingredients, and how non-parametric methods like LOESS and kernel smoothing allow the data itself to define its path.

But these tools are not merely mathematical flourishes. They are design choices.

Choosing the right method isn’t just a technical exercise—it’s a question of **interpretability vs. flexibility**, **locality vs. generality**, and **bias vs. variance**. Modeling non-linearity is about knowing when to trust the model, when to trust the data, and when to let them speak to each other.

More often than not, it’s not the algorithm that fails—it’s the assumptions we make about the world the data lives in. The models we explored aren’t just function approximators—they are lenses, and each lens highlights different facets of the truth.

Whether you're modeling the rise and fall of a stock price, the arc of a child’s growth, or the seasonal dance of demand curves, the lesson is clear:

> **Non-linearity is not a complication to fear. It’s a structure to uncover.**

If you've made it this far, you now hold a versatile toolkit. Use it not just to predict, but to understand.

Let the curve teach you something.
