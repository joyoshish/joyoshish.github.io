---
layout: post
title: "Generalized Linear Models: A Unified Framework for Modern Regression"
date: 2025-02-08
description: "A thorough guide to Generalized Linear Models (GLMs)—covering exponential family distributions, link functions, and models like logistic, Poisson, and negative binomial regression. Learn GLM structure, MLE, IRLS optimization, and model diagnostics for diverse response types."
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


## Why Go Beyond Linear Regression?

Linear regression is one of the most foundational models in statistics and machine learning. It assumes a linear relationship between input features and the output variable, with errors that are normally distributed, independent, and homoscedastic. For many real-world problems, this model serves as a reliable starting point.

But what happens when the assumptions break?

Suppose you're trying to model:

* Whether a customer will click on an ad (yes or no)
* How many times a patient visits a hospital in a year
* The number of purchases a user makes in a session
* Whether a transaction is fraudulent

These are **not** problems where the response variable is continuous and normally distributed. Instead:

* A binary outcome like *click* or *no click* violates the assumption of continuous and Gaussian residuals.
* A count outcome like *number of visits* is inherently discrete and non-negative, and its variance typically increases with the mean.
* Outcomes such as *claim amounts* or *duration until an event* may be right-skewed or heavy-tailed.

If we apply standard linear regression to such problems, several issues arise:

* Predictions can fall outside the feasible range (e.g., predicting probabilities less than 0 or more than 1)
* The model may assign constant variance to heteroscedastic data
* Estimations and inference become unreliable and misleading

In short, **linear regression is not flexible enough** to handle response variables with **non-Gaussian characteristics**. Forcing it onto such data can lead to poor predictive performance and invalid interpretations.

This motivates a broader question:

**Can we build a general modeling framework that retains the interpretability of linear models, but adapts to different types of outcome variables?**

The answer is yes—and it leads us directly to the framework of **Generalized Linear Models (GLMs)**.

GLMs extend linear regression to a much wider class of problems, including classification, count modeling, and more, by:

* Allowing the response variable to follow any distribution from the **exponential family**
* Introducing a **link function** that connects the expected value of the response to a linear predictor

This framework not only preserves the elegant structure of linear models, but also adapts to the stochastic nature of real-world data. In the next section, we’ll build this structure from the ground up.

## The GLM Framework: A Unified View of Regression

At its heart, the Generalized Linear Model (GLM) is a principled extension of linear regression. It retains the essential interpretability and structure of linear models while generalizing them to handle a much broader class of response variables.

A GLM is defined by **three core components**:

---

### 1. Random Component: Modeling the Response Variable

In classical linear regression, we assume that the response variable $$y$$ follows a normal distribution with constant variance:


$$
y \sim \mathcal{N}(\mu, \sigma^2)
$$


In GLMs, this assumption is replaced by a much more flexible condition:
**The response variable is drawn from the exponential family of distributions.**

The exponential family has a canonical form:


$$
p(y; \theta, \phi) = \exp\left( \frac{y \cdot \theta - b(\theta)}{\phi} + c(y, \phi) \right)
$$


Where:

* $$\theta$$ is the **canonical (natural) parameter**
* $$\phi$$ is the **dispersion parameter** (fixed in many GLMs, e.g., Poisson)
* $$b(\theta)$$ is a cumulant function ensuring proper normalization
* $$c(y, \phi)$$ adjusts for the specific form of the response variable

Many commonly used distributions are special cases of this form:

<div style="overflow-x: auto;">
  <table class="simple-table">
    <thead>
      <tr>
        <th>Distribution</th>
        <th>Support of $$y$$</th>
        <th>Mean $$\mu$$</th>
        <th>Variance</th>
        <th>Canonical Parameter $$\theta$$</th>
        <th>Canonical Link</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Gaussian</td>
        <td>$$\mathbb{R}$$</td>
        <td>$$\mu$$</td>
        <td>$$\sigma^2$$</td>
        <td>$$\theta = \mu$$</td>
        <td>Identity</td>
      </tr>
      <tr>
        <td>Bernoulli</td>
        <td>$$\{0,1\}$$</td>
        <td>$$\mu = p$$</td>
        <td>$$p(1-p)$$</td>
        <td>$$\theta = \log\left(\frac{p}{1 - p}\right)$$</td>
        <td>Logit</td>
      </tr>
      <tr>
        <td>Poisson</td>
        <td>$$\mathbb{N}_0$$</td>
        <td>$$\mu$$</td>
        <td>$$\mu$$</td>
        <td>$$\theta = \log(\mu)$$</td>
        <td>Log</td>
      </tr>
      <tr>
        <td>Gamma</td>
        <td>$$\mathbb{R}_+$$</td>
        <td>$$\mu$$</td>
        <td>$$\mu^2 / k$$</td>
        <td>$$\theta = -1/\mu$$</td>
        <td>Inverse</td>
      </tr>
    </tbody>
  </table>
</div>

This generalization allows us to work with a variety of data types—binary, count, skewed continuous—within a single coherent modeling framework.

---

### 2. Systematic Component: The Linear Predictor

Just like in linear regression, GLMs assume that the **input features** affect the outcome via a linear combination. We define the linear predictor as:


$$
\eta = \mathbf{Xw}
$$


Where:

* $$\mathbf{X} \in \mathbb{R}^{n \times p}$$ is the **design matrix** of inputs
* $$\mathbf{w} \in \mathbb{R}^p$$ is the **coefficient vector**
* $$\eta \in \mathbb{R}^n$$ is the **linear predictor**, one per data point

This linear predictor serves as the bridge between input features and the modeled outcome.

---

### 3. Link Function: Connecting Mean to Linear Predictor

In most GLMs, the **mean** of the response variable, $$\mu = \mathbb{E}[y]$$, is **not** equal to the linear predictor $$\eta$$. Instead, they are connected via a **link function** $$g(\cdot)$$:


$$
g(\mu) = \eta = \mathbf{Xw}
$$


Equivalently, we write:


$$
\mu = g^{-1}(\eta)
$$


The link function transforms the expected value of the response into the linear scale of predictors. The choice of the link depends on both the distribution and interpretability. The **canonical link** for each distribution results in convenient mathematical properties, particularly for maximum likelihood estimation.

---

### Visualizing the GLM Generalization

Let's take a moment to visualize what's happening conceptually.

* In **linear regression**, we directly model the mean as a linear combination:
  $$\mu = \eta = \mathbf{Xw}$$
  Here, the link function is the **identity**: $$g(\mu) = \mu$$.

* In **logistic regression**, we model the probability of a binary outcome:
  $$\mu = \mathbb{E}[y] = \mathbb{P}(y = 1)$$
  We relate this to the linear predictor via the **logit link**:
  $$g(\mu) = \log\left(\frac{\mu}{1 - \mu}\right) = \eta$$.

* In **Poisson regression**, we model count outcomes:
  $$\mu = \mathbb{E}[y] \in \mathbb{R}_+$$
  The link is the **log function**:
  $$g(\mu) = \log(\mu) = \eta$$.

Thus, linear regression is simply a **special case** of GLMs where:

* The distribution is Gaussian
* The link is identity

GLMs generalize this by allowing flexible response distributions and appropriate transformations through the link.

---


## The Exponential Family of Distributions

The cornerstone of Generalized Linear Models is the assumption that the response variable comes from a **distribution within the exponential family**. This family is large and expressive—it includes many common distributions used in regression, classification, and count modeling.

But what exactly is the exponential family?

---

### General Form

A probability distribution belongs to the exponential family if it can be written in the following canonical form:


$$
p(y; \theta, \phi) = \exp\left( \frac{y \cdot \theta - b(\theta)}{\phi} + c(y, \phi) \right)
$$


Where:

* $$y$$ is the observed response variable
* $$\theta$$ is the **natural (canonical) parameter**
* $$\phi$$ is the **dispersion parameter** (fixed in some distributions)
* $$b(\theta)$$ is the **log-partition function**, which ensures the distribution integrates to one
* $$c(y, \phi)$$ is a function of the data and dispersion

This decomposition might look abstract at first, so let's walk through it.

---

### Intuition Behind Each Term

* **Natural parameter $$\theta$$**: This is the parameter that links directly to the sufficient statistic (here, $$y$$). The form $$y \cdot \theta$$ ensures that the distribution is *linear in the sufficient statistic*.
* **Log-partition function $$b(\theta)$$**: This term normalizes the distribution. It also plays a crucial role in determining the mean and variance.
* **Dispersion parameter $$\phi$$**: It controls the spread or scale of the distribution. In many GLMs (like Poisson or Bernoulli), $$\phi = 1$$ by convention.
* **Function $$c(y, \phi)$$**: Ensures the entire function is properly normalized in terms of the observed data.

---

### Why Use the Exponential Family?

The key advantage is **tractability**. Distributions in this family allow for:

* Convex log-likelihoods
* Analytical derivatives (score functions, Hessians)
* Clean expressions for mean and variance

And most importantly, they yield GLMs that are easy to optimize using maximum likelihood methods.

---

### Computing Mean and Variance

For any distribution in the exponential family:

* The **mean** is given by:

  
  $$
  \mu = \mathbb{E}[y] = b'(\theta)
  $$
  

* The **variance** is:

  
  $$
  \text{Var}(y) = \phi \cdot b''(\theta)
  $$
  

So, once we know $$b(\theta)$$, we can derive the full behavior of the distribution in terms of mean and variance.

---

### Examples of Exponential Family Members

Let's see how some familiar distributions fit into this general form.

<div style="overflow-x: auto;">
  <table class="simple-table">
    <thead>
      <tr>
        <th>Distribution</th>
        <th>Support</th>
        <th>$$\theta$$ (Canonical Parameter)</th>
        <th>$$b(\theta)$$</th>
        <th>$$\phi$$</th>
        <th>Mean $$\mu$$</th>
        <th>Variance</th>
        <th>Canonical Link</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Gaussian</td>
        <td>$$\mathbb{R}$$</td>
        <td>$$\theta = \mu$$</td>
        <td>$$b(\theta) = \frac{\theta^2}{2}$$</td>
        <td>$$\sigma^2$$</td>
        <td>$$\mu$$</td>
        <td>$$\sigma^2$$</td>
        <td>Identity</td>
      </tr>
      <tr>
        <td>Bernoulli</td>
        <td>$$\{0, 1\}$$</td>
        <td>$$\theta = \log\left(\frac{p}{1-p}\right)$$</td>
        <td>$$b(\theta) = \log(1 + e^\theta)$$</td>
        <td>1</td>
        <td>$$\mu = p$$</td>
        <td>$$p(1 - p)$$</td>
        <td>Logit</td>
      </tr>
      <tr>
        <td>Poisson</td>
        <td>$$\mathbb{N}_0$$</td>
        <td>$$\theta = \log(\lambda)$$</td>
        <td>$$b(\theta) = e^\theta$$</td>
        <td>1</td>
        <td>$$\mu = \lambda$$</td>
        <td>$$\lambda$$</td>
        <td>Log</td>
      </tr>
      <tr>
        <td>Gamma</td>
        <td>$$\mathbb{R}_+$$</td>
        <td>$$\theta = -\frac{1}{\mu}$$</td>
        <td>$$b(\theta) = -\log(-\theta)$$</td>
        <td>$$\phi = \frac{1}{k}$$</td>
        <td>$$\mu$$</td>
        <td>$$\mu^2 / k$$</td>
        <td>Inverse</td>
      </tr>
    </tbody>
  </table>
</div>

---

### Beyond the Big Four: Mentioning Tweedie Models

The **Tweedie family** is a more general class that spans distributions between Poisson and Gamma. It includes:

* Normal: variance $$\propto \mu^0$$
* Poisson: variance $$\propto \mu^1$$
* Gamma: variance $$\propto \mu^2$$

While Tweedie models are beyond the scope of basic GLMs, they offer powerful extensions in fields like insurance and energy forecasting, where distributions are often zero-inflated and right-skewed.

---

## Choosing the Right Link Function

The **link function** is what makes Generalized Linear Models truly flexible and powerful. It acts as a bridge between the expected value of the response variable and the linear predictor. While the linear predictor $$\eta = \mathbf{Xw}$$ remains the same across all GLMs, the transformation applied through the link function determines how the model adapts to different types of data.

Mathematically, the link function is defined as:


$$
g(\mu) = \eta = \mathbf{Xw}
$$


Where:

* $$\mu = \mathbb{E}[y]$$ is the mean of the response variable
* $$g(\cdot)$$ is a monotonic, differentiable function that maps the range of $$\mu$$ into the entire real line

Choosing the right link function ensures:

* The output of the model lies within the valid domain of the response
* Interpretability aligns with the nature of the problem
* Estimation becomes tractable

Let's explore some common link functions and where they are used.

---

### Identity Link

This is the simplest possible link function:


$$
g(\mu) = \mu
$$

So that:

$$
\mu = \eta = \mathbf{Xw}
$$


Used in:

* **Linear Regression**

This link assumes that the mean of the response can take any real value, making it suitable for continuous outputs that are approximately normally distributed.

It is also the **canonical link** for the Gaussian distribution.

---

### Logit Link

The logit function is defined as:


$$
g(\mu) = \log\left( \frac{\mu}{1 - \mu} \right)
$$

Which implies:

$$
\mu = \frac{1}{1 + e^{-\eta}} = \sigma(\eta)
$$


Used in:

* **Logistic Regression** (for binary classification)

This link maps probabilities (which lie in the interval $$[0, 1]$$) to the real line, allowing us to model binary outcomes while ensuring predictions remain valid probabilities.

The logit is the **canonical link** for the Bernoulli distribution.

---

### Probit Link

The probit link uses the inverse CDF of the standard normal distribution:


$$
g(\mu) = \Phi^{-1}(\mu)
$$

So that:

$$
\mu = \Phi(\eta)
$$


Where $$\Phi(\cdot)$$ is the CDF of the standard normal distribution.

Used in:

* **Probit Regression** (alternative to logistic regression)

While both logit and probit provide sigmoid-shaped mappings, the probit link assumes latent Gaussian noise and is often favored in Bayesian settings due to conjugacy with Gaussian priors.

Unlike the logit, the probit link is **not** canonical for the Bernoulli distribution.

---

### Log Link

The log function is defined as:


$$
g(\mu) = \log(\mu)
$$

Which implies:

$$
\mu = \exp(\eta)
$$


Used in:

* **Poisson Regression** (for count data)
* **Gamma Regression** (for skewed positive data)

The log link ensures that the predicted mean $$\mu$$ remains strictly positive, which is critical when modeling counts or non-negative continuous quantities.

This is the **canonical link** for the Poisson distribution.

---

### Canonical vs Non-Canonical Links

Each exponential family distribution has a **canonical link function** that simplifies the mathematics of estimation, particularly in deriving the score function and Hessian matrix. However, one is **not restricted** to the canonical link. Non-canonical links can be useful when:

* Interpretation is more natural in another scale (e.g., using identity link for binomial proportions)
* Robustness or regularization properties differ
* Prior distributions align better (in Bayesian modeling)

The choice of link affects:

* **Interpretability**: Odds ratios (logit), probabilities (identity), hazard rates (log)
* **Computational behavior**: Canonical links often lead to faster convergence in estimation
* **Model fit**: Some datasets might prefer a non-canonical link for better predictive performance

---

To summarize, the link function governs how the model expresses the relationship between the inputs and the expected output. The selection should reflect both the distributional assumptions and the practical meaning of model coefficients.

---


## Examples of Common GLMs

The power of the Generalized Linear Model framework lies in its versatility. By appropriately choosing a **distribution** from the exponential family and selecting a suitable **link function**, we can derive a wide range of models tailored to different types of response variables.

In this section, we will revisit several well-known models—some you've likely encountered before—as special cases of GLMs. This unified perspective not only clarifies their assumptions and limitations but also illuminates how they are all structurally similar under the GLM umbrella.

We begin with the most familiar of all: **Linear Regression**.

---

### 1. Linear Regression (Revisiting OLS as a GLM)

To appreciate the power and generality of Generalized Linear Models, it helps to start with something familiar: **Ordinary Least Squares (OLS)** linear regression.

Though often treated as a separate classical method, linear regression fits naturally into the GLM framework—it's simply the case where:

* The **response variable** is assumed to follow a **Gaussian distribution**
* The **link function** is the **identity**, i.e., $$g(\mu) = \mu$$

Let's walk through this explicitly.

---

#### The Classical Setup

In standard linear regression, we model the response as:


$$
y = \mathbf{x}^\top \mathbf{w} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)
$$


For a dataset with $$n$$ observations and $$p$$ features, the full model becomes:


$$
\mathbf{y} = \mathbf{Xw} + \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})
$$


We assume the noise terms are:

* **Independent**
* **Identically distributed**
* **Normally distributed** with mean 0 and constant variance $$\sigma^2$$

This leads to the familiar **least squares objective**:


$$
\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{Xw}\|_2^2
$$


The closed-form solution is given by the **normal equation**:


$$
\hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$


---

#### GLM Perspective: Expressing OLS as a Generalized Linear Model

Let's now reinterpret this through the GLM lens.

* **Distribution (Random Component):** We assume each $$y_i$$ is drawn independently from a normal distribution:

  
  $$
  y_i \sim \mathcal{N}(\mu_i, \sigma^2)
  $$
  

* **Linear Predictor (Systematic Component):** The mean is modeled as:

  
  $$
  \mu_i = \eta_i = \mathbf{x}_i^\top \mathbf{w}
  $$
  

* **Link Function:** We use the **identity link**, so that:

  
  $$
  g(\mu_i) = \mu_i = \eta_i
  $$  
  

This satisfies all the conditions of a GLM:

* Response from exponential family (Normal distribution is a member)
* Linear predictor
* Link function

---

#### Connection to Maximum Likelihood Estimation

To cement the GLM connection, let's derive the OLS solution from **maximum likelihood**.

We assume each response $$y_i$$ is distributed as:


$$
p(y_i \mid \mu_i, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(y_i - \mu_i)^2}{2\sigma^2} \right)
$$


Since $$\mu_i = \mathbf{x}_i^\top \mathbf{w}$$, the **log-likelihood** becomes:


$$
\log \mathcal{L}(\mathbf{w}) = \sum_{i=1}^n \log p(y_i \mid \mathbf{x}_i, \mathbf{w}) = -\frac{n}{2} \log(2\pi \sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - \mathbf{x}_i^\top \mathbf{w})^2
$$


Maximizing the likelihood is equivalent to minimizing the negative log-likelihood, which simplifies to:


$$
\min_{\mathbf{w}} \sum_{i=1}^n (y_i - \mathbf{x}_i^\top \mathbf{w})^2
$$


This is precisely the **least squares** objective. So, from a GLM standpoint, linear regression is not just algebraically neat—it is also the **maximum likelihood estimator** for a Gaussian response with an identity link.

---

### 2. Logistic Regression (GLM Perspective)

When the response variable is binary—such as **clicked or not**, **fraud or no fraud**, **yes or no**—a linear model with Gaussian assumptions is inappropriate. Probabilities must lie between 0 and 1, and linear regression can’t guarantee that.

Enter **Logistic Regression**, a model that naturally arises in the GLM framework when:

* The response variable follows a **Bernoulli distribution**
* The link function is the **logit**, mapping probabilities to the real line

---

#### Random Component: Bernoulli Distribution

We assume the binary response variable $$y_i \in \{0, 1\}$$ is drawn from a Bernoulli distribution with probability $$p_i = \mathbb{P}(y_i = 1)$$:


$$
y_i \sim \text{Bernoulli}(p_i)
$$


The Bernoulli distribution can be expressed in exponential family form:


$$
p(y_i; \theta_i) = \exp\left( y_i \cdot \theta_i - \log(1 + e^{\theta_i}) \right)
$$

$$
\text{with } \theta_i = \log\left( \frac{p_i}{1 - p_i} \right)
$$


Here, $$\theta_i$$ is the **canonical parameter**, and this form confirms that Bernoulli is indeed a member of the exponential family.

---

#### Link Function: Logit

To ensure $$p_i \in (0, 1)$$, we use the **logit link**:


$$
g(p_i) = \log\left( \frac{p_i}{1 - p_i} \right) = \eta_i = \mathbf{x}_i^\top \mathbf{w}
$$


Taking the inverse gives the **sigmoid** function:


$$
p_i = \frac{1}{1 + e^{-\eta_i}} = \sigma(\eta_i)
$$


This transformation maps real-valued linear predictors to valid probabilities. The logit is the **canonical link** for the Bernoulli distribution.

---

#### Interpretation: Odds and Log-Odds

One of the appealing features of logistic regression is the **interpretability** of its coefficients in terms of **odds ratios**.

* The **odds** of a positive outcome are:

  
  $$
  \text{Odds}(y_i = 1) = \frac{p_i}{1 - p_i}
  $$
  

* The **log-odds** (logit) are linear in the features:

  
  $$
  \log\left( \frac{p_i}{1 - p_i} \right) = \mathbf{x}_i^\top \mathbf{w}
  $$  
  

Each coefficient $$w_j$$ represents the change in **log-odds** associated with a one-unit increase in feature $$x_j$$, holding all other features constant.

---

#### MLE Objective: Cross-Entropy Loss

The likelihood of a Bernoulli response under the model is:


$$
\mathcal{L}(\mathbf{w}) = \prod_{i=1}^n p_i^{y_i} (1 - p_i)^{1 - y_i}
$$


Taking logs gives the **log-likelihood**:


$$
\log \mathcal{L}(\mathbf{w}) = \sum_{i=1}^n \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]
$$


This is the **negative cross-entropy loss**, which we minimize to estimate $$\mathbf{w}$$. Unlike OLS, there’s no closed-form solution; we must use **iterative optimization** methods like gradient descent or IRLS.

---


While logistic regression is most commonly used for **classification**, here we’ve framed it purely as a **regression model for binary data** using the GLM formalism.

We’ll revisit logistic regression in much greater detail—including classification boundaries, ROC curves, decision thresholds, and regularization—in the dedicated classification blog series.

---


### 3. Probit Regression

Like logistic regression, **Probit Regression** is designed for modeling binary response variables. It arises from the same GLM framework, with the key difference being the **link function**: instead of the logit, probit uses the **inverse cumulative distribution function (CDF)** of the standard normal distribution.

This subtle change makes Probit regression particularly useful in **Bayesian statistics**, latent variable modeling, and cases where a **Gaussian error structure** is assumed for the underlying decision process.

---

#### Random Component: Bernoulli Response

Just like in logistic regression, the outcome $$y_i \in \{0, 1\}$$ is modeled as:


$$
y_i \sim \text{Bernoulli}(p_i)
$$


But here, $$p_i$$ is linked to the input features via a different transformation.

---

#### Link Function: Probit (Inverse Gaussian CDF)

Instead of using the logit function, we apply the inverse of the standard normal CDF, denoted by $$\Phi^{-1}(\cdot)$$:


$$
g(p_i) = \Phi^{-1}(p_i) = \eta_i = \mathbf{x}_i^\top \mathbf{w}
$$


Taking the inverse of the link gives the **probit model**:


$$
p_i = \Phi(\eta_i) = \int_{-\infty}^{\eta_i} \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{z^2}{2}\right) dz
$$


This maps the linear predictor $$\eta_i$$ to the probability space $$[0, 1]$$ through the standard normal CDF.

---

#### Geometric and Statistical Interpretation

You can think of probit regression as arising from a **latent Gaussian process**. Imagine there’s an unobserved continuous variable $$z_i$$ defined as:


$$
z_i = \mathbf{x}_i^\top \mathbf{w} + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, 1)
$$


Then:

* $$y_i = 1$$ if $$z_i > 0$$
* $$y_i = 0$$ otherwise

This leads to:


$$
\mathbb{P}(y_i = 1) = \mathbb{P}(z_i > 0) = \mathbb{P}(\mathbf{x}_i^\top \mathbf{w} + \varepsilon_i > 0) = \Phi(\mathbf{x}_i^\top \mathbf{w})
$$


This latent variable formulation makes probit regression especially appealing in **Bayesian modeling**, where Gaussian priors on $$\mathbf{w}$$ yield conjugacy with the likelihood.

---

#### Comparison to Logistic Regression

Both probit and logit links produce sigmoid-shaped curves, but with slightly different tails:

* **Logistic CDF**: Has **heavier tails**, allowing slightly more extreme probabilities
* **Gaussian CDF (Probit)**: Is **more concentrated** near the center

Despite this, the predictions from both models are often very similar in practice. The choice between them typically hinges on:

* **Theoretical considerations** (e.g., latent Gaussian assumptions favor probit)
* **Bayesian tractability** (probit is easier to integrate into Gibbs sampling)
* **Software or domain conventions**

---

#### Practical Considerations

* **Interpretability**: Coefficients in probit models are interpreted less intuitively than in logit models, since they reflect changes in the latent normal scale.
* **Usage**: Probit regression is less common in applied machine learning but more frequently used in economics, psychometrics, and Bayesian statistics.

---

In summary, probit regression provides a statistically elegant and Bayesian-friendly alternative to logistic regression, particularly when Gaussian noise is a natural modeling assumption.

---

### 4. Multinomial / Softmax Regression

When the response variable can take on more than two discrete categories—such as predicting which product a customer will buy or which language a user is typing in—we need a model that generalizes logistic regression to the **multi-class** setting.

This leads us to **Multinomial Regression**, also known as **Softmax Regression** in machine learning literature. It is the natural GLM extension of logistic regression when the target is **categorical with more than two classes**.

---

#### Random Component: Multinomial Distribution

Assume the response variable $$y_i$$ takes values in the set $$\{1, 2, \dots, K\}$$ for $$K > 2$$. The conditional probability is modeled using a **categorical distribution**, which is a special case of the multinomial (with one trial per observation):


$$
\mathbb{P}(y_i = k) = p_{ik}, \quad \text{for } k = 1, \dots, K
$$


Each $$p_{ik}$$ is the probability that observation $$i$$ belongs to class $$k$$, and we require:


$$
\sum_{k=1}^K p_{ik} = 1
$$


---

#### Link Function: Generalized Logit (Softmax)

We define a separate linear predictor for each class $$k$$:


$$
\eta_{ik} = \mathbf{x}_i^\top \mathbf{w}_k
$$


Then apply the **softmax** transformation:


$$
p_{ik} = \frac{e^{\eta_{ik}}}{\sum_{j=1}^K e^{\eta_{ij}}}
$$


This maps each vector of real-valued logits $$[\eta_{i1}, \dots, \eta_{iK}]$$ into a valid probability distribution over $$K$$ classes.

Note: To ensure identifiability, one of the $$\mathbf{w}_k$$ (often the last) is typically set to zero.

---

#### GLM Interpretation

* The multinomial distribution is a member of the exponential family
* The softmax function serves as the **generalized canonical link** for multi-class categorical data
* The model can be fit using **maximum likelihood**, leading to the **categorical cross-entropy** loss

---


While multinomial regression is a GLM in every sense, its primary usage is in **classification tasks** rather than regression. Topics such as:

* One-vs-rest vs full softmax modeling
* Decision boundaries in high-dimensional space
* Regularization (e.g., multinomial logistic with $$\ell_1$$ or $$\ell_2$$ penalty)
* Evaluation metrics (accuracy, top-k, F1)

will be discussed in detail in the upcoming **classification blog series**.

---


### 5. Poisson Regression

In many real-world problems, the response variable represents **counts**—how many times an event occurs. Examples include:

* Number of clicks on an ad
* Number of doctor visits in a month
* Number of claims filed by an insurance policyholder
* Number of purchases during a session

These are non-negative integers ($$0, 1, 2, \dots$$) with typically **skewed distributions** and **heteroscedastic variance**. Linear regression fails to capture these properties.

**Poisson Regression** offers a GLM-based solution by modeling the response using the **Poisson distribution** and linking its mean to a linear predictor through a **log link**.

---

#### Random Component: Poisson Distribution

The response variable $$y_i \in \mathbb{N}_0 = \{0, 1, 2, \dots\}$$ is assumed to follow a **Poisson distribution** with rate parameter $$\lambda_i > 0$$:


$$
\mathbb{P}(y_i = k) = \frac{e^{-\lambda_i} \lambda_i^k}{k!}, \quad k \in \mathbb{N}_0
$$


This distribution belongs to the exponential family with:

* **Canonical parameter**: $$\theta_i = \log(\lambda_i)$$
* **Log-partition function**: $$b(\theta_i) = e^{\theta_i}$$
* **Dispersion**: $$\phi = 1$$ (fixed in standard Poisson)

---

#### Mean and Variance

A key property of the Poisson distribution:


$$
\mathbb{E}[y_i] = \lambda_i, \quad \text{Var}(y_i) = \lambda_i
$$


The variance equals the mean, a restriction that will later lead us to **Negative Binomial regression** for overdispersed data.

---

#### Link Function: Log Link

To ensure $$\lambda_i > 0$$, we use the **log link function**:


$$
g(\lambda_i) = \log(\lambda_i) = \eta_i = \mathbf{x}_i^\top \mathbf{w}
$$


This gives us the modeling equation:


$$
\lambda_i = \exp(\mathbf{x}_i^\top \mathbf{w})
$$


The log link is the **canonical link** for the Poisson distribution and guarantees positivity of predicted rates.

---

#### Likelihood and Estimation

The log-likelihood for the Poisson model is:


$$
\log \mathcal{L}(\mathbf{w}) = \sum_{i=1}^n \left[ y_i \log \lambda_i - \lambda_i - \log(y_i!) \right]
= \sum_{i=1}^n \left[ y_i \cdot \mathbf{x}_i^\top \mathbf{w} - \exp(\mathbf{x}_i^\top \mathbf{w}) - \log(y_i!) \right]
$$


This is a **convex function**, and the MLE $$\hat{\mathbf{w}}$$ can be estimated using **iteratively reweighted least squares (IRLS)** or **gradient-based optimization**.

---

#### Use Cases

Poisson regression is ideal for scenarios where:

* The target is a **non-negative count**
* The variance is roughly equal to the mean
* Events occur **independently and randomly** in time or space

Common applications include:

* **Insurance**: Number of claims per policyholder
* **Marketing**: Ad impressions or clicks
* **Healthcare**: Number of visits or prescriptions
* **Web Analytics**: Pageviews, session counts

---


### 6. Negative Binomial Regression

Poisson regression is elegant, but it makes a **strong assumption**: that the variance of the response equals the mean:


$$
\text{Var}(y_i) = \mathbb{E}[y_i] = \lambda_i
$$


In practice, however, count data often exhibit **overdispersion**—the observed variance exceeds the mean:


$$
\text{Var}(y_i) > \mathbb{E}[y_i]
$$


This can arise from unobserved heterogeneity, clustered events, or long-tailed distributions. When overdispersion is present, the Poisson model **underestimates variability**, leading to biased standard errors and poor model fit.

**Negative Binomial Regression** extends Poisson regression by explicitly modeling overdispersion while remaining within the GLM framework.

---

#### From Poisson to Negative Binomial

One way to derive the Negative Binomial distribution is as a **Poisson-Gamma mixture**:

* Assume the Poisson rate parameter $$\lambda_i$$ is not fixed, but follows a **Gamma distribution**:

  
  $$
  \lambda_i \sim \text{Gamma}(r, p), \quad y_i \mid \lambda_i \sim \text{Poisson}(\lambda_i)
  $$
  

* Marginalizing over the latent Gamma noise gives the **Negative Binomial distribution**:

  
  $$
  \mathbb{P}(y_i = k) = \frac{\Gamma(k + r)}{k! \, \Gamma(r)} \left( \frac{p}{1 + p} \right)^r \left( \frac{1}{1 + p} \right)^k
  $$
  

This compound formulation introduces **extra variance** through the Gamma prior on $$\lambda_i$$, allowing the model to fit overdispersed data.

---

#### Mean and Variance

For Negative Binomial-distributed responses, the mean and variance are:


$$
\mathbb{E}[y_i] = \mu_i, \quad \text{Var}(y_i) = \mu_i + \alpha \mu_i^2
$$


Where:

* $$\mu_i = \exp(\mathbf{x}_i^\top \mathbf{w})$$ is the predicted count
* $$\alpha > 0$$ is an overdispersion parameter

This quadratic variance structure allows **more flexible modeling** than Poisson regression.

---

#### Link Function: Log Link

As with Poisson regression, we use the **log link** to ensure non-negative predictions:


$$
\log(\mu_i) = \eta_i = \mathbf{x}_i^\top \mathbf{w}
$$

$$
\quad \text{or equivalently} \quad \mu_i = \exp(\mathbf{x}_i^\top \mathbf{w})
$$


The log link remains interpretable and is often used in count models.

---

#### Estimation and Usage

The log-likelihood for the Negative Binomial model includes a shape parameter (usually estimated via maximum likelihood or profile likelihood). Most modern libraries (e.g., `statsmodels`, `R`'s `MASS::glm.nb`) handle this seamlessly.

Negative Binomial regression is widely used in fields such as:

* **Healthcare**: modeling hospital visits or lengths of stay
* **Ecology**: modeling species counts
* **Insurance and Finance**: modeling claim frequency with variability
* **Web analytics**: modeling user behavior over variable-length sessions

---

#### Summary

By introducing a **Gamma-distributed noise term** into the Poisson rate, Negative Binomial regression provides a principled way to model count data with extra variance. It retains the GLM structure, uses the same log link, and improves fit when Poisson's mean–variance constraint is too rigid.

---

With this, we complete the tour of the most common GLMs. In the next section, we'll briefly discuss how these models are fit and evaluated in practice, and when to use each one depending on your data and task.

---


## Maximum Likelihood Estimation in GLMs

All models within the Generalized Linear Model framework share a common estimation philosophy: **maximum likelihood**. Whether you're modeling a continuous outcome via linear regression or counts via Poisson regression, the goal is to find the parameter vector $$\mathbf{w}$$ that maximizes the likelihood of observing the data under the assumed model.

In this section, we'll explore how parameter estimation works across GLMs, focusing on the **score function**, the role of the **link function**, and the iterative algorithms used to solve the optimization problem.

---

### Objective: Maximizing the Likelihood

Suppose we observe $$n$$ independent data points $$(\mathbf{x}_i, y_i)$$. Under a GLM, the likelihood of the full dataset is:


$$
\mathcal{L}(\mathbf{w}) = \prod_{i=1}^n p(y_i; \theta_i(\mathbf{x}_i^\top \mathbf{w}), \phi)
$$


Taking the logarithm, we obtain the **log-likelihood function**:


$$
\ell(\mathbf{w}) = \sum_{i=1}^n \left[ \frac{y_i \cdot \theta_i - b(\theta_i)}{\phi} + c(y_i, \phi) \right]
$$


Where $$\theta_i$$ is the **canonical parameter**, expressed as a function of the linear predictor:


$$
\eta_i = \mathbf{x}_i^\top \mathbf{w}, \quad \theta_i = g^{-1}(\eta_i) \text{ (possibly with transformation)}
$$


The goal is to solve:


$$
\hat{\mathbf{w}} = \arg\max_{\mathbf{w}} \, \ell(\mathbf{w})
$$


In general, this problem does **not** admit a closed-form solution (except for linear regression). We must use **numerical optimization**.

---

### Deriving the Score Function

To solve for $$\mathbf{w}$$, we compute the **score function**, the gradient of the log-likelihood:


$$
\frac{\partial \ell(\mathbf{w})}{\partial \mathbf{w}} = \sum_{i=1}^n \left( \frac{\partial \ell}{\partial \mu_i} \cdot \frac{\partial \mu_i}{\partial \eta_i} \cdot \frac{\partial \eta_i}{\partial \mathbf{w}} \right)
$$


This can be organized neatly using matrix notation:


$$
\nabla_{\mathbf{w}} \ell(\mathbf{w}) = \mathbf{X}^\top \mathbf{W}^{-1} (\mathbf{y} - \boldsymbol{\mu})
$$


Where:

* $$\mathbf{X}$$ is the design matrix
* $$\boldsymbol{\mu}$$ is the vector of predicted means: $$\mu_i = \mathbb{E}[y_i]$$
* $$\mathbf{W}$$ is a diagonal matrix of weights involving variances and derivatives of the link

This gradient points us in the direction of **steepest ascent** in log-likelihood, and can be used in iterative algorithms like **gradient ascent** or **Newton-Raphson**.

---

### IRLS: Iteratively Reweighted Least Squares

The most commonly used method for solving GLM estimation is **IRLS**—Iteratively Reweighted Least Squares. It is a second-order method that approximates the log-likelihood surface locally as a quadratic, then solves a weighted least squares problem at each step.

IRLS solves:


$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + (\mathbf{X}^\top \mathbf{W}^{(t)} \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{W}^{(t)} \mathbf{z}^{(t)}
$$


Where:

* $$\mathbf{W}^{(t)}$$ is a diagonal matrix of weights based on the variance and derivative of the link
* $$\mathbf{z}^{(t)}$$ is the **working response**, a linear approximation to the link-transformed outcome:

  
  $$
  z_i^{(t)} = \eta_i^{(t)} + \frac{y_i - \mu_i^{(t)}}{g'(\mu_i^{(t)})}
  $$
  

Each iteration is equivalent to solving a **weighted least squares** problem, where weights depend on the model's curvature at the current parameter estimate.

---

### Convergence Properties

IRLS is efficient and has strong convergence properties under the following conditions:

* The log-likelihood is **concave** (which holds for canonical links)
* The Fisher Information Matrix $$\mathbf{X}^\top \mathbf{W} \mathbf{X}$$ is full-rank
* The response distribution is properly specified

However:

* Poorly scaled data can cause numerical instability
* Overdispersion (e.g., Poisson when variance > mean) can distort curvature
* Non-canonical links may slow convergence or make the Hessian harder to compute

In modern practice, IRLS is often replaced or supplemented by:

* **Quasi-Newton methods** (e.g., BFGS)
* **Stochastic gradient descent (SGD)** for large datasets
* **Bayesian MCMC** for posterior estimation in fully probabilistic settings

---

## Model Evaluation and Diagnostics for GLMs

Fitting a GLM is only the beginning. Once you've estimated the model parameters, the next crucial step is to **evaluate how well the model fits the data**, determine whether the assumptions hold, and compare across candidate models.

Unlike linear regression—where we have well-defined tools like $$R^2$$ and residual plots—GLMs require a slightly more general toolkit. In this section, we explore the most common diagnostics and model selection tools for GLMs.

---

### Deviance: The GLM Analog of Residual Sum of Squares

In OLS, we evaluate model fit using the **Residual Sum of Squares (RSS)**:


$$
\text{RSS} = \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$


In GLMs, the equivalent measure is the **deviance**, which quantifies the distance between the fitted model and a **saturated model** (a hypothetical model that fits each data point exactly).

Formally, the deviance is defined as:


$$
D = 2 \left[ \ell_{\text{sat}} - \ell(\hat{\mathbf{w}}) \right]
$$


Where:
* $$\ell_{\text{sat}}$$ is the log-likelihood of the saturated model
* $$\ell(\hat{\mathbf{w}})$$ is the log-likelihood under the fitted model

Smaller deviance implies better fit. For comparing two nested GLMs, the **difference in deviance** follows an approximate $$\chi^2$$ distribution and can be used for hypothesis testing.

---

### AIC: Penalized Likelihood for Model Selection

The **Akaike Information Criterion (AIC)** balances goodness-of-fit with model complexity. It is defined as:


$$
\text{AIC} = -2 \ell(\hat{\mathbf{w}}) + 2k
$$


Where:
* $$\ell(\hat{\mathbf{w}})$$ is the maximized log-likelihood
* $$k$$ is the number of estimated parameters (including the intercept)

AIC is used to **compare different models on the same dataset**: lower AIC indicates a better model, accounting for overfitting.

For smaller sample sizes, a corrected version—**AICc**—is often used:


$$
\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n - k - 1}
$$


---

### Residuals in GLMs

GLMs generalize the concept of residuals. Different types of residuals provide diagnostic insight into various aspects of model fit.

#### 1. Pearson Residuals

These standardize the difference between observed and expected responses:


$$
r_i^{(P)} = \frac{y_i - \hat{\mu}_i}{\sqrt{\text{Var}(y_i)}}
$$


Useful for detecting **outliers** and **non-constant variance** patterns.

#### 2. Deviance Residuals

Derived from the contribution of each data point to the model deviance:


$$
r_i^{(D)} = \text{sign}(y_i - \hat{\mu}_i) \cdot \sqrt{d_i}
$$


Where $$d_i$$ is the contribution of observation $$i$$ to the total deviance.

Deviance residuals are symmetric and often preferred for visual diagnostics like **residual vs. fitted** plots.

#### 3. Anscombe Residuals

These are designed to **normalize skewness** in residuals and are especially useful when the response distribution is highly asymmetric (e.g., Gamma or Poisson).

---

### Checking for Overdispersion

Overdispersion occurs when the observed variance exceeds the model's assumed variance. It's most common in:
* **Poisson regression**: assumes $$\text{Var}(y) = \mu$$
* **Binomial models**: assumes $$\text{Var}(y) = n \cdot p(1 - p)$$

To test for overdispersion:

1. Compute the **Pearson dispersion statistic**:

   
   $$
   \hat{\phi} = \frac{1}{n - p} \sum_{i=1}^n \left( \frac{y_i - \hat{\mu}_i}{\sqrt{\text{Var}(y_i)}} \right)^2
   $$

   \quad \text{If } \hat{\phi} > 1, \text{ there is evidence of overdispersion.}
   

2. Refit the model using **quasi-likelihood** or **Negative Binomial** variants.

---

### Goodness-of-Fit Tests

Some tests assess whether the model adequately describes the data:
* **Hosmer-Lemeshow Test** (for binary outcomes): Compares observed and expected frequencies across quantile bins.
* **Deviance Test**: Compares model deviance to degrees of freedom.
* **Likelihood Ratio Test (LRT)**: Compares two nested models.

These are often supplemented by **visual diagnostics** such as:
* Residual plots
* Q-Q plots of deviance residuals
* Leverage and Cook's distance for influence analysis

---

## When to Use Which GLM?

Choosing the right Generalized Linear Model hinges on understanding the **nature of your response variable** and the **distributional assumptions** appropriate for it. Each GLM is specialized to handle a different kind of outcome behavior, and selecting the correct model is essential for valid inference and predictive performance.

The table below summarizes which GLM to use depending on the structure of your target variable:

<div style="overflow-x: auto;">
  <table class="simple-table">
    <thead>
      <tr>
        <th>Response Type</th>
        <th>Distribution</th>
        <th>Link Function</th>
        <th>Recommended GLM</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Binary (0/1)</td>
        <td>Bernoulli</td>
        <td>Logit / Probit</td>
        <td>Logistic Regression / Probit Regression</td>
      </tr>
      <tr>
        <td>Multi-class (Categorical)</td>
        <td>Multinomial</td>
        <td>Generalized Logit (Softmax)</td>
        <td>Multinomial (Softmax) Regression</td>
      </tr>
      <tr>
        <td>Count (Variance $$\\approx$$ Mean)</td>
        <td>Poisson</td>
        <td>Log</td>
        <td>Poisson Regression</td>
      </tr>
      <tr>
        <td>Count (Variance $$>$$ Mean)</td>
        <td>Negative Binomial</td>
        <td>Log</td>
        <td>Negative Binomial Regression</td>
      </tr>
      <tr>
        <td>Continuous (Unbounded)</td>
        <td>Gaussian (Normal)</td>
        <td>Identity</td>
        <td>Linear Regression (OLS)</td>
      </tr>
    </tbody>
  </table>
</div>

---

### A Few Guiding Principles

* Use **Logistic Regression** when your outcome is binary and you’re interested in interpretability via odds ratios.
* Use **Probit Regression** if you have latent Gaussian motivations or are building a Bayesian model.
* Use **Multinomial Regression** for multi-class classification tasks, especially when class probabilities matter.
* Use **Poisson Regression** for non-negative count data where the variance is close to the mean.
* Use **Negative Binomial Regression** when you detect overdispersion in count data.
* Use **Linear Regression** for continuous numeric outcomes with approximately normal residuals.

Always check residuals, variance structure, and dispersion before settling on your model. If in doubt, fit multiple candidate models and compare using **AIC**, **cross-validation**, and **diagnostic plots**.

---

## Closing Thoughts

Generalized Linear Models (GLMs) represent one of the most elegant and powerful frameworks in statistical modeling. By extending linear regression to accommodate a variety of response types—binary, count, categorical, and more—GLMs unify a wide spectrum of models under a common mathematical structure.

At their core, GLMs provide:

* A **random component** drawn from the exponential family
* A **systematic component** via linear predictors
* A **link function** connecting them

This modular design brings both interpretability and extensibility. It allows us to handle real-world data complexities—like skewness, discreteness, and heteroscedasticity—while preserving the conceptual clarity of linear models.

Importantly, GLMs serve as a **bridge** between classical statistics and modern machine learning:

* They form the foundation of **regularized regression**, **generalized additive models (GAMs)**, and **Bayesian regression**
* Their probabilistic nature connects smoothly to **likelihood-based inference**, **uncertainty quantification**, and **Bayesian priors**
* Their optimization structure aligns with gradient-based methods used in **deep learning** and **large-scale modeling**

Understanding GLMs equips data scientists with a principled approach to choosing the right model for the task at hand—not just because it “works,” but because it fits the data-generating process in a meaningful way.

As we transition to more complex tasks like **classification**, **structured outputs**, or **nonlinear modeling**, the ideas introduced here—link functions, exponential families, and likelihood-based estimation—will remain foundational.

In future posts, we’ll explore how these ideas extend into the **classification domain** (Logistic, Softmax, and beyond), and how GLMs inspire the design of flexible nonlinear models in both Bayesian and ML contexts.