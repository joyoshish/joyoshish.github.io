---
layout: post
title: Probability & Statistics for Data Science - Probability Distributions
date: 2022-06-05
description: Probability & Statistics 2 - Mathematics for Machine Learning
tags: ml ai probability-statistics math
math: true
categories: machine-learning math math-for-ml
# thumbnail: assets/img/linalg_banner.png
giscus_comments: true
pretty_table: true
related_posts: true
chart:
  plotly: true
toc:
  beginning: true
---



Probability distributions are the foundation of machine learning. They shape how we simulate data, quantify uncertainty, and reason about model behavior. Whether you are generating synthetic samples, fitting probabilistic models, or understanding errors, distributions provide the structure for everything that follows.

In this article, we will explore:

- Important **discrete and continuous distributions**
- Concepts around **multivariate distributions**
- Key **applications in machine learning and data science**
- Demonstrations using **Python and Plotly visualizations**

---

## Discrete Distributions

Discrete probability distributions represent outcomes that are countable — such as binary labels, event counts, or integer results of repeated trials. These are particularly relevant in classification tasks, anomaly detection, and simulations of real-world processes.

---

### 1. Bernoulli Distribution

The Bernoulli distribution is used to model a single trial with two possible outcomes: success (1) or failure (0). It is the building block of binary classification problems.

**Probability Mass Function (PMF)**:
$$
P(X = x) = p^x (1 - p)^{1 - x}, \quad x \in \{0,1\}
$$

- **Mean**: $$\mathbb{E}[X] = p$$  
- **Variance**: $$\text{Var}(X) = p(1 - p)$$

**Applications**:
- Binary classification (e.g., logistic regression target)
- Simulating binary labels for synthetic datasets


> The plot below illustrates a Bernoulli distribution where the probability of success (1) is 0.7 and failure (0) is 0.3. This type of distribution is ideal for binary classification tasks where outcomes are either "yes" or "no", such as predicting if a customer will churn. The interactive bar chart helps visualize how the probability is distributed between the two outcomes.


{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/probstat2_1_bernoulli.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/probstat2_1_bernoulli.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}




---

### 2. Binomial Distribution

The Binomial distribution generalizes the Bernoulli trial to $$n$$ repeated trials. It models the number of successes in a fixed number of independent experiments.

**PMF**:
$$
P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}
$$

- **Mean**: $$\mathbb{E}[X] = np$$  
- **Variance**: $$\text{Var}(X) = np(1 - p)$$

**Applications**:
- Ensemble model success probability
- Simulating repeated trials


> This interactive chart represents the probability of getting a certain number of successes in 10 independent trials, each with a 50% success rate. The binomial distribution models many real-world situations in machine learning, such as predicting how many models in an ensemble will make correct predictions. Notice how the distribution is symmetric when $$p = 0.5$$ and peaks around $$n \cdot p = 5$$.

{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/probstat2_2_binomial.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/probstat2_2_binomial.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}


---

### 3. Poisson Distribution

The Poisson distribution models the number of events that occur in a fixed interval of time or space, assuming the events occur independently and at a constant rate $$\lambda$$.

**PMF**:
$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
$$

- **Mean**: $$\mathbb{E}[X] = \lambda$$  
- **Variance**: $$\text{Var}(X) = \lambda$$

**Applications**:
- Anomaly detection (e.g., fraud or system failures)
- Modeling event frequency (web traffic, queueing)

```python
from scipy.stats import poisson

lmbda = 4
x = np.arange(0, 15)
pmf = poisson.pmf(x, lmbda)

plt.bar(x, pmf)
plt.title(f"Poisson Distribution (λ={lmbda})")
plt.xlabel("k (events)")
plt.ylabel("Probability")
plt.show()
```

> The Poisson distribution is commonly used for modeling the number of times an event occurs in a fixed interval. In the plot below, we use $$\lambda = 4$$ to show the probability distribution of event counts. This distribution is especially useful in anomaly detection — for example, identifying if the number of failed login attempts is abnormally high.

<div id="poissonPlot" style="width:100%;max-width:700px; margin: 2rem auto;"></div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
  document.addEventListener("DOMContentLoaded", function () {
    const poissonX = Array.from({ length: 15 }, (_, i) => i);
    const lambda = 4;
    const factorial = n => n ? n * factorial(n - 1) : 1;
    const poissonY = poissonX.map(k => (Math.pow(lambda, k) * Math.exp(-lambda)) / factorial(k));

    Plotly.newPlot("poissonPlot", [{
      x: poissonX,
      y: poissonY,
      type: "bar",
      marker: { color: "slateblue" }
    }], {
      title: "Poisson Distribution (λ = 4)",
      xaxis: { title: "k (Events)" },
      yaxis: { title: "Probability" }
    });
  });
</script>



---

## Continuous Distributions

Continuous distributions represent variables that can take on any value within a range. They are essential in feature modeling, regression analysis, generative modeling, and more.

---

### 1. Uniform Distribution

This distribution assigns equal probability to all values in an interval $[a, b]$.

**PDF**:
$$
f(x) = \frac{1}{b - a}, \quad a \leq x \leq b
$$

**Applications**:
- Random initialization (e.g., neural network weights)
- Data simulation and control baselines

```python
samples = np.random.uniform(0, 1, 10000)
sns.histplot(samples, bins=30, kde=True)
plt.title("Uniform Distribution [0,1]")
plt.show()
```



---

### 2. Normal (Gaussian) Distribution

The Normal distribution is the most widely used distribution in statistics and machine learning, due to the Central Limit Theorem.

**PDF**:
$$
f(x) = \frac{1}{\sqrt{2\pi \sigma^2}} e^{- \frac{(x - \mu)^2}{2\sigma^2}}
$$

- **Mean**: $$\mu$$  
- **Variance**: $$\sigma^2$$

**Applications**:
- Linear regression assumptions
- PCA and feature decorrelation
- Modeling errors and noise

```python
mu, sigma = 0, 1
samples = np.random.normal(mu, sigma, 10000)
sns.histplot(samples, bins=30, kde=True)
plt.title("Normal Distribution (μ=0, σ=1)")
plt.show()
```
> The bell-shaped curve shown below represents the standard normal distribution. This is foundational to many algorithms in statistics and machine learning. It models noise, errors, and is used in methods like Principal Component Analysis (PCA). Thanks to the Central Limit Theorem, many sample-based statistics tend to follow this distribution even when the original data is not normal.


{% raw %}
<div style="display: flex; justify-content: center;">
  <div id="normalPlot" style="width:100%;max-width:700px;"></div>
</div>
<script>
  const normalX = Array.from({length: 100}, (_, i) => (i - 50) / 10);
  const mu = 0, sigma = 1;
  const normalY = normalX.map(x => (1 / (Math.sqrt(2 * Math.PI) * sigma)) * Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2)));

  Plotly.newPlot('normalPlot', [{
    x: normalX,
    y: normalY,
    type: 'scatter',
    mode: 'lines',
    line: { color: 'seagreen' }
  }], {
    title: 'Normal Distribution (μ = 0, σ = 1)',
    xaxis: { title: 'x' },
    yaxis: { title: 'Density' }
  });
</script>
{% endraw %}


---

### 3. Exponential Distribution

The exponential distribution describes the time between events in a Poisson process.

**PDF**:
$$
f(x) = \lambda e^{-\lambda x}, \quad x \geq 0
$$

**Applications**:
- Survival and reliability analysis
- Modeling waiting times

```python
lmbda = 1
samples = np.random.exponential(1/lmbda, 10000)
sns.histplot(samples, bins=30, kde=True)
plt.title("Exponential Distribution")
plt.show()
```

> This visualization shows the exponential distribution, which is often used to model the time between events in a Poisson process. It's especially useful in survival analysis and reliability engineering. The curve declines rapidly, showing that events are most likely to happen shortly after the last one, and become less likely over time.

{% raw %}
<div style="display: flex; justify-content: center;">
  <div id="exponentialPlot" style="width:100%;max-width:700px;"></div>
</div>
<script>
  const expX = Array.from({length: 100}, (_, i) => i * 0.1);
  const lambdaExp = 1;
  const expY = expX.map(x => lambdaExp * Math.exp(-lambdaExp * x));

  Plotly.newPlot('exponentialPlot', [{
    x: expX,
    y: expY,
    type: 'scatter',
    mode: 'lines',
    line: { color: 'tomato' }
  }], {
    title: 'Exponential Distribution (λ = 1)',
    xaxis: { title: 'x' },
    yaxis: { title: 'Density' }
  });
</script>
{% endraw %}


---

### 4. Beta Distribution

A flexible distribution on the interval [0,1], defined by two shape parameters $$\alpha$$ and $$\beta$$.

**PDF**:
$$
f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}
$$

**Applications**:
- Bayesian modeling of probabilities
- Thompson sampling in reinforcement learning

```python
from scipy.stats import beta

x = np.linspace(0, 1, 1000)
for a, b in [(2, 5), (5, 2), (2, 2)]:
    plt.plot(x, beta.pdf(x, a, b), label=f"α={a}, β={b}")

plt.title("Beta Distributions")
plt.legend()
plt.show()
```

> The Beta distribution is defined over the interval [0,1] and is commonly used to model probabilities themselves. In the visualization below, we use shape parameters $$\alpha = 2$$ and $$\beta = 5$$. This creates a distribution skewed toward 0, reflecting a belief that lower probability values are more likely. Beta distributions are essential in Bayesian statistics and exploration-exploitation algorithms like Thompson Sampling.

{% raw %}
<div style="display: flex; justify-content: center;">
  <div id="betaPlot" style="width:100%;max-width:700px;"></div>
</div>
<script>
  function gamma(n) {
    if (n === 1) return 1;
    return (n - 1) * gamma(n - 1);
  }

  function betaPDF(x, a, b) {
    const B = (a, b) => (gamma(a) * gamma(b)) / gamma(a + b);
    return Math.pow(x, a - 1) * Math.pow(1 - x, b - 1) / B(a, b);
  }

  const betaX = Array.from({length: 100}, (_, i) => i / 100);
  const betaY = betaX.map(x => betaPDF(x, 2, 5));

  Plotly.newPlot('betaPlot', [{
    x: betaX,
    y: betaY,
    type: 'scatter',
    mode: 'lines',
    line: { color: 'steelblue' }
  }], {
    title: 'Beta Distribution (α = 2, β = 5)',
    xaxis: { title: 'x (0 to 1)' },
    yaxis: { title: 'Density' }
  });
</script>
{% endraw %}


---

### 5. Gamma Distribution

A two-parameter generalization of the exponential distribution.

**PDF**:
$$
f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}
$$

**Applications**:
- Waiting time models
- Priors in Bayesian inference

```python
from scipy.stats import gamma

x = np.linspace(0, 20, 1000)
for shape in [1, 2, 5]:
    plt.plot(x, gamma.pdf(x, shape), label=f"Shape={shape}")

plt.title("Gamma Distributions")
plt.legend()
plt.show()
```

---

## Multivariate Distributions

Multivariate distributions model **joint behavior of multiple variables**, especially when they are correlated.

---

### Joint, Marginal, and Conditional Distributions

- **Joint distribution**: $$P(X, Y)$$ gives the probability of two variables occurring together.
- **Marginal distribution**: $$P(X) = \sum_Y P(X, Y)$$, projecting one variable.
- **Conditional distribution**: $$P(Y \mid X) = \frac{P(X, Y)}{P(X)}$$

These are critical for reasoning about dependencies, causality, and generative models.

---

### Multivariate Normal Distribution

Defined by a mean vector $$\mu$$ and a covariance matrix $$\Sigma$$.

$$
X \sim \mathcal{N}(\mu, \Sigma)
$$

**Applications**:
- PCA (eigen decomposition of $$\Sigma$$)
- Gaussian Mixture Models (GMMs)
- Modeling correlated features

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]

samples = np.random.multivariate_normal(mean, cov, size=1000)
sns.scatterplot(x=samples[:, 0], y=samples[:, 1])
plt.title("Samples from Bivariate Normal Distribution")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis("equal")
plt.show()
```


---

## Simulating Datasets in Practice

Simulating data allows us to prototype models, test ideas, and understand algorithms.

### Example: Binary Classification Dataset

```python
from sklearn.datasets import make_classification
import pandas as pd

X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_classes=2)
df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
df["Target"] = y
print(df.head())
```

### Example: Two-Class Gaussian Clusters

```python
class0 = np.random.normal(loc=-2, scale=1, size=(500, 2))
class1 = np.random.normal(loc=2, scale=1, size=(500, 2))
labels = np.array([0]*500 + [1]*500)
X = np.vstack([class0, class1])
```

---

## Summary Table: Choosing the Right Distribution

| **Distribution**     | **Common Application**                    |
|----------------------|-------------------------------------------|
| Bernoulli            | Binary classification                     |
| Binomial             | Ensemble voting, success trials           |
| Poisson              | Anomaly detection, event counts           |
| Normal               | Error modeling, PCA, regression           |
| Exponential          | Time-to-event, survival modeling          |
| Beta                 | Probability modeling, Bayesian inference  |
| Gamma                | Duration modeling, Bayesian priors        |
| Multivariate Normal  | Feature correlation, PCA, GMM             |

---


Probability distributions are at the core of every model you build in machine learning. They guide how you generate, structure, and analyze data. Whether you are simulating features, modeling uncertainty, or decomposing variance in PCA, the right distribution makes all the difference.

In the next post, we’ll move into the world of **Bayesian inference** and learn how concepts like **MLE, MAP**, and **priors** shape our understanding of parameter estimation.



