---
layout: post
title: Probability & Statistics for Data Science - Foundations of Probability
date: 2022-06-01
description: Probability & Statistics 1 - Mathematics for Machine Learning
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


When we build machine learning models, we’re not just dealing with data — we're dealing with **uncertainty**. Whether we’re classifying emails, predicting stock prices, or detecting fraud, our models are built on probabilistic foundations that allow them to reason under uncertainty.

This post covers the **core probability theory concepts** that underpin machine learning, deep learning, and AI. We'll explore how concepts like **sample space, events, random variables, expectation, variance, the Law of Large Numbers**, and the **Central Limit Theorem** connect directly to real-world ML applications.

---

## Sample Space, Events, and Conditional Probability

To understand how probability plays a role in ML, we must first start with the basics.

- **Sample Space ( $$\Omega$$ )**: The set of all possible outcomes of an experiment.
- **Event**: A subset of the sample space. It's a collection of outcomes we're interested in.
- **Probability Function**: Assigns a number between 0 and 1 to each event, satisfying the axioms of probability.

**Example**: Suppose we build a binary classifier to detect spam.

- $$\Omega = \{\text{spam}, \text{not spam}\}$$
- $$P(\text{spam}) = 0.4$$, $$P(\text{not spam}) = 0.6$$

### Independence

Two events $$A$$ and $$B$$ are **independent** if the occurrence of one does not affect the probability of the other:

$$
P(A \cap B) = P(A) \cdot P(B)
$$

In ML, this concept appears in **Naive Bayes classifiers**, where we assume that features are conditionally independent given the class label — a simplification that works surprisingly well in practice.

### Conditional Probability

Conditional probability tells us the probability of an event $$A$$ given that event $$B$$ has occurred:

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}
$$

This is the foundation of **Bayes’ Theorem**, which is used to update predictions as new information becomes available — exactly what happens when your email spam filter learns from new messages.

---

📸 **Image idea**:  
A Venn diagram illustrating $$P(A \cap B)$$ and $$P(A \mid B)$$ inside a sample space.

---

## Random Variables (Discrete and Continuous)

A **random variable** maps outcomes from the sample space to numerical values.

### Discrete Random Variables

These take **countable** values (like integers). Examples include:

- Number of clicks on an ad
- Whether a transaction is fraudulent (0 or 1)

A discrete random variable $$X$$ has a **probability mass function (PMF)**:

$$
P(X = x) = p(x)
$$

### Continuous Random Variables

These take **uncountably infinite** values (e.g., any real number). Examples:

- The exact temperature in a room
- Probability of customer spending

A continuous random variable has a **probability density function (PDF)** such that:

$$
P(a \leq X \leq b) = \int_a^b f(x) \, dx
$$

In ML, we use random variables to model data distributions. For example, we assume weights in Bayesian models come from a Gaussian prior — a continuous random variable.

---

📸 **Image idea**:  
Side-by-side plots of a PMF (e.g., Binomial) and a PDF (e.g., Normal distribution).

---

## Expectation, Variance, and Standard Deviation

These are the building blocks of **descriptive statistics** and are critical to understanding model behavior.

### Expectation

Also called the **mean** or expected value:

- For discrete $$X$$:
  $$
  \mathbb{E}[X] = \sum_x x \cdot P(X = x)
  $$

- For continuous $$X$$:
  $$
  \mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx
  $$

Think of it as the **center of mass** of the distribution.

### Variance

Measures the **spread** of the random variable around the mean:

$$
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]
$$

This gives us insight into how uncertain a prediction might be. A model with low variance makes **consistent predictions**.

### Standard Deviation

The square root of variance:

$$
\sigma = \sqrt{\text{Var}(X)}
$$

Used commonly in ML to **normalize features** and analyze error distributions.

---

📸 **Image idea**:  
A bell curve showing mean, one standard deviation, and two standard deviations.

---

## Law of Large Numbers (LLN)

The Law of Large Numbers states that as the sample size $$n$$ increases, the **sample mean** of random variables converges to the **true mean**:

$$
\lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^n X_i = \mathbb{E}[X]
$$

This justifies why **averaging predictions across many models (ensembles)** improves performance — individual errors average out.


Let’s simulate this idea by drawing samples from a uniform distribution $$\mathcal{U}(0, 1)$$ and watching how the sample mean stabilizes:

```python
import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# Generate 10,000 samples from a Uniform[0,1] distribution
samples = np.random.uniform(0, 1, 10000)

# Compute sample means as we increase sample size
sample_means = [np.mean(samples[:i]) for i in range(1, len(samples) + 1)]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(sample_means, label='Sample Mean')
plt.axhline(y=0.5, color='red', linestyle='--', label='True Mean = 0.5')
plt.xlabel("Number of Samples")
plt.ylabel("Sample Mean")
plt.title("Law of Large Numbers (LLN)")
plt.legend()
plt.show()
```
{% raw %}
<div style="display: flex; justify-content: center;">
  <div id="llnPlot" style="width:100%;max-width:700px;"></div>
</div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
  const xVals = Array.from({length: 1000}, (_, i) => i + 1);
  const yVals = xVals.map(n => {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += Math.random();
    }
    return sum / n;
  });

  const trace1 = {
    x: xVals,
    y: yVals,
    mode: 'lines',
    name: 'Sample Mean'
  };

  const trace2 = {
    x: xVals,
    y: Array(xVals.length).fill(0.5),
    mode: 'lines',
    name: 'True Mean = 0.5',
    line: {dash: 'dot', color: 'red'}
  };

  Plotly.newPlot('llnPlot', [trace1, trace2], {
    title: 'Law of Large Numbers',
    xaxis: {title: 'Number of Samples'},
    yaxis: {title: 'Sample Mean'}
  });
</script>
{% endraw %}



As shown in the plot, the sample mean starts off noisy but **converges toward 0.5**, which is the true expected value of a uniform distribution over [0, 1].


### ML Insight:
In **bagging methods** (e.g., Random Forests), we train many models on bootstrapped samples and average their predictions. The LLN guarantees that as the number of trees increases, the aggregate prediction becomes more stable and closer to the true signal.


---

## Central Limit Theorem (CLT)

The CLT is one of the most powerful ideas in all of statistics.

> **Theorem**: The sum (or average) of a large number of **independent, identically distributed (i.i.d.)** random variables approaches a **Normal distribution**, regardless of the original distribution.

Formally, for i.i.d. variables $$X_1, X_2, \dots, X_n$$:

$$
\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i \quad \text{and} \quad \frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1)
$$

This explains **why the normal distribution is so prevalent** in ML — it naturally arises when averaging data, sampling errors, or even when computing model parameter estimates.

---


Let’s visualize this with a non-Gaussian distribution — the **exponential distribution**:

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Draw means from 1000 samples, each of size 50, from an Exponential distribution
means = []
for _ in range(1000):
    sample = np.random.exponential(scale=1.0, size=50)
    means.append(np.mean(sample))

# Plotting
plt.figure(figsize=(10, 5))
sns.histplot(means, bins=30, kde=True)
plt.title("Central Limit Theorem (CLT): Means from Exponential Distribution")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.show()
```

{% raw %}
<div style="display: flex; justify-content: center;">
  <div id="cltPlot" style="width:100%;max-width:700px;"></div>
</div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
  const sampleMeans = [];
  for (let i = 0; i < 1000; i++) {
    let sum = 0;
    for (let j = 0; j < 50; j++) {
      const u = Math.random();
      const expSample = -Math.log(1 - u); // inverse transform sampling
      sum += expSample;
    }
    sampleMeans.push(sum / 50);
  }

  const trace = {
    x: sampleMeans,
    type: 'histogram',
    marker: {color: 'skyblue'}
  };

  Plotly.newPlot('cltPlot', [trace], {
    title: 'Central Limit Theorem: Sampling Means from Exponential Distribution',
    xaxis: {title: 'Sample Mean'},
    yaxis: {title: 'Frequency'}
  });
</script>
{% endraw %}


Despite the exponential distribution being **highly skewed**, the **histogram of sample means is bell-shaped**, showing the CLT in action.

---

## Applications in Machine Learning

Now that we’ve built up the mathematical foundations, let’s see how they **directly impact practical machine learning workflows**.

---

### Probabilistic Modeling

Probabilistic models like:

- **Naive Bayes**
- **Gaussian Mixture Models**
- **Bayesian Linear Regression**

...are all rooted in these probability concepts. They define **likelihoods**, **priors**, and **posteriors**, which use **conditional probability**, **distributions**, and **expectation**.

Even advanced models like **Variational Autoencoders (VAEs)** rely on these basics — using expectations, KL divergence, and Normal distributions.

---

### Uncertainty Quantification

Models that output probabilities (like softmax classifiers or probabilistic regressors) provide a **distribution over outputs**, allowing you to quantify uncertainty.

- Knowing the **variance** of a model’s prediction helps determine confidence.
- Using the **CLT**, you can estimate confidence intervals around predictions.

This is critical in **high-stakes applications** like medicine, finance, and autonomous driving.

---

### Why Empirical Means Work (Ensemble Models)

Ensemble techniques like:

- **Bagging (Random Forests)**
- **Boosting (XGBoost, LightGBM)**
- **Model Averaging (e.g., Stacking)**

...all rely on the **Law of Large Numbers** and **Central Limit Theorem**.

By aggregating multiple noisy models, the final prediction has:

- Lower variance
- Greater stability
- Better generalization

The magic lies in averaging — and the math tells us why that works.

---

## Wrapping Up

Probability theory is more than a theoretical curiosity — it's the **engine behind machine learning**. From modeling uncertainties, understanding feature distributions, building generative models, and even constructing deep learning layers — these foundational ideas are everywhere.

---

**Next Up**: In the next post, we’ll dive into **Probability Distributions** — understanding Bernoulli, Binomial, Gaussian, and how they shape the models we build.