---
layout: post
title: Probability & Statistics for Data Science - Information Theory, Correlation & Statistical Learning Theory
date: 2022-10-17
description: Probability & Statistics 7 - Mathematics for Machine Learning
tags: ml ai probability-statistics math
categories: machine-learning math math-for-ml
thumbnail: assets/img/probstat_banner.png
math: true
giscus_comments: true
related_posts: true
chart:
  plotly: true
toc:
  beginning: true
---


## 1. Introduction

Let us take a journey that intertwines **information theory**, **statistical correlation**, and the bedrock of **statistical learning theory**. We’re stepping into that sweet spot where ideas from multiple domains converge to clarify how we reason about uncertainty, how we measure relationships, and how we build models that can generalize from limited data.

The topics might sound separate at first—**entropy** on one side, **covariance** on another, and **learning theory** on yet another. But in data science and machine learning, these concepts weave together to inform our everyday decisions about modeling, feature selection, evaluating loss functions, or understanding the fundamental limits of learning from finite samples.

In this article, we’ll take a thorough, in-depth tour:

1. **Information Theory**  
   - How do we measure *surprise*?  
   - Why do *entropy*, *cross-entropy*, and *KL divergence* matter so much in machine learning?  
   - How does *mutual information* help us measure dependencies?

2. **Covariance & Correlation**  
   - Revisiting *covariance*, *Pearson/Spearman* correlations, and the *covariance matrix*.  
   - Understanding how *Principal Component Analysis (PCA)* leverages covariance to find directions of maximal variance.

3. **Statistical Learning Theory**  
   - The *bias-variance tradeoff*.  
   - *PAC Learning* (Probably Approximately Correct) and *VC Dimension*.  
   - *Occam’s Razor*, generalization bounds, and how they tie into real-world model training.

4. **Applications**  
   - Why *cross-entropy loss* is standard for classification (spoiler: it’s all about *KL divergence*).  
   - How *feature selection* might leverage mutual information or correlation-based heuristics.  
   - The interplay of *regularization*, *model complexity*, and how these concepts show up in *deep learning* generalization.

We’ll close with some reflections on how all these threads connect, especially when building robust solutions from incomplete or uncertain data. Let’s begin where so many theoretical frameworks in ML begin: with **information** itself.

---

## 2. Information Theory

The field of **information theory** was largely shaped by Claude E. Shannon’s seminal 1948 paper, *A Mathematical Theory of Communication*. Shannon sought to quantify “information” in messages, bridging the gap between abstract math and practical engineering constraints (telecommunications, cryptography, data compression). Over the decades, these ideas have deeply influenced machine learning—particularly in understanding **uncertainty**, **similarity of probability distributions**, and how to measure the *information* shared between random variables.

### 2.1 Entropy, Cross-Entropy, and KL Divergence

#### Entropy

At the heart of information theory lies **entropy**, which measures the *average level of “uncertainty”* or *“surprise”* in a probability distribution. Suppose you have a discrete random variable  
$$
X \quad \text{taking values in} \quad \{x_1, x_2, \ldots, x_n\} \quad \text{with} \quad p(x_i) = P(X = x_i).
$$

The **entropy** (in bits, if we use log base 2) is defined as:

$$
H(X) = - \sum_{i=1}^n \; p(x_i) \; \log_2 \, p(x_i).
$$

- If a distribution is highly “peaked”—meaning one outcome is very likely—its entropy is *low*, because there’s little surprise.  
- If a distribution is spread out—many equally likely outcomes—its entropy is *higher*, reflecting greater uncertainty.

**Why it matters**: In machine learning, entropy often appears when we talk about the *purity of a node in decision trees*, or the *uncertainty in a classification problem*, or even *regularization of autoencoders*. High-entropy distributions are more “random”; low-entropy distributions are more “predictable.”

#### Cross-Entropy

When we compare a *true* (or “target”) distribution $$p(x)$$ to a *predicted* distribution $$q(x)$$, we often measure how “off” the prediction is using **cross-entropy**:

$$
H(p, q) = - \sum_{x} \; p(x) \; \log_2 \, q(x).
$$

- In machine learning, especially classification, if we interpret $$p(x)$$ as a one-hot “true label” distribution, and $$q(x)$$ as the model’s predicted probability distribution, then minimizing cross-entropy is basically pushing $$q(x)$$ to match $$p(x)$$.  
- This is exactly why **cross-entropy loss** is so standard for classification tasks. We want $$q(x) \approx p(x)$$ so that the log-likelihood $$\log_2 \, q(x)$$ for the correct class is maximized.

Interestingly, **minimizing cross-entropy** is equivalent to **minimizing KL divergence** plus the constant $$H(p)$$ (the entropy of the true distribution). Which brings us to…

#### Kullback–Leibler (KL) Divergence

**KL divergence** (sometimes called the *relative entropy*) measures how one probability distribution $$p(x)$$ diverges from a second distribution $$q(x)$$:

$$
D_{\mathrm{KL}} \bigl(p \mid\mid q\bigr) = \sum_{x} \; p(x) \;\; \log_2 \!\biggl(\frac{p(x)}{q(x)}\biggr).
$$

- It’s *nonnegative*, and zero only if $$p(x) = q(x)$$ for all $$x$$.  
- It’s *not* symmetric: $$D_{\mathrm{KL}}(p \mid\mid q) \neq D_{\mathrm{KL}}(q \mid\mid p).$$

Also:

$$
H(p, q) = H(p) + D_{\mathrm{KL}}(p \mid\mid q).
$$

So **cross-entropy** can be decomposed into the “true entropy” plus the “extra cost” we pay for using $$q(x)$$ instead of $$p(x)$$.

**Why it matters**: In many ML contexts (logistic regression, neural networks, language modeling, etc.), we see “minimize cross-entropy.” That’s effectively “minimize $$D_{\mathrm{KL}}(p \mid\mid q)$$.” Equivalently, “force $$q$$ to match $$p$$.” The role of KL also surfaces in more advanced topics (variational autoencoders, Bayesian machine learning), as an objective to measure how far a posterior distribution is from a tractable approximation.

### 2.2 Mutual Information

**Mutual information (MI)** quantifies how much knowing one random variable reduces the uncertainty in another. Formally, for discrete variables $$X$$ and $$Y$$:

$$
I(X;Y) 
= \sum_{x, y} \; p(x,y) \; \log_2 \!\biggl(\frac{p(x,y)}{p(x)\,p(y)}\biggr).
$$

Another way to view it:

$$
I(X;Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X),
$$

where $$H(X \mid Y)$$ is the conditional entropy (the average uncertainty in $$X$$ once you know $$Y$$).

**Interpretation**: If $$X$$ and $$Y$$ are *independent*, then $$p(x,y) = p(x)p(y)$$, so $$I(X;Y) = 0$$. If they’re perfectly correlated in some sense, the mutual information is larger.

**Use in ML**:  
1. **Feature selection**: We might pick features $$X_i$$ that have high mutual information with the target $$Y$$ to ensure those features reduce our uncertainty about the label.  
2. **Clustering**: Some advanced approaches to clustering and dimension reduction try to maximize mutual information with the “latent representation.”  
3. **Data transmission**: If we model $$X$$ as signals and $$Y$$ as received signals, MI guides how many bits of “true information” pass through a noisy channel.

---

## 3. Covariance & Correlation

While information theory frames *uncertainty* and *information content*, **covariance and correlation** measure linear (or rank-based) relationships in data. They also feed directly into principal component analysis, which is fundamental for dimensionality reduction.

### 3.1 Covariance Matrix and Pearson/Spearman Correlation

#### Covariance

The **covariance** between two random variables $$X$$ and $$Y$$ is:

$$
\mathrm{Cov}(X, Y) 
= E[(X - E[X])(Y - E[Y])]
= E[XY] - E[X]\;E[Y].
$$

- If $$\mathrm{Cov}(X, Y) > 0$$, then when $$X$$ is above its mean, $$Y$$ tends to be above its mean too (positively associated).  
- If $$\mathrm{Cov}(X, Y) < 0$$, then they move in opposite directions.  
- If $$\mathrm{Cov}(X, Y) = 0$$, they are uncorrelated (though not necessarily independent).

For a set of random variables $$\mathbf{X} = (X_1, \ldots, X_d)$$, the **covariance matrix** $$\Sigma$$ is:

$$
\Sigma_{ij} = \mathrm{Cov}(X_i, X_j).
$$

**Interpretation**: Each entry of $$\Sigma$$ says how strongly variables $$i$$ and $$j$$ co-vary linearly. The diagonal elements $$\Sigma_{ii} = \mathrm{Var}(X_i)$$ are simply the variances of each coordinate.

#### Pearson Correlation

The **Pearson correlation coefficient** between $$X$$ and $$Y$$ is the *normalized* covariance:

$$
\rho_{X,Y}
= \frac{\mathrm{Cov}(X, Y)}
{\sqrt{\mathrm{Var}(X)\,\mathrm{Var}(Y)}}.
$$

It always lies in the range $$[-1, +1]$$. If it’s close to +1, the variables show a strong positive linear relationship; if close to -1, a strong negative linear relationship.

**Why it matters**: Pearson’s correlation often arises in data analysis to see if two features move together linearly. In some ML approaches, we might drop features that have extremely high correlation with others (to avoid redundancy) or extremely low correlation with the target variable (assuming they might not be predictive).

#### Spearman Rank Correlation

Sometimes we care more about *monotonic relationships* than strict linear ones. **Spearman’s rank correlation** measures the correlation between the *rank* of $$X$$ and the *rank* of $$Y$$. If $$X$$ and $$Y$$ move together in a strictly monotonic way, Spearman’s correlation is high, even if it’s not a linear relationship.

**In practice**: Spearman can be more robust to outliers or to nonlinear but monotonic relationships. It’s common in *non-parametric* statistics or quick data explorations to check whether a simpler monotonic link might exist even if linear correlation is small.

### 3.2 Principal Component Analysis (PCA)

**PCA** is a direct, elegant application of covariance to dimensionality reduction. If you have data matrix $$\mathbf{X} \in \mathbb{R}^{N \times d}$$ (with $$N$$ samples, each a $$d$$-dimensional vector), PCA tries to find new axes (principal components) capturing the largest variance in your data.

1. **Compute the Covariance Matrix**  
   We typically first center the data so each feature has mean zero. Then the covariance matrix is:

   $$
   \Sigma = \frac{1}{N} \mathbf{X}^\top \mathbf{X}
   \quad (\text{assuming mean-centered data}).
   $$

2. **Eigen-Decomposition**  
   The principal components are the eigenvectors $$\mathbf{v}_1, \mathbf{v}_2, \ldots$$ of $$\Sigma$$, sorted by their corresponding eigenvalues $$\lambda_1 \ge \lambda_2 \ge \ldots \ge 0$$. The largest eigenvalue indicates the direction of maximum variance.  

3. **Dimension Reduction**  
   To reduce from dimension $$d$$ to dimension $$k$$, you keep only the top $$k$$ eigenvectors:

   $$
   \mathbf{Z} = \mathbf{X} \, [\mathbf{v}_1 \ldots \mathbf{v}_k],
   $$

   so $$\mathbf{Z} \in \mathbb{R}^{N \times k}$$. This often preserves the “most informative” structure in the data, especially if the covariance is well-captured by a few principal directions.

---

## 4. Statistical Learning Theory

While **information theory** addresses uncertainty and data distributions, and **correlation measures** help us see relationships, **statistical learning theory** provides the *mathematical scaffolding* for how we *learn* from data. It asks: “Given finite samples, how well can we expect our model to generalize to unseen data?”

### 4.1 Bias-Variance Tradeoff

One of the earliest conceptual pillars in learning theory is the **bias-variance tradeoff**. Suppose we have a learning algorithm trying to predict a target variable $$Y$$ from features $$X$$. If we train different models on different random samples, the predictions might vary. This variation can be decomposed (under some assumptions) as:

$$
\mathbb{E}\bigl[(Y - \hat{f}(X))^2\bigr]
= \mathrm{Bias}^2(\hat{f}) + \mathrm{Var}(\hat{f}) + \sigma^2,
$$

where

- **Bias** measures how far the *average* model’s predictions are from the *true* target function. A model with *high bias* is underfitting (too simple, missing the real structure).  
- **Variance** measures how sensitive the model is to the particular training set. A model with *high variance* is overfitting (it changes drastically with small perturbations of data).  
- $$\sigma^2$$ is the irreducible noise.

**Tradeoff**: Typically, if you try to reduce bias by using a more complex model, you risk increasing variance. If you clamp down on variance by using a simpler model or stronger regularization, you might increase bias. The sweet spot is to find that balance that yields the best generalization performance.

### 4.2 PAC Learning and VC Dimension

In the late 1980s and early 1990s, a formal mathematical framework emerged called **Probably Approximately Correct (PAC) Learning**. It tries to answer: “Under what conditions, and how quickly, can a learner converge to a hypothesis that is ‘good enough’ with high probability?”

#### PAC Learning

**PAC** states that an algorithm is a PAC learner for a concept class $$\mathcal{C}$$ if, for **any** distribution over the input space and for any $$\epsilon > 0$$ (the accuracy threshold) and $$\delta > 0$$ (the confidence level), there exists a polynomial sample size $$m(\epsilon, \delta)$$ and polynomial time algorithm such that with probability at least $$1 - \delta$$, the learned hypothesis $$h$$ has an error at most $$\epsilon$$.

Formally, if the true concept is $$c \in \mathcal{C}$$, and we measure error as

$$
\text{error}(h) = P_{x \sim D}[\,h(x) \neq c(x)\,],
$$

PAC learning demands

$$
P\Bigl(\text{error}(h) \le \epsilon\Bigr) \ge 1 - \delta,
$$

given enough samples $$m(\epsilon, \delta)$$ from distribution $$D$$. The question is: **how large** must $$m$$ be so that we can guarantee a small error with high probability?

#### VC Dimension

**Vapnik–Chervonenkis (VC) dimension** is a measure of the **capacity** (or complexity) of a set of classifiers. Informally, the VC dimension of a hypothesis class $$\mathcal{H}$$ is the largest number of points that can be **shattered** by $$\mathcal{H}$$—that is, labeled in **all** possible ways by some hypothesis in $$\mathcal{H}$$.

- **Example**: A simple linear classifier in 2D has a VC dimension of 3 (actually it’s infinite in 2D, if we allow real-valued parameters—but for lines in 2D we can show certain bounds). For the sake of simpler illustration, consider that a 1D threshold classifier has a VC dimension of 1.  
- **Interpretation**: If a model class (like a deep neural network or a set of decision trees) has a large VC dimension, it can represent a wide variety of decision boundaries. This means it has a high capacity, potentially can fit very complicated data, but might also risk overfitting.

**Why it matters**: The **VC dimension** ties directly into sample complexity. If your model class has a certain dimension $$d$$, you typically need on the order of $$\mathcal{O}(\frac{d + \ln(1/\delta)}{\epsilon})$$ samples to PAC-learn. This gives a formal handle on how many data points we need relative to the complexity of the model class to ensure good generalization.

### 4.3 Occam’s Razor and Generalization Bounds

**Occam’s Razor** informally states: *Among competing hypotheses that explain the data equally well, the simplest one is likely the best.* In learning theory, a more precise restatement is: *The more complex your model, the more data you need to avoid overfitting, or the stricter your regularization must be.* 

**Generalization bounds** in statistical learning often reflect exactly that principle. For instance, Rademacher complexity, covering numbers, or bounds based on the VC dimension all show that if the class of hypotheses is large, we can fit many possible data patterns. But we pay for that flexibility with the risk of picking up spurious patterns in the sample. The *bound* on test error grows with the complexity measure but shrinks with more data or more regularization.

**In practice**: This is why we see consistent themes like L2-regularization, weight decay, dropout, or simpler model architectures. They limit the complexity of the hypothesis space, effectively harnessing Occam’s Razor: simpler (or better-regularized) hypotheses generalize more reliably, all else being equal.

---

## 5. Applications

These theoretical pillars—**information theory**, **correlation/covariance**, **statistical learning**—translate into a variety of practical data science tasks. Let’s highlight four big areas.

### 5.1 Loss Functions in Classification

**Cross-entropy loss**: As hinted in Section 2, **cross-entropy** is intimately tied to **KL divergence**. For a classification problem with true label distribution $$p(x)$$ (often a one-hot vector) and predicted distribution $$q(x)$$:

$$
\text{CrossEntropy}(p, q)
= - \sum_x \; p(x) \; \log q(x)
= H(p, q).
$$

Minimizing cross-entropy effectively tries to minimize $$D_{\mathrm{KL}}(p \mid\mid q)$$ as well. That’s *the* reason it’s the gold standard for classification tasks: we want our predicted distribution $$q$$ to match the “true” distribution $$p$$ as closely as possible.

- In logistic regression or neural networks, we typically implement cross-entropy with a softmax output layer.  
- Alternative losses (like hinge loss) also exist, but cross-entropy maps beautifully onto the probabilistic interpretation.

### 5.2 Feature Selection and Reduction

- **Mutual Information**: We can measure $$I(X_i; Y)$$ between a candidate feature $$X_i$$ and the target $$Y$$. If mutual information is high, that feature strongly reduces uncertainty in $$Y$$. We might select top features accordingly.  
- **Correlation Coefficients**: We might filter features by correlation with the target or remove sets of features highly correlated with each other.  
- **PCA**: If we suspect redundant dimensions or large amounts of correlated input variables, applying PCA can yield a more compact representation. We keep the top principal components to preserve most variance.

In each case, the core ideas revolve around measuring *how much “signal” a feature or dimension carries* about the problem. That’s precisely the domain where correlation or mutual information excel. Meanwhile, PCA stands out for linear transformations to reduce dimension, often used as a “preprocessing step” for classification or regression tasks.

### 5.3 Regularization and Model Complexity Trade-Offs

**Statistical learning theory** taught us about the **bias-variance** tightrope and **VC dimension**. This directly informs how we handle:

- **Regularization** (like $$L_2$$ or weight decay) shrinks parameters, effectively reducing the capacity of the model.  
- **Dropout** in neural nets randomly zeroes out neurons, making the network less likely to memorize small training-set quirks.  
- **Data augmentation** in vision tasks artificially inflates the training set, reducing overfitting risk, akin to increasing the effective sample size.

All of these serve the same function: keep the model from “overfitting” or “over-expressing itself.” In practice, we want an approach that balances complexity with generalization, consistent with Occam’s Razor.

### 5.4 Understanding Generalization in Deep Learning

Modern deep networks often have **parameter counts** that exceed the size of the training set. Classical generalization bounds (like strict VC dimension arguments) might seem insufficient. Yet in practice, deep nets can generalize well. Why?

- **Implicit Regularization**: Even though the network is huge, the training procedure (stochastic gradient descent + random initialization) doesn’t necessarily search the entire parameter space—only certain directions that yield stable minima. Some argue this exerts an *implicit Occam’s Razor*.  
- **Information Bottleneck Hypothesis**: Proposed by Tishby et al., it relates network layers to *progressively discarding irrelevant information* about the inputs while retaining relevant info about the labels—a viewpoint borrowed from information theory.  
- **Large-scale data**: If we’re working with enormous training sets, the capacity of the model can be matched by the capacity of the data itself. So long as the bias-variance interplay is well-managed (through dropout, weight decay, etc.), we can exploit large capacity without catastrophic overfitting.

Deep learning is thus partly an interplay between *huge capacity* and *various forms of regularization*, explicit or implicit, enabling impressive generalization even when classical bounds might suggest caution.

---

## 6. Conclusion

As we’ve dived deeply into the synergy between **information theory** (uncertainty, entropy, mutual information), **correlation** (covariance matrices, PCA, linear relationships), and **statistical learning theory** (bias-variance, PAC learning, VC dimension), we see how they converge on a shared mission: **understanding data and building models that generalize**.

1. **Information theory** guides how we measure uncertainty and compare distributions.  
2. **Covariance & correlation** help us see linear or monotonic structures, crucial for dimension reduction and data exploration.  
3. **Statistical learning theory** puts a mathematical lens on model complexity, sample requirements, and the trade-offs that ensure we don’t just memorize but genuinely *learn* from finite data.

In the day-to-day world of machine learning, these concepts drive practical choices: we minimize **cross-entropy** to match distributions, we might leverage **mutual information** or **correlation** for feature selection, we keep an eye on the **bias-variance** interplay to choose model capacity, and we rely on **Occam’s Razor** (and sometimes large data) to maintain generalization.  

This interplay is a testament to how integrated these seemingly disparate ideas are. When you tune your model’s complexity, you’re obeying a principle that arises just as naturally from *VC dimension arguments* as it does from *Occam’s Razor*. When you select a feature set, you’re chasing maximum *mutual information* or removing *redundant correlations*. And when you finalize a classification model, **KL divergence** is right there in your cross-entropy objective. It’s all the same tapestry.

In many ways, the story of modern machine learning is about bridging these theoretical concepts to practical algorithms that handle messy real-world data. And as we continue refining these connections—be it in interpretability research, new forms of regularization, or advanced generative models—the synergy among **information theory**, **correlation-based methods**, and **learning-theoretic** frameworks remains at the core of the field’s creativity and progress.
