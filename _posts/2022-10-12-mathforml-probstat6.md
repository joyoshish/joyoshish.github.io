---
layout: post
title: Probability & Statistics for Data Science - Expectation-Maximization (EM) for Unsupervised Learning
date: 2022-10-12
description: Probability & Statistics 6 - Mathematics for Machine Learning
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


Somewhere in the late 1970s, a trio of statisticians—Arthur Dempster, Nan Laird, and Donald Rubin—were wrestling with an age-old challenge: *How do we reliably estimate parameters for a model if some data are missing?* You can imagine them poring over tables of incomplete medical records or half-finished datasets, searching for a principled method to fill those knowledge gaps. Their brilliant insight came in the form of the **Expectation-Maximization (EM) algorithm**, published in a now-classic 1977 paper. Although that might sound like dry academic history, it was pivotal: the EM framework quickly became a foundation of **unsupervised learning** techniques in statistics and machine learning, and remains so today.

I remember first encountering EM in a course on statistical learning, feeling both excited and intimidated. The gist was: EM elegantly solves *latent* or *unobserved* data problems—like figuring out hidden cluster memberships or unknown transition states—by iterating between *guessing* what the hidden parts might be (the **E-step**) and then *re-fitting* your model to those guessed values (the **M-step**). Repeat until your guesses stabilize.

If you’ve heard of **Gaussian Mixture Models (GMMs)** or **Hidden Markov Models (HMMs)**, you’ve probably seen EM in action. At its heart, EM is a log-likelihood maximizer that tiptoes around the fact that we don’t have the full picture—some data or variables are *latent*. But instead of giving up or making random assumptions, EM says, “Let’s make our best estimate of the hidden part, then optimize the parameters as if that guess were correct. We’ll keep iterating until we reach a stable point.” The result, quite often, is a neat solution to an otherwise tricky unsupervised or semi-supervised problem.

In the sections that follow, I’ll explore EM step by step: from the **intuition** behind it, to the **mathematical derivation**, to how it underpins **Gaussian Mixture Models** and **Hidden Markov Models**. We’ll also see it in action for tasks like *soft clustering* or *generative modeling*. But for now, let’s start with the basics:

---

## Relevance in Unsupervised Frameworks

### The Big Picture of EM

The **Expectation-Maximization (EM) algorithm** is a strategy to **estimate parameters** of a statistical model when some variables are hidden or missing. In many data scenarios, we can’t observe all the relevant information directly—perhaps the data themselves are only partially recorded, or we have unlabeled classes that we suspect exist but can’t measure. EM helps us *systematically* handle these hidden variables by alternating between two key steps:

- **E-step (Expectation)**: Guess (or “infer”) the hidden variables in some probabilistic sense, based on the current estimate of the model parameters.
- **M-step (Maximization)**: Update the parameters by maximizing the likelihood (or log-likelihood) using the “completed” data from the E-step.

We keep cycling through these two steps until convergence—meaning the parameter estimates don’t change much anymore. Formally, EM is derived to *locally maximize* the (log) likelihood of the observed data, even though we never see the full picture.

### Latent Variables and Incomplete Data

To understand *why* we need EM, think of a **Gaussian Mixture Model** (GMM). Suppose we have data points $$x_1, x_2, \ldots, x_N$$ coming from, say, *two* different Gaussian distributions, but we don’t know which data point came from which distribution. The *cluster assignments*—the “which mixture component do I belong to?” labels—are *latent variables*. If we had those assignments, we could trivially compute the parameters of each Gaussian. But we don’t. Similarly, in **Hidden Markov Models**, the underlying states (often called *hidden states*) are unobserved. We only see some emissions or observations and want to figure out the hidden sequence.

EM’s brilliant trick is: “Let’s guess the hidden parts (probabilistically) using our current best guess for parameters, then update the parameters as if those guessed hidden parts were correct.” Mathematically, we break the log-likelihood into two parts and use a lower bound that’s tightest at the current guesses, ensuring that each iteration can only increase (or keep the same) the observed log-likelihood.

### EM as Log-Likelihood Maximization

Formally, if our observed dataset is $$\mathbf{X} = \{x_1, \ldots, x_N\}$$ and our *latent* data are $$\mathbf{Z} = \{z_1, \ldots, z_N\},$$ we want to maximize the **log-likelihood**:

$$
\ell(\theta \mid \mathbf{X}) = \log p(\mathbf{X} \mid \theta).
$$

But because $$\mathbf{Z}$$ is hidden, we consider the **complete-data log-likelihood**:

$$
\log p(\mathbf{X}, \mathbf{Z} \mid \theta),
$$

which is easier to handle if we only *knew* $$\mathbf{Z}$$. Since we don’t, EM does a neat dance:

1. **E-step:** Compute the *expected value* of the complete-data log-likelihood, using the *current* parameter guess $$\theta^{(t)}$$:

   $$
   Q(\theta \mid \theta^{(t)}) = \mathbb{E}\big[\log p(\mathbf{X}, \mathbf{Z} \mid \theta)\mid \mathbf{X}, \theta^{(t)}\big].
   $$

2. **M-step:** Choose $$\theta^{(t+1)}$$ to maximize that expectation:

   $$
   \theta^{(t+1)} = \arg \max_{\theta} \, Q(\theta \mid \theta^{(t)}).
   $$

This procedure ensures (or tries to ensure) that each iteration bumps the observed-data log-likelihood $$\log p(\mathbf{X} \mid \theta)$$ up—or at least doesn’t decrease it. And that’s the essence of EM: *iterative, guaranteed to converge to a local maximum*, and surprisingly straightforward to code once you know the formulas for your particular problem.

---


## Where EM Fits in the Machine Learning Workflow

In many real-world scenarios—customer segmentation, natural language processing, recommendation systems, you name it—we either lack labeled data or have partially labeled data. The result? We can’t do a standard “supervised” approach that directly fits a function $$f(x) \rightarrow \text{labels}$$. Instead, we suspect there’s some *hidden structure* in the data.

- In **customer segmentation**, we might hypothesize that there are several “types” of customers (latent classes), each with different buying behaviors. Those classes are never explicitly labeled, but we can guess them from transaction patterns.
- In **topic modeling** (like LDA, which ironically also uses a form of EM under the hood), the topics in text documents are unobserved. We want to uncover them from word distributions.
- In **bioinformatics**, we might guess that a set of gene expressions is generated from a mixture of different cell types. The “cell type membership” is latent.

Where do we turn when faced with unknown or partially unknown data structures? EM is often the first hammer in the toolbox—especially if the underlying model is from the **exponential family** (like Gaussians, Poissons, Bernoullis, etc.).

## Relationship to Other Methods

If you’re coming from a purely supervised background, you might wonder, *Why not just guess the hidden variables and do gradient descent?* In principle, we can do that, but it can become very messy: the objective function can get complicated, and the partial derivatives with respect to hidden variables usually lead to intractable expressions. EM sidesteps that by rewriting the objective in a *neater* form.

In fact, **k-means clustering** can be viewed as a *special case* of the EM algorithm for GMMs when the covariance matrices are forced to be the identity matrix and we treat cluster membership as “hard.” The E-step just assigns points to their nearest cluster, and the M-step recomputes cluster centroids. That’s less flexible but computationally simpler. EM for GMMs generalizes that idea by allowing *soft* assignments and *varying covariance structures*.

## Efficiency, Convergence, and Practical Importance

**Why** is EM so popular? Beyond the theoretical neatness, there’s a practical reason: it’s often *easy to implement*. Once you write down the formula for your model’s complete-data likelihood, the E-step simply uses current parameters to form posterior probabilities of hidden variables (like “the probability that data point $$x_i$$ came from cluster 1”). The M-step is typically a standard maximization problem (e.g., re-estimating the means and covariances for each Gaussian). Repeat until stable.

Moreover, EM is guaranteed to converge to *a* local maximum (though not necessarily the global one). In many real tasks, local maxima are still decent solutions. Finally, it’s widely used because it’s conceptually intuitive: *guess the missing data, then fit as if that guess were correct*. Because of that conceptual clarity—and the fact that real-world data are often incomplete or unlabeled—EM remains a go-to method for unsupervised tasks.

---

## Derivation of the EM Algorithm

The EM algorithm revolves around a seemingly simple question: *How do we maximize the log-likelihood* 
$$\log p(\mathbf{X} \mid \theta)$$ 
*if there are hidden variables* 
$$\mathbf{Z}$$ 
*we can’t observe?* The direct approach—integrating out or summing over every possible 
$$\mathbf{Z}$$
—is often computationally explosive or analytically daunting. EM tames that problem by *introducing* a “clever trick” involving **lower bounds** and **Jensen’s inequality**. Let’s walk through the details.

---

### Incomplete-Data and Complete-Data Log-Likelihoods

We start with two important expressions:

1. **Observed (Incomplete) Log-Likelihood**:

   $$
   \ell(\theta \mid \mathbf{X})
   =
   \log p(\mathbf{X} \mid \theta)
   =
   \log \int p(\mathbf{X}, \mathbf{Z} \mid \theta)\, d\mathbf{Z}.
   $$

   Here, 
   $$\mathbf{Z}$$ 
   represents the *unobserved* or *latent* variables. Because of the integral (or sum, if 
   $$\mathbf{Z}$$ 
   is discrete), it’s typically hard to differentiate and maximize this expression directly with respect to 
   $$\theta$$.

2. **Complete-Data Log-Likelihood**:

   $$
   \log p(\mathbf{X}, \mathbf{Z} \mid \theta).
   $$

   If we somehow *knew* 
   $$\mathbf{Z}$$, 
   we could write down the maximum-likelihood estimates easily by treating 
   $$(\mathbf{X}, \mathbf{Z})$$ 
   as a fully observed dataset. EM leverages this simpler *complete-data* view, but in an iterative framework that accounts for the fact 
   $$\mathbf{Z}$$ 
   is not actually known.

---

### The Jensen’s Inequality Trick

The heart of the derivation is to create a *lower bound* on the observed-data log-likelihood. Suppose we define a distribution 
$$q(\mathbf{Z})$$ 
over the latent variables. (We’ll later make a clever choice for 
$$q$$.) Then:

$$
\log p(\mathbf{X} \mid \theta)
=
\log \int q(\mathbf{Z}) \frac{p(\mathbf{X}, \mathbf{Z} \mid \theta)}{q(\mathbf{Z})} \, d\mathbf{Z}.
$$

By Jensen’s inequality—which says that 
$$\log(\mathbb{E}[\cdot]) \ge \mathbb{E}[\log(\cdot)]$$—we know:

$$
\log \int q(\mathbf{Z}) \frac{p(\mathbf{X}, \mathbf{Z} \mid \theta)}{q(\mathbf{Z})} \, d\mathbf{Z}
\;\ge\;
\int q(\mathbf{Z}) \, \log \Bigl(\frac{p(\mathbf{X}, \mathbf{Z} \mid \theta)}{q(\mathbf{Z})}\Bigr) d\mathbf{Z}.
$$

Define:

$$
\begin{aligned}
\mathcal{L}(\theta, q)
&=
\int q(\mathbf{Z}) \,\log \bigl(p(\mathbf{X}, \mathbf{Z} \mid \theta)\bigr) \, d\mathbf{Z}
-
\int q(\mathbf{Z}) \,\log q(\mathbf{Z}) \, d\mathbf{Z} \\
&=
\mathbb{E}_{q(\mathbf{Z})}[\log p(\mathbf{X}, \mathbf{Z} \mid \theta)]
-
\mathrm{KL}\bigl(q(\mathbf{Z}) \mid\mid p(\mathbf{Z} \mid \mathbf{X}, \theta)\bigr)
+
\text{constant}.
\end{aligned}
$$

The difference between 
$$\log p(\mathbf{X} \mid \theta)$$ 
and 
$$\mathcal{L}(\theta, q)$$ 
is exactly the **Kullback–Leibler (KL) divergence** between 
$$q(\mathbf{Z})$$ 
and the true posterior 
$$p(\mathbf{Z} \mid \mathbf{X}, \theta)$$. So we have:

$$
\log p(\mathbf{X} \mid \theta)
=
\mathcal{L}(\theta, q)
+
\mathrm{KL}\bigl(q(\mathbf{Z}) \mid\mid p(\mathbf{Z} \mid \mathbf{X}, \theta)\bigr).
$$

Since KL divergence is always nonnegative, 
$$\mathcal{L}(\theta, q)$$ 
is a *lower bound* on 
$$\log p(\mathbf{X} \mid \theta)$$:

$$
\mathcal{L}(\theta, q)
\;\le\;
\log p(\mathbf{X} \mid \theta).
$$

---

### Choosing $$q$$ to Tighten the Bound

To make this bound as tight as possible, we notice that:

- The **KL** term vanishes exactly if

  $$
  q(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X}, \theta).
  $$

So the best choice of 
$$q(\mathbf{Z})$$ 
(to maximize the bound) is the *posterior* over 
$$\mathbf{Z}$$ 
given 
$$\mathbf{X}$$ 
and 
$$\theta$$. That means:

$$
\begin{aligned}
\mathcal{L}(\theta, q) 
&= 
\int p(\mathbf{Z} \mid \mathbf{X}, \theta)\,
\log \!\Bigl(\tfrac{p(\mathbf{X}, \mathbf{Z} \mid \theta)}{p(\mathbf{Z} \mid \mathbf{X}, \theta)}\Bigr)\, d\mathbf{Z} \\
&= 
\mathbb{E}_{\mathbf{Z} \sim p(\mathbf{Z} \mid \mathbf{X}, \theta)} 
\bigl[\log p(\mathbf{X}, \mathbf{Z} \mid \theta)\bigr] 
\;-\;
\mathbb{E}_{\mathbf{Z} \sim p(\mathbf{Z} \mid \mathbf{X}, \theta)} 
\bigl[\log p(\mathbf{Z} \mid \mathbf{X}, \theta)\bigr].
\end{aligned}
$$


Rewriting that expectation leads us to the function we call 
$$Q(\theta \mid \theta^{(t)})$$ 
in EM.

---

### The E-Step and M-Step from the Bound Perspective

**Now** we come to the classic two-step iteration:

1. **E-step (Expectation step):**  
   Given the current parameter estimate 
   $$\theta^{(t)}$$, 
   we let:

   $$
   q(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X}, \theta^{(t)}).
   $$

   Equivalently, in many texts:

   $$
   Q(\theta \mid \theta^{(t)})
   =
   \mathbb{E}_{\mathbf{Z} \sim p(\mathbf{Z} \mid \mathbf{X}, \theta^{(t)})}
   \bigl[\log p(\mathbf{X}, \mathbf{Z} \mid \theta)\bigr].
   $$

   This *pins down* the distribution over 
   $$\mathbf{Z}$$ 
   with the *current* parameters, effectively “guessing” the latent information.

2. **M-step (Maximization step):**  
   We update 
   $$\theta$$ 
   by maximizing this *expected* complete-data log-likelihood:

   $$
   \theta^{(t+1)}
   =
   \arg\max_\theta 
   \,
   Q(\theta \mid \theta^{(t)}).
   $$

   By doing so, we maximize the lower bound 
   $$\mathcal{L}(\theta, q(\mathbf{Z}))$$, 
   which in turn ensures 
   $$\log p(\mathbf{X} \mid \theta)$$ 
   *cannot go down*.

Putting it all together:

1. **We fix** 
   $$\theta^{(t)}$$.  
2. **We compute** the distribution of latent variables *given* those parameters, 
   $$p(\mathbf{Z} \mid \mathbf{X}, \theta^{(t)})$$.  
3. **We form** the function 
   $$Q(\theta \mid \theta^{(t)})$$.  
4. **We maximize** 
   $$Q$$ 
   w.r.t. 
   $$\theta$$ 
   to get 
   $$\theta^{(t+1)}$$.  
5. **Repeat** until convergence.

In essence, the E-step *improves our guess of the hidden structure*, and the M-step *improves the parameters given that guess*.

---

### Why It Works

Each iteration increases (or at least does not decrease) the observed-data log-likelihood. The reason is:

- At the E-step, we *tighten the bound* by choosing 
  $$q(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X}, \theta^{(t)})$$.  
- At the M-step, we *maximize* the bound w.r.t. 
  $$\theta$$.

Hence the bound moves upward, and the log-likelihood is at least as high as it was in the previous iteration. This ensures *monotonic convergence* to a local maximum of 
$$\log p(\mathbf{X} \mid \theta)$$.

---

### Key Takeaways of the Derivation

1. **EM is a bound-maximization method.** We don’t attempt to directly maximize 
   $$\log p(\mathbf{X} \mid \theta)$$. Instead, we iteratively maximize a *lower bound* that meets 
   $$\log p(\mathbf{X} \mid \theta)$$ 
   at 
   $$\theta^{(t)}$$.
2. **The E-step** sets 
   $$q(\mathbf{Z})$$ 
   to the posterior of 
   $$\mathbf{Z}$$ 
   under the current parameters, which “completes” the data in a probabilistic way.
3. **The M-step** then solves a simpler, more standard maximization problem involving the complete-data likelihood.
4. **Local maxima** are guaranteed, not the global maximum, but in many real scenarios that’s perfectly acceptable—especially when better global techniques might be far too computationally heavy.

---

With the derivation in hand, we can now apply EM to specific models. Two canonical examples are:

- **Gaussian Mixture Models (GMMs):** We treat cluster memberships 
  $$z_i$$ 
  as latent variables and iterate between assigning data to clusters (E-step) and updating cluster parameters (M-step).
- **Hidden Markov Models (HMMs):** We treat the hidden states of a Markov chain as latent. The E-step uses *Forward-Backward* to find state probabilities, and the M-step updates transition and emission parameters.

We’ll dive into these next. But keep the derivation in mind—it’s the conceptual backbone of all EM-based solutions, whether you’re clustering images, modeling customer segments, or inferring hidden states in a time series. The details of the E-step and M-step change per model, but the structure remains the same: *guess the unobserved data, then maximize the likelihood as if those guesses were true, and repeat*.

---

## EM for Gaussian Mixture Models (GMMs)

Gaussian Mixture Models (GMMs) are a classic example of how the EM algorithm is applied in practice. The high-level idea is simple yet powerful: assume your data come from a mixture of several Gaussian (normal) distributions, each with its own mean and covariance. But we *don’t know* which Gaussian “generated” each data point—and that missing assignment is exactly the **latent variable** EM can handle.

Below, we’ll break down GMMs and show how EM’s E-step and M-step look in this context, then walk through an example in Python.

---

### A Quick Overview of GMMs

A **Gaussian Mixture Model** posits that each data point 
$$x_i \in \mathbb{R}^d$$ 
is drawn from one of 
$$K$$ 
different Gaussian distributions, each with:

- A mixing weight 
  $$\pi_k$$, 
  where 
  $$\sum_{k=1}^K \pi_k = 1$$.  
- A mean vector 
  $$\mu_k \in \mathbb{R}^d$$.  
- A covariance matrix 
  $$\Sigma_k \in \mathbb{R}^{d \times d}$$.

Formally, the probability density for a single data point 
$$x$$ 
under a GMM is:

$$
p(x \mid \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K)
=
\sum_{k=1}^K \pi_k \, \mathcal{N}\bigl(x \mid \mu_k, \Sigma_k\bigr),
$$

where 
$$\mathcal{N}(x \mid \mu, \Sigma)$$ 
denotes the multivariate Gaussian density with mean 
$$\mu$$ 
and covariance 
$$\Sigma$$. The *latent variable* here is the index 
$$z_i \in \{1, \dots, K\}$$ 
that indicates which Gaussian component generated 
$$x_i$$. We don’t observe 
$$z_i$$ 
directly.

---

### Why Use GMMs and Why EM Is Crucial

1. **Soft Clustering**: Unlike k-means, which assigns each point *definitively* to a single cluster, GMMs allow *partial* memberships. A single point might be 60% likely to belong to cluster 1, 30% to cluster 2, etc. This better reflects the uncertain or overlapping nature of real-world data.

2. **Flexible Shapes**: Each cluster can have its own covariance structure, so GMMs can capture elongated or oriented clusters in multidimensional space—far more realistic than the spherical clusters in standard k-means.

3. **Unobserved Cluster Assignments**: We never see which Gaussian distribution generated each data point. If those assignments *were* known, we could directly maximize a *complete-data* log-likelihood by standard formulas. But they’re not—so the *incomplete-data* log-likelihood becomes a sum (or integral) over all possible assignments, which is computationally intractable to optimize directly.

4. **EM as the Solution**: Expectation-Maximization tackles this head-on by iteratively guessing the unobserved component assignments and then re-estimating the parameters. During the **E-step**, it calculates the *expected* cluster membership for each data point (the “responsibilities”). During the **M-step**, it updates cluster parameters as if those responsibilities were correct. This procedure sidesteps the impossibly large sum and reliably converges to a local maximum of the observed log-likelihood.

In short, EM is *essential* for GMMs because it deftly handles the problem of *missing cluster assignments* by weaving between soft membership updates (E-step) and classical parameter estimation (M-step).

---

### EM in Action for GMMs

Let’s denote our observed data by

$$
\mathbf{X} = \{x_1, x_2, \ldots, x_N\}.
$$

We aim to estimate parameters:

$$
\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K.
$$

Recall that EM alternates between:

1. **E-step (Expectation)**  
   - Compute 
     $$\gamma_{ik}$$, 
     the *posterior probability* that data point 
     $$x_i$$ 
     belongs to component 
     $$k$$, 
     given the current parameter estimates. Formally,

     $$
     \gamma_{ik} 
     = 
     p(z_i = k \mid x_i, \theta^{(t)})
     =
     \frac{\pi_k^{(t)} \,\mathcal{N}(x_i \mid \mu_k^{(t)}, \Sigma_k^{(t)})}
          {\sum_{j=1}^K \pi_j^{(t)} \,\mathcal{N}(x_i \mid \mu_j^{(t)}, \Sigma_j^{(t)})}.
     $$

   - Here, 
     $$\theta^{(t)}$$ 
     indicates the parameter estimates from the previous iteration.

2. **M-step (Maximization)**  
   - Update each parameter by maximizing the *expected complete-data likelihood*, which ends up having closed-form solutions for GMMs:
     - Mixing weights:

       $$
       \pi_k^{(t+1)} 
       =
       \frac{1}{N}\,\sum_{i=1}^N \gamma_{ik}.
       $$

     - Means:

       $$
       \mu_k^{(t+1)}
       =
       \frac{\sum_{i=1}^N \gamma_{ik}\,x_i}{\sum_{i=1}^N \gamma_{ik}}.
       $$

     - Covariances:

       $$
       \Sigma_k^{(t+1)}
       =
       \frac{\sum_{i=1}^N \gamma_{ik}\,(x_i - \mu_k^{(t+1)})(x_i - \mu_k^{(t+1)})^\top}{\sum_{i=1}^N \gamma_{ik}}.
       $$

   - Here 
     $$\gamma_{ik}$$ 
     acts like a “soft count” of how many points belong to cluster 
     $$k$$.

3. **Repeat**  
   - Continue iterating E-step and M-step until convergence (i.e., until the parameter updates don’t change much or the log-likelihood improvement is below a threshold).

The result is a locally optimal set of mixture weights, means, and covariances.

---

### Typical Use Cases

GMMs are employed in tasks like:

- **Clustering**: A more nuanced alternative to k-means, especially when clusters aren’t spherical.  
- **Density Estimation**: Helpful for anomaly detection (points with low mixture probability might be outliers).  
- **Feature Extraction**: Common in speech recognition tasks (modeling different phonemes or subword units).  
- **Partial Memberships in Segmentation**: E.g., in marketing or biology, where an individual could reasonably belong to multiple groups with different degrees of membership.

---

### An Illustrative Example in Python

Let’s illustrate with a short example. We’ll:

1. Generate synthetic 2D data from three Gaussians.  
2. Fit a GMM with EM (from scratch).  
3. Visualize the (soft) cluster assignments.

#### Generating Synthetic Data

```python
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for three Gaussians
means = [np.array([0, 0]), np.array([5, 5]), np.array([-5, 5])]
covs = [np.array([[1, 0], [0, 1]]),
         np.array([[1.5, 0.3], [0.3, 1]]),
         np.array([[1, -0.4], [-0.4, 1]])]
weights = [0.3, 0.5, 0.2]  # Must sum to 1

# Number of points
N = 600

# Generate data
X = []
for i in range(N):
    # Pick a component based on the weights
    k = np.random.choice(len(weights), p=weights)
    sample = np.random.multivariate_normal(means[k], covs[k], 1)
    X.append(sample[0])
X = np.array(X)
```

We now have a 2D dataset `X` that’s drawn from one of three Gaussians (latent). We only see `(x, y)`, not which cluster generated each point.

#### EM Implementation from Scratch

```python
def gaussian_pdf(x, mean, cov):
    """Compute multivariate Gaussian density at point x."""
    d = len(x)
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    norm_const = 1.0 / ( (2*np.pi)**(d/2) * np.sqrt(det_cov) )
    diff = (x - mean).reshape(-1,1)
    exponent = -0.5 * diff.T.dot(inv_cov).dot(diff)
    return float(norm_const * np.exp(exponent))

def gmm_em(X, K=3, max_iters=100, tol=1e-4):
    N, d = X.shape
    
    # 1. Initialize parameters randomly
    np.random.seed(0)
    idx = np.random.choice(N, K, replace=False)
    mus = X[idx, :]               # means
    covs = [np.eye(d) for _ in range(K)]  # identity cov
    pis = np.ones(K)/K            # mixing weights
    
    prev_log_likelihood = -np.inf
    
    for iteration in range(max_iters):
        # E-step: compute responsibilities gamma[i, k]
        gamma = np.zeros((N, K))
        for i in range(N):
            for k in range(K):
                gamma[i, k] = pis[k] * gaussian_pdf(X[i], mus[k], covs[k])
            # Normalize across k
            gamma[i, :] /= np.sum(gamma[i, :])
        
        # M-step: update pis, mus, and covs
        Nk = np.sum(gamma, axis=0)  # effective number of points per cluster
        
        # Update mixing weights
        pis = Nk / N
        
        # Update means
        mus = np.array([
            np.sum(gamma[:, k].reshape(-1,1) * X, axis=0) / Nk[k]
            for k in range(K)
        ])
        
        # Update covariances
        for k in range(K):
            cov_k = np.zeros((d, d))
            for i in range(N):
                diff = (X[i] - mus[k]).reshape(-1,1)
                cov_k += gamma[i, k] * diff.dot(diff.T)
            covs[k] = cov_k / Nk[k]
        
        # Compute log-likelihood for convergence check
        log_likelihood = 0.0
        for i in range(N):
            mix_sum = 0.0
            for k in range(K):
                mix_sum += pis[k] * gaussian_pdf(X[i], mus[k], covs[k])
            log_likelihood += np.log(mix_sum + 1e-12)  # add small term for safety
        
        if np.abs(log_likelihood - prev_log_likelihood) < tol:
            break
        prev_log_likelihood = log_likelihood
    
    return pis, mus, covs, gamma, log_likelihood

pis_est, mus_est, covs_est, gamma_est, final_ll = gmm_em(X, K=3, max_iters=100)
print("Estimated mixing weights:", pis_est)
print("Estimated means:\n", mus_est)
print("Estimated covariances:\n", covs_est)
print("Final log-likelihood:", final_ll)
```


- **E-step**: We compute 
  $$\gamma_{ik}$$ 
  (here called `gamma[i,k]`) for each point 
  $$x_i$$ 
  and each mixture component 
  $$k$$.
- **M-step**: We use these “soft assignments” to update mixing weights 
  $$\pi_k$$, 
  means 
  $$\mu_k$$, 
  and covariances 
  $$\Sigma_k$$.
- **Log-likelihood** is checked to see if we should stop.

**Expected Output**:

```text
Estimated mixing weights: [0.30809167 0.2086329  0.48327543]

Estimated means:
[[ 0.01233236  0.12055429]
 [-5.02351291  5.06080423]
 [ 5.03601621  5.14670795]]

Estimated covariances:
[array([[ 1.09963883, -0.01221444],
       [-0.01221444,  1.0573512 ]]),
 array([[ 0.95633523, -0.43613791],
       [-0.43613791,  1.27380056]]),
 array([[1.33755208, 0.22850451],
       [0.22850451, 0.82785559]])]

Final log-likelihood: -2349.5595212288563
```





#### Interpreting the Output

1. **Estimated Mixing Weights (`pis_est`)**: These should sum to approximately 1 and reflect the proportion of data points that the algorithm assigns to each of the three Gaussians (on average).  
2. **Estimated Means (`mus_est`)**: Each row corresponds to the mean of one mixture component. If things go well, they should be reasonably close to the original `means` values we used to generate the data (around `[0, 0]`, `[5, 5]`, and `[-5, 5]`).  
3. **Estimated Covariances (`covs_est`)**: Each element will be a 
   $$2\times2$$ 
   matrix reflecting the shape/orientation of each Gaussian cluster. Again, we expect these to be *similar* to our ground-truth covariances if the algorithm converges well.  
4. **Final Log-Likelihood (`final_ll`)**: A numeric measure of how plausible our data are under the fitted model. Over iterations, this value typically *increases* until EM converges.

Because EM converges to a *local* maximum, exact matches to our “true” parameters aren’t guaranteed. Nonetheless, you’ll often see results in the ballpark of the original mixture components—especially if you start from decent initial guesses or run multiple times with different random seeds.

#### Visualizing Results

```python
import matplotlib.pyplot as plt

# Predict cluster labels by picking the argmax of gamma
labels = np.argmax(gamma_est, axis=1)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=labels)
plt.title("GMM Clustering via EM")
plt.show()
```

<div style="text-align: center; margin: 2rem 0;">
  {% include figure.liquid 
      path="assets/img/probstat6_1.png" 
      class="img-fluid rounded shadow-sm" 
      loading="eager" 
      zoomable=true 
      alt="" 
  %}
</div>

You see three blobs corresponding (roughly) to the three Gaussian components. Points near cluster boundaries may be assigned in a “soft” manner (they might have nearly equal responsibilities for more than one cluster)—but for visualization, we show the *most likely* cluster each point belongs to.

---

### Takeaways

- **EM for GMM** is a prime example of how we handle *latent variables* (the unknown cluster assignments) in an iterative manner.  
- The procedure *monotonically increases* the log-likelihood until convergence, though it’s susceptible to local maxima.  
- Libraries like `scikit-learn` provide a user-friendly implementation (`GaussianMixture`), but the from-scratch version clarifies the mechanics behind the scenes.

Next, we’ll turn to **Hidden Markov Models (HMMs)**—another area where EM thrives, but with hidden states in sequences rather than hidden cluster memberships in i.i.d. data points.


---

## EM for Hidden Markov Models (HMMs)

Hidden Markov Models (HMMs) are another quintessential setting for the EM algorithm—this time in **sequence modeling**. Whereas a Gaussian Mixture Model treats cluster assignments as latent variables for i.i.d. data points, an HMM treats *state sequences* as latent variables for a chain of observations. You see only the emitted symbols (or values), but not the underlying states that generated them. EM steps in to iteratively guess those hidden states (E-step) and then refine the transition and emission parameters (M-step).

Below, we’ll walk through what an HMM is, why EM is essential, and how the algorithm (often called the **Baum-Welch** algorithm in this context) actually works in detail—complete with a small Python example.

---

### A Quick Glance at HMMs

In a **Hidden Markov Model**, we assume:

1. There are $$K$$ hidden states $$s_1, s_2, \ldots, s_K$$.  
2. At each time $$t$$, the system is in some hidden state $$z_t \in \{s_1, \ldots, s_K\}$$.  
3. The next state $$z_{t+1}$$ depends *only* on the current state $$z_t$$ (the Markov assumption), via a transition probability:

   $$
   a_{ij} = p\bigl(z_{t+1} = s_j \mid z_t = s_i\bigr).
   $$

4. Each hidden state $$s_i$$ emits an observation $$x_t$$ according to some distribution $$b_i(x)$$. For discrete observations, that might be:

   $$
   b_i(x) = p\bigl(x_t = x \mid z_t = s_i\bigr).
   $$

5. We observe a sequence of outputs 
   $$
   \mathbf{X} = (x_1, x_2, \ldots, x_T)
   $$
   without ever seeing the hidden state sequence 
   $$
   \mathbf{Z} = (z_1, z_2, \ldots, z_T).
   $$

The model parameters often include:

1. An **initial state distribution** 
   $$
   \pi = [\pi_1, \ldots, \pi_K],
   $$
   where 
   $$
   \pi_i = p\bigl(z_1 = s_i\bigr).
   $$  
2. A **transition matrix** 
   $$
   A = [a_{ij}],
   $$
   with 
   $$
   a_{ij} = p\bigl(z_{t+1} = s_j \mid z_t = s_i\bigr).
   $$
3. An **emission distribution** $$B$$. In the discrete case, 
   $$
   b_{i}(x)
   $$
   is the probability of emitting symbol $$x$$ if the HMM is in state $$s_i$$.

We want to find or refine these parameters 
$$
\theta = (\pi, A, B)
$$
to explain our observed sequence(s). Since we never see the state sequence 
$$\mathbf{Z}$$, we turn to **EM**—a specialized form known as **Baum-Welch**.

---

In an HMM, we have a sequence of observations that are produced by a chain of *hidden states.* If we knew exactly what that hidden sequence of states was, we could use standard maximum-likelihood formulas: just count how often each state and state transition appears, and how often each state emits each symbol. But the catch is that we never observe those states directly, and enumerating all possible state sequences is generally intractable as the sequence length grows.

**Why HMMs Need EM**

1. **Latent (Hidden) States:** Much like missing cluster memberships in a Gaussian Mixture Model, we don’t directly see the hidden states in an HMM. If the states were given, maximum likelihood estimation would be straightforward. But they’re not—so we have *incomplete data*.

2. **Exponential Possibilities:** A sequence of length $$T$$ has up to $$K^T$$ possible hidden state configurations if the HMM has $$K$$ states. Attempting to handle them all explicitly is impossible for nontrivial $$T$$.

3. **“Guess Latent, Then Optimize”**: EM provides a structured way to estimate the posterior probability of the hidden states (the “guess” part) and then re-estimate the model parameters as if that guess were truth (the “optimize” part). Repeating these two steps converges to a local maximum of the likelihood.

**Baum-Welch as a Special Case of EM**

- **EM, at its core,** alternates between an *Expectation (E)-step* that computes the *expected values* of “hidden” variables (or hidden data) given the current parameters, and a *Maximization (M)-step* that updates the parameters to best fit those newly filled-in data.  
- **For HMMs,** the hidden data are the *state sequences.* We use the **Forward-Backward** algorithm (a dynamic programming approach) to compute the posterior distribution over states—i.e., how likely we were in state $$s_i$$ at time $$t$$, or how likely we transitioned from state $$s_i$$ at time $$t$$ to $$s_j$$ at time $$t+1$$. This *is* the E-step, creating the “soft counts” of how often each state or transition occurs.  
- **Next,** we update the model parameters (initial distribution, transition probabilities, and emission probabilities) by normalizing these “counts.” This *is* the M-step.  
- **“Baum-Welch”** is just the name given to this EM procedure in the context of HMMs, derived from the foundational 1970 paper by Leonard E. Baum and others. Under the hood, it’s the same principle as EM:  
  1. *Impute* (E-step) the missing data (the unobserved states) in a probabilistic sense.  
  2. *Re-estimate* (M-step) parameters using those imputed values.  
  3. Repeat until convergence.

Hence, Baum-Welch = EM specialized to HMM structure. Instead of summing over an exponential set of state paths, we use forward-backward to do it efficiently. Then, at the M-step, we have elegant closed-form updates for the transition and emission probabilities—very much like the GMM case (where the E-step was computing cluster responsibilities, and the M-step was re-estimating means, covariances, and mixing weights).

In short, EM is indispensable here because we can’t do direct maximum likelihood without enumerating or otherwise dealing with all possible state paths. The forward-backward procedure in the E-step elegantly sums over them implicitly in $$ \mathcal{O}(K^2 T) $$ time.

---

### How EM (Baum-Welch) Works for HMMs

Let’s define our data as a sequence of observations:

$$
\mathbf{X} = (x_1, x_2, \ldots, x_T).
$$

We want to maximize

$$
\log p\bigl(\mathbf{X} \mid \theta\bigr),
$$

where 
$$
\theta = (\pi, A, B).
$$ 

The hidden data 
$$
\mathbf{Z}
$$ 
is the sequence of states:

$$
\mathbf{Z} = (z_1, z_2, \ldots, z_T).
$$

The E-step uses the **Forward-Backward** algorithm to compute “soft counts” of how often each state or state transition is used at each time. The M-step then uses those counts to update 
$$\pi, A,$$ and $$B$$.

#### E-step: Forward-Backward

1. **Forward Pass** ($$\alpha$$-values):  
   - $$\alpha_t(i)$$ = probability of the partial observation 
     $$x_1, \ldots, x_t$$ **and** being in state 
     $$s_i$$ at time 
     $$t$$.  
   - Initialize:

     $$
     \alpha_1(i) = \pi_i \, b_i\bigl(x_1\bigr).
     $$

   - Recursively compute:

     $$
     \alpha_{t+1}(j) 
     = \Bigl[\sum_{i=1}^K \alpha_t(i)\,a_{ij}\Bigr]\; b_j\bigl(x_{t+1}\bigr).
     $$

2. **Backward Pass** ($$\beta$$-values):  
   - $$\beta_t(i)$$ = probability of the **future** observations 
     $$x_{t+1}, \ldots, x_T$$ given that we’re in state 
     $$s_i$$ at time 
     $$t$$.  
   - Initialize:

     $$
     \beta_T(i) = 1 \quad \text{for all } i.
     $$

   - Recursively compute (going backward):

     $$
     \beta_t(i) 
     = \sum_{j=1}^K a_{ij}\,b_j\bigl(x_{t+1}\bigr)\,\beta_{t+1}(j).
     $$

3. **Compute Responsibilities**:  
   - Probability that we are in state $$s_i$$ at time $$t$$:

     $$
     \gamma_t(i) = p\bigl(z_t = s_i \mid \mathbf{X}, \theta\bigr)
     = \frac{\alpha_t(i)\,\beta_t(i)}{p\bigl(\mathbf{X} \mid \theta\bigr)},
     $$

     where 

     $$
     p\bigl(\mathbf{X} \mid \theta\bigr) 
     = \sum_{i=1}^K \alpha_T(i)
     = \sum_{i=1}^K \alpha_t(i)\,\beta_t(i)\quad \text{(for any } t).
     $$

   - Probability we transition from $$s_i$$ at time $$t$$ to $$s_j$$ at time $$t+1$$:

     $$
     \xi_t(i,j) 
     = p\bigl(z_t=s_i,\,z_{t+1}=s_j \mid \mathbf{X}, \theta\bigr)
     = \frac{\alpha_t(i)\,a_{ij}\,b_j\bigl(x_{t+1}\bigr)\,\beta_{t+1}(j)}{p\bigl(\mathbf{X} \mid \theta\bigr)}.
     $$

$$\gamma_t(i)$$ and $$\xi_t(i,j)$$ are the “soft counts” that, in the M-step, let us re-estimate $$\pi, A, B$$.

#### M-step: Parameter Updates

Using the expected counts from the E-step, the standard re-estimations are:

1. **Initial State Distribution**:

   $$
   \pi_i^{(t+1)} = \gamma_1(i).
   $$

   (Probability of starting in state $$s_i$$ is just the expected probability we begin in $$s_i$$.)

2. **Transition Matrix**:

   $$
   a_{ij}^{(t+1)}
   = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}
          {\sum_{t=1}^{T-1} \gamma_t(i)}.
   $$

   (Probability of going from $$s_i$$ to $$s_j$$ equals the fraction of times we’re in $$s_i$$ at time $$t$$ that we then transition to $$s_j$$ at $$t+1$$.)

3. **Emission Probabilities** (discrete case):

   $$
   b_i^{(t+1)}(x) 
   = \frac{\sum_{t=1}^T \gamma_t(i)\,\mathbb{I}\bigl[x_t = x\bigr]}
          {\sum_{t=1}^T \gamma_t(i)}.
   $$

   We count how often state $$s_i$$ “emits” symbol $$x$$ in the sequence, relative to how often we’re in $$s_i$$.

---

### Putting It All Together

In practice, the **Baum-Welch** (EM) algorithm for HMMs goes like this:

1. **Initialize** $$\pi, A, B$$ randomly or heuristically.  
2. **E-step**: Use Forward-Backward to compute $$\alpha_t(i)$$, $$\beta_t(i)$$, then find $$\gamma_t(i)$$ and $$\xi_t(i,j)$$.  
3. **M-step**: Update $$\pi_i$$, $$a_{ij}$$, and $$b_i(x)$$ according to the formulas above.  
4. **Check** log-likelihood $$p(\mathbf{X}\mid\theta)$$. If it’s changed very little from last iteration, stop. Otherwise, repeat.

---

### A Mini Python Example

Below is a simplified example for a discrete HMM with 2 hidden states ($$K=2$$) and a small observation alphabet (e.g. $$\{0,1\}$$). We’ll illustrate the core Forward-Backward plus re-estimation steps for a single observation sequence.

```python
import numpy as np

def forward_backward(observations, pi, A, B):
    """
    Perform the forward-backward algorithm on a single sequence of observations.
    observations: array-like of shape (T,)
    pi: initial distribution, shape (K,)
    A: transition matrix, shape (K,K)
    B: emission matrix, shape (K,V) for discrete outputs
    returns alpha, beta, and the log-likelihood of the sequence
    """
    T = len(observations)
    K = len(pi)
    
    # Forward pass (alpha)
    alpha = np.zeros((T, K))
    alpha[0] = pi * B[:, observations[0]]
    for t in range(1, T):
        for j in range(K):
            alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, observations[t]]
    
    # Backward pass (beta)
    beta = np.zeros((T, K))
    beta[T-1] = 1
    for t in range(T-2, -1, -1):
        for i in range(K):
            beta[t, i] = np.sum(A[i, :] * B[:, observations[t+1]] * beta[t+1])
    
    # Probability of the entire sequence
    seq_prob = np.sum(alpha[T-1])
    log_likelihood = np.log(seq_prob + 1e-12)
    
    return alpha, beta, log_likelihood

def compute_xi_gamma(alpha, beta, A, B, observations):
    """
    Compute xi_t(i,j) and gamma_t(i) for an HMM with discrete emissions.
    """
    T, K = alpha.shape
    xi = np.zeros((T-1, K, K))
    gamma = np.zeros((T, K))
    
    seq_prob = np.sum(alpha[-1])
    
    # xi
    for t in range(T-1):
        denom = seq_prob
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = alpha[t, i] * A[i, j] * B[j, observations[t+1]] * beta[t+1, j]
        xi[t] /= (denom + 1e-12)
    
    # gamma
    for t in range(T):
        gamma[t] = (alpha[t] * beta[t]) / (seq_prob + 1e-12)
    
    return xi, gamma

def baum_welch(observations, K, V, max_iter=100, tol=1e-4):
    """
    Baum-Welch for a single observation sequence of discrete emissions.
    K: number of hidden states
    V: number of possible emission symbols
    """
    np.random.seed(0)
    
    T = len(observations)
    
    # Random initialization
    pi = np.ones(K) / K
    A = np.random.dirichlet(np.ones(K), K)  # each row sums to 1
    B = np.random.dirichlet(np.ones(V), K)  # each row sums to 1 (emission distribution)
    
    prev_log_likelihood = -np.inf
    
    for iteration in range(max_iter):
        # E-step
        alpha, beta, ll = forward_backward(observations, pi, A, B)
        xi, gamma = compute_xi_gamma(alpha, beta, A, B, observations)
        
        # M-step
        # Update pi
        pi = gamma[0]
        
        # Update A
        for i in range(K):
            denom = np.sum(gamma[:-1, i])
            for j in range(K):
                A[i, j] = np.sum(xi[:, i, j])
            A[i, :] /= (denom + 1e-12)
        
        # Update B
        for i in range(K):
            denom = np.sum(gamma[:, i])
            for v in range(V):
                mask = (observations == v)
                B[i, v] = np.sum(gamma[mask, i])
            B[i, :] /= (denom + 1e-12)
        
        if abs(ll - prev_log_likelihood) < tol:
            break
        prev_log_likelihood = ll
    
    return pi, A, B, ll

# Example usage
observations = np.array([0,1,1,0,1,0,0,1])  # small discrete sequence
K = 2  # two hidden states
V = 2  # two possible observation symbols: {0,1}

pi_est, A_est, B_est, final_ll = baum_welch(observations, K, V, max_iter=50)
print("Estimated initial distribution:", pi_est)
print("Estimated transition matrix:\n", A_est)
print("Estimated emission matrix:\n", B_est)
print("Final log-likelihood:", final_ll)
```

In this snippet:

- **forward_backward** calculates $$\alpha$$ and $$\beta$$ arrays and returns the log-likelihood.  
- **compute_xi_gamma** uses $$\alpha$$ and $$\beta$$ to produce $$\xi_t(i,j)$$ and $$\gamma_t(i)$$.  
- **baum_welch** does the E-step (forward-backward + $$\xi, \gamma$$) and M-step (reestimate $$\pi, A, B$$), iterating until convergence or a max iteration count.

**Expected output:**

```text
Estimated initial distribution: [1.00000000e+00 4.32581465e-66]

Estimated transition matrix:
 [[0.24991271 0.75008729]
 [0.6665381  0.3334619 ]]

Estimated emission matrix:
 [[9.99999999e-01 1.18532237e-09]
 [1.29670998e-04 9.99870329e-01]]

Final log-likelihood: -4.159082490200387
```
Here’s how we can interpret the output:

1. **Estimated Initial Distribution**:  
   ```
   [1.00000000e+00 4.32581465e-66]
   ```
   - This vector says the model believes there’s $$\approx 100\%$$ chance the sequence *starts* in State 0 and essentially $$0\%$$ chance it starts in State 1. (The extremely small $$4.33 \times 10^{-66}$$ is a rounding artifact indicating it almost never starts in State 1.)  
   - This outcome usually reflects that your sequence likely begins with an observation best explained by State 0 in the final model.

2. **Estimated Transition Matrix**:  
   ```
   [[0.24991271 0.75008729]
    [0.6665381  0.3334619 ]]
   ```
   - For the first row (State 0): $$25\%$$ chance of staying in State 0, $$75\%$$ chance of transitioning to State 1 at the next time step.  
   - For the second row (State 1): $$67\%$$ chance of jumping back to State 0 next, $$33\%$$ chance of staying in State 1.  
   - In other words, the model sees frequent alternation between states—although from State 1 there’s a two-thirds likelihood to move back to State 0, and from State 0 there’s a three-fourths likelihood to move to State 1.

3. **Estimated Emission Matrix**:  
   ```
   [[9.99999999e-01 1.18532237e-09]
    [1.29670998e-04 9.99870329e-01]]
   ```
   - Each row corresponds to a hidden state, each column to an observed symbol (0 or 1).  
   - First row (State 0) is approximately $$[1.0,\, 1.18 \times 10^{-9}]$$. That means if you’re in State 0, there’s effectively $$100\%$$ chance to emit 0 and almost no chance to emit 1.  
   - Second row (State 1) is approximately $$[1.30 \times 10^{-4},\, 0.99987]$$. That means if you’re in State 1, you’ll almost always emit 1, and only extremely rarely emit 0.

   This strongly indicates the model has learned that **State 0** $$\approx$$ always emits 0 and **State 1** $$\approx$$ always emits 1.

4. **Final Log-Likelihood**:
   ```
   -4.159082490200387
   ```
   - The log-likelihood of the data given this model is about $$-4.159$$. Over the EM iterations, you probably saw this value increase (become less negative) until convergence.

---

### Overall Interpretation

- **State 0** is effectively the “0-emitting” state.  
- **State 1** is effectively the “1-emitting” state.  
- **Initial State**: The sequence almost certainly starts in State 0.  
- **Transitions**: Although each state has a strong preference for a particular symbol, it’s quite willing to switch back and forth ($$0 \to 1$$ at $$75\%$$ chance, $$1 \to 0$$ at $$66\%$$).

Given the input sequence $$[0,1,1,0,1,0,0,1]$$, it’s not surprising the model has latched onto a straightforward “0-state” and “1-state,” with a decent probability of flipping between them—this closely matches the pattern of the data. The final log-likelihood of $$-4.159$$ simply tells you how well this model explains that short sequence; it is a locally optimal fit given your initialization and the EM updates.

---

### Real-World Applications

1. **Speech Recognition**: A classic domain where HMMs model phonemes/states and observed acoustic signals.  
2. **Bioinformatics**: Gene prediction or protein sequence analysis, where hidden states might represent functional regions in the DNA/protein.  
3. **Financial Time Series**: Modeling market regimes as hidden states and price movements as emissions.  
4. **Natural Language Processing**: Part-of-speech tagging can be approached via HMMs, with hidden states as POS tags and observed words as emissions.

Wherever you suspect a sequence of hidden causes that produce observable data, HMMs (with EM) are a time-tested approach—though more modern techniques (like deep learning or more flexible generative models) can sometimes surpass them.

---

### Key Points

- **Hidden states** in an HMM are the analog of “which cluster?” in a GMM, except they form a Markov chain.  
- **Forward-Backward** is the *core* procedure for the E-step, summarizing over all possible hidden-state paths.  
- **Baum-Welch** is EM by another name: we guess the hidden sequences (E-step) and refine parameters (M-step).  
- Like all EM-based methods, it guarantees *monotonic improvements* in likelihood but can get stuck in local maxima.

With that, you have the essential structure of the EM algorithm for Hidden Markov Models. While the GMM scenario deals with i.i.d. points, HMMs apply EM logic to time-dependent (or sequence-dependent) data. The underlying principle is the same: *impute the unobserved variables (states), then do a standard MLE update, repeat, converge.*


---

## Applications of EM

### 1. Soft Clustering

One of the most intuitive applications of EM is **soft clustering**, where each data point is *partially* (rather than exclusively) assigned to one or more clusters. 

- **Contrast with Hard Clustering**  
  In hard clustering (e.g., **k-means**), each data point $$x_i$$ is definitively assigned to one cluster index. But in many real-world scenarios—customer segmentation, image processing, topic discovery—points can plausibly belong to multiple clusters with varying degrees of membership.  

- **Gaussian Mixture Models (GMMs)**  
  EM’s canonical example for soft clustering is the **Gaussian Mixture Model** (GMM). Each data point $$x_i$$ is assumed to come from one of $$K$$ Gaussian components with parameters 
  $$
  \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K,
  $$
  but we *don’t know* which component. During the **E-step**, we compute the “responsibility”  
  $$
  \gamma_{ik} = p(z_i = k \mid x_i, \theta),
  $$
  which is a *fractional* membership of point $$x_i$$ in cluster $$k$$. Thus, instead of a single “best” cluster, each point has a probability distribution over clusters.  

- **Why Soft Assignments Matter**  
  1. **Overlap**: Real data often has overlapping group structures (e.g., a person might be both a “bargain hunter” and “tech enthusiast”).  
  2. **Uncertainty Quantification**: Soft memberships let you gauge *how* confidently a point belongs to a given cluster. You might see that one point is 90% likely in cluster A and 10% in cluster B.  
  3. **Smarter Decision-Making**: Instead of forcing a point into one group, you can weight analyses by partial memberships.  

In practice, you can use GMM-based clustering to **segment** images, **classify** documents by topic in a probabilistic way, or **group** customers with partial overlap. EM ensures that the cluster parameters and membership probabilities converge to a local optimum of the likelihood.

---

### 2. Generative Modeling

**Generative models** try to capture the underlying data distribution so that we can *generate new samples* resembling the original data. EM is particularly handy in such models because of its iterative approach to fill in “missing” or “latent” structure:

- **Gaussian Mixtures as Generators**  
  Once you fit a GMM to data, you can generate new points by:
  1. Sampling a cluster index $$k$$ according to $$\pi_k$$.  
  2. Sampling a data point from the Gaussian distribution 
     $$
     \mathcal{N}(x \mid \mu_k, \Sigma_k).
     $$  
  The resulting synthetic data has *similar statistical properties* to the original dataset, capturing multiple modes (clusters) if they exist.

- **Hidden Markov Models (HMMs) for Sequence Generation**  
  With an HMM, once you learn the transition matrix $$A$$ and emission distributions $$B$$, you can generate new sequences by:
  1. Sampling an initial state $$z_1 \sim \pi$$.  
  2. Iteratively sampling the next state $$z_{t+1} \sim p(\cdot \mid z_t, A)$$ and an emission $$x_{t+1} \sim b_{z_{t+1}}(\cdot).$$  
  This is invaluable in speech synthesis, text generation, or any domain where you want realistic sequences that follow a Markov structure.

- **Why Generative Models?**  
  1. **Simulation & Synthetic Data**: If you need more data for training or testing, a well-fit generative model can produce plausible samples.  
  2. **Understanding Complex Distributions**: Instead of a single “best guess,” you get an entire data-generating process.  
  3. **Handling Missing Data**: Generative models can impute or fill in missing values by sampling from the learned distribution. EM naturally complements this because it was designed to handle incomplete or latent information.

By formulating your model with latent variables, EM helps find parameter estimates that maximize the likelihood of observed data—turning your model into a powerful generative tool.

---

### 3. Unsupervised Learning with Latent Variables

Perhaps the **broadest** umbrella under which EM operates is **unsupervised learning** with latent variables. In many tasks, the variables or labels you’d love to have are either partially observed or entirely missing. EM’s entire philosophy is: “Guess those latent parts, then optimize.” Some prominent scenarios include:

1. **Clustering Without Labels**  
   - We never had class labels, so the cluster assignments are *latent*. 
   - GMM-based clustering is a perfect example, but more complex forms (e.g., mixtures of Bernoulli, Poisson, or other distributions) also rely on EM to handle hidden assignments.

2. **Topic Modeling**  
   - In Latent Dirichlet Allocation (LDA), the *topic distributions* of each document and the *word distributions* of each topic are latent. Although LDA typically uses Variational Inference or Gibbs Sampling, an EM formulation is entirely possible (and historically used). 
   - The E-step guesses how likely each word is assigned to each topic. The M-step re-estimates topic-word distributions.

3. **Dimensionality Reduction & Factor Analysis**  
   - When the data is high-dimensional, you might hypothesize a lower-dimensional latent structure. For instance, in **Factor Analysis**, each observation $$x_i$$ is generated from a lower-dimensional latent vector $$z_i$$ plus noise. EM can be used to iteratively estimate the latent factors $$z_i$$ and the loading matrix that maps latent factors to the observed space.

4. **Incomplete Data (Missing Values)**  
   - Suppose your dataset has missing entries (like half-filled medical forms). EM can help by treating the unknown values as latent variables. In the E-step, you “fill in” or compute the distribution of those missing entries given current parameters. The M-step then updates your model (e.g., means, covariances) using these estimates.

Across all these examples, the unifying idea is that **part of your data is “hidden” or not directly observable**, but you can still systematically estimate it with the E-step, then update your parameters with the M-step. This general scheme is precisely **EM**—and it has become an essential technique in unsupervised learning, whenever we suspect *latent structures* underlie our observed data.

---


As we’ve taken a thorough dive into the nuts and bolts of Expectation-Maximization—walking through Gaussian Mixture Models, Hidden Markov Models, and the underlying math—we now find ourselves at the closing stretch of this journey. Yet the stories and insights that drive EM continue to fascinate.

We began by facing the frustration of incomplete data, exploring how each E-step “fills in” what’s missing and each M-step optimizes parameters given that new guess. Along the way, we saw how EM underlies soft clustering (GMMs), generative modeling (sampling new data once we’ve fit a mixture or HMM), and more general unsupervised tasks (discovering latent variables wherever labels are absent or partial).

There’s an old anecdote circulating about Leonard E. Baum—whose contributions to **Baum-Welch** propelled HMMs into the spotlight. He was working with colleagues at the Institute for Defense Analyses on cryptographic challenges and pattern detection, obsessing over how to model signals with elusive, ever-shifting “true states.” Legend has it he’d stare at reams of partially encrypted data for hours, muttering that we only had *half* the story—the half we couldn’t see was the real puzzle. Out of this frustration came the recognition that *if* they could somehow guess those hidden states (or at least the probabilities of them), they could “complete” the data, estimate their model, and *then re-evaluate those hidden states more accurately.* Rinse, repeat.

That mindset captures EM’s essence. Sometimes what drives discovery isn’t a neat, clean dataset but the *gaps* in our knowledge—those half-finished puzzles. By systematically filling in that “invisible half,” we find a richer understanding of the processes that shape our observations. Decades later, EM remains at the heart of numerous machine learning methods precisely because *gaps* and *incompleteness* aren’t rare exceptions: they’re the norm. And in many ways, we owe that to the mathematicians who peered into the hidden states of their data and decided *this is a feature, not a deal-breaker*.