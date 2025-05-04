---
layout: post
title: Optimization Strategies in Modern Machine Learning (Non-Convexity, Hyperparameters, and Meta-Learning)
date: 2024-05-22
description: Optimization & Multivariable Calculus 7 - Mathematics for Data Science
tags: ml ai optimization calculus math
categories: machine-learning math data-science-math
thumbnail: assets/img/optmcalc_banner.jpg
math: true
giscus_comments: true
related_posts: true
chart:
  plotly: true
toc:
  beginning: true
---


Almost all introductory machine learning courses start with a simple idea: minimize a loss function. You define an objective, compute gradients, and follow them downhill. If your function is convex, you're in luck—any local minimum is also a global one, and optimization is largely a matter of computation.

But this picture breaks down quickly in deep learning.

Modern models—from deep neural networks to GANs and neural ODEs—are **highly non-convex**. Their loss landscapes are full of **saddle points**, **flat plateaus**, **sharp cliffs**, and **chaotic basins**. Optimization becomes unstable, unpredictable, and sensitive to initialization and learning rate schedules. Standard gradient descent may stall, oscillate, or diverge entirely.

To make things more complex, training a deep model isn’t a single optimization task. It’s often a **hierarchy of interdependent optimizations**:

* Learning weights via stochastic gradient descent (SGD)
* Tuning hyperparameters like learning rate and batch size
* Solving inner optimization loops in meta-learning
* Navigating min-max games in adversarial training
* Differentiating through entire optimization processes (e.g., in implicit layers or neural ODEs)

This blog explores how modern optimization techniques address these challenges. We’ll examine:

* Why non-convexity makes deep learning optimization hard
* How hyperparameter search methods—grid, random, Bayesian—are used in practice
* Where **proximal gradient methods** and **manifold optimization** appear in ML pipelines
* How **meta-learning algorithms** (like MAML and Reptile) work as nested optimization schemes
* Why optimization in **GANs** introduces new convergence issues

We’ll back each section with mathematical insights, numerical examples, and Python code—so you’ll not only understand the theory but also know how to apply it in real-world machine learning.

---

## Non-Convex Optimization

In convex optimization, things are simple: the function curves upward, and any local minimum is a global one. But most machine learning models—especially deep neural networks—lead to **non-convex loss functions**.

<div style="white-space: nowrap;">
$$
\text{A function } f: \mathbb{R}^n \to \mathbb{R} \text{ is convex if:}
$$
</div>

<div style="white-space: nowrap;">
$$
f(\lambda x + (1 - \lambda) y) \leq \lambda f(x) + (1 - \lambda) f(y), \quad \text{for all } x, y \in \mathbb{R}^n \text{ and } \lambda \in [0, 1]
$$
</div>

Geometrically, this means the line segment connecting any two points on the graph of $$f$$ lies above the function itself.

In contrast, a function is **non-convex** if it violates this inequality in any region. Non-convex functions can have:

* **Multiple local minima**
* **Saddle points**
* **Flat regions (plateaus)**
* **Sharp valleys and ridges**

In deep learning, the loss surface <span style="white-space: nowrap;">$$\mathcal{L}(\theta)$$</span>—where <span style="white-space: nowrap;">$$\theta$$</span> are model parameters—is almost always non-convex due to layer composition, activation non-linearities, and parameter interactions.

---

### Why It Matters in ML

Let’s take a basic deep network as an example:

<div style="white-space: nowrap;">
$$
\hat{y} = f(x; \theta) = W_3 \sigma(W_2 \sigma(W_1 x + b_1) + b_2) + b_3
$$
</div>

Even with ReLU activations <span style="white-space: nowrap;">$$\sigma(z) = \max(0, z)$$</span>, the composition of affine transforms and non-linearities produces a loss surface that is **piecewise smooth and highly non-convex**.

This makes optimization harder:

* We can’t guarantee convergence to a global minimum  
* SGD may get stuck or behave unpredictably  
* Initialization and learning rate have a major impact  

---

### Example

Let’s visualize a simple non-convex function often used in optimization studies:

<div style="white-space: nowrap;">
$$
f(x, y) = x^4 - x^2 + y^2
$$
</div>

<iframe src="/assets/plotly/non_convex_surface.html" width="100%" height="500" frameborder="0"></iframe>

While it appears bowl-like in some regions, it contains both flat regions and sharp dips—making optimization more complex than convex functions. You can observe the valley along the <span style="white-space: nowrap;">$$y$$</span>-axis and the local well-structure along the <span style="white-space: nowrap;">$$x$$</span>-direction. The mixed curvature is what introduces instability.

---

### Saddle Points: Where Gradient Descent Gets Confused

A **saddle point** is a stationary point (i.e., gradient is zero) that is neither a local minimum nor maximum.

Mathematically, for a point <span style="white-space: nowrap;">$$x^*$$</span>:

* <span style="white-space: nowrap;">$$\nabla f(x^*) = 0$$</span>  
* <span style="white-space: nowrap;">$$\text{The Hessian } \nabla^2 f(x^*) \text{ has both positive and negative eigenvalues}$$</span>

This means the surface curves up in some directions and down in others. Gradient descent may slow down near these points or get "stuck" due to tiny gradient magnitudes.

---

#### A Simple Saddle Function

Consider:

<div style="white-space: nowrap;">
$$
f(x, y) = x^2 - y^2
$$
</div>

This has a saddle at:

<div style="white-space: nowrap;">
$$
(0, 0)
$$
</div>

* Gradient:

  <div style="white-space: nowrap;">
  $$
  \nabla f = [2x, -2y] \Rightarrow \nabla f(0, 0) = [0, 0]
  $$
  </div>

* Hessian:

  $$
  H = \begin{bmatrix}
  2 & 0 \\
  0 & -2
  \end{bmatrix}
  $$

  which has eigenvalues:

  <div style="white-space: nowrap;">
  $$
  +2 \quad \text{and} \quad -2
  $$
  </div>

  indicating a saddle point.

Let’s visualize this.

<iframe src="/assets/plotly/saddle_point_surface.html" width="100%" height="500" frameborder="0"></iframe>

This is a canonical saddle surface. The origin is a critical point: the gradient vanishes, but it's **not** a local minimum or maximum. Instead, it's a point of inflection—downhill in one direction, uphill in another. Optimizers like vanilla gradient descent may stall or oscillate here unless noise (like that in SGD) helps escape.

---

#### Implication in Deep Learning

In high dimensions, saddle points are **much more common** than local minima. In fact, most critical points with zero gradient are likely to be saddle points due to random curvature directions in large parameter spaces (per results from Choromanska et al., 2015).

This explains why:

* Training gets slow in early epochs  
* Gradient norms vanish  
* Escaping saddles requires stochasticity (e.g., using SGD with noise)  

---



### Plateaus and Chaotic Basins

After saddle points, two more landscape features often cause trouble during training: **plateaus** and **chaotic basins**. They emerge frequently in deep networks, particularly when the initialization is poor or the architecture is highly sensitive.

#### Plateaus

A **plateau** is a broad region of the loss surface where the gradient is nearly zero:

<div style="white-space: nowrap;">
$$
\|\nabla f(\boldsymbol{\theta})\| \approx 0
$$
</div>

but this is **not** a minimum — just a flat space. Often, the **Hessian is close to zero** too:

<div style="white-space: nowrap;">
$$
\nabla^2 f(\boldsymbol{\theta}) \approx 0
$$
</div>

This means there is no clear slope to descend, and first-order methods like SGD make **very slow progress**.

Common causes:

* Use of **sigmoid activations**, which saturate to 0 or 1 and kill gradients  
* **Poor weight initialization**, especially in deep feedforward networks  
* Lack of **normalization** like BatchNorm or LayerNorm  

##### Visualization: Plateau Region

<iframe src="/assets/plotly/plateau_loss_surface.html" width="100%" height="500" frameborder="0"></iframe>

This surface is defined by:

<div style="white-space: nowrap;">
$$
f(x, y) = \sigma(x)^2 + \sigma(y)^2, \quad \text{where} \quad \sigma(z) = \frac{1}{1 + e^{-z}}
$$
</div>

This is a realistic example of what happens in deep learning with **sigmoid activations**:

* When <span style="white-space: nowrap;">$$x \ll 0$$</span> or <span style="white-space: nowrap;">$$x \gg 0$$</span>, sigmoid saturates to 0 or 1  
* The derivative <span style="white-space: nowrap;">$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$</span> becomes vanishingly small  
* Gradients essentially die out **across large regions** of parameter space  

These **extended flat regions** are exactly what cause deep networks to *stall* in early epochs, especially without proper initialization or normalization. The optimizer "crawls blindly" across the plateau, unable to make meaningful updates.

---

#### Chaotic Basins: Where Everything Becomes Sensitive

A **chaotic basin** is a highly unstable region of the loss surface. Tiny changes in the model parameters <span style="white-space: nowrap;">$$\boldsymbol{\theta}$$</span> can cause wild swings in the loss:

<div style="white-space: nowrap;">
$$
f(\boldsymbol{\theta} + \epsilon) - f(\boldsymbol{\theta}) \gg \|\epsilon\|
$$
</div>

Such regions often arise:

* Around **sharp minima** or **ridges**  
* In **adversarial training** setups, like GANs  
* When weight scales explode due to **no regularization** or **bad initialization**  

In chaotic basins, **gradients can oscillate or explode**, making convergence **unstable**.

##### Visualization: Chaotic Loss Surface

<iframe src="/assets/plotly/chaotic_loss_surface.html" width="100%" height="500" frameborder="0"></iframe>

The above surface illustrates a **chaotic basin**, defined by the function:

<div style="white-space: nowrap;">
$$
f(x, y) = \sin(3x) \cdot \cos(3y) + 0.3 \cdot \sin(5xy)
$$
</div>

This kind of landscape is **not uncommon in deep learning**, especially in complex networks, GANs, or situations involving adversarial dynamics.

What makes this surface chaotic?

* **High-frequency oscillations**: The sine and cosine terms generate a rugged landscape with **sharp ridges and narrow valleys**.  
* **Unstable curvature**: The Hessian varies rapidly in space, making second-order methods unreliable.  
* **Exploding or zig-zag gradients**: Tiny parameter changes can produce large, non-smooth jumps in the loss.  

Mathematically, the gradient here fluctuates rapidly:

$$
\nabla f(x, y) =
\begin{bmatrix}
3 \cos(3x) \cos(3y) + 1.5 y \cos(5xy) \\
-3 \sin(3x) \sin(3y) + 1.5 x \cos(5xy)
\end{bmatrix}
$$

So even small changes in <span style="white-space: nowrap;">$$x$$</span> or <span style="white-space: nowrap;">$$y$$</span> can cause major swings in <span style="white-space: nowrap;">$$f(x, y)$$</span>, especially due to the <span style="white-space: nowrap;">$$\cos(5xy)$$</span> term.

In practice, such chaotic behavior:

* **Slows convergence** or makes it unstable  
* Requires careful **learning rate decay** and **gradient clipping**  
* Is mitigated using **weight normalization**, **adaptive optimizers**, or architectural changes  

This helps explain why optimizers like **Adam** or **RMSProp** are popular: they adapt to this type of curvature instability better than raw SGD.

---


### Summary Table

Here’s how these components affect optimization in ML models:

<div style="overflow-x: auto;">
<table class="simple-table">
  <thead>
    <tr>
      <th>Phenomenon</th>
      <th>Mathematical Cause</th>
      <th>Effect on Optimization</th>
      <th>Mitigation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Saddle Points</td>
      <td><span style="white-space: nowrap;">$$\nabla f = 0$$</span>, <span style="white-space: nowrap;">Hessian has mixed signs</span></td>
      <td>Slow convergence, stuck updates</td>
      <td>Noise (SGD), Momentum</td>
    </tr>
    <tr>
      <td>Plateaus</td>
      <td><span style="white-space: nowrap;">Low gradient and curvature</span></td>
      <td>Extremely slow learning</td>
      <td>BatchNorm, better init</td>
    </tr>
    <tr>
      <td>Chaotic Basins</td>
      <td><span style="white-space: nowrap;">Sharp curvature, instability</span></td>
      <td>Exploding gradients, oscillation</td>
      <td>Gradient clipping, LR decay</td>
    </tr>
  </tbody>
</table>
</div>


---

## Implicit Differentiation: Differentiating Without Solving

Sometimes in optimization and machine learning, we’re stuck with constraints or nested dependencies where solving for a variable explicitly just isn’t feasible. That’s where implicit differentiation becomes a powerful tool.

Rather than solving for a variable and then differentiating, we differentiate first, while treating certain variables as implicitly dependent on others.

---

### The Classic Idea

Suppose we have an equation:

$$
x^2 + y^2 = 25
$$

This defines a $$\text{circle}$$, and we can’t solve for $$y$$ explicitly in all regions without splitting into branches. But we might still want to know:

$$
\text{How does } y \text{ change with } x?
$$

That is, we want $$\frac{dy}{dx}$$ — even though $$y$$ is defined implicitly.

Let’s differentiate both sides with respect to $$x$$, treating $$y$$ as a function of $$x$$:

$$
\frac{d}{dx}(x^2 + y^2) = \frac{d}{dx}(25)
$$

Using the chain rule:

$$
2x + 2y \frac{dy}{dx} = 0
$$

Solving for $$\frac{dy}{dx}$$:

$$
\frac{dy}{dx} = -\frac{x}{y}
$$

This is the implicit derivative of $$y$$ with respect to $$x$$ on the curve defined by $$x^2 + y^2 = 25$$.

---

### General Case

Suppose you have a function defined implicitly by:

$$
F(x, y) = 0
$$

and you want $$\frac{dy}{dx}$$.

Differentiate both sides:

$$
\frac{dF}{dx} = \frac{\partial F}{\partial x} + \frac{\partial F}{\partial y} \cdot \frac{dy}{dx} = 0
$$

Solve for $$\frac{dy}{dx}$$:

$$
\frac{dy}{dx} = -\frac{\frac{\partial F}{\partial x}}{\frac{\partial F}{\partial y}}
$$

This result is elegant and powerful. It tells you how $$y$$ changes with $$x$$ even when the relationship between them is not explicitly given.

---

### Why It Matters in Machine Learning

Implicit differentiation is the backbone of several advanced applications in ML and DL:

* Hyperparameter optimization through bilevel problems
* Learning dynamics in Neural ODEs
* Meta-learning (e.g., MAML) where gradients flow through inner optimization loops
* Differentiating through constraints in optimization layers (e.g., OptNet)

---

### Bilevel Optimization

Suppose you have a nested optimization problem:

$$
\begin{aligned}
&\min_{\lambda} \quad \mathcal{L}_{\text{outer}}(\theta^*(\lambda), \lambda) \\
&\text{where } \theta^*(\lambda) = \arg\min_{\theta} \ \mathcal{L}_{\text{inner}}(\theta, \lambda)
\end{aligned}
$$

You want to optimize a hyperparameter $$\lambda$$, but $$\theta^*(\lambda)$$ depends implicitly on $$\lambda$$ via the solution to the inner problem.

To get $$\frac{d\mathcal{L}_{\text{outer}}}{d\lambda}$$, you need to compute:

$$
\frac{d\mathcal{L}_{\text{outer}}}{d\lambda} = \frac{\partial \mathcal{L}_{\text{outer}}}{\partial \lambda} + \frac{\partial \mathcal{L}_{\text{outer}}}{\partial \theta^*} \cdot \frac{d\theta^*}{d\lambda}
$$

But since $$\theta^*$$ comes from an argmin condition (i.e., a solution to $$\nabla_\theta \mathcal{L}_{\text{inner}}(\theta^*, \lambda) = 0$$), we use implicit differentiation on this gradient condition:

$$
\frac{d}{d\lambda} \nabla_\theta \mathcal{L}_{\text{inner}}(\theta^*, \lambda) = 0
$$

Which gives:

$$
\frac{d\theta^*}{d\lambda} = - \left( \frac{\partial^2 \mathcal{L}_{\text{inner}}}{\partial \theta^2} \right)^{-1} \cdot \frac{\partial^2 \mathcal{L}_{\text{inner}}}{\partial \theta \partial \lambda}
$$

Plug this into the chain rule and you have a clean way to differentiate through the solution of an optimization problem.

---

### Another Example

Let’s take a concrete example:

$$
F(x, y) = x^3 + y^3 - 3xy = 0
$$

Differentiate both sides:

$$
\frac{dF}{dx} = 3x^2 - 3y + 3y^2 \cdot \frac{dy}{dx} - 3x \cdot \frac{dy}{dx} = 0
$$

Group terms:

$$
(3y^2 - 3x)\frac{dy}{dx} = 3y - 3x^2
$$

Solve for $$\frac{dy}{dx}$$:

$$
\frac{dy}{dx} = \frac{3y - 3x^2}{3y^2 - 3x}
$$

---


## Tuning the Machine: Hyperparameter Optimization Strategies

In machine learning and deep learning, we often focus heavily on optimizing the *model parameters* (weights and biases) using techniques like gradient descent. However, there's another layer of parameters that dictates *how* the learning process itself happens: **hyperparameters**. These aren't learned from data during training; instead, they are set *before* training begins. Examples include the learning rate, the number of layers in a neural network, the regularization strength ($$L_1$$ or $$L_2$$ penalty), the type of optimizer (Adam, SGD), or the number of trees in a random forest.

Finding the optimal set of hyperparameters is crucial for achieving peak model performance. Poor hyperparameter choices can lead to slow convergence, suboptimal models, or even failure to converge at all. The process of finding the best set of hyperparameters is known as **Hyperparameter Optimization (HPO)**.

Mathematically, if our model is parameterized by $$\theta$$ and trained on data $$D$$, and its performance is evaluated by a loss function $$\mathcal{L}_{val}$$ on a validation set, the learning process finds $$\theta^* = \arg \min_{\theta} \mathcal{L}_{train}(\theta \mid D)$$. However, the training process itself, and thus the resulting $$\theta^*$$ and its validation performance, depend on a set of hyperparameters $$\lambda$$. HPO aims to solve:

$$
\lambda^* = \arg \min_{\lambda \in \Lambda} \mathcal{L}_{val}\bigl(\theta^*(\lambda) \mid D_{val}\bigr)
$$

where $$\Lambda$$ is the search space of possible hyperparameter configurations, and $$\theta^*(\lambda)$$ represents the model parameters obtained after training with hyperparameters $$\lambda$$. Evaluating $$\mathcal{L}_{val}\bigl(\theta^*(\lambda)\bigr)$$ typically involves training the model fully for each candidate $$\lambda$$, making HPO potentially very computationally expensive. This is often treated as a **black‑box optimization problem**, as we usually don't have access to the gradients of the validation loss with respect to the hyperparameters.

Let's explore three common strategies for tackling this challenge.

### 1. Grid Search: The Exhaustive Approach

Grid Search is perhaps the most intuitive HPO method. You define a discrete grid of possible values for each hyperparameter you want to tune, and the algorithm evaluates *every single combination* of these values.

**Concept:**  
Imagine you want to tune two hyperparameters: learning rate (`lr`) and regularization strength (`alpha`). You might define the following search space:  
* `lr`: [0.1, 0.01, 0.001]  
* `alpha`: [0.0, 0.1, 1.0]

Grid Search will train and evaluate the model for all $$3 \times 3 = 9$$ combinations: (0.1, 0.0), (0.1, 0.1), (0.1, 1.0), (0.01, 0.0), ..., (0.001, 1.0). The combination yielding the best validation performance is selected.

**Mathematical Background:**  


Imagine you have a couple of knobs (your hyperparameters, let's call a specific set of knob settings $$\lambda$$) that control your model‑building machine. You want to find the setting $$\lambda$$ that makes the machine produce the model with the best score on your validation data (let's call this score $$\mathcal{L}_{\text{val}}(\lambda)$$ – lower is usually better).

Grid Search is like deciding on a few specific positions for each knob.  
* For Knob 1 (e.g., `learning_rate`), you choose positions: [0.1, 0.01, 0.001]. Let’s call this set $$\Lambda_{1}$$.  
* For Knob 2 (e.g., `alpha` for regularization), you choose positions: [0.0, 0.1, 1.0]. Let’s call this set $$\Lambda_{2}$$.

The total “search space” $$\Lambda$$ is every possible combination you can make by picking one position from $$\Lambda_{1}$$ and one from $$\Lambda_{2}$$. This forms a grid:  
* (0.1, 0.0), (0.1, 0.1), (0.1, 1.0)  
* (0.01, 0.0), (0.01, 0.1), (0.01, 1.0)  
* (0.001, 0.0), (0.001, 0.1), (0.001, 1.0)

Mathematically, this grid $$\Lambda$$ is the *Cartesian product* of the individual sets:  
$$
\Lambda \;=\; \Lambda_{1} \;\times\; \Lambda_{2}.
$$  
If you had more knobs (hyperparameters), it would be  
$$
\Lambda \;=\; \Lambda_{1} \;\times\; \Lambda_{2} \;\times\; \dots \;\times\; \Lambda_{d}.
$$

Grid Search then *systematically tries every single setting* $$\lambda$$ in this grid $$\Lambda$$. For each one, it builds the model and calculates the validation score $$\mathcal{L}_{\text{val}}(\lambda)$$.

The goal is to find the setting $$\lambda^{*}$$ that gives the best score:  
$$
\lambda^{*} = \arg\min_{\lambda \in \Lambda}\; \mathcal{L}_{\text{val}}(\lambda).
$$  
This simply means: “Find the hyperparameter combination $$\lambda$$ from our grid $$\Lambda$$ that results in the minimum validation loss (or maximum accuracy, depending on how you score it).” Grid Search guarantees finding the best one *on the grid* because it checks every combination.

**When, How, and Where to Use:**  
* **When:** Best suited when you have only a few hyperparameters (typically 2‑4) and each has a small number of discrete values to test.  
* **How:** Define the hyperparameter names and the lists of values you want to try for each. Use libraries like Scikit‑learn’s `GridSearchCV`.  
* **Where:** Useful for initial explorations or when you have a strong prior belief about a small range of good values. Also good when reproducibility and exploring *every* specified combination is paramount.

**Pros:**  
* Simple to understand and implement.  
* Guaranteed to find the best combination *within the specified grid*.

**Cons:**  
* Suffers heavily from the **curse of dimensionality**. The number of combinations grows exponentially with the number of hyperparameters.  
* Inefficient, as it spends equal time on all combinations, even in regions of the space that quickly prove suboptimal.  
* May miss optimal values that lie *between* grid points, especially for continuous hyperparameters.

**Code Walkthrough: Implementing Grid Search with GridSearchCV**

Let's see how Grid Search can be implemented in practice using Python's popular Scikit-learn library. We'll use a synthetic dataset and a Support Vector Classifier (SVC) model for illustration. First, we import the necessary tools, generate some data, and split it. Then, we define our model and specify the `param_grid` – this dictionary holds the hyperparameters we want to tune (`C`, `gamma`, `kernel`) and the specific values we want to try for each. `GridSearchCV` will systematically test every single combination.

```python
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the model
model = SVC(probability=True)  # Example model

# Define the grid of hyperparameters
param_grid = {
    'C': [0.1, 1, 10, 100],       # Regularization parameter
    'gamma': [1, 0.1, 0.01, 0.001],  # Kernel coefficient
    'kernel': ['rbf', 'linear']
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

grid_search.fit(X_train, y_train)

print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
```

In this example, `GridSearchCV` takes our SVC `estimator`, the `param_grid`, and configuration details like the cross-validation strategy (`cv=5`). Setting `n_jobs=-1` is highly recommended to speed up the search by using all available CPU cores. The `fit` method runs the exhaustive search. Afterwards, we can access the best hyperparameter combination found via `grid_search.best_params_` and the corresponding mean cross-validated score via `grid_search.best_score_`. You can see how quickly the number of combinations (and thus computation time) grows as you add more parameters or values to the grid.

### 2. Random Search: A More Efficient Exploration

Random Search works differently. Instead of defining a discrete grid, you define a *distribution* for each hyperparameter (e.g., a uniform distribution between 0.001 and 0.1 for learning rate). The algorithm then samples a fixed number of combinations randomly from these distributions.

**Concept:**  
Instead of trying all points on a grid, Random Search picks random points within the defined search space bounds. Research (Bergstra & Bengio, 2012) showed that Random Search is often more efficient than Grid Search, especially when some hyperparameters are much more important than others. It's more likely to find good values for the important parameters because it doesn't waste evaluations on refining unimportant ones across many grid points.

**Mathematical Background:**  

Grid Search can be slow because it checks *every* point on the grid, even if early results suggest a whole area is bad. What if we just tried a bunch of *random* combinations instead? That's Random Search.

Instead of defining specific values for each hyperparameter knob, you define a *range* or a *way to pick randomly*.
* For Knob 1 (`learning_rate`), maybe pick any value *uniformly* between 0.0001 and 0.1. We write this like $$\lambda_1 \sim \text{Uniform}(0.0001, 0.1)$$. The "~" means "is randomly sampled from".
* For Knob 2 (`number_of_layers`), maybe pick a random *integer* between 2 and 10. We could write $$\lambda_2 \sim \text{Integer}(2, 10)$$.
* For Knob 3 (`activation_function`), maybe pick randomly from a *list*: ['relu', 'tanh', 'sigmoid']. We could write $$\lambda_3 \in \{\text{'relu', 'tanh', 'sigmoid'}\}$$.

The search space $$\Lambda$$ is now defined by these ranges and distributions. Trying *all* combinations is often impossible (especially with continuous ranges). So, Random Search just says: "Pick $$N$$ random combinations $$\{\lambda^{(1)}, \lambda^{(2)}, \dots, \lambda^{(N)}\}$$ according to the rules we set for each knob." You decide how many combinations ($$N$$) you have the budget to try.

For each randomly chosen combination $$\lambda^{(i)}$$, you build the model and get the validation score $$\mathcal{L}_{\text{val}}(\lambda^{(i)})$$.

The goal is still to find the best hyperparameters, but since we only tested $$N$$ random points, we find the best one *among those we tested*:

$$
\lambda^* \approx \arg \min_{i \in \{1,\dots,N\}} \mathcal{L}_{\text{val}}(\lambda^{(i)})
$$

The "$$\approx$$" (approximately equals) sign is there because we didn't test *all* possibilities, just $$N$$ random ones. However, studies show that if $$N$$ is reasonably large, Random Search is surprisingly good at finding very good, if not the absolute best, hyperparameter settings—often much faster than Grid Search in higher dimensions.

**When, How, and Where to Use:**  
* **When:** When dealing with a larger number of hyperparameters (e.g., > 3‑4), including continuous ones, or when the computational budget is limited (you can fix the number of trials, $$N$$).  
* **How:** Define hyperparameter names and their corresponding distributions (e.g., uniform, log‑uniform, discrete choices). Use libraries like Scikit‑learn’s `RandomizedSearchCV` or tools like `hyperopt`.  
* **Where:** Often a better default choice than Grid Search for deep learning models or complex pipelines due to higher efficiency in high‑dimensional spaces.

**Pros:**  
* More efficient than Grid Search, especially in higher dimensions.  
* Less likely to miss good values “between the grid points.”  
* Easier to control the computational budget by setting the number of iterations ($$N$$).

**Cons:**  
* No guarantee of finding the optimal combination (it's probabilistic).  
* Doesn't learn from past evaluations; each trial is independent.

**Code Walkthrough: Implementing Random Search with RandomizedSearchCV**

Now, let's look at Random Search, again using Scikit-learn. The setup is similar, but instead of providing fixed lists of values, we provide *distributions* from which to sample using `scipy.stats`. We also specify `n_iter`, the total number of random combinations to try, which gives us direct control over the computational budget.

```python
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy.stats import expon, randint

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the model
model = SVC(probability=True)

# Define the distributions for hyperparameters
param_dist = {
    'C': expon(scale=100),                # Exponential distribution
    'gamma': expon(scale=0.1),
    'kernel': ['rbf', 'poly', 'sigmoid'],  # Discrete choices
    'degree': randint(2, 5)               # Integer between 2 and 4
}

# Set up RandomizedSearchCV
n_iter_search = 20
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5, n_jobs=-1,
                                   verbose=2, random_state=42, scoring='accuracy')

random_search.fit(X_train, y_train)

print(f"Best parameters found: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")
```

Here, `RandomizedSearchCV` uses the `param_distributions` dictionary. Notice how we can mix discrete choices (like `kernel`) with continuous distributions (`expon` for `C` and `gamma`) and integer distributions (`randint` for `degree`). The key parameter `n_iter=20` dictates that only 20 combinations will be sampled and evaluated, regardless of how large the theoretical space is. This makes Random Search much more scalable than Grid Search for higher dimensions or continuous parameters. The results (`best_params_`, `best_score_`) represent the best found within those 20 random trials.

### 3. Bayesian Optimization: Intelligent Search

Bayesian Optimization takes a more sophisticated approach. It builds a probabilistic model (a **surrogate model**) of the relationship between hyperparameters and the validation loss. It then uses this model to intelligently decide which hyperparameter combination to try next, balancing *exploration* (trying uncertain areas) and *exploitation* (trying areas likely to yield high performance based on current knowledge).

**Concept:**  
1. **Surrogate Model:** A probabilistic model, often a Gaussian Process (GP), is used to approximate the expensive black‑box function $$\mathcal{L}_{val}(\lambda)$$. The GP provides not only a prediction of the loss for untested $$\lambda$$ but also an estimate of the uncertainty around that prediction.  
2. **Acquisition Function:** An acquisition function (e.g., Expected Improvement (EI), Probability of Improvement (PI), Upper Confidence Bound (UCB)) uses the surrogate model's predictions and uncertainties to quantify the "value" of evaluating a particular point $$\lambda$$. EI, for instance, calculates the expected improvement over the best loss found so far.  
3. **Iteration:** The algorithm selects the next hyperparameter combination $$\lambda_{\text{next}}$$ by maximizing the acquisition function. It then evaluates the true $$\mathcal{L}_{val}(\lambda_{\text{next}})$$ (by training the model), updates the surrogate model with this new data point, and repeats the process.

**Mathematical Background:**  

Imagine finding the best hyperparameters $$\lambda$$ is like trying to find the lowest valley in a landscape (where height represents the validation loss $$\mathcal{L}_{\text{val}}(\lambda)$$). The catch is, the landscape is covered in fog! You can only know the height at specific points where you land a probe (i.e., train and evaluate a model), which is expensive and time-consuming.

Grid Search tries landing probes on a fixed grid. Random Search lands probes randomly. Bayesian Optimization tries to be smarter, like a hiker using previous measurements to guess where the valley might be deepest.

Here's the process:


1.  **Build a Belief (Surrogate Model):**  
    After landing a few initial probes (evaluating some initial hyperparameter sets $$\lambda$$), Bayesian Optimization builds a "probabilistic map" or "belief" about what the entire landscape *might* look like. This map is called a **surrogate model**. It's often a **Gaussian Process ($$\mathcal{GP}$$)**.
    
    * Think of the $$\mathcal{GP}$$ as drawing a smooth, flexible surface that fits the points you've measured.
    * Crucially, it also estimates its *uncertainty* – it knows where it's just guessing versus where it has solid data. Mathematically, for any point $$\lambda$$, it gives a predicted average height $$\mu(\lambda)$$ and an uncertainty measure (related to the variance, derived from a kernel function $$k$$). We write this as:

      $$
      \mathcal{L}_{\text{val}}(\lambda) \sim \mathcal{GP}(\mu(\lambda), k(\lambda, \lambda'))
      $$



2.  **Decide Where to Probe Next (Acquisition Function):**  
    Based on this uncertain map (the surrogate model), where's the *most useful* place to land the next expensive probe? This is decided by an **acquisition function**. This function balances two desires:

    * **Exploitation:** Probing near the lowest points found so far, hoping to find even lower points nearby. (Drilling near a known oil well).
    * **Exploration:** Probing in areas where the map is very uncertain. These areas might contain an unexpectedly deep valley. (Drilling in a completely new region with promising but uncertain geological signs).

    A common acquisition function is **Expected Improvement (EI)**. It calculates, for each possible next point $$\lambda$$:

    > "How much lower do we *expect* the valley floor to be here compared to the lowest point ($$y_{\text{best}}$$) we've found so far, considering both our map's prediction $$\mu(\lambda)$$ and its uncertainty?"

    We write this as:

    $$
    a_{EI}(\lambda) = \mathbb{E}[\max(0, y_{\text{best}} - \mathcal{L}_{\text{val}}(\lambda)) \mid \text{Data}]
    $$



3.  **Optimize the Acquisition:**  
    Find the point $$\lambda_{\text{next}}$$ that *maximizes* the acquisition function (e.g., gives the highest Expected Improvement). This step is cheap because it only involves calculations on our surrogate map, not training the actual model.

    $$
    \lambda_{\text{next}} = \arg \max_{\lambda \in \Lambda} a(\lambda \mid \text{Data})
    $$



4.  **Land the Probe and Update:**  
    Now, actually run the expensive model training and evaluation using the chosen hyperparameters $$\lambda_{\text{next}}$$ to get the true score $$y_{\text{next}} = \mathcal{L}_{\text{val}}(\lambda_{\text{next}})$$. Add this new information $$(\lambda_{\text{next}}, y_{\text{next}})$$ to your set of known points. Update the surrogate map ($$\mathcal{GP}$$) with this new data point – the map becomes slightly more accurate. Repeat from step 2 until your budget (e.g., number of probes/evaluations) runs out.

---

In essence, Bayesian Optimization uses probability and past results to make educated guesses about where to search next, aiming to find the lowest point (best hyperparameters) using fewer expensive evaluations than random guessing or exhaustive grid checks.

**When, How, and Where to Use:**  
* **When:** When function evaluations are expensive (minutes, hours, days).  
* **How:** Use libraries like `scikit‑optimize`, `Optuna`, `Hyperopt`, or Facebook’s `Ax`/`BoTorch`. Define the search space and an objective that returns validation loss (or accuracy).  
* **Where:** The method of choice for deep learning HPO, scientific simulations, and any scenario where training time dominates.

**Pros:**  
* Generally more sample‑efficient than Grid or Random Search.  
* Effectively balances exploration and exploitation.  
* Adapts based on previous results.

**Cons:**  
* More complex to set up.  
* Surrogate‑model overhead may matter if evaluations are very cheap.  
* Sensitivity to the choice of surrogate and acquisition function.  
* Can sometimes converge to local optima in hyperparameter space.

**Code Walkthrough: Implementing Bayesian Optimization with gp_minimize**

Finally, let's illustrate Bayesian Optimization using the `scikit-optimize` library (`skopt`). This approach requires a bit more setup. We need to define the search space using special objects (`Real`, `Integer`, `Categorical`) from `skopt.space`. More importantly, we must define an `objective` function. This function takes a set of hyperparameters, trains the model with them, evaluates it (often using cross-validation), and returns the score we want to *minimize* (e.g., error rate or `1 - accuracy`). `gp_minimize` then uses this objective function, iteratively choosing the next hyperparameters to try based on its internal surrogate model (a Gaussian Process by default).

```python
# Needs: pip install scikit-optimize
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# Data
X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)

# Space
search_space = [
    Integer(10, 500, name='n_estimators'),
    Integer(1, 10, name='max_depth'),
    Real(1e-6, 1e-1, "log-uniform", name='min_samples_split'),
    Real(1e-6, 1e-1, "log-uniform", name='min_samples_leaf'),
    Categorical(['sqrt', 'log2', None], name='max_features')
]

@use_named_args(search_space)
def objective(**params):
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    score = np.mean(cross_val_score(model, X, y, cv=3, scoring='accuracy'))
    print(f"Params: {params} -> Score: {score:.4f}")
    return -score  # gp_minimize minimizes

result = gp_minimize(objective, search_space, n_calls=30,
                     n_initial_points=10, acq_func='EI', random_state=42,
                     verbose=True)

print(f"\nBest parameters: {result.x}")
best_params = {space.name: val for space, val in zip(search_space, result.x)}
print(f"Best parameters dict: {best_params}")
print(f"Best accuracy achieved: {-result.fun:.4f}")
```

In this `scikit-optimize` example, we define the `search_space` with specific types and ranges (note the use of `Real` with a `"log-uniform"` scale, often beneficial for parameters like learning rates or regularization strengths). The `objective` function encapsulates the model training and evaluation logic, returning the negative accuracy because `gp_minimize` aims to minimize its input. The `gp_minimize` function orchestrates the process: it calls `objective` `n_calls` times in total. It starts with `n_initial_points` random evaluations (similar to Random Search) to bootstrap the process, then iteratively uses the Gaussian Process model and the specified acquisition function (`'EI'`) to choose the subsequent, most promising points to evaluate. The final `result` object contains the best parameters found (`result.x`) and the best score achieved (`result.fun`). This approach intelligently navigates the search space, often reaching better results with fewer expensive model training calls compared to random or grid search.

### Summary: Choosing Your Weapon

* **Grid Search:** Simple and exhaustive; viable only for a handful of discrete hyperparameters.  
* **Random Search:** Budget‑friendly, scales better, and often preferred over a grid in higher‑dimensional spaces.  
* **Bayesian Optimization:** Most sample‑efficient for expensive evaluations; sophisticated but powerful.

Hyperparameter optimization is an essential part of the modern ML/DL workflow. While these three methods form the bedrock, the field is active, with connections to meta‑learning and Neural Architecture Search (NAS) continually pushing the boundaries of automated model building. Choosing the right HPO technique depends on your specific problem, computational budget, and the complexity of the hyperparameter space.

---


## Proximal Gradient Methods: When Smooth Meets Nonsmooth

In many ML problems, the loss function isn’t just a nice smooth curve. Often, we deal with *composite objectives*:

$$
\min_{\boldsymbol{x}} \ \underbrace{f(\boldsymbol{x})}_{\text{smooth}} \;+\; \underbrace{g(\boldsymbol{x})}_{\text{possibly non‑smooth}}
$$

For example:

* In **Lasso regression**, $$f(\boldsymbol{x}) = \frac{1}{2}\,\|\boldsymbol{y} - X\boldsymbol{x}\|^{2}$$ is smooth,  
  but $$g(\boldsymbol{x}) = \lambda\,\|\boldsymbol{x}\|_{1}$$ is not.
* In **image denoising**, $$f$$ might be a data‑fidelity term, and $$g$$ could encode total variation or sparsity.

Standard gradient descent doesn’t work well when $$g$$ is non‑differentiable (e.g., absolute value, indicator functions, etc.).

---

### Idea Behind Proximal Gradient

At each step, we want to move in the direction that decreases the smooth part $$f$$ **while also respecting the structure or constraint encoded in $$g$$**.

$$
\boldsymbol{x}^{(k+1)}
  = \operatorname{prox}_{\eta\,g}\!\Bigl(
      \boldsymbol{x}^{(k)} \;-\; \eta\,\nabla f\bigl(\boldsymbol{x}^{(k)}\bigr)
    \Bigr)
$$

Here:

* $$\eta$$ is the step size  
* $$\nabla f(\boldsymbol{x})$$ is the gradient of the smooth part  
* $$\operatorname{prox}_{\eta g}(\cdot)$$ is the **proximal operator** of $$g$$, defined as  

  $$
  \operatorname{prox}_{\eta g}(\boldsymbol{v})
    = \arg\min_{\boldsymbol{x}}
        \Bigl\{\,g(\boldsymbol{x}) + \tfrac{1}{2\eta}\,\|\boldsymbol{x}-\boldsymbol{v}\|^{2}\Bigr\}
  $$

> “Take a gradient step for the smooth part, then *project* or *shrink* that point toward what $$g$$ wants.”

---

### Example: Lasso Regression

Objective:

$$
\min_{\boldsymbol{w}} \;
  \tfrac{1}{2}\,\|\boldsymbol{y} - X\boldsymbol{w}\|^{2}
  \;+\;
  \lambda\,\|\boldsymbol{w}\|_{1}
$$

Decompose:

* $$f(\boldsymbol{w}) = \tfrac{1}{2}\,\|\boldsymbol{y} - X\boldsymbol{w}\|^{2}$$ — smooth, differentiable  
* $$g(\boldsymbol{w}) = \lambda\,\|\boldsymbol{w}\|_{1}$$ — non‑smooth but proximable  

**Gradient Step**

$$
\boldsymbol{v}^{(k)}
  = \boldsymbol{w}^{(k)}
    \;-\;
    \eta\,X^{\!\top}\!\bigl(X\boldsymbol{w}^{(k)} - \boldsymbol{y}\bigr)
$$

**Proximal Step (Soft‑thresholding)**

$$
\boldsymbol{w}^{(k+1)}
  = \operatorname{prox}_{\eta\lambda\,\|\cdot\|_{1}}\!
    \bigl(\boldsymbol{v}^{(k)}\bigr)
  = \operatorname{sign}\!\bigl(\boldsymbol{v}^{(k)}\bigr)
    \,\cdot\,
    \max\!\bigl(|\boldsymbol{v}^{(k)}| - \eta\lambda,\;0\bigr)
$$

This shrinks small entries to zero — perfect for inducing **sparsity**.

---

### When Are These Useful?

Proximal methods shine in problems where:

* $$g$$ imposes **sparsity** or **constraints** (e.g., $$\ell_{1}$$, indicator functions)  
* You want to **split smooth and nonsmooth parts** for separate treatment  
* You deal with **high‑dimensional convex or structured optimization** problems  
* You need **coordinate‑wise operations**, e.g., shrinkage or projections  

---

### Connection to ISTA and FISTA

* **ISTA** (Iterative Shrinkage‑Thresholding Algorithm) is the basic proximal‑gradient method applied to Lasso.  
* **FISTA** accelerates convergence, achieving $$\mathcal{O}\!\bigl(1/k^{2}\bigr)$$ versus $$\mathcal{O}\!\bigl(1/k\bigr)$$ for ISTA.

---

### Summary: Why We Use Proximal Methods

<div style="overflow-x: auto;">
  <table class="simple-table">
    <thead>
      <tr>
        <th>Scenario</th>
        <th>Why&nbsp;Proximal?</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Lasso&nbsp;or&nbsp;Elastic&nbsp;Net</td>
        <td>Handle $$\ell_{1}$$ regularization via shrinkage</td>
      </tr>
      <tr>
        <td>TV&nbsp;Denoising</td>
        <td>Smooth + total‑variation penalty</td>
      </tr>
      <tr>
        <td>Convex&nbsp;constraints</td>
        <td>Use projection via proximal operators</td>
      </tr>
      <tr>
        <td>Sparse&nbsp;Coding</td>
        <td>Enforce sparsity while fitting dictionary</td>
      </tr>
    </tbody>
  </table>
</div>

---

## Manifold Optimization: Learning on Curved Spaces

Most optimization methods in machine learning assume that parameters live in **Euclidean space** <span style="white-space: nowrap;">$$\mathbb{R}^n$$</span>. But what if our model's parameters must lie on **a curved surface**, such as:

* a **sphere** (unit norm constraint),
* an **orthogonal group** (e.g., orthonormal columns),
* or a **low-rank matrix manifold**?

In such cases, plain gradient descent doesn’t work — because it takes steps that leave the feasible space. This is where **manifold optimization** comes in: it generalizes first-order and second-order methods to **Riemannian manifolds**.

---

### What Is a Manifold?

A **manifold** is a space that **locally looks like Euclidean space**, even if it's curved globally.

Formally, an <span style="white-space: nowrap;">$$n$$</span>-dimensional manifold <span style="white-space: nowrap;">$$\mathcal{M}$$</span> is a topological space where every point has a neighborhood homeomorphic to <span style="white-space: nowrap;">$$\mathbb{R}^n$$</span>.

#### Common examples in machine learning:

* **Sphere**:

  $$
  \mathbb{S}^{n-1} = \{ \boldsymbol{x} \in \mathbb{R}^n \mid \|\boldsymbol{x}\|_2 = 1 \}
  $$

* **Stiefel manifold**:

  $$
  \mathcal{V}_{k}(\mathbb{R}^n) = \{ W \in \mathbb{R}^{n \times k} \mid W^\top W = I_k \}
  $$

  (orthonormal <span style="white-space: nowrap;">$$k$$</span>-frames in <span style="white-space: nowrap;">$$\mathbb{R}^n$$</span>)

* **Grassmannian**:  
  The space of <span style="white-space: nowrap;">$$k$$</span>-dimensional subspaces of <span style="white-space: nowrap;">$$\mathbb{R}^n$$</span>.

* **Positive semi-definite cone**:

  $$
  \{ X \in \mathbb{R}^{n \times n} \mid X = X^\top, X \succeq 0 \}
  $$

These structures show up in neural networks, kernel methods, dimensionality reduction, and quantum information.

---

### Why Euclidean Gradient Descent Fails

In unconstrained optimization, we update:

$$
\boldsymbol{x}_{k+1} = \boldsymbol{x}_k - \eta \nabla f(\boldsymbol{x}_k)
$$

But if <span style="white-space: nowrap;">$$\boldsymbol{x}_k \in \mathcal{M}$$</span>, the Euclidean gradient step may **exit the manifold**:

* It doesn’t respect constraints (e.g., orthogonality, normalization).  
* Projection back to the manifold may not be differentiable.  

We need an optimization algorithm that **stays on the manifold at every step**.

---

### Anatomy of Manifold Optimization

To optimize a function <span style="white-space: nowrap;">$$f : \mathcal{M} \to \mathbb{R}$$</span> where <span style="white-space: nowrap;">$$\mathcal{M}$$</span> is a Riemannian manifold, we replace the components of gradient descent with their manifold analogs.

---

#### 1. Tangent Space

At each point <span style="white-space: nowrap;">$$\boldsymbol{x} \in \mathcal{M}$$</span>, the **tangent space** <span style="white-space: nowrap;">$$T_{\boldsymbol{x}} \mathcal{M}$$</span> is a vector space that "touches" the manifold at that point.

For instance, on the sphere:

$$
T_{\boldsymbol{x}} \mathbb{S}^{n-1} = \{ \boldsymbol{v} \in \mathbb{R}^n \mid \boldsymbol{x}^\top \boldsymbol{v} = 0 \}
$$

This means tangent vectors must be **orthogonal** to the current point on the sphere.

---

#### 2. Riemannian Gradient

The **Riemannian gradient** is the projection of the standard Euclidean gradient onto the tangent space:

$$
\nabla_{\mathcal{M}} f(\boldsymbol{x}) = \Pi_{T_{\boldsymbol{x}}\mathcal{M}} \nabla f(\boldsymbol{x})
$$

Example: On the sphere <span style="white-space: nowrap;">$$\mathbb{S}^{n-1}$$</span>,

$$
\Pi_{T_{\boldsymbol{x}} \mathbb{S}^{n-1}} (\boldsymbol{g}) = \boldsymbol{g} - (\boldsymbol{x}^\top \boldsymbol{g}) \boldsymbol{x}
$$

This operation removes any component of <span style="white-space: nowrap;">$$\boldsymbol{g}$$</span> that points off the manifold.

---

#### 3. Retraction

After taking a step in the tangent space, you need to **map the point back** to the manifold. A **retraction** is a smooth map:

$$
\text{Retr}_{\boldsymbol{x}}(\boldsymbol{v}) \approx \boldsymbol{x} + \boldsymbol{v}
$$

such that <span style="white-space: nowrap;">$$\text{Retr}_{\boldsymbol{x}}(\boldsymbol{0}) = \boldsymbol{x}$$</span> and the image stays on <span style="white-space: nowrap;">$$\mathcal{M}$$</span>.

Example: On the sphere, a simple retraction is **normalization**:

$$
\text{Retr}_{\boldsymbol{x}}(\boldsymbol{v}) = \frac{\boldsymbol{x} + \boldsymbol{v}}{\|\boldsymbol{x} + \boldsymbol{v}\|}
$$

---

### Full Algorithm: Riemannian Gradient Descent

At each iteration <span style="white-space: nowrap;">$$k$$</span>:

1. Compute the Euclidean gradient: <span style="white-space: nowrap;">$$\nabla f(\boldsymbol{x}_k)$$</span>  
2. Project to tangent space:

   $$
   \boldsymbol{v}_k = \Pi_{T_{\boldsymbol{x}_k} \mathcal{M}} \nabla f(\boldsymbol{x}_k)
   $$

3. Update along tangent:

   $$
   \boldsymbol{x}_{k+1} = \text{Retr}_{\boldsymbol{x}_k}(-\eta \boldsymbol{v}_k)
   $$

This ensures the iterate stays on the manifold **throughout training**.

---

### Real-World Applications

<div style="overflow-x: auto;">
<table class="simple-table">
<thead>
<tr>
  <th>Use Case</th>
  <th>Manifold</th>
  <th>Constraint</th>
</tr>
</thead>
<tbody>
<tr>
  <td>Orthogonal layers in neural nets</td>
  <td>Stiefel manifold</td>
  <td><span style="white-space: nowrap;">$$W^\top W = I$$</span></td>
</tr>
<tr>
  <td>Low-rank matrix factorization</td>
  <td>Fixed-rank manifold</td>
  <td><span style="white-space: nowrap;">$$\text{rank}(X) = r$$</span></td>
</tr>
<tr>
  <td>Word embeddings in hyperbolic space</td>
  <td>Hyperbolic manifold</td>
  <td>Negative curvature</td>
</tr>
<tr>
  <td>Pose estimation, quantum states</td>
  <td>Unit sphere / <span style="white-space: nowrap;">$$SO(n)$$</span></td>
  <td><span style="white-space: nowrap;">$$\|\boldsymbol{x}\| = 1$$</span> or rotation matrices</td>
</tr>
</tbody>
</table>
</div>

---

### Summary

**Manifold optimization** extends gradient descent to curved spaces by replacing:

* Euclidean gradients → Riemannian gradients  
* Vector updates → Tangent space updates  
* Naive step → Retraction back to manifold  

This is essential for constrained or structured ML models, and plays a foundational role in geometry-aware learning.

---


## Meta-Learning: How to Learn New Tasks, Fast

Imagine training a machine learning model not just to do one thing well—say, classify digits—but to **quickly adapt** to **new tasks** it’s never seen before, like classifying alphabets or different character styles.

That’s what **meta-learning** is about.

Instead of just learning *to perform*, the model learns *how to learn*. It’s like teaching someone the skill of picking up new languages fast, rather than just one specific language.

---

### The Meta-Learning Problem Setup

We assume that tasks come from some distribution:

$$\mathcal{T} \sim p(\mathcal{T})$$

Each task $$\mathcal{T}$$ has its own dataset, typically split into:

* A **training set**: $$\mathcal{D}_{\mathcal{T}}^{\text{train}}$$ — for quick adaptation
* A **validation set**: $$\mathcal{D}_{\mathcal{T}}^{\text{val}}$$ — for measuring performance after adaptation

The goal is to find **initial model parameters** $$\theta$$ that can quickly adapt to any task $$\mathcal{T}$$ using just a few training examples from $$\mathcal{D}_{\mathcal{T}}^{\text{train}}$$.

We’re asking:

> What $$\theta$$ should we learn so that, after one or two gradient steps on a new task, we do well on that task?

---

### Step Inside a Task: Two Loops of Learning

This gives rise to **two loops** in meta-learning:

* An **inner loop**: adaptation within a task, using a few gradient descent steps
* An **outer loop**: meta-optimization across many tasks

Let’s make this more concrete using **MAML**, one of the most influential algorithms in this space.

---

### MAML: Model-Agnostic Meta-Learning

MAML (Finn et al., 2017) learns a single parameter vector $$\theta$$ that can **quickly adapt to new tasks** via gradient descent.

Instead of learning a fixed model, we learn an **initialization** that performs well after fine-tuning on any task.

#### Algorithm Breakdown

1. **Sample a batch of tasks**:

   $$\{\mathcal{T}_1, \mathcal{T}_2, \dots, \mathcal{T}_B\} \sim p(\mathcal{T})$$

2. **For each task** $$\mathcal{T}_i$$:

   **(a) Inner loop:** Perform one or more gradient steps using the task’s training data:

   $$\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\text{train}}(\theta)$$

   This gives us task-specific adapted weights $$\theta'_i$$.

   **(b) Outer loop:** Evaluate how well those adapted weights perform on the task’s validation data:

   $$\mathcal{L}_{\mathcal{T}_i}^{\text{val}}(\theta'_i)$$

3. **Meta-update (outer loop):** Update the original $$\theta$$ by minimizing the validation losses across tasks:

   $$\theta \leftarrow \theta - \beta \nabla_\theta \sum_{i=1}^B \mathcal{L}_{\mathcal{T}_i}^{\text{val}}(\theta'_i)$$

   Here’s the twist: we’re taking gradients **through** the inner update, so this involves **second-order derivatives**.

---



Imagine each task has its own “valley” in the loss landscape. MAML tries to find an initial point $$\theta$$ that’s **close to all of them**—so one or two steps get you to the bottom, no matter which task you're on.

It doesn't make $$\theta$$ perfect for any single task—but it makes it easy to reach optimal weights for many.

---

#### Why Use MAML?

* It’s **model-agnostic**: works with any model you can train with gradient descent.
* Great for **few-shot learning**: excels when data is scarce for each task.
* Theoretically elegant: links meta-learning with optimization.

---

### Reptile: Simpler, Faster Meta-Learning

MAML is powerful—but computationally expensive because of second-order gradients. **Reptile** (Nichol et al., 2018) offers a simpler, **first-order approximation** that still captures the meta-learning idea.

Instead of differentiating through gradients, Reptile just **moves the initialization $$\theta$$ toward the task-specific solution $$\theta'$$**.

---

#### Reptile Algorithm

1. Sample a task $$\mathcal{T}$$

2. Train on the task for $$k$$ steps of gradient descent using $$\mathcal{D}_{\mathcal{T}}^{\text{train}}$$:

   $$\theta' = \text{SGD}_k(\theta, \mathcal{D}_{\mathcal{T}}^{\text{train}})$$

3. Meta-update:

   $$\theta \leftarrow \theta + \epsilon (\theta' - \theta)$$

This is not a traditional gradient update. It’s more like nudging $$\theta$$ in the direction of a good task-specific solution.

---

#### Why Does Reptile Work?

Even though it doesn’t use gradient-of-gradient calculations, it still encourages $$\theta$$ to be a **good starting point** across many tasks.

It’s like averaging the useful parts of the task-specific optimizations without computing second-order terms.

---

### MAML vs Reptile: A Side-by-Side View

<table class="simple-table">
  <thead>
    <tr>
      <th>Aspect</th>
      <th>MAML</th>
      <th>Reptile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Gradient Order</td>
      <td>Second-order (or first-order variant)</td>
      <td>First-order only</td>
    </tr>
    <tr>
      <td>Computation</td>
      <td>Slower due to backprop through gradients</td>
      <td>Faster and simpler</td>
    </tr>
    <tr>
      <td>Update Mechanism</td>
      <td>Gradient of validation loss after adaptation</td>
      <td>Move toward adapted parameters</td>
    </tr>
    <tr>
      <td>Intuition</td>
      <td>Learn to adapt via gradient descent</td>
      <td>Average useful adaptations across tasks</td>
    </tr>
  </tbody>
</table>

---

### Why This Matters in Practice

Meta-learning lets us build systems that:

* **Adapt quickly** to new users, new data, or new environments
* **Use fewer training examples** per task
* Generalize better across tasks

MAML and Reptile both offer frameworks for doing this **using only gradient descent**—the core machinery behind most of deep learning.

They also highlight how powerful optimization can be when you don’t just optimize for performance—but **optimize for adaptability**.

---

## Optimization in GANs: Dancing on a Min-Max Edge

Training a Generative Adversarial Network (GAN) feels more like trying to balance a see-saw than descending a hill. You’re not just minimizing or maximizing one function—you’re **doing both at once**.

At its heart, GAN training is a **min-max optimization problem**, also known as a **two-player game**.

---

### The GAN Objective: A Two-Player Game

GANs involve two neural networks:

* A **generator** $$G(z; \theta_G)$$ that tries to produce realistic data from noise $$z$$
* A **discriminator** $$D(x; \theta_D)$$ that tries to distinguish real data $$x$$ from fake data $$G(z)$$

Their objectives are in direct opposition. The discriminator wants to assign high scores to real samples and low scores to generated ones. The generator wants to fool the discriminator.

Mathematically, the original GAN objective is:

$$
\min_{\theta_G} \ \max_{\theta_D} \ \mathbb{E}_{x \sim p_{\text{data}}} \left[ \log D(x) \right] + \mathbb{E}_{z \sim p(z)} \left[ \log (1 - D(G(z))) \right]
$$

Let’s unpack this:

* The **discriminator** $$D$$ tries to **maximize** the likelihood of correctly labeling:

  * Real data $$x$$ as 1: $$\log D(x)$$
  * Fake data $$G(z)$$ as 0: $$\log(1 - D(G(z)))$$

* The **generator** $$G$$ tries to **minimize** this same loss (i.e., fool the discriminator).

So you get a **min-max game**:

$$
\min_G \max_D \ \mathcal{L}(G, D)
$$

---

### Why It’s Hard to Optimize

This isn’t a standard optimization problem. In regular machine learning, we minimize a loss over parameters:

$$
\min_\theta \ \mathcal{L}(\theta)
$$

In GANs, both $$G$$ and $$D$$ are moving targets. When one improves, it changes the loss landscape for the other. The optimization becomes **non-stationary** and **non-convex** in both directions.

It’s like:

> Trying to hit a moving bullseye while blindfolded, where your actions also move the bullseye.

---

### Saddle Point Formulation

Let’s define the objective as a game:

$$
\min_{x \in \mathcal{X}} \max_{y \in \mathcal{Y}} f(x, y)
$$

The **optimal solution** is a **saddle point**:

$$
f(x^*, y) \leq f(x^*, y^*) \leq f(x, y^*)
$$

for all $$x \in \mathcal{X}$$, $$y \in \mathcal{Y}$$.

But in practice, gradients computed for $$x$$ and $$y$$ often interfere with each other. Standard gradient descent/ascent doesn’t guarantee convergence, and can cause **cycling**, **divergence**, or chaotic dynamics.

---

### Dynamics of Simultaneous Gradient Updates

When training both networks via gradient descent (for the generator) and ascent (for the discriminator), we often do:

* Update discriminator:

  $$
  \theta_D \leftarrow \theta_D + \eta_D \nabla_{\theta_D} \mathcal{L}(G, D)
  $$

* Update generator:

  $$
  \theta_G \leftarrow \theta_G - \eta_G \nabla_{\theta_G} \mathcal{L}(G, D)
  $$

But this **simultaneous gradient step** may **not converge**, because gradients interfere with each other when the objective is non-convex and non-concave.

---

### A Toy Example: Rotating Vector Field

Even in a simple function like:

$$
f(x, y) = xy
$$

we get:

* $$\nabla_x f = y$$
* $$\nabla_y f = x$$

If you perform simultaneous gradient descent on $$x$$ and ascent on $$y$$, the updates are:

$$
x_{t+1} = x_t - \eta y_t \\
y_{t+1} = y_t + \eta x_t
$$

This forms a **rotation**, not convergence. The iterates spiral in the parameter space.

This is why GAN training is often unstable or oscillatory.

---

### How Do We Make It Better?

Researchers have developed many techniques to stabilize GAN optimization:

#### 1. **Alternative Objectives**

Instead of the original cross-entropy loss, we can use **non-saturating losses** or **Wasserstein distance**.

For example, in **Wasserstein GANs**, the objective becomes:

$$
\min_G \max_{D \in \mathcal{D}_1} \ \mathbb{E}_{x \sim p_{\text{data}}} [D(x)] - \mathbb{E}_{z \sim p(z)} [D(G(z))]
$$

Where $$\mathcal{D}_1$$ is the set of 1-Lipschitz functions. This gives smoother gradients and better convergence.

---

#### 2. **Gradient Penalties and Regularization**

To keep the discriminator smooth and avoid overfitting, we add penalties:

$$
\lambda \cdot \mathbb{E}_{\hat{x}} \left[ (\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2 \right]
$$

This encourages Lipschitz continuity and stabilizes learning.

---

#### 3. **Optimizers and Learning Rates**

* Using **Adam** instead of SGD helps with the noisy updates.
* Slowing down discriminator learning avoids overpowering the generator.

---

#### 4. **Two-Time Scale Updates**

Update the discriminator more frequently or with a smaller step size. This simulates solving the inner maximization more fully before updating the generator.

Formally:

$$
\eta_D \ll \eta_G \quad \text{or} \quad n_{\text{disc}} > n_{\text{gen}}
$$

This makes GAN optimization resemble **bilevel learning** (inner maximization + outer minimization).

---

### Summary: GAN Optimization Is Game Theory

Training GANs is not just optimization—it’s **game-theoretic optimization**.

You're looking for a **Nash equilibrium**, where neither the generator nor the discriminator can improve unilaterally.

But in practice:

* The game is non-convex and non-concave
* Gradients rotate or diverge
* Regular optimization tools aren’t enough

Which is why GAN optimization is a rich area of research in its own right.

---


## Conclusion: Optimization Beyond Convexity

Modern machine learning operates in optimization landscapes that are far from ideal. The classical guarantees we enjoy in convex optimization—unique global minima, predictable convergence—often break down in practice. Instead, we face a terrain filled with **saddle points**, **plateaus**, **chaotic basins**, and **adversarial objectives** that evolve during training.

This blog explored how these challenges manifest and how advanced optimization techniques attempt to address them:

* **Non-convex optimization** introduces ambiguity in convergence, requiring careful tuning and stochasticity to escape poor regions of the loss surface.
* **Saddle points** and **flat regions** slow training progress and require either structural insights or noisy dynamics to overcome.
* **Implicit differentiation** allows gradients to propagate through constrained systems and nested optimization problems, which are common in bilevel learning and neural ODEs.
* **Meta-learning algorithms** like MAML and Reptile optimize for adaptability, not just performance—shifting the goal from static solutions to rapid learning strategies.
* **GANs** turn optimization into a min-max game, where traditional descent-ascent methods can fail to converge without regularization or careful design.

In each case, the underlying challenge is not simply to minimize a loss, but to **design optimization procedures that work in unstable, interdependent, or ill-behaved systems**.

As machine learning models grow more expressive and their objectives more complex, the burden on optimization grows accordingly. Understanding the mathematical structure of these problems—gradients, Hessians, fixed points, constraint geometry—is essential not just for theory, but for building systems that actually work.

This shift from "solve this function" to "optimize this process" is at the core of what makes optimization in modern ML both difficult and intellectually rich. It also explains why many recent breakthroughs in deep learning and AI—such as better initialization schemes, adaptive optimizers, or gradient flow improvements—are fundamentally rooted in the language of optimization.

The deeper we go into learning systems, the more clear it becomes: **optimization isn't an afterthought—it is the architecture beneath learning itself**.

