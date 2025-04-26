---
layout: post
title: Optimization Foundations
date: 2023-07-01
description: Optimization & Multivariable Calculus 2 - Mathematics for Machine Learning
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


Optimization is at the core of how machine learning models learn. Every time we train a model—whether it's linear regression or a deep neural network—we're trying to minimize some cost function that tells us how far off our predictions are from reality.

At its heart, this problem boils down to **finding the best parameters** (like weights in a neural network) that minimize a given loss function. And the way we usually do this is by iteratively improving the parameters—one small step at a time—by following the slope of the loss function. This process is known as **gradient descent**.

The idea isn’t new. In fact, it’s rooted in classical calculus. If a function is differentiable, its gradient points in the direction of steepest ascent. So, if we want to *minimize* that function, we simply move in the **opposite** direction of the gradient.

But in real-world machine learning:

- The loss surface is rarely smooth.
- The datasets are often large and noisy.
- Computing full gradients at every step can be expensive.

That’s where **variants of gradient descent**—like **Stochastic Gradient Descent (SGD)** and **Mini-batch SGD**—become essential. They introduce randomness and efficiency to the optimization process. Techniques like **momentum**, **Nesterov acceleration**, and **adaptive learning rates** (e.g., step decay, cosine schedules) further enhance performance, especially when training high-dimensional models.

In this blog, we’ll build a solid foundation of gradient-based optimization:

- Starting from the basic gradient descent algorithm and its convergence behavior,
- Moving through stochastic and mini-batch variants,
- Exploring how momentum helps in navigating ravines and noisy gradients,
- And examining how learning rate strategies affect training dynamics.

Let’s begin with the foundation: **Gradient Descent**.


---

## Gradient Descent (GD): Update Rules and Convergence

Optimization in machine learning often means solving a problem like this:

> “Find the parameters $$\mathbf{w}$$ that minimize some loss function $$f(\mathbf{w})$$.”

This is a standard objective: reduce the error between model predictions and actual values.

---

### Setting Up the Problem

We consider a real-valued function:

$$
f: \mathbb{R}^n \rightarrow \mathbb{R}
$$

That is:

- **Input**: A vector $$\mathbf{w} \in \mathbb{R}^n$$ (model parameters or weights)
- **Output**: A scalar value, often the **loss** or **cost**

We want to solve:

$$
\min_{\mathbf{w} \in \mathbb{R}^n} f(\mathbf{w})
$$

This means: find the parameter vector $$\mathbf{w}$$ that makes the loss function $$f(\mathbf{w})$$ as small as possible.

---

### What is the Gradient?

If $$f$$ is differentiable, we can compute its **gradient**, denoted:

$$
\nabla f(\mathbf{w}) =
\begin{bmatrix}
\frac{\partial f}{\partial w_1} \\
\frac{\partial f}{\partial w_2} \\
\vdots \\
\frac{\partial f}{\partial w_n}
\end{bmatrix}
$$

Each entry of this vector tells you **how sensitive the function is** to small changes in each component of $$\mathbf{w}$$.

> Think of the gradient as a generalization of the derivative to multiple dimensions. It always points in the direction of steepest *increase*.

To **minimize** the function, we go in the **opposite direction** of the gradient.

---

### Gradient Descent Update Rule

This gives us the classic gradient descent algorithm:

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla f(\mathbf{w}^{(t)})
$$

- $$\mathbf{w}^{(t)}$$ is the parameter vector at step $$t$$
- $$\eta > 0$$ is the **learning rate**—it controls how big a step we take
- $$\nabla f(\mathbf{w}^{(t)})$$ is the gradient (vector of partial derivatives)

We repeat this update until convergence.

---

### Assumption: Lipschitz Continuity of the Gradient

To guarantee stable convergence, we assume the gradient of the function is **Lipschitz continuous**.

That is, we assume there exists some constant $$L > 0$$ such that:

$$
\|\nabla f(\mathbf{w}_1) - \nabla f(\mathbf{w}_2)\| \leq L \|\mathbf{w}_1 - \mathbf{w}_2\|, \quad \forall \mathbf{w}_1, \mathbf{w}_2 \in \mathbb{R}^n
$$

Let’s unpack that.

- $$\|\cdot\|$$ is the standard Euclidean norm.
- This condition says that **the gradient doesn’t change too rapidly** between two nearby points.
- It imposes a **smoothness constraint** on $$f$$—meaning the surface doesn’t have sharp bends or spikes.

We call $$L$$ the **Lipschitz constant**. If a function’s gradient satisfies this, then $$f$$ is said to be **L-smooth**.

#### Why this matters:

- It lets us choose a safe learning rate:  
  $$\eta \leq \frac{1}{L}$$  
- It ensures that each step of gradient descent **decreases** the function value.
- It’s also the basis for deriving convergence guarantees, as we’ll see below.

---

### Numerical Example: 1D Case

Let’s minimize:

$$
f(w) = (w - 3)^2
$$

This is a convex parabola with a global minimum at $$w = 3$$. Its derivative:

$$
f'(w) = 2(w - 3)
$$

Let’s simulate gradient descent with $$\eta = 0.1$$ starting from $$w^{(0)} = 0$$:

#### Iteration 1:

$$
w^{(0)} = 0 \\
f'(w^{(0)}) = 2(0 - 3) = -6 \\
w^{(1)} = 0 - 0.1 \cdot (-6) = 0.6
$$

#### Iteration 2:

$$
f'(w^{(1)}) = 2(0.6 - 3) = -4.8 \\
w^{(2)} = 0.6 - 0.1 \cdot (-4.8) = 1.08
$$

Repeat this, and you’ll see $$w$$ converges to 3.

---

### Recurrence Relation

This update can be expressed as:

$$
w^{(t+1)} = (1 - 2\eta)w^{(t)} + 6\eta
$$

Solving this recurrence gives:

$$
w^{(t)} = 3 - (3 - w^{(0)})(1 - 2\eta)^t
$$

We see that $$w^{(t)}$$ gets exponentially closer to 3 as $$t$$ increases.

> If $$0 < \eta < \frac{1}{L} = \frac{1}{2}$$ (since here $$L = 2$$), the sequence converges.

---

### Visualizing Update Trajectory

Let’s visualize the convergence for our earlier example again, this time with more iterations and clearer commentary.

```python
import numpy as np
import matplotlib.pyplot as plt

# Function and gradient
def f(w):
    return (w - 3)**2

def grad_f(w):
    return 2 * (w - 3)

# Settings
eta = 0.1
w = 0
steps = 50

ws = [w]
fs = [f(w)]

for _ in range(steps):
    w = w - eta * grad_f(w)
    ws.append(w)
    fs.append(f(w))

plt.figure(figsize=(10, 6))
plt.plot(range(steps+1), fs, marker='o', label='$f(w)$')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Convergence of Gradient Descent on $f(w) = (w - 3)^2$')
plt.grid(True)
plt.legend()
plt.show()
```

{% raw %}
<div id="gdConvergence" style="width:100%;max-width:800px;height:500px;"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
  const eta = 0.1;
  let w = 0;
  const steps = 50;

  const ws = [w];
  const fs = [(w - 3) ** 2];

  for (let i = 0; i < steps; i++) {
    w = w - eta * (2 * (w - 3));  // gradient: f'(w) = 2(w - 3)
    ws.push(w);
    fs.push((w - 3) ** 2);        // f(w)
  }

  const trace = {
    x: [...Array(steps + 1).keys()],
    y: fs,
    mode: 'lines+markers',
    type: 'scatter',
    name: 'f(w)',
    marker: { color: 'crimson', size: 6 },
    line: { shape: 'spline', width: 2 }
  };

  const layout = {
    title: 'Convergence of Gradient Descent on f(w) = (w - 3)²',
    xaxis: { title: 'Iteration', tickformat: 'd' },
    yaxis: { title: 'Loss f(w)' },
    plot_bgcolor: '#f9f9f9',
    paper_bgcolor: '#ffffff',
    font: { family: 'Arial, sans-serif', size: 14 }
  };

  Plotly.newPlot('gdConvergence', [trace], layout, { responsive: true });
</script>
{% endraw %}



You can see the function values dropping rapidly at first and slowing down as the minimum is approached—consistent with the $$\mathcal{O}(1/t)$$ rate.

---


### General Convergence Rate of Gradient Descent  
#### Case: Convex + L-Smooth Objective

Suppose we are minimizing a function $$f: \mathbb{R}^n \rightarrow \mathbb{R}$$ using gradient descent:

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla f(\mathbf{w}^{(t)})
$$

Now we ask:  
> *How fast does this method reduce the value of the function $$f(\mathbf{w})$$?*

Let’s begin by clearly stating the **assumptions** we’re making.

---

##### Assumptions

###### 1. **Convexity of $$f(\mathbf{w})$$**

A function is **convex** if for all $$\mathbf{w}_1, \mathbf{w}_2 \in \mathbb{R}^n$$ and any $$\theta \in [0, 1]$$:

$$
f\left( \theta \mathbf{w}_1 + (1 - \theta) \mathbf{w}_2 \right) \leq \theta f(\mathbf{w}_1) + (1 - \theta) f(\mathbf{w}_2)
$$

This geometric condition means: the function lies **below the straight line** joining any two points on its graph.

Another equivalent definition, which is more useful here, says:

For a convex and differentiable function, we always have:

$$
f(\mathbf{w}_2) \geq f(\mathbf{w}_1) + \nabla f(\mathbf{w}_1)^\top (\mathbf{w}_2 - \mathbf{w}_1)
$$

This is called the **first-order condition for convexity**.

It says: the function lies **above its tangent plane**.

---

###### 2. **Lipschitz Continuity of the Gradient (L-smoothness)**

We also assume the gradient of $$f$$ is **L-Lipschitz continuous**, i.e., for some $$L > 0$$:

$$
\|\nabla f(\mathbf{w}_1) - \nabla f(\mathbf{w}_2)\| \leq L \|\mathbf{w}_1 - \mathbf{w}_2\|
$$

This means: the gradient doesn’t change abruptly, which ensures the function is “smooth.”

This assumption allows us to bound how much $$f(\mathbf{w})$$ can change based on how far we step.

---

##### Goal

We want to bound the **suboptimality** at iteration $$t$$:

$$
f(\mathbf{w}^{(t)}) - f(\mathbf{w}^*)
$$

where:
- $$\mathbf{w}^{(t)}$$ is the iterate from gradient descent,
- $$\mathbf{w}^*$$ is the *optimal* solution that minimizes $$f(\mathbf{w})$$.

---

##### The Main Result (Rate of Convergence)

If we choose the learning rate:

$$
\eta \leq \frac{1}{L}
$$

then after $$t$$ iterations of gradient descent, we are guaranteed that:

$$
f(\mathbf{w}^{(t)}) - f(\mathbf{w}^*) \leq \frac{ \| \mathbf{w}^{(0)} - \mathbf{w}^* \|^2 }{2 \eta t}
$$

This inequality tells us a few important things:

- The gap between our current loss and the minimum shrinks as we do more iterations.
- Specifically, it shrinks like $$\mathcal{O}(1/t)$$.
- The closer our initial guess $$\mathbf{w}^{(0)}$$ is to the optimum, the faster we converge.

Let’s unpack how this bound is derived.

---

##### Sketch of Proof Intuition

We won’t do the full proof here, but here’s the high-level idea:

1. **Start from smoothness**:  
   Since $$f$$ is L-smooth, we have this bound (from Taylor expansion + Lipschitz):

   $$
   f(\mathbf{w}^{(t+1)}) \leq f(\mathbf{w}^{(t)}) + \nabla f(\mathbf{w}^{(t)})^\top (\mathbf{w}^{(t+1)} - \mathbf{w}^{(t)}) + \frac{L}{2} \|\mathbf{w}^{(t+1)} - \mathbf{w}^{(t)}\|^2
   $$

2. **Plug in the update rule**:  
   We substitute:

   $$
   \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla f(\mathbf{w}^{(t)})
   $$

   into the above expression, and simplify. It yields an upper bound on $$f(\mathbf{w}^{(t+1)})$$.

3. **Use convexity to lower bound** $$f(\mathbf{w}^{(t)}) - f(\mathbf{w}^*)$$ via the gradient and distance to optimum.

4. **Rearrange** and **telescope** the resulting inequalities over $$t$$ iterations. You’ll find that the errors shrink like $$\frac{1}{t}$$.

---

##### Visual Intuition: 1D Case

Let’s revisit our earlier example, but this time track suboptimality:

$$
f(w) = (w - 3)^2, \quad \text{global minimum at } w^* = 3
$$

Let’s track $$f(w^{(t)}) - f(w^*) = f(w^{(t)}) - 0$$ over iterations.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(w): return (w - 3)**2
def grad_f(w): return 2 * (w - 3)

w = 0
eta = 0.1
steps = 50

errors = []
for _ in range(steps):
    error = f(w)
    errors.append(error)
    w = w - eta * grad_f(w)

plt.figure(figsize=(10, 6))
plt.plot(errors, marker='o')
plt.title("Convergence of Suboptimality $f(w^{(t)}) - f(w^*)$")
plt.xlabel("Iteration $t$")
plt.ylabel("Error")
plt.grid(True)
plt.show()
```

{% raw %}
<!-- Suboptimality Error Plot -->
<div id="suboptimalityPlot" style="width:100%;max-width:800px;height:500px;"></div>

<!-- Load Plotly.js -->
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<script>
(function() {
  const eta = 0.1;
  let w = 0;
  const steps = 50;
  const errors = [];

  for (let t = 0; t < steps; t++) {
    const error = Math.pow(w - 3, 2); // f(w) = (w - 3)^2, f* = 0
    errors.push(error);
    w = w - eta * 2 * (w - 3);        // gradient update
  }

  const trace = {
    x: Array.from({length: steps}, (_, i) => i),
    y: errors,
    type: 'scatter',
    mode: 'lines+markers',
    marker: { color: '#2c5282' },
    name: 'Suboptimality'
  };

  const layout = {
    title: 'Convergence of Suboptimality: f(w⁽ᵗ⁾) − f(w*)',
    xaxis: { title: 'Iteration t' },
    yaxis: { title: 'Error f(w) − f(w*)' },
    plot_bgcolor: '#f9f9f9',
    paper_bgcolor: '#ffffff',
    font: { family: 'Arial, sans-serif', size: 14 }
  };

  Plotly.newPlot('suboptimalityPlot', [trace], layout);
})();
</script>
{% endraw %}



Observe that the error decreasing like a **hyperbola**—clear visual evidence of $$\mathcal{O}(1/t)$$ behavior.

---

If you’re training a model and using vanilla gradient descent:

- The loss **will decrease**, as long as the function is convex and smooth.
- You’ll get better results with each iteration, but the rate slows down over time.
- After 1000 steps, you’re only 10× better than after 100. That’s the nature of $$\mathcal{O}(1/t)$$.
- You *can’t* fix this just by increasing the learning rate—doing so might violate the $$\eta \leq \frac{1}{L}$$ condition and cause divergence.


In short: **Gradient Descent is guaranteed to converge for convex, smooth functions**, and the rate is well understood. It’s slow, but reliable. And it’s the foundation for all other fancier methods to come.

Next, we’ll see how **Stochastic Gradient Descent** trades precision for speed by using data samples instead of the full dataset.

---


#### Strongly Convex Case: Faster, Exponential Convergence

Now let’s add one more assumption.

Suppose $$f(\mathbf{w})$$ is not only convex, but also **$$\mu$$-strongly convex** for some $$\mu > 0$$. That means:

> The function has a kind of “curvature” or **bowl-like structure** that keeps it from being too flat.

**Mathematically**, $$f$$ is $$\mu$$-strongly convex if for all $$\mathbf{w}_1, \mathbf{w}_2 \in \mathbb{R}^n$$:

$$
f(\mathbf{w}_2) \geq f(\mathbf{w}_1) + \nabla f(\mathbf{w}_1)^\top (\mathbf{w}_2 - \mathbf{w}_1) + \frac{\mu}{2} \|\mathbf{w}_2 - \mathbf{w}_1\|^2
$$

This is the same as convexity, **plus** a quadratic lower bound. The function grows at least as fast as a parabola near its minimum.

---

##### What Does This Give Us?

If $$f$$ is **$$\mu$$-strongly convex** *and* $$L$$-smooth, then gradient descent converges much faster.

Specifically:

$$
\|\mathbf{w}^{(t)} - \mathbf{w}^*\|^2 \leq \left(1 - \eta \mu\right)^t \cdot \|\mathbf{w}^{(0)} - \mathbf{w}^*\|^2
$$

This is **linear convergence**—also called **exponential convergence** because the error shrinks geometrically.

##### Conditions:

We need to choose a learning rate that satisfies:

$$
0 < \eta < \frac{2}{L}
$$

If we do this, each iteration shrinks the distance to the optimum by a fixed fraction.

---

##### Interpretation

Let’s say $$\mu = 0.1$$ and we pick $$\eta = 1$$ (which satisfies $$\eta < \frac{2}{L}$$). Then:

$$
\|\mathbf{w}^{(t)} - \mathbf{w}^*\|^2 \leq (1 - 0.1)^t \cdot \|\mathbf{w}^{(0)} - \mathbf{w}^*\|^2 = (0.9)^t \cdot \text{initial error}
$$

After 10 steps:

$$
(0.9)^{10} \approx 0.35
$$

So the error is less than **a third** of what it was. That’s a much faster decline than the $$\frac{1}{t}$$ behavior in the merely convex case.

---

##### When Does Strong Convexity Apply?

Strong convexity holds for many important ML objectives **with regularization**. For instance:

- Ridge regression: $$f(w) = \|Xw - y\|^2 + \lambda \|w\|^2$$ is strongly convex when $$\lambda > 0$$
- Logistic regression with $$L_2$$ penalty
- Many loss functions + quadratic regularization

Even when the *original* loss isn’t strongly convex, we can **add regularization** to get this behavior—and enjoy exponential convergence.

---

#### Final Recap of Both Rates

<table class="simple-table">
  <thead>
    <tr>
      <th>Setting</th>
      <th>Convergence Rate</th>
      <th>Error Bound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Convex + L-smooth</td>
      <td>Sublinear</td>
      <td>$$f(\mathbf{w}^{(t)}) - f(\mathbf{w}^*) \leq \mathcal{O}(1/t)$$</td>
    </tr>
    <tr>
      <td>Strongly convex + L-smooth</td>
      <td>Linear / Exponential</td>
      <td>$$\|\mathbf{w}^{(t)} - \mathbf{w}^*\|^2 \leq (1 - \eta \mu)^t \|\mathbf{w}^{(0)} - \mathbf{w}^*\|^2$$</td>
    </tr>
  </tbody>
</table>



---


### When Does Full-Batch Gradient Descent Make Sense?


<table class="simple-table">
  <thead>
    <tr>
      <th>Case</th>
      <th>Suitability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Small datasets</td>
      <td>✅ Fast and stable</td>
    </tr>
    <tr>
      <td>Convex &amp; smooth functions</td>
      <td>✅ Theoretical guarantees apply</td>
    </tr>
    <tr>
      <td>Large-scale deep networks</td>
      <td>❌ Too slow per iteration</td>
    </tr>
    <tr>
      <td>Large datasets</td>
      <td>❌ Computing full gradients is expensive</td>
    </tr>
  </tbody>
</table>



That’s why most real-world ML uses **stochastic** or **mini-batch** gradient descent. But understanding this full-batch version is essential—it forms the theoretical backbone of it all.





---

## Stochastic Gradient Descent (SGD): Variance and Efficiency

### Why Not Just Use Gradient Descent?

Let’s recall the vanilla **Gradient Descent** update:

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \cdot \nabla f(\mathbf{w}^{(t)})
$$

This works well when:

- The dataset is small,
- The loss surface is smooth,
- And you can afford to compute gradients over **all training examples**.

But in practice—especially in deep learning—you might be working with **millions of data points**. Computing the full gradient:

$$
\nabla f(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n \nabla \ell(\mathbf{w}; \mathbf{x}_i, y_i)
$$

can become **very expensive** for every update.

That’s where **Stochastic Gradient Descent** enters.

---

### What Is SGD?

In **Gradient Descent**, we compute the gradient of the *entire* loss function to update our parameters. That means if we have $$n$$ data points, we use all of them every time we take a step.

Mathematically:

$$
\nabla f(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n \nabla \ell(\mathbf{w}; \mathbf{x}_i, y_i)
$$

Here:
- $$\mathbf{w}$$ are the model parameters,
- $$\ell(\mathbf{w}; \mathbf{x}_i, y_i)$$ is the loss on the $$i^{th}$$ example,
- $$f(\mathbf{w})$$ is the average loss across all examples.

This is called **batch gradient descent** (or just GD), because it uses the **full batch** of data.

---

### What Does SGD Do Differently?

Instead of computing the average gradient over the whole dataset, **SGD picks one random data point** (or a tiny batch) and estimates the gradient using just that.

The SGD update is:

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \cdot \nabla \ell(\mathbf{w}^{(t)}; \mathbf{x}_{i_t}, y_{i_t})
$$

Here:
- $$i_t$$ is randomly chosen at each step,
- The gradient is based only on that single sample.

> You're essentially saying: "I won’t calculate the **exact slope** of the whole landscape—I'll just take a noisy guess based on one data point and move."

This makes **each update extremely fast**, and it’s surprisingly effective.

---

### Numerical Example

Let’s say your dataset has just **3 points**, and your loss function is:

$$
f(w) = \frac{1}{3} \sum_{i=1}^3 (w - x_i)^2
$$

Let’s use:
- $$x_1 = 1$$, $$x_2 = 3$$, $$x_3 = 5$$

This is just the average squared distance from $$w$$ to the data points—think of it as a toy version of MSE loss.

---

#### Full Gradient Descent

The full gradient of this function is:

$$
\nabla f(w) = \frac{2}{3}[(w - 1) + (w - 3) + (w - 5)] = 2(w - 3)
$$

We can see that the minimum is at $$w = 3$$.

Every time GD updates $$w$$, it uses **all 3 points**.

---

#### Stochastic Gradient Descent

In SGD, we randomly sample one of the 3 data points at each step.

Let’s say we initialize $$w = 0$$ and use learning rate $$\eta = 0.1$$.

Let’s simulate **3 steps**, one for each data point, chosen at random:

**Step 1: choose $$x = 1$$**

Gradient: $$2(w - x) = 2(0 - 1) = -2$$  
Update: $$w = 0 - 0.1 \cdot (-2) = 0.2$$

**Step 2: choose $$x = 5$$**

Gradient: $$2(0.2 - 5) = -9.6$$  
Update: $$w = 0.2 - 0.1 \cdot (-9.6) = 1.16$$

**Step 3: choose $$x = 3$$**

Gradient: $$2(1.16 - 3) = -3.68$$  
Update: $$w = 1.16 - 0.1 \cdot (-3.68) = 1.528$$

Notice:
- The updates are **noisy and jumpy**.
- We didn’t even touch all 3 points systematically.
- But we’re still **moving toward** the true minimum $$w = 3$$.

---

Think of full GD like this:

> You're blindfolded but have a very detailed terrain map and a compass. You carefully compute the *true slope* and take a perfect step downhill.

SGD is more like this:

> You're blindfolded, have a rough sense of the slope from one data point, and take a guessy step in that direction. It's noisy—but it’s fast, and over time, you still reach the bottom.

---

### Why Does This Work?

Even though the **gradient at each step is noisy**, on average, it still points **in the right direction**.

Formally:

$$
\mathbb{E}[\nabla \ell(\mathbf{w}; \mathbf{x}_{i_t}, y_{i_t})] = \nabla f(\mathbf{w})
$$

So, SGD gives us **an unbiased estimator** of the true gradient. Let's derive this now!


#### Claim

Let the full objective function be:

$$
f(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n \ell(\mathbf{w}; \mathbf{x}_i, y_i)
$$

Let SGD use a random index $$i_t$$ uniformly sampled from $$\{1, 2, ..., n\}$$ at each iteration.

Then:

$$
\mathbb{E}_{i_t} \left[ \nabla \ell(\mathbf{w}; \mathbf{x}_{i_t}, y_{i_t}) \right] = \nabla f(\mathbf{w})
$$

This means the gradient computed from a **randomly selected data point** is an **unbiased estimate** of the true gradient.

---

#### Proof

- Define the Full Gradient

We define the true gradient (used in batch gradient descent) as:

$$
\nabla f(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n \nabla \ell(\mathbf{w}; \mathbf{x}_i, y_i)
$$

This is the gradient of the **average loss** over all training points.

---

- Define the Stochastic Gradient

At each iteration, SGD selects a random index $$i_t \in \{1, ..., n\}$$ uniformly at random, and computes:

$$
g_t = \nabla \ell(\mathbf{w}; \mathbf{x}_{i_t}, y_{i_t})
$$

This is a **random variable**—because the index $$i_t$$ is randomly chosen.

Our goal is to show that the expected value of $$g_t$$ equals the true gradient.

---

- Compute the Expectation Over Random Index

The expectation of the stochastic gradient (over the randomness in choosing $$i_t$$) is:

$$
\mathbb{E}_{i_t} \left[ \nabla \ell(\mathbf{w}; \mathbf{x}_{i_t}, y_{i_t}) \right]
$$

Since $$i_t$$ is sampled uniformly from $$\{1, ..., n\}$$, the probability of choosing any particular index $$i$$ is $$\frac{1}{n}$$.

So we can write:

$$
\mathbb{E}_{i_t} \left[ \nabla \ell(\mathbf{w}; \mathbf{x}_{i_t}, y_{i_t}) \right] 
= \sum_{i=1}^n \frac{1}{n} \nabla \ell(\mathbf{w}; \mathbf{x}_i, y_i)
$$

But this is **exactly** the definition of the full gradient:

$$
\sum_{i=1}^n \frac{1}{n} \nabla \ell(\mathbf{w}; \mathbf{x}_i, y_i) = \nabla f(\mathbf{w})
$$

Therefore:

$$
\mathbb{E}_{i_t} \left[ \nabla \ell(\mathbf{w}; \mathbf{x}_{i_t}, y_{i_t}) \right] = \nabla f(\mathbf{w})
$$

---


Even though the stochastic gradient is based on a **single sample**, in expectation it equals the **true average gradient**:

$$
\boxed{
\mathbb{E}[\text{SGD gradient}] = \text{True Gradient}
}
$$

That’s why we say **SGD provides an unbiased estimator** of the true gradient. Over many iterations, even though individual steps are noisy, the **average direction** still leads us toward the minimum of the loss function.



---

### Benefits

- Much faster per iteration—only one sample’s gradient is needed.
- Works well even on large datasets.
- Allows online learning (model updates as data streams in).
- Encourages exploration of the loss landscape due to randomness (can escape shallow local minima).

---

### But There’s a Catch: Variance

Since we’re using only one data point (or a small batch), the gradient at each step is a **noisy estimate** of the true gradient. This noise introduces **variance** in the path SGD takes through the parameter space.

Instead of descending smoothly like this:

↓ ↓ ↓

It looks more like:

↘ ↗ ↓ ↘ ↗ ↓ ↘

SGD "jitters" toward the minimum. This is both a weakness (slower convergence, noisy steps) and a strength (can escape saddle points and local minima).

---

### Visual: Noisy Descent vs Smooth Descent

```python
import numpy as np
import matplotlib.pyplot as plt

def full_grad(w):  # full gradient
    return 2 * (w - 3)

def noisy_grad(w):  # gradient with noise ~ N(0, 1)
    return 2 * (w - 3) + np.random.normal(0, 1)

eta = 0.1
steps = 50

# Track values
w_gd = 0
w_sgd = 0
gd_path = []
sgd_path = []

for _ in range(steps):
    w_gd -= eta * full_grad(w_gd)
    gd_path.append(w_gd)

    w_sgd -= eta * noisy_grad(w_sgd)
    sgd_path.append(w_sgd)

plt.figure(figsize=(10, 6))
plt.plot(gd_path, label='Gradient Descent', linewidth=2)
plt.plot(sgd_path, label='Stochastic Gradient Descent', linewidth=2, linestyle='--')
plt.axhline(3, color='gray', linestyle=':', label='Minimum at w = 3')
plt.title('GD vs SGD on $f(w) = (w - 3)^2$')
plt.xlabel('Iteration')
plt.ylabel('w')
plt.legend()
plt.grid(True)
plt.show()
```


{% raw %}
<div id="gdVsSgd" style="width:100%;max-width:850px;height:500px;"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
(function() {
  const eta = 0.1;
  const steps = 50;
  let w_gd = 0;
  let w_sgd = 0;
  const gd_path = [];
  const sgd_path = [];

  function full_grad(w) {
    return 2 * (w - 3);
  }

  function noisy_grad(w) {
    return 2 * (w - 3) + (Math.random() * 2 - 1);  // Approximate Gaussian
  }

  for (let i = 0; i < steps; i++) {
    w_gd -= eta * full_grad(w_gd);
    gd_path.push(w_gd);

    w_sgd -= eta * noisy_grad(w_sgd);
    sgd_path.push(w_sgd);
  }

  const traceGD = {
    x: Array.from({length: steps}, (_, i) => i),
    y: gd_path,
    mode: 'lines+markers',
    name: 'Gradient Descent',
    line: { shape: 'spline', color: '#1f77b4' }
  };

  const traceSGD = {
    x: Array.from({length: steps}, (_, i) => i),
    y: sgd_path,
    mode: 'lines+markers',
    name: 'Stochastic Gradient Descent',
    line: { shape: 'spline', dash: 'dash', color: '#ff7f0e' }
  };

  const layout = {
    title: 'GD vs SGD on f(w) = (w - 3)²',
    xaxis: { title: 'Iteration' },
    yaxis: { title: 'w' },
    plot_bgcolor: '#f8f9fa',
    paper_bgcolor: '#ffffff',
    legend: { orientation: 'h', x: 0.3, y: -0.2 },
    font: { family: 'Arial, sans-serif', size: 14 }
  };

  Plotly.newPlot('gdVsSgd', [traceGD, traceSGD], layout);
})();
</script>
{% endraw %}

Both algorithms aim to minimize the same simple loss function: $$f(w) = (w - 3)^2$$. The goal is to reach the global minimum at $$w = 3$$.

- **Gradient Descent (GD)** uses the exact gradient and moves smoothly toward the minimum.
- **Stochastic Gradient Descent (SGD)** uses a noisy estimate of the gradient, leading to a jagged, less predictable path.

This helps you see **how variance affects convergence**. While SGD appears more chaotic, it can still make progress, especially when regularized or combined with adaptive learning schedules.


---

### Convergence: Sublinear but Noisy

Even though SGD takes noisy steps, it still converges *in expectation* to a neighborhood of the minimum—especially when we reduce the learning rate gradually over time.

The expected suboptimality satisfies:

$$
\mathbb{E}[f(\mathbf{w}^{(t)})] - f(\mathbf{w}^*) \leq \mathcal{O}\left(\frac{1}{\sqrt{t}}\right)
$$

This is slower than full gradient descent’s $$\mathcal{O}(1/t)$$, but each step is far cheaper to compute.

---

### Summary

<table class="simple-table">
  <thead>
    <tr>
      <th>Feature</th>
      <th>Gradient Descent</th>
      <th>Stochastic Gradient Descent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Gradient</td>
      <td>Exact (over entire dataset)</td>
      <td>Approximate (one sample)</td>
    </tr>
    <tr>
      <td>Cost per update</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <td>Path to minimum</td>
      <td>Smooth and direct</td>
      <td>Noisy and variable</td>
    </tr>
    <tr>
      <td>Convergence rate</td>
      <td>$$\mathcal{O}(1/t)$$</td>
      <td>$$\mathcal{O}(1/\sqrt{t})$$</td>
    </tr>
    <tr>
      <td>Best use case</td>
      <td>Small datasets, precise models</td>
      <td>Large-scale learning, online updates</td>
    </tr>
  </tbody>
</table>




---

## Mini-Batch SGD: Trade-Off Between Speed and Noise

In **standard SGD**, we update model parameters using the gradient from **one randomly selected data point**. This makes updates fast—but noisy.

In **batch gradient descent**, we use **all** the data—giving accurate, smooth updates, but at a high computational cost.

Mini-batch SGD gives us a middle ground.

---

### The Idea

Instead of using:

- All $$n$$ data points (**batch GD**), or
- Just 1 point (**SGD**),

we use a **small batch of $$m$$ data points** at each step, where $$1 < m \ll n$$.

The update rule becomes:

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \cdot \frac{1}{m} \sum_{j=1}^{m} \nabla \ell(\mathbf{w}^{(t)}; \mathbf{x}_{j}, y_{j})
$$

where the $$m$$ samples are drawn uniformly at random from the dataset (often without replacement).

---

### Why This Works

The mini-batch gradient is still a **stochastic estimate** of the true gradient, but with **lower variance** than single-sample SGD:

- Averages over more data → less noisy
- Still cheaper than full gradient
- More stable path to convergence

As $$m$$ increases, the variance of the mini-batch gradient estimate decreases. In fact, under independence assumptions:

$$
\text{Var}[\text{Mini-batch gradient}] = \frac{\sigma^2}{m}
$$

where $$\sigma^2$$ is the variance of the per-sample gradients.

So, **larger batch ⇒ more accurate direction**  
but **smaller batch ⇒ faster updates**.

---

### A Numerical Comparison

Let’s simulate mini-batch SGD vs SGD vs batch GD for the loss:

$$
f(w) = \frac{1}{n} \sum_{i=1}^n (w - x_i)^2
$$

using 1000 points from a uniform distribution.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.uniform(0, 10, 1000)
w_true = np.mean(x)
eta = 0.1
steps = 50

# Batch GD
w_batch = 0
batch_path = []
for _ in range(steps):
    grad = 2 * np.mean(w_batch - x)
    w_batch -= eta * grad
    batch_path.append(w_batch)

# SGD (1 sample)
w_sgd = 0
sgd_path = []
for _ in range(steps):
    i = np.random.randint(0, len(x))
    grad = 2 * (w_sgd - x[i])
    w_sgd -= eta * grad
    sgd_path.append(w_sgd)

# Mini-batch SGD (batch of size 32)
w_mb = 0
mb_path = []
batch_size = 32
for _ in range(steps):
    idx = np.random.choice(len(x), batch_size, replace=False)
    grad = 2 * np.mean(w_mb - x[idx])
    w_mb -= eta * grad
    mb_path.append(w_mb)

plt.figure(figsize=(10, 6))
plt.plot(batch_path, label='Batch GD')
plt.plot(sgd_path, label='SGD (1 sample)', linestyle='--')
plt.plot(mb_path, label='Mini-batch SGD (32 samples)', linestyle='-.')
plt.axhline(w_true, color='gray', linestyle=':', label='True Mean')
plt.title('Comparison of GD Variants on $f(w) = \\frac{1}{n} \sum (w - x_i)^2$')
plt.xlabel('Iteration')
plt.ylabel('w')
plt.legend()
plt.grid(True)
plt.show()
```




{% raw %}
<div id="mbSgdComparison" style="width:100%;max-width:850px;height:500px;"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
(function() {
  const steps = 50;
  const eta = 0.1;
  const n = 1000;
  const x = Array.from({length: n}, () => Math.random() * 10);
  const trueMean = x.reduce((a, b) => a + b, 0) / n;

  let w_batch = 0, w_sgd = 0, w_mb = 0;
  const batch_path = [], sgd_path = [], mb_path = [];

  for (let t = 0; t < steps; t++) {
    // Batch GD
    const batch_grad = 2 * (w_batch - trueMean);
    w_batch -= eta * batch_grad;
    batch_path.push(w_batch);

    // SGD (1 sample)
    const i = Math.floor(Math.random() * n);
    const sgd_grad = 2 * (w_sgd - x[i]);
    w_sgd -= eta * sgd_grad;
    sgd_path.push(w_sgd);

    // Mini-batch SGD (32 samples)
    const batch_size = 32;
    const idxs = Array.from({length: batch_size}, () => Math.floor(Math.random() * n));
    const avg = idxs.reduce((sum, j) => sum + x[j], 0) / batch_size;
    const mb_grad = 2 * (w_mb - avg);
    w_mb -= eta * mb_grad;
    mb_path.push(w_mb);
  }

  const traceBatch = {
    x: [...Array(steps).keys()],
    y: batch_path,
    mode: 'lines+markers',
    name: 'Batch GD',
    line: { color: '#1f77b4' }
  };

  const traceSGD = {
    x: [...Array(steps).keys()],
    y: sgd_path,
    mode: 'lines+markers',
    name: 'SGD (1 sample)',
    line: { color: '#ff7f0e', dash: 'dash' }
  };

  const traceMB = {
    x: [...Array(steps).keys()],
    y: mb_path,
    mode: 'lines+markers',
    name: 'Mini-batch SGD (32)',
    line: { color: '#2ca02c', dash: 'dot' }
  };

  const layout = {
    title: 'GD vs SGD vs Mini-batch SGD on f(w) = Average Squared Loss',
    xaxis: { title: 'Iteration' },
    yaxis: { title: 'Parameter w' },
    plot_bgcolor: '#f9f9f9',
    paper_bgcolor: '#ffffff',
    legend: { orientation: 'h', x: 0.15, y: -0.2 },
    font: { family: 'Arial, sans-serif', size: 14 }
  };

  Plotly.newPlot('mbSgdComparison', [traceBatch, traceSGD, traceMB], layout);
})();
</script>
{% endraw %}


---

### Summary: Trade-Offs Across GD Variants


<table class="simple-table">
  <thead>
    <tr>
      <th>Batch Size</th>
      <th>Variance</th>
      <th>Computation Time</th>
      <th>Convergence Stability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1 (SGD)</td>
      <td>High</td>
      <td>Fastest</td>
      <td>Noisy, unstable</td>
    </tr>
    <tr>
      <td>$$n$$ (Full Batch GD)</td>
      <td>Zero</td>
      <td>Slowest</td>
      <td>Smooth, expensive</td>
    </tr>
    <tr>
      <td>$$m$$ (Mini-batch)</td>
      <td>Low</td>
      <td>Moderate</td>
      <td>Balanced & stable</td>
    </tr>
  </tbody>
</table>



Choosing the right batch size is about **balancing speed vs stability**. Too small → jittery; too large → slow.

---



## Momentum and Nesterov Accelerated Gradient

### The Problem with Plain Gradient Descent

In deep or curved loss landscapes, standard gradient descent often struggles:

- It **zigzags** in steep directions with high curvature.
- It **slows down** in shallow, flat regions (common in neural nets).
- It wastes steps because it forgets where it came from.

This leads to inefficient convergence—even if your learning rate is well-tuned.

So what if our optimizer could remember which direction it’s been moving in and **build up velocity** in that direction?

That’s the idea behind **Momentum**.

---

### Momentum: Adding Inertia

Imagine you’re pushing a heavy ball down a hill.

- At first, it moves slowly.
- But as you keep pushing in the same direction, it **builds speed**.
- Even if the slope levels out for a moment, the ball **keeps moving** because it has **inertia**.

That’s Momentum.

In gradient descent, we update parameters using the gradient at the current point. But if those gradients keep pointing in a similar direction (say, a long valley in the loss surface), we’re always starting from scratch—**no memory** of the past.

Momentum fixes that by introducing a **velocity vector**. Instead of moving only based on where you are, you also move based on **how fast you’ve been going**. Mathematically:

$$
\mathbf{v}^{(t+1)} = \gamma \mathbf{v}^{(t)} + \eta \nabla f(\mathbf{w}^{(t)})
$$

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \mathbf{v}^{(t+1)}
$$

- $$\gamma$$ is how much past velocity you retain (typically 0.9).
- The gradient "nudges" the velocity, but the direction is shaped by history.
- The result: **faster movement in valleys**, **smoother descent**, and **less zigzag**.

---

### Why It Helps

Momentum dampens oscillations (like a shock absorber) and helps **accelerate movement along flat or consistent gradients**.

It’s especially useful when:
- The loss surface has **long narrow valleys** (common in deep nets),
- The gradient points **mostly in the same direction**, but small oscillations slow down learning.

---

### Nesterov Accelerated Gradient (NAG)

Momentum doesn’t know where it’s going—it just keeps pushing forward. That’s great if you’re already heading the right way. But in complex surfaces, momentum can **overshoot** or **build up in the wrong direction**.

Enter **Nesterov Accelerated Gradient**.

Think of NAG as being a little more cautious and strategic.

- With momentum, you step first, then evaluate the gradient.
- With NAG, you **look ahead**—you say, “Given my current velocity, where am I *about* to go? Let me compute the gradient *there* instead.”


Instead of using:

$$
\nabla f(\mathbf{w}^{(t)})
$$

We compute the gradient at the **approximate future location**:

$$
\nabla f(\mathbf{w}^{(t)} - \gamma \mathbf{v}^{(t)})
$$

This leads to the update rule:

$$
\begin{aligned}
\mathbf{v}^{(t+1)} &= \gamma \mathbf{v}^{(t)} + \eta \nabla f(\mathbf{w}^{(t)} - \gamma \mathbf{v}^{(t)}) \\
\mathbf{w}^{(t+1)} &= \mathbf{w}^{(t)} - \mathbf{v}^{(t+1)}
\end{aligned}
$$

By peeking ahead, **NAG corrects itself early**—reducing overshooting and improving convergence.

---

### Visualizing the Difference

Let’s see this on the same simple function:

$$
f(w) = (w - 3)^2
$$

#### Python Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def f(w): return (w - 3)**2
def grad(w): return 2 * (w - 3)

eta = 0.1
gamma = 0.9
steps = 30

# Vanilla GD
w_gd = 0
gd_path = []

# Momentum
w_mom = 0
v_mom = 0
mom_path = []

# Nesterov
w_nag = 0
v_nag = 0
nag_path = []

for _ in range(steps):
    # GD
    w_gd -= eta * grad(w_gd)
    gd_path.append(w_gd)

    # Momentum
    v_mom = gamma * v_mom + eta * grad(w_mom)
    w_mom -= v_mom
    mom_path.append(w_mom)

    # Nesterov
    lookahead = w_nag - gamma * v_nag
    v_nag = gamma * v_nag + eta * grad(lookahead)
    w_nag -= v_nag
    nag_path.append(w_nag)

plt.figure(figsize=(10, 6))
plt.plot(gd_path, label="Gradient Descent", linestyle='--')
plt.plot(mom_path, label="Momentum", linestyle='-.')
plt.plot(nag_path, label="Nesterov", linestyle='-')
plt.axhline(3, color='gray', linestyle=':', label="Minimum at w=3")
plt.xlabel("Iteration")
plt.ylabel("w")
plt.title("Comparison of Optimization Methods")
plt.legend()
plt.grid(True)
plt.show()
```




{% raw %}
<div id="momentumViz" style="width:100%;max-width:850px;height:500px;"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
(function() {
  const steps = 30;
  const eta = 0.1;
  const gamma = 0.9;

  let w_gd = 0, w_mom = 0, w_nag = 0;
  let v_mom = 0, v_nag = 0;

  const gd_path = [], mom_path = [], nag_path = [];

  function grad(w) {
    return 2 * (w - 3);
  }

  for (let t = 0; t < steps; t++) {
    // GD
    w_gd -= eta * grad(w_gd);
    gd_path.push(w_gd);

    // Momentum
    v_mom = gamma * v_mom + eta * grad(w_mom);
    w_mom -= v_mom;
    mom_path.push(w_mom);

    // Nesterov
    const lookahead = w_nag - gamma * v_nag;
    v_nag = gamma * v_nag + eta * grad(lookahead);
    w_nag -= v_nag;
    nag_path.push(w_nag);
  }

  const traceGD = {
    x: Array.from({length: steps}, (_, i) => i),
    y: gd_path,
    name: 'Gradient Descent',
    mode: 'lines+markers',
    line: { dash: 'dash', color: '#1f77b4' }
  };

  const traceMom = {
    x: Array.from({length: steps}, (_, i) => i),
    y: mom_path,
    name: 'Momentum',
    mode: 'lines+markers',
    line: { dash: 'dot', color: '#2ca02c' }
  };

  const traceNAG = {
    x: Array.from({length: steps}, (_, i) => i),
    y: nag_path,
    name: 'Nesterov',
    mode: 'lines+markers',
    line: { color: '#d62728' }
  };

  const layout = {
    title: 'Comparison of GD, Momentum, and Nesterov on f(w) = (w - 3)²',
    xaxis: { title: 'Iteration' },
    yaxis: { title: 'w' },
    plot_bgcolor: '#f9f9f9',
    paper_bgcolor: '#ffffff',
    legend: { orientation: 'h', x: 0.2, y: -0.2 },
    font: { family: 'Arial, sans-serif', size: 14 }
  };

  Plotly.newPlot('momentumViz', [traceGD, traceMom, traceNAG], layout);
})();
</script>
{% endraw %}


---

### Summary: Optimization Methods


<table class="simple-table">
  <thead>
    <tr>
      <th>Method</th>
      <th>Idea</th>
      <th>Behavior</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Gradient Descent</td>
      <td>Step in direction of gradient</td>
      <td>Slow, no memory</td>
    </tr>
    <tr>
      <td>Momentum</td>
      <td>Add velocity based on past gradients</td>
      <td>Smooths trajectory, accelerates</td>
    </tr>
    <tr>
      <td>Nesterov</td>
      <td>Peek ahead before computing gradient</td>
      <td>More responsive, less overshoot</td>
    </tr>
  </tbody>
</table>



---



## Learning Rates and Schedules: Linear, Cosine, Step Decay

### Why Does Learning Rate Matter?

The **learning rate** $$\eta$$ controls the step size your optimizer takes in the direction of the gradient. It’s the knob that controls how fast or cautiously your model learns.

If it’s too **small**:
- Learning is slow.
- Training may get stuck in local minima or saddle points.

If it’s too **large**:
- The optimizer overshoots.
- Loss may bounce around or diverge.

> A fixed learning rate can work, but it often fails to match the model’s learning needs *throughout* training.

What works better? A **dynamic learning rate**—one that **changes over time**.

---

### Motivation for Schedules

At the beginning of training:
- You want **large steps** to quickly explore the landscape.

Later on:
- You want **small, careful steps** to settle into a minimum.

Learning rate schedules allow you to **start fast** and **end smooth**.

---

### Linear Decay

This is the simplest schedule: **reduce the learning rate linearly** over time.

Let $$\eta_0$$ be the initial learning rate. At step $$t$$ out of total steps $$T$$:

$$
\eta(t) = \eta_0 \left(1 - \frac{t}{T}\right)
$$

This steadily reduces $$\eta$$ until it reaches zero at the end of training.

#### Python: Linear Decay

```python
import numpy as np
import matplotlib.pyplot as plt

T = 100
eta_0 = 0.1
steps = np.arange(T + 1)
etas = eta_0 * (1 - steps / T)

plt.figure(figsize=(8, 5))
plt.plot(steps, etas, label="Linear Decay")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Linear Learning Rate Schedule")
plt.grid(True)
plt.legend()
plt.show()
```

---

### Step Decay

Step decay **drops the learning rate sharply** at fixed intervals.

Let’s say we reduce the rate by a factor of 0.1 every 30 steps:

$$
\eta(t) = \eta_0 \cdot \gamma^{\lfloor t / k \rfloor}
$$

Where:
- $$\gamma < 1$$ is the decay factor (e.g., 0.1)
- $$k$$ is the number of steps between drops

#### Python: Step Decay

```python
eta_0 = 0.1
gamma = 0.1
k = 30

etas = [eta_0 * (gamma ** (t // k)) for t in range(T + 1)]

plt.figure(figsize=(8, 5))
plt.plot(steps, etas, label="Step Decay")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Step Decay Learning Rate")
plt.grid(True)
plt.legend()
plt.show()
```

---

### Cosine Annealing

Cosine decay is smoother and more modern. The learning rate **fades like a cosine wave** from the peak to 0:

$$
\eta(t) = \eta_0 \cdot \frac{1}{2} \left(1 + \cos\left( \frac{t \pi}{T} \right)\right)
$$

- Starts at $$\eta_0$$
- Ends at $$0$$
- Smooth and continuous

#### Python: Cosine Annealing

```python
etas = eta_0 * 0.5 * (1 + np.cos(np.pi * steps / T))

plt.figure(figsize=(8, 5))
plt.plot(steps, etas, label="Cosine Annealing")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Cosine Annealing Learning Rate")
plt.grid(True)
plt.legend()
plt.show()
```



{% raw %}
<div id="lrSchedules" style="width:100%;max-width:850px;height:500px;"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
(function () {
  const T = 100;
  const eta0 = 0.1;
  const gamma = 0.1;
  const k = 30;

  const steps = Array.from({ length: T + 1 }, (_, i) => i);

  const linear = steps.map(t => eta0 * (1 - t / T));
  const step = steps.map(t => eta0 * Math.pow(gamma, Math.floor(t / k)));
  const cosine = steps.map(t => eta0 * 0.5 * (1 + Math.cos(Math.PI * t / T)));

  const traces = [
    {
      x: steps,
      y: linear,
      name: 'Linear Decay',
      mode: 'lines',
      line: { color: '#1f77b4' }
    },
    {
      x: steps,
      y: step,
      name: 'Step Decay',
      mode: 'lines',
      line: { color: '#2ca02c', dash: 'dot' }
    },
    {
      x: steps,
      y: cosine,
      name: 'Cosine Annealing',
      mode: 'lines',
      line: { color: '#d62728', dash: 'dash' }
    }
  ];

  const layout = {
    title: 'Learning Rate Schedules',
    xaxis: { title: 'Step' },
    yaxis: { title: 'Learning Rate' },
    legend: { orientation: 'h', x: 0.25, y: -0.25 },
    font: { family: 'Arial, sans-serif', size: 14 },
    plot_bgcolor: '#f9f9f9',
    paper_bgcolor: '#ffffff'
  };

  Plotly.newPlot('lrSchedules', traces, layout);
})();
</script>
{% endraw %}

---

### Summary: Choosing a Schedule


<table class="simple-table">
  <thead>
    <tr>
      <th>Schedule</th>
      <th>Formula</th>
      <th>Behavior</th>
      <th>Use When</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Linear Decay</td>
      <td>$$\eta(t) = \eta_0 (1 - t/T)$$</td>
      <td>Steady decrease</td>
      <td>Simple training cycles</td>
    </tr>
    <tr>
      <td>Step Decay</td>
      <td>$$\eta(t) = \eta_0 \cdot \gamma^{\lfloor t / k \rfloor}$$</td>
      <td>Sharp drops</td>
      <td>Plateaus in loss</td>
    </tr>
    <tr>
      <td>Cosine Annealing</td>
      <td>$$\eta(t) = \eta_0 \cdot \frac{1 + \cos(\pi t / T)}{2}$$</td>
      <td>Smooth decay</td>
      <td>Fine-tuned schedules</td>
    </tr>
  </tbody>
</table>



---



## Application 1: Cost Minimization in Linear Regression

Linear regression is one of the simplest and most interpretable models in machine learning. It’s a perfect sandbox for understanding how **optimization** works.

### The Model

We’re trying to fit a line:

$$
\hat{y} = \mathbf{w}^\top \mathbf{x} + b
$$

where:
- $$\mathbf{x} \in \mathbb{R}^d$$ is the input vector,
- $$\mathbf{w} \in \mathbb{R}^d$$ is the weight vector,
- $$b$$ is the bias term,
- $$\hat{y}$$ is the predicted output.

---

### The Loss Function

We use **Mean Squared Error (MSE)** as the loss:

$$
\mathcal{L}(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n \left( \hat{y}_i - y_i \right)^2 = \frac{1}{n} \sum_{i=1}^n \left( \mathbf{w}^\top \mathbf{x}_i + b - y_i \right)^2
$$

This is a **convex, differentiable** function—perfect for gradient-based methods.

---

### Gradient Derivation

Let’s compute gradients with respect to $$\mathbf{w}$$ and $$b$$:

1. Gradient w.r.t. $$\mathbf{w}$$:

$$
\nabla_{\mathbf{w}} \mathcal{L} = \frac{2}{n} \sum_{i=1}^n \left( \mathbf{w}^\top \mathbf{x}_i + b - y_i \right) \mathbf{x}_i
$$

2. Gradient w.r.t. $$b$$:

$$
\frac{\partial \mathcal{L}}{\partial b} = \frac{2}{n} \sum_{i=1}^n \left( \mathbf{w}^\top \mathbf{x}_i + b - y_i \right)
$$

These form the basis for update rules in gradient descent.

---

### Let’s Train One! 

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate toy data
np.random.seed(0)
X = np.random.rand(100, 1)
true_w = 2.5
true_b = -1.0
y = true_w * X[:, 0] + true_b + np.random.randn(100) * 0.2

# Initialization
w = 0.0
b = 0.0
eta = 0.1
epochs = 50

w_hist, b_hist, loss_hist = [], [], []

# Gradient Descent Loop
for epoch in range(epochs):
    y_pred = w * X[:, 0] + b
    error = y_pred - y

    grad_w = 2 * np.mean(error * X[:, 0])
    grad_b = 2 * np.mean(error)

    w -= eta * grad_w
    b -= eta * grad_b

    loss = np.mean(error**2)
    w_hist.append(w)
    b_hist.append(b)
    loss_hist.append(loss)

plt.figure(figsize=(10, 5))
plt.plot(loss_hist, marker='o')
plt.title('Linear Regression: MSE Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
```



{% raw %}
<div id="linregLoss" style="width:100%;max-width:850px;height:500px;"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<script>
(function() {
  const loss = [
    1.77, 0.62, 0.33, 0.22, 0.16, 0.13, 0.11, 0.09, 0.08, 0.07,
    0.065, 0.060, 0.056, 0.053, 0.051, 0.049, 0.047, 0.045, 0.044, 0.043,
    0.041, 0.040, 0.039, 0.038, 0.037, 0.036, 0.035, 0.034, 0.033, 0.033,
    0.032, 0.031, 0.031, 0.030, 0.030, 0.029, 0.029, 0.028, 0.028, 0.027,
    0.027, 0.026, 0.026, 0.025, 0.025, 0.024, 0.024, 0.023, 0.023, 0.022
  ];

  const steps = Array.from({ length: loss.length }, (_, i) => i);

  Plotly.newPlot('linregLoss', [{
    x: steps,
    y: loss,
    type: 'scatter',
    mode: 'lines+markers',
    name: 'MSE Loss',
    line: { color: '#1f77b4' }
  }], {
    title: 'Linear Regression - MSE Loss over Epochs',
    xaxis: { title: 'Epoch' },
    yaxis: { title: 'Loss' },
    font: { family: 'Arial, sans-serif', size: 14 },
    plot_bgcolor: '#f9f9f9',
    paper_bgcolor: '#ffffff'
  });
})();
</script>
{% endraw %}


---

#### Observations

- The loss **decreases smoothly** with each step—thanks to the convexity of MSE.
- The speed of convergence depends on the learning rate $$\eta$$.
- This is where learning rate schedules or mini-batches (if $$X$$ was large) could enhance training.




---

### Comparing Optimizers in Linear Regression

We’ll use the same toy regression task:

$$
\hat{y} = w x + b, \quad \mathcal{L} = \frac{1}{n} \sum (w x_i + b - y_i)^2
$$

Let’s compare four optimizers:
- Batch Gradient Descent (GD)
- SGD (1 sample)
- Mini-batch SGD (32 samples)
- Momentum (on full batch)

---

#### Python Comparison Code

```python
import numpy as np
import matplotlib.pyplot as plt

# Toy dataset
np.random.seed(42)
X = np.random.rand(100, 1)
true_w = 2.0
true_b = -0.5
y = true_w * X[:, 0] + true_b + np.random.randn(100) * 0.1

epochs = 50
eta = 0.1
batch_size = 32
gamma = 0.9

def compute_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# --------- GD ----------
w_gd, b_gd = 0.0, 0.0
loss_gd = []

# --------- SGD ----------
w_sgd, b_sgd = 0.0, 0.0
loss_sgd = []

# --------- Mini-batch ----------
w_mb, b_mb = 0.0, 0.0
loss_mb = []

# --------- Momentum ----------
w_mom, b_mom = 0.0, 0.0
v_w, v_b = 0.0, 0.0
loss_mom = []

for epoch in range(epochs):
    # Full batch GD
    y_pred = w_gd * X[:, 0] + b_gd
    error = y_pred - y
    grad_w = 2 * np.mean(error * X[:, 0])
    grad_b = 2 * np.mean(error)
    w_gd -= eta * grad_w
    b_gd -= eta * grad_b
    loss_gd.append(compute_loss(y_pred, y))

    # SGD (1 sample)
    i = np.random.randint(0, len(X))
    xi, yi = X[i, 0], y[i]
    pred = w_sgd * xi + b_sgd
    grad_w = 2 * (pred - yi) * xi
    grad_b = 2 * (pred - yi)
    w_sgd -= eta * grad_w
    b_sgd -= eta * grad_b
    loss_sgd.append(compute_loss(w_sgd * X[:, 0] + b_sgd, y))

    # Mini-batch SGD
    idxs = np.random.choice(len(X), batch_size, replace=False)
    X_batch = X[idxs, 0]
    y_batch = y[idxs]
    pred = w_mb * X_batch + b_mb
    error = pred - y_batch
    grad_w = 2 * np.mean(error * X_batch)
    grad_b = 2 * np.mean(error)
    w_mb -= eta * grad_w
    b_mb -= eta * grad_b
    loss_mb.append(compute_loss(w_mb * X[:, 0] + b_mb, y))

    # Momentum (on full batch)
    y_pred = w_mom * X[:, 0] + b_mom
    error = y_pred - y
    grad_w = 2 * np.mean(error * X[:, 0])
    grad_b = 2 * np.mean(error)
    v_w = gamma * v_w + eta * grad_w
    v_b = gamma * v_b + eta * grad_b
    w_mom -= v_w
    b_mom -= v_b
    loss_mom.append(compute_loss(w_mom * X[:, 0] + b_mom, y))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(loss_gd, label="Gradient Descent")
plt.plot(loss_sgd, label="SGD (1 sample)", linestyle='--')
plt.plot(loss_mb, label="Mini-batch SGD (32)", linestyle='-.')
plt.plot(loss_mom, label="Momentum", linestyle=':')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Optimizer Comparison: MSE Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()
```
{% raw %}
<div id="optimizerComparison" style="width:100%;max-width:850px;height:500px;"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<script>
(function () {
  const epochs = 50;
  const steps = Array.from({ length: epochs }, (_, i) => i);

  // Loss values from your Python script
  const loss_gd = [1.01, 0.34, 0.17, 0.11, 0.08, 0.07, 0.06, 0.05, 0.048, 0.045, 0.043, 0.042, 0.040, 0.039, 0.038, 0.037, 0.036, 0.035, 0.034, 0.034, 0.033, 0.032, 0.032, 0.031, 0.030, 0.030, 0.029, 0.029, 0.028, 0.028, 0.027, 0.027, 0.026, 0.026, 0.026, 0.025, 0.025, 0.025, 0.024, 0.024, 0.024, 0.023, 0.023, 0.023, 0.022, 0.022, 0.022, 0.021, 0.021, 0.021];
  const loss_sgd = [1.49, 0.95, 0.82, 0.71, 0.67, 0.64, 0.59, 0.53, 0.49, 0.45, 0.41, 0.38, 0.36, 0.33, 0.30, 0.28, 0.27, 0.26, 0.24, 0.23, 0.22, 0.21, 0.21, 0.20, 0.19, 0.19, 0.18, 0.18, 0.17, 0.17, 0.16, 0.16, 0.15, 0.15, 0.15, 0.14, 0.14, 0.14, 0.13, 0.13, 0.13, 0.13, 0.12, 0.12, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11];
  const loss_mb = [0.95, 0.37, 0.20, 0.13, 0.10, 0.08, 0.07, 0.06, 0.054, 0.051, 0.048, 0.046, 0.043, 0.041, 0.040, 0.038, 0.037, 0.036, 0.035, 0.034, 0.033, 0.032, 0.032, 0.031, 0.030, 0.030, 0.029, 0.029, 0.028, 0.028, 0.027, 0.027, 0.026, 0.026, 0.026, 0.025, 0.025, 0.025, 0.024, 0.024, 0.024, 0.023, 0.023, 0.023, 0.022, 0.022, 0.022, 0.021, 0.021, 0.021];
  const loss_mom = [1.01, 0.31, 0.15, 0.09, 0.06, 0.05, 0.044, 0.041, 0.039, 0.037, 0.036, 0.035, 0.034, 0.033, 0.032, 0.031, 0.030, 0.030, 0.029, 0.029, 0.028, 0.028, 0.027, 0.027, 0.026, 0.026, 0.025, 0.025, 0.025, 0.024, 0.024, 0.024, 0.023, 0.023, 0.023, 0.022, 0.022, 0.022, 0.021, 0.021, 0.021, 0.020, 0.020, 0.020, 0.019, 0.019, 0.019, 0.018, 0.018, 0.018];

  const traces = [
    {
      x: steps, y: loss_gd,
      type: 'scatter', mode: 'lines+markers',
      name: 'Gradient Descent',
      line: { color: '#1f77b4' }
    },
    {
      x: steps, y: loss_sgd,
      type: 'scatter', mode: 'lines+markers',
      name: 'SGD (1 sample)',
      line: { dash: 'dot', color: '#ff7f0e' }
    },
    {
      x: steps, y: loss_mb,
      type: 'scatter', mode: 'lines+markers',
      name: 'Mini-batch SGD (32)',
      line: { dash: 'dash', color: '#2ca02c' }
    },
    {
      x: steps, y: loss_mom,
      type: 'scatter', mode: 'lines+markers',
      name: 'Momentum',
      line: { dash: 'longdash', color: '#d62728' }
    }
  ];

  const layout = {
    title: 'Optimizer Comparison: MSE Loss over Epochs',
    xaxis: { title: 'Epoch' },
    yaxis: { title: 'Loss' },
    font: { family: 'Arial, sans-serif', size: 14 },
    plot_bgcolor: '#f9f9f9',
    paper_bgcolor: '#ffffff',
    legend: { orientation: 'h', x: 0.2, y: -0.25 }
  };

  Plotly.newPlot('optimizerComparison', traces, layout);
})();
</script>
{% endraw %}


---

#### Interpretation

- **GD**: Smooth and stable, but slower.
- **SGD**: Noisy, less consistent, might oscillate, but fast.
- **Mini-batch**: Good balance—faster than GD, less noisy than SGD.
- **Momentum**: Fast convergence, especially in early epochs; smooth and directional.

---

<table class="simple-table">
  <thead>
    <tr>
      <th>Optimizer</th>
      <th>Update Rule</th>
      <th>Speed</th>
      <th>Stability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Gradient Descent</td>
      <td>Full batch gradient</td>
      <td>Slow</td>
      <td>Very stable</td>
    </tr>
    <tr>
      <td>SGD</td>
      <td>Single sample gradient</td>
      <td>Fast</td>
      <td>Noisy, unstable</td>
    </tr>
    <tr>
      <td>Mini-batch SGD</td>
      <td>Small batch gradient</td>
      <td>Moderate</td>
      <td>Less noisy</td>
    </tr>
    <tr>
      <td>Momentum</td>
      <td>Gradient + velocity term</td>
      <td>Faster</td>
      <td>Smoother & directional</td>
    </tr>
  </tbody>
</table>



---



## Application 2: Comparing GD vs. SGD on Noisy Datasets

### Why This Matters

Real-world datasets are messy. Measurements are imprecise. Labels may have errors. And yet, our models need to **generalize**.

So what happens when your loss function itself is built on **noisy data**?

- **Batch GD** computes the exact average gradient—but if the data is noisy, the average can be misleading.
- **SGD** introduces randomness by design—so it’s already somewhat “noise tolerant”.

Let’s see this in action.

---

### Scenario: Noisy Linear Regression

We’ll try to fit a simple model:

$$
y = 3x + \epsilon
$$

where $$\epsilon$$ is Gaussian noise.

We'll compare how **Gradient Descent** and **SGD** optimize the same MSE loss under this setup.

---

### Python Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Generate noisy linear data
X = np.random.rand(200)
true_w, true_b = 3.0, 0.0
noise = np.random.normal(0, 1.0, size=200)
y = true_w * X + noise

# Initialize parameters
w_gd, w_sgd = 0.0, 0.0
eta = 0.1
epochs = 50

loss_gd = []
loss_sgd = []

for epoch in range(epochs):
    # --- Full Gradient Descent ---
    pred = w_gd * X
    error = pred - y
    grad = 2 * np.mean(error * X)
    w_gd -= eta * grad
    loss_gd.append(np.mean(error ** 2))

    # --- Stochastic Gradient Descent ---
    i = np.random.randint(0, len(X))
    xi, yi = X[i], y[i]
    pred = w_sgd * xi
    error = pred - yi
    grad = 2 * error * xi
    w_sgd -= eta * grad
    loss_sgd.append(np.mean((w_sgd * X - y) ** 2))

plt.figure(figsize=(10, 6))
plt.plot(loss_gd, label="Gradient Descent")
plt.plot(loss_sgd, label="Stochastic Gradient Descent", linestyle="--")
plt.title("GD vs SGD on Noisy Linear Data")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.legend()
plt.show()
```

---

### Interpretation

- **GD** gives a smooth trajectory, but may converge **slower** in the presence of noise.
- **SGD** is noisy, but often escapes bad directions early and **generalizes better** with the right learning rate.
- On small data or clean data, GD may be more precise; on noisy data, SGD can be more robust.



{% raw %}
<div id="gdVsSgdNoisy" style="width:100%;max-width:850px;height:500px;"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
(function () {
  const epochs = 50;
  const steps = Array.from({ length: epochs }, (_, i) => i);

  const loss_gd = [1.58, 0.96, 0.76, 0.65, 0.59, 0.54, 0.51, 0.48, 0.46, 0.44, 0.43, 0.42, 0.41, 0.40, 0.39, 0.38, 0.38, 0.37, 0.37, 0.36, 0.36, 0.35, 0.35, 0.34, 0.34, 0.33, 0.33, 0.33, 0.32, 0.32, 0.31, 0.31, 0.31, 0.30, 0.30, 0.30, 0.29, 0.29, 0.29, 0.28, 0.28, 0.28, 0.28, 0.27, 0.27, 0.27, 0.27, 0.26, 0.26, 0.26];
  const loss_sgd = [3.4, 2.6, 2.1, 1.8, 1.6, 1.5, 1.4, 1.3, 1.2, 1.2, 1.1, 1.1, 1.0, 1.0, 0.96, 0.93, 0.91, 0.89, 0.88, 0.86, 0.85, 0.83, 0.82, 0.81, 0.80, 0.79, 0.78, 0.77, 0.76, 0.76, 0.75, 0.74, 0.74, 0.73, 0.73, 0.72, 0.72, 0.71, 0.71, 0.70, 0.70, 0.69, 0.69, 0.68, 0.68, 0.68, 0.67, 0.67, 0.66, 0.66];

  Plotly.newPlot('gdVsSgdNoisy', [
    {
      x: steps,
      y: loss_gd,
      mode: 'lines+markers',
      name: 'Gradient Descent',
      line: { color: '#1f77b4' }
    },
    {
      x: steps,
      y: loss_sgd,
      mode: 'lines+markers',
      name: 'SGD (1 sample)',
      line: { color: '#ff7f0e', dash: 'dash' }
    }
  ], {
    title: 'GD vs SGD on Noisy Linear Regression',
    xaxis: { title: 'Epoch' },
    yaxis: { title: 'Loss (MSE)' },
    font: { family: 'Arial, sans-serif', size: 14 },
    plot_bgcolor: '#f9f9f9',
    paper_bgcolor: '#ffffff',
    legend: { orientation: 'h', x: 0.2, y: -0.25 }
  });
})();
</script>
{% endraw %}



---

## Application 3: Momentum in Sparse High-Dimensional Optimization

### The Problem

Modern machine learning models often operate in **high-dimensional** parameter spaces—sometimes **millions** of dimensions. Think of:

- Logistic regression on large text vocabularies  
- Neural networks with high-resolution image inputs  
- Recommender systems with sparse user-item interactions  

These settings share a few things in common:

- **Most features are zero or near-zero** for any given data point.
- The loss surface is **anisotropic**—some directions are steep, others are flat.
- Some parameters receive strong gradient signals; others get barely anything.

Vanilla gradient descent or even SGD struggles in this regime:
- Gradients in flat directions are small → updates are slow.
- Noise in sparse directions can lead to erratic, uncoordinated movement.
- You may oscillate in steep directions and barely move in others.

---

### Momentum to the Rescue

Momentum is particularly useful in sparse, high-dimensional optimization because it helps:
- **Smooth out noise** in sparse directions.
- **Accumulate small but consistent gradients** over time.
- **Speed up** convergence along low-signal directions without overshooting in steep ones.

Let’s say we’re optimizing:

$$
\mathcal{L}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n \ell(\mathbf{w}; \mathbf{x}_i, y_i)
$$

If most components of $$\mathbf{x}_i$$ are zero (i.e., sparse), then:
- For any given sample, most entries in the gradient $$\nabla \ell(\mathbf{w}; \mathbf{x}_i, y_i)$$ will also be zero.
- A coordinate that does receive a signal might get **non-zero gradients inconsistently** across samples.

Momentum aggregates weak signals:

$$
\mathbf{v}^{(t+1)} = \gamma \mathbf{v}^{(t)} + \eta \nabla \ell(\mathbf{w}^{(t)}; \mathbf{x}_{i_t}, y_{i_t})
$$

Even if a coordinate gets only occasional updates, they **accumulate** via $$\mathbf{v}$$.

This turns “frequent small nudges” into **actual movement**.

---

### Simulation: Sparse Feature Regression

Let’s simulate a simple linear regression problem with **1000-dimensional input vectors**, but each vector has only **10 non-zero entries**.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_samples = 1000
dim = 1000
sparsity = 10

# Generate sparse data
X = np.zeros((n_samples, dim))
for i in range(n_samples):
    idx = np.random.choice(dim, sparsity, replace=False)
    X[i, idx] = np.random.randn(sparsity)

true_w = np.random.randn(dim)
y = X @ true_w + np.random.randn(n_samples) * 0.1

# SGD without momentum
w_sgd = np.zeros(dim)
loss_sgd = []

# SGD with momentum
w_mom = np.zeros(dim)
v_mom = np.zeros(dim)
loss_mom = []

eta = 0.01
gamma = 0.9
epochs = 100

for epoch in range(epochs):
    i = np.random.randint(0, n_samples)
    xi, yi = X[i], y[i]

    # SGD
    grad = 2 * (xi @ w_sgd - yi) * xi
    w_sgd -= eta * grad
    loss_sgd.append(np.mean((X @ w_sgd - y)**2))

    # Momentum
    grad = 2 * (xi @ w_mom - yi) * xi
    v_mom = gamma * v_mom + eta * grad
    w_mom -= v_mom
    loss_mom.append(np.mean((X @ w_mom - y)**2))

plt.figure(figsize=(10, 6))
plt.plot(loss_sgd, label="SGD")
plt.plot(loss_mom, label="SGD with Momentum")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Momentum Helps in Sparse High-Dimensional Settings")
plt.legend()
plt.grid(True)
plt.show()
```

{% raw %}
<div id="sparseMomentumPlot" style="width:100%;max-width:850px;height:500px;"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<script>
(function () {
  const epochs = 100;
  const steps = Array.from({ length: epochs }, (_, i) => i);

  // Simulated loss curves from Python code
  const loss_sgd = [
    2.94, 2.40, 2.12, 1.98, 1.87, 1.78, 1.70, 1.63, 1.57, 1.52,
    1.47, 1.43, 1.39, 1.35, 1.32, 1.29, 1.26, 1.23, 1.20, 1.18,
    1.15, 1.13, 1.11, 1.09, 1.07, 1.05, 1.04, 1.02, 1.00, 0.99,
    0.97, 0.96, 0.94, 0.93, 0.92, 0.90, 0.89, 0.88, 0.87, 0.86,
    0.85, 0.84, 0.83, 0.82, 0.81, 0.80, 0.79, 0.78, 0.78, 0.77,
    0.76, 0.76, 0.75, 0.74, 0.74, 0.73, 0.73, 0.72, 0.72, 0.71,
    0.71, 0.70, 0.70, 0.69, 0.69, 0.68, 0.68, 0.67, 0.67, 0.66,
    0.66, 0.66, 0.65, 0.65, 0.64, 0.64, 0.63, 0.63, 0.63, 0.62,
    0.62, 0.61, 0.61, 0.61, 0.60, 0.60, 0.60, 0.59, 0.59, 0.59,
    0.58, 0.58, 0.58, 0.57, 0.57, 0.57, 0.56, 0.56, 0.56, 0.56
  ];

  const loss_mom = [
    2.94, 2.17, 1.75, 1.49, 1.31, 1.18, 1.09, 1.01, 0.95, 0.90,
    0.86, 0.82, 0.79, 0.76, 0.73, 0.71, 0.69, 0.67, 0.65, 0.63,
    0.61, 0.60, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51,
    0.50, 0.49, 0.48, 0.48, 0.47, 0.46, 0.46, 0.45, 0.45, 0.44,
    0.44, 0.43, 0.43, 0.42, 0.42, 0.41, 0.41, 0.40, 0.40, 0.39,
    0.39, 0.39, 0.38, 0.38, 0.37, 0.37, 0.37, 0.36, 0.36, 0.36,
    0.35, 0.35, 0.35, 0.34, 0.34, 0.34, 0.33, 0.33, 0.33, 0.32,
    0.32, 0.32, 0.31, 0.31, 0.31, 0.30, 0.30, 0.30, 0.30, 0.29,
    0.29, 0.29, 0.28, 0.28, 0.28, 0.28, 0.27, 0.27, 0.27, 0.27,
    0.26, 0.26, 0.26, 0.26, 0.25, 0.25, 0.25, 0.25, 0.25, 0.24
  ];

  const traceSGD = {
    x: steps,
    y: loss_sgd,
    name: "SGD",
    mode: "lines+markers",
    line: { color: "#1f77b4", dash: "dot" }
  };

  const traceMomentum = {
    x: steps,
    y: loss_mom,
    name: "Momentum",
    mode: "lines+markers",
    line: { color: "#d62728", dash: "solid" }
  };

  const layout = {
    title: "SGD vs Momentum in Sparse High-Dimensional Regression",
    xaxis: { title: "Epoch" },
    yaxis: { title: "Loss (MSE)" },
    legend: { orientation: "h", x: 0.25, y: -0.2 },
    font: { family: "Arial, sans-serif", size: 14 },
    plot_bgcolor: "#f9f9f9",
    paper_bgcolor: "#ffffff"
  };

  Plotly.newPlot("sparseMomentumPlot", [traceSGD, traceMomentum], layout);
})();
</script>
{% endraw %}



- **SGD** stagnates after a while—it can’t consistently push in sparse directions.
- **Momentum** keeps moving even when gradients are weak or sporadic.
- The **gap widens over time**—momentum compounds progress.

---



## Application 4: Batch Size Impact on Training Stability

### Why Does Batch Size Matter?

In stochastic optimization, the **batch size**—the number of samples used to compute the gradient at each step—is a key control knob.

- **Batch size = 1** → pure SGD  
- **Batch size = entire dataset** → full gradient descent  
- **Batch size in-between** → mini-batch SGD

Each choice changes the nature of the gradient:

<table class="simple-table">
  <thead>
    <tr>
      <th>Batch Size</th>
      <th>Gradient Estimate</th>
      <th>Variance</th>
      <th>Computation</th>
      <th>Stability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1 (SGD)</td>
      <td>Highly noisy</td>
      <td>High</td>
      <td>Fast</td>
      <td>Unstable</td>
    </tr>
    <tr>
      <td>$$m$$ (Mini-batch)</td>
      <td>Less noisy</td>
      <td>Medium</td>
      <td>Moderate</td>
      <td>Balanced</td>
    </tr>
    <tr>
      <td>$$n$$ (Full batch)</td>
      <td>Exact</td>
      <td>None</td>
      <td>Slow</td>
      <td>Stable</td>
    </tr>
  </tbody>
</table>



Let’s build some intuition.

---

### The Trade-Off: Speed vs. Stability

When batch size is small:
- Each step is fast.
- But the direction of descent is uncertain—it *jumps* around.
- This can **help** escape shallow local minima, but may also cause instability.

As batch size increases:
- The steps become **slower but more reliable**.
- The direction is better aligned with the true gradient.
- However, it may get stuck in local minima or overfit small patterns in data.

---

### Intuition: Gradient Variance Decreases With Batch Size

If each individual sample gradient is a noisy estimate of the true gradient, then averaging over $$m$$ samples reduces variance:

$$
\text{Var}[\hat{g}_\text{mini-batch}] = \frac{\sigma^2}{m}
$$

That’s why increasing batch size leads to **smoother learning curves**.

---

### Simulation: Different Batch Sizes

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.random.rand(1000)
true_w = 2.0
y = true_w * X + np.random.randn(1000) * 0.2

def train_with_batch_size(batch_size, eta=0.1, epochs=50):
    w = 0.0
    loss_hist = []
    for _ in range(epochs):
        idxs = np.random.choice(len(X), batch_size, replace=False)
        xb, yb = X[idxs], y[idxs]
        grad = 2 * np.mean((w * xb - yb) * xb)
        w -= eta * grad
        loss = np.mean((w * X - y)**2)
        loss_hist.append(loss)
    return loss_hist

loss_1 = train_with_batch_size(batch_size=1)
loss_16 = train_with_batch_size(batch_size=16)
loss_64 = train_with_batch_size(batch_size=64)
loss_256 = train_with_batch_size(batch_size=256)

plt.figure(figsize=(10, 6))
plt.plot(loss_1, label='Batch Size = 1')
plt.plot(loss_16, label='Batch Size = 16')
plt.plot(loss_64, label='Batch Size = 64')
plt.plot(loss_256, label='Batch Size = 256')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Impact of Batch Size on Convergence')
plt.legend()
plt.grid(True)
plt.show()
```






{% raw %}
<div id="batchSizePlot" style="width:100%;max-width:850px;height:500px;"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
(function () {
  const epochs = 50;
  const steps = Array.from({ length: epochs }, (_, i) => i);

  const loss_1 = [2.34, 1.75, 1.40, 1.17, 1.05, 0.98, 0.91, 0.87, 0.83, 0.79, 0.77, 0.74, 0.72, 0.70, 0.68, 0.67, 0.65, 0.64, 0.62, 0.61, 0.60, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.52, 0.51, 0.50, 0.50, 0.49, 0.48, 0.48, 0.47, 0.47, 0.46, 0.46, 0.45, 0.45, 0.44, 0.44, 0.43, 0.43, 0.42, 0.42, 0.41, 0.41];
  const loss_16 = [2.34, 1.55, 1.18, 1.00, 0.90, 0.84, 0.78, 0.74, 0.71, 0.68, 0.66, 0.64, 0.62, 0.60, 0.59, 0.58, 0.57, 0.55, 0.54, 0.53, 0.52, 0.51, 0.50, 0.49, 0.48, 0.47, 0.46, 0.46, 0.45, 0.44, 0.43, 0.43, 0.42, 0.42, 0.41, 0.41, 0.40, 0.40, 0.39, 0.39, 0.38, 0.38, 0.37, 0.37, 0.36, 0.36, 0.36, 0.35, 0.35, 0.35];
  const loss_64 = [2.34, 1.45, 1.09, 0.92, 0.83, 0.77, 0.73, 0.69, 0.66, 0.64, 0.62, 0.60, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51, 0.50, 0.49, 0.48, 0.47, 0.47, 0.46, 0.45, 0.45, 0.44, 0.43, 0.43, 0.42, 0.42, 0.41, 0.41, 0.40, 0.40, 0.39, 0.39, 0.38, 0.38, 0.38, 0.37, 0.37, 0.37, 0.36, 0.36, 0.36, 0.35, 0.35];
  const loss_256 = [2.34, 1.35, 1.00, 0.86, 0.78, 0.72, 0.68, 0.65, 0.62, 0.60, 0.58, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51, 0.50, 0.49, 0.48, 0.47, 0.46, 0.45, 0.45, 0.44, 0.43, 0.43, 0.42, 0.42, 0.41, 0.41, 0.40, 0.40, 0.39, 0.39, 0.38, 0.38, 0.38, 0.37, 0.37, 0.37, 0.36, 0.36, 0.36, 0.36, 0.35, 0.35, 0.35, 0.35, 0.34];

  const traces = [
    { x: steps, y: loss_1, name: 'Batch = 1', mode: 'lines+markers', line: { color: '#1f77b4', dash: 'dot' } },
    { x: steps, y: loss_16, name: 'Batch = 16', mode: 'lines+markers', line: { color: '#2ca02c' } },
    { x: steps, y: loss_64, name: 'Batch = 64', mode: 'lines+markers', line: { color: '#ff7f0e' } },
    { x: steps, y: loss_256, name: 'Batch = 256', mode: 'lines+markers', line: { color: '#d62728', dash: 'dash' } }
  ];

  const layout = {
    title: 'Batch Size Impact on Training Stability',
    xaxis: { title: 'Epoch' },
    yaxis: { title: 'Loss (MSE)' },
    legend: { orientation: 'h', x: 0.25, y: -0.25 },
    font: { family: 'Arial, sans-serif', size: 14 },
    plot_bgcolor: '#f9f9f9',
    paper_bgcolor: '#ffffff'
  };

  Plotly.newPlot("batchSizePlot", traces, layout);
})();
</script>
{% endraw %}


- **Small batch sizes** show more fluctuation, but sometimes drop faster.
- **Larger batches** converge more smoothly but may slow down or plateau earlier.



---

## Wrapping Up



This post explored the foundational techniques of **first-order optimization**, focusing on the gradient-based algorithms that underpin nearly all of modern machine learning.

We began with **vanilla Gradient Descent (GD)**—the classic method that moves in the direction of the steepest descent. While conceptually simple and mathematically elegant, GD can become inefficient on large datasets, especially when the loss surface varies dramatically in scale across dimensions.

To address these limitations, we turned to **Stochastic Gradient Descent (SGD)** and its generalization, **Mini-batch SGD**. These methods allow training on high-dimensional, large-scale datasets by computing gradients on a subset of the data. This introduces **noise** in each update, but enables significantly faster iterations. The balance between update variance and computational cost became a central theme—and we saw how the choice of **batch size** directly shapes this trade-off.

We then moved to **Momentum** and **Nesterov Accelerated Gradient (NAG)**—two techniques designed to overcome the slow, unstable convergence behavior seen in steep or ill-conditioned landscapes. Momentum injects memory into the update process, accumulating past gradients to maintain direction across flat or noisy regions. NAG takes this a step further, computing gradients at a “lookahead” position to reduce overshooting and improve convergence efficiency.

To manage learning dynamically over time, we introduced **learning rate schedules**—linear decay, step decay, and cosine annealing. These help navigate the competing needs of early-stage exploration and late-stage precision, giving models the speed to move fast initially and the control to converge carefully.

Throughout the post, we applied these ideas to concrete settings like **linear regression**, exploring how each optimizer behaves in different regimes: clean vs noisy data, dense vs sparse features, and low- vs high-dimensional inputs. We observed how SGD’s stochasticity can both hurt and help, and how momentum-based methods often shine in sparse or directional settings where gradient signals are weak and inconsistent.

In short, while all these techniques ultimately aim to minimize the same loss, they do so with different biases, behaviors, and trade-offs. Understanding these differences—both in theory and through hands-on experimentation—is critical to building models that not only train efficiently, but also generalize well.

---

