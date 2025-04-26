---
layout: post
title: Taylor Expansions & Second-Order Thinking
date: 2023-07-15
description: Optimization & Multivariable Calculus 3 - Mathematics for Data Science
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

28-year-old Isaac Newton, yet to publish *Principia*, sat at Cambridge during a plague lockdown. He was playing with curves—not with data, but with functions. If a function wobbled at a point, could he describe it using just a few key ingredients? A slope, a curvature, maybe a few more? Could he peel back the function like layers of an onion?

Turns out, he could. What Newton figured out—and what Brook Taylor later formalized—was that nearly any "nice" function could be locally mimicked by a **polynomial**. And this idea of local approximation by peeling layer after layer of derivatives became what we now call the **Taylor series**.

Today, that simple idea forms the bedrock of *how we optimize*, *how we estimate uncertainty*, and even *how we simulate the unknown*. Whether you're tuning weights in a neural network or approximating posterior distributions in Gaussian processes, you're secretly invoking Taylor.

But here's the twist. Most of us stop at the first layer—the slope. That’s gradient descent. But the second layer? The bend? That’s where things get really smart.

That’s what this blog is about.

We’ll walk through:
- How Taylor approximations work for multivariable functions
- Why second-order information matters
- How Newton’s method uses the Hessian to take smarter steps
- Why Quasi-Newton methods like BFGS are a clever cheat
- Where this all shows up in data science—from logistic regression to Gaussian processes

If gradients tell you which way is downhill, **curvature tells you how fast the valley falls**. Welcome to second-order thinking.

---


## Taylor Series for Multivariable Functions

Let’s start simple. Imagine you’re standing on a hilly landscape. The height of the land at any point $$(x, y)$$ is given by some smooth function $$f(x, y)$$—maybe it’s temperature, maybe elevation, maybe loss in a neural network. Now, if you move a little in the $$x$$ direction or a little in the $$y$$ direction, how can you estimate how the function will change?

If you’ve taken calculus, your first instinct might be: “Take the gradient!”  
And you’d be right—for small movements, the **gradient** gives the direction and rate of steepest ascent. But what if you want more? What if you want to approximate the function’s actual *value* nearby? That’s where Taylor expansions shine.

---

### The Core Idea

In single-variable calculus, the Taylor series of a function $$f(x)$$ centered at $$x = a$$ looks like:

$$
f(x) = f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2 + \cdots
$$

It’s a polynomial that uses the function’s value, slope, curvature, and so on at $$a$$ to build a local approximation around $$a$$.

Now generalize this to $$f(\mathbf{x})$$ where $$\mathbf{x} \in \mathbb{R}^n$$. Let’s center the approximation at a point $$\mathbf{a} \in \mathbb{R}^n$$.

The Taylor series becomes:

$$
f(\mathbf{x}) = f(\mathbf{a}) + \nabla f(\mathbf{a})^\top (\mathbf{x} - \mathbf{a}) + \frac{1}{2} (\mathbf{x} - \mathbf{a})^\top H_f(\mathbf{a}) (\mathbf{x} - \mathbf{a}) + \cdots
$$

Let’s unpack this gently:

- $$f(\mathbf{a})$$ — the value at the center
- $$\nabla f(\mathbf{a})$$ — the **gradient**: a vector of partial derivatives, giving the slope in each direction
- $$H_f(\mathbf{a})$$ — the **Hessian matrix**: a matrix of second partial derivatives, telling you how the slope itself is changing

This expansion is essentially a *local 3D map* of your function:  
- The gradient tells you which way is steep.  
- The Hessian tells you how the steepness changes—whether you’re standing in a bowl or on a ridge.


---

### Numerical Example: Building the Taylor Expansion Step-by-Step

To make things concrete, let’s work through a real example. Consider the following 2D function:

$$
f(x, y) = 3 + 2x + y^2
$$

We’ll expand this function around the point $$(x_0, y_0) = (0, 0)$$.  
The goal is to approximate $$f(x, y)$$ near $$(0, 0)$$ using a Taylor series.

---

**Step 1: Zeroth-Order Term (Function Value at the Expansion Point)**

The zeroth-order approximation is simply the value of the function at the point $$(0, 0)$$:

$$
f(0, 0) = 3
$$

This gives us a **constant approximation**: near the origin, $$f(x, y) \approx 3$$ if we completely ignore changes in $$x$$ and $$y$$.

---

**Step 2: First-Order Term (Gradient)**

Next, we incorporate information about how the function *changes* — that is, its **gradient**:

The gradient vector of $$f(x, y)$$ is:

$$
\nabla f(x, y) = 
\begin{bmatrix}
\frac{\partial f}{\partial x} \\
\frac{\partial f}{\partial y}
\end{bmatrix}
=
\begin{bmatrix}
2 \\
2y
\end{bmatrix}
$$

Now, plug in the point $$(0, 0)$$:

$$
\nabla f(0, 0) = 
\begin{bmatrix}
2 \\
0
\end{bmatrix}
$$

This tells us that:
- The function increases at a rate of **2 units per unit increase** in $$x$$,
- and is **flat along the $$y$$-direction** at $$(0,0)$$.

Thus, adding the linear (first-order) correction:

$$
\text{Linear term} = \nabla f(0, 0)^T \cdot 
\begin{bmatrix}
x \\
y
\end{bmatrix}
= 2x
$$

---

**Step 3: Second-Order Term (Hessian)**

To capture the *curvature* — how the function bends — we need the **Hessian matrix**, made up of all second partial derivatives:

$$
H_f(x, y) = 
\begin{bmatrix}
\frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\
\frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2}
\end{bmatrix}
=
\begin{bmatrix}
0 & 0 \\
0 & 2
\end{bmatrix}
$$

- No curvature along $$x$$ (since $$\frac{\partial^2 f}{\partial x^2} = 0$$).
- Curvature of **2** along $$y$$ (since $$\frac{\partial^2 f}{\partial y^2} = 2$$).

The second-order correction is:

$$
\frac{1}{2} 
\begin{bmatrix}
x & y
\end{bmatrix}
H_f(0, 0)
\begin{bmatrix}
x \\
y
\end{bmatrix}
= \frac{1}{2} (2y^2) = y^2
$$

---


Summing up the constant, linear, and quadratic terms, we get the second-order Taylor expansion:

$$
f(x, y) \approx 3 + 2x + y^2
$$

Notice something special here:  
The Taylor expansion **exactly reconstructs** the original function!  
That’s because $$f(x, y)$$ is already a second-degree polynomial — there are no higher-order terms left out.

---


- The **constant** part ($$3$$) gives the base height.
- The **linear** part ($$2x$$) tilts the surface along the $$x$$-axis.
- The **quadratic** part ($$y^2$$) adds a gentle curve along the $$y$$-axis.

Thus, around $$(0,0)$$, the function surface looks like a tilted paraboloid.

---



### Let’s Visualize That

Here’s a small Python snippet to see this in action:

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the original function
def f(x, y):
    return 3 + 2*x + y**2

# Create a grid
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.9)
ax.set_title("Surface of $f(x, y) = 3 + 2x + y^2$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
plt.tight_layout()
plt.show()
```

{% raw %}
<div id="taylorSurface" style="width:100%; max-width:800px; height:500px; margin: 2rem auto;"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
  // Generate mesh grid
  const xRange = [], yRange = [], zData = [];
  const xSteps = 50, ySteps = 50;
  const xMin = -2, xMax = 2, yMin = -2, yMax = 2;

  for (let i = 0; i < xSteps; i++) {
    xRange.push(xMin + i * (xMax - xMin) / (xSteps - 1));
  }

  for (let j = 0; j < ySteps; j++) {
    yRange.push(yMin + j * (yMax - yMin) / (ySteps - 1));
  }

  for (let j = 0; j < ySteps; j++) {
    const row = [];
    for (let i = 0; i < xSteps; i++) {
      const x = xRange[i];
      const y = yRange[j];
      row.push(3 + 2 * x + y * y);
    }
    zData.push(row);
  }

  const surface = {
    type: 'surface',
    x: xRange,
    y: yRange,
    z: zData,
    colorscale: 'Viridis',
    showscale: false
  };

  const layout = {
    title: '$$f(x, y) = 3 + 2x + y^2$$',
    scene: {
      xaxis: { title: 'x' },
      yaxis: { title: 'y' },
      zaxis: { title: 'f(x, y)' }
    },
    margin: { l: 0, r: 0, t: 50, b: 0 }
  };

  Plotly.newPlot('taylorSurface', [surface], layout);
</script>
{% endraw %}



This surface is gently sloping in the $$x$$ direction (linear) and curving in the $$y$$ direction (quadratic). That’s literally what our Taylor expansion captures.

---

### Why This Matters in Data Science

Let’s connect this to machine learning. Suppose you’re minimizing a loss function $$L(\theta)$$ for some parameter vector $$\theta$$. Using only the gradient (first-order info) tells you *which direction* to go.

But imagine you could also feel the curvature—how quickly the loss flattens or steepens in each direction. That’s second-order info. That’s what optimizers like Newton’s method exploit.

You’re no longer blindly following the slope—you’re skiing the terrain with full awareness of its ridges and bowls.

---



## First-Order Approximation (Linearization)

You’ve probably heard it in calculus: “Approximate a function near a point using a tangent line.”  
But what does that look like when your function lives in two, ten, or even a thousand dimensions?

At the heart of gradient descent, backpropagation, and most machine learning optimizers lies a deceptively simple idea: if you zoom in close enough, any smooth function *looks flat*. Linearization is that first zoom.

### A Quick Refresher in One Dimension

In single-variable calculus, the first-order Taylor expansion of a function $$f(x)$$ near some point $$x_0$$ is:

$$
f(x) \approx f(x_0) + f'(x_0)(x - x_0)
$$

It’s just a line: the height at $$x_0$$ plus a slope times how far you’ve moved.

But in data science, we rarely work in one dimension. So let’s generalize.

---

### The Gradient Takes Over in Higher Dimensions

Let your input now be a vector: $$\mathbf{x} \in \mathbb{R}^n$$. We want to approximate a scalar-valued function $$f(\mathbf{x})$$ around a point $$\mathbf{a} \in \mathbb{R}^n$$.

The first-order approximation is:

$$
f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^\top (\mathbf{x} - \mathbf{a})
$$

Here’s what each term does:

- $$f(\mathbf{a})$$ — the current function value
- $$\nabla f(\mathbf{a})$$ — the gradient at that point (direction and steepness)
- $$(\mathbf{x} - \mathbf{a})$$ — your movement away from the point

This formula constructs a **tangent hyperplane**—a flat surface that hugs the function at $$\mathbf{a}$$ and shares its slope.

---

### A Concrete Numerical Example

Let’s consider the same function we’ve been working with:

$$
f(x, y) = 3 + 2x + y^2
$$

We’ll linearize it at $$(x_0, y_0) = (0, 0)$$.

- The function value is:  
  $$f(0, 0) = 3$$

- The gradient at $$(0, 0)$$ is:  
  $$
  \nabla f(0, 0) = 
  \begin{bmatrix}
  \frac{\partial f}{\partial x} \\
  \frac{\partial f}{\partial y}
  \end{bmatrix}
  =
  \begin{bmatrix}
  2 \\
  0
  \end{bmatrix}
  $$

So, the linear approximation becomes:

$$
f(x, y) \approx 3 + 2x
$$

Note that this approximation ignores the $$y^2$$ term—it’s purely a flat surface aligned with the local slope in the $$x$$ direction. That’s exactly what we expect from a linearization: it flattens the terrain to the gradient.

---

### Seeing the Flat and the Curved Together

Let’s visualize both the actual function and its linear approximation. This makes the idea of local flattening vivid—you’ll see a bowl-like surface (the true function) and a tilted sheet (the linear approximation) brushing against it at the base point.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Grid definition
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)

# True function: f(x, y) = 3 + 2x + y^2
Z_true = 3 + 2 * X + Y**2

# Linear approximation at (0, 0): f(x, y) ≈ 3 + 2x
Z_linear = 3 + 2 * X

# Plotting
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# True surface
ax.plot_surface(X, Y, Z_true, alpha=0.8, cmap='viridis', edgecolor='k', label='True Function')

# Linear approximation surface
ax.plot_surface(X, Y, Z_linear, alpha=0.5, cmap='hot', edgecolor='k')

ax.set_title('True Function vs Linear Approximation')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.tight_layout()
plt.show()
```
<div style="text-align: center; margin: 2rem 0;">
  <iframe src="/assets/plotly/linear_approx.html" width="100%" height="500" frameborder="0"></iframe>
</div>



---

### Why It Matters in Data Science

Every time we update model parameters using a gradient, we’re using this linear approximation:

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla L(\theta_{\text{old}})
$$

We’re assuming that the loss function $$L(\theta)$$ behaves like a flat plane locally. And that assumption is often good enough—especially when steps are small, and the function is smooth.

Linearization is the foundation of gradient-based learning.  
It’s not perfect, but it’s fast, efficient, and *good enough* to get us moving in the right direction.

---



## Second-Order Expansion with Hessians - From Slope to Curvature  

Let’s start with a thought experiment. Imagine you’re walking blindfolded on a landscape, and all you can do is feel the slope under your feet (that’s the gradient). Now imagine if you could also sense how the slope changes in every direction—whether you're in a bowl, on a ridge, or on a saddle.

That extra feeling? That’s what the **Hessian matrix** gives you. It’s the second derivative of a multivariable function, and it captures the **curvature**.

---

### Recalling the Taylor Series

The second-order Taylor expansion of a scalar-valued function $$f: \mathbb{R}^n \rightarrow \mathbb{R}$$ at point $$\mathbf{a}$$ is:

$$
f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^\top (\mathbf{x} - \mathbf{a}) + \frac{1}{2} (\mathbf{x} - \mathbf{a})^\top H_f(\mathbf{a}) (\mathbf{x} - \mathbf{a})
$$

Let’s unpack this:

- $$\nabla f(\mathbf{a})$$: the gradient, giving direction and steepness
- $$H_f(\mathbf{a})$$: the Hessian, a matrix of all second partial derivatives
- The quadratic term tells you how the slope is changing—it adjusts the first-order estimate to better hug the function's true shape.

---

### What the Hessian Matrix Tells You

For a function $$f(x, y)$$, the Hessian is:

$$
H_f(x, y) =
\begin{bmatrix}
\frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\
\frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2}
\end{bmatrix}
$$

It’s symmetric for well-behaved functions. The signs and values of the eigenvalues of this matrix tell you:
- If the point is a **local minimum** (both eigenvalues positive),
- A **local maximum** (both negative),
- Or a **saddle point** (mixed signs).

---

### Numerical Example: Revisiting Our Function

Consider:

$$
f(x, y) = 3 + 2x + y^2
$$

We expand around $$(0, 0)$$.

- $$f(0, 0) = 3$$  
- $$\nabla f(0, 0) = \begin{bmatrix} 2 \\ 0 \end{bmatrix}$$  
- $$H_f(0, 0) = \begin{bmatrix} 0 & 0 \\ 0 & 2 \end{bmatrix}$$

So the second-order approximation is:

$$
f(x, y) \approx 3 + 2x + \frac{1}{2} \cdot 2 \cdot y^2 = 3 + 2x + y^2
$$

Again, this recovers the original function because it *is* quadratic.

---

### Visualization: First-order vs Second-order Approximation

Let’s compare:
- The **true function**
- The **first-order approximation** (flat plane)
- The **second-order approximation** (curved surface using the Hessian)


```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)

Z_true = 3 + 2 * X + Y**2
Z_first_order = 3 + 2 * X
Z_second_order = 3 + 2 * X + Y**2

fig = plt.figure(figsize=(18, 6))

# True surface
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z_true, cmap='viridis', alpha=0.9)
ax1.set_title("True Function")

# First-order
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z_first_order, cmap='hot', alpha=0.9)
ax2.set_title("First-Order Approximation")

# Second-order
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, Z_second_order, cmap='cividis', alpha=0.9)
ax3.set_title("Second-Order Approximation")

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')

plt.tight_layout()
plt.show()
```



<div style="text-align: center; margin: 2rem 0;">
  <iframe src="/assets/plotly/second_order_approx.html" width="100%" height="520" frameborder="0"></iframe>
</div>

Notice how:
- The **first-order** surface quickly diverges from the true function as we move along $$y$$, because it has no notion of curvature.
- The **second-order** approximation aligns beautifully with the true surface near $$(0, 0)$$. It bends correctly in $$y$$, thanks to the second derivative.

---

### Where This Shows Up in Data Science

Hessians are the core ingredient in **second-order optimization**:

- **Newton’s Method**: Uses Hessians to adjust steps based on curvature.
- **Trust Region Methods**: Use second-order approximations to decide how far to move.
- **Uncertainty Estimation**: Curvature tells us about confidence in parameter estimates.
- **Gaussian Processes**: Taylor expansions help approximate kernels and predictive means.

When your function is well-behaved, the Hessian gives you a lens into its shape. You’re no longer stumbling down a slope—you’re navigating with a terrain map.

---


## Newton’s Method: Intuition and Math  
### Optimization with Second-Order Eyes

Most of machine learning uses **first-order** optimization—just gradients. You’re like a hiker standing on a slope, asking “which way is downhill?” But Newton’s method asks a better question:

> “Given how steep it is *and* how the slope is changing, where is the minimum likely to be?”

It doesn't just feel the slope—it reads the terrain.

---

### A One-Dimensional Warm-Up

In 1D, Newton’s method is a root-finding algorithm. It finds where a function $$f(x)$$ equals zero by approximating it with a tangent line and jumping to where that line hits the axis.

The update rule is:

$$
x_{\text{new}} = x - \frac{f(x)}{f'(x)}
$$

Now, suppose instead of solving $$f(x) = 0$$, we want to **minimize a function** $$f(x)$$. Then we apply Newton’s method to its derivative:

$$
x_{\text{new}} = x - \frac{f'(x)}{f''(x)}
$$

This is Newton’s method for optimization: take a step proportional to the slope, but scaled by the curvature.

---

### Now to Multiple Dimensions

Let $$f: \mathbb{R}^n \rightarrow \mathbb{R}$$ be a twice-differentiable function. Newton’s method for finding a **local minimum** of $$f(\mathbf{x})$$ uses the update rule:

$$
\mathbf{x}_{\text{new}} = \mathbf{x} - H_f(\mathbf{x})^{-1} \nabla f(\mathbf{x})
$$

Where:
- $$\nabla f(\mathbf{x})$$ is the gradient (first derivative)
- $$H_f(\mathbf{x})$$ is the Hessian (second derivative matrix)
- $$H^{-1} \nabla f$$ is called the **Newton direction**

So you’re still moving downhill, but instead of using a fixed step size, you “scale and twist” your direction using the Hessian.

---

### Why Is This Better Than Gradient Descent?

Gradient descent takes steps proportional to the slope:

$$
\mathbf{x}_{\text{new}} = \mathbf{x} - \eta \nabla f(\mathbf{x})
$$

But this can be **slow in poorly conditioned problems**, especially when the function curves steeply in one direction but not in another (like long narrow valleys).

Newton’s method automatically rescales the space so you take **balanced steps** in all directions, making convergence much faster—especially near the minimum.

---

### Numerical Example : $$f(x, y) = x^2 + 5y^2$$

Let’s apply Newton’s method to this function at the point $$(x, y) = (2.5, 2.5)$$.

#### 1. **Gradient:**

The gradient is:

$$
\nabla f(x, y) = \begin{bmatrix} 2x \\ 10y \end{bmatrix}
$$

So at $$(2.5, 2.5)$$:

$$
\nabla f(2.5, 2.5) = \begin{bmatrix} 5 \\ 25 \end{bmatrix}
$$

#### 2. **Hessian Matrix:**

Because this is a quadratic function, the second derivative is constant:

$$
H_f = \begin{bmatrix} 2 & 0 \\ 0 & 10 \end{bmatrix}
$$

#### 3. **Newton Step:**

We apply Newton’s update rule:

$$
\mathbf{x}_{\text{new}} = \mathbf{x} - H_f^{-1} \nabla f(\mathbf{x})
$$

So we compute:

$$
H_f^{-1} = \begin{bmatrix} \frac{1}{2} & 0 \\ 0 & \frac{1}{10} \end{bmatrix}
$$

Then:

$$
\mathbf{x}_{\text{new}} = 
\begin{bmatrix} 2.5 \\ 2.5 \end{bmatrix}
-
\begin{bmatrix} \frac{1}{2} & 0 \\ 0 & \frac{1}{10} \end{bmatrix}
\begin{bmatrix} 5 \\ 25 \end{bmatrix}
=
\begin{bmatrix} 2.5 \\ 2.5 \end{bmatrix}
-
\begin{bmatrix} 2.5 \\ 2.5 \end{bmatrix}
=
\begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$



We reach the **global minimum** at $$(0, 0)$$ in **one step**.

---


Let’s see both Gradient Descent and Newton’s Method take steps from the same starting point.


<div style="text-align: center; margin: 2rem 0;">
  <iframe src="/assets/plotly/newton_vs_gd_iterative.html" width="100%" height="560" frameborder="0"></iframe>
</div>


This visualization puts both algorithms in motion.

- **Gradient Descent** (red) takes many small, cautious steps—bouncing across the narrow valley before finally closing in.
- **Newton’s Method** (blue) leverages curvature to take an aggressive, *direct path* to the optimum. On a quadratic function like this, it can jump to the minimum in just **one or two steps**.

This plot reveals why Newton’s method is so powerful near minima—but also why it’s expensive: it needs to compute and invert the Hessian. On the flip side, gradient descent is slower but easy to compute, making it a workhorse for deep learning.


The takeaway is powerful: while gradient descent can zigzag slowly in narrow valleys, Newton’s method reshapes the problem to make direct, balanced steps. In smooth, convex problems, this can lead to **quadratic convergence**—that is, reaching the optimum *exponentially faster*.


---

## Quasi-Newton Methods: BFGS and L-BFGS  
### Learning Curvature Without Computing It

Imagine trying to navigate a mountainous terrain. Newton’s method hands you a full 3D model of the landscape—super accurate, but heavy and expensive to carry. Gradient descent gives you just a compass—cheap and light, but kind of clueless.  
**Quasi-Newton methods** build the map *as you go*, updating their understanding of the terrain based on your past steps and gradients.

It’s a clever compromise.

---

### Why We Need Them

Newton’s method requires calculating and inverting the Hessian:

$$
\mathbf{x}_{\text{new}} = \mathbf{x} - H_f^{-1} \nabla f(\mathbf{x})
$$

But in high-dimensional spaces:
- Computing the full Hessian is $$\mathcal{O}(n^2)$$
- Inverting it is $$\mathcal{O}(n^3)$$

That’s a showstopper when $$n$$ is in the thousands or millions—as in neural nets, large-scale logistic regression, or NLP models.

So what do we do?

We **approximate** the Hessian, or more precisely, its inverse. And we do it iteratively, using **only gradients**.

---

### BFGS: The Core Idea

BFGS stands for **Broyden-Fletcher-Goldfarb-Shanno**, named after the four mathematicians who independently developed it in the 1970s.

Here’s what it does:
- Keeps a running estimate of the **inverse Hessian**, $$H_k$$
- Uses gradient and position updates to refine that estimate

Let:
- $$\mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k$$ (change in position)
- $$\mathbf{y}_k = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)$$ (change in gradient)

Then BFGS updates its Hessian approximation via:

$$
H_{k+1} = \left( I - \frac{\mathbf{s}_k \mathbf{y}_k^\top}{\mathbf{y}_k^\top \mathbf{s}_k} \right) H_k \left( I - \frac{\mathbf{y}_k \mathbf{s}_k^\top}{\mathbf{y}_k^\top \mathbf{s}_k} \right)
+ \frac{\mathbf{s}_k \mathbf{s}_k^\top}{\mathbf{y}_k^\top \mathbf{s}_k}
$$

You never compute the real Hessian—but you slowly build a good estimate of its inverse.

---

### L-BFGS: Memory-Efficient BFGS

L-BFGS stands for **Limited-memory BFGS**. It’s the same algorithm in spirit, but instead of storing a full $$n \times n$$ matrix, it stores a handful of recent $$(\mathbf{s}_k, \mathbf{y}_k)$$ vectors and reconstructs the inverse Hessian approximation *on the fly*.

This reduces both **storage** and **computation**, making it ideal for large-scale optimization.

It’s used in practice everywhere: from **scikit-learn’s LogisticRegression** to **XGBoost**, **TensorFlow**, and **PyTorch's optim.LBFGS**.

---



## Application: Newton’s Method for Logistic Regression  

In logistic regression briefly, we model the probability of class 1 as:

$$
P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{x}^\top \boldsymbol{\beta}) = \frac{1}{1 + \exp(-\mathbf{x}^\top \boldsymbol{\beta})}
$$

We then estimate the parameters $$\boldsymbol{\beta}$$ by **maximum likelihood**, which is equivalent to minimizing the **negative log-likelihood (NLL)**:

$$
L(\boldsymbol{\beta}) = - \sum_{i=1}^{n} \left[ y_i \log \sigma(z_i) + (1 - y_i) \log (1 - \sigma(z_i)) \right], \quad z_i = \mathbf{x}_i^\top \boldsymbol{\beta}
$$

---

### Why Use Newton’s Method?

The loss function is **convex**, but has no closed-form solution. Gradient descent can work, but convergence can be slow.

Newton’s method, on the other hand, can converge in **very few iterations**—especially when the number of features is small or moderate—because it leverages second-order curvature from the Hessian.

---

### Derivatives We Need

Let’s derive what Newton needs.

Let $$\boldsymbol{\beta} \in \mathbb{R}^d$$ be the parameter vector, and $$X \in \mathbb{R}^{n \times d}$$ the design matrix.

#### 1. Gradient of the NLL

$$
\nabla L(\boldsymbol{\beta}) = X^\top (\sigma(X\boldsymbol{\beta}) - \mathbf{y})
$$

#### 2. Hessian of the NLL

$$
H = X^\top S X \quad \text{where } S \text{ is a diagonal matrix with } s_i = \sigma(z_i)(1 - \sigma(z_i))
$$

This is known as the **Fisher Information Matrix** in statistics.

---

### Newton’s Update Rule

Using the gradient and Hessian, the Newton-Raphson update is:

$$
\boldsymbol{\beta}_{\text{new}} = \boldsymbol{\beta} - (X^\top S X)^{-1} X^\top (\sigma(X\boldsymbol{\beta}) - \mathbf{y})
$$

---



### Numerical Example  
We’ll work with **2 data points**, **2 features**, and include a **bias term**, so we have:

- $$n = 2$$ examples  
- $$d = 2$$ features + 1 bias = 3 weights to estimate  
- Goal: fit logistic regression via Newton’s Method

---

#### Step 1: Define Inputs

Let’s say we have the following feature matrix $$X$$ (already including a bias column):

$$
X =
\begin{bmatrix}
1 & 1 & 2 \\
1 & 2 & 1
\end{bmatrix}
$$

Each row is an example:  
- First column = bias term  
- Second, third columns = actual features

Let the labels be:

$$
\mathbf{y} =
\begin{bmatrix}
1 \\
0
\end{bmatrix}
$$

Initialize weights:

$$
\boldsymbol{\beta}^{(0)} =
\begin{bmatrix}
0 \\
0 \\
0
\end{bmatrix}
$$

---

#### Step 2: Compute Predictions

We compute the linear predictor $$z = X \boldsymbol{\beta}$$:

Since $$\boldsymbol{\beta} = \mathbf{0}$$:

$$
z =
X \boldsymbol{\beta} =
\begin{bmatrix}
1 & 1 & 2 \\
1 & 2 & 1
\end{bmatrix}
\begin{bmatrix}
0 \\
0 \\
0
\end{bmatrix}
=
\begin{bmatrix}
0 \\
0
\end{bmatrix}
$$

Apply the **sigmoid** function:

$$
p = \sigma(z) =
\begin{bmatrix}
0.5 \\
0.5
\end{bmatrix}
$$

---

#### Step 3: Compute the Gradient

The gradient of the NLL is:

$$
\nabla L(\boldsymbol{\beta}) = X^\top (p - y)
$$

So:

$$
p - y =
\begin{bmatrix}
0.5 \\
0.5
\end{bmatrix}
-
\begin{bmatrix}
1 \\
0
\end{bmatrix}
=
\begin{bmatrix}
-0.5 \\
0.5
\end{bmatrix}
$$

Now:

$$
X^\top (p - y) =
\begin{bmatrix}
1 & 1 \\
1 & 2 \\
2 & 1
\end{bmatrix}
\begin{bmatrix}
-0.5 \\
0.5
\end{bmatrix}
=
\begin{bmatrix}
0 \\
0.5 \\
-0.5
\end{bmatrix}
$$

So the gradient is:

$$
\nabla L(\boldsymbol{\beta}) =
\begin{bmatrix}
0 \\
0.5 \\
-0.5
\end{bmatrix}
$$

---

#### Step 4: Compute the Hessian

The Hessian is:

$$
H = X^\top S X
$$

Where $$S$$ is a diagonal matrix with entries $$\sigma(z_i)(1 - \sigma(z_i))$$:

Since $$p_i = 0.5$$ for both examples:

$$
S = \text{diag}(0.25, 0.25)
$$

Now compute:

$$
H = X^\top S X =
\begin{bmatrix}
1 & 1 \\
1 & 2 \\
2 & 1
\end{bmatrix}
\begin{bmatrix}
0.25 & 0 \\
0 & 0.25
\end{bmatrix}
\begin{bmatrix}
1 & 1 & 2 \\
1 & 2 & 1
\end{bmatrix}
$$

Step-by-step:

1. Compute $$S X$$:

$$
S X =
\begin{bmatrix}
0.25 & 0 \\
0 & 0.25
\end{bmatrix}
\begin{bmatrix}
1 & 1 & 2 \\
1 & 2 & 1
\end{bmatrix}
=
\begin{bmatrix}
0.25 & 0.25 & 0.5 \\
0.25 & 0.5 & 0.25
\end{bmatrix}
$$

2. Then:

$$
H = X^\top (S X) =
\begin{bmatrix}
1 & 1 \\
1 & 2 \\
2 & 1
\end{bmatrix}
\begin{bmatrix}
0.25 & 0.25 & 0.5 \\
0.25 & 0.5 & 0.25
\end{bmatrix}
=
\begin{bmatrix}
0.5 & 0.75 & 0.75 \\
0.75 & 1.25 & 1.25 \\
0.75 & 1.25 & 1.25
\end{bmatrix}
$$

---

#### Step 5: Newton’s Update

Now compute:

$$
\boldsymbol{\beta}_{\text{new}} = \boldsymbol{\beta} - H^{-1} \nabla L(\boldsymbol{\beta})
$$

We already have:

- $$\nabla L(\boldsymbol{\beta}) = \begin{bmatrix} 0 \\ 0.5 \\ -0.5 \end{bmatrix}$$  
- $$H = \text{matrix above}$$

Use your favorite numerical method to compute $$H^{-1} \nabla L$$ (say, in Python), and subtract that from $$\boldsymbol{\beta}$$ to get the new weights.

---


Even with this tiny dataset, you can see:

- The gradient tells us how to shift $$\boldsymbol{\beta}$$.
- The Hessian accounts for *how confidently* we can shift—curvature.
- Newton’s update goes straight toward the optimal solution by combining both.

You’ll reach convergence in just a few steps—even without tuning any learning rate.

---



## Application: Second-Order Curvature in Loss Functions  
### How Loss Geometry Shapes Optimization

We’ve talked about Taylor expansions, gradients, Hessians, Newton’s method, and BFGS. But where do they *really* show up in the life of a data scientist?

Right here—inside the **loss functions** we minimize during model training.

If you’ve ever wondered why:
- Gradient descent seems to zigzag forever,
- Newton’s method converges in just one or two confident steps,
- Optimizers like **Adam**, **RMSProp**, and **L-BFGS** often feel smarter…

The answer is: **curvature**.

---

### Curvature in the Loss Landscape

Loss functions aren’t flat. They **curve**—and not always equally in all directions.

Take this example:

$$
f(\theta_1, \theta_2) = 10\theta_1^2 + 100\theta_2^2
$$

- The loss increases gradually along $$\theta_1$$
- But very sharply along $$\theta_2$$

The curvature is directional. Mathematically, that’s captured by the **Hessian matrix**:

$$
H_f(\boldsymbol{\theta}) = 
\begin{bmatrix}
20 & 0 \\
0 & 200
\end{bmatrix}
$$

Its eigenvalues tell us the function curves *ten times more* in $$\theta_2$$ than in $$\theta_1$$. And that’s a problem for naive optimizers.

---

### What This Means in Practice

If you’re using **gradient descent**:

- It takes steps proportional to the slope
- But steep curvature in one direction forces **tiny step sizes** to avoid instability
- Meanwhile, in flat directions, it barely moves

You either make slow progress, or you overshoot. This is what the infamous **zigzag path** looks like in a loss valley.

Now here’s where Newton’s method—or BFGS—shines.

---

### Numerical Example: How Newton Fixes the Problem

Let’s run the math for:

$$
f(\theta_1, \theta_2) = 10\theta_1^2 + 100\theta_2^2
$$

We start from:

$$
\boldsymbol{\theta}^{(0)} = 
\begin{bmatrix}
2 \\
2
\end{bmatrix}
$$

---

#### Step 1: Gradient

$$
\nabla f = 
\begin{bmatrix}
20\theta_1 \\
200\theta_2
\end{bmatrix}
\Rightarrow 
\begin{bmatrix}
40 \\
400
\end{bmatrix}
$$

---

#### Step 2: Gradient Descent Step (with $$\eta = 0.01$$)

$$
\boldsymbol{\theta}^{(1)} = 
\begin{bmatrix}
2 \\
2
\end{bmatrix}
-
0.01 \cdot
\begin{bmatrix}
40 \\
400
\end{bmatrix}
=
\begin{bmatrix}
1.6 \\
-2.0
\end{bmatrix}
$$

Gradient descent makes a very small improvement in $$\theta_1$$ and overshoots $$\theta_2$$—classic instability in curved loss functions.

---

#### Step 3: Newton’s Method Update

Hessian:

$$
H = 
\begin{bmatrix}
20 & 0 \\
0 & 200
\end{bmatrix}
\quad \Rightarrow \quad
H^{-1} = 
\begin{bmatrix}
\frac{1}{20} & 0 \\
0 & \frac{1}{200}
\end{bmatrix}
$$

Then:

$$
\boldsymbol{\theta}^{(1)} = 
\boldsymbol{\theta}^{(0)} -
H^{-1} \nabla f = 
\begin{bmatrix}
2 \\
2
\end{bmatrix}
-
\begin{bmatrix}
2 \\
2
\end{bmatrix}
=
\begin{bmatrix}
0 \\
0
\end{bmatrix}
$$

Newton jumps *directly* to the global minimum. No overshooting, no zigzagging.

---


This is the 3D loss surface:

<div style="text-align: center; margin: 2rem 0;">
  <iframe src="/assets/plotly/loss_curvature_surface.html" width="100%" height="600" frameborder="0"></iframe>
</div>

It shows a steep ridge in $$\theta_2$$ and a gently sloped direction in $$\theta_1$$.


This plot makes second-order curvature tangible:

- The **steep wall along $$\theta_2$$** is what causes instability in gradient descent.
- The **shallow bowl along $$\theta_1$$** is what causes slow progress.
- Newton’s method rescales the update using the Hessian, effectively *flattening* the valley so you can move efficiently in all directions.

This is why second-order methods—and modern optimizers that approximate them—work so well in practice.


---

## Application: Trust Region Methods  


Let’s say you have a nonlinear loss function, and you want to minimize it. So you do what we’ve already learned:

- Use a **Taylor expansion** to approximate it locally (gradient + Hessian)
- Solve the quadratic approximation (i.e., use Newton’s method)

Sounds solid, right?

But here's the catch: if your current point is far from the optimum, your local Taylor approximation might be **wildly wrong**. You might step into a region where the approximation is no longer valid—and instead of converging, you spiral out.

This is where **trust region methods** come in. They say:

> “I’ll trust this approximation—but only within a small, defined neighborhood.”

---

### The Core Idea

Instead of taking a Newton step blindly, you **constrain the step** to lie within a region around the current point—a **trust region**—where the Taylor approximation is expected to be reliable.

Formally, instead of solving:

$$
\min_{\Delta \boldsymbol{\theta}} \; f(\boldsymbol{\theta} + \Delta \boldsymbol{\theta})
$$

we solve the **trust region subproblem**:

$$
\min_{\Delta \boldsymbol{\theta}} \; Q(\Delta \boldsymbol{\theta}) = f(\boldsymbol{\theta}) + \nabla f^\top \Delta \boldsymbol{\theta} + \frac{1}{2} \Delta \boldsymbol{\theta}^\top H \Delta \boldsymbol{\theta}
$$

subject to:

$$
\| \Delta \boldsymbol{\theta} \| \leq \delta
$$

where:
- $$Q$$ is the second-order Taylor approximation (our “local guess”)
- $$\delta$$ is the **trust region radius**

---

### How It Works in Practice

1. Propose a step by solving the trust region subproblem.
2. Evaluate whether that step actually improved the true loss.
3. If yes → increase the radius (trust more).
4. If no → shrink the radius (trust less).

It’s a feedback loop that helps you:
- Take large steps in well-behaved regions
- Take cautious steps in tricky terrain

---


Imagine your loss surface is a warped landscape. You place a **bowl-shaped quadratic approximation** centered at your current point. But you only move **within a circle** around that point—because that’s where the bowl is most accurate.

Move too far? You might fall off a cliff. Trust region protects you from that.


Think of Newton’s method like asking someone directions and immediately following them *for 100 miles*.

Trust region is like saying:

> “Okay, I’ll follow your suggestion—but only for 1 mile. Then I’ll check my map again.”

It’s cautious, adaptive, and resilient. Exactly what you want when the terrain is curved and unpredictable.

---

### Where This Shows Up in Data Science

- **Scipy's `trust-constr` and `trust-ncg`** solvers are based on trust region logic.
- Used in **constrained optimization problems**, e.g., Lagrangian-based approaches.
- In **generalized linear models (GLMs)** with weird-shaped likelihoods.
- Anytime you’re solving nonlinear least squares problems robustly (e.g., curve fitting, Gaussian processes).

Some machine learning libraries like TensorFlow and PyTorch use **line search variants**, but trust region methods are still the gold standard in **scientific optimization libraries** like IPOPT, SNOPT, and SciPy’s `minimize()`.

---



## Application: Fitting Gaussian Processes Using Taylor Approximation  


Gaussian Processes (GPs) are powerful. They don't just fit a curve—they model **distributions over functions**. Given a set of inputs, GPs give you:
- A **mean prediction** at each point
- A **variance (uncertainty)** that reflects how confident the model is

They’re elegant, flexible, and probabilistically grounded. But… they're also expensive.

### The Bottleneck

GPs require computing:

$$
K^{-1} \mathbf{y}
$$

Where:
- $$K$$ is the kernel matrix (size $$n \times n$$),
- $$\mathbf{y}$$ is the vector of observations,
- $$K$$ depends on a **kernel function** (like RBF, Matern, etc.)

This is:
- $$\mathcal{O}(n^3)$$ for inversion
- Costly when $$n$$ is large

So researchers have explored ways to **approximate GPs**. And one of the smartest ideas?  
**Use a Taylor expansion of the kernel function**.

---

### Where Taylor Comes In



A popular kernel in GPs is the **Radial Basis Function (RBF)**:

$$
k(\mathbf{x}, \mathbf{x}') = \exp\left(-\frac{1}{2\ell^2} \| \mathbf{x} - \mathbf{x}' \|^2 \right)
$$

Now, if $$\mathbf{x}'$$ is close to $$\mathbf{x}$$, we can use a **second-order Taylor expansion** to approximate the exponential:

$$
\exp(-z) \approx 1 - z + \frac{z^2}{2}
$$

Set:

$$
z = \frac{1}{2\ell^2} \|\mathbf{x} - \mathbf{x}'\|^2
$$

Then:

$$
k(\mathbf{x}, \mathbf{x}') \approx 1 - \frac{1}{2\ell^2} \|\mathbf{x} - \mathbf{x}'\|^2 + \frac{1}{8\ell^4} \|\mathbf{x} - \mathbf{x}'\|^4
$$

So instead of computing full kernel evaluations, we can use **polynomial expansions** around each point—built from distances and derivatives.

---

### Why This Works

- GPs are fundamentally local: the kernel mostly cares about nearby points.
- Taylor expansions are *also* local—accurate near a point of expansion.
- So if we approximate kernel entries locally using Taylor expansions, we can:
  - **Reduce computation**
  - **Keep probabilistic interpretability**
  - **Accelerate inference on large datasets**

---

### Kernel Derivatives and the Hessian

Another bonus:  
When you expand a kernel, you get not just values, but **derivatives**:
- First derivatives → capture directional change
- Second derivatives → curvature information

This makes Taylor-approximated GPs useful in:
- **Active learning**
- **Bayesian optimization**
- **Physics-informed ML** (where derivative information is valuable)

---

### When This Is Used in Practice

You’ll find this trick in:
- **Sparse GPs** (e.g., FITC, DTC approximations)
- **Fast kernel interpolation** methods (e.g., KISS-GP)
- Papers that reduce GP complexity using **local approximations or basis expansions**
- Taylor-based **low-rank approximations** of $$K$$

In short: Taylor expansions give you a fast, localized way to build surrogate models that still respect uncertainty.

---

 
## Wrapping Up

Second-order methods extend optimization and approximation beyond gradient-based approaches by incorporating curvature through the Hessian or its approximations.

Across this blog, we examined:
- Taylor expansions for multivariable functions and their role in approximating complex landscapes.
- Newton’s method and quasi-Newton algorithms like BFGS, which use curvature to accelerate convergence.
- Trust region methods, which constrain updates to locally valid approximations.
- Gaussian Process regression, where Taylor approximations help construct scalable, uncertainty-aware models.

These techniques are foundational in numerical optimization, probabilistic modeling, and kernel-based learning. Understanding and applying second-order methods enables more stable, efficient, and interpretable solutions in modern data science workflows.

