---
layout: post
title: Optimization & Multivariable Calculus for Data Science - Multivariable Functions & Gradients
date: 2023-06-05
description: Optimization & Multivariable Calculus 1 - Mathematics for Machine Learning
tags: ml ai optimization calculus math
categories: machine-learning math math-for-ml
thumbnail: assets/img/optmcalc_banner.jpg
math: true
giscus_comments: true
related_posts: true
chart:
  plotly: true
toc:
  beginning: true
---



Before a neural network can learn, before a model loss can be minimized, and before gradients can flow through layers—there’s one quiet foundational idea beneath it all:

**Functions of many variables.**

In school, we learn functions like $$f(x) = x^2$$ or $$g(x) = \sin(x)$$—clean, one-dimensional stories. But data science doesn’t live in one dimension. Our models don’t take a single input. Our loss functions don’t live on a line.

Instead, machine learning is built on top of functions that eat whole vectors:

$$
f(x_1, x_2, \dots, x_n)
$$

If you’re training a model, the loss might depend on a thousand weights. It’s no longer a curve—it’s a surface. Or worse: a terrain in 10,000-dimensional space.

And to move toward a minimum, we need to ask questions like:

- Which direction should we move? (that’s the gradient)
- How fast should we move? (that’s the gradient magnitude)
- What does the terrain look like nearby? (enter the Hessian)

Let’s start from the ground up.

---



## Functions of Several Variables

Suppose you’re training a regression model with two features, $$x_1$$ and $$x_2$$. A typical loss function might look like:

$$
f(x_1, x_2) = x_1^2 + 4x_2^2
$$

This function takes two inputs and produces a single output—it maps $$\mathbb{R}^2 \rightarrow \mathbb{R}$$. Visually, this defines a smooth surface: a 3D bowl where the lowest point represents the minimum loss.

Let’s visualize what this surface looks like. Here’s how it behaves as we tweak $$x_1$$ and $$x_2$$:

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = X1**2 + 4*X2**2

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
plt.title('f(x1, x2) = x1² + 4x2²')
plt.show()
```

{% raw %}
<div id="surface-plot" style="width:100%; max-width:800px; height:500px;"></div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
  const x1 = [];
  const x2 = [];
  const steps = 50;
  const x1_min = -5, x1_max = 5;
  const x2_min = -5, x2_max = 5;

  for (let i = 0; i < steps; i++) {
    x1.push(x1_min + i * (x1_max - x1_min) / (steps - 1));
    x2.push(x2_min + i * (x2_max - x2_min) / (steps - 1));
  }

  const z = [];
  for (let i = 0; i < steps; i++) {
    const row = [];
    for (let j = 0; j < steps; j++) {
      const val = Math.pow(x1[j], 2) + 4 * Math.pow(x2[i], 2);
      row.push(val);
    }
    z.push(row);
  }

  const data = [{
    x: x1,
    y: x2,
    z: z,
    type: 'surface',
    colorscale: 'Viridis'
  }];

  const layout = {
    title: '3D Surface: f(x₁, x₂) = x₁² + 4x₂²',
    autosize: true,
    margin: { l: 0, r: 0, b: 0, t: 30 },
    scene: {
      xaxis: { title: 'x₁' },
      yaxis: { title: 'x₂' },
      zaxis: { title: 'f(x₁, x₂)' }
    }
  };

  Plotly.newPlot('surface-plot', data, layout);
</script>
{% endraw %}

What we’re seeing here is a quadratic bowl. Imagine dropping a ball onto the surface—it’ll roll toward the lowest point. The way it moves is shaped by the slope, direction, and curvature of the surface. That movement—how functions change as inputs vary—is what we’ll now study using derivatives.

Let’s start with partial derivatives.

---

## Partial vs. Total Derivatives

In single-variable calculus, we use the derivative:

$$
\frac{df}{dx}
$$

But for multivariable functions, we need to ask: which direction are we moving? Are we changing just one variable, or several?

### Partial Derivatives

A **partial derivative** measures the rate of change of a function with respect to *one* variable, while holding the others fixed.

For our function:

$$
f(x_1, x_2) = x_1^2 + 4x_2^2
$$

We compute:

$$
\frac{\partial f}{\partial x_1} = 2x_1
$$

$$
\frac{\partial f}{\partial x_2} = 8x_2
$$

These partial derivatives tell us how steep the function is in the $$x_1$$ and $$x_2$$ directions independently.

Later on, we’ll stack these partials into a **gradient vector** to find the overall direction of steepest ascent or descent.

---

### Total Derivatives

Partial derivatives are local—one direction at a time. But in many real situations, inputs change together.

Suppose:

$$
x_1(t) = t, \quad x_2(t) = t^2
$$

Then:

$$
f(t) = f(x_1(t), x_2(t)) = x_1^2 + 4x_2^2 = t^2 + 4t^4
$$

To compute how $$f$$ evolves with time:

$$
\frac{df}{dt} = \frac{\partial f}{\partial x_1} \cdot \frac{dx_1}{dt} + \frac{\partial f}{\partial x_2} \cdot \frac{dx_2}{dt}
$$

Compute each term:

- $$\frac{\partial f}{\partial x_1} = 2x_1 = 2t$$  
- $$\frac{\partial f}{\partial x_2} = 8x_2 = 8t^2$$  
- $$\frac{dx_1}{dt} = 1, \quad \frac{dx_2}{dt} = 2t$$

So:

$$
\frac{df}{dt} = 2t \cdot 1 + 8t^2 \cdot 2t = 2t + 16t^3
$$

This total derivative tells us the rate at which the function changes as both inputs evolve over time—this idea becomes crucial in understanding how loss behaves during optimization when multiple parameters are updated together.

---



## Gradient Vector: $$\nabla f(x)$$

Partial derivatives tell us how a function changes when we vary one input at a time. But in most real-world scenarios—especially in machine learning—we don’t move along just one axis. We move through parameter space, tweaking multiple variables simultaneously.

To navigate that space wisely, we need a direction that tells us where the function increases fastest. That direction is given by the **gradient vector**.

For a function $$f(x_1, x_2, \dots, x_n)$$, the gradient is:

$$
\nabla f(x) = \left[ \frac{\partial f}{\partial x_1},\ \frac{\partial f}{\partial x_2},\ \dots,\ \frac{\partial f}{\partial x_n} \right]^T
$$

This vector points in the direction of **steepest ascent**—where the function increases most rapidly. In optimization problems, especially in machine learning, we typically want to *minimize* a function, so we follow the **negative gradient** to move "downhill."

---

### Example

We have

$$
f(x_1, x_2) = x_1^2 + 4x_2^2
$$

The gradient is:

$$
\nabla f(x_1, x_2) = \begin{bmatrix}
2x_1 \\
8x_2
\end{bmatrix}
$$

The function increases faster along the $$x_2$$ axis than the $$x_1$$ axis, due to the steeper curvature from the 4 multiplier.

---

### Contour Visualization with Gradients

Let’s plot the contours of this function, and at each point, draw the gradient vector to show how the function behaves locally.

```python
import numpy as np
import matplotlib.pyplot as plt

# Grid definition
x1 = np.linspace(-4, 4, 40)
x2 = np.linspace(-4, 4, 40)
X1, X2 = np.meshgrid(x1, x2)

# Function and gradient
Z = X1**2 + 4 * X2**2
df_dx1 = 2 * X1
df_dx2 = 8 * X2
grad_magnitude = np.sqrt(df_dx1**2 + df_dx2**2)

# Plot setup
plt.figure(figsize=(10, 8))

# Contour plot
contours = plt.contour(X1, X2, Z, levels=25, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=9, fmt="%.0f")

# Gradient arrows with magnitude coloring
skip = (slice(None, None, 2), slice(None, None, 2))  # downsample for readability
quiver = plt.quiver(
    X1[skip], X2[skip], df_dx1[skip], df_dx2[skip],
    grad_magnitude[skip], cmap='inferno', scale=100, width=0.005
)
cb = plt.colorbar(quiver, label='Gradient Magnitude')

# Labels and aesthetics
plt.xlabel(r'$x_1$', fontsize=12)
plt.ylabel(r'$x_2$', fontsize=12)
plt.title(r'Contours and Gradient Field of $f(x_1, x_2) = x_1^2 + 4x_2^2$', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axis('equal')
plt.tight_layout()
plt.show()
```


<div style="text-align: center; margin: 2rem 0;">
  {% include figure.liquid 
      path="assets/img/optmcalc1_1.png" 
      class="img-fluid rounded shadow-sm" 
      loading="eager" 
      zoomable=true 
      alt="" 
  %}
</div>



What you see in this plot are the elliptical contours of the function and a field of red arrows, each representing the gradient at that point. Notice how:
- The arrows grow longer farther from the origin—indicating steeper slopes.
- They are perpendicular to the contour lines—because the gradient always points orthogonally to level sets.
- Near the minimum at the origin, the gradient vanishes—the function is flat.

This is not just mathematical trivia; it’s the very geometry of optimization. When we use **gradient descent**, we move against the gradient to descend the surface toward a minimum:

$$
\theta^{(t+1)} = \theta^{(t)} - \eta \cdot \nabla f(\theta^{(t)})
$$

The learning rate $$\eta$$ controls how far we step, but the direction is always determined by $$\nabla f$$.



---

## Directional Derivatives and Steepest Descent

We know that the gradient vector $$\nabla f(x)$$ tells us the direction in which a function increases most rapidly. But what if we’re not moving exactly in that direction?

In many optimization problems, especially in constrained settings, we may be forced to move in a specific direction. Naturally, the question arises:

**How fast does the function increase (or decrease) if I move in a particular direction?**

This is where **directional derivatives** come in.

---

### The Idea

Given a direction vector $$\mathbf{v}$$, the directional derivative of a function $$f(x)$$ at a point $$x$$ in the direction of $$\mathbf{v}$$ is defined as:

$$
D_{\mathbf{v}} f(x) = \nabla f(x) \cdot \mathbf{v}
$$

This is the **dot product** between the gradient and the direction vector.

Geometrically, this measures the projection of the gradient onto $$\mathbf{v}$$—that is, how much of the gradient’s influence actually acts along the direction $$\mathbf{v}$$.

---

### A Simple Case

Let’s reuse the same function:

$$
f(x_1, x_2) = x_1^2 + 4x_2^2
$$

We already know the gradient:

$$
\nabla f(x_1, x_2) = \begin{bmatrix} 2x_1 \\ 8x_2 \end{bmatrix}
$$

Now suppose we want to compute the directional derivative at point $$x = (1, 1)$$ in the direction:

$$
\mathbf{v} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

(That is, moving diagonally northeast in 2D.)

Compute the gradient at that point:

$$
\nabla f(1, 1) = \begin{bmatrix} 2 \\ 8 \end{bmatrix}
$$

Then:

$$
D_{\mathbf{v}} f(1, 1) = \nabla f(1,1) \cdot \mathbf{v} = 
\begin{bmatrix} 2 \\ 8 \end{bmatrix} \cdot 
\frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}
= \frac{1}{\sqrt{2}}(2 + 8) = \frac{10}{\sqrt{2}} \approx 7.07
$$

So the function increases at a rate of about 7.07 per unit movement in that direction.

---

### Why the Dot Product?

The dot product captures two things at once:
- The **alignment** between the gradient and your chosen direction: if you move perfectly in the gradient’s direction, you get the full steepness.
- The **magnitude** of the gradient: stronger gradients push the function value more rapidly.

This leads to a key observation:

> The directional derivative is **maximized** when $$\mathbf{v}$$ is perfectly aligned with $$\nabla f(x)$$.

This is the mathematical way of saying: *the gradient points in the direction of steepest ascent*.

---

### Steepest Descent in Practice

Since the gradient tells us where the function increases fastest, it follows that:

- $$\nabla f(x)$$ points **uphill**
- $$-\nabla f(x)$$ points **downhill**

In gradient descent, we take steps like:

$$
x^{(t+1)} = x^{(t)} - \eta \nabla f(x^{(t)})
$$

We’re not just picking any downhill path—we’re taking the **steepest descent**, the fastest way down the slope at each step.

This behavior becomes clearer when visualized.

---

### Visualization: Directional Derivatives and Descent Arrows

We’ll visualize a few arrows at a point, each representing the directional derivative in a particular direction.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define the function and its gradient
def f(x1, x2):
    return x1**2 + 4*x2**2

def grad_f(x1, x2):
    return np.array([2*x1, 8*x2])

# Create the grid
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

# Base point and gradient
x0 = np.array([1.0, 1.0])
gradient = grad_f(*x0)

# Normalize the gradient for display (only for arrow length—not math)
grad_unit = gradient / np.linalg.norm(gradient)

# Define direction vectors (unit vectors at 45° intervals)
angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
directions = np.array([[np.cos(a), np.sin(a)] for a in angles])
dot_products = directions @ gradient

# Scale down direction vectors for visualization
arrow_scale = 0.75
scaled_directions = directions * arrow_scale

# Plot
fig, ax = plt.subplots(figsize=(7, 7))
contours = ax.contour(X1, X2, Z, levels=20, cmap='viridis')
ax.clabel(contours, inline=True, fontsize=8)

# Gradient arrow (red)
ax.quiver(*x0, *(grad_unit * arrow_scale * 1.5), 
          angles='xy', scale_units='xy', scale=1, 
          color='red', label='Gradient (normalized)', linewidth=2)

# Directional arrows (gray)
for i, vec in enumerate(scaled_directions):
    dp = dot_products[i]
    ax.quiver(*x0, *vec, angles='xy', scale_units='xy', scale=1, 
              color='gray', alpha=0.6, linestyle='--')

# Annotations
ax.plot(*x0, 'ko')  # Base point
ax.text(x0[0]+0.1, x0[1]+0.1, 'x₀ = (1,1)', fontsize=10)

# Aesthetics
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Directional Derivatives and Gradient at $x_0 = (1,1)$')
ax.grid(True)
ax.legend()
ax.set_aspect('equal')

plt.tight_layout()
plt.show()
```


<div style="text-align: center; margin: 2rem 0;">
  {% include figure.liquid 
      path="assets/img/optmcalc1_2.png" 
      class="img-fluid rounded shadow-sm" 
      loading="eager" 
      zoomable=true 
      alt="" 
  %}
</div>

Each gray arrow here represents a possible direction. The longer the arrow, the faster the function increases in that direction. The red arrow is the gradient—always the longest, always the steepest.

---

Directional derivatives give us a full map of how a function changes in *any* direction. And when paired with the gradient, they form the compass and topographic lines that guide learning algorithms across loss landscapes.

In the next section, we’ll make this even more concrete: by visualizing **level sets**—those topographic lines—and seeing how they relate to gradients and optimization paths.


---

## Level Sets and Contour Plots for Loss Surfaces

If you’ve ever hiked with a topographic map, you’ve seen level sets in action. The contour lines on those maps connect points of equal elevation—walk along one, and you won’t go uphill or downhill. In multivariable calculus, we borrow the same idea. A **level set** of a function is the collection of all points where the function holds a constant value.

For a function $$f(x_1, x_2)$$, the level set corresponding to a constant $$c$$ is defined as:

$$
\{ (x_1, x_2) \in \mathbb{R}^2 \mid f(x_1, x_2) = c \}
$$

You’ve already seen these in action earlier in this blog—when we plotted the contour lines for our quadratic function. Those closed curves weren’t just for aesthetics. Each one traced out a level set of the function, showing all the places where the loss (or height, or energy) stays the same.

---

### Why Level Sets Matter in Optimization

Level sets give us a kind of "top-down view" of the function. Rather than staring at a 3D surface, we flatten it into slices—like MRI scans of a brain, each layer showing constant intensity. In optimization, this is incredibly helpful:

- We can visualize the **shape** of the objective function.
- We see whether it’s symmetric, skewed, or steep in some directions.
- And perhaps most importantly: we see how the **gradient interacts with the terrain**.

One key geometric insight:

> The gradient vector at a point is always **perpendicular to the level set** passing through that point.

This makes sense intuitively: if you're walking along a path where the function doesn’t change, then the direction in which it *does* change must be orthogonal to your path. The gradient points that way—cutting across contour lines at 90 degrees.

---

Let’s revisit the same function from before:

$$
f(x_1, x_2) = x_1^2 + 4x_2^2
$$

Its level sets are ellipses. Why ellipses? Because $$x_1$$ and $$x_2$$ contribute differently to the value of $$f$$—the $$x_2$$ term grows four times as fast, so the contours are "squeezed" along that axis.

Here’s a cleaner contour plot with the gradient field superimposed—now normalized so you can clearly see the relationship.

```python
import numpy as np
import matplotlib.pyplot as plt

# Function and gradient
def f(x1, x2):
    return x1**2 + 4*x2**2

def grad_f(x1, x2):
    return np.array([2*x1, 8*x2])

# Grid for plotting
x1 = np.linspace(-3, 3, 30)
x2 = np.linspace(-3, 3, 30)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

# Compute gradients
grad_x1 = 2 * X1
grad_x2 = 8 * X2

# Normalize gradients for consistent arrow size
norm = np.sqrt(grad_x1**2 + grad_x2**2)
grad_x1_norm = grad_x1 / norm
grad_x2_norm = grad_x2 / norm

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
contours = ax.contour(X1, X2, Z, levels=20, cmap='viridis')
ax.clabel(contours, inline=True, fontsize=8)

# Quiver (gradient field)
ax.quiver(X1, X2, grad_x1_norm, grad_x2_norm, color='red', alpha=0.7)

# Annotations
ax.set_title('Level Sets and Gradient Field of $f(x_1, x_2) = x_1^2 + 4x_2^2$')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.grid(True)
ax.set_aspect('equal')
plt.tight_layout()
plt.show()
```
<div style="text-align: center; margin: 2rem 0;">
  {% include figure.liquid 
      path="assets/img/optmcalc1_3.png" 
      class="img-fluid rounded shadow-sm" 
      loading="eager" 
      zoomable=true 
      alt="" 
  %}
</div>


- Each **contour line** corresponds to a constant value of the function. Think of them as elevation rings.
- The **red arrows** represent gradient directions—always pointing perpendicular to the contour lines.
- Where the lines are dense (along the $$x_2$$ axis), the slope is steep; where they’re sparse, it’s flatter.

This is more than geometry—it’s optimization logic. In gradient descent, your updates slice across level sets, driving the parameters downhill. The shape of these contours can tell you whether your optimizer might struggle: narrow valleys, plateaus, saddle points—they all show up in this view.

---




## Jacobian Matrix: Shape, Interpretation, and When It Matters

Until now, we’ve focused on scalar-valued functions—those that take a vector and return a number. Loss functions, for instance, are like that: they take in weights or inputs and return a single value to minimize.

But many functions we work with in machine learning go a step further. They take a vector and return another vector.

Think about:
- The output of a neural network layer  
- The softmax function over logits  
- A transformation from an input image to an embedding vector

These are **vector-valued functions**, written generally as:

$$
\mathbf{f} : \mathbb{R}^n \rightarrow \mathbb{R}^m
$$

So how do we talk about “how the function changes” now? There’s no longer just one slope to follow—there are multiple outputs, each changing with multiple inputs.

That’s where the **Jacobian matrix** comes in.

---

### What Is the Jacobian?

Suppose we have a function:

$$
\mathbf{f}(x) = \begin{bmatrix}
f_1(x_1, \dots, x_n) \\
f_2(x_1, \dots, x_n) \\
\vdots \\
f_m(x_1, \dots, x_n)
\end{bmatrix}
$$

The **Jacobian matrix** collects all the partial derivatives into an $$m \times n$$ matrix:

$$
J_{\mathbf{f}}(x) = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

Each row corresponds to the gradient of a single output component. Each column tells us how a particular input influences all the outputs.

---

### A Concrete Example

Let’s say:

$$
\mathbf{f}(x_1, x_2) =
\begin{bmatrix}
x_1^2 + x_2 \\
\sin(x_1) + x_1 x_2
\end{bmatrix}
$$

Then the Jacobian is:

$$
J_{\mathbf{f}}(x_1, x_2) =
\begin{bmatrix}
2x_1 & 1 \\
\cos(x_1) + x_2 & x_1
\end{bmatrix}
$$

At $$(x_1, x_2) = (1, 2)$$:

$$
J_{\mathbf{f}}(1, 2) \approx
\begin{bmatrix}
2 & 1 \\
2.54 & 1
\end{bmatrix}
$$

(using $$\cos(1) \approx 0.54$$)

This matrix tells us exactly how changes in $$x_1$$ and $$x_2$$ influence both components of $$\mathbf{f}$$ at that point.

---

### Geometric Interpretation — What the Jacobian *Really* Does

Let’s pause here—and really sit with this. Because this is where the Jacobian turns from a matrix of partial derivatives into something you can *see* and *feel*.

Imagine you’re standing at a point in $$\mathbb{R}^2$$—say, a point in the input space of a neural network layer. Now imagine nudging that point slightly—maybe one unit step in the $$x_1$$ direction, or maybe a diagonal step that mixes $$x_1$$ and $$x_2$$.

What happens to the output?

That’s the question the **Jacobian answers**.

---

#### Think of the Jacobian as a Local Transformer

If $$\mathbf{f} : \mathbb{R}^n \rightarrow \mathbb{R}^m$$ is differentiable, then around any point $$\mathbf{x}_0$$, it behaves *almost* like a linear map:

$$
\mathbf{f}(\mathbf{x}_0 + \Delta \mathbf{x}) \approx \mathbf{f}(\mathbf{x}_0) + J_{\mathbf{f}}(\mathbf{x}_0) \cdot \Delta \mathbf{x}
$$

So, locally, the function **is its Jacobian**. Tiny input changes get multiplied by the Jacobian to yield tiny output changes.

---

#### Visual Intuition in 2D

Back to our earlier example:

$$
\mathbf{f}(x_1, x_2) =
\begin{bmatrix}
x_1^2 + x_2 \\
\sin(x_1) + x_1 x_2
\end{bmatrix}
$$

At $$x = (1, 2)$$, we can ask:

- What happens if I take a small step in the $$x_1$$ direction?
- What happens if I take a small step in the $$x_2$$ direction?

The Jacobian tells us: here’s the resulting direction and magnitude in output space. And those output vectors become the **columns of the Jacobian**. That’s why the Jacobian captures not just the rate of change, but *the entire directional behavior*.

---

#### A Local Linear Map

You can think of the Jacobian as defining a new local basis in output space.

- A square in input space becomes a **parallelogram** in output space.
- A circle might get **sheared** or **stretched**.

If the Jacobian is diagonal with large entries → it stretches.  
If it's singular → it squashes input into a lower dimension.  
If it includes off-diagonal terms → it twists and shears.

---

#### Visualizing the Transformation

```python
import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x1, x2):
    f1 = x1**2 + x2
    f2 = np.sin(x1) + x1 * x2
    return np.array([f1, f2])

# Define Jacobian
def jacobian(x1, x2):
    df1_dx1 = 2 * x1
    df1_dx2 = 1
    df2_dx1 = np.cos(x1) + x2
    df2_dx2 = x1
    return np.array([
        [df1_dx1, df1_dx2],
        [df2_dx1, df2_dx2]
    ])

# Evaluate at point
x0 = np.array([1.0, 2.0])
J = jacobian(*x0)

# Unit vectors in input space
unit_x = np.array([1, 0])
unit_y = np.array([0, 1])

# Map through Jacobian
mapped_x = J @ unit_x
mapped_y = J @ unit_y

# Plot
fig, ax = plt.subplots(figsize=(7, 6))

# Original unit vectors (input space)
ax.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='blue', label='Input: $x_1$')
ax.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='green', label='Input: $x_2$')

# Jacobian-transformed vectors (output space)
ax.quiver(0, 0, mapped_x[0], mapped_x[1], angles='xy', scale_units='xy', scale=1,
          color='blue', alpha=0.5, label='Output: $J \\cdot x_1$')
ax.quiver(0, 0, mapped_y[0], mapped_y[1], angles='xy', scale_units='xy', scale=1,
          color='green', alpha=0.5, label='Output: $J \\cdot x_2$')

# Plot settings
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 4)
ax.set_aspect('equal')
ax.set_xlabel('Output $f_1$')
ax.set_ylabel('Output $f_2$')
ax.set_title('Jacobian as a Local Linear Map')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
```


<div style="text-align: center; margin: 2rem 0;">
  {% include figure.liquid 
      path="assets/img/optmcalc1_4.png" 
      class="img-fluid rounded shadow-sm" 
      loading="eager" 
      zoomable=true 
      alt="" 
  %}
</div>

This plot shows how small, axis-aligned steps in input space transform into new vectors in output space. The directions and magnitudes of these arrows encode the **local geometry** of $$\mathbf{f}$$ at that point.

---

#### Where the Jacobian Matters in ML

- In **backpropagation**, each layer’s transformation is tracked via Jacobians. Gradients pass backward by chaining Jacobians via the multivariable chain rule.
- In **normalizing flows**, the **Jacobian determinant** tracks how volume changes during transformation—a core step in computing likelihoods.
- In **coordinate transformations**, it accounts for how measures like length, area, or probability density distort under nonlinear mappings.

---

So to sum it up:

> The Jacobian is the best linear approximation to a function at a point. It doesn’t just say *how fast* something changes—it tells you *in what direction, how much, and with what distortion*. It’s the local shape of the function—captured in a matrix.

When linear isn’t enough—when curvature, twists, and second-order behavior begin to matter—we need something more powerful.

Let’s now explore the Hessian matrix, and see how the landscape bends.



---

## Hessian Matrix: Curvature, Convexity, and Second-Order Conditions

The gradient tells us how steeply a function rises or falls in different directions. But that’s only part of the story. It tells you the slope—but not how the slope is *changing*.

What if the function curves gently in one direction but sharply in another? What if the slope is zero, but you're at a saddle point, not a minimum?

To answer these questions, we need second derivatives. And in multivariable calculus, they come packed into a matrix: the **Hessian**.

---

### What Is the Hessian?

Let’s start with a scalar-valued function:

$$
f: \mathbb{R}^n \rightarrow \mathbb{R}
$$

The **Hessian matrix** of $$f$$ at a point $$x$$ is the matrix of all second-order partial derivatives:

$$
H_f(x) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
$$

Each entry in the matrix tells you how one slope is changing with respect to another variable.

In short:
- Diagonal entries → curvature along coordinate axes
- Off-diagonal entries → interaction between variables (e.g., how $$x_1$$ affects the rate of change along $$x_2$$)

---

### What About Vector-Valued Functions?

So far, we’ve assumed that our function maps from $$\mathbb{R}^n$$ to $$\mathbb{R}$$—a single output. But in many machine learning settings, especially in neural nets and transformation models, our functions are **vector-valued**:

$$
\mathbf{f} : \mathbb{R}^n \rightarrow \mathbb{R}^m
$$

When $$m = 1$$, we can use the Hessian directly.

But when $$m > 1$$, things get more complex. There's **no single Hessian matrix**—instead, each output dimension can have its own Hessian.

So the second-order derivative of a vector-valued function becomes a **rank-3 tensor**:

- For each output component $$f_i$$, you can compute a Hessian:  
  $$H_{f_i} \in \mathbb{R}^{n \times n}$$
- The full second-order derivative of $$\mathbf{f}$$ is then a stack of $$m$$ Hessian matrices:
  
  $$
  \frac{\partial^2 \mathbf{f}}{\partial \mathbf{x}^2} \in \mathbb{R}^{m \times n \times n}
  $$

This shows up in advanced applications like:
- Second-order backpropagation  
- Differential geometry  
- Optimization with vector-valued objectives (e.g., multi-task learning)

But in most ML workflows, we apply the Hessian primarily to **scalar loss functions**—even when they come from a vector-valued prediction model.

---

### Example: A Classic Quadratic Bowl

Let’s return to a familiar surface:

$$
f(x_1, x_2) = x_1^2 + 4x_2^2
$$

Compute the partials:

- $$\frac{\partial f}{\partial x_1} = 2x_1$$, $$\frac{\partial^2 f}{\partial x_1^2} = 2$$  
- $$\frac{\partial f}{\partial x_2} = 8x_2$$, $$\frac{\partial^2 f}{\partial x_2^2} = 8$$  
- Mixed partials: $$\frac{\partial^2 f}{\partial x_1 \partial x_2} = 0$$

So the Hessian is:

$$
H_f(x) = \begin{bmatrix}
2 & 0 \\
0 & 8
\end{bmatrix}
$$

This tells us something immediately about the shape of the function: it curves **more steeply along the $$x_2$$ direction** than along $$x_1$$.

---

### What It Means for a Function to "Curve More Steeply"

Let’s unpack that visually and intuitively.

The second derivative with respect to $$x_1$$ is 2, and with respect to $$x_2$$ is 8. That means:

> If you walk in the $$x_1$$ direction, the slope increases gradually.  
> But if you walk in the $$x_2$$ direction, the slope rises much faster.  
> The function is bending upward **four times faster** along $$x_2$$.

And that matches what we saw earlier in the contour plots:
- The level curves are **ellipses**, stretched wide along $$x_1$$, squished along $$x_2$$.
- Contour lines are farther apart where the function grows slowly, and closer together where the surface bends sharply.
- So the Hessian, through its entries, is encoding the very curvature we visualized before.

---

### Interpreting the Hessian

#### 1. **Curvature**
The Hessian tells you the local curvature of the function. If you think of $$f$$ as a landscape, the Hessian tells you if you're standing on a peak, a pit, a ridge, or a saddle.

#### 2. **Convexity**
The Hessian also tells you whether the function is convex or not.

- If $$H_f(x)$$ is **positive definite** (all eigenvalues > 0), the function is strictly convex.
- If it’s **indefinite** (some eigenvalues negative), you're in a saddle region.
- If it’s **negative definite**, you're sitting on a hilltop.

So convex optimization boils down to checking the **eigenvalues of the Hessian**.

---

### Geometric Visualization: Contours + Curvature

Let’s revisit the same quadratic surface, now with the Hessian's influence embedded in the shape.

```python
import numpy as np
import matplotlib.pyplot as plt

# Quadratic function and constant Hessian
def f(x1, x2):
    return x1**2 + 4*x2**2

def hessian():
    return np.array([[2, 0], [0, 8]])

# Grid for contour plot
x1 = np.linspace(-2.5, 2.5, 400)
x2 = np.linspace(-2.5, 2.5, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

# Compute Hessian eigendecomposition
H = hessian()
eigvals, eigvecs = np.linalg.eig(H)

# Set up plot
fig, ax = plt.subplots(figsize=(7, 7))
contours = ax.contour(X1, X2, Z, levels=40, cmap='viridis')
ax.clabel(contours, inline=True, fontsize=8)

# Curvature vectors from eigenvectors
origin = np.array([0, 0])
colors = ['crimson', 'darkorange']
labels = [f'Curvature dir 1 ($\\lambda$={eigvals[0]:.0f})',
          f'Curvature dir 2 ($\\lambda$={eigvals[1]:.0f})']
display_length = 2.2

for i in range(2):
    direction = eigvecs[:, i] / np.linalg.norm(eigvecs[:, i])
    vec = direction * display_length
    ax.quiver(*origin, vec[0], vec[1],
              angles='xy', scale_units='xy', scale=1,
              color=colors[i], width=0.017,
              headwidth=8, headlength=10,
              label=labels[i])

# Add boxed text for annotation labels
ax.text(2.3, -0.2, 'Gentler curvature',
        color='black', fontsize=10, va='center',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

ax.text(0.4, 1.6, 'Steeper curvature',
        color='black', fontsize=10, va='center', rotation=90,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

# Final plot aesthetics
ax.set_title('Contours and Principal Curvature Directions of $f(x_1, x_2) = x_1^2 + 4x_2^2$')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.grid(True)
ax.set_aspect('equal')
ax.legend()
plt.tight_layout()
plt.show()
```

<div style="text-align: center; margin: 2rem 0;">
  {% include figure.liquid 
      path="assets/img/optmcalc1_5.png" 
      class="img-fluid rounded shadow-sm" 
      loading="eager" 
      zoomable=true 
      alt="" 
  %}
</div>


This plot shows how the **Hessian’s eigenvectors** correspond to the **principal curvature directions**—and their eigenvalues tell us how steeply the function bends in each of those directions.

---

### Why the Hessian Matters in Machine Learning

- **Newton’s method**: Optimization algorithms like Newton’s method use the Hessian to take curvature-aware steps.
- **Second-order optimization**: When the gradient alone isn’t enough, the Hessian can help escape saddle points or accelerate convergence.
- **Uncertainty estimation**: In Bayesian learning, the inverse of the Hessian approximates the posterior variance near a maximum likelihood solution.
- **Vector-valued systems**: Even though we typically apply Hessians to scalar losses, many of those losses come from vector-valued models—so understanding when and how to generalize second-order derivatives matters in deep learning and differential programming.

---

The gradient points us toward lower ground. But the Hessian tells us what the ground *feels* like—whether it’s sloping evenly, or bending underfoot, or about to flip from valley to ridge.

And in deep learning, understanding this second-order behavior can be the difference between fast, stable convergence and slow, unstable wandering.

---

## Applications

Perfect. Now that we've built strong geometric and mathematical intuition for gradients, directional derivatives, level sets, Jacobians, and Hessians, it's time to put that machinery to work in the kinds of problems data scientists and ML engineers actually face.

Let’s now transition into real-world **applications** of these concepts—starting with the behavior of gradients on **loss landscapes**.

---

### Interpreting Gradients in Loss Landscapes

In machine learning, the loss function plays the role of the landscape. Each point in parameter space corresponds to a particular set of model weights, and the loss tells us how well those weights perform.

And just like in our geometric visualizations earlier, the **gradient** tells us which way to move in that space to reduce error.

But here’s the twist: unlike a clean quadratic bowl, real-world loss surfaces aren’t always smooth, convex, or well-behaved.

They're messy. They have flat regions, sharp cliffs, narrow ravines, and saddle points.

And gradients are our only compass.

---

#### What Does a Loss Landscape Look Like?

Let’s imagine a basic example: linear regression with two parameters.

We can visualize the **mean squared error (MSE)** loss surface for this model:

$$
\hat{y} = w_1 x_1 + w_2 x_2 \\
\mathcal{L}(w_1, w_2) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

The function $$\mathcal{L}(w_1, w_2)$$ is quadratic and convex. That means:
- The loss surface is shaped like a bowl
- The gradient at any point points directly toward the minimum
- The contours are ellipses, centered at the best-fit weights

Let’s build a synthetic dataset and visualize what the gradient field looks like.

---

#### Visualization: Gradient Field on a Loss Surface

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.random.randn(100, 2)
true_w = np.array([2.0, -3.0])
y = X @ true_w + np.random.randn(100) * 0.5

# Loss function (MSE)
def loss(w):
    y_pred = X @ w
    return np.mean((y - y_pred) ** 2)

# Gradient of loss
def grad(w):
    y_pred = X @ w
    return -2 * X.T @ (y - y_pred) / len(y)

# Create a grid over weight space
w1_vals = np.linspace(-4, 5, 40)
w2_vals = np.linspace(-6, 3, 40)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
Z = np.array([[loss(np.array([w1, w2])) for w1, w2 in zip(row1, row2)]
              for row1, row2 in zip(W1, W2)])

# Gradient vectors at each grid point
Gx = np.zeros_like(W1)
Gy = np.zeros_like(W2)

for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        w = np.array([W1[i, j], W2[i, j]])
        g = grad(w)
        Gx[i, j], Gy[i, j] = -g  # Negative gradient (descent direction)

# Normalize gradients for display
norm = np.sqrt(Gx**2 + Gy**2)
Gx /= norm + 1e-8
Gy /= norm + 1e-8

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
contours = ax.contour(W1, W2, Z, levels=30, cmap='viridis')
ax.clabel(contours, inline=True, fontsize=8)

ax.quiver(W1, W2, Gx, Gy, color='red', alpha=0.7, width=0.003)

# Mark the true minimum
ax.plot(true_w[0], true_w[1], 'ko', label='True minimum')

# Aesthetics
ax.set_title('Gradient Field on Loss Surface (Linear Regression)')
ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
```


<div style="text-align: center; margin: 2rem 0;">
  {% include figure.liquid 
      path="assets/img/optmcalc1_6.png" 
      class="img-fluid rounded shadow-sm" 
      loading="eager" 
      zoomable=true 
      alt="" 
  %}
</div>

- The **red arrows** show the gradient descent direction at each point.
- All arrows point inward, toward the global minimum—the best-fit weights.
- Contours are **elliptical**, reflecting the quadratic structure of MSE.
- The gradient magnitudes are stronger further out, and shrink as we approach the minimum—matching our earlier theory of gradient flow.

This is the **ideal case**: smooth, convex, easy to optimize. But real-world models, like deep neural networks, rarely give you this luxury.

---

#### Beyond Quadratics: When Gradients Struggle

In high-dimensional models:
- Gradients might vanish (flat regions)
- Or explode (near cliffs or chaotic boundaries)
- Or zigzag due to curvature imbalance (like a narrow ravine)

The surface might be **non-convex**, filled with:
- Multiple local minima
- Saddle points
- Flat valleys and deceptive ridges

In these cases, understanding gradients—and their limitations—becomes even more important. And that’s where second-order information like the **Hessian** can help.

But even in simpler models like logistic regression, you’ll see how gradients steer learning in powerful ways.

---


### Visualizing Cost Functions for Logistic Regression

Great—let’s now look at how these geometric and differential ideas show up in another foundational ML model: **logistic regression**.

While the linear regression loss surface was a perfect bowl, logistic regression introduces a non-linearity via the sigmoid, which slightly warps the loss surface—but still keeps it **convex**. This makes it a beautiful example for understanding gradient flow and cost geometry beyond quadratics.

In logistic regression, we’re not predicting continuous values—we’re predicting probabilities. Specifically, we model:

$$
\hat{y} = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}
$$

Then we compare $$\hat{y}$$ with the actual labels $$y \in \{0, 1\}$$ using the **binary cross-entropy (log loss)**:

$$
\mathcal{L}(w) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right]
$$

This function is still **convex**, but not quadratic. Its surface is warped—flatter in some areas, steeper in others—and has no closed-form minimum like MSE. But gradients still point us toward the best weights.

---

#### Let’s Visualize the Cost Surface

We’ll create a simple 2D dataset with binary labels, train a logistic regression model, and plot how the loss surface behaves as we vary $$w_1$$ and $$w_2$$.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

# Create toy classification dataset
np.random.seed(42)
X = np.random.randn(100, 2)
true_w = np.array([2.0, -3.0])
logits = X @ true_w
y = (sigmoid(logits) > 0.5).astype(int)

# Logistic loss function
def loss(w):
    logits = X @ w
    preds = sigmoid(logits)
    eps = 1e-10  # numerical stability
    return -np.mean(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps))

# Grid over weight space
w1_vals = np.linspace(-4, 5, 100)
w2_vals = np.linspace(-6, 4, 100)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
Z = np.array([[loss(np.array([w1, w2])) for w1, w2 in zip(row1, row2)]
              for row1, row2 in zip(W1, W2)])

# Plot the loss surface
fig, ax = plt.subplots(figsize=(8, 6))
contours = ax.contour(W1, W2, Z, levels=40, cmap='plasma')
ax.clabel(contours, inline=True, fontsize=8)

# Mark the true weights
ax.plot(true_w[0], true_w[1], 'ko', label='True weights')

# Aesthetics
ax.set_title('Logistic Regression Loss Surface')
ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
```

<div style="text-align: center; margin: 2rem 0;">
  {% include figure.liquid 
      path="assets/img/optmcalc1_7.png" 
      class="img-fluid rounded shadow-sm" 
      loading="eager" 
      zoomable=true 
      alt="" 
  %}
</div>


- The surface is **not symmetric**—it curves differently along $$w_1$$ and $$w_2$$.
- Still, it's **convex**—there’s only one global minimum.
- As you move toward the true weights, the loss consistently decreases.
- Some regions are *flatter*—meaning smaller gradients, slower learning.
- Others are *steeper*—meaning faster change and more dramatic updates.

Even without a closed-form solution, gradient descent works well here, precisely because of that convex shape. The gradient always points in a helpful direction—even if you don’t know where the minimum is.

---

#### Comparing to Linear Regression

<div style="border-left: 4px solid #2c5282; background: #f7fafc; padding: 1rem; border-radius: 6px; margin: 1rem 0; font-size: 0.95rem;"> <table style="width:100%; border-collapse: collapse; text-align: left;"> <thead style="background: #2c5282; color: white;"> <tr> <th style="padding: 8px;">Feature</th> <th style="padding: 8px;">Linear Regression</th> <th style="padding: 8px;">Logistic Regression</th> </tr> </thead> <tbody> <tr style="border-bottom: 1px solid #ccc;"> <td style="padding: 8px;">Loss Shape</td> <td style="padding: 8px;">Perfect bowl (quadratic)</td> <td style="padding: 8px;">Warped bowl (log-convex)</td> </tr> <tr style="border-bottom: 1px solid #ccc;"> <td style="padding: 8px;">Minimum</td> <td style="padding: 8px;">Closed-form solution</td> <td style="padding: 8px;">Requires numerical optimization</td> </tr> <tr style="border-bottom: 1px solid #ccc;"> <td style="padding: 8px;">Contour Shape</td> <td style="padding: 8px;">Elliptical, symmetric</td> <td style="padding: 8px;">Elliptical-ish, stretched or tilted</td> </tr> <tr style="border-bottom: 1px solid #ccc;"> <td style="padding: 8px;">Gradient Behavior</td> <td style="padding: 8px;">Constant rate of change</td> <td style="padding: 8px;">Varying rates; flatter near optimum</td> </tr> <tr> <td style="padding: 8px;">Optimization Path</td> <td style="padding: 8px;">Direct and steady</td> <td style="padding: 8px;">More sensitive to step size</td> </tr> </tbody> </table> </div>

---

This sets us up beautifully for more complex models where the surface is **not even convex**, and gradients can point in *unhelpful* directions. But before jumping to deep nets, we can build intuition for how gradients behave **across layers**, which leads us naturally to...

---


### Layer-Wise Gradients in Neural Networks

In neural networks, we often hear about **backpropagation**—the algorithm that trains deep models by "flowing gradients backward." But what does that actually mean, mathematically?

To understand this, we need to return to two ideas we’ve already built up:
- The **Jacobian**, which tells us how vector-valued functions transform space.
- The **chain rule**, which helps us understand how derivatives compose through nested functions.

Backpropagation is simply the **systematic application of the chain rule across layers**—a beautifully recursive structure of Jacobians and gradients.

---

#### The Setup

Let’s say we have a simple feedforward neural network:

$$
\mathbf{x} \xrightarrow{\;W^{(1)}\;} \mathbf{z}^{(1)} \xrightarrow{\;\phi\;} \mathbf{a}^{(1)} \xrightarrow{\;W^{(2)}\;} \mathbf{z}^{(2)} \xrightarrow{\;\phi\;} \dots \xrightarrow{\text{final layer}} \hat{y}
$$

Each layer applies:
- A **linear transformation**: $$\mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$
- A **nonlinear activation**: $$\mathbf{a}^{(l)} = \phi(\mathbf{z}^{(l)})$$

We want to compute the gradient of the loss with respect to each layer’s parameters: $$\frac{\partial \mathcal{L}}{\partial W^{(l)}}$$.

To do this, we propagate the gradient of the loss **backward**—from output to input.

---

#### Backprop: Layer-by-Layer

At the heart of backpropagation is this rule:

$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \cdot \left(\mathbf{a}^{(l-1)}\right)^T
$$

Where:
- $$\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}}$$ is the *error signal* at layer $$l$$
- $$\mathbf{a}^{(l-1)}$$ is the activation from the previous layer

This gradient is a matrix. The outer product structure arises because we’re taking derivatives of a scalar loss with respect to a matrix of weights.

The recursion for the error signal is:

$$
\delta^{(l)} = \left(W^{(l+1)}\right)^T \delta^{(l+1)} \odot \phi'(\mathbf{z}^{(l)})
$$

This is the multivariable chain rule in action—composed through Jacobians.

---

#### Geometric Interpretation

Each layer applies a function:

$$
\mathbf{a}^{(l)} = f^{(l)}(\mathbf{a}^{(l-1)})
$$

So the gradient of the loss w.r.t. earlier layers is:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(l-1)}} =
J_{f^{(l)}}^T \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(l)}}
$$

That is, **we multiply the incoming gradient by the transpose of the Jacobian of the layer**.

Layer-by-layer, the gradients are shaped by:
- The activations ($$\mathbf{a}$$)
- The Jacobians of each transformation
- The structure of the loss at the end

---

#### Table: What Flows Through the Network?

<div style="border-left: 4px solid #2c5282; background: #f7fafc; padding: 1rem; border-radius: 6px; margin: 1rem 0; font-size: 0.95rem;">

<table style="width:100%; border-collapse: collapse; text-align: left;">
  <thead style="background: #2c5282; color: white;">
    <tr>
      <th style="padding: 8px;">Quantity</th>
      <th style="padding: 8px;">Forward Pass</th>
      <th style="padding: 8px;">Backward Pass</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #ccc;">
      <td style="padding: 8px;">Input</td>
      <td style="padding: 8px;">$$\mathbf{x}$$</td>
      <td style="padding: 8px;">–</td>
    </tr>
    <tr style="border-bottom: 1px solid #ccc;">
      <td style="padding: 8px;">Linear transform</td>
      <td style="padding: 8px;">$$\mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)}$$</td>
      <td style="padding: 8px;">Multiply by $$\left(W^{(l)}\right)^T$$</td>
    </tr>
    <tr style="border-bottom: 1px solid #ccc;">
      <td style="padding: 8px;">Nonlinearity</td>
      <td style="padding: 8px;">$$\mathbf{a}^{(l)} = \phi(\mathbf{z}^{(l)})$$</td>
      <td style="padding: 8px;">Elementwise multiply by $$\phi'(\mathbf{z}^{(l)})$$</td>
    </tr>
    <tr>
      <td style="padding: 8px;">Loss</td>
      <td style="padding: 8px;">$$\hat{y}, \mathcal{L}(\hat{y}, y)$$</td>
      <td style="padding: 8px;">Start from $$\frac{\partial \mathcal{L}}{\partial \hat{y}}$$ and chain backward</td>
    </tr>
  </tbody>
</table>

</div>

---

#### A Numerical Demo of Backpropagation

Let’s walk through a **numerical example** using a tiny 1-hidden-layer neural network with ReLU activation and one output neuron.

##### Problem Setup:

- Input: $$\mathbf{x} = [1, 2]$$  
- Hidden layer: 2 neurons, ReLU  
- Output: 1 neuron, linear  
- True label: $$y = 10$$  
- Loss: Mean Squared Error (MSE)

---

##### Network Parameters

- Hidden layer weights:

$$
W^{[1]} = \begin{bmatrix}
1 & -1 \\
2 & 0
\end{bmatrix}, \quad
\mathbf{b}^{[1]} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

- Output layer weights:

$$
W^{[2]} = \begin{bmatrix} 3 & -1 \end{bmatrix}, \quad b^{[2]} = 0
$$

---

Step 1: Forward Pass

**Input to hidden layer:**

$$
\mathbf{z}^{[1]} = W^{[1]} \cdot \mathbf{x} + \mathbf{b}^{[1]} =
\begin{bmatrix}
1 & -1 \\
2 & 0
\end{bmatrix}
\cdot
\begin{bmatrix}
1 \\
2
\end{bmatrix}
=
\begin{bmatrix}
1 - 2 \\
2 + 0
\end{bmatrix}
=
\begin{bmatrix}
-1 \\
2
\end{bmatrix}
$$

**Apply ReLU:**

$$
\mathbf{a}^{[1]} = \phi(\mathbf{z}^{[1]}) =
\begin{bmatrix}
\max(0, -1) \\
\max(0, 2)
\end{bmatrix}
=
\begin{bmatrix}
0 \\
2
\end{bmatrix}
$$

**Output layer:**

$$
\hat{y} = W^{[2]} \cdot \mathbf{a}^{[1]} + b^{[2]} = [3, -1] \cdot [0, 2] = -2
$$

---

Step 2: Compute Loss

Use MSE:

$$
\mathcal{L} = \frac{1}{2} (y - \hat{y})^2 = \frac{1}{2} (10 - (-2))^2 = \frac{1}{2} \cdot 144 = 72
$$

---

Step 3: Backpropagation

Step 3.1: Gradient of loss w.r.t. output

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y} - y = -2 - 10 = -12
$$

Step 3.2: Gradient w.r.t. output layer weights

We use:

$$
\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W^{[2]}} = -12 \cdot \mathbf{a}^{[1]} = -12 \cdot \begin{bmatrix} 0 \\ 2 \end{bmatrix} = \begin{bmatrix} 0 \\ -24 \end{bmatrix}
$$

So:

$$
\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \begin{bmatrix} 0 & -24 \end{bmatrix}
$$

---

Step 3.3: Backpropagate to hidden layer (chain rule)

We need:

$$
\delta^{[1]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[1]}} = \left( W^{[2]} \right)^T \cdot \frac{\partial \mathcal{L}}{\partial \hat{y}} \odot \phi'(\mathbf{z}^{[1]})
$$

- $$\phi'(\mathbf{z}^{[1]}) = [0, 1]$$ (since ReLU derivative is 0 for negative inputs, 1 for positive)
- $$W^{[2]} = [3, -1]$$ → $$\left(W^{[2]}\right)^T = \begin{bmatrix} 3 \\ -1 \end{bmatrix}$$

So:

$$
\delta^{[1]} = \begin{bmatrix} 3 \\ -1 \end{bmatrix} \cdot (-12) \odot \begin{bmatrix} 0 \\ 1 \end{bmatrix}
= \begin{bmatrix} -36 \\ 12 \end{bmatrix} \odot \begin{bmatrix} 0 \\ 1 \end{bmatrix}
= \begin{bmatrix} 0 \\ 12 \end{bmatrix}
$$

---

Step 3.4: Gradient w.r.t. hidden layer weights

Use:

$$
\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \delta^{[1]} \cdot (\mathbf{x})^T
= \begin{bmatrix} 0 \\ 12 \end{bmatrix}
\cdot
\begin{bmatrix} 1 & 2 \end{bmatrix}
=
\begin{bmatrix}
0 & 0 \\
12 & 24
\end{bmatrix}
$$

---

##### Final Gradients Summary

<div style="border-left: 4px solid #2c5282; background: #f7fafc; padding: 1rem; border-radius: 6px; margin: 1rem 0; font-size: 0.95rem;">

<table style="width:100%; border-collapse: collapse; text-align: left;">
  <thead style="background: #2c5282; color: white;">
    <tr>
      <th style="padding: 8px;">Parameter</th>
      <th style="padding: 8px;">Gradient</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #ccc;">
      <td style="padding: 8px;">$$W^{[2]}$$</td>
      <td style="padding: 8px;">$$[0,\ -24]$$</td>
    </tr>
    <tr>
      <td style="padding: 8px;">$$W^{[1]}$$</td>
      <td style="padding: 8px;">$$\begin{bmatrix} 0 & 0 \\ 12 & 24 \end{bmatrix}$$</td>
    </tr>
  </tbody>
</table>

</div>

---



### Jacobians in Backpropagation

Great—we’ve just seen how gradients flow backward through neural networks using chain rules and Jacobians. Now let’s step into how Jacobians (and sometimes Hessians) show up *inside* those gradients, especially when the outputs themselves are vector-valued.

Backpropagation is often taught as a mechanical algorithm: compute forward, cache values, compute gradients backward. But under the hood, what we’re really doing is composing **Jacobian matrices** of every layer.

The power of this interpretation? It lets you understand:
- Why gradients flow the way they do
- How they’re affected by each transformation
- How automatic differentiation frameworks like PyTorch and TensorFlow actually work

---

#### Recap: The Chain Rule, Vector Version

For scalar functions, we’re used to:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

In vector calculus, this becomes:

$$
J_{f \circ g}(x) = J_f(g(x)) \cdot J_g(x)
$$

That is, the **Jacobian of a composition is the product of Jacobians**.

This is the real backbone of backpropagation.

---

#### A Simple Example: 2-Layer Network with Vector Output

Suppose your network looks like this:

$$
\mathbf{x} \in \mathbb{R}^2
$$

$$
\mathbf{z}^{[1]} = W^{[1]} \cdot \mathbf{x} \in \mathbb{R}^3
$$

$$
\mathbf{a}^{[1]} = \phi(\mathbf{z}^{[1]}) \in \mathbb{R}^3
$$

$$
\mathbf{z}^{[2]} = W^{[2]} \cdot \mathbf{a}^{[1]} \in \mathbb{R}^2
$$

$$
\hat{\mathbf{y}} = \phi(\mathbf{z}^{[2]}) \in \mathbb{R}^2
$$

Assume your loss function is:

$$
\mathcal{L} = \frac{1}{2} \|\hat{\mathbf{y}} - \mathbf{y}\|^2
$$

To compute $$\frac{\partial \mathcal{L}}{\partial W^{[1]}}$$, we need:

1. The Jacobian of the loss with respect to $$\hat{\mathbf{y}}$$
2. The Jacobian of $$\hat{\mathbf{y}}$$ with respect to $$\mathbf{z}^{[2]}$$
3. The Jacobian of $$\mathbf{z}^{[2]}$$ with respect to $$\mathbf{a}^{[1]}$$
4. The Jacobian of $$\mathbf{a}^{[1]}$$ with respect to $$\mathbf{z}^{[1]}$$
5. And finally, $$\frac{\partial \mathbf{z}^{[1]}}{\partial W^{[1]}}$$

---

#### Matrix Multiplication of Jacobians

Let’s define each transformation and then apply the **chain rule with Jacobians**.

##### Step-by-step Jacobian flow:

$$
\frac{\partial \mathcal{L}}{\partial W^{[1]}} =
\underbrace{\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}}}_{\text{(2x1)}}
\cdot
\underbrace{J_{\hat{\mathbf{y}} \leftarrow \mathbf{z}^{[2]}}}_{(2x2)}
\cdot
\underbrace{J_{\mathbf{z}^{[2]} \leftarrow \mathbf{a}^{[1]}}}_{(2x3)}
\cdot
\underbrace{J_{\mathbf{a}^{[1]} \leftarrow \mathbf{z}^{[1]}}}_{(3x3)}
\cdot
\underbrace{\frac{\partial \mathbf{z}^{[1]}}{\partial W^{[1]}}}_{(3x2x2)}
$$

> Technically, the last term is a **tensor**—because we’re differentiating a matrix output w.r.t. a matrix input. But you can also think of it row-by-row.

Each Jacobian tells you: how does output move if I wiggle the previous layer? And backprop is just: multiply those movements step by step, using the **transpose of the Jacobians**.

---

#### How Autograd Libraries Do It

Autograd engines like PyTorch and TensorFlow build a **computation graph** behind the scenes. Each node:
- Stores its forward value
- Stores the **Jacobian** (or more precisely, its **pullback rule**)
- Chains them backward using reverse-mode differentiation

In PyTorch:
- `.backward()` walks through this chain of Jacobians
- It avoids computing full Jacobian matrices—just applies **Jacobian-vector products (JVPs)**

This is much more memory efficient than storing all Jacobians.

---

#### Table: Jacobians Through Layers

<div style="border-left: 4px solid #2c5282; background: #f7fafc; padding: 1rem; border-radius: 6px; margin: 1rem 0; font-size: 0.95rem;">

<table style="width:100%; border-collapse: collapse; text-align: left;">
  <thead style="background: #2c5282; color: white;">
    <tr>
      <th style="padding: 8px;">Operation</th>
      <th style="padding: 8px;">Forward</th>
      <th style="padding: 8px;">Jacobian or Gradient</th>
    </tr>
  </thead>
  <tbody>

    <tr style="background-color: #edf2f7; border-bottom: 1px solid #ccc;">
      <td style="padding: 8px;"><em>Loss (MSE)</em></td>
      <td style="padding: 8px;">$$\mathcal{L} = \frac{1}{2} \|\hat{y} - y\|^2$$</td>
      <td style="padding: 8px;">$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y} - y$$</td>
    </tr>

    <tr style="border-bottom: 1px solid #ccc;">
      <td style="padding: 8px;">Linear</td>
      <td style="padding: 8px;">$$\mathbf{z} = W \cdot \mathbf{x}$$</td>
      <td style="padding: 8px;">$$\mathbb{R}^{\text{out} \times \text{in}}$$</td>
    </tr>

    <tr style="border-bottom: 1px solid #ccc;">
      <td style="padding: 8px;">ReLU</td>
      <td style="padding: 8px;">$$\mathbf{a} = \max(0,\ \mathbf{z})$$</td>
      <td style="padding: 8px;">$$\text{Diagonal} \in \mathbb{R}^{n \times n}$$</td>
    </tr>

    <tr>
      <td style="padding: 8px;">Sigmoid</td>
      <td style="padding: 8px;">$$\sigma(z) = \frac{1}{1 + e^{-z}}$$</td>
      <td style="padding: 8px;">$$\text{Diagonal: } \sigma(z)(1 - \sigma(z))$$</td>
    </tr>

  </tbody>
</table>

</div>
---

#### Why This Matters

Seeing backprop through the lens of Jacobians gives you:
- A deeper understanding of how **gradient flow** works
- The ability to **debug and design custom layers**
- A way to extend to **second-order methods**, where Jacobians and Hessians matter



---

### Hessians in Newton’s Method

Gradient descent tells you *which way* to move to reduce loss. But it doesn't tell you how **far** to move with confidence. It assumes a simple landscape and takes cautious steps.

What if the surface isn’t symmetric?  
What if it curves more sharply in some directions than others?

That’s exactly what the **Hessian matrix** captures—and Newton’s method uses it to take **curvature-aware steps**.

---

#### Gradient Descent vs. Newton’s Method

- **Gradient descent** updates:
  
  $$
  \theta_{t+1} = \theta_t - \eta \cdot \nabla \mathcal{L}(\theta_t)
  $$

- **Newton’s method** uses:

  $$
  \theta_{t+1} = \theta_t - H^{-1}(\theta_t) \cdot \nabla \mathcal{L}(\theta_t)
  $$

Here, $$H$$ is the **Hessian of the loss function**, and the inverse adjusts for the local curvature. When the surface is steep in one direction and flat in another, Newton’s method scales the step accordingly—big where it's flat, small where it's sharp.

---

#### How It Works Geometrically

Let’s say you’re minimizing a scalar function $$f: \mathbb{R}^n \rightarrow \mathbb{R}$$. Around any point $$\theta$$, you can do a second-order Taylor approximation:

$$
f(\theta + \Delta \theta) \approx f(\theta) + \nabla f(\theta)^T \Delta \theta + \frac{1}{2} \Delta \theta^T H \Delta \theta
$$

To minimize this quadratic approximation, set the derivative w.r.t. $$\Delta \theta$$ to zero:

$$
\nabla f(\theta) + H \Delta \theta = 0
$$

Solving gives:

$$
\Delta \theta = - H^{-1} \nabla f(\theta)
$$

This is **Newton’s update step**.

---

**Numerical Example: Newton’s Method on a Simple 2D Quadratic**

Let’s apply this to a concrete function:

**Function:**

$$
f(x_1, x_2) = x_1^2 + 4x_2^2 - 4x_1 - 8x_2
$$

This is a convex quadratic with an obvious minimum.

---

- Step 1: Compute Gradient

$$
\nabla f(x_1, x_2) = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2}
\end{bmatrix}
=
\begin{bmatrix}
2x_1 - 4 \\
8x_2 - 8
\end{bmatrix}
$$

---

- Step 2: Compute Hessian

$$
H = \nabla^2 f = \begin{bmatrix}
2 & 0 \\
0 & 8
\end{bmatrix}
$$

Note: it's constant since the function is quadratic.

---

- Step 3: Choose Starting Point

Let’s start at $$x = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

---

- Step 4: Compute Gradient at Starting Point

$$
\nabla f(0, 0) = \begin{bmatrix} -4 \\ -8 \end{bmatrix}
$$

---

- Step 5: Newton Step

Use:

$$
x_{\text{new}} = x - H^{-1} \cdot \nabla f(x)
$$

We compute:

-- Inverse of Hessian:

  $$
  H^{-1} = \begin{bmatrix} 1/2 & 0 \\ 0 & 1/8 \end{bmatrix}
  $$

-- Step:

  $$
  \Delta x = - H^{-1} \cdot \begin{bmatrix} -4 \\ -8 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}
  $$

-- Update:

  $$
  x_{\text{new}} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} + \begin{bmatrix} 2 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}
  $$

---

- Step 6: Verify Minimum

Compute gradient at $$x = [2,\ 1]$$:

$$
\nabla f(2, 1) = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

We’ve reached the **global minimum in one step**.

---

**Gradient Descent vs. Newton’s Method**

<div style="border-left: 4px solid #2c5282; background: #f7fafc; padding: 1rem; border-radius: 6px; margin: 1rem 0; font-size: 0.95rem;">

<table style="width:100%; border-collapse: collapse; text-align: left;">
  <thead style="background: #2c5282; color: white;">
    <tr>
      <th style="padding: 8px;">Method</th>
      <th style="padding: 8px;">Step Formula</th>
      <th style="padding: 8px;">Pros</th>
      <th style="padding: 8px;">Cons</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #ccc;">
      <td style="padding: 8px;">Gradient Descent</td>
      <td style="padding: 8px;">$$x_{t+1} = x_t - \eta \nabla f(x_t)$$</td>
      <td style="padding: 8px;">Simple, stable, scalable</td>
      <td style="padding: 8px;">Slow in flat regions; ignores curvature</td>
    </tr>
    <tr>
      <td style="padding: 8px;">Newton’s Method</td>
      <td style="padding: 8px;">$$x_{t+1} = x_t - H^{-1} \nabla f(x_t)$$</td>
      <td style="padding: 8px;">Fast (quadratic convergence near min)</td>
      <td style="padding: 8px;">Requires Hessian and matrix inversion</td>
    </tr>
  </tbody>
</table>

</div>

---

Newton’s method isn’t widely used in deep learning—because computing and inverting the Hessian is expensive—but it **informs the design** of many **second-order approximations** like:
- **L-BFGS**
- **Adam (which estimates 2nd-order behavior via momentum + variance)**
- **Natural gradient methods**

---

## Wrapping Up

We’ve covered a lot of ground—from gradients and directional derivatives to Jacobians and Hessians. At first glance, these might seem like purely academic tools—concepts you meet in a calculus course, solve a few problems with, and move on.

But once you start building machine learning models—especially the kind that don’t behave like nice, neat bowls—you realize: this math doesn’t go away. It just gets buried behind high-level APIs and layers of abstraction.

- Gradients tell us which way to move.
- Jacobians explain how outputs stretch and shift.
- Hessians warn us when the slope changes shape.

These are the core machinery behind optimization, training stability, learning rates, and even architecture design.

What’s powerful is that once you really get these ideas—not just the equations, but the *geometry* behind them—you start to see why models behave the way they do. Why gradients vanish in one layer and explode in another. Why some loss surfaces feel like gentle slopes and others like twisted ravines. Why second-order methods aren’t used much in deep learning, but still shape the thinking behind optimizers like Adam or AdaGrad.
