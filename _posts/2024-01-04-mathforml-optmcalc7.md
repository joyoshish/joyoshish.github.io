---
layout: post
title: Matrix Calculus for Machine Learning
date: 2024-01-04
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


<div style="font-style: italic; color: #444; padding: 0.8rem 1rem; background: #f8f8f8; border-left: 4px solid #2c5282; border-radius: 6px;">
In 1943, Warren McCulloch and Walter Pitts scribbled down a few equations that captured the logic of a neuron. That paper gave birth to the idea of neural networks, but they didn’t know how to train one.  
Backpropagation, the algorithm that finally cracked the learning part, didn’t arrive until the 1980s.  
And at the heart of it — was matrix calculus.
</div>

If you’ve ever trained a neural network, optimized a loss function, or simply tried to make your model a little less wrong, you’ve likely relied on gradients flowing through vectors, matrices, and tensors. These gradients were computed — carefully, systematically — using the machinery of **matrix calculus**.

This blog is about that machinery. Not the abstract, hand-wavy kind. But the real thing — the rules, tricks, and identities that power everything from a simple linear regression to backpropagation in massive language models.

---

In undergrad calculus, we learn to differentiate functions like $$f(x) = x^2 + 3x$$. Neat and useful — for scalar variables. But machine learning isn’t scalar land. Models work with:

- Vectors of weights: $$\mathbf{w} \in \mathbb{R}^n$$  
- Data matrices: $$X \in \mathbb{R}^{n \times p}$$  
- Outputs: sometimes scalars, often vectors, occasionally entire matrices

And when we want to compute how a scalar-valued loss changes with respect to all of these — we need **vector derivatives**, **Jacobian matrices**, **matrix-by-matrix derivatives**, and more.

They're the basis of:

- Training algorithms (SGD, Adam, Newton's method)  
- Regularization (L2, Frobenius norm)  
- Dimensionality reduction (PCA, SVD)  
- Deep learning (backprop, chain rule in matrix form)  
- Probabilistic models (matrix normal distributions, determinant derivatives)

Even the automatic differentiation tools in PyTorch or TensorFlow rely on these rules under the hood. When things break, or when you're writing something novel (a custom loss, a new model class), **you need to know how the derivatives actually work**.

We’re going to build the entire toolbox — step by step.

---



## Scalar-by-Vector Derivatives: $$\frac{\partial f}{\partial \mathbf{x}}$$

Let’s start simple.

Suppose you’ve got a scalar-valued function — say, a loss function — that depends on multiple variables:  
$$f: \mathbb{R}^n \rightarrow \mathbb{R}$$

This means: you feed in a vector $$\mathbf{x} \in \mathbb{R}^n$$ and out comes a single number. In machine learning, this is incredibly common — almost every cost function is of this type. You feed in a vector of weights, and get a scalar loss.

Our job?  
To figure out: *how does this scalar change as we wiggle each coordinate of $$\mathbf{x}$$?*

That’s where **gradients** and **directional derivatives** come in.

---

### From Partial Derivatives to the Gradient

Let’s say:
$$
\mathbf{x} = \begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}, \quad
f(\mathbf{x}) \in \mathbb{R}
$$

Then the **gradient** of $$f$$ with respect to $$\mathbf{x}$$ is just a vector of partial derivatives:

$$
\nabla_{\mathbf{x}} f = \frac{\partial f}{\partial \mathbf{x}} =
\begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
\in \mathbb{R}^n
$$

This vector tells us *how much and in what direction* the function $$f$$ changes as each component of $$\mathbf{x}$$ changes.

> Think of it like this: if you’re hiking on a scalar-valued elevation map (like a hill), and your coordinates are given by $$\mathbf{x}$$, then the gradient vector points *uphill*. Its direction gives the steepest ascent, and its magnitude gives the steepness.

---

### A Simple Example

Let’s take a warm-up example:

**Function:**

$$
f(\mathbf{x}) = \mathbf{x}^\top \mathbf{x} = \sum_{i=1}^n x_i^2
$$

That’s just the squared L2 norm, i.e. $$f(\mathbf{x}) = \|\mathbf{x}\|^2$$.

**Gradient:**

Let’s compute:
$$
\frac{\partial f}{\partial \mathbf{x}} = 
\begin{bmatrix}
\frac{\partial}{\partial x_1}(x_1^2 + \dots + x_n^2) \\
\vdots \\
\frac{\partial}{\partial x_n}(x_1^2 + \dots + x_n^2)
\end{bmatrix}
=
\begin{bmatrix}
2x_1 \\
2x_2 \\
\vdots \\
2x_n
\end{bmatrix}
= 2\mathbf{x}
$$

**Interpretation:** The gradient of the squared norm points directly *away from the origin*, scaled by 2. That makes sense — the farther you are from the origin, the faster the norm grows.

---

### Directional Derivative — *the Gradient in Disguise*


Given a scalar-valued function  
$$f : \mathbb{R}^n \rightarrow \mathbb{R}$$  
and a **direction vector**  
$$\mathbf{v} \in \mathbb{R}^n,$$

the **directional derivative** of $$f$$ at point $$\mathbf{x}$$ in the direction $$\mathbf{v}$$ is defined as:

$$
D_{\mathbf{v}} f(\mathbf{x}) := \lim_{h \to 0} \frac{f(\mathbf{x} + h\,\mathbf{v}) - f(\mathbf{x})}{h}
$$

If you prefer unit directions, set  
$$\mathbf{u} = \frac{\mathbf{v}}{\lVert\mathbf{v}\rVert}$$  
and then  
$$D_{\mathbf{u}} f$$  
measures the rate of change *per unit distance* traveled.

---

#### Why the Gradient Pops Out in Directional Derivatives

Suppose you’re at a point $$\mathbf{x}$$ in $$n$$-dimensional space, and you’re looking at a scalar function $$f(\mathbf{x})$$. You want to ask:  
*“If I take a small step in some arbitrary direction $$\mathbf{v}$$, how much will $$f$$ change?”*

This is **not** a partial derivative — those only describe change *along one coordinate axis*. But in real-world problems, especially in ML optimization, we often care about arbitrary directions. So we turn to the **directional derivative**.

Let’s build up to the formula and see how the gradient quietly reveals itself.

---

 **Step 1: Start from the Definition**

We want the instantaneous rate of change of $$f$$ at $$\mathbf{x}$$ in the direction $$\mathbf{v}$$:

$$
D_{\mathbf{v}} f(\mathbf{x}) := \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{v}) - f(\mathbf{x})}{h}
$$

---

 **Step 2: Use Taylor Expansion**

If $$f$$ is differentiable, then for small $$h$$, we can expand:

$$
f(\mathbf{x} + h\mathbf{v}) \approx f(\mathbf{x}) + h\, \nabla f(\mathbf{x})^\top \mathbf{v} + o(h)
$$

where:

- $$h\mathbf{v}$$ is your step,
- $$\nabla f(\mathbf{x})^\top \mathbf{v}$$ is the dot product between gradient and direction,
- and $$o(h)$$ vanishes faster than $$h$$ as $$h \to 0$$.

Now plug this into the definition:

$$
\begin{aligned}
D_{\mathbf{v}} f(\mathbf{x})
&= \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{v}) - f(\mathbf{x})}{h} \\\\
&= \lim_{h \to 0} \frac{f(\mathbf{x}) + h\,\nabla f(\mathbf{x})^\top \mathbf{v} + o(h) - f(\mathbf{x})}{h} \\\\
&= \lim_{h \to 0} \left( \nabla f(\mathbf{x})^\top \mathbf{v} + \frac{o(h)}{h} \right) \\\\
&= \nabla f(\mathbf{x})^\top \mathbf{v}
\end{aligned}
$$

That’s the result:  
the **directional derivative is just a dot product between the gradient and the direction vector**.

---

 **Step 3: Intuition — Why a Dot Product?**

Recall:

$$
\nabla f^\top \mathbf{v} = \sum_{i=1}^n \frac{\partial f}{\partial x_i} \cdot v_i
$$

So the gradient acts as a **weighted sum of sensitivities**: how much each coordinate direction contributes to change in $$f$$, scaled by how much you're moving in that direction.

- If $$\mathbf{v}$$ points in the **same direction** as $$\nabla f$$ → large positive dot product → **steepest ascent**.
- If $$\mathbf{v}$$ is **orthogonal** to $$\nabla f$$ → dot product is zero → **no change**.
- If $$\mathbf{v}$$ points **opposite** to $$\nabla f$$ → negative dot product → **steepest descent**.

This is the geometric meaning behind "gradient = steepest ascent direction."

---

 **Step 4: The Gradient as a Local Linear Model**

Think of $$f(\mathbf{x})$$ as the altitude at position $$\mathbf{x}$$ on a hilly surface.

Then:

- The gradient $$\nabla f(\mathbf{x})$$ tells you which way is *uphill fastest*.
- Moving orthogonally to the gradient is like walking *around* the hill at constant altitude.
- The directional derivative tells you *how much altitude changes* if you take a small step in any direction $$\mathbf{v}$$ — and it’s exactly the projection of $$\mathbf{v}$$ onto the gradient.

---

#### 2D Example 

Take the simple quadratic surface:

$$
f(x, y) = x^2 + y^2
$$

At point $$(1, 2)$$:

- The gradient is  
  $$\nabla f = \begin{bmatrix} 2x \\ 2y \end{bmatrix} = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$$
- Take a direction vector:  
  $$\mathbf{v} = \begin{bmatrix} 3 \\ -4 \end{bmatrix}$$

Then:

$$
\begin{aligned}
D_{\mathbf{v}} f(1, 2)
&= \nabla f^\top \mathbf{v} \\\\
&= \begin{bmatrix} 2 & 4 \end{bmatrix} \cdot \begin{bmatrix} 3 \\ -4 \end{bmatrix} \\\\
&= 2 \cdot 3 + 4 \cdot (-4) = 6 - 16 = -10
\end{aligned}
$$

**Interpretation:**  
A tiny step in the direction $$(3, -4)$$ causes the function to **decrease** at a rate of 10 units per unit step — because that direction leans *against* the gradient.

---

#### Key Takeaways

- The **gradient vector** encodes how a function changes in *every possible direction*.
- The **directional derivative** in direction $$\mathbf{v}$$ is simply $$\nabla f^\top \mathbf{v}$$ — a projection of movement onto the gradient.
- In machine learning, this is the theoretical justification behind **gradient descent**: to decrease a loss function fast, go in direction $$-\nabla f$$.


---

### Matrix Notation

In matrix calculus, we often prefer this notation:

If $$f(\mathbf{x})$$ is scalar and $$\mathbf{x} \in \mathbb{R}^{n \times 1}$$, then:

- The gradient is typically a **column vector** $$\frac{\partial f}{\partial \mathbf{x}} \in \mathbb{R}^{n \times 1}$$  
- But some references (especially in engineering) write it as a row vector — be aware of this ambiguity.

In this blog series, we’ll **default to column vectors** for gradients.

---

### Why This Matters in Machine Learning

Any loss function you optimize — mean squared error, cross-entropy, hinge loss — boils down to:

$$
\min_{\mathbf{w}} \; f(\mathbf{w})
$$

And your optimizer (SGD, Adam, etc.) needs $$\nabla_{\mathbf{w}} f$$.

Without understanding this gradient — how it’s formed, how it's directional — you're just running someone else's black box. And when you want to *build your own* (say, with custom loss, or reparameterized weights), you’ll need this math.

---

Real-world models — from logistic regression to deep neural networks — don’t just optimize over vectors. They learn **matrices of weights**, like the parameter matrix $$W$$ in a linear transformation:

$$
Z = XW
$$

We need to compute how a **scalar-valued loss** changes with respect to **every entry of a matrix**.

So the natural next question becomes:

> *What does the gradient look like when the variable is a matrix?*

That’s where we’re headed now.

Let’s step into **scalar-by-matrix derivatives**, and see how gradients flow through entire affine maps — the backbone of modern ML.



## Scalar-by-Matrix Derivatives: $$\frac{\partial L}{\partial W}$$

Most machine learning models funnel a whole batch of data through an **affine map**:

$$
\mathbf{Z} = XW + \mathbf{b}
$$

where:

- $$X \in \mathbb{R}^{m \times d}$$ holds $$m$$ examples with $$d$$ features each,  
- $$W \in \mathbb{R}^{d \times k}$$ is the weight matrix we want to learn,  
- $$\mathbf{b} \in \mathbb{R}^{1 \times k}$$ (or broadcasted) is a bias, and  
- $$\mathbf{Z}$$ feeds a softmax, sigmoid, or some other non-linearity.

Our task is to figure out how a **scalar loss**

$$
L = \mathcal{L}(\mathbf{Z},\, \mathbf{y})
$$

changes when we nudge each entry of $$W$$.

---

### Warm-Up: Mean-Squared Error with a Single Output

Let’s take plain linear regression with a single output:

$$
\hat{y} = X\mathbf{w}, \qquad L = \frac{1}{2m}\;\lVert\hat{y} - y\rVert^2
$$

Here we write $$W$$ as a column vector $$\mathbf{w}$$ just to keep notation light.

**Derivative with respect to $$\mathbf{w}$$:**

$$
\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{m}\,X^\top (X\mathbf{w} - y)
$$

That’s the familiar normal equation gradient.

> **Takeaway** — When the loss is quadratic and predictions are linear in weights, the gradient boils down to *“design-matrix-transpose times residuals.”*

---

### General Rule: Scalar Loss w.r.t. Matrix

Suppose:

$$
L = f(Z(W)), \qquad Z = XW \in \mathbb{R}^{m \times k}
$$

Using the trace trick, we express the scalar loss as:

$$
L = \mathrm{tr}(f(Z))
$$

Now use the identity:

$$
\boxed{
\frac{\partial\, \mathrm{tr}(A^\top X W)}{\partial W} = X^\top A
}
$$

Let:

- $$G = \frac{\partial L}{\partial Z} \in \mathbb{R}^{m \times k}$$ be the "upstream" gradient  
- $$Z = XW$$ as before

Then apply the chain rule in matrix form:

$$
\begin{aligned}
\frac{\partial L}{\partial W}
&= \frac{\partial\, \mathrm{tr}(G^\top Z)}{\partial W} \\
&= \frac{\partial\, \mathrm{tr}(G^\top XW)}{\partial W} \\
&= X^\top G
\end{aligned}
$$

This formula — $$X^\top G$$ — is the workhorse of backpropagation for fully connected layers.



---

### Logistic Regression Example

For a binary classification task, the loss per sample is:

$$
\ell_i = -\left[y_i \log \sigma(z_i) + (1 - y_i)\log (1 - \sigma(z_i))\right], \quad z_i = x_i^\top \mathbf{w}
$$

Then the full batch loss is:

$$
L = \frac{1}{m} \sum_{i=1}^m \ell_i
$$

The derivative of the loss with respect to each logit $$z_i$$ is:

$$
\frac{\partial L}{\partial z_i} = \frac{1}{m}(\hat{y}_i - y_i)
$$

Stack this into a matrix form:

$$
G = \frac{1}{m}(\hat{y} - y), \qquad G \in \mathbb{R}^{m \times 1}
$$

Then the gradient with respect to weights is:

$$
\boxed{
\frac{\partial L}{\partial \mathbf{w}} = X^\top G = \frac{1}{m} X^\top (\hat{y} - y)
}
$$

Again — the usual rule hidden inside every ML library.

---

### Key Patterns to Remember

<div style="overflow-x: auto;"> <table class="simple-table"> <thead> <tr> <th>Situation</th> <th>Upstream Gradient</th> <th><span style="white-space: nowrap;">\(\frac{\partial L}{\partial W}\)</span></th> </tr> </thead> <tbody> <tr> <td><span style="white-space: nowrap;">\(Z = XW\)</span>, arbitrary scalar loss</td> <td><span style="white-space: nowrap;">\(G = \frac{\partial L}{\partial Z}\)</span></td> <td><span style="white-space: nowrap;">\(X^\top G\)</span></td> </tr> <tr> <td><span style="white-space: nowrap;">\(L = \frac{1}{2} \lVert XW - y \rVert^2\)</span> (MSE)</td> <td><span style="white-space: nowrap;">\(G = XW - y\)</span></td> <td><span style="white-space: nowrap;">\(X^\top (XW - y)\)</span></td> </tr> <tr> <td>Softmax + Cross-Entropy</td> <td><span style="white-space: nowrap;">\(G = \hat{Y} - Y\)</span></td> <td><span style="white-space: nowrap;">\(X^\top (\hat{Y} - Y)\)</span></td> </tr> </tbody> </table> </div>

---

### Common Pitfalls

1. **Shape mismatches** — $$X^\top G$$ must match the shape of $$W$$ exactly  
2. **Missing constants** — don’t forget the factor $$\frac{1}{m}$$ when averaging over batches  
3. **Nonlinearities** — gradients must *pass through* activation functions before reaching $$W$$

---

By now, we’ve moved beyond gradients of scalars — and entered the terrain where **entire vectors flow through our models**. The Jacobian is what keeps track of how those outputs shift, stretch, or twist when the inputs change. It's the key to understanding how vector-valued functions respond to vector inputs, and it underpins the machinery of backpropagation, reparameterization tricks, and more.

But even this isn’t the full story.

In deep learning, we often work with parameters, activations, and losses that are **matrices**. Layers are matrix functions of matrices. The derivatives we need? They aren’t just gradients or Jacobians — they’re something more structured.

Let’s now turn to that next level of complexity:  
**what happens when both the function *and* the variable are matrices?**

---



## Vector-by-Vector Derivatives: $$\frac{\partial \mathbf{f}}{\partial \mathbf{x}}$$


Imagine a function that takes in a vector and spits out another vector:

$$
\mathbf{f} : \mathbb{R}^n \rightarrow \mathbb{R}^m
$$

This setup is everywhere in machine learning and data science:

- Neural network layers (input → activations)
- Batch transformations
- Coordinate transformations in loss landscapes

So, how do we represent **the derivative of a vector-valued function with respect to a vector**?

The answer is the **Jacobian matrix**.

---

### What Is the Jacobian?

If:

$$
\mathbf{f}(\mathbf{x}) = \begin{bmatrix}
f_1(\mathbf{x}) \\
f_2(\mathbf{x}) \\
\vdots \\
f_m(\mathbf{x})
\end{bmatrix}
\quad \text{where } \mathbf{x} \in \mathbb{R}^n,
$$

then the **Jacobian** of $$\mathbf{f}$$ with respect to $$\mathbf{x}$$ is the matrix of all partial derivatives:

$$
\boxed{
\frac{\partial \mathbf{f}}{\partial \mathbf{x}} =
J_{\mathbf{f}}(\mathbf{x}) =
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
}
$$

- Each **row** corresponds to the gradient of a single component function $$f_i$$.
- Each **column** corresponds to the sensitivity of all outputs to a single input dimension $$x_j$$.

If you're familiar with gradients, you can think of the Jacobian as **a stacked collection of gradients**, one per output dimension.

---

### Example: Simple Vector Function

Let:

$$
\mathbf{f}(\mathbf{x}) = 
\begin{bmatrix}
x_1 + 2x_2 \\
x_1^2 \\
\sin(x_2)
\end{bmatrix},
\quad \mathbf{x} =
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

Then the Jacobian is:

$$
\frac{\partial \mathbf{f}}{\partial \mathbf{x}} =
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} \\
\frac{\partial f_3}{\partial x_1} & \frac{\partial f_3}{\partial x_2}
\end{bmatrix}
=
\begin{bmatrix}
1 & 2 \\
2x_1 & 0 \\
0 & \cos(x_2)
\end{bmatrix}
$$

Each row is a directional map: how each component of $$\mathbf{f}$$ changes when you nudge $$\mathbf{x}$$.

---

### When Do Jacobians Appear in ML?

- **Neural network layers:** When an input vector flows through a layer, the weights form a Jacobian.
- **Backpropagation:** Chain rules over Jacobians allow gradients to be pushed through layers.
- **Autoencoders, GANs, normalizing flows:** Invertibility and determinant of Jacobians matter.
- **Coordinate transforms in variational inference:** Changing variables requires multiplying by Jacobian determinants.

---

### The Chain Rule for Jacobians

Let’s say we have two functions:

- $$\mathbf{f} : \mathbb{R}^n \rightarrow \mathbb{R}^m$$  
- $$\mathbf{g} : \mathbb{R}^m \rightarrow \mathbb{R}^p$$  

and we define a composition:

$$
\mathbf{h}(\mathbf{x}) = \mathbf{g}(\mathbf{f}(\mathbf{x}))
$$

Then the Jacobian of $$\mathbf{h}$$ with respect to $$\mathbf{x}$$ is given by the **matrix chain rule**:

$$
\boxed{
\frac{\partial \mathbf{h}}{\partial \mathbf{x}} =
\frac{\partial \mathbf{g}}{\partial \mathbf{f}} \cdot \frac{\partial \mathbf{f}}{\partial \mathbf{x}}
}
$$

This is just matrix multiplication — as long as the shapes line up!

> **Important:** the order matters. Matrix multiplication is not commutative.  
You apply the outer function’s Jacobian first, then multiply by the inner function’s Jacobian.

---

### Jacobian Shapes: Keep It Straight

<div style="overflow-x: auto;">
  <table class="simple-table">
    <thead>
      <tr>
        <th>Function</th>
        <th>Input Dim</th>
        <th>Output Dim</th>
        <th>Jacobian Shape</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><span style="white-space: nowrap;">\(\mathbf{f} : \mathbb{R}^n \rightarrow \mathbb{R}\)</span></td>
        <td><span style="white-space: nowrap;">\(n\)</span></td>
        <td><span style="white-space: nowrap;">\(1\)</span></td>
        <td><span style="white-space: nowrap;">\((1 \times n)\)</span> (row vector)</td>
      </tr>
      <tr>
        <td><span style="white-space: nowrap;">\(\mathbf{f} : \mathbb{R}^n \rightarrow \mathbb{R}^m\)</span></td>
        <td><span style="white-space: nowrap;">\(n\)</span></td>
        <td><span style="white-space: nowrap;">\(m\)</span></td>
        <td><span style="white-space: nowrap;">\((m \times n)\)</span> (Jacobian matrix)</td>
      </tr>
    </tbody>
  </table>
</div>

---

### Summary: Why the Jacobian Matters

- It generalizes the gradient when your function outputs multiple values.  
- It’s essential in backprop, where gradients move **backward** through layers of composed vector functions.  
- It’s the foundation for more complex structures: **Hessians**, **vector-Jacobian products**, and even **determinants of transformations**.

If you're using PyTorch or JAX, you’ve likely seen `torch.autograd.grad` or `jacrev()` — they’re computing this exact object under the hood.

---


## Matrix-by-Matrix Derivatives: $$\frac{\partial F}{\partial X}$$


In most real-world ML architectures — especially deep networks — layers don't just map vectors to scalars or vectors to vectors. Instead:

- Inputs are often **mini-batch matrices** (e.g., $$X \in \mathbb{R}^{m \times d}$$),
- Outputs of layers are **activation matrices** (e.g., $$Z \in \mathbb{R}^{m \times k}$$),
- And the transformations themselves involve **matrix-valued parameters**.

So to compute how one matrix affects another, we need to differentiate **matrix-valued functions with respect to matrices**. That’s where matrix-by-matrix derivatives come in.

---

### What Are Matrix-by-Matrix Derivatives?

Suppose:

$$
F : \mathbb{R}^{m \times n} \rightarrow \mathbb{R}^{p \times q}
$$

and you want to compute:

$$
\frac{\partial F}{\partial X}, \quad \text{where } X \in \mathbb{R}^{m \times n}, \; F(X) \in \mathbb{R}^{p \times q}
$$

Unlike scalar-valued functions, this derivative isn’t a matrix — it’s technically a **4D tensor** of shape $$p \times q \times m \times n$$. That’s hard to visualize or work with directly.

So instead, we often **vectorize** both inputs and outputs.

---

### Vectorization: Flattening the Problem

Define:

- $$\operatorname{vec}(X)$$: stacks the columns of $$X$$ into a column vector
- $$\operatorname{vec}(F(X))$$: same for the output

Then we work with the **Jacobian matrix** of vectorized output with respect to vectorized input:

$$
\boxed{
\frac{\partial \operatorname{vec}(F)}{\partial \operatorname{vec}(X)} \in \mathbb{R}^{(pq) \times (mn)}
}
$$

This is how libraries like PyTorch and TensorFlow think of matrix derivatives internally: *everything gets flattened into vectors*, and the derivative is just a big Jacobian.

---

### Example: Matrix Multiplication as a Function

Let:

$$
F(X) = AXB
$$

where:

- $$A \in \mathbb{R}^{p \times m}$$,
- $$B \in \mathbb{R}^{n \times q}$$,
- $$X \in \mathbb{R}^{m \times n}$$,
- So that $$F(X) \in \mathbb{R}^{p \times q}$$

The vectorized version of the output is:

$$
\operatorname{vec}(F(X)) = (B^\top \otimes A) \operatorname{vec}(X)
$$

> Here, $$\otimes$$ is the **Kronecker product**.

So the Jacobian (matrix-by-matrix derivative) is:

$$
\boxed{
\frac{\partial \operatorname{vec}(F)}{\partial \operatorname{vec}(X)} = B^\top \otimes A
}
$$

This rule is foundational. In fact, the entire backpropagation process in fully connected layers boils down to a series of **matrix multiplications** and this derivative.

---


### Kronecker Product

The Kronecker product $$A \otimes B$$ between two matrices creates a **block matrix**, where each entry of $$A$$ is replaced by that entry **multiplied by the entire matrix $$B$$**.

Formally, if:

$$
A = \begin{bmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{bmatrix},
\quad
B = \begin{bmatrix}
b_{11} & b_{12} \\
b_{21} & b_{22}
\end{bmatrix}
$$

then:

$$
A \otimes B =
\begin{bmatrix}
a_{11} B & a_{12} B \\
a_{21} B & a_{22} B
\end{bmatrix}
=
\begin{bmatrix}
a_{11}b_{11} & a_{11}b_{12} & a_{12}b_{11} & a_{12}b_{12} \\
a_{11}b_{21} & a_{11}b_{22} & a_{12}b_{21} & a_{12}b_{22} \\
a_{21}b_{11} & a_{21}b_{12} & a_{22}b_{11} & a_{22}b_{12} \\
a_{21}b_{21} & a_{21}b_{22} & a_{22}b_{21} & a_{22}b_{22}
\end{bmatrix}
$$

So it **inflates** a small matrix into a much larger one — and this inflated object turns out to be the Jacobian of vectorized matrix multiplication.

---

### Worked Example: Derivative of a Matrix Product

Let’s say:

$$
F(X) = AX + C
$$

where:

- $$A \in \mathbb{R}^{2 \times 2}$$  
- $$X \in \mathbb{R}^{2 \times 2}$$ is our variable  
- $$C \in \mathbb{R}^{2 \times 2}$$ is a constant bias matrix

Suppose:

$$
A = \begin{bmatrix}
1 & 2 \\
0 & -1
\end{bmatrix}, \quad
X = \begin{bmatrix}
x_1 & x_2 \\
x_3 & x_4
\end{bmatrix}, \quad
C = \begin{bmatrix}
5 & 0 \\
0 & 0
\end{bmatrix}
$$

So:

$$
F(X) = A X + C =
\begin{bmatrix}
1 & 2 \\
0 & -1
\end{bmatrix}
\begin{bmatrix}
x_1 & x_2 \\
x_3 & x_4
\end{bmatrix}
+
\begin{bmatrix}
5 & 0 \\
0 & 0
\end{bmatrix}
$$

Compute the matrix multiplication:

$$
A X = 
\begin{bmatrix}
1 \cdot x_1 + 2 \cdot x_3 & 1 \cdot x_2 + 2 \cdot x_4 \\
0 \cdot x_1 + (-1) \cdot x_3 & 0 \cdot x_2 + (-1) \cdot x_4
\end{bmatrix}
=
\begin{bmatrix}
x_1 + 2x_3 & x_2 + 2x_4 \\
- x_3 & -x_4
\end{bmatrix}
$$

Now add the bias:

$$
F(X) =
\begin{bmatrix}
x_1 + 2x_3 + 5 & x_2 + 2x_4 \\
- x_3 & -x_4
\end{bmatrix}
$$

Now let’s say we want to compute the derivative of a scalar function of this output. For instance:

$$
L = \operatorname{tr}(F(X)) = (x_1 + 2x_3 + 5) + (-x_4) = x_1 + 2x_3 - x_4 + 5
$$

Then the gradient of this scalar $$L$$ with respect to each variable in $$X$$ is:

- $$\frac{\partial L}{\partial x_1} = 1$$  
- $$\frac{\partial L}{\partial x_2} = 0$$  
- $$\frac{\partial L}{\partial x_3} = 2$$  
- $$\frac{\partial L}{\partial x_4} = -1$$

So the gradient matrix is:

$$
\frac{\partial L}{\partial X} =
\begin{bmatrix}
1 & 0 \\
2 & -1
\end{bmatrix}
$$

> This matches the rule:  
> If $$L = \operatorname{tr}(A X)$$, then $$\frac{\partial L}{\partial X} = A^\top$$

In our case, $$A^\top = \begin{bmatrix} 1 & 0 \\ 2 & -1 \end{bmatrix}$$ — exactly the matrix above.

---

### Application: Backpropagation in Deep Learning

Let’s say you have a fully connected layer:

$$
Z = XW + b
$$

with:

- $$X \in \mathbb{R}^{m \times d}$$ (input batch),
- $$W \in \mathbb{R}^{d \times k}$$ (weights),
- $$Z \in \mathbb{R}^{m \times k}$$ (pre-activation outputs),
- $$b \in \mathbb{R}^{1 \times k}$$ (broadcasted bias)

Now suppose you receive the upstream gradient:

$$
\frac{\partial L}{\partial Z} = G \in \mathbb{R}^{m \times k}
$$

You want to compute:

- $$\frac{\partial L}{\partial W}$$ — how the loss changes with respect to the weights
- $$\frac{\partial L}{\partial X}$$ — how the loss changes with respect to the inputs (useful for chaining to previous layers)

The rules are:

$$
\boxed{
\frac{\partial L}{\partial W} = X^\top G, \quad \frac{\partial L}{\partial X} = G W^\top
}
$$

These come directly from **matrix-by-matrix derivatives** of the form:

- $$\frac{\partial \operatorname{tr}(A^\top XW)}{\partial W} = X^\top A$$  
- $$\frac{\partial \operatorname{tr}(A^\top XW)}{\partial X} = A W^\top$$

> Think of these as matrix calculus chain rules — they let gradients flow *through* layers.

---

### Recap: When Do You Actually Need This?

You’ll be using matrix-by-matrix derivatives (implicitly or explicitly) whenever you:

- Backprop through matrix-valued functions in deep networks
- Implement custom layers in PyTorch or JAX
- Derive gradients in matrix factorization models (e.g., $$XW^\top$$)
- Work with vectorized representations (via `torch.nn.Linear`, `einsum`, or `reshape`)

Most of the time, you’ll rely on automatic differentiation. But understanding the underlying derivatives — and being able to write them down — is what makes debugging and custom design possible.


---

## Chain Rule in High Dimensions  
**Goal:** Efficiently compute derivatives when functions are **composed**, and **inputs/outputs are vectors or matrices**

---

### What We Already Know

In scalar calculus, you learned:

$$
\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)
$$

Now what happens if:

- Both $$f$$ and $$g$$ are **vector-valued**?
- Or they operate on **matrices**?
- And what if you have **many layers** of such functions — like in deep learning?

We need a more general, more structural rule — one that handles these layered compositions.

---

### General Setup

Let:

- $$\mathbf{x} \in \mathbb{R}^n$$  
- $$\mathbf{f} : \mathbb{R}^n \rightarrow \mathbb{R}^m$$  
- $$\mathbf{g} : \mathbb{R}^m \rightarrow \mathbb{R}^p$$

and define:

$$
\mathbf{h}(\mathbf{x}) = \mathbf{g}(\mathbf{f}(\mathbf{x}))
$$

Then the Jacobian of $$\mathbf{h}$$ is given by the **matrix chain rule**:

$$
\boxed{
\frac{\partial \mathbf{h}}{\partial \mathbf{x}} =
\frac{\partial \mathbf{g}}{\partial \mathbf{f}} \cdot \frac{\partial \mathbf{f}}{\partial \mathbf{x}}
}
$$

> Same idea as scalar calculus — but now we’re multiplying **Jacobians** instead of scalars.

---

### Shape Check

Let’s break down the dimensions to make sure it all lines up:

<div style="overflow-x: auto;">
  <table class="simple-table">
    <thead>
      <tr>
        <th>Derivative</th>
        <th>Shape</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><span style="white-space: nowrap;">\(\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\)</span></td>
        <td><span style="white-space: nowrap;">\((m \times n)\)</span></td>
      </tr>
      <tr>
        <td><span style="white-space: nowrap;">\(\frac{\partial \mathbf{g}}{\partial \mathbf{f}}\)</span></td>
        <td><span style="white-space: nowrap;">\((p \times m)\)</span></td>
      </tr>
      <tr>
        <td><strong>Total:</strong> <span style="white-space: nowrap;">\(\frac{\partial \mathbf{h}}{\partial \mathbf{x}}\)</span></td>
        <td><span style="white-space: nowrap;">\((p \times n)\)</span></td>
      </tr>
    </tbody>
  </table>
</div>

So you’re safe to multiply — as long as you’re careful with matrix shapes and multiplication order.

---

### Example: Deep Network as Function Composition

In a feedforward neural network, each layer is a function:

$$
\begin{aligned}
\mathbf{h}^{(1)} &= \sigma(W^{(1)} \mathbf{x}) \\
\mathbf{h}^{(2)} &= \sigma(W^{(2)} \mathbf{h}^{(1)}) \\
\vdots \\
\mathbf{y}_{\text{pred}} &= W^{(L)} \mathbf{h}^{(L-1)}
\end{aligned}
$$

So the full network is:

$$
\mathbf{y}_{\text{pred}} = f^{(L)} \circ f^{(L-1)} \circ \cdots \circ f^{(1)}(\mathbf{x})
$$

To compute gradients for training, we need:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} =
\frac{\partial \mathcal{L}}{\partial \mathbf{y}_{\text{pred}}} \cdot
\frac{\partial \mathbf{y}_{\text{pred}}}{\partial \mathbf{h}^{(L-1)}} \cdot
\cdots \cdot
\frac{\partial \mathbf{h}^{(1)}}{\partial \mathbf{x}}
$$

This is just **the chain rule, applied over and over**, moving from the output layer back to the inputs.

---

### Special Case: Scalar Output

If the final output is a **scalar loss** $$L$$, and intermediate steps are vector/matrix-valued, then the chain rule becomes:

$$
\boxed{
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} \cdot \frac{\partial Z}{\partial X}
}
$$

- Here $$Z$$ could be an intermediate matrix
- $$X$$ is a parameter or input
- The gradient “backflows” through the layers

This is exactly what **backpropagation** automates.

---

### One More Example (Explicit):

Let:

- $$f(\mathbf{x}) = A \mathbf{x}$$, where $$A \in \mathbb{R}^{m \times n}$$  
- $$g(\mathbf{z}) = \sigma(\mathbf{z})$$ elementwise activation

Then $$h(\mathbf{x}) = g(f(\mathbf{x})) = \sigma(A \mathbf{x})$$

You want $$\frac{\partial h}{\partial \mathbf{x}}$$. Chain rule says:

$$
\frac{\partial h}{\partial \mathbf{x}} = \frac{\partial g}{\partial \mathbf{z}} \cdot A
$$

Since $$\frac{\partial g}{\partial \mathbf{z}}$$ is a **diagonal matrix** with entries $$\sigma'(z_i)$$, this becomes:

$$
\boxed{
\frac{\partial h}{\partial \mathbf{x}} = \operatorname{diag}(\sigma'(A\mathbf{x})) \cdot A
}
$$

This kind of expression is everywhere in the derivative logic of neural nets.

---

### Why This Chain Rule Matters

- **Mathematical foundation** of backpropagation
- It generalizes gradient descent to **composite**, **high-dimensional**, **multi-layered** models
- It allows modular, local gradient computation that’s reused across layers

Whether you’re implementing custom loss functions, writing your own neural network modules, or reading research papers — this chain rule is the connective tissue that makes modern machine learning trainable.


---

## Vectorization Techniques  
*Kronecker products and the vec operator — flattening matrices, sharpening calculus*

---

### “What if we could flatten the chaos?”

That’s the idea behind vectorization.

In machine learning, we’re rarely dealing with just scalars. A single layer of a neural net transforms entire batches of feature matrices. Backpropagation involves matrices flowing through functions of other matrices — and the gradients? They must know how to flow back efficiently through all of that.

That’s where **vectorization** becomes your secret weapon.

We introduce two essential tools in matrix calculus:

- The **vec** operator: which flattens matrices to vectors  
- The **Kronecker product**: which helps us do calculus on these flattened forms  

---

### The vec Operator: Turning Matrices Into Long Skinny Vectors

Let’s start simple. Suppose you have a matrix:

$$
X =
\begin{bmatrix}
x_{11} & x_{12} \\
x_{21} & x_{22}
\end{bmatrix}
$$

Then:

$$
\operatorname{vec}(X) =
\begin{bmatrix}
x_{11} \\
x_{21} \\
x_{12} \\
x_{22}
\end{bmatrix}
$$

The rule is: **stack columns, top to bottom**.

More generally, for $$X \in \mathbb{R}^{m \times n}$$, the operator

$$
\operatorname{vec}: \mathbb{R}^{m \times n} \rightarrow \mathbb{R}^{mn}
$$

produces a column vector with the columns of $$X$$ stacked vertically.

---


As discussed in Matrix-by-Matrix Derivatives, the **Kronecker product**, denoted $$A \otimes B$$, creates a block matrix from two smaller ones. Formally:

If $$A \in \mathbb{R}^{p \times q}$$ and $$B \in \mathbb{R}^{r \times s}$$, then

$$
A \otimes B \in \mathbb{R}^{pr \times qs}
$$

and:

$$
A \otimes B =
\begin{bmatrix}
a_{11} B & a_{12} B & \cdots & a_{1q} B \\
a_{21} B & a_{22} B & \cdots & a_{2q} B \\
\vdots   & \vdots   & \ddots & \vdots \\
a_{p1} B & a_{p2} B & \cdots & a_{pq} B \\
\end{bmatrix}
$$

---

### Vectorization Identities



#### 1. $$\operatorname{vec}(AX) = (I \otimes A)\operatorname{vec}(X)$$

**Proof**:

Let $$X \in \mathbb{R}^{m \times n}$$. Then the $$j^{\text{th}}$$ column of $$AX$$ is:

$$
AX_{[:, j]} = A \cdot \text{(column } j \text{ of } X)
$$

So:

$$
\operatorname{vec}(AX) =
\begin{bmatrix}
A X_{[:, 1]} \\
A X_{[:, 2]} \\
\vdots \\
A X_{[:, n]}
\end{bmatrix}
= (I_n \otimes A)\, \operatorname{vec}(X)
$$

This identity follows by applying $$A$$ to each column of $$X$$ and stacking the results — which is precisely what the Kronecker product $$I_n \otimes A$$ does when multiplying $$\operatorname{vec}(X)$$.

---

#### 2. $$\operatorname{vec}(X B) = (B^\top \otimes I)\operatorname{vec}(X)$$

**Proof**:

Let $$X \in \mathbb{R}^{m \times n},\ B \in \mathbb{R}^{n \times p}$$.

Each row of the matrix product $$XB$$ is a linear combination of the rows of $$B$$ weighted by the entries of $$X$$.

So the $$i^{\text{th}}$$ row of $$XB$$ is:

$$
(XB)_{i,:} = \sum_{j=1}^{n} X_{i,j} B_{j,:}
$$

This row-wise operation corresponds to multiplying $$\operatorname{vec}(X)$$ by $$B^\top \otimes I$$.

Hence:

$$
\operatorname{vec}(X B) = (B^\top \otimes I)\, \operatorname{vec}(X)
$$

---

#### 3. $$\operatorname{vec}(A X B) = (B^\top \otimes A)\operatorname{vec}(X)$$


This elegant identity lies at the heart of matrix calculus and vectorization. It allows us to reduce a seemingly complex matrix expression into a clean, computable form — and it appears everywhere from backpropagation mechanics to second-order optimization.

Let’s walk through it step by step.

---

Suppose we have a function defined as:

$$
F(X) = A X B
$$

where:

$$
A \in \mathbb{R}^{p \times m}, \quad
X \in \mathbb{R}^{m \times n}, \quad
B \in \mathbb{R}^{n \times q}
$$

Then the output is:

$$
F(X) \in \mathbb{R}^{p \times q}
$$

Our goal is to compute the **vectorized** form of this result — that is, express:

$$
\operatorname{vec}(F(X)) = \operatorname{vec}(A X B)
$$

in terms of $$\operatorname{vec}(X)$$.

The identity we aim to derive is:

$$
\boxed{
\operatorname{vec}(A X B) = (B^\top \otimes A)\, \operatorname{vec}(X)
}
$$

---

We’ll build this from two well-established vectorization rules:

1. If $$A \in \mathbb{R}^{p \times m}$$ and $$X \in \mathbb{R}^{m \times n}$$, then:

$$
\operatorname{vec}(A X) = (I_n \otimes A)\, \operatorname{vec}(X)
$$

2. If $$X \in \mathbb{R}^{m \times n}$$ and $$B \in \mathbb{R}^{n \times q}$$, then:

$$
\operatorname{vec}(X B) = (B^\top \otimes I_m)\, \operatorname{vec}(X)
$$

These identities essentially say: you can translate matrix multiplication into Kronecker product multiplication after flattening.

---

Now rewrite the expression:

$$
A X B = (A X) B
$$

and apply the vectorization rule for post-multiplication:

$$
\begin{aligned}
\operatorname{vec}(A X B)
&= \operatorname{vec}((A X) B) \\
&= (B^\top \otimes I_p)\, \operatorname{vec}(A X)
\end{aligned}
$$

Then apply the pre-multiplication identity to $$\operatorname{vec}(A X)$$:

$$
\operatorname{vec}(A X) = (I_n \otimes A)\, \operatorname{vec}(X)
$$

Putting it all together:

$$
\begin{aligned}
\operatorname{vec}(A X B)
&= (B^\top \otimes I_p)\, (I_n \otimes A)\, \operatorname{vec}(X)
\end{aligned}
$$

---

The final step is to simplify this Kronecker product composition. A standard identity in Kronecker algebra tells us:

$$
(B^\top \otimes I_p)(I_n \otimes A) = B^\top \otimes A
$$

This follows from the mixed-product property:

$$
(A \otimes B)(C \otimes D) = (AC) \otimes (BD)
$$

when the matrix dimensions are compatible.

So we conclude:

$$
\operatorname{vec}(A X B) = (B^\top \otimes A)\, \operatorname{vec}(X)
$$

Exactly as claimed.


---




### Example: Quadratic Form with Vectorization

Let:

$$
L = \operatorname{tr}(A X B X^\top)
$$

We want to compute:

$$
\frac{\partial L}{\partial X}
$$

This expression arises in:

- Mahalanobis distance: $$\mathbf{x}^\top \Sigma^{-1} \mathbf{x}$$  
- Matrix-variate Gaussians  
- Multivariate optimization problems

**Step 1: Use vec and trace identity**

Use the identity:

$$
\operatorname{tr}(A X B X^\top) = \operatorname{vec}(X)^\top (B^\top \otimes A) \operatorname{vec}(X)
$$

**Proof sketch**:

We know:

$$
\operatorname{tr}(A X B X^\top)
= \operatorname{tr}(X^\top A X B)
= \operatorname{vec}(X)^\top (B^\top \otimes A) \operatorname{vec}(X)
$$

(See Petersen & Pedersen "The Matrix Cookbook" for this identity.)

**Step 2: Take derivative of quadratic form**

Let:

$$
L = \mathbf{x}^\top Q \mathbf{x}
$$

Then:

$$
\frac{dL}{d\mathbf{x}} = (Q + Q^\top)\, \mathbf{x}
$$

When $$Q$$ is symmetric, this becomes:

$$
\frac{dL}{d\mathbf{x}} = 2 Q \mathbf{x}
$$

In our case:

- $$\mathbf{x} = \operatorname{vec}(X)$$  
- $$Q = B^\top \otimes A$$ (symmetric if $$B = B^\top$$ and $$A = A^\top$$)

So:

$$
\frac{\partial L}{\partial \operatorname{vec}(X)} = 2 (B^\top \otimes A) \operatorname{vec}(X)
$$

If needed, reshape back to matrix:

$$
\frac{\partial L}{\partial X} = \text{reshape to } m \times n
$$

---




### Summary 



- The `vec` operator helps translate matrix calculus into vector calculus  
- The Kronecker product allows those translations to become clean linear maps  
- Useful identities like $$\operatorname{vec}(A X B) = (B^\top \otimes A)\, \operatorname{vec}(X)$$  
  give us elegant tools to differentiate messy matrix functions

---


## Trace and determinant derivatives



### Why This Matters

If you’ve spent any time with loss functions in linear models, Gaussian likelihoods, matrix normal distributions, or multivariate optimization, you’ve seen trace and determinant terms pop up.

In this section, we’ll explore:

- The **linearity and cyclic rules** for traces  
- How to differentiate **trace expressions** with respect to matrices  
- What the **Jacobi formula** tells us about the derivative of a determinant  
- And why these results are so useful in machine learning

Let’s begin with the friendliest one of them all: the trace.

---

### The Trace Operator: Definition and Properties

For any square matrix $$A \in \mathbb{R}^{n \times n}$$:

$$
\operatorname{tr}(A) = \sum_{i=1}^{n} A_{ii}
$$

That’s it — just the sum of the diagonal entries.

But this innocent-looking function hides two powerful algebraic properties that make it a star in matrix calculus:

1. **Linearity**:  
   $$
   \operatorname{tr}(A + B) = \operatorname{tr}(A) + \operatorname{tr}(B), \quad \operatorname{tr}(c A) = c \operatorname{tr}(A)
   $$

2. **Cyclic Permutation**:  
   $$
   \operatorname{tr}(A B) = \operatorname{tr}(B A)
   $$

   More generally:

   $$
   \operatorname{tr}(A B C) = \operatorname{tr}(C A B) = \operatorname{tr}(B C A)
   $$

   **Caution**: You can cyclically permute **but not reorder arbitrarily**.

This second rule is especially important. It allows us to rearrange matrix products inside a trace to our advantage when differentiating.

---

### Derivatives Involving the Trace

Let’s build some derivative identities now, starting with the trace of matrix expressions. These are derived using component-wise differentiation and the properties above.

#### 1. Scalar by Matrix:

Let $$A \in \mathbb{R}^{n \times n}$$ and $$X \in \mathbb{R}^{n \times n}$$.

Then:

$$
\frac{\partial}{\partial X} \operatorname{tr}(A^\top X) = A
$$

This one is straightforward. It’s the matrix analog of the scalar identity $$\frac{d}{dx} (a x) = a$$.

More generally, for any expression $$\operatorname{tr}(A X)$$ or $$\operatorname{tr}(X A)$$:

$$
\frac{\partial}{\partial X} \operatorname{tr}(A X) = A^\top
$$

(because differentiation is with respect to each entry of $$X$$, not the transpose.)

#### 2. Quadratic Forms:

Let’s say $$X \in \mathbb{R}^{n \times p}$$, and $$A \in \mathbb{R}^{p \times p}$$ is symmetric.

Then:

$$
\frac{\partial}{\partial X} \operatorname{tr}(X A X^\top) = X (A + A^\top)
$$

And if $$A$$ is symmetric (common in ML):

$$
\frac{\partial}{\partial X} \operatorname{tr}(X A X^\top) = 2 X A
$$

This comes up often in Mahalanobis distances and multivariate Gaussians.

#### 3. Frobenius Norm:

The Frobenius norm of a matrix $$X$$ is:

$$
\|X\|_F^2 = \operatorname{tr}(X^\top X)
$$

Then:

$$
\frac{\partial}{\partial X} \|X\|_F^2 = 2 X
$$

This result powers the regularization terms in Ridge regression and neural network weight decay.

---

### Jacobi’s Formula: The Determinant Derivative

This one is a bit more subtle but incredibly important.

Let $$X \in \mathbb{R}^{n \times n}$$ be an invertible matrix. Then:

$$
\frac{d}{dX} \det(X) = \det(X) \cdot (X^{-1})^\top
$$

So in full derivative form:

$$
\frac{\partial}{\partial X} \log \det(X) = (X^{-1})^\top
$$

This is known as **Jacobi’s formula**, and it plays a starring role in probabilistic modeling — especially when working with log-likelihoods of Gaussian distributions.

It gives us:

- A way to differentiate log-determinants in closed form  
- The building block for matrix-variate distributions (like the matrix normal)  
- Clean expressions for entropy and KL divergence in terms of covariance matrices

---

### Real-World Machine Learning Examples

- In **multivariate Gaussian log-likelihoods**, the term  
  $$
  \log \det(\Sigma)
  $$  
  appears in the normalization constant. Its gradient? Exactly what Jacobi gives us.

- In **information theory**, when computing KL divergence between Gaussians, we encounter:  
  $$
  \operatorname{tr}(\Sigma_2^{-1} \Sigma_1) - \log \frac{\det(\Sigma_1)}{\det(\Sigma_2)}
  $$  
  and the derivatives of trace and log-determinant terms show up in natural gradient methods.

- In **regularization**, the Frobenius norm penalty translates to  
  $$
  \lambda \operatorname{tr}(W^\top W)
  $$  
  and the derivative is just $$2 \lambda W$$.

---

Perfect — let’s now step into a concept that often feels like black magic the first time you see it: **how to differentiate through a matrix inverse**.

We'll walk through it carefully, building intuition and math from the ground up — and as always, **all math is inside `$$...$$` for clean Jekyll rendering**.

---

## Derivatives of Matrix Inverses  


### Why You Need This

If you’re working with second-order optimization (e.g. Newton’s method), statistical estimation (like linear or multivariate Gaussian models), or algorithms that involve solving matrix equations repeatedly, then derivatives of **inverse matrices** are everywhere.

And yet... the inverse operation is **nonlinear**, **non-elementwise**, and **nontrivial** to differentiate.

Still, with some matrix calculus magic, we can express this derivative in closed form. And once you know it, you’ll see it pop up all over machine learning theory.

---

### The Goal

Let’s say $$X \in \mathbb{R}^{n \times n}$$ is an **invertible**, differentiable matrix-valued function.

We want to compute:

$$
\frac{\partial}{\partial X} (X^{-1})
$$

That is, how does the inverse of a matrix change as we perturb the original matrix?

---

### The Key Identity

The answer is:

$$
\boxed{
\frac{d}{dX} (X^{-1}) = -X^{-1} \left( \frac{dX}{dX} \right) X^{-1}
}
$$

In practice, this becomes:

$$
\frac{\partial X^{-1}}{\partial X} = -X^{-1} \otimes X^{-1}
$$

Let’s unpack that.

This means: if you **flatten** the output matrix $$X^{-1}$$ into a vector (via the vec operator), and do the same for the input matrix $$X$$, then the Jacobian is a big $$n^2 \times n^2$$ matrix given by:

$$
\operatorname{vec}(dX^{-1}) = - (X^{-1} \otimes X^{-1})\, \operatorname{vec}(dX)
$$

This is your tool for automatic differentiation, matrix backprop, and symbolic derivations in second-order methods.

---

### Derivation: From First Principles

We begin with a classic trick:

If $$X$$ is invertible, then by definition:

$$
X X^{-1} = I
$$

Differentiate both sides:

$$
d(X X^{-1}) = dI = 0
$$

Now apply the product rule:

$$
dX \cdot X^{-1} + X \cdot d(X^{-1}) = 0
$$

Rearranging:

$$
X \cdot d(X^{-1}) = -dX \cdot X^{-1}
$$

Multiply both sides on the left by $$X^{-1}$$:

$$
d(X^{-1}) = - X^{-1} dX X^{-1}
$$

This gives us a compact and powerful result:

$$
\boxed{
d(X^{-1}) = - X^{-1} dX X^{-1}
}
$$

This is the **differential** form. It can be translated into vectorized form via the Kronecker product if needed for Jacobians or autodiff systems.

---

### Practical Use Case: Newton's Method

In optimization, Newton's method uses the update:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - H^{-1} \nabla f
$$

where $$H$$ is the Hessian matrix of the loss function.

To compute derivatives of the update rule itself (for meta-learning, for example), you may need to differentiate through $$H^{-1}$$ — and this is exactly where our identity helps.

---

### In Statistics: Covariance Matrix Inverses

In multivariate Gaussian distributions, the likelihood often includes the term:

$$
(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})
$$

To perform MLE or Bayesian estimation, we need the gradient with respect to $$\Sigma$$, the covariance matrix.

Differentiating through $$\Sigma^{-1}$$ uses:

$$
\frac{\partial \Sigma^{-1}}{\partial \Sigma} = -\Sigma^{-1} \otimes \Sigma^{-1}
$$

This result appears in:

- Natural gradient descent  
- Fisher information matrix  
- EM algorithm updates  
- Precision matrix estimation in graphical models

---

### Summary Box

<div style="overflow-x: auto;">
  <table class="simple-table">
    <thead>
      <tr>
        <th>Expression</th>
        <th>Derivative</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><span style="white-space: nowrap;">\(d(X^{-1})\)</span></td>
        <td><span style="white-space: nowrap;">\(-X^{-1} dX X^{-1}\)</span></td>
        <td>Differential form</td>
      </tr>
      <tr>
        <td><span style="white-space: nowrap;">\(\operatorname{vec}(dX^{-1})\)</span></td>
        <td><span style="white-space: nowrap;">\(- (X^{-1} \otimes X^{-1}) \operatorname{vec}(dX)\)</span></td>
        <td>Vectorized form</td>
      </tr>
      <tr>
        <td><span style="white-space: nowrap;">\(\frac{\partial}{\partial X} (X^{-1})\)</span></td>
        <td><span style="white-space: nowrap;">\(- X^{-1} \otimes X^{-1}\)</span></td>
        <td>Jacobian matrix (optional)</td>
      </tr>
    </tbody>
  </table>
</div>

---

### Closing Thought

Matrix inverses may feel tricky — and rightly so. But this compact identity:

$$
d(X^{-1}) = - X^{-1} dX X^{-1}
$$

gives you the full derivative in one clean sweep. Whether you're modeling covariance matrices or optimizing matrix-valued functions, this result will keep showing up. It’s one of the most useful “mental macros” in all of matrix calculus.

---

## Derivatives of Eigenvalues and Eigenvectors  
*When matrices change, what happens to their spectrum?*

---

### The Setup

Let $$A \in \mathbb{R}^{n \times n}$$ be a real symmetric matrix. Suppose:

$$
A \mathbf{v} = \lambda \mathbf{v}
$$

with:

- $$\lambda$$ being an eigenvalue  
- $$\mathbf{v}$$ the corresponding (unit-norm) eigenvector: $$\|\mathbf{v}\| = 1$$

Now we perturb $$A$$ slightly — how do $$\lambda$$ and $$\mathbf{v}$$ change?

This is a **sensitivity analysis** problem: if $$A \rightarrow A + \varepsilon H$$, what is:

- $$\frac{d\lambda}{d\varepsilon}$$  
- $$\frac{d\mathbf{v}}{d\varepsilon}$$

---

### 1. Derivative of the Eigenvalue

Let’s start with the easier one.

Assume $$\mathbf{v}$$ is normalized and $$A$$ is symmetric.

Then the derivative of the eigenvalue $$\lambda$$ with respect to a perturbation $$H$$ is:

$$
\boxed{
\frac{d\lambda}{d\varepsilon} = \mathbf{v}^\top H \mathbf{v}
}
$$

This means: if you nudge the matrix $$A$$ in the direction of some matrix $$H$$, the eigenvalue $$\lambda$$ changes at a rate given by this simple quadratic form.

**Interpretation:**  
It’s just the Rayleigh quotient of $$H$$ with respect to $$\mathbf{v}$$ — a beautifully compact expression.

---

### 2. Derivative of the Eigenvector

This part is trickier, and assumes $$\lambda$$ is **simple** (has multiplicity one).

Let $$A$$ be symmetric, and suppose:

$$
A \mathbf{v}_i = \lambda_i \mathbf{v}_i
$$

Let $$H$$ be the perturbation direction for $$A$$.

Then, the directional derivative of the eigenvector $$\mathbf{v}_i$$ is:

$$
\boxed{
\frac{d\mathbf{v}_i}{d\varepsilon}
= \sum_{j \ne i} \frac{\mathbf{v}_j^\top H \mathbf{v}_i}{\lambda_i - \lambda_j}\, \mathbf{v}_j
}
$$

So the rate of change of $$\mathbf{v}_i$$ depends on:

- All the **other** eigenvectors $$\mathbf{v}_j$$  
- The **spectral gap** $$\lambda_i - \lambda_j$$  
- How the perturbation $$H$$ interacts with them

---

### Why This Matters for PCA

In Principal Component Analysis (PCA), we compute the eigenvectors of a covariance matrix:

$$
C = \frac{1}{n} X^\top X
$$

The first principal component is the eigenvector $$\mathbf{v}_1$$ corresponding to the **largest** eigenvalue.

If your data $$X$$ changes (e.g., online updates or batch re-training), the covariance matrix changes too. These derivative formulas help answer:

- How **stable** are the principal directions to noise?
- How can we **differentiate through PCA** for optimization (e.g., in differentiable dimensionality reduction)?

These insights lead to:

- Differentiable subspace learning (e.g. in neural networks that embed PCA steps)  
- Understanding convergence of online PCA algorithms  
- Regularizing PCA when eigenvalues are close together (small spectral gap → unstable eigenvectors)

---

### Real-World Analogy

Think of a guitar string.

- The **tension** of the string is like your matrix $$A$$  
- The **frequencies** (eigenvalues) change smoothly as you tune the tension  
- But the **shape of vibration modes** (eigenvectors) can **flip wildly** when two frequencies get close

This is why eigenvector derivatives are more fragile, and why **PCA can become unstable** when leading eigenvalues are nearly equal.

---



### Closing Notes

- The eigenvalue derivative is simple and intuitive — it's just the projection of your perturbation onto the eigendirection.  
- The eigenvector derivative is messier — but critical when PCA must be **differentiated**, **stabilized**, or **learned**.  
- These ideas generalize to **singular value decomposition (SVD)**, which powers many other dimensionality reduction techniques.

---



## Derivatives of Matrix Norms  
*Regularization’s best friend: the Frobenius norm and how it learns*

---

### What’s a Matrix Norm, and Why Do We Care?

In optimization, we often need to **measure the “size” of a matrix**. This comes up when:

- We want to penalize large weights in linear regression  
- We apply weight decay in neural nets  
- We impose smoothness or sparsity in optimization problems

The most common choice? The **Frobenius norm**, defined for a matrix $$X \in \mathbb{R}^{m \times n}$$ as:

$$
\|X\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} X_{ij}^2}
= \sqrt{\operatorname{tr}(X^\top X)}
$$

This is just the Euclidean norm of the matrix when viewed as a long vector.

---

### The Squared Frobenius Norm

We usually work with the **squared** Frobenius norm in loss functions, since it's differentiable everywhere and easier to manipulate:

$$
\|X\|_F^2 = \sum_{i,j} X_{ij}^2 = \operatorname{tr}(X^\top X)
$$

This term frequently appears in regularization penalties. For instance, in Ridge regression:

$$
L(\mathbf{w}) = \frac{1}{2m} \|X\mathbf{w} - \mathbf{y}\|^2 + \frac{\lambda}{2} \|\mathbf{w}\|^2
$$

That second term is just the Frobenius norm of a vector — which is equivalent to the $$\ell_2$$ norm.

---

### The Derivative of the Frobenius Norm

Let’s derive the gradient step by step.

#### If $$X$$ is a scalar:

Nothing new — this is just:

$$
\frac{d}{dX} X^2 = 2X
$$

#### If $$X$$ is a vector:

Same rule applies:

$$
\frac{d}{dX} \|X\|^2 = 2X
$$

#### Now for a full matrix:

Let $$X \in \mathbb{R}^{m \times n}$$. Then:

$$
\|X\|_F^2 = \operatorname{tr}(X^\top X)
$$

Using the identity:

$$
\frac{\partial}{\partial X} \operatorname{tr}(X^\top X) = 2X
$$

we conclude:

$$
\boxed{
\frac{\partial}{\partial X} \|X\|_F^2 = 2X
}
$$

This is remarkably clean: **same rule applies, no matter the dimension**.

---

### Application: Ridge Regression

Ridge regression adds an $$\ell_2$$ penalty on the weights:

$$
L(\mathbf{w}) = \frac{1}{2m} \|X\mathbf{w} - \mathbf{y}\|^2 + \frac{\lambda}{2} \|\mathbf{w}\|^2
$$

Take the derivative of the second term:

$$
\frac{d}{d\mathbf{w}} \left( \frac{\lambda}{2} \|\mathbf{w}\|^2 \right) = \lambda \mathbf{w}
$$

This leads to the closed-form solution:

$$
\mathbf{w}^* = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}
$$

That $$\lambda I$$ term comes **directly** from differentiating the norm penalty.

---

### Application: Neural Network Weight Decay

In deep learning, weight decay (also called $$\ell_2$$ regularization) penalizes large parameter magnitudes.

If your loss is:

$$
L(\theta) = \text{CrossEntropy} + \frac{\lambda}{2} \sum_{l} \|W^{(l)}\|_F^2
$$

then each gradient step gets:

$$
\frac{\partial L}{\partial W^{(l)}} = \text{Backprop gradient} + \lambda W^{(l)}
$$

Again — the derivative of the squared Frobenius norm is **just the weight matrix itself**, scaled by $$\lambda$$.

---



### Takeaways

- The Frobenius norm is just the Euclidean norm generalized to matrices.  
- Its squared version has a clean derivative: $$2X$$.  
- This simplicity makes it a natural choice for regularization in many machine learning models.  
- If you're implementing Ridge regression or weight decay, you're already using this identity — whether you knew it or not.

---


## Derivatives of Structured Matrices  



### Why Structure Matters

Most real-world machine learning problems — especially those involving dimensionality reduction, optimization over manifolds, or latent variable models — don’t deal with arbitrary matrices.

They often involve:

- **Symmetric matrices** (e.g., covariance matrices, Hessians)
- **Orthogonal matrices** (e.g., PCA loadings, rotation matrices, SVD components)

These structures aren't accidental. They encode geometric constraints and ensure the results make physical, statistical, or optimization sense.

The challenge? **Their derivatives must also respect the structure.**

---

### 1. Differentiating Symmetric Matrices

Let’s say $$S \in \mathbb{R}^{n \times n}$$ is symmetric:  
$$S = S^\top$$

Then if you define a scalar-valued function:

$$
f(S) = \operatorname{tr}(S A)
$$

with $$A$$ being arbitrary, the naive derivative $$\frac{\partial f}{\partial S} = A$$ doesn’t always preserve symmetry.

To ensure the gradient remains symmetric, we must **symmetrize** the derivative:

$$
\boxed{
\frac{\partial f}{\partial S} = \frac{1}{2}(A + A^\top)
}
$$

Why? Because any variation $$dS$$ must satisfy $$dS = dS^\top$$ — the space of symmetric matrices is a subspace, and we’re projecting gradients onto that subspace.

---

#### Practical Example: Covariance Matrix Estimation

Suppose you estimate a sample covariance matrix:

$$
\Sigma = \frac{1}{n} X^\top X
$$

Now you're optimizing a likelihood involving $$\Sigma$$ — but you must ensure it stays symmetric and positive semi-definite.

So, if the loss contains:

$$
L = \log \det \Sigma + \operatorname{tr}(A \Sigma)
$$

you differentiate and then **symmetrize** the result:

$$
\frac{\partial L}{\partial \Sigma} = \Sigma^{-1} + A \quad \Rightarrow \quad \text{Use } \frac{1}{2}(\Sigma^{-1} + A + \Sigma^{-1} + A)^\top
$$

---

### 2. Differentiating Orthogonal Matrices

Let’s now consider orthogonal matrices:

$$
Q^\top Q = I
$$

Common in:

- PCA (principal axes)
- SVD (U and V matrices)
- Rotation layers in computer vision

The constraint introduces **nontrivial geometry**: the set of orthogonal matrices lies on a **manifold**, meaning you can't perturb $$Q$$ arbitrarily and still stay on the manifold.

So, if you're optimizing a loss:

$$
L(Q)
$$

subject to $$Q^\top Q = I$$, then you must restrict derivatives to the **tangent space** of the orthogonal group.

---

### Tangent Space to Orthogonal Matrices

If $$Q \in \mathbb{R}^{n \times n}$$ is orthogonal, its tangent vectors $$dQ$$ must satisfy:

$$
Q^\top dQ + dQ^\top Q = 0
$$

This is the **skew-symmetry condition**. So: the directional derivative $$dQ$$ must lie in the space of **skew-symmetric** matrices.

Therefore, any gradient $$\nabla_Q L$$ must be **projected** onto this tangent space.

---

### Projected Gradient on Orthogonal Group

To get a valid gradient that stays on the manifold:

$$
\boxed{
\text{Proj}_{\mathcal{T}_Q}(\nabla_Q L) = \nabla_Q L - Q \nabla_Q L^\top Q
}
$$

This is the Riemannian gradient — it tells you how to move **along the manifold** of orthogonal matrices.

You’ll encounter this in **optimization over SO(n)** or **Stiefel manifolds** — used in:

- Variational inference  
- Geometric deep learning  
- ICA and dimensionality reduction  


---

### Why It Matters

- **Symmetric matrices** dominate in second-order methods, covariance estimation, and graphical models  
- **Orthogonal matrices** arise in projection methods, PCA, and rotation-based learning algorithms  
- Derivatives that ignore structure may leak off the feasible region, making optimization unstable or incorrect

So: when you know your matrices have structure, you **must let your gradients know too**.

---



## Generalized and Pseudo-Inverses  


### Why Pseudoinverses?

In many real-world problems, we encounter **non-square matrices** — where the inverse doesn’t exist. Or worse, **ill-conditioned** matrices, where the inverse is unstable or poorly defined.

That’s where the **Moore–Penrose pseudoinverse** comes in. It gives us a principled way to “invert” rectangular or singular matrices and find **least-squares solutions**.

---

### Definition of the Moore–Penrose Pseudoinverse

Given any matrix $$A \in \mathbb{R}^{m \times n}$$, its **Moore–Penrose pseudoinverse** is denoted:

$$
A^+ \in \mathbb{R}^{n \times m}
$$

and satisfies four conditions:

1. $$A A^+ A = A$$  
2. $$A^+ A A^+ = A^+$$  
3. $$(A A^+)^\top = A A^+$$  
4. $$(A^+ A)^\top = A^+ A$$

This defines a **unique** generalized inverse.

---

### Computing the Pseudoinverse via SVD

If we compute the singular value decomposition:

$$
A = U \Sigma V^\top
$$

where:

- $$U \in \mathbb{R}^{m \times m}$$  
- $$V \in \mathbb{R}^{n \times n}$$  
- $$\Sigma \in \mathbb{R}^{m \times n}$$ is diagonal with singular values $$\sigma_i$$

Then the pseudoinverse is:

$$
A^+ = V \Sigma^+ U^\top
$$

where:

- $$\Sigma^+$$ is obtained by inverting each non-zero singular value $$\sigma_i$$ and transposing the matrix.

If $$\Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_r, 0, \dots, 0)$$, then:

$$
\Sigma^+ = \operatorname{diag}\left(\frac{1}{\sigma_1}, \dots, \frac{1}{\sigma_r}, 0, \dots, 0\right)^\top
$$

---

### Application: Least Squares Solution

Suppose you want to solve the system:

$$
A \mathbf{x} = \mathbf{b}
$$

When $$A$$ is tall and thin ($$m > n$$) and **has full rank**, this is **overdetermined**, and no exact solution may exist.

The best we can do is minimize:

$$
\min_{\mathbf{x}} \|A \mathbf{x} - \mathbf{b}\|^2
$$

The solution is given by:

$$
\boxed{
\mathbf{x}^* = A^+ \mathbf{b}
}
$$

And if $$A$$ has full column rank, we recover the familiar normal equation:

$$
A^+ = (A^\top A)^{-1} A^\top
$$

This is the expression used in closed-form **linear regression**.

---

### Application: Ridge Regression and Stability

When $$A^\top A$$ is **ill-conditioned**, pseudoinverses help with **regularization**.

In Ridge regression, we replace:

$$
(A^\top A)^{-1} A^\top
$$

with:

$$
(A^\top A + \lambda I)^{-1} A^\top
$$

This is a **regularized pseudoinverse**, and it's **more numerically stable** — small singular values are not inverted all the way.

---

### Geometric Interpretation

Let’s suppose:

- $$A$$ maps from $$\mathbb{R}^n$$ to $$\mathbb{R}^m$$  
- We want to find a solution $$\mathbf{x}$$ such that $$A \mathbf{x} \approx \mathbf{b}$$

Then:

- $$A^+ \mathbf{b}$$ is the point in the **column space of A** that’s closest to $$\mathbf{b}$$  
- Among all such solutions, it finds the one with **minimum Euclidean norm**

This is what makes $$A^+$$ **the canonical least-norm solution**.

---



### Takeaways

- The pseudoinverse is **central to least squares**, linear models, and underdetermined systems  
- It **extends matrix inversion** to non-square or rank-deficient matrices  
- Using SVD to compute $$A^+$$ is **numerically stable**, especially when $$A^\top A$$ is near singular  
- Pseudoinverses show up in **Ridge regression**, **low-rank approximations**, and even **deep learning** for layer inversion and model analysis

---



## Matrix Decompositions and Their Derivatives  
*Breaking matrices apart — and tracking how they move when perturbed*

---

### Why Matrix Decompositions Matter in ML

Matrix decompositions are how we **understand**, **compress**, and **solve** linear systems — especially when direct inversion or full eigendecompositions are too expensive or unstable.

You’ll see:

- **LU decomposition** in solvers and backpropagation through linear systems  
- **QR decomposition** in optimization problems with orthogonality constraints  
- **SVD** in PCA, collaborative filtering, and matrix factorization

But beyond just computing them, we often need to **differentiate through** these decompositions — especially in deep learning or probabilistic models where **intermediate matrix steps are part of the learnable pipeline**.

---

### 1. LU Decomposition and Derivatives

The LU decomposition writes a square matrix as:

$$
A = L U
$$

where:

- $$L$$ is lower triangular with ones on the diagonal  
- $$U$$ is upper triangular

This is primarily used in solving linear systems efficiently, but not all matrices have an LU decomposition without **row pivoting**.

If:

$$
A \in \mathbb{R}^{n \times n}, \quad A = LU
$$

and we perturb $$A \rightarrow A + \delta A$$, then under certain regularity conditions:

$$
\delta A = \delta L \cdot U + L \cdot \delta U
$$

One can derive expressions for $$\delta L$$ and $$\delta U$$ using triangular projection operators, but LU is **not differentiable everywhere**, especially where row swaps (pivoting) occur.

Hence: **LU derivatives are useful for solvers**, but less common in end-to-end differentiable models.

---

### 2. QR Decomposition and Derivatives

Every matrix $$A \in \mathbb{R}^{m \times n}$$ ($$m \geq n$$) has a **QR decomposition**:

$$
A = Q R
$$

where:

- $$Q \in \mathbb{R}^{m \times n}$$ has orthonormal columns  
- $$R \in \mathbb{R}^{n \times n}$$ is upper triangular

This is powerful when you want to **orthonormalize** matrices — used in:

- Least squares solvers  
- Constrained optimization on the Stiefel manifold  
- Regression with orthogonality constraints

#### Derivative (Sketch)

If $$A = QR$$ and we perturb $$A \rightarrow A + \delta A$$, then:

$$
\delta A = \delta Q \cdot R + Q \cdot \delta R
$$

But since $$Q^\top Q = I$$, $$\delta Q$$ must lie in the **tangent space to the Stiefel manifold**. Therefore:

$$
Q^\top \delta Q + \delta Q^\top Q = 0
$$

So the derivative of $$Q$$ must be **skew-symmetric**, as with orthogonal matrices.

Libraries like PyTorch and JAX now implement **differentiable QR** for use in deep learning layers.

---

### 3. SVD and Its Derivatives

This is the most celebrated decomposition:

$$
A = U \Sigma V^\top
$$

- $$U \in \mathbb{R}^{m \times m}$$, orthogonal  
- $$\Sigma \in \mathbb{R}^{m \times n}$$, diagonal (singular values)  
- $$V \in \mathbb{R}^{n \times n}$$, orthogonal

It underpins:

- **PCA** (projecting onto directions of maximal variance)  
- **Low-rank approximations**  
- **Collaborative filtering**, topic models, embedding compression  

#### Derivatives of SVD

Suppose we perturb $$A \rightarrow A + \delta A$$ and want to track how $$U, \Sigma, V$$ change.

This is non-trivial because:

- The singular values may be **non-differentiable** when repeated  
- The eigenvectors (columns of $$U$$ and $$V$$) may **rotate discontinuously**

Still, under full-rank, non-degenerate singular values, we have:

##### Derivative of $$\Sigma$$:

Let $$\sigma_i$$ be the $$i^{th}$$ singular value.

Then:

$$
\delta \sigma_i = u_i^\top \, \delta A \, v_i
$$

That is: the change in the $$i^{th}$$ singular value is the projection of $$\delta A$$ onto the $$i^{th}$$ left and right singular vectors.

##### Derivatives of $$U, V$$:

More complex, but involves **resolvent operators** and **eigenvalue gap assumptions**.

If you’re building optimization algorithms or probabilistic models involving SVD, tools like [Autograd/JAX](https://jax.readthedocs.io/en/latest/notebooks/SVD_derivatives.html) and [Matrix Cookbook](https://matrixcookbook.com) can help.

---

### Application: PCA via SVD

To perform PCA:

- Center the data: $$X \leftarrow X - \bar{X}$$  
- Compute SVD: $$X = U \Sigma V^\top$$  
- Take top $$k$$ singular vectors: $$V_k$$  
- Project: $$Z = X V_k$$

This compresses data to $$k$$ dimensions.

Now, suppose your **PCA directions are part of a learnable pipeline** (e.g., in unsupervised learning). You need to **differentiate through the SVD**, which is why automatic SVD gradients are crucial in deep learning libraries.

---

### Summary Table

<div style="overflow-x: auto;">
  <table class="simple-table">
    <thead>
      <tr>
        <th>Decomposition</th>
        <th>Form</th>
        <th>Used For</th>
        <th>Derivative Highlights</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>LU</td>
        <td>\(A = LU\)</td>
        <td>Linear system solvers</td>
        <td>Non-differentiable at pivot points</td>
      </tr>
      <tr>
        <td>QR</td>
        <td>\(A = QR\)</td>
        <td>Orthonormal basis, least squares</td>
        <td>Project derivatives onto Stiefel manifold</td>
      </tr>
      <tr>
        <td>SVD</td>
        <td>\(A = U \Sigma V^\top\)</td>
        <td>PCA, low-rank approximation</td>
        <td>Delicate near degenerate singular values</td>
      </tr>
    </tbody>
  </table>
</div>

---

### Final Notes

- Matrix decompositions are **not just algebraic tools** — they appear in backpropagation, variational inference, and matrix estimation tasks  
- Differentiating through decompositions allows **end-to-end learning** when layers include matrix ops like PCA, whitening, or factorization  
- Automatic differentiation libraries now support **SVD/QR gradients**, but knowing the theory helps when things go wrong

---



## Multivariate Distributions and Matrix Calculus  



### From Multivariate Gaussians to Matrix Normals

Let’s start with the familiar.

A multivariate normal distribution for a vector $$\mathbf{x} \in \mathbb{R}^d$$ looks like:

$$
\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \Sigma)
= \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
$$

Now suppose instead of a single $$\mathbf{x}$$, we observe an entire **matrix** $$X \in \mathbb{R}^{n \times p}$$ — say, $$n$$ samples and $$p$$ variables — and we want to model **correlation across both rows and columns**.

This leads us to the **matrix normal distribution**.

---

### The Matrix Normal Distribution

Let $$X \in \mathbb{R}^{n \times p}$$. We say:

$$
X \sim \mathcal{MN}(M, U, V)
$$

if:

$$
\operatorname{vec}(X) \sim \mathcal{N}\bigl(\operatorname{vec}(M),\, V \otimes U \bigr)
$$

That is:

- $$M \in \mathbb{R}^{n \times p}$$ is the **mean matrix**  
- $$U \in \mathbb{R}^{n \times n}$$ is the **row covariance**  
- $$V \in \mathbb{R}^{p \times p}$$ is the **column covariance**

The full density:

$$
p(X) =
\frac{1}{(2\pi)^{np/2} |U|^{p/2} |V|^{n/2}}
\exp\left( -\frac{1}{2} \operatorname{tr}\left( U^{-1} (X - M) V^{-1} (X - M)^\top \right) \right)
$$

---

### Why Matrix Calculus?

Let’s say we’re doing **maximum likelihood estimation**. We have a dataset $$\{X^{(i)}\}$$, and we want to learn the MLE of $$M, U, V$$. Or perhaps we’re doing **Bayesian inference**, and need the gradient of the log-likelihood for variational optimization.

In either case, we must compute derivatives of the form:

- $$\frac{\partial}{\partial M} \log p(X)$$  
- $$\frac{\partial}{\partial U} \log p(X)$$  
- $$\frac{\partial}{\partial V} \log p(X)$$

This is where **trace derivatives**, **log-determinant derivatives**, and **matrix inverse derivatives** come together — all the things we’ve built so far.

---

### Gradient w.r.t. the Mean Matrix

Let’s compute the gradient of the log-likelihood w.r.t. the mean matrix $$M$$.

From the density above:

$$
\log p(X) = \text{const} - \frac{1}{2} \operatorname{tr}\left( U^{-1} (X - M) V^{-1} (X - M)^\top \right)
$$

Use matrix calculus:

Let $$E = X - M$$. Then:

\[
\begin{aligned}
\frac{\partial}{\partial M} \log p(X)
&= \frac{1}{2} \frac{\partial}{\partial M} \operatorname{tr}\left( U^{-1} E V^{-1} E^\top \right) \\
&= \frac{1}{2} \cdot (-2 U^{-1} E V^{-1}) \\
&= -U^{-1} (X - M) V^{-1}
\end{aligned}
\]

So the gradient is:

$$
\boxed{
\frac{\partial \log p(X)}{\partial M} = - U^{-1} (X - M) V^{-1}
}
$$

---

### Gradient w.r.t. Covariance Matrices

Here’s where things get a bit trickier — but powerful.

Let’s differentiate:

$$
\log p(X) = \text{const} - \frac{p}{2} \log |U| - \frac{n}{2} \log |V| - \frac{1}{2} \operatorname{tr}\left( U^{-1} E V^{-1} E^\top \right)
$$

#### For the row covariance $$U$$:

Using:

- $$\frac{\partial \log |U|}{\partial U} = U^{-1}$$  
- $$\frac{\partial}{\partial U} \operatorname{tr}(U^{-1} A) = -U^{-1} A U^{-1}$$

We get:

$$
\frac{\partial \log p(X)}{\partial U} =
- \frac{p}{2} U^{-1}
+ \frac{1}{2} U^{-1} E V^{-1} E^\top U^{-1}
$$

Similarly for $$V$$:

$$
\frac{\partial \log p(X)}{\partial V} =
- \frac{n}{2} V^{-1}
+ \frac{1}{2} V^{-1} E^\top U^{-1} E V^{-1}
$$

These terms are structurally symmetric — you just flip $$E$$.

---

### Application: Probabilistic PCA

Probabilistic PCA (Tipping & Bishop, 1999) models data as:

$$
X = Z W^\top + \epsilon
$$

where $$Z$$ is a latent matrix and $$\epsilon \sim \mathcal{MN}(0, I, \sigma^2 I)$$.

Then:

$$
p(X \mid W, \sigma^2) = \mathcal{MN}(0, I, WW^\top + \sigma^2 I)
$$

Optimizing this log-likelihood (via EM or gradient methods) requires the matrix derivatives above — especially w.r.t. the covariance.

---

### The Matrix t-Distribution

A heavier-tailed cousin of the matrix normal, often used in **robust models**.

If:

$$
X \sim \mathcal{MT}(M, U, V, \nu)
$$

Then:

$$
p(X) \propto
|U|^{-p/2} |V|^{-n/2} \left[1 + \frac{1}{\nu} \operatorname{tr}(U^{-1}(X - M)V^{-1}(X - M)^\top)\right]^{-(\nu + np)/2}
$$

Similar derivatives can be computed using matrix calculus — just with heavier algebra.

---

### Summary Table

<div style="overflow-x: auto;">
  <table class="simple-table">
    <thead>
      <tr>
        <th>Quantity</th>
        <th>Derivative</th>
        <th>Interpretation</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Mean matrix \(M\)</td>
        <td>\(-U^{-1}(X - M)V^{-1}\)</td>
        <td>Gradient of log-likelihood</td>
      </tr>
      <tr>
        <td>Row covariance \(U\)</td>
        <td>\(-\frac{p}{2} U^{-1} + \frac{1}{2} U^{-1} E V^{-1} E^\top U^{-1}\)</td>
        <td>MLE or MAP gradient</td>
      </tr>
      <tr>
        <td>Column covariance \(V\)</td>
        <td>\(-\frac{n}{2} V^{-1} + \frac{1}{2} V^{-1} E^\top U^{-1} E V^{-1}\)</td>
        <td>Symmetric to \(U\)</td>
      </tr>
    </tbody>
  </table>
</div>

---

### Final Thoughts

The matrix normal distribution is more than a notational trick. It’s a **powerful modeling tool** that captures structure across **rows and columns** — essential in multivariate regression, matrix-variate factor models, and Gaussian processes.

And once you understand how to differentiate through it, you unlock a full spectrum of applications in **statistical learning**, **Bayesian inference**, and **deep probabilistic models**.

---



We’ve spent the past several sections laying down the mathematical bricks — gradients, Jacobians, Kronecker products, trace tricks, pseudoinverses. It’s a beautiful edifice of abstract computation. But machine learning isn’t built on equations alone.

Every time you train a neural network, minimize a loss, or regularize a model — you’re using these derivatives.

Now it’s time to flip the switch. Let’s move from *how* matrix calculus works to *where* it works.  

This section will walk through applications where matrix calculus powers core algorithms.


## Application 1: Backpropagation with Matrix Calculus  




Backpropagation is often introduced as a recursive algorithm involving layers, local gradients, and lots of chain rule applications. That’s true, but at its heart, backprop is really just **vectorized calculus** over a **composite matrix function**.

And once we understand the right rules — like how to differentiate a scalar loss with respect to a matrix — the entire algorithm becomes cleaner, easier to generalize, and surprisingly elegant.

Let’s break it down from scratch.

---

### The Setting: A 2-Layer Neural Network

Consider a simple neural network with:

- One hidden layer using ReLU activation  
- A final output layer with linear activation (for regression)

The forward pass looks like this:

$$
\begin{aligned}
Z^{(1)} &= X W^{(1)} + \mathbf{b}^{(1)} \\
A^{(1)} &= \phi(Z^{(1)}) \quad \text{(ReLU)} \\
Z^{(2)} &= A^{(1)} W^{(2)} + \mathbf{b}^{(2)} \\
\hat{Y} &= Z^{(2)} \\
L &= \frac{1}{2m} \| \hat{Y} - Y \|^2
\end{aligned}
$$

Where:

- $$X \in \mathbb{R}^{m \times d}$$: Input batch  
- $$W^{(1)} \in \mathbb{R}^{d \times h}$$: Weights from input to hidden layer  
- $$W^{(2)} \in \mathbb{R}^{h \times o}$$: Weights from hidden to output  
- $$\phi(\cdot)$$: ReLU activation  
- $$L$$: Scalar loss (MSE)

---

### Step-by-Step Derivatives Using Matrix Calculus

We now compute the gradients of the loss $$L$$ with respect to the weights and biases.

---

#### Step 1: Derivative w.r.t. $$Z^{(2)}$$

Since:

$$
L = \frac{1}{2m} \| Z^{(2)} - Y \|^2
$$

We get:

$$
\frac{\partial L}{\partial Z^{(2)}} = \frac{1}{m} (Z^{(2)} - Y) =: \delta^{(2)}
$$

---

#### Step 2: Gradient w.r.t. $$W^{(2)}$$

From:

$$
Z^{(2)} = A^{(1)} W^{(2)} + \mathbf{b}^{(2)}
$$

Using:

$$
\frac{\partial L}{\partial W^{(2)}} = (A^{(1)})^\top \delta^{(2)}
$$

Similarly, for the bias:

$$
\frac{\partial L}{\partial \mathbf{b}^{(2)}} = \text{row-wise sum of } \delta^{(2)}
$$

---

#### Step 3: Backpropagate to Hidden Layer

We compute the upstream gradient:

$$
\delta^{(1)} = \delta^{(2)} (W^{(2)})^\top \odot \phi'(Z^{(1)})
$$

Where:

- $$\odot$$: Hadamard (elementwise) product  
- $$\phi'(Z^{(1)})$$: Derivative of ReLU, i.e., indicator for positive entries in $$Z^{(1)}$$

---

#### Step 4: Gradient w.r.t. $$W^{(1)}$$

$$
\frac{\partial L}{\partial W^{(1)}} = X^\top \delta^{(1)}
$$

And for the bias:

$$
\frac{\partial L}{\partial \mathbf{b}^{(1)}} = \text{row-wise sum of } \delta^{(1)}
$$

---

### Recap: What Matrix Calculus Let Us Do

Without breaking into individual neuron derivatives, we used these matrix calculus identities:

- $$\frac{\partial L}{\partial W} = X^\top G$$, when $$Z = XW$$  
- Chain rule for $$\delta = \text{upstream} \times \text{local derivative}$$  
- Elementwise application of nonlinear derivatives (like ReLU)

This gave us fully vectorized gradient expressions — scalable, efficient, and readable.

---

### Summary Table

<div style="overflow-x: auto;">
  <table class="simple-table">
    <thead>
      <tr>
        <th>Quantity</th>
        <th>Matrix Derivative</th>
        <th>Dimensions</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><span style="white-space: nowrap;">$$ \frac{\partial L}{\partial W^{(2)}} $$</span></td>
        <td><span style="white-space: nowrap;">$$ (A^{(1)})^\top \delta^{(2)} $$</span></td>
        <td><span style="white-space: nowrap;">$$ h \times o $$</span></td>
      </tr>
      <tr>
        <td><span style="white-space: nowrap;">$$ \frac{\partial L}{\partial W^{(1)}} $$</span></td>
        <td><span style="white-space: nowrap;">$$ X^\top \delta^{(1)} $$</span></td>
        <td><span style="white-space: nowrap;">$$ d \times h $$</span></td>
      </tr>
      <tr>
        <td><span style="white-space: nowrap;">$$ \delta^{(1)} $$</span></td>
        <td><span style="white-space: nowrap;">$$ \delta^{(2)} (W^{(2)})^\top \odot \phi'(Z^{(1)}) $$</span></td>
        <td><span style="white-space: nowrap;">$$ m \times h $$</span></td>
      </tr>
    </tbody>
  </table>
</div>

---


## Application 2: Closed-form Solutions in Linear Models  



### Problem Setup: Linear Regression


Suppose we’re given a dataset of $$m$$ samples:

- **Inputs:** $$X \in \mathbb{R}^{m \times d}$$ — each row is a feature vector  
- **Targets:** $$y \in \mathbb{R}^{m \times 1}$$ — a column vector of real-valued labels  
- **Weights to learn:** $$\mathbf{w} \in \mathbb{R}^{d \times 1}$$

Our prediction is:

$$
\hat{y} = X \mathbf{w}
$$

And the **mean squared error loss** is:

$$
L(\mathbf{w}) = \frac{1}{2m} \left\| X \mathbf{w} - y \right\|^2
$$

This is a **scalar function** of a **vector variable** $$\mathbf{w}$$, which makes it ideal for matrix calculus.

---

### Goal

We want to minimize $$L(\mathbf{w})$$ by finding the best weight vector $$\mathbf{w}^\star$$ that solves:

$$
\min_{\mathbf{w}} \; L(\mathbf{w})
$$

And we want to do this **analytically**, using calculus.

---

### Step 1: Expand the Loss

Let’s first express the loss more compactly using inner products:

$$
L(\mathbf{w}) = \frac{1}{2m} (X \mathbf{w} - y)^\top (X \mathbf{w} - y)
$$

We’ll now differentiate this with respect to $$\mathbf{w}$$.

---

### Step 2: Differentiate Using Matrix Calculus

We use the standard identity:

$$
\frac{d}{d\mathbf{w}} \left( \mathbf{w}^\top A \mathbf{w} \right) = (A + A^\top) \mathbf{w}
$$

(If $$A$$ is symmetric, this reduces to $$2A \mathbf{w}$$)

We also use:

$$
\frac{d}{d\mathbf{w}} \left( \mathbf{a}^\top \mathbf{w} \right) = \mathbf{a}
$$

Let’s differentiate:

$$
\begin{aligned}
L(\mathbf{w}) &= \frac{1}{2m} \left[ \mathbf{w}^\top X^\top X \mathbf{w} - 2 y^\top X \mathbf{w} + y^\top y \right] \\
\frac{\partial L}{\partial \mathbf{w}} &= \frac{1}{2m} \left( 2 X^\top X \mathbf{w} - 2 X^\top y \right) \\
&= \frac{1}{m} \left( X^\top X \mathbf{w} - X^\top y \right)
\end{aligned}
$$

---

### Step 3: Set Gradient to Zero

To find the minimum, we set the gradient to zero:

$$
X^\top X \mathbf{w} = X^\top y
$$

Assuming $$X^\top X$$ is invertible, we get the **normal equation**:

$$
\boxed{
\mathbf{w}^\star = (X^\top X)^{-1} X^\top y
}
$$

That’s our closed-form solution.

---

### When Is This Useful?

- For small to medium datasets, this is extremely fast — no need for iterative solvers.
- It appears in $$\text{OLS regression}$$, $$\text{Ridge regression}$$ (with a slight tweak), $$\text{PCA}$$, and many classical models.

---

### Important Notes

1. **Computational cost**: Inverting $$X^\top X$$ has time complexity $$\mathcal{O}(d^3)$$. So this is rarely used in high-dimensional ML directly — but still foundational for theory and diagnostics.

2. **Numerical stability**: Instead of directly computing the inverse, use:

   $$
   \mathbf{w}^\star = \text{solve}(X^\top X, X^\top y)
   $$

   using something like QR or Cholesky decomposition.

3. **Connection to projections**: This solution is projecting $$y$$ onto the column space of $$X$$. That’s why the residual is orthogonal to the fitted predictions:

   $$
   X^\top (\hat{y} - y) = 0
   $$


---

### Summary Table

<div style="overflow-x: auto;">
  <table class="simple-table">
    <thead>
      <tr>
        <th>Expression</th>
        <th>Interpretation</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><span style="white-space: nowrap;">$$ L(\mathbf{w}) = \frac{1}{2m} \| X \mathbf{w} - y \|^2 $$</span></td>
        <td>Quadratic loss function</td>
      </tr>
      <tr>
        <td><span style="white-space: nowrap;">$$ \nabla_{\mathbf{w}} L = \frac{1}{m} (X^\top X \mathbf{w} - X^\top y) $$</span></td>
        <td>Gradient via matrix calculus</td>
      </tr>
      <tr>
        <td><span style="white-space: nowrap;">$$ \mathbf{w}^\star = (X^\top X)^{-1} X^\top y $$</span></td>
        <td>Closed-form solution (normal equation)</td>
      </tr>
    </tbody>
  </table>
</div>

---

## Application 3: Convolutional Neural Networks (CNNs)  
*Gradient computation through convolutional layers using matrix calculus*

---

Convolutional Neural Networks revolutionized computer vision by mimicking how we process spatial patterns — edges, corners, textures — through local filters. But for these filters to *learn*, they must be differentiable. And not just scalar-by-scalar differentiable, but **differentiable as matrix functions** that act over batched inputs, multiple channels, and shared parameters.

This is where matrix calculus flexes its power. Rather than deriving gradients through messy loops or hand-coding derivatives for every layer, we treat the entire convolution as a composed matrix transformation — and derive gradients cleanly, efficiently, and fully vectorized.

---

### 1D Convolution

Let’s begin with a minimalist setup to build intuition: a 1D convolution with one input row and one filter.

---

#### Step 1: The Forward Pass in Matrix Clothing

Suppose you have an input vector:

$$
X = \begin{bmatrix} x_1 & x_2 & x_3 & x_4 & x_5 \end{bmatrix} \in \mathbb{R}^{1 \times 5}
$$

and a filter:

$$
w = \begin{bmatrix} w_1 & w_2 & w_3 \end{bmatrix} \in \mathbb{R}^{1 \times 3}
$$

With stride 1 and no padding, the filter moves across 3 positions. At each step, it "sees" a window of 3 input elements — called a **receptive field**.

We extract these into overlapping **patches**:

$$
X_{\text{col}} =
\begin{bmatrix}
x_1 & x_2 & x_3 \\
x_2 & x_3 & x_4 \\
x_3 & x_4 & x_5
\end{bmatrix}
\in \mathbb{R}^{3 \times 3}
$$

This reshaping — often implemented as `im2col` — converts the convolution into a standard matrix product:

$$
Z = X_{\text{col}} \cdot w^\top \in \mathbb{R}^{3 \times 1}
$$

Each row of $$Z$$ is the dot product between a filter and a sliding window from the input — i.e., a classic 1D convolution output.

---

#### Step 2: The Backward Pass — Let the Gradients Flow

Now suppose we compute some scalar loss $$L = \mathcal{L}(Z)$$ (like MSE or cross-entropy).

Let $$G = \frac{\partial L}{\partial Z}$$ be the gradient flowing back from the loss to the convolution output.

Matrix calculus gives us:

- **Gradient w.r.t. the filter**:

  $$
  \frac{\partial L}{\partial w} = G^\top X_{\text{col}} \in \mathbb{R}^{1 \times 3}
  $$

- **Gradient w.r.t. the input patches**:

  $$
  \frac{\partial L}{\partial X_{\text{col}}} = G \cdot w \in \mathbb{R}^{3 \times 3}
  $$

This gradient can be mapped back to the original 1D input space using a `col2im` operation.

> **Sidenote:** In PyTorch or TensorFlow, this step is automatically handled by the backward pass of the convolution op, but under the hood, it’s doing exactly this — summing overlapping contributions from each patch.

---

### 2D Convolution

Now let’s scale up to the 2D case: where both inputs and filters are matrices.

---

#### Step 1: Matrix View of 2D Convolution

Imagine a $3 \times 3$ image being filtered by a $2 \times 2$ kernel:

$$
X = 
\begin{bmatrix}
x_{11} & x_{12} & x_{13} \\
x_{21} & x_{22} & x_{23} \\
x_{31} & x_{32} & x_{33}
\end{bmatrix}
\in \mathbb{R}^{3 \times 3}
\quad
W =
\begin{bmatrix}
w_{11} & w_{12} \\
w_{21} & w_{22}
\end{bmatrix}
\in \mathbb{R}^{2 \times 2}
$$

Stride 1, no padding.

This gives 4 positions for the filter to slide:

1. $$x_{11}, x_{12}, x_{21}, x_{22}$$  
2. $$x_{12}, x_{13}, x_{22}, x_{23}$$  
3. $$x_{21}, x_{22}, x_{31}, x_{32}$$  
4. $$x_{22}, x_{23}, x_{32}, x_{33}$$

We collect these into:

$$
X_{\text{col}} =
\begin{bmatrix}
x_{11} & x_{12} & x_{21} & x_{22} \\
x_{12} & x_{13} & x_{22} & x_{23} \\
x_{21} & x_{22} & x_{31} & x_{32} \\
x_{22} & x_{23} & x_{32} & x_{33}
\end{bmatrix}
\in \mathbb{R}^{4 \times 4}
$$

Flatten the filter:

$$
w = \begin{bmatrix} w_{11} & w_{12} & w_{21} & w_{22} \end{bmatrix} \in \mathbb{R}^{1 \times 4}
$$

Now compute:

$$
Z = X_{\text{col}} \cdot w^\top \in \mathbb{R}^{4 \times 1}
\Rightarrow \text{reshape to } 2 \times 2
$$

---

#### Step 2: Gradients via Matrix Calculus

Assume loss:

$$
L = \frac{1}{2} \| Z - T \|^2
$$

Then:

- **Output gradient**:

  $$
  G = \frac{\partial L}{\partial Z} = Z - T \in \mathbb{R}^{4 \times 1}
  $$

- **Filter gradient**:

  $$
  \frac{\partial L}{\partial w} = G^\top X_{\text{col}} \in \mathbb{R}^{1 \times 4}
  $$

- **Input patch gradient**:

  $$
  \frac{\partial L}{\partial X_{\text{col}}} = G \cdot w \in \mathbb{R}^{4 \times 4}
  $$

To reconstruct $$\frac{\partial L}{\partial X}$$ (the gradient in image form), we undo the patchification using `col2im`.

---

### Stride and Padding: What Changes?

Let:

- $$n$$: Input size  
- $$k$$: Kernel size  
- $$s$$: Stride  
- $$p$$: Padding

Then:

$$
\text{output size} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1
$$

**Implications:**

- **Larger strides** reduce overlap → fewer gradients per input location  
- **More padding** preserves input size → more overlap → more spatial coverage  
- **Gradient shapes** depend entirely on this arithmetic and are accounted for automatically during `im2col` / `col2im`

---

### Weight Sharing and Summation of Gradients

Since the same kernel applies across all spatial locations, the total gradient with respect to the filter is the sum of contributions from each patch:

$$
\frac{\partial L}{\partial w} = \sum_{i=1}^P g_i \cdot x_i
$$

Or vectorized:

$$
\frac{\partial L}{\partial w} = G^\top X_{\text{col}}
$$

This summation is what gives convolutional filters their spatial generalization: a filter that sharpens an edge in one corner can also learn to do so in another.

---

## Wrapping Up

By now, you’ve seen how matrix calculus fits into machine learning — from basic derivatives to the way gradients move through layers during backpropagation. We’ve looked at Jacobians, vectorization tricks, and how concepts like the trace and determinant show up in real models.

The goal wasn’t to cover every identity — it was to build a way of thinking. One where shapes, dimensions, and transformations become familiar, and where equations start to make sense beyond just symbol pushing.

You’re now in a position to read through a model description and understand where the gradients come from. You can trace the flow of derivatives through a network and spot the logic behind the updates.

And from here, it only gets more interesting. Try applying these ideas to your own models. Look at how your framework handles gradient flow. Work through a few derivations by hand.

This foundation will serve you well — in deep learning, in optimization, and anywhere gradients drive learning.