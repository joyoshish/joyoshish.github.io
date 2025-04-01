---
layout: post
title: Linear Algebra Basics for ML - Vector Operations, Norms, and Projections
date: 2024-05-13 11:20:00
description: Linear Algebra 1 - Mathematics for Machine Learning
tags: ml ai linear-algebra math
math: true
categories: machine-learning math
# thumbnail: assets/img/linalg_banner.png
giscus_comments: true
related_posts: true
chart:
  plotly: true
toc:
  beginning: true
---


In machine learning, every problem—whether it’s image recognition, natural language processing, or anomaly detection—begins with how we represent data. In this post, we’ll take a deep dive into vectors and vector spaces, exploring the underlying mathematics that enables ML algorithms to learn from data. We’ll follow a problem-driven approach: each section starts with a real-world ML challenge, introduces the mathematical tool needed, explains the theory from the ground up, and concludes with a detailed solution along with Python coding examples and real-world applications in NLP and Computer Vision.

---

## Vector Addition and Scalar Multiplication

Imagine you're training a neural network. At each step, your model needs to update its parameters—those weights that define how the network behaves. But how exactly are these updates performed? Behind the scenes, you’re combining current weights with gradient information and scaling them based on how much you want to change. This simple operation, which powers the heart of deep learning, relies entirely on vector addition and scalar multiplication.

Let’s break it down.

A vector $$\mathbf{v}$$ in $$\mathbb{R}^n$$ is an ordered list of $$n$$ real numbers:

$$
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
$$

Vectors represent points, directions, or quantities in space—and in machine learning, they can represent feature values, model parameters, or gradients.

Adding two vectors $$\mathbf{u}$$ and $$\mathbf{v}$$ looks like this:

$$
\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}
$$


This operation is performed element-wise. You can think of it as combining two data points or updating a parameter by adding the change suggested by a gradient.

Now, multiplying a vector by a scalar $$\alpha$$ stretches or shrinks it:

$$
\alpha \mathbf{u} = \begin{bmatrix} \alpha u_1 \\ \alpha u_2 \\ \vdots \\ \alpha u_n \end{bmatrix}
$$

This changes the length of the vector (its magnitude), but not its direction—unless $$\alpha$$ is negative, in which case the vector flips.

This brings us to the classic weight update rule in gradient descent:

$$
\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} - \alpha \nabla \mathbf{w}
$$

Here, $$\nabla \mathbf{w}$$ is the gradient vector, and $$\alpha$$ is the learning rate. You're subtracting a scaled version of the gradient from the current weights—a simple but powerful operation that helps your model learn.

Let’s see this in action:

```python
import numpy as np

# Define weight vector and gradient vector
weights = np.array([0.5, -0.3, 0.8])
gradients = np.array([0.1, -0.05, 0.2])

# Update rule using gradient descent (learning rate = 0.1)
learning_rate = 0.1
new_weights = weights - learning_rate * gradients
print("Updated Weights:", new_weights)
```

Running this snippet simulates one step of gradient descent. Each component of the weight vector is nudged slightly in the direction opposite to the gradient, scaled by how aggressively you want to learn (i.e., the learning rate).

These operations form the computational backbone of most optimization routines in machine learning. For example, gradient descent, stochastic gradient descent (SGD), and their variants (like Adam or RMSProp) all rely on vector addition and scalar multiplication to iteratively update model parameters.

In reinforcement learning, policy gradients are updated via vector-based gradient steps to maximize expected returns. In graph neural networks (GNNs), feature propagation across nodes often involves combining node and neighbor vectors—again using addition and scaling operations.

Even in time-series forecasting, models like LSTMs and GRUs perform cell updates using vector operations that combine new and past information. These operations are not only simple but also form the atomic operations used to build and train large-scale models across supervised, unsupervised, and self-supervised learning paradigms.

---

## Linear Combinations, Span, Basis, and Dimensionality

In many machine learning applications, especially in domains like text and image processing, the data we work with lives in very high-dimensional spaces. Word embeddings might have 300 dimensions, images can have thousands of pixel values—and all of this contributes to increased computation, memory usage, and sometimes even noise. But do we really need all those dimensions?

Often, we don’t. The trick lies in representing high-dimensional data more compactly—without losing the essence of what makes that data useful. To do that, we need to understand the concepts of linear combinations, span, basis, and dimensionality.

Let’s start with the basics.

A **linear combination** allows us to build a new vector using a set of existing vectors. Suppose we have vectors $$\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k$$ in $$\mathbb{R}^n$$, then a linear combination is:

$$
\mathbf{v} = \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \dots + \alpha_k \mathbf{v}_k
$$

In other words, we scale each vector by a coefficient and add them together.

The **span** of a set of vectors is the collection of all possible vectors you can form using linear combinations of those vectors. If your set spans $$\mathbb{R}^n$$, then it’s powerful enough to represent any point in that space.

Now, when you want the most compact and efficient representation, you need a **basis**: a set of linearly independent vectors that spans the entire space. With a basis, every vector in the space can be uniquely expressed as a linear combination of these basis vectors.

The number of vectors in the basis gives us the **dimension** of the space. And in machine learning, reducing this dimensionality—while preserving the most important structure in the data—is exactly what techniques like PCA (Principal Component Analysis) aim to do.

PCA finds a new basis where each new vector (called a principal component) captures as much variance in the data as possible. These new basis vectors are orthogonal, and often, you only need the first few to explain most of your data's structure. This simplifies your dataset, making models faster and potentially more robust.

Here’s a quick example of how we can represent a vector using a standard basis:

```python
import numpy as np

# Representing a point in 2D space using the standard basis
basis_vectors = np.array([[1, 0], [0, 1]])
coefficients = np.array([3, 4])
new_vector = np.dot(coefficients, basis_vectors)
print("New Vector from Linear Combination:", new_vector)
```

This gives us the vector $$[3, 4]$$ as a combination of the basis vectors $$[1, 0]$$ and $$[0, 1]$$.

The idea of expressing data using a minimal set of representative directions is ubiquitous in ML. Autoencoders, for instance, learn a compressed latent space—a learned basis—into which input data is encoded and later decoded. This compressed representation reduces dimensionality and noise while preserving the key structure of the input.

In signal processing, sparse coding and dictionary learning aim to represent signals as linear combinations of a few basis elements. In finance, factor models such as PCA or ICA are used to explain asset returns through a few economic factors.

Dimensionality reduction is also crucial in medical imaging, where thousands of features (pixels or voxels) are compressed into fewer latent variables for classification tasks like tumor detection. In genomics, where gene expression data is high-dimensional, basis discovery helps in clustering, feature selection, and disease classification.

---

## Orthogonality and Projections

Imagine you're working on a computer vision task and want to compress image data without losing important information. You could reduce the number of features, but how do you ensure you're keeping the parts that matter?

The answer lies in projections. More specifically, *orthogonal projections* onto lower-dimensional subspaces help us reduce dimensionality while preserving the most important aspects of the data.

In math terms, two vectors $$\mathbf{u}$$ and $$\mathbf{v}$$ are said to be orthogonal if their dot product is zero:

$$
\mathbf{u} \cdot \mathbf{v} = 0
$$

This means the vectors point in completely independent directions—think of axes in 3D space.

When you project one vector onto another, you’re essentially extracting its component in that direction. The projection of $$\mathbf{u}$$ onto $$\mathbf{v}$$ is calculated as:

$$
\text{proj}_{\mathbf{v}}(\mathbf{u}) = \left( \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \right) \mathbf{v}
$$

This formula plays a major role in Principal Component Analysis (PCA), where data is projected onto orthogonal axes—called principal components—that maximize variance. In simpler terms, PCA finds the directions in which your data varies the most and projects it there, compressing the information without much loss.

Here’s a simple example to visualize a projection:

```python
import numpy as np

# Define a data point and a principal component direction
data_point = np.array([3, 4])
principal_component = np.array([1, 0])

# Project the data point onto the principal component
proj_scalar = np.dot(data_point, principal_component) / np.dot(principal_component, principal_component)
projection = proj_scalar * principal_component
print("Projection of Data Point onto Principal Component:", projection)
```

<div style="display: flex; justify-content: center;">
  <div id="projectionEnhanced"></div>
</div>
<div id="projectionEnhanced"></div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
  // Original point and principal component
  const point = [3, 4];
  const pc = [1, 0]; // principal component vector

  // Projection calculation
  const dot = point[0]*pc[0] + point[1]*pc[1];
  const pcNorm2 = pc[0]*pc[0] + pc[1]*pc[1];
  const scale = dot / pcNorm2;
  const proj = [scale * pc[0], scale * pc[1]]; // [3, 0]

  // Helper function to make an arrow shape
  function arrow(x0, y0, x1, y1, color, width = 3, dash = "solid") {
    return {
      type: "line",
      x0: x0, y0: y0,
      x1: x1, y1: y1,
      line: {
        color: color,
        width: width,
        dash: dash
      },
      xref: "x",
      yref: "y"
    };
  }

  const layout = {
    title: {
      text: "",
      font: { size: 20 }
    },
    width: 650,
    height: 600,
    showlegend: false,
    plot_bgcolor: "#fcfcfc",
    paper_bgcolor: "#ffffff",
    xaxis: {
      title: "X-axis",
      range: [-1, 5],
      showgrid: true,
      gridcolor: "#eaeaea",
      zeroline: true,
    },
    yaxis: {
      title: "Y-axis",
      range: [-1, 5],
      showgrid: true,
      gridcolor: "#eaeaea",
      zeroline: true,
    },
    shapes: [
      // Principal component direction (dashed)
      arrow(0, 0, 4.5, 0, "orange", 2, "dot"),

      // Arrow from origin to original point
      arrow(0, 0, point[0], point[1], "blue", 4),

      // Arrow from origin to projection
      arrow(0, 0, proj[0], proj[1], "green", 4),

      // Dashed perpendicular connector
      arrow(point[0], point[1], proj[0], proj[1], "gray", 2, "dot")
    ],
    annotations: [
      {
        x: point[0], y: point[1],
        text: "Data Point (3, 4)",
        showarrow: true,
        arrowhead: 2,
        ax: -40, ay: -40,
        font: { color: "blue", size: 14 }
      },
      {
        x: proj[0], y: proj[1],
        text: "Projection (3, 0)",
        showarrow: true,
        arrowhead: 2,
        ax: -20, ay: 40,
        font: { color: "green", size: 14 }
      },
      {
        x: 4.5, y: -0.2,
        text: "Principal Component (PC₁)",
        showarrow: false,
        font: { color: "orange", size: 13 }
      }
    ]
  };

  const data = [
    {
      type: "scatter",
      mode: "markers",
      x: [point[0], proj[0]],
      y: [point[1], proj[1]],
      marker: {
        size: 10,
        color: ["blue", "green"]
      }
    }
  ];

  Plotly.newPlot("projectionEnhanced", data, layout);
</script>


Orthogonality is a core concept in many areas of ML that deal with feature decorrelation. For instance, independent component analysis (ICA) extends PCA by aiming for statistically independent (not just uncorrelated) components, which is useful in blind source separation tasks like speech signal decomposition.

In computer vision, projections are used in dimensionality reduction pipelines (e.g., PCA, t-SNE, UMAP) to extract visually salient features. Orthogonality also plays a role in orthogonal initialization of deep neural networks, which helps preserve variance and avoid vanishing gradients in very deep architectures.

In physics-informed ML and scientific computing, projections are used to map complex nonlinear states into simpler bases for simulation or differential equation modeling. This helps compress high-dimensional simulations like fluid dynamics into learnable latent dynamics.

---

## Vector Norms and Model Complexity

Now suppose you’re training a regression model, and it starts overfitting—performing great on training data but terribly on unseen examples. One common trick to fix this is regularization, which involves penalizing large weights. But that raises a question: how do you actually measure the "size" of a vector?

That’s where vector norms come in.

The **L1 norm** (also known as the Manhattan norm) sums up the absolute values of all the vector components:

$$
\|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i|
$$

The **L2 norm** (Euclidean norm) is the straight-line distance from the origin:

$$
\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2}
$$

And the **L∞ norm** captures the largest absolute value in the vector:

$$
\|\mathbf{v}\|_\infty = \max_i |v_i|
$$

Each of these norms gives a different perspective on vector size, and each is used in different types of regularization. Lasso regression uses the L1 norm to encourage sparsity (pushing some weights to zero), while Ridge regression uses the L2 norm to shrink all weights evenly. L∞ isn’t as common, but it can be useful when you care about controlling the biggest contributor.

Here’s a quick example showing how to calculate all three:

```python
import numpy as np

v = np.array([1, -2, 3])

l1_norm = np.sum(np.abs(v))
l2_norm = np.sqrt(np.sum(v ** 2))
linf_norm = np.max(np.abs(v))

print("L1 Norm:", l1_norm)
print("L2 Norm:", l2_norm)
print("L∞ Norm:", linf_norm)
```



Regularization using norms is a widely adopted strategy to prevent overfitting and improve generalization. In logistic regression and linear classifiers, L1 and L2 regularization help constrain the coefficient space, thereby simplifying the decision boundary and increasing robustness.

In deep learning, the weight decay trick (essentially L2 norm penalty) is used alongside batch normalization and dropout to stabilize training. In federated learning, client updates are sometimes clipped using norm thresholds to avoid noisy or adversarial contributions.

Beyond regularization, norms are also used in anomaly detection (e.g., distance from cluster centroids), metric learning (contrastive and triplet losses depend on Euclidean or cosine distances), and adversarial robustness (where L∞, L2, and L1 norms define allowed perturbation bounds). Whether you're compressing models for edge deployment or defending them against attacks, norms guide and constrain model behavior effectively.

---

## Inner and Outer Products

Let’s say you’re building a recommendation engine or clustering users based on their behavior. One of the first things you’ll need to do is measure how similar two data points are. But how do you quantify "similarity" in a vector space?

That’s where the **inner product** comes in.

Given two vectors $$\mathbf{u}$$ and $$\mathbf{v}$$, the inner product (or dot product) is:

$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i
$$

If the result is large, it means the vectors are pointing in similar directions—i.e., they are similar. In fact, this is the basis for cosine similarity, a widely used metric in NLP for comparing word vectors.

On the other hand, the **outer product** builds a full matrix of interactions between two vectors. For $$\mathbf{u} \in \mathbb{R}^m$$ and $$\mathbf{v} \in \mathbb{R}^n$$, the outer product looks like this:

$$
\mathbf{u} \otimes \mathbf{v} =
\begin{bmatrix}
u_1v_1 & u_1v_2 & \cdots & u_1v_n \\
u_2v_1 & u_2v_2 & \cdots & u_2v_n \\
\vdots & \vdots & \ddots & \vdots \\
u_mv_1 & u_mv_2 & \cdots & u_mv_n \\
\end{bmatrix}
$$

While the inner product gives us a single similarity score, the outer product creates a full interaction map—useful when we want to understand how features influence each other.

Here’s how to compute both in Python:

```python
import numpy as np

u = np.array([1, 2])
v = np.array([3, 4])

# Inner product: similarity
inner_product = np.dot(u, v)
print("Inner Product:", inner_product)

# Outer product: interaction matrix
outer_product = np.outer(u, v)
print("Outer Product:\n", outer_product)
```

The inner product is at the heart of similarity calculations across many algorithms—whether it's comparing embeddings in a semantic space or computing attention weights in transformers. Cosine similarity, a normalized inner product, is frequently used in clustering, information retrieval, and question-answering systems.

In kernel methods, such as support vector machines (SVMs), the inner product is generalized into kernel functions to measure similarity in high-dimensional (sometimes infinite-dimensional) spaces. The outer product, on the other hand, forms the basis of covariance matrices, used in PCA, Gaussian processes, and multivariate statistics.

In deep learning, attention mechanisms use scaled dot-product attention, which essentially involves outer products to compute weighted combinations across key-query-value triplets. Outer products are also fundamental to tensor factorization and bilinear pooling methods used in multi-modal learning, such as combining image and text inputs in visual question answering (VQA).

---


From updating model parameters with gradient descent to compressing high-dimensional data, measuring vector magnitudes, and analyzing feature interactions—vectors and vector spaces are the hidden framework behind much of machine learning.

These mathematical ideas may seem abstract at first, but they solve incredibly concrete problems. They power dimensionality reduction in PCA, make regularization work in regression models, and drive similarity searches in recommendation engines and NLP systems.

If you’ve followed along this far, you've just walked through the foundational math that makes many machine learning techniques possible. And we're just getting started—these concepts will show up again and again as we explore more advanced topics in the Math for ML series.