---
layout: post
title: Linear Algebra Basics for ML - Vector Operations, Norms, and Projections
date: 2024-05-13 11:20:00
description: Linear Algebra 1 - Mathematics for Machine Learning
tags: ml ai linear_algebra math
math: true
categories: machine-learning math
# thumbnail: assets/img/linalg_banner.png
giscus_comments: true
related_posts: true
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

These same operations show up everywhere. In NLP, we update word embeddings using gradients—every word vector in a model like Word2Vec gets refined through such updates. In Computer Vision, convolutional filters (which are essentially matrices or higher-dimensional tensors made up of vector-like slices) are tuned via backpropagation, relying on vector addition and scaling.

Whether you’re fine-tuning the weights of a deep network or learning dense word representations, these basic operations—vector addition and scalar multiplication—are always in play. They may seem simple, but they’re fundamental to everything that follows in the world of machine learning.


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

In practice, especially in NLP, we apply dimensionality reduction techniques like PCA to compress high-dimensional word vectors into more manageable sizes. This helps in both visualization and reducing complexity for downstream models.

In computer vision, PCA is used for image compression—transforming large image data into a smaller set of values that still retain the key visual features. This can drastically reduce storage and computation without sacrificing much accuracy in tasks like recognition or classification.

So whether you’re dealing with text or pixels, these foundational ideas—linear combinations, span, basis, and dimension—are what let us tame the curse of dimensionality and make sense of our data in smarter ways.

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

In real-world NLP tasks, we use projections to reduce the dimensionality of word vectors, helping to clean up noisy embeddings and improve interpretability. In computer vision, orthogonal projections help extract relevant image features, enabling better classification, detection, or compression.

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

In NLP, we use norm-based regularization to keep our word embeddings generalizable. In computer vision, we regularize convolutional filters to avoid overfitting to noise in the training images. Measuring and controlling vector magnitude is key to making our models not just accurate, but robust.

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

In NLP, we use inner products to compare embeddings and cluster similar words or documents. Outer products come in handy when building attention mechanisms or covariance matrices that model relationships between features. In computer vision, outer products are used to analyze spatial patterns in images, enabling algorithms to detect textures, faces, and more.

---


From updating model parameters with gradient descent to compressing high-dimensional data, measuring vector magnitudes, and analyzing feature interactions—vectors and vector spaces are the hidden framework behind much of machine learning.

These mathematical ideas may seem abstract at first, but they solve incredibly concrete problems. They power dimensionality reduction in PCA, make regularization work in regression models, and drive similarity searches in recommendation engines and NLP systems.

If you’ve followed along this far, you've just walked through the foundational math that makes many machine learning techniques possible. And we're just getting started—these concepts will show up again and again as we explore more advanced topics in the Math for ML series.