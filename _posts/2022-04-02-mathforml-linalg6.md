---
layout: post
title: Linear Algebra Basics for ML - Advanced Topics
date: 2022-02-07
description: Linear Algebra 6 - Mathematics for Machine Learning
tags: ml ai linear-algebra math
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

You're absolutely right—and thank you for your patience. Below is the **formally written**, **fully Markdown-compatible** version with **all inline math inside `$$...$$`** (even single-variable expressions), so it renders correctly in Jekyll using MathJax or KaTeX.

---

# Advanced Matrix Factorizations in Machine Learning

Matrix factorizations are a foundational tool in linear algebra and play a critical role in modern machine learning. They simplify matrix computations, enable numerical stability, and reveal latent structures in data. This post explores several advanced matrix factorizations used in machine learning and deep learning, with complete mathematical derivations and practical insights.

This section covers:

- LU Decomposition  
- Cholesky Decomposition  
- QR Decomposition  
- Non-negative Matrix Factorization (NMF)

All mathematical expressions are enclosed in `$$...$$` for compatibility with Jekyll and MathJax.

---

## 1. LU Decomposition

### Definition

LU decomposition factors a square matrix $$A \in \mathbb{R}^{n \times n}$$ into the product of two matrices:

$$
A = LU
$$

Where:
- $$L$$ is a lower triangular matrix with ones on its diagonal ($$L_{ii} = 1$$),
- $$U$$ is an upper triangular matrix.

If row pivoting is required, the decomposition is written as:

$$
PA = LU
$$

Where $$P$$ is a permutation matrix.

### Algorithm

Given:

$$
A = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nn}
\end{bmatrix},
$$

we use Gaussian elimination to eliminate entries below the diagonal. The multipliers used are stored in $$L$$, and the remaining upper triangular form becomes $$U$$.

For each row $$j > i$$, compute the multiplier:

$$
l_{ji} = \frac{a_{ji}}{a_{ii}}
$$

Then update row $$j$$:

$$
a_{j\cdot} = a_{j\cdot} - l_{ji} \cdot a_{i\cdot}
$$

After completing all iterations, the matrix $$A$$ is factorized into $$L$$ and $$U$$.

### Applications in Machine Learning

- Efficiently solving systems of equations $$Ax = b$$ by solving $$Ly = b$$ then $$Ux = y$$,
- Matrix inversion,
- Numerical optimization routines in regression and convex problems.

---

## 2. Cholesky Decomposition

### Definition

Cholesky decomposition applies to **symmetric, positive-definite** matrices. For a matrix $$A \in \mathbb{R}^{n \times n}$$ satisfying:

- $$A = A^T$$ (symmetry),
- $$x^T A x > 0$$ for all non-zero $$x \in \mathbb{R}^n$$ (positive-definiteness),

there exists a unique lower triangular matrix $$L$$ such that:

$$
A = LL^T
$$

### Construction

We compute each entry of $$L$$ as follows:

For diagonal entries:

$$
L_{ii} = \sqrt{A_{ii} - \sum_{k=1}^{i-1} L_{ik}^2}
$$

For off-diagonal entries where $$i > j$$:

$$
L_{ij} = \frac{1}{L_{jj}} \left( A_{ij} - \sum_{k=1}^{j-1} L_{ik} L_{jk} \right)
$$

All entries above the diagonal are zero.

### Applications in Machine Learning

- Efficient sampling from multivariate Gaussian distributions,
- Gaussian Processes (for inverting the kernel matrix),
- Kalman filters and Bayesian updates.

---

## 3. QR Decomposition

### Definition

For any matrix $$A \in \mathbb{R}^{m \times n}$$ where $$m \geq n$$, the QR decomposition expresses $$A$$ as:

$$
A = QR
$$

Where:
- $$Q \in \mathbb{R}^{m \times m}$$ is orthogonal ($$Q^T Q = I$$),
- $$R \in \mathbb{R}^{m \times n}$$ is upper triangular.

For economy-size decomposition (when only $$n$$ orthogonal vectors are needed), we use:

$$
A = Q_{\text{red}} R, \quad Q_{\text{red}} \in \mathbb{R}^{m \times n}
$$

### Gram-Schmidt Orthogonalization

Let $$a_1, a_2, \ldots, a_n$$ be the columns of $$A$$. We generate orthonormal vectors $$q_1, q_2, \ldots, q_n$$ using:

1. Initialization:

$$
u_1 = a_1, \quad q_1 = \frac{u_1}{\|u_1\|}
$$

2. For $$k = 2, \ldots, n$$:

$$
u_k = a_k - \sum_{j=1}^{k-1} \langle a_k, q_j \rangle q_j
$$

$$
q_k = \frac{u_k}{\|u_k\|}
$$

Then define:

$$
Q = [q_1, q_2, \ldots, q_n], \quad R = Q^T A
$$

### Least Squares via QR

Given an overdetermined system $$Ax = b$$, we solve:

$$
A = QR \Rightarrow QRx = b
$$

Then:

$$
Rx = Q^T b
$$

Since $$R$$ is upper triangular, this is solved efficiently via back substitution.

### Applications in Machine Learning

- Numerically stable solution for linear regression,
- Eigenvalue computations using the QR algorithm,
- Orthonormal basis construction.

---

## 4. Non-negative Matrix Factorization (NMF)

### Definition

Given a non-negative matrix $$A \in \mathbb{R}^{m \times n}$$ (i.e., $$A_{ij} \geq 0$$), NMF seeks matrices $$W \in \mathbb{R}^{m \times k}$$ and $$H \in \mathbb{R}^{k \times n}$$ such that:

$$
A \approx WH
$$

subject to:

$$
W \geq 0, \quad H \geq 0
$$

### Optimization Problem

The factorization is found by solving:

$$
\min_{W, H \geq 0} \|A - WH\|_F^2
$$

where $$\|\cdot\|_F$$ denotes the Frobenius norm.

### Multiplicative Update Rules

A common approach (Lee & Seung, 2001) involves the following update rules:

1. Update $$H$$:

$$
H_{ij} \leftarrow H_{ij} \cdot \frac{(W^T A)_{ij}}{(W^T W H)_{ij}}
$$

2. Update $$W$$:

$$
W_{ij} \leftarrow W_{ij} \cdot \frac{(A H^T)_{ij}}{(W H H^T)_{ij}}
$$

These updates are applied iteratively until convergence.

### Applications in Machine Learning

- **Topic modeling** from document-term matrices,
- **Collaborative filtering** in recommendation systems,
- **Image compression and decomposition**,
- **Clustering** with parts-based representation.

---



# Orthogonalization Techniques in Machine Learning and Deep Learning

Orthogonalization plays a central role in linear algebra and is extensively used in various machine learning and deep learning tasks. Whether it’s **constructing orthonormal bases**, **decorrelating features**, or **stabilizing neural network training**, orthogonal structures are powerful due to their numerical and geometric properties.

Now we shall cover:

- The **Gram-Schmidt orthogonalization process** for constructing orthonormal bases,
- **Orthogonal initialization** in neural networks for improved stability and convergence.

---

## 1. Orthogonalization and Orthonormal Bases

### Motivation

Given a set of linearly independent vectors $$\{v_1, v_2, \ldots, v_n\}$$ in $$\mathbb{R}^n$$, it is often desirable to construct an orthonormal basis $$\{q_1, q_2, \ldots, q_n\}$$ that spans the same subspace, where:

- Vectors are **orthogonal**: $$\langle q_i, q_j \rangle = 0$$ for $$i \ne j$$,
- Vectors are **normalized**: $$\|q_i\| = 1$$.

Orthonormal bases are easier to work with:
- Projections are straightforward,
- Matrix representations (e.g., $$Q^T Q = I$$) are numerically stable,
- Useful for dimensionality reduction and decorrelation (e.g., PCA).

---

## 2. The Gram-Schmidt Process

### Definition

The **Gram-Schmidt process** transforms a set of linearly independent vectors $$\{v_1, v_2, \ldots, v_n\}$$ into an orthonormal set $$\{q_1, q_2, \ldots, q_n\}$$ spanning the same subspace.

This is achieved by iteratively subtracting projections onto previously computed vectors and normalizing.

---

### Step-by-step Algorithm

Let the input be vectors $$v_1, v_2, \ldots, v_n \in \mathbb{R}^d$$.

1. **Initialize** the first orthonormal vector:

$$
q_1 = \frac{v_1}{\|v_1\|}
$$

2. **Iterate** for $$k = 2, \ldots, n$$:

   - Project $$v_k$$ onto each previous $$q_j$$:

   $$
   \text{proj}_{q_j}(v_k) = \langle v_k, q_j \rangle q_j
   $$

   - Subtract the projections:

   $$
   u_k = v_k - \sum_{j=1}^{k-1} \langle v_k, q_j \rangle q_j
   $$

   - Normalize to get the next orthonormal vector:

   $$
   q_k = \frac{u_k}{\|u_k\|}
   $$

After the process, the vectors $$\{q_1, \ldots, q_n\}$$ form an **orthonormal basis** of the span of $$\{v_1, \ldots, v_n\}$$.

---

### Example

Let:

$$
v_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$

Compute:

$$
q_1 = \frac{v_1}{\|v_1\|} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

Project $$v_2$$ onto $$q_1$$:

$$
\langle v_2, q_1 \rangle = \frac{1}{\sqrt{2}}(1 \cdot 1 + 0 \cdot 1) = \frac{1}{\sqrt{2}}
$$

Then:

$$
u_2 = v_2 - \langle v_2, q_1 \rangle q_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix} - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} - \frac{1}{2} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} \frac{1}{2} \\ -\frac{1}{2} \end{bmatrix}
$$

Normalize:

$$
q_2 = \frac{u_2}{\|u_2\|} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

So the orthonormal basis is:

$$
q_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad q_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

---

### Applications

- **QR Decomposition**: Gram-Schmidt is used to compute orthogonal matrix $$Q$$ in QR.
- **PCA and SVD**: Constructs orthonormal eigenvectors or singular vectors.
- **Feature Embeddings**: Ensures orthogonality between embedding dimensions for decorrelation.

---

## 3. Orthogonal Initialization in Neural Networks

### Motivation

Deep networks are sensitive to weight initialization. Poorly chosen initializations can lead to:

- Vanishing or exploding gradients,
- Poor convergence,
- Suboptimal generalization.

Orthogonal initialization addresses these issues by ensuring that:

- The weight matrix $$W$$ preserves the norm of the signal during forward and backward passes,
- Gradients are not distorted across layers.

---

### Mathematical Principle

Let $$W \in \mathbb{R}^{n \times n}$$ be an orthogonal matrix, so that:

$$
W^T W = WW^T = I
$$

If input $$x \in \mathbb{R}^n$$ is passed through such a weight matrix:

$$
y = Wx \Rightarrow \|y\|_2 = \|x\|_2
$$

Hence, the norm is preserved, avoiding scaling issues layer by layer.

In backpropagation, the gradient flow also remains stable:

$$
\frac{\partial L}{\partial x} = W^T \frac{\partial L}{\partial y}
$$

If $$W$$ is orthogonal, the gradient is rotated but not scaled, avoiding vanishing/exploding gradients.

---

### How to Initialize Orthogonal Matrices

- Generate a random matrix $$A$$,
- Perform QR decomposition: $$A = QR$$,
- Use $$Q$$ as the initialization matrix (optionally scale by a factor $$\sigma$$):

$$
W = \sigma Q
$$

### Implementing in Deep Learning Libraries

In PyTorch:

```python
import torch.nn as nn
nn.init.orthogonal_(tensor, gain=1.0)
```

In TensorFlow:

```python
initializer = tf.keras.initializers.Orthogonal(gain=1.0)
```

### Applications

- **Recurrent Neural Networks (RNNs)**: Orthogonal or unitary weight matrices are crucial for long-term memory retention.
- **Deep Fully Connected Networks**: Improves training dynamics for very deep MLPs.
- **Transformer Layers**: Can help in initializing dense layers to preserve signal variance.

---

Orthogonalization is not only a theoretical concept but also a practical tool that:
- Enhances numerical stability,
- Helps decorrelate features,
- Enables better gradient flow in deep neural networks.

Whether through **Gram-Schmidt orthogonalization** for structured bases or **orthogonal initialization** for training deep networks, mastering these tools improves both understanding and implementation of modern ML models.



---

# Kernel Methods and Hilbert Spaces in Machine Learning

Kernel methods are powerful tools that enable learning algorithms to operate in **high-dimensional feature spaces** without explicitly computing those spaces. They are foundational to algorithms like **Support Vector Machines (SVMs)**, **Gaussian Processes (GPs)**, and **Kernel PCA**. This is achieved through the **kernel trick** and formalized via the theory of **Hilbert spaces** and **Mercer’s theorem**.

In this section, we will cover:

- Inner products in high-dimensional (possibly infinite-dimensional) spaces,
- The concept of Reproducing Kernel Hilbert Spaces (RKHS),
- The kernel trick and its use in ML algorithms,
- Mercer’s theorem and the mathematical foundation of kernels.

---

## 1. Inner Products in High-Dimensional Spaces

### Motivation

In many ML algorithms, we compute **inner products** between feature vectors:

$$
\langle x, x' \rangle = x^\top x'
$$

However, linear models using this inner product are limited to **linear decision boundaries**.

To handle **non-linear patterns**, we can **map** the inputs into a higher-dimensional space:

$$
\phi: \mathcal{X} \rightarrow \mathcal{H}, \quad x \mapsto \phi(x)
$$

where $$\mathcal{H}$$ is a **Hilbert space** (a complete inner product space). Then, inner products become:

$$
\langle \phi(x), \phi(x') \rangle
$$

If $$\phi$$ maps into a **very high-dimensional** or **infinite-dimensional** space, computation becomes infeasible. However, we can avoid this cost using **kernels**.

---

## 2. Kernel Functions and the Kernel Trick

### Definition

A **kernel function** $$k: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$$ is defined as:

$$
k(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}}
$$

This allows computing the inner product **without explicitly computing** $$\phi(x)$$ or $$\phi(x')$$.

This is known as the **kernel trick**.

### Examples of Kernel Functions

- **Linear kernel**:

$$
k(x, x') = x^\top x'
$$

- **Polynomial kernel** (degree $$d$$):

$$
k(x, x') = (x^\top x' + c)^d
$$

- **Radial Basis Function (RBF) / Gaussian kernel**:

$$
k(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)
$$

- **Sigmoid kernel**:

$$
k(x, x') = \tanh(\alpha x^\top x' + c)
$$

---

### Why It Works

Many algorithms (like SVMs) rely on computing dot products between data points:

- Training: $$x_i^\top x_j$$
- Prediction: $$x^\top x_i$$

By replacing these with $$k(x_i, x_j)$$ and $$k(x, x_i)$$, we effectively operate in the high-dimensional space **without ever computing it** explicitly.

This is computationally and memory-efficient and allows fitting complex, non-linear decision boundaries.

---

## 3. Reproducing Kernel Hilbert Space (RKHS)

### Definition

A **Reproducing Kernel Hilbert Space** is a Hilbert space $$\mathcal{H}$$ of functions from $$\mathcal{X} \rightarrow \mathbb{R}$$ such that:

1. For all $$x \in \mathcal{X}$$, the evaluation functional $$f \mapsto f(x)$$ is continuous.
2. There exists a **kernel function** $$k(x, \cdot) \in \mathcal{H}$$ such that for all $$f \in \mathcal{H}$$:

$$
f(x) = \langle f, k(x, \cdot) \rangle_{\mathcal{H}}
$$

This is called the **reproducing property**, and it guarantees that the kernel uniquely defines the Hilbert space.

### Intuition

RKHS is the space induced by the kernel. Every function in the RKHS can be written in terms of kernels evaluated at training points. This leads to **kernel representer theorems** and simplifies optimization.

---

## 4. Mercer’s Theorem

### Statement

Mercer's Theorem provides a condition under which a function is a **valid kernel** (i.e., corresponds to an inner product in some Hilbert space).

Let $$k(x, x')$$ be a continuous, symmetric, and positive semi-definite kernel on a compact domain $$\mathcal{X} \subset \mathbb{R}^n$$. Then:

- There exists a sequence of **orthonormal eigenfunctions** $$\{\phi_i\}$$ and **non-negative eigenvalues** $$\{\lambda_i\}$$ such that:

$$
k(x, x') = \sum_{i=1}^{\infty} \lambda_i \phi_i(x) \phi_i(x')
$$

This shows that a kernel corresponds to an inner product in a (possibly infinite-dimensional) feature space.

### Practical Use

Mercer’s Theorem justifies that functions like the RBF or polynomial kernel are **valid kernels** and thus induce a real Hilbert space with meaningful inner products.

---

## 5. Applications in Machine Learning

### 5.1 Support Vector Machines (SVMs)

The dual form of the SVM optimization problem involves only dot products. With a kernel function:

$$
\text{maximize}_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j k(x_i, x_j)
$$

Predictions are made using:

$$
f(x) = \sum_i \alpha_i y_i k(x, x_i) + b
$$

This enables **non-linear classification** by implicitly mapping into a high-dimensional space.

---

### 5.2 Gaussian Processes (GPs)

A Gaussian Process is a distribution over functions:

$$
f(x) \sim \mathcal{GP}(0, k(x, x'))
$$

The kernel function $$k$$ defines the **covariance** between function values. This allows modeling smoothness, periodicity, and other function properties.

Prediction involves computing:

$$
\mathbb{E}[f(x_*)] = k(x_*, X) [K(X, X) + \sigma^2 I]^{-1} y
$$

Where:
- $$K(X, X)$$ is the Gram matrix over the training data using kernel $$k$$,
- $$x_*$$ is the test input.

---

### 5.3 Kernel PCA

Standard PCA uses the covariance matrix:

$$
C = \frac{1}{n} \sum_{i=1}^n x_i x_i^T
$$

Kernel PCA generalizes this using the kernel trick by computing the **kernel Gram matrix**:

$$
K_{ij} = k(x_i, x_j)
$$

Then perform eigen-decomposition on the **centered** kernel matrix:

$$
K_c = K - \mathbf{1}K - K\mathbf{1} + \mathbf{1}K\mathbf{1}
$$

Where $$\mathbf{1}$$ is the centering matrix.

The result is non-linear dimensionality reduction using kernel-defined similarities.

---


## Summary

| Concept            | Mathematical Foundation                           | Role in Machine Learning                         |
| :----------------- | :-----------------------------------------------: | -----------------------------------------------: |
| Feature Mapping     | $$\phi: x \mapsto \mathcal{H}$$                   | Transforms data into high-dimensional space       |
| Kernel Function     | $$k(x, x') = \langle \phi(x), \phi(x') \rangle$$ | Computes inner products without explicit mapping  |
| RKHS                | Function space induced by a valid kernel         | Guarantees expressiveness and optimization theory |
| Mercer’s Theorem    | $$k(x, x') = \sum \lambda_i \phi_i(x) \phi_i(x')$$ | Validates kernel as inner product in Hilbert space |
| Kernel Trick        | Replace $$\langle x, x' \rangle$$ with $$k(x, x')$$ | Enables non-linear learning with linear algorithms |



---


Kernel methods let us apply powerful linear algorithms in high-dimensional non-linear spaces—**without computing those spaces directly**. The kernel trick, rooted in Hilbert space theory and Mercer’s theorem, forms the backbone of some of the most elegant and effective machine learning algorithms.


---

# Sparse Matrices and Sparsity in Machine Learning

Modern machine learning applications often involve **high-dimensional data**, where most entries are **zero**. In such settings, **sparsity** becomes a powerful property to exploit—both for **computational efficiency** and for **improving generalization**.

This section explores:

- Sparse matrix representation and storage,
- The role of sparsity in large-scale ML problems,
- Compressed sensing and sparse coding,
- Applications in recommender systems, NLP, computer vision, and feature selection.

---

## 1. Sparse Representations

### Definition

A matrix $$A \in \mathbb{R}^{m \times n}$$ is called **sparse** if most of its entries are zero:

$$
\text{Sparsity}(A) = \frac{\text{Number of zero entries}}{mn} \gg 0
$$

Equivalently, it has **only a few non-zero entries** compared to its total size.

---

### Storage and Efficiency

Instead of storing all $$mn$$ entries, we store only the non-zero elements and their indices. Common sparse matrix formats include:

- **Compressed Sparse Row (CSR)**:
  Stores non-zero values, column indices, and row pointer.

- **Compressed Sparse Column (CSC)**:
  Similar to CSR, but column-wise.

- **Coordinate (COO)**:
  Stores each non-zero entry as a tuple $$(i, j, A_{ij})$$.

These formats enable:
- **O(1)** access to non-zero elements,
- Efficient matrix-vector multiplication in $$O(\text{nnz})$$ time,
- Reduced memory usage.

---

### Use Cases

- **Recommender Systems**:
  User-item interaction matrices are sparse; most users rate very few items.

- **NLP**:
  Bag-of-Words, TF-IDF, one-hot encoding—all result in sparse representations.

- **Graph ML**:
  Adjacency matrices of large graphs (social networks, web graphs) are sparse.

---

## 2. Compressed Sensing

### Motivation

Compressed sensing addresses the question:

**Can we recover high-dimensional signals from few linear measurements if the signal is sparse?**

The answer is yes—under specific conditions.

---

### Problem Setup

Let:
- $$x \in \mathbb{R}^n$$ be the **original signal**, sparse in some basis.
- $$y \in \mathbb{R}^m$$ be the **observed measurements**, where $$m \ll n$$.
- $$A \in \mathbb{R}^{m \times n}$$ be a measurement matrix.

We want to solve:

$$
y = A x \quad \text{with} \quad x \text{ sparse}
$$

Since $$m < n$$, this is underdetermined. But if $$x$$ is sparse (i.e., $$\|x\|_0 \ll n$$), recovery is possible.

---

### Recovery via Optimization

The sparse recovery problem is posed as:

$$
\min_x \|x\|_0 \quad \text{subject to} \quad y = Ax
$$

But this is NP-hard. Instead, we solve:

$$
\min_x \|x\|_1 \quad \text{subject to} \quad y = Ax
$$

This is known as **Basis Pursuit**, and under the **Restricted Isometry Property (RIP)**, it provably recovers the sparse signal.

### Lasso (Relaxed version)

If measurements are noisy:

$$
y = Ax + \varepsilon
$$

Then we solve:

$$
\min_x \|y - Ax\|_2^2 + \lambda \|x\|_1
$$

This is the **Lasso** (Least Absolute Shrinkage and Selection Operator) formulation.

---

### Applications

- **Medical imaging**: MRI and CT scan reconstruction from fewer samples.
- **Signal processing**: Denoising, compression.
- **NLP and CV**: Learning compact word/image representations.

---

## 3. Sparse Coding

### Definition

Given an input signal $$x \in \mathbb{R}^d$$, sparse coding assumes:

$$
x \approx D h
$$

Where:
- $$D \in \mathbb{R}^{d \times k}$$ is a **dictionary** of basis vectors (atoms),
- $$h \in \mathbb{R}^k$$ is a **sparse code** (i.e., few non-zero entries).

The goal is to **learn $$D$$ and $$h$$** such that $$x$$ is well-represented with a **sparse $$h$$**.

---

### Optimization Objective

Given training data $$X = [x^{(1)}, \ldots, x^{(n)}]$$, we solve:

$$
\min_{D, H} \sum_{i=1}^n \left( \|x^{(i)} - D h^{(i)}\|_2^2 + \lambda \|h^{(i)}\|_1 \right)
$$

Subject to normalization constraints on $$D$$ (e.g., $$\|d_j\|_2 \leq 1$$).

This problem is **bi-convex** and typically solved via alternating minimization:
1. Fix $$D$$, update sparse codes $$H$$.
2. Fix $$H$$, update dictionary $$D$$.

---

### Interpretation

- Sparse coding learns **overcomplete dictionaries** that can represent inputs efficiently.
- The sparse codes $$h$$ capture **salient features** with fewer active components.

---

### Applications

- **Image processing**:
  - Denoising, super-resolution, texture synthesis.
- **NLP**:
  - Sparse embedding representations.
- **Feature learning**:
  - Unsupervised pretraining for deep networks.
- **Compression**:
  - Reduces storage and computation for large models.

---

## Summary

| Concept             | Mathematical Description                                 | Applications                                  |
| :------------------ | :------------------------------------------------------: | --------------------------------------------: |
| Sparse Matrix        | Mostly zero entries, stored in CSR/COO format            | Recommender systems, NLP, graph ML            |
| Compressed Sensing   | $$y = Ax, \|x\|_0 \ll n$$, recover $$x$$ from $$y$$       | Imaging, signal processing, low-data regimes  |
| Lasso                | $$\min_x \|y - Ax\|_2^2 + \lambda \|x\|_1$$              | Feature selection, regularization             |
| Sparse Coding        | $$x \approx D h$$ with sparse $$h$$                      | Feature learning, representation compression  |

---


Sparsity is a crucial structural assumption in many ML settings. Whether it's handling massive sparse data structures, recovering signals from limited observations, or learning compressed representations, **sparsity enables scalable and interpretable learning**.

The underlying mathematical principles—$$\ell_0$$ and $$\ell_1$$ norms, underdetermined systems, and structured optimization—form the basis of compressed sensing and sparse coding.

---

# Numerical Stability and Conditioning in Machine Learning

In large-scale machine learning models and numerical computations, **numerical stability** plays a vital role in ensuring accuracy, convergence, and generalization. When data is high-dimensional, features are correlated, or optimization problems are ill-posed, small numerical errors can lead to large deviations in results.

This section explores:

- Matrix conditioning and condition numbers,
- Their effect on optimization and linear models,
- Regularization strategies like **Tikhonov regularization** and **Ridge regression**,
- Real-world applications after each theoretical concept.

---

## 1. Matrix Conditioning and Condition Numbers

### Definition

Given a matrix $$A \in \mathbb{R}^{n \times n}$$, its **condition number** in the 2-norm is:

$$
\kappa(A) = \|A\|_2 \cdot \|A^{-1}\|_2
$$

If $$A$$ is symmetric positive definite:

$$
\kappa(A) = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

Where $$\lambda_{\max}$$ and $$\lambda_{\min}$$ are the largest and smallest eigenvalues of $$A$$.

This value measures the sensitivity of the solution $$x$$ to small perturbations in the system $$Ax = b$$.

---

### Application: Linear Systems and Inversion Stability

In **linear regression** or **least squares**, solving:

$$
\hat{x} = (X^T X)^{-1} X^T y
$$

can be unstable if $$X^T X$$ is ill-conditioned (e.g., due to multicollinearity).

**Real-World Scenarios**:
- High-dimensional regression models,
- Polynomial regression (where powers of features become highly correlated),
- PCA: covariance matrix conditioning affects eigenvalue computation.

---

## 2. Instability in Optimization

### Gradient Descent Sensitivity

Consider optimizing a quadratic loss:

$$
f(x) = \frac{1}{2} x^T A x
$$

Gradient descent update:

$$
x_{k+1} = x_k - \eta A x_k
$$

If $$A$$ is ill-conditioned (i.e., eigenvalues vary widely), the optimization will:
- Converge slowly,
- Zigzag across the cost surface,
- Require small learning rates to remain stable.

---

### Application: Neural Network Training

In **deep networks**, layers may learn at drastically different rates due to ill-conditioning. Common symptoms include:
- Exploding or vanishing gradients,
- Slow convergence even on simple tasks,
- Difficulty tuning learning rates.

**Example**: In early training of deep MLPs or RNNs, poor weight scaling leads to ill-conditioned Jacobians and Hessians.

---

## 3. Tikhonov Regularization

### Theory

To solve an ill-posed least squares problem:

$$
\min_x \|Ax - b\|_2^2
$$

we add a regularization term:

$$
\min_x \|Ax - b\|_2^2 + \lambda \|x\|_2^2
$$

This is **Tikhonov regularization**. The solution becomes:

$$
\hat{x}_\lambda = (A^T A + \lambda I)^{-1} A^T b
$$

This improves conditioning by ensuring the matrix being inverted is better-behaved.

---

### Application: Ill-posed Inverse Problems

**Tikhonov regularization** is used in:
- **Image deblurring and denoising**,
- **Medical imaging (MRI, CT)**,
- **Physics-based simulations** with uncertain measurements.

In ML, it improves:
- **Matrix inversion stability** in linear models,
- **Numerical robustness** in batch/mini-batch computations.

---

## 4. Ridge Regression

### Theory

Ridge regression is a specific case of Tikhonov regularization applied to linear regression. Given $$X \in \mathbb{R}^{n \times d}$$ and $$y \in \mathbb{R}^n$$:

$$
\min_w \|Xw - y\|_2^2 + \lambda \|w\|_2^2
$$

Solution:

$$
w = (X^T X + \lambda I)^{-1} X^T y
$$

Benefits:
- Stabilizes matrix inversion,
- Reduces overfitting in high-dimensional settings.

---

### Application: High-Dimensional Linear Models

Ridge regression is essential when:
- The number of features exceeds the number of samples,
- Features are highly correlated (multicollinearity),
- Predictors are noisy or redundant.

**Example**:
```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(X, y)
```

Common in:
- **Genomics** (p >> n),
- **Text regression models** with bag-of-words features,
- **Financial models** with redundant indicators.

---

## 5. Conditioning in Deep Learning

### Common Problems

In deep neural networks:
- Weight matrices may become poorly conditioned,
- Gradients may vanish or explode during backpropagation,
- Activations may saturate, leading to optimization stalls.

---

### Solutions and Their Mathematical Roles

1. **Orthogonal Initialization**: Weight matrices initialized to be orthogonal preserve input norm and maintain conditioning.

   $$
   W^T W = I \Rightarrow \|Wx\|_2 = \|x\|_2
   $$

   **Code**:
   ```python
   torch.nn.init.orthogonal_(tensor)
   ```

2. **Weight Decay** (L2 regularization): Equivalent to Ridge on weights. Controls weight growth, stabilizes learning.

   $$
   \min_w \mathcal{L}(w) + \lambda \|w\|_2^2
   $$

3. **Gradient Clipping**: Prevents gradient explosion by clipping:

   $$
   \nabla \mathcal{L} \leftarrow \frac{\nabla \mathcal{L}}{\max(1, \|\nabla \mathcal{L}\| / \tau)}
   $$

   **Code**:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
   ```

---

### Application: Training Stability in Deep Networks

These techniques are used in:
- **RNNs**: To avoid exploding gradients through time.
- **Transformers**: LayerNorm and initialization stabilize long-depth training.
- **CNNs**: Weight decay improves generalization and convergence.

---

## Summary

| Concept                 | Mathematical Description                                           | Role in ML / DL                                      |
| :---------------------- | :---------------------------------------------------------------: | ---------------------------------------------------: |
| Condition Number        | $$\kappa(A) = \|A\| \cdot \|A^{-1}\|$$                              | Measures sensitivity to noise                       |
| Ill-conditioning        | $$\kappa(A) \gg 1$$                                                | Leads to instability in training and optimization   |
| Tikhonov Regularization | $$\min_x \|Ax - b\|^2 + \lambda \|x\|^2$$                          | Improves matrix invertibility                       |
| Ridge Regression        | $$w = (X^T X + \lambda I)^{-1} X^T y$$                             | Stabilizes regression with correlated features      |
| Orthogonal Init.        | $$W^T W = I$$                                                      | Preserves norm in forward/backward pass             |
| Weight Decay            | $$\min_w \mathcal{L} + \lambda \|w\|^2$$                           | Regularizes weights and enhances generalization     |
| Gradient Clipping       | $$\nabla \mathcal{L} \rightarrow \frac{\nabla \mathcal{L}}{\max(\cdot)}$$ | Prevents exploding gradients during training        |

---


Understanding numerical stability and matrix conditioning helps you:
- Design models that train efficiently,
- Use optimization methods that converge reliably,
- Avoid silent failures due to ill-conditioning.

By incorporating **regularization**, **better initialization**, and **gradient control**, you can ensure your machine learning models are not only performant—but also numerically robust.


---

# Rotation, Reflection, and Markov Matrices in Machine Learning

Geometric and probabilistic transformations form the backbone of many machine learning systems—especially in computer vision, robotics, and probabilistic reasoning. This section explores two foundational categories of matrices:

1. **Rotation and Reflection matrices** (used in spatial transformations),
2. **Markov (Stochastic) matrices** (used in probabilistic models and temporal systems).

---

## Rotation and Reflection Matrices

Geometric transformations such as rotation and reflection are represented by orthogonal matrices in linear algebra. These are crucial for tasks in **computer vision**, **robotics**, **3D graphics**, and **data augmentation**.

---

### Rotation Matrices

A **rotation matrix** in $$\mathbb{R}^n$$ rotates a vector about the origin while preserving its norm. A matrix $$R$$ is a rotation matrix if:

$$
R^T R = RR^T = I \quad \text{and} \quad \det(R) = 1
$$

#### 2D Rotation

In two dimensions, rotation by an angle $$\theta$$ counterclockwise is given by:

$$
R(\theta) = \begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$

For any vector $$x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$, the rotated vector is:

$$
x' = R(\theta) x
$$

#### 3D Rotation and Euler Angles

In 3D, rotations can be performed around each axis. The elementary rotations are:

- Around $$x$$-axis:
  $$
  R_x(\theta) = \begin{bmatrix}
  1 & 0 & 0 \\
  0 & \cos\theta & -\sin\theta \\
  0 & \sin\theta & \cos\theta
  \end{bmatrix}
  $$

- Around $$y$$-axis:
  $$
  R_y(\theta) = \begin{bmatrix}
  \cos\theta & 0 & \sin\theta \\
  0 & 1 & 0 \\
  -\sin\theta & 0 & \cos\theta
  \end{bmatrix}
  $$

- Around $$z$$-axis:
  $$
  R_z(\theta) = \begin{bmatrix}
  \cos\theta & -\sin\theta & 0 \\
  \sin\theta & \cos\theta & 0 \\
  0 & 0 & 1
  \end{bmatrix}
  $$

Any 3D rotation can be represented by a combination of these, often using **Euler angles**.

---

### Application: Computer Vision and Robotics

- **Image Augmentation**: Rotating images during training increases robustness to orientation.
- **Pose Estimation**: Estimating camera or robot orientation using rotation matrices.
- **3D Reconstruction**: Applying transformations to point clouds and mesh data.
- **Robotics Control**: Planning and executing movement using rotation matrices in kinematics.

**Example**:
```python
import torchvision.transforms as T
T.RandomRotation(degrees=15)
```

---

### Reflection Matrices

A **reflection matrix** flips a vector across a subspace (hyperplane). It is also orthogonal, but unlike a rotation matrix:

$$
\det(R) = -1
$$

#### Reflection in 2D

Reflection across the $$x$$-axis:

$$
R = \begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
$$

Reflection across an arbitrary line through the origin with unit normal vector $$n$$:

$$
R = I - 2nn^T
$$

Where $$n \in \mathbb{R}^d$$ and $$\|n\| = 1$$.

---

### Application: Data Augmentation in CV

Reflections are used to simulate different perspectives of the same object.

- **Horizontal Flip**:
```python
T.RandomHorizontalFlip(p=0.5)
```

- **Vertical Flip**:
```python
T.RandomVerticalFlip(p=0.5)
```

- **Symmetry-based Learning**: Useful in object detection and scene understanding.

---

## Markov Matrices and Stochastic Processes

Markov models describe systems that evolve probabilistically over time, where the future depends only on the current state. Their underlying structure is captured by **stochastic (Markov) matrices**.

---

### Stochastic Matrices

A **Markov matrix** (or **row-stochastic matrix**) is a square matrix $$P \in \mathbb{R}^{n \times n}$$ where:

1. $$P_{ij} \geq 0$$ for all $$i, j$$,
2. $$\sum_{j=1}^n P_{ij} = 1$$ for all $$i$$ (each row sums to 1).

This means $$P_{ij}$$ is the probability of transitioning from state $$i$$ to state $$j$$.

Let $$x_t$$ be a **probability distribution vector** at time $$t$$. Then:

$$
x_{t+1} = x_t P
$$

This recurrence describes the **evolution of a Markov chain** over time.

---

### Stationary Distribution

A distribution $$\pi$$ is **stationary** if:

$$
\pi = \pi P
$$

This represents the long-term distribution of states. Under mild conditions (irreducibility, aperiodicity), every Markov chain has a unique stationary distribution.

---

### Application: PageRank and Random Walks

**PageRank Algorithm**:
- Models the web as a Markov chain.
- Pages are states, links are transitions.
- Uses a stochastic matrix with damping:

$$
P' = \alpha P + (1 - \alpha) \frac{1}{n} \mathbf{1}\mathbf{1}^T
$$

Stationary distribution $$\pi$$ is computed such that:

$$
\pi = \pi P'
$$

Other Applications:
- **Language Modeling**: Character or word-level Markov chains.
- **Hidden Markov Models (HMMs)**: NLP, speech, time-series.
- **Graph algorithms**: Random walk-based node ranking and clustering.

---

## Summary

| Concept                 | Mathematical Description                                        | Applications                                   |
| :---------------------- | :------------------------------------------------------------: | ---------------------------------------------: |
| Rotation Matrix         | $$R^T R = I, \ \det(R) = 1$$                                    | CV, robotics, 3D vision, pose estimation       |
| Reflection Matrix       | $$R = I - 2nn^T, \ \det(R) = -1$$                              | Data augmentation, symmetry modeling           |
| Euler Angles            | Composition of axis-wise rotations                             | Robotics, camera modeling                      |
| Stochastic Matrix       | $$P_{ij} \geq 0, \ \sum_j P_{ij} = 1$$                         | Markov chains, PageRank, probabilistic models  |
| Stationary Distribution | $$\pi = \pi P$$                                                | Long-term behavior modeling                    |

---



Rotation and reflection matrices form the mathematical backbone of **spatial transformations** in ML applications like vision and robotics. Meanwhile, Markov matrices provide a **probabilistic framework** to model temporal evolution in tasks such as **language modeling**, **search ranking**, and **sequence prediction**.



---

# Advanced Projections: Random Projections and the Johnson–Lindenstrauss Lemma

In modern machine learning and data science, high-dimensional datasets are common—particularly in fields like **natural language processing**, **image processing**, and **information retrieval**. However, high dimensionality brings computational and storage challenges, as well as the **curse of dimensionality**.

One elegant solution to this is **random projection**, a method for dimensionality reduction that is fast, scalable, and surprisingly effective. Its theoretical foundation is the **Johnson–Lindenstrauss Lemma**, which guarantees that random projections approximately preserve distances between points.

This section explains:

- The theory and math behind random projections,
- The Johnson–Lindenstrauss Lemma and its guarantees,
- Applications in NLP, IR, privacy, and beyond.

---

## 1. The Need for Dimensionality Reduction

Let $$X \in \mathbb{R}^{n \times d}$$ be a dataset with $$n$$ samples in a high-dimensional space $$\mathbb{R}^d$$.

Problems with large $$d$$:
- **Computational inefficiency**: Matrix operations are expensive.
- **Memory consumption**: Storing all features is costly.
- **Overfitting**: Too many features relative to data points.
- **Distance concentration**: In high dimensions, pairwise distances become less informative.

Goal: Reduce the dimensionality from $$d$$ to $$k \ll d$$ such that **geometric structure (e.g., pairwise distances)** is preserved.

---

## 2. Random Projections

Instead of learning an optimal projection (like PCA), **random projections** use a **random linear map**:

Let $$R \in \mathbb{R}^{k \times d}$$ be a random matrix. Then:

$$
z_i = \frac{1}{\sqrt{k}} R x_i \in \mathbb{R}^k
$$

for each data point $$x_i \in \mathbb{R}^d$$.

The random matrix $$R$$ typically has entries sampled from:
- Standard Gaussian: $$R_{ij} \sim \mathcal{N}(0, 1)$$,
- Sparse sign matrices: $$R_{ij} \in \{-1, 0, +1\}$$ with controlled sparsity.

---

## 3. Johnson–Lindenstrauss Lemma

The **Johnson–Lindenstrauss Lemma** states that a small set of points in high-dimensional space can be mapped into a lower-dimensional space such that pairwise distances are approximately preserved.

### Theorem (JL Lemma)

For any $$0 < \epsilon < 1$$ and integer $$n$$, let $$X = \{x_1, \ldots, x_n\} \subset \mathbb{R}^d$$ be a set of $$n$$ points. Then for:

$$
k = O\left(\frac{\log n}{\epsilon^2}\right),
$$

there exists a linear map $$f: \mathbb{R}^d \rightarrow \mathbb{R}^k$$ such that for all $$i, j$$:

$$
(1 - \epsilon)\|x_i - x_j\|_2^2 \leq \|f(x_i) - f(x_j)\|_2^2 \leq (1 + \epsilon)\|x_i - x_j\|_2^2
$$

This means that **random projections preserve pairwise distances up to small distortion** with high probability.

---

### Intuition

- The JL Lemma shows that **no information-theoretic bottleneck** exists when compressing data from $$\mathbb{R}^d$$ to $$\mathbb{R}^k$$, as long as $$k = O(\log n)$$.
- This is **data-independent**: No need to look at the data when designing the projection.

---

## 4. Construction of Projection Matrix

Let $$R \in \mathbb{R}^{k \times d}$$ be the random matrix. Some common constructions:

### 4.1 Gaussian Random Projection

Each entry is drawn i.i.d. from:

$$
R_{ij} \sim \mathcal{N}(0, 1)
$$

Then the projection is:

$$
f(x) = \frac{1}{\sqrt{k}} R x
$$

This satisfies the JL lemma with high probability.

---

### 4.2 Sparse Random Projection

To improve speed and memory:

$$
R_{ij} = \sqrt{s} \cdot
\begin{cases}
+1 & \text{with probability } \frac{1}{2s}, \\
0  & \text{with probability } 1 - \frac{1}{s}, \\
-1 & \text{with probability } \frac{1}{2s}
\end{cases}
$$

For example, $$s = 3$$ gives ~67% sparsity.

---

## 5. Applications

### 5.1 Natural Language Processing (NLP)

- **TF-IDF vectors** for documents can have tens of thousands of dimensions.
- Random projections reduce dimensionality for:
  - **Document classification**,
  - **Similarity search**,
  - **Topic modeling** pre-processing.

**Example**:
```python
from sklearn.random_projection import GaussianRandomProjection
transformer = GaussianRandomProjection(n_components=100)
X_new = transformer.fit_transform(X_tfidf)
```

---

### 5.2 Information Retrieval and ANN Search

- Used to **index high-dimensional vectors** for approximate nearest neighbors (ANN).
- Efficiently reduce the dimension of image embeddings, word embeddings, etc.
- Compatible with LSH (Locality-Sensitive Hashing).

---

### 5.3 Differential Privacy and Data Privacy

- Random projections are used to **obscure sensitive dimensions** while preserving utility.
- Also appear in **private matrix factorization** and federated learning pipelines.

---

### 5.4 Kernel Approximation

- **Random Fourier Features** approximate Gaussian/RBF kernels by projecting into a low-dimensional space.
- Scales kernel methods to large datasets:
  $$
  k(x, y) \approx \phi(x)^T \phi(y)
  $$

Where $$\phi(x)$$ is obtained via random projections.

---

## 6. Comparison with PCA

| Aspect                | PCA                                           | Random Projection                                  |
| :-------------------- | :-------------------------------------------: | -------------------------------------------------: |
| Data-dependent        | Yes                                           | No                                                 |
| Computational cost    | High (SVD-based)                              | Low (matrix multiplication)                        |
| Distance preservation | Optimal for top-variance directions           | Approximate, but probabilistically guaranteed      |
| Scalability           | Less scalable for large $$d$$                 | Highly scalable                                    |
| Interpretability      | High (axes = principal components)            | Low                                                |

---

## Summary

| Concept                    | Mathematical Idea                                         | Application Domains                                |
| :------------------------- | :--------------------------------------------------------: | -------------------------------------------------: |
| Johnson–Lindenstrauss Lemma| $$k = O\left(\frac{\log n}{\epsilon^2}\right)$$           | Distance-preserving low-dim embedding              |
| Gaussian Projection        | $$R_{ij} \sim \mathcal{N}(0, 1)$$                         | NLP, embeddings, privacy                           |
| Sparse Projection          | $$R_{ij} \in \{-1, 0, +1\}$$ with sparsity                | Faster computation                                 |
| Random Fourier Features    | Approx. kernel projection via random bases                | Kernel methods on large datasets                   |
| Distance Preservation      | $$\|x_i - x_j\| \approx \|f(x_i) - f(x_j)\|$$              | ANN, clustering, manifold learning                 |

---


**Random projections** offer a principled, efficient, and theoretically sound method to **compress high-dimensional data** while preserving its **geometric structure**. Thanks to the **Johnson–Lindenstrauss Lemma**, we can apply these projections without worrying about distortion—making them perfect for large-scale ML systems.

Whether you're dealing with:
- Large vocabulary document matrices in NLP,
- Embedding vectors in image retrieval,
- Kernel methods in high dimensions,

random projections are a tool you should definitely have in your toolbox.
