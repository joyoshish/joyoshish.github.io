---
layout: post
title: Linear Algebra Basics for ML - Eigenvalues, Eigenvectors, and Singular Value Decomposition
date: 2022-02-03
description: Linear Algebra 4 - Mathematics for Machine Learning
tags: ml ai linear-algebra math
math: true
categories: machine-learning math math-for-ml
# thumbnail: assets/img/linalg_banner.png
giscus_comments: true
related_posts: true
chart:
  plotly: true
toc:
  beginning: true
---


Understanding how data transforms under linear mappings is core to machine learning, and eigenvalues and eigenvectors lie at the heart of this. Whether we're analyzing variance in high-dimensional data or compressing information with minimal loss, these concepts help us simplify and interpret complex systems. In this post, we’ll build intuition, explore the math behind eigen decomposition and diagonalization, and connect the dots to key ML applications.

---

## What Makes Eigenvectors Special?

When we apply a matrix transformation to a vector, we typically expect its direction to change. But some vectors resist this change—they may get stretched or shrunk, but they stay on the same line. These are the **eigenvectors** of the transformation.

Formally, if $$A$$ is a square matrix, and $$\mathbf{v}$$ is a vector such that:

$$
A \mathbf{v} = \lambda \mathbf{v},
$$

then $$\mathbf{v}$$ is called an eigenvector of $$A$$ and $$\lambda$$ is the corresponding **eigenvalue**. The transformation scales the vector by $$\lambda$$ without rotating it.

To visualize this, imagine applying a matrix that stretches space vertically. Vectors aligned with the vertical axis simply stretch—those are eigenvectors with eigenvalues greater than 1. Vectors aligned with the horizontal axis may stay the same length (eigenvalue = 1) or stretch differently. But vectors at diagonal angles typically change direction; they’re not eigenvectors.

This geometric idea is powerful in ML: eigenvectors identify the key **directions** in which data varies, and eigenvalues tell us **how much** variation lies in each direction.

---

## Why Do Eigenvectors Matter in ML?

In machine learning, eigen decomposition helps us understand structure and simplify computations. Take **Principal Component Analysis (PCA)** as an example—it identifies the directions (principal components) along which data varies most. These directions are eigenvectors of the covariance matrix. The corresponding eigenvalues tell us how much variance each direction captures.

This insight drives a wide range of applications:
- **Dimensionality reduction:** Eliminate less informative directions to compress data.
- **Feature selection:** Identify directions with the most meaningful variation.
- **Interpretability:** Understand how transformations like matrix multiplications affect data.
- **Stability and convergence:** In optimization and iterative methods, the dominant eigenvalue often controls convergence speed.

In short, eigenvalues and eigenvectors distill the essence of a transformation.

---

## Eigen-Decomposition and Diagonalization

Let’s now formalize this intuition. Given a square matrix $$A \in \mathbb{R}^{n \times n}$$, an eigen-decomposition expresses it in terms of its eigenvectors and eigenvalues. First, we solve the **characteristic equation**:

$$
\det(A - \lambda I) = 0
$$

The roots $$\lambda_1, \lambda_2, \dots, \lambda_n$$ are the eigenvalues of $$A$$. For each eigenvalue, we solve:

$$
(A - \lambda I)\mathbf{v} = 0
$$

to find the corresponding eigenvector $$\mathbf{v}$$.

If $$A$$ has $$n$$ linearly independent eigenvectors, we can form a matrix $$Q$$ whose columns are those eigenvectors. Let $$\Lambda$$ be the diagonal matrix of the corresponding eigenvalues:

$$
Q = [\mathbf{v}_1\ \mathbf{v}_2\ \cdots\ \mathbf{v}_n], \quad \Lambda = \begin{bmatrix}
\lambda_1 & 0 & \cdots & 0 \\
0 & \lambda_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_n
\end{bmatrix}
$$

Then the matrix $$A$$ can be written as:

$$
A = Q \Lambda Q^{-1}
$$

This is the **diagonalization** of $$A$$. It tells us that in the basis formed by eigenvectors, the transformation $$A$$ just scales each coordinate by its eigenvalue.

For **symmetric matrices**, the situation is even better: the eigenvectors are orthogonal and $$Q$$ becomes an orthogonal matrix (i.e., $$Q^{-1} = Q^T$$), giving:

$$
A = Q \Lambda Q^T
$$

This is known as the **spectral theorem**, and it holds for any real symmetric matrix—like covariance matrices, Laplacians, and Gram matrices in ML. The eigenvalues are always real, and the eigenvectors form an orthonormal basis.

---

## A Concrete Example: Diagonalizing a Matrix

Let’s walk through an example. Suppose:

$$
A = \begin{pmatrix}
3 & 1 \\
0 & 2
\end{pmatrix}
$$

To find the eigenvalues, solve:

$$
\det(A - \lambda I) = (3 - \lambda)(2 - \lambda) = 0
$$

The roots are $$\lambda_1 = 3$$ and $$\lambda_2 = 2$$.

To find eigenvectors:
- For $$\lambda_1 = 3$$, solve $$(A - 3I)\mathbf{v} = 0$$:

$$
\begin{pmatrix}
0 & 1 \\
0 & -1
\end{pmatrix}
\begin{pmatrix}
v_1 \\
v_2
\end{pmatrix}
= 0
\quad \Rightarrow \quad v_2 = 0
$$

Choose $$v_1 = 1$$. So one eigenvector is $$\mathbf{v}^{(1)} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$.

- For $$\lambda_2 = 2$$, solve $$(A - 2I)\mathbf{v} = 0$$:

$$
\begin{pmatrix}
1 & 1 \\
0 & 0
\end{pmatrix}
\begin{pmatrix}
v_1 \\
v_2
\end{pmatrix}
= 0
\quad \Rightarrow \quad v_2 = -v_1
$$

Choose $$v_1 = 1$$. So $$\mathbf{v}^{(2)} = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$.

Now we build:

$$
Q = \begin{pmatrix}
1 & 1 \\
0 & -1
\end{pmatrix}, \quad
\Lambda = \begin{pmatrix}
3 & 0 \\
0 & 2
\end{pmatrix}
$$

We can verify that:

$$
A = Q \Lambda Q^{-1}
$$

In this eigenbasis, applying $$A$$ is as simple as scaling by 3 and 2 along the new axes.

---

## Why Diagonalization Matters in Machine Learning

In ML, diagonalization is not just a theoretical convenience—it enables faster computation and deeper insight.

When matrices are diagonalizable, operations like computing $$A^k$$, solving $$A \mathbf{x} = \mathbf{b}$$, or evaluating matrix exponentials become much easier. This is vital in **graph analysis**, **dynamical systems**, and **deep learning**, where matrix powers and iterative updates are common.

In **PCA**, the covariance matrix is symmetric and positive semi-definite, so we can always decompose it into eigenvectors and eigenvalues. The eigenvectors give us **uncorrelated directions** in feature space, and eigenvalues quantify how much variance lies in each direction. Keeping only the top $$k$$ components (largest eigenvalues) gives a compressed yet informative view of the data.

Diagonalization also underlies **spectral clustering**, **recommendation systems**, and **natural language processing**. In these cases, the eigenvectors often correspond to **latent patterns** or **community structures**, and the eigenvalues measure their significance. For instance, in spectral clustering, the eigenvectors of a graph Laplacian can reveal cluster boundaries. In NLP, eigenvectors from co-occurrence matrices highlight semantic dimensions in word relationships.

Another powerful insight comes from **power iteration**—a technique that leverages repeated matrix multiplication to find the dominant eigenvector. This behavior is at the core of algorithms like **PageRank**.

Lastly, the diagonalized form $$A = Q \Lambda Q^{-1}$$ gives us a **spectrum** of the transformation: the eigenvalues summarize key properties like total variance (via the trace) and overall scaling (via the determinant). These quantities link abstract algebra to concrete geometry and help us interpret the transformation as acting independently along each eigenvector axis.

## Properties of Eigenvalues and Eigenvectors

Understanding the properties of eigenvalues and eigenvectors reveals deep insights into linear transformations, variance, and the structure of data in machine learning. Here's a concise overview:

- **Scaling and Linearity**  
  If $$A \mathbf{v} = \lambda \mathbf{v}$$, then for any scalar $$c$$,  
  $$A (c \mathbf{v}) = \lambda (c \mathbf{v})$$.  
  So, any nonzero scalar multiple of an eigenvector is still an eigenvector associated with the same eigenvalue.  
  - The set of all such vectors (plus the zero vector) forms the **eigenspace**.  
  - A linear combination of two eigenvectors is generally not an eigenvector unless they share the same eigenvalue.

- **Invariant Subspaces**  
  Eigenvectors define one-dimensional invariant subspaces under transformation.  
  - The span of multiple eigenvectors (with corresponding eigenvalues $$\{\lambda_1, \lambda_2\}$$) also forms an invariant subspace.  
  - In PCA, the subspace spanned by the top $$k$$ eigenvectors captures the principal variance in data.

- **Geometric Interpretation**  
  Eigenvectors are directions in which the matrix acts as a pure stretch/compression:
  - $$\mid\lambda\mid > 1$$ → stretched  
  - $$\mid\lambda\mid < 1$$ → compressed  
  - $$\lambda = -1$$ → flipped (180° rotation)  
  - $$\lambda = 0$$ → squashed to the origin  
  In symmetric matrices, eigenvectors form orthogonal axes—ideal for interpreting transformations.

- **Variance Interpretation (PCA)**  
  For a covariance matrix $$S$$:
  - The eigenvector with the largest eigenvalue corresponds to the direction of **maximum variance**.
  - Eigenvalues represent the **amount of variance** along their eigenvector directions.
  - If eigenvalues are $$\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n$$ with eigenvectors $$e_1, e_2, \dots, e_n$$, then:
    - $$e_1$$ is the direction of highest variance.
    - $$\frac{\lambda_1}{\sum_i \lambda_i}$$ is the proportion of total variance captured by that direction.
  - The **sum** of all eigenvalues equals the **trace** of $$S$$ (total variance).

- **Orthogonality (for symmetric matrices)**  
  - Eigenvectors corresponding to different eigenvalues are orthogonal:  
    $$e_i^T e_j = 0 \text{ for } i \ne j$$  
  - We can choose an **orthonormal** set of eigenvectors for symmetric matrices, making projections and decompositions simpler.

- **Sum and Product of Eigenvalues**  
  - The **sum** of the eigenvalues of $$A$$ is equal to its **trace**:
    $$\sum_{i=1}^{n} \lambda_i = \text{tr}(A)$$  
  - The **product** of the eigenvalues equals the **determinant** of $$A$$:
    $$\prod_{i=1}^{n} \lambda_i = \det(A)$$  
  - If any eigenvalue is zero, $$A$$ is **singular**. If all eigenvalues are positive, $$A$$ is **positive definite**.

- **Left vs Right Eigenvectors**  
  - Typically, we refer to **right eigenvectors** satisfying $$A \mathbf{v} = \lambda \mathbf{v}$$.  
  - **Left eigenvectors** satisfy $$\mathbf{w}^T A = \lambda \mathbf{w}^T$$.  
  - For **symmetric matrices**, left and right eigenvectors are transposes of each other.
  - In applications like **PageRank**, the left eigenvector of the transition matrix (eigenvector of $$P^T$$) represents the stationary distribution.

These properties show how eigenvalues and eigenvectors help us understand invariant directions, the strength of transformation along those directions, and variance structure in data.


## Principal Component Analysis (PCA) Using Eigen Decomposition

Principal Component Analysis (PCA) is a powerful technique that reduces the dimensionality of data while preserving as much variance as possible. Given an $$m \times n$$ data matrix $$X$$ (with $$m$$ samples and $$n$$ features), the key idea is to find new axes—orthogonal directions in the feature space—that best capture the variance in the data. These axes are the principal components, which correspond to the top eigenvectors of the data’s covariance matrix.

To derive PCA, we first center the data by subtracting the mean from each feature, so that each column of $$X$$ has mean zero. Then, we compute the sample covariance matrix:

$$
S = \frac{1}{m - 1} X^T X
$$

The goal is to find a unit vector $$\mathbf{w} \in \mathbb{R}^n$$ such that the projection of the data onto $$\mathbf{w}$$ has the maximum possible variance. The variance along $$\mathbf{w}$$ is given by:

$$
\mathrm{Var}(X\mathbf{w}) = \mathbf{w}^T S \mathbf{w}
$$

We maximize this subject to the constraint $$\|\mathbf{w}\| = 1$$. This leads us to an eigenvalue problem. Since $$S$$ is symmetric and positive semidefinite, we can write:

$$
S = Q \Lambda Q^T
$$

Here, $$\Lambda$$ is a diagonal matrix of eigenvalues $$\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_n \ge 0$$, and $$Q$$ contains the corresponding orthonormal eigenvectors $$\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_n$$ as columns.

If we express $$\mathbf{w}$$ as $$Q \mathbf{v}$$ for some unit vector $$\mathbf{v}$$ (since $$Q$$ is orthonormal), the objective becomes:

$$
\mathbf{w}^T S \mathbf{w} = \mathbf{v}^T \Lambda \mathbf{v} = \sum_{i=1}^n \lambda_i v_i^2
$$

To maximize this weighted average of eigenvalues, we assign all weight to the largest eigenvalue by choosing $$\mathbf{v} = (1, 0, 0, \dots, 0)^T$$. This corresponds to choosing $$\mathbf{w} = \mathbf{q}_1$$, the eigenvector associated with $$\lambda_1$$. Thus, the first principal component is the direction of greatest variance in the data.

Subsequent components are found by selecting eigenvectors orthogonal to the previous ones, associated with the next highest eigenvalues. This way, projecting data onto the top $$k$$ eigenvectors captures the most variance possible among all $$k$$-dimensional subspaces.

Another perspective is that PCA finds the best rank-$$k$$ approximation of the data in terms of minimizing reconstruction error. This makes PCA not only a tool for understanding variance but also for efficient data compression and noise reduction.

To summarize, compute the covariance matrix:

$$
S = \frac{1}{m - 1} X^T X
$$

Then find eigenvectors $$\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_n$$ and corresponding eigenvalues $$\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_n$$. The proportion of variance captured by the top $$k$$ components is:

$$
\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{n} \lambda_i}
$$

Typically, one selects $$k$$ so that this ratio exceeds a threshold like 95%, ensuring most of the information is retained while significantly reducing dimensionality.

Once eigenvalues are computed, we often visualize them using a scree plot. This plot displays eigenvalues in descending order versus their component indices. A sharp "elbow" in the curve suggests that components beyond that point contribute little variance. For example, in the Iris dataset with four features, the scree plot shows the first eigenvalue around 4.2, the second about 0.24, and the remaining two near zero. This suggests that one or two principal components are sufficient to explain most of the variance.

In practice, PCA is widely used for visualization, especially when reducing high-dimensional data to 2 or 3 components. It also serves as a preprocessing step for other algorithms, helping to mitigate the curse of dimensionality or reduce noise. Since the principal components are uncorrelated (due to orthogonality of eigenvectors), they provide cleaner inputs for downstream models.

Here’s a Python example using scikit-learn on the Iris dataset:

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load and center the data
iris = load_iris()
X = iris.data
X = X - X.mean(axis=0)

# Fit PCA
pca = PCA(n_components=4)
pca.fit(X)

print("Eigenvalues:", pca.explained_variance_)
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Typical output might be:

```
Eigenvalues: [4.228 0.243 0.078 0.024]
Explained variance ratio: [0.9246 0.0531 0.0171 0.0052]
```

This confirms that the first component alone captures over 92% of the total variance, and the first two together over 97%. Projecting the data into these two dimensions gives a near-complete representation of the structure, making it ideal for plotting and exploratory analysis.

Finally, PCA also helps denoise data. Noise tends to appear as low-variance components (low eigenvalues), and by discarding those, we remove irrelevant variation. In real-world ML pipelines, one typically performs PCA on training data, selects the number of components via a scree or cumulative variance plot, and then transforms both training and test data into the lower-dimensional space for modeling. Libraries like scikit-learn internally use efficient algorithms like Singular Value Decomposition (SVD) to compute PCA, especially when the number of features is large.


---


# Singular Value Decomposition (SVD)

Eigen-decomposition we discussed is for square matrices (and especially symmetric matrices in many ML contexts). Singular Value Decomposition (SVD) is a more general matrix factorization that works for any $$m \times n$$ matrix (square or rectangular, symmetric or not). SVD is one of the most important algorithms in linear algebra – often described as the “Swiss Army knife” of matrix factorizations – because it has a wide range of applications from solving inverse problems to dimensionality reduction.


The Singular Value Decomposition of a matrix $$M$$ is a factorization of the form:

$$
M = U\, \Sigma\, V^T,
$$

where:

$$M$$ is an $$m \times n$$ real matrix (think of m samples and n features, or an image matrix, etc.).

$$U$$ is an $$m \times m$$ orthogonal matrix (its columns are orthonormal vectors $$u_1, u_2, \dots, u_m$$, called left singular vectors).

$$V$$ is an $$n \times n$$ orthogonal matrix (its columns are orthonormal vectors $$v_1, \dots, v_n$$, called right singular vectors).

$$\Sigma$$ is an $$m \times n$$ diagonal matrix (not necessarily square) with nonnegative values on the diagonal. These diagonal entries $$\sigma_1, \sigma_2, \dots$$ are called singular values, and by convention are ordered $$\sigma_1 \ge \sigma_2 \ge \dots \ge 0$$.

Only the first $$\min(m,n)$$ diagonal entries of $$\Sigma$$ are non-zero; the rest of $$\Sigma$$ (if $$m \ne n$$) are effectively padded with zeros. Intuitively, SVD says that any linear transformation $$M$$ can be viewed as a rotation (by $$V^T$$), then a scaling (by $$\Sigma$$), then another rotation (by $$U$$). This is analogous to eigen-decomposition but doesn’t require the matrix to be square or symmetric. In fact, if $$M$$ is symmetric and orthogonally diagonalizable, SVD and eigen-decomposition coincide (U and V will be the same aside from dimension).

Historical note: SVD was discovered in the 19th century by Eugenio Beltrami (1873) and Camille Jordan (1874) independentl (What Is the Singular Value Decomposition? – Nick Higham)】, though it wasn’t widely used until much later. It was popularized in numerical computing by Gene Golub and collaborators in the 1960s, who developed efficient algorithms (the Golub–Reinsch algorithm) to compute i (What Is the Singular Value Decomposition? – Nick Higham)】. With the advent of computers, SVD became a fundamental tool because it’s stable and reliable for solving linear systems and least squares, even when matrices are ill-conditioned or singular.

## Mathematical Intuition and Relation to Eigenvalues

SVD can be derived or understood via the eigen-decomposition of $$M^T M$$ (an $$n \times n$$ symmetric matrix) or $$M M^T$$ (an $$m \times m$$ symmetric matrix). Note that $$M^T M$$ is symmetric positive semidefinite, so it has an eigen-decomposition: 

$$
M^T M = V\,D\,V^T,
$$

where $$D$$ is diagonal with eigenvalues $$\mu_1, \mu_2, \dots, \mu_n \ge 0$$, and $$V$$’s columns are the eigenvectors (right singular vectors of $$M$$). It turns out the non-zero eigenvalues of $$M^T M$$ (and $$M M^T$$) are the squares of the singular values ($$\sigma_i^2$$). In fact:

- The singular values $$\sigma_i = \sqrt{\mu_i}$$
- The eigenvectors of $$M^T M$$ are the right singular vectors
- Similarly, eigenvectors of $$M M^T$$ (which has the same non-zero eigenvalues) are the left singular vectors (columns of $$U$$)

$$
M M^T = U D' U^T
$$

with the same non-zero eigenvalues $$\mu_i$$ filling $$D'$$ (plus additional zeros if $$m > n$$).

So the way to compute the SVD is: find eigenvalues and eigenvectors of $$M^T M$$. Suppose $$\mu_1 \ge \mu_2 \ge \dots \ge \mu_r > 0$$ are the non-zero eigenvalues ($$r = \text{rank of } M$$), with eigenvectors $$v_1, \dots, v_r$$. Set $$\sigma_i = \sqrt{\mu_i}$$. Let $$v_i$$ be the i-th column of $$V$$. Then define:

$$
u_i = \frac{1}{\sigma_i} M v_i
$$

One can show $$u_i$$ are unit vectors and are eigenvectors of $$M M^T$$. These form the first $$r$$ columns of $$U$$. The remaining columns of $$U$$ (if any) can be any orthonormal vectors completing the basis. Similarly, if $$n > r$$, the last columns of $$V$$ can be any orthonormal completion (or correspond to the zero eigenvalues). This construction yields:

$$
M = \sum_{i=1}^r \sigma_i u_i v_i^T,
$$

which is the SVD.

In summary, $$U$$’s columns are eigenvectors of $$M M^T$$, $$V$$’s columns are eigenvectors of $$M^T M$$, and $$\Sigma$$’s diagonal entries are the square roots of those eigenvalues.

Another intuition: think of $$M$$ as transforming an n-dimensional vector (input) into an m-dimensional vector (output). The SVD tells us there is an orthonormal basis of input vectors $$v_i$$ and an orthonormal basis of output vectors $$u_i$$ such that:

$$
M v_i = \sigma_i u_i
$$

For those familiar with eigenvectors, this looks almost like an eigen equation, except $$v_i$$ and $$u_i$$ live in different spaces if $$m \ne n$$. But:

$$
M^T M v_i = \sigma_i^2 v_i
$$

does hold (so $$v_i$$ is an eigenvector of $$M^T M$$ with eigenvalue $$\sigma_i^2$$). So SVD is an extension of eigen-decomposition to rectangular matrices.

### Geometric interpretation

As mentioned, any linear map can be seen as a rotation/reflection (by $$V^T$$), then axis-aligned stretching (by $$\Sigma$$), then another rotation/reflection (by $$U$$). If you imagine a unit sphere in $$\mathbb{R}^n$$, applying $$M$$ to it produces an m-dimensional ellipsoid. The principal semi-axes of that ellipsoid are $$u_1, u_2, \dots$$ (the left singular vectors), and their lengths are the singular values $$\sigma_1, \sigma_2, \dots$$. They tell us the directions in the input space that get mapped to those principal axes in the output. So singular values are basically the “strengths” of $$M$$ along those special input/output directions.

---

## SVD in Practice: Low-Rank Approximations and Computation

One of the most useful aspects of SVD is that it gives the best low-rank approximations of a matrix. If we have:

$$
M = U \Sigma V^T = \sum_{i=1}^r \sigma_i (u_i v_i^T)
$$

(where $$r = \text{rank of } M$$), we can approximate $$M$$ by truncating this sum to $$k < r$$ terms:

$$
M_k = \sum_{i=1}^k \sigma_i\, u_i\, v_i^T
$$

This $$M_k$$ is a rank-$$k$$ matrix (only $$k$$ singular values/vectors used). It turns out $$M_k$$ is the best approximation of $$M$$ among all rank-$$k$$ matrices in terms of least-squares error (minimum Frobenius norm error) – this is known as the **Eckart–Young theorem**.

The intuition is that since the singular values are in descending order, we are keeping the largest “components” of $$M$$ and discarding the rest. If singular values drop off quickly, then $$M$$ is well-approximated by a much smaller rank matrix.

---

For example, if you have a big matrix of data but most of its structure lies in a few dimensions (e.g., a topic-document matrix where only a few underlying topics span the data), the top singular vectors capture those and the rest may be noise. In PCA terms, using SVD on the data matrix directly can give similar results to PCA on the covariance.

Computing SVD: In code, computing SVD is easy with libraries. For instance, using NumPy:

```python
import numpy as np
A = np.random.rand(6, 4)  # a 6x4 matrix
U, s, Vt = np.linalg.svd(A, full_matrices=False)
```

This returns $$U$$ (6×4 in this case, since `full_matrices=False` gives the reduced form), the singular values in array $$s$$ (length 4), and $$V^T$$ (4×4). We can reconstruct $$A$$ via:

```python
U @ np.diag(s) @ Vt
```

and it will match the original (within numerical precision). If we only want the top $$k$$ singular values, we could take `U[:,:k]`, `s[:k]`, `Vt[:k,:]` to build the rank-$$k$$ approximation.

For very large matrices, one can use iterative methods or truncated SVD algorithms (like `scipy.sparse.linalg.svds` or scikit-learn’s `TruncatedSVD` which is useful for sparse matrices). In fact, PCA is often computed via SVD on the centered data matrix rather than eigen-decomposition of the covariance, because it’s more efficient when the number of features is large. Scikit-learn’s PCA, for example, uses an SVD under the hood. This is related to the fact that:

$$
X^T X v = \lambda v
$$

is equivalent to the SVD of $$X$$.

---

## Let’s illustrate SVD with a concrete example: image compression

We take a grayscale image (which is basically a matrix of pixel intensities) and apply SVD to it. By keeping only a few singular values, we can reconstruct an approximation of the image.

Consider the example image below. The first panel is the original image (512×512 pixels). The next panels show reconstructions using only the top 5 singular values, top 20, and top 50 (out of 512 total).

As we increase the number of singular values in the reconstruction, the image quality improves. With 50 singular values (~10% of the possible components for this 512×512 image), the image is already largely recognizable.

In code, the compression might look like this:

```python
import matplotlib.pyplot as plt
from skimage import data

# Load example image as a matrix
img = data.camera().astype(float)  # cameraman test image, shape (512,512)

# Compute SVD
U, s, Vt = np.linalg.svd(img, full_matrices=False)

# Choose k
k = 50
img_approx = U[:,:k] @ np.diag(s[:k]) @ Vt[:k,:]

plt.imshow(img_approx, cmap='gray')
```

One can experiment with different $$k$$. What SVD is doing here is essentially finding an optimal basis for the image’s pixel space such that you can truncate it.

---

The power of SVD goes beyond images

- **Natural language processing**: the technique called **Latent Semantic Analysis (LSA)** uses SVD on a term-document matrix to uncover “topics” (the singular vectors) and reduce noise in text data.

- **Recommender systems**: SVD (or similar matrix factorization methods) is used to decompose the user-item ratings matrix into a product of lower-dimensional matrices.

- **Solving least squares problems**: SVD provides a robust method to compute the pseudoinverse and solve $$Ax \approx b$$ even if $$A$$ is singular or ill-conditioned.

To connect SVD back to eigen-decomposition vs PCA: PCA can be obtained from SVD of the data matrix. If $$X$$ is $$m \times n$$ (m samples, n features), doing SVD:

$$
X = U \Sigma V^T
$$

Here $$U$$ is $$m \times m$$, $$V$$ is $$n \times n$$. The columns of $$V$$ (right singular vectors) are the eigenvectors of $$X^T X$$ (the covariance matrix up to scaling), and the singular values relate to eigenvalues:

$$
\sigma_i^2 = \lambda_i (m-1)
$$

if using sample covariance. In fact, the **principal components** = columns of $$V$$, and $$X V = U \Sigma$$ gives the principal component scores (projections) scaled by singular values.

Scikit-learn’s PCA uses an SVD internally for efficiency. The difference is mostly in whether you subtract the mean and whether you divide by $$\sqrt{m-1}$$ in the singular values.

---


## Applications of SVD in ML

We’ve already touched on a few:

### Dimensionality Reduction

SVD is at the heart of PCA (as explained) and more generally is used to reduce dimensionality of very large, sparse datasets (e.g., text). Sklearn’s TruncatedSVD is often used as a PCA alternative for sparse data (it doesn’t center the data, which can be fine for text frequency data). The advantage of SVD is it can work directly on the data without needing to form a covariance matrix, which is memory-intensive for large feature sets. For example, with a term-document matrix of size 100k documents, you wouldn’t explicitly form a $$100k \times 100k$$ covariance; you’d do truncated SVD to get, say, 100 topics.

### Latent Semantic Analysis (LSA) in NLP

As mentioned, by doing SVD on a term-document matrix (rows = documents, columns = terms, values = say TF-IDF scores), you get: $$U$$ = document vectors in topic-space, $$V$$ = term vectors in topic-space, $$\Sigma$$ = strengths of each “topic”. This uncovers latent concepts. After truncating, each document or term is represented in a reduced-dimensional semantic space. Queries and docs can be compared in this space to improve search (even if exact words don’t match, related concepts can be matched via the latent factors).

### Recommendation Systems (Collaborative Filtering)

If you have a user-item rating matrix, SVD gives you matrices that represent users in a latent factor space and items in the same space. A rating is approximated by the dot product of user and item factor vectors. This reveals, for example, that a movie’s factor vector might encode how much it is comedy vs drama vs action, and a user’s factor vector encodes their preference for those genres. In practice, direct SVD on a sparse rating matrix may require filling missing entries, so often specialized alternating least squares are used, but conceptually it’s similar. The Netflix Prize solution was an ensemble of such factorization models.

### Image Processing

Beyond compression, SVD (and PCA, which for images are sometimes called “eigenfaces” in face recognition) can decompose image datasets. For denoising images, one can keep only the largest singular values which tend to capture the main structures and discard smaller ones that often represent noise. SVD is also used in algorithms like image alignment and structure-from-motion in computer vision because it can find optimal linear alignments.

### Matrix Solvers and Model Compression

SVD is used to compute the Moore-Penrose pseudoinverse for solving linear systems. If a model has a large weight matrix (for instance, a fully connected layer in a neural network), one can compress it by approximating that weight matrix with a low-rank decomposition using SVD. Essentially, you replace a big weight matrix by two thinner matrices (which correspond to keeping only top singular vectors), which reduces the number of parameters and can speed up inference. In deep learning, after training a model, one can do SVD on weight layers to see if they have low effective rank and truncate them (with some fine-tuning to recover accuracy). SVD has also been used inside training algorithms for RNNs to enforce a well-conditioned weight matrix.

---

## Comparison Between PCA and SVD

It’s worth clarifying the relationship and differences between PCA and SVD, as they often get mentioned together:

### Relationship

PCA is essentially a specific application of SVD. If you take your data matrix $$X$$ (with zero-mean for each feature) and perform SVD $$X = U \Sigma V^T$$, the right singular vectors $$V$$ are the principal component directions, and the singular values $$\sigma_i$$ are related to the standard deviation of data along those components. In fact, 

$$
\lambda_i = \frac{\sigma_i^2}{m-1}
$$

would be the eigenvalues of the covariance. Thus, PCA results can be obtained by SVD. Conversely, if you have the covariance matrix 

$$
S = V \Lambda V^T
$$

and do eigen-decomposition, you could reconstruct an SVD of $$X$$ by setting 

$$
U = X V \Lambda^{-1/2}
$$

(for nonzero eigenvalues). So mathematically, PCA and SVD are tightly linked—PCA often just means applying SVD to a mean-centered data matrix and interpreting the results in terms of variance.

### Differences in Focus

PCA is defined as a statistical procedure: it focuses on the covariance structure of data and identifies directions of maximal variance (making the data in the new coordinates uncorrelated). SVD is purely a matrix factorization that doesn’t inherently carry a statistical interpretation unless you connect it to something like $$X^T X$$.

SVD can be applied to matrices that are not covariance matrices (e.g., the term-document matrix, or a rectangular transformation matrix). So one difference: PCA typically involves first normalizing or standardizing data (and always centering), and it implicitly assumes we care about variance. SVD will happily factorize any matrix as is. For PCA, you must decide how to handle scaling of features (since covariance is sensitive to scale); SVD doesn’t care about that but if you gave it an unnormalized data matrix, the singular vectors would be dominated by whatever features have larger numeric scales.

In practice, one usually scales features for PCA (e.g., using correlation matrix instead of covariance if units differ).

### Computational Differences

PCA (eigen-decomposition) computes eigenvectors of an $$n \times n$$ covariance matrix (if $$n$$ = number of features). This can be expensive when $$n$$ is large (e.g., $$n = 10000$$ would mean a $$10000 \times 10000$$ matrix to diagonalize). SVD can work directly on the $$m \times n$$ data matrix, which might be more feasible if $$n$$ is large but $$m$$ is smaller, or if the matrix is sparse.

For example, if you have $$10^6$$ samples and $$10^5$$ features (very tall matrix), eigen-decomposition of the covariance ($$10^5 \times 10^5$$) is huge, but SVD on $$10^6 \times 10^5$$ might be manageable with iterative methods.

Similarly, SVD can handle missing data by algorithms that operate only on observed entries, whereas PCA (covariance computation) can’t directly handle missing values.

### Use-Cases

PCA is typically used for exploratory data analysis, feature reduction, visualization, and sometimes preprocessing to decorrelate features. SVD has broader applications: beyond data variance, it’s used in matrix completion, inverse problems, etc.

For example, if you have a linear system $$Ax = b$$, SVD helps solve it (via pseudoinverse) especially if $$A$$ is not full rank. PCA wouldn’t be referenced in that context.

So PCA is a subset of what SVD can do, focusing on variance in a dataset.

### Output Interpretation

In PCA, we talk about “principal components”, “explained variance”, and we often care about how many components to choose, interpret the components, etc. In SVD of an arbitrary matrix, we talk about “singular vectors” and “singular values”, and their magnitudes, but not usually “variance explained” unless it’s specifically data matrix.

In PCA we often drop components with small eigenvalues because they are mostly noise. In SVD, dropping small singular values gives a low-rank approximation which might be for noise reduction or compression. These are analogous ideas (and indeed the same operation), but PCA frames it as data compression, SVD frames it as matrix compression.

---

### Summary: When to Use PCA vs SVD

- They will give the same result if applied consistently (PCA on covariance vs SVD on centered data).
- If your data is very high-dimensional (many features), use SVD to compute PCA.
- If your data matrix is sparse (like text data or recommender systems), use truncated SVD directly.
- If the goal is interpretability in terms of original features and variance, PCA terminology is used.
- If the goal is matrix factorization or solving a linear algebra problem, SVD terminology is used.
- PCA uses SVD under the hood.

As a fun fact, if you perform PCA on un-centered data (i.e., don’t subtract mean), that’s equivalent to doing an SVD on the raw data matrix (with an extra singular vector corresponding to the mean direction usually appearing). Generally, we center data for PCA.

---

Not to brag, but we just wrangled some serious math today. We went from stretching vectors to compressing images — not bad for one post!

We started by understanding what makes eigenvectors special: they’re the directions that remain unchanged (except for scaling) under a transformation. We saw how eigenvalues and eigenvectors reveal the internal structure of matrices and how they’re deeply tied to variance, stability, and interpretability in machine learning.

We then looked at eigen-decomposition and diagonalization, and how these concepts simplify complex operations, especially when dealing with symmetric matrices like covariance matrices. This naturally led us to Principal Component Analysis (PCA), where eigenvectors define the principal directions of variance, and eigenvalues tell us how important each direction is.

Then came Singular Value Decomposition (SVD) — a more general, powerful factorization that works for any matrix, not just square or symmetric ones. We explored how SVD connects back to eigenvalues, how it gives us the best low-rank approximation of data, and how it’s used in practice for tasks like image compression, noise reduction, LSA in NLP, recommender systems, and model compression.

We also clarified how PCA and SVD are related: PCA can be computed via SVD, but SVD has broader applications beyond just analyzing variance.

In the end, these tools — eigenvalues, eigenvectors, PCA, and SVD — give us more than math. They give us a way to see the essence of data, simplify it, compress it, and make it more interpretable. And in a field as noisy and high-dimensional as machine learning, that’s incredibly powerful.