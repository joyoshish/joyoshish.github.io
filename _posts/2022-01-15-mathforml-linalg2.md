---
layout: post
title: Matrices and Matrix Operations
date: 2022-01-15
description: Linear Algebra 2 - Mathematics for Data Science
tags: ml ai linear-algebra math
math: true
categories: machine-learning math data-science-math
thumbnail: assets/img/linalg_banner.png
giscus_comments: true
related_posts: true
chart:
  plotly: true
toc:
  beginning: true
---

In machine learning, whether you're manipulating datasets, performing linear transformations in neural networks, or analyzing graph structures, matrices provide a compact and efficient way to represent and process information. In this post, we’ll explore the world of matrices and matrix operations through a problem-driven approach. Each section grounds the math in real ML use cases, walks through the concepts clearly, and wraps up with code and applications.

---

## Matrices and Matrix Operations

Think of a matrix as a grid—a two-dimensional array of numbers. On the surface, it may just look like a neat way to organize data, but in machine learning, it does so much more. Matrices are the building blocks for operations like transforming input features, propagating signals through neural networks, and encoding relationships in graph data.

Suppose you're representing three data points, each with two features. You might organize this as a matrix:

$$
X = \begin{bmatrix} 1.2 & 3.4 \\ 2.1 & 0.7 \\ 4.0 & 1.5 \end{bmatrix}
$$

Each row is a sample, and each column is a feature. This basic structure appears everywhere from data preprocessing to forward passes in deep learning.

---

## Matrix Addition and Multiplication

Let’s start with two toy matrices:

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
$$

**Matrix Addition**:

Two matrices can be added if they have the same dimensions. Element-wise addition gives:

$$
A + B = \begin{bmatrix} a_{11}+b_{11} & a_{12}+b_{12} \\ a_{21}+b_{21} & a_{22}+b_{22} \end{bmatrix} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}
$$

**Properties**:
- Commutative: $$A + B = B + A$$
- Associative: $$(A + B) + C = A + (B + C)$$
- Additive identity: $$A + 0 = A$$
- Additive inverse: $$A + (-A) = 0$$

**Matrix Multiplication**:

Matrix multiplication is defined when the number of columns in $$A$$ matches the number of rows in $$B$$:

$$
(AB)_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
$$

For our example:

$$
AB = \begin{bmatrix} (1\cdot5 + 2\cdot7) & (1\cdot6 + 2\cdot8) \\ (3\cdot5 + 4\cdot7) & (3\cdot6 + 4\cdot8) \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
$$

**Properties**:
- Associative: $$(AB)C = A(BC)$$
- Distributive: $$A(B + C) = AB + AC$$
- Not necessarily commutative: $$AB \neq BA$$ in general

**Data Science Applications**:
- **Linear transformations** in neural networks: $$Z = WX + b$$
- **Embeddings** in NLP: word matrices transformed into contextual vectors
- **Recommender Systems**: low-rank approximations: $$\hat{R} = U \Sigma V^T$$

---

## Transpose, Inverse, Determinant, Trace, and Rank

Let’s dive deeper into the matrix:

$$
A = \begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix}
$$

### Transpose

The transpose of $$A$$ flips its rows and columns:

$$
A^T = \begin{bmatrix} a_{11} & a_{21} \\ a_{12} & a_{22} \end{bmatrix} = \begin{bmatrix} 4 & 2 \\ 7 & 6 \end{bmatrix}
$$

**Properties**:
- $$(A^T)^T = A$$
- $$(A + B)^T = A^T + B^T$$
- $$(AB)^T = B^T A^T$$


### Cofactor Matrix

The **cofactor matrix** of a square matrix is formed by replacing each element with its cofactor. The cofactor of an element $$a_{ij}$$ is calculated using:

$$
C_{ij} = (-1)^{i+j} \cdot M_{ij}
$$

where:
- $$M_{ij}$$ is the **minor** of $$a_{ij}$$, i.e., the determinant of the submatrix obtained by removing the $$i$$-th row and $$j$$-th column from the original matrix.
- $$(-1)^{i+j}$$ gives the correct sign based on the position of the element (row $$i$$, column $$j$$).

#### Steps to Find the Cofactor Matrix

1. For each element $$a_{ij}$$ in the matrix:
   - Remove the $$i$$-th row and $$j$$-th column to form a smaller submatrix.
   - Calculate the determinant of this submatrix to get the minor $$M_{ij}$$.
   - Multiply the minor by $$(-1)^{i+j}$$ to get the cofactor $$C_{ij}$$.
2. The cofactor matrix is the matrix where each entry is the corresponding cofactor $$C_{ij}$$.

#### Example

Given a $$3 \times 3$$ matrix:
$$
A = \begin{bmatrix}
1 & 0 & -2 \\
3 & -1 & 2 \\
4 & 5 & 6
\end{bmatrix}
$$

To find the cofactor of the element $$0$$ (in the first row, second column):
- Remove the first row and second column:
  $$
  \begin{bmatrix}
  3 & 2 \\
  4 & 6
  \end{bmatrix}
  $$
- Compute the minor:
  $$
  M_{12} = (3 \times 6) - (4 \times 2) = 18 - 8 = 10
  $$
- Apply the sign:
  $$
  C_{12} = (-1)^{1+2} \times 10 = -10
  $$

Repeat this process for every element to construct the full cofactor matrix.



### Determinant

The determinant of a 2x2 matrix $$A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$$ is:

$$
\det(A) = ad - bc
$$

The general formula (Laplace expansion) for an $$n \times n$$ matrix is:

$$
\det(A) = \sum_{j=1}^{n} (-1)^{1+j} a_{1j} M_{1j}
$$

Where $$M_{1j}$$ is the determinant of the $$(n-1) \times (n-1)$$ minor matrix obtained by removing the first row and $$j$$-th column.

**Properties**:
- $$\det(AB) = \det(A) \cdot \det(B)$$
- $$\det(A^T) = \det(A)$$
- $$\det(cA) = c^n \cdot \det(A)$$ for scalar $$c$$
- $$\det(I) = 1$$

---

### Inverse of a Matrix

The inverse of a matrix is the key to "undoing" a linear transformation. For a square matrix $$A$$, its inverse $$A^{-1}$$ satisfies:

$$
A A^{-1} = A^{-1} A = I
$$

However, not every matrix has an inverse. Only **non-singular square matrices**—those with $$\det(A) \ne 0$$—can be inverted.

---

#### Inverse of a 2×2 Matrix

Let’s begin with the simplest case.

Let  
$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$

##### Step 1: Check Invertibility

Compute the determinant:

$$
\det(A) = ad - bc
$$

- If $$\det(A) = 0$$ → matrix is **singular** → **no inverse exists**.
- If $$\det(A) \ne 0$$ → proceed to next step.

##### Step 2: Use the Inverse Formula

$$
A^{-1} = \frac{1}{\det(A)} 
\begin{bmatrix} 
d & -b \\ 
-c & a 
\end{bmatrix}
$$

##### Example

Given  
$$
A = \begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix}
$$

Calculate the determinant:

$$
\det(A) = 4 \cdot 6 - 7 \cdot 2 = 24 - 14 = 10
$$

Hence, the inverse is:

$$
A^{-1} = \frac{1}{10} 
\begin{bmatrix} 
6 & -7 \\ 
-2 & 4 
\end{bmatrix}
$$

---

#### Inverse of a 3×3 Matrix

The procedure becomes more involved but follows the same principles.

Let $$A$$ be a 3×3 matrix.

##### Step 1: Compute Determinant

Use cofactor (Laplace) expansion, typically along the first row:

$$
\begin{aligned}
\det(A) &= a_{11} 
\begin{vmatrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{vmatrix} 
- a_{12} 
\begin{vmatrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{vmatrix} 
+ a_{13} 
\begin{vmatrix} a_{21} & a_{22} \\ a_{31} & a_{32} \end{vmatrix}
\end{aligned}
$$

##### Step 2: Compute Cofactor Matrix

Each entry:

$$
C_{ij} = (-1)^{i+j} \cdot \det(\text{minor of } a_{ij})
$$

##### Step 3: Compute Adjugate Matrix

Transpose the cofactor matrix:

$$
\mathrm{adj}(A) = C^\top
$$

##### Step 4: Divide by Determinant

The inverse is:

$$
A^{-1} = \frac{1}{\det(A)} \cdot \mathrm{adj}(A)
$$

---

#### Complete Example (3×3)

Let  
$$
A = \begin{bmatrix} 
7 & 2 & 1 \\ 
0 & 3 & -1 \\ 
-3 & 4 & -2 
\end{bmatrix}
$$

##### 1. Determinant


$$
\begin{aligned}
\det(A) 
&= 7 \cdot 
\begin{vmatrix} 
3 & -1 \\ 
4 & -2 
\end{vmatrix} 
\,-\, 2 \cdot 
\begin{vmatrix} 
0 & -1 \\ 
-3 & -2 
\end{vmatrix} 
\,+\, 1 \cdot 
\begin{vmatrix} 
0 & 3 \\ 
-3 & 4 
\end{vmatrix} \\
&= 7(-6 + 4) \,-\, 2(0 - (-3)) \,+\, 1(0 + 9) \\
&= 7(-2) \,-\, 2(-3) \,+\, 9 \\
&= -14 + 6 + 9 \\
&= 1
\end{aligned}
$$


Hence, invertible.

##### 2. Cofactor Matrix

$$
C = \begin{bmatrix} 
-2 & -3 & 9 \\ 
8 & -11 & -34 \\ 
-5 & 7 & 21 
\end{bmatrix}
$$

##### 3. Adjugate

$$
\mathrm{adj}(A) = C^\top = \begin{bmatrix} 
-2 & 8 & -5 \\ 
-3 & -11 & 7 \\ 
9 & -34 & 21 
\end{bmatrix}
$$

##### 4. Final Inverse

$$
A^{-1} = \frac{1}{1} 
\begin{bmatrix} 
-2 & 8 & -5 \\ 
-3 & -11 & 7 \\ 
9 & -34 & 21 
\end{bmatrix}
$$

---

#### Sanity Checks and Verification

- **Determinant Test**:  
  $$\det(A) = 0$$ → matrix is not invertible.

- **Multiplication Test**:  
  $$A \cdot A^{-1} = I$$ must hold. Otherwise, recheck calculation.

- **Inverse Property Tests**:
  - $$(A^{-1})^{-1} = A$$
  - $$(A^T)^{-1} = (A^{-1})^T$$
  - $$\det(A^{-1}) = \frac{1}{\det(A)}$$

- **Special Cases**:  
  For **triangular matrices**, $$A^{-1}$$ will also be triangular. Diagonal entries of the inverse are reciprocals of the original diagonal entries (if all are non-zero).

---

#### Summary of Properties

- A matrix is invertible **iff** $$\det(A) \ne 0$$
- $$(AB)^{-1} = B^{-1} A^{-1}$$
- $$(A^T)^{-1} = (A^{-1})^T$$
- $$(A^{-1})^{-1} = A$$


---


### Pseudo-Inverse of a Matrix

Not all matrices are square or invertible—but in machine learning, we still need to “solve” equations like $$Ax = b$$, even when $$A$$ is rectangular or singular.

This is where the **Moore–Penrose pseudo-inverse** comes in.

Given a matrix $$A \in \mathbb{R}^{m \times n}$$, its **pseudo-inverse**, denoted $$A^+$$, is a matrix such that:

$$
A^+ = \arg\min_{X} \|AX - I\|_F
$$

It satisfies four key properties:

1. $$A A^+ A = A$$  
2. $$A^+ A A^+ = A^+$$  
3. $$(A A^+)^T = A A^+$$  
4. $$(A^+ A)^T = A^+ A$$

---

#### How to Compute It

There are multiple ways to compute $$A^+$$, but the most robust is via **Singular Value Decomposition (SVD)**:

Let  
$$
A = U \Sigma V^T
$$  
Then,  
$$
A^+ = V \Sigma^+ U^T
$$

Where:
- $$\Sigma^+$$ is formed by taking the reciprocal of the non-zero singular values in $$\Sigma$$ and transposing the result.


We'll revisit **SVD (Singular Value Decomposition)** in more depth later when we explore **eigenvalues, eigenvectors**, and decompositions like **PCA** and **SVD-based dimensionality reduction**. For now, just note that it's a powerful tool that lets us compute pseudo-inverses even for non-square or rank-deficient matrices.


---

#### Example: Rectangular Matrix

Suppose:

$$
A = \begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6
\end{bmatrix}
$$

We want to solve $$Ax = b$$ in the least squares sense.

We compute:

$$
A^+ = (A^T A)^{-1} A^T
$$

This gives a matrix that maps a target vector $$b$$ to the best-fitting $$x$$ minimizing:

$$
\|Ax - b\|^2
$$

---

#### Application in Machine Learning

The pseudo-inverse appears frequently when:
- Solving **normal equations** in linear regression:
  $$
  \hat{\beta} = (X^T X)^{-1} X^T y = X^+ y
  $$
- Working with **non-square matrices** in overdetermined or underdetermined systems
- Implementing **closed-form least squares solutions**
- Dealing with **non-invertible covariance matrices** in statistics
- Backpropagation in networks where you need a generalized inverse

---

If the inverse is a rare perfect square peg, the pseudo-inverse is the flexible wrench—it works when nothing else does. It allows us to perform meaningful matrix "division" even in messy, high-dimensional settings common in ML and data science.



---



### Trace

The trace of a square matrix is the sum of its diagonal elements:

$$
\text{tr}(A) = \sum_{i=1}^{n} A_{ii}
$$

**Properties**:
- $$\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$$
- $$\text{tr}(cA) = c \cdot \text{tr}(A)$$
- $$\text{tr}(AB) = \text{tr}(BA)$$
- $$\text{tr}(A^T) = \text{tr}(A)$$

### Rank

The rank is the number of linearly independent rows (or columns). This can be found using row-reduction or SVD.

**Data Science Applications**:
- **Regression**: Normal equations: $$\hat{\beta} = (X^TX)^{-1}X^Ty$$
- **PCA**: Eigen-decomposition of covariance matrices relies on trace and rank
- **Multicollinearity**: Detected via rank deficiency
- **Normalizing Flows**: Require computing $$\log |\det(Jacobian)|$$

---


## Special Matrices in ML

### Identity Matrix

$$
I_n = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}
$$

**Properties**:
- $$AI = IA = A$$ for any matrix $$A$$ of compatible size
- $$I^T = I$$
- $$I^{-1} = I$$

### Diagonal Matrix

$$
D = \text{diag}(\lambda_1, \lambda_2, ..., \lambda_n) = \begin{bmatrix} \lambda_1 & 0 & \cdots & 0 \\ 0 & \lambda_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_n \end{bmatrix}
$$

**Properties**:
- $$D^T = D$$ (symmetric)
- $$D^{-1} = \text{diag}(1/\lambda_1, ..., 1/\lambda_n)$$ if all $$\lambda_i \neq 0$$
- $$\det(D) = \prod_{i=1}^n \lambda_i$$
- $$\text{tr}(D) = \sum_{i=1}^n \lambda_i$$

### Symmetric Matrix

$$
A = A^T
$$

**Properties**:
- All eigenvalues of a real symmetric matrix are real
- Can be orthogonally diagonalized: $$A = Q \Lambda Q^T$$ where $$Q$$ is orthogonal and $$\Lambda$$ is diagonal

### Orthogonal Matrix

$$
Q^T Q = QQ^T = I
$$

**Properties**:
- $$Q^{-1} = Q^T$$
- $$\det(Q) = \pm 1$$
- Preserves vector norms: $$\|Qx\| = \|x\|$$
- Preserves inner products: $$\langle Qx, Qy \rangle = \langle x, y \rangle$$


---


### Positive Definite and Semi-Definite Matrices

Positive definite and positive semi-definite matrices often hide behind the scenes in ML models, but they're doing critical work—especially in optimization, regularization, and probabilistic modeling.

#### Definitions

Let $$A \in \mathbb{R}^{n \times n}$$ be a **symmetric** matrix.

- **Positive Definite (PD)**:
  $$
  \forall x \in \mathbb{R}^n, \; x \ne 0 \Rightarrow x^T A x > 0
  $$
- **Positive Semi-Definite (PSD)**:
  $$
  \forall x \in \mathbb{R}^n, \; x \ne 0 \Rightarrow x^T A x \ge 0
  $$

That is, a matrix is PD if it always stretches any nonzero vector in such a way that the result is strictly aligned (positive projection) with the vector. If it *could* flatten the vector (project to zero), it’s PSD.


---


When we say a matrix is **positive definite (PD)**, we mean the following:

> For **any non-zero vector** $$x$$, if you transform it using the matrix $$A$$ (by computing $$Ax$$), and then measure how aligned the result is with the original $$x$$—the alignment is **always positive**.

Mathematically, this alignment is given by:

$$
x^T A x
$$

If this value is **strictly positive** for all $$x \ne 0$$, then $$A$$ is **positive definite**.

---

Now, if that value is allowed to be **zero** (but never negative), the matrix is called **positive semi-definite (PSD)**.

So you can think of it like this:

- **Positive Definite (PD)** → always pushes every vector in a direction that's at least somewhat aligned with itself (never flat, never reversed)
- **Positive Semi-Definite (PSD)** → *may* flatten some vectors completely (i.e., project them to 0), but still never bends anything backwards

---

#### Visualize


We want to understand **what it means** for a matrix to be:
- **Positive Definite (PD)**: $$x^T A x > 0$$ for *every* non-zero vector $$x$$
- **Positive Semi-Definite (PSD)**: $$x^T A x \ge 0$$ for all $$x$$, but it **might be zero** for some

But those formulas can feel abstract. So here’s the core idea:

> Pick a bunch of directions (vectors $$x$$), plug them into $$x^T A x$$, and ask:  
> “How much is this matrix ‘amplifying’ or ‘flattening’ that direction?”

---

{% raw %}
<div id="pd-vs-psd" style="width:100%;max-width:700px;margin:auto;"></div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
const theta = Array.from({length: 200}, (_, i) => 2 * Math.PI * i / 199);
const x = theta.map(t => Math.cos(t));
const y = theta.map(t => Math.sin(t));

// Quadratic form for PD matrix A = [[2, 0], [0, 1]]
const qf_pd = x.map((xi, i) => 2 * xi**2 + y[i]**2);
const x_pd = x.map((xi, i) => xi * qf_pd[i]);
const y_pd = y.map((yi, i) => yi * qf_pd[i]);

// Quadratic form for PSD matrix A = [[1, -1], [-1, 1]]
const qf_psd = x.map((xi, i) => (xi - y[i])**2);
const x_psd = x.map((xi, i) => xi * qf_psd[i]);
const y_psd = y.map((yi, i) => yi * qf_psd[i]);

const data = [
  {
    x: x_pd,
    y: y_pd,
    mode: 'markers',
    type: 'scatter',
    marker: {size: 5, color: qf_pd, colorscale: 'Viridis', colorbar: {title: 'xᵀAx'}},
    name: 'PD: xᵀAx',
    visible: true
  },
  {
    x: x,
    y: y,
    mode: 'lines',
    type: 'scatter',
    line: {dash: 'dash', color: 'gray'},
    name: 'Unit Circle',
    visible: true
  },
  {
    x: x_psd,
    y: y_psd,
    mode: 'markers',
    type: 'scatter',
    marker: {size: 5, color: qf_psd, colorscale: 'Plasma', colorbar: {title: 'xᵀAx'}},
    name: 'PSD: xᵀAx',
    visible: false
  },
  {
    x: x,
    y: y,
    mode: 'lines',
    type: 'scatter',
    line: {dash: 'dash', color: 'gray'},
    name: 'Unit Circle (PSD)',
    visible: false
  }
];

const layout = {
  title: 'Positive Definite vs. Semi-Definite Matrices',
  xaxis: {title: 'x', scaleanchor: 'y'},
  yaxis: {title: 'y'},
  showlegend: false,
  updatemenus: [{
    buttons: [
      {
        label: 'Positive Definite',
        method: 'update',
        args: [
          {visible: [true, true, false, false]},
          {title: 'Positive Definite Matrix — All xᵀAx > 0'}
        ]
      },
      {
        label: 'Positive Semi-Definite',
        method: 'update',
        args: [
          {visible: [false, false, true, true]},
          {title: 'Positive Semi-Definite Matrix — Some xᵀAx = 0'}
        ]
      }
    ],
    direction: 'down',
    showactive: true,
    x: 0.5,
    xanchor: 'center',
    y: 1.15,
    yanchor: 'top'
  }]
};

Plotly.newPlot('pd-vs-psd', data, layout);
</script>
{% endraw %}

---

##### **What We Plotted**

We sampled 200 points that lie evenly around the **unit circle**—meaning we’re probing every possible direction in 2D with a vector of length 1.

For each direction vector $$x$$, we compute:

$$
x^T A x
$$

This gives us a number telling us how the matrix $$A$$ stretches or flattens that direction.

Then, we scale each point in that direction by its computed value. So in the plot, a point **farther from the origin** means $$x^T A x$$ was larger—it shows stronger stretching. If it’s **close to the origin**, the matrix barely affected that direction.

The color of each dot also encodes the magnitude of $$x^T A x$$.

---

##### **Plot: Positive Definite Matrix**

Here, the matrix is:

$$
A = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}
$$

This matrix stretches along the $$x$$-axis more than the $$y$$-axis, but it still positively "responds" to **every** direction.

**What you see**:
- All the dots stay away from the origin—none collapse
- The pattern is elliptical, showing stronger stretching in some directions (especially along the x-axis)
- Every direction yields a strictly positive $$x^T A x$$

**Why this matters**:  
A positive definite matrix **always increases** the value of $$x^T A x$$ for any non-zero $$x$$. You never lose information, never flatten. That's why the shape in the plot is smooth and full—no gaps, no collapses.

---

##### **Plot: Positive Semi-Definite Matrix**

This time, the matrix is:

$$
A = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
$$

It’s symmetric, but **not invertible**, and has **rank 1**. That means it only responds to vectors lying in a particular direction—and ignores others completely.

**What you see**:
- Most points are stretched away from the origin, just like before
- But along specific directions (e.g., the diagonal $$[1, 1]$$), the dots shrink back to the center—they completely vanish

**Interpretation**:  
In those collapsed directions, $$x^T A x = 0$$. The matrix doesn’t "see" those vectors at all—it acts as if they don’t exist. That’s the essence of **positive semi-definiteness**: some directions may be ignored (zero energy), but none are flipped or made negative.




---

#### Properties

- All **eigenvalues** of a PD matrix are **strictly positive**
- All eigenvalues of a PSD matrix are **non-negative**
- A **symmetric matrix** is:
  - PD if and only if all eigenvalues > 0  
  - PSD if and only if all eigenvalues ≥ 0
- Covariance matrices are always **PSD**, and **PD** if the data has full rank
- Every PD matrix is **invertible**, but not every PSD matrix is

---

#### Why It Matters in ML

- **Covariance matrices** (e.g., in Gaussian models, PCA, multivariate normal) must be PSD
- **Kernel matrices** in SVMs and Gaussian processes must be PSD (they define valid inner products)
- In **convex optimization**, if your loss function’s Hessian is PD, the function is strictly convex—and you’re guaranteed a unique minimum
- **Ridge regression** uses:
  $$
  (X^T X + \lambda I)^{-1}
  $$
  This term is guaranteed to be PD for $$\lambda > 0$$, even if $$X^T X$$ is only PSD
- In **PCA**, we diagonalize the covariance matrix—its PSD nature ensures real, non-negative eigenvalues for interpretable principal components

---

#### Quick Test (for symmetric matrices)

Let $$A$$ be symmetric. Then:

- If all the **leading principal minors** (i.e., the determinants of the top-left $$k \times k$$ submatrices for $$k = 1, 2, ..., n$$) of a symmetric matrix $$A$$ are positive, then $$A$$ is **positive definite**. This is known as **Sylvester’s Criterion**, and it provides a quick way to test positive definiteness without computing all eigenvalues.
- If all eigenvalues of $$A$$ are ≥ 0 → $$A$$ is PSD  
- Try computing $$x^T A x$$ for random vectors $$x$$—if it's always positive, the matrix is likely PD


---

## Block and Partitioned Matrices

Sometimes, a matrix is too big or too complex to work with all at once. Instead of treating it as one big chunk, we break it down into smaller, more manageable pieces—called **blocks**.

Think of it like splitting a large spreadsheet into smaller tables that still fit together. This helps with both understanding and computation.

---

### A Simple Example

Let’s take a $$4 \times 4$$ matrix:

$$
A = \begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12 \\
13 & 14 & 15 & 16
\end{bmatrix}
$$

We can divide this into four $$2 \times 2$$ blocks like this:

$$
A = \begin{bmatrix}
A_{11} & A_{12} \\
A_{21} & A_{22}
\end{bmatrix}
$$

Where:

- $$A_{11} = \begin{bmatrix} 1 & 2 \\ 5 & 6 \end{bmatrix}$$  
- $$A_{12} = \begin{bmatrix} 3 & 4 \\ 7 & 8 \end{bmatrix}$$  
- $$A_{21} = \begin{bmatrix} 9 & 10 \\ 13 & 14 \end{bmatrix}$$  
- $$A_{22} = \begin{bmatrix} 11 & 12 \\ 15 & 16 \end{bmatrix}$$

Each block is just a smaller matrix, but together they still represent the original structure.

---

### Why Use Blocks?

Block matrices are useful when:

- You're dealing with **structured data** (e.g., images, time-series, graphs)
- You want to **optimize computations** using smaller chunks
- Your matrix represents multiple **modalities or features** (e.g., text + image)
- You're distributing operations across **multiple processors or nodes**

---

### Block Matrix Multiplication

Let’s say you have two block-partitioned matrices:

$$
A = \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}, \quad
B = \begin{bmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \end{bmatrix}
$$

If the inner dimensions match (i.e., each block multiplication is defined), you can compute the product blockwise:

$$
AB = \begin{bmatrix}
A_{11} B_{11} + A_{12} B_{21} & A_{11} B_{12} + A_{12} B_{22} \\
A_{21} B_{11} + A_{22} B_{21} & A_{21} B_{12} + A_{22} B_{22}
\end{bmatrix}
$$

So instead of multiplying the full matrices all at once, you're multiplying and adding smaller blocks—this is especially useful in large-scale computing.

---

### Applications

Block matrices show up in more places than you might expect:

- **Distributed computing**: In big-data systems like Spark or Dask, matrices are often partitioned into blocks for parallel processing. Each node handles a piece.
  
- **Mini-batch learning**: In neural networks, training often happens on batches. Each mini-batch can be viewed as a block of the full data matrix.
  
- **Multi-modal learning**: Say you're combining image features and text features—each type can be represented as a block in a larger feature matrix.
  
- **Graph learning**: Adjacency matrices of graphs can be block-partitioned to reflect communities or clusters. This is helpful in spectral clustering or modularity detection.




Block matrices are one of those “simple once you see it” ideas. They don’t just make computation faster—they also mirror the real-world structure in many ML tasks. If your data or model has parts, blocks are a natural way to think and compute.

---

## Final Thoughts

We started with a humble matrix and ended up unlocking a toolkit that powers nearly every machine learning algorithm today. Whether you're transforming features, training deep nets, or decomposing graphs—matrices are behind the scenes, doing the heavy lifting.

