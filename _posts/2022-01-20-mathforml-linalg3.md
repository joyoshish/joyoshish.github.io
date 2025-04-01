---
layout: post
title: Linear Algebra Basics for ML - Systems of Linear Equations
date: 2022-01-20
description: Linear Algebra 3 - Mathematics for Machine Learning
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

---

Many machine learning tasks boil down to optimizing model parameters that best fit the observed data. For instance, finding the best-fit line in linear regression is equivalent to solving a system of linear equations. In this post, we’ll dive into the mathematics behind solving these systems using multiple methods. We’ll explore row reduction, Gaussian elimination, and Cramer’s Rule—each backed by real-world ML context, intuitive math, and working code.

---

## Systems of Linear Equations in Machine Learning

In many machine learning problems—such as multiple linear regression, parameter estimation in models, and even network analysis—we often need to solve systems of equations that look like:

$$
A \mathbf{x} = \mathbf{b}
$$

where $$A$$ is an $$m \times n$$ matrix of coefficients, $$\mathbf{x}$$ is the vector of unknowns (model parameters), and $$\mathbf{b}$$ is the result or outcome vector. The goal is to determine the values of $$\mathbf{x}$$ that satisfy this equation—an essential step in fitting many models.

---

## Solving Systems Using Row Reduction

Let’s start with something familiar: multiple linear regression. When you derive the solution analytically, you arrive at the normal equations:

$$
X^T X \mathbf{\beta} = X^T \mathbf{y}
$$

Here, $$X$$ is the feature matrix, $$\mathbf{y}$$ is the target vector, and $$\mathbf{\beta}$$ is the vector of regression coefficients you're trying to find. This is a classic system of linear equations that we can solve using **row reduction**.

To do that, we first write the augmented matrix $$ [ A \mid b ] $$. Take, for example, the system:

$$
\begin{aligned}
2x + y &= 8 \\
x + 3y &= 13
\end{aligned}
$$

which becomes:

$$
\left[\begin{array}{cc|c}
2 & 1 & 8 \\
1 & 3 & 13 \\
\end{array}\right]
$$

Using elementary row operations—swapping rows, scaling rows, and adding multiples of rows to one another—we aim to convert this matrix into an upper triangular or reduced row-echelon form. Once it’s simplified, back substitution helps us solve for each variable one by one.

Here’s a Python implementation of row reduction in action:

```python
import numpy as np

def row_reduce(A, b):
    Ab = np.hstack((A.astype(float), b.reshape(-1, 1).astype(float)))
    n = Ab.shape[0]
    for i in range(n):
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]
        Ab[i] = Ab[i] / Ab[i, i]
        for j in range(i+1, n):
            Ab[j] = Ab[j] - Ab[i] * Ab[j, i]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])
    return x

A = np.array([[2, 1],
              [1, 3]])
b = np.array([8, 13])
beta = row_reduce(A, b)
print("Solution using row reduction:", beta)
```

In machine learning, row reduction isn’t just a math exercise—it tells us something about our data. For instance, if a solution doesn’t exist or isn’t unique, it may be due to multicollinearity in the dataset. And while libraries use faster numerical methods, the logic here underpins those solvers.

Whether you’re solving regression models, performing parameter estimation, or diagnosing issues in data, understanding row reduction equips you with both intuition and control over the structure of your solution.

---

## Gaussian Elimination for Larger Systems

For larger systems, manually applying row operations isn’t practical. This is where Gaussian elimination shines—a structured algorithm that eliminates variables step-by-step, reducing your system to a manageable triangular form.

At its core, Gaussian elimination performs forward elimination to zero out entries below the diagonal, followed by back substitution to solve for each variable from bottom to top.

The method also relies on **partial pivoting**—swapping rows to ensure numerical stability, especially important when working with floating-point data in real-world ML scenarios.

Here’s how you might code it:

```python
import numpy as np

def gaussian_elimination(A, b):
    n = A.shape[0]
    Ab = np.hstack((A.astype(float), b.reshape(-1, 1).astype(float)))
    for i in range(n):
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]
        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:], x[i+1:])) / Ab[i, i]
    return x

A = np.array([[3, 2, -4],
              [2, 3, 3],
              [5, -3, 1]])
b = np.array([3, 15, 14])
solution = gaussian_elimination(A, b)
print("Solution using Gaussian elimination:", solution)
```

In machine learning, this method is commonly used when solving normal equations in regression. It's also part of the foundation behind algorithms like LU decomposition and QR factorization. Gaussian elimination is particularly useful in data fitting, interpolation, and models that involve large-scale linear systems. While it may not always be used directly, it's hiding beneath many optimization libraries used in practice.

---

## Cramer’s Rule: A Theoretical Lens

For smaller systems—or when you're looking for a more theoretical perspective—Cramer’s Rule offers an elegant way to find each variable explicitly using determinants.

Given a system $$A\mathbf{x} = \mathbf{b}$$, Cramer’s Rule gives each solution as:

$$
x_i = \frac{\det(A_i)}{\det(A)}
$$

where $$A_i$$ is the matrix formed by replacing the $$i$$-th column of $$A$$ with the vector $$\mathbf{b}$$.

While this approach is computationally expensive for large systems (since it requires calculating multiple determinants), it’s incredibly valuable for understanding how solutions relate to the structure of the system. For instance, a zero determinant tells you that the system doesn’t have a unique solution—critical for understanding singularities and data redundancy.

Here’s an implementation in Python:

```python
import numpy as np

def cramer_rule(A, b):
    det_A = np.linalg.det(A)
    if np.isclose(det_A, 0):
        raise ValueError("The system has no unique solution (determinant is zero).")
    n = A.shape[0]
    x = np.zeros(n)
    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b
        x[i] = np.linalg.det(A_i) / det_A
    return x

A = np.array([[2, -1, 3],
              [1,  0, 2],
              [4,  1, 8]])
b = np.array([5, 4, 12])
solution_cramer = cramer_rule(A, b)
print("Solution using Cramer's Rule:", solution_cramer)
```

Though not practical for large ML pipelines, Cramer’s Rule is ideal for educational and theoretical purposes. It’s often used to validate the structure of small systems and to perform **sensitivity analysis**—seeing how small changes in input affect the output. It's especially insightful when exploring linear dependencies between variables in regression or in understanding how features influence predictions at a theoretical level.

---

## Conclusion

Solving systems of linear equations isn’t just a math skill—it’s a fundamental tool in the machine learning toolbox. Whether you’re fitting a linear regression model or designing a network analysis algorithm, these techniques will inevitably come into play.

Row reduction gives you a practical handle on small systems and helps interpret model behavior. Gaussian elimination scales that power for larger systems and forms the foundation of more advanced solvers. And Cramer’s Rule offers theoretical clarity on how and why solutions behave the way they do.

Mastering these approaches deepens your understanding of both data and models—ensuring you’re not just using ML tools, but also understanding what’s happening behind the scenes.

