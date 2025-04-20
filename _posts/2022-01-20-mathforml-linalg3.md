---
layout: post
title: Linear Algebra Basics for ML - Systems of Linear Equations
date: 2022-01-20
description: Linear Algebra 3 - Mathematics for Machine Learning
tags: ml ai linear-algebra math
math: true
categories: machine-learning math math-for-ml
thumbnail: assets/img/linalg_banner.png
giscus_comments: true
related_posts: true
chart:
  plotly: true
toc:
  beginning: true
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


## Solving Systems via Row Reduction

**Generic Algorithmic Steps**  
Given any square system  
$$
A\,x = b,\quad A\in\mathbb R^{n\times n},\;b\in\mathbb R^n,
$$  
you can solve it by:

1. **Form the augmented matrix**  
   $$
   \bigl[\,A \mid b\,\bigr]\;\in\;\mathbb R^{n\times(n+1)}.
   $$
2. **Forward Elimination**  
   For each pivot row $$i=1,\dots,n$$:  
   - **Pivot selection:** swap row $$i$$ with a lower row having largest absolute entry in column $$i$$ to reduce round‑off.  
   - **Normalize** row $$i$$: divide the entire row by the pivot element so that the leading entry becomes 1.  
   - **Eliminate** below: for every row $$j>i$$, subtract (row $$i$$ × that row’s $$i$$‑th entry).
3. **Check for singularities**  
   - If at any step the pivot is (near) zero, the matrix is singular, yielding no solution or infinitely many.  
   - You can detect singularities via the ranks of \(A\) and \([A\mid b]\):

$$
\begin{aligned}
\operatorname{rank}(A)
&<\;
\operatorname{rank}\bigl[\,A\mid b\,\bigr]
\quad\Longrightarrow\quad
\text{no solution},\\[6pt]
\operatorname{rank}(A)
&=\;
\operatorname{rank}\bigl[\,A\mid b\,\bigr]
\;<\;
n
\quad\Longrightarrow\quad
\text{infinitely many solutions}.
\end{aligned}
$$

4. **Back Substitution**  
   Once you have an upper‑triangular system, solve for $$x_n$$ first, then substitute upward to find $$x_{n-1},\dots,x_1$$.

---

### Numeric Example

Solve  
$$
\begin{cases}
2x + y = 8,\\
x + 3y = 13.
\end{cases}
$$

1. **Augmented matrix**  
   $$
   \left[\begin{array}{cc|c}
     2 & 1 & 8 \\[6pt]
     1 & 3 & 13
   \end{array}\right]
   $$
2. **Swap** row 1 and row 2 (so pivot is 1):  
   $$
   \left[\begin{array}{cc|c}
     1 & 3 & 13 \\[3pt]
     2 & 1 & 8
   \end{array}\right]
   $$
3. **Eliminate** below: R₂ → R₂ – 2 × R₁  
   $$
   \left[\begin{array}{cc|c}
     1 & 3 & 13 \\[3pt]
     0 & -5 & -18
   \end{array}\right]
   $$
4. **Normalize** row 2: divide by –5  
   $$
   \left[\begin{array}{cc|c}
     1 & 3 & 13 \\[3pt]
     0 & 1 & \tfrac{18}{5}
   \end{array}\right]
   $$
5. **Back‑substitute**: R₁ → R₁ – 3 × R₂  
   $$
   \left[\begin{array}{cc|c}
     1 & 0 & \tfrac{11}{5} \\[6pt]
     0 & 1 & \tfrac{18}{5}
   \end{array}\right]
   $$

Hence  
$$
x = \tfrac{11}{5} = 2.2,\quad y = \tfrac{18}{5} = 3.6.
$$


---

### Geometric Visualization (2‑D)

Two lines in $$\mathbb R^2$$ intersect at one point if and only if the system has a unique solution. Embed this Plotly snippet in your Jekyll post:

{% raw %}
<div style="display: flex; justify-content: center; margin: 2rem 0;">
  <div id="lineIntersect" style="width:100%; max-width:700px; height:450px;"></div>
</div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
  const x = Array.from({length: 50}, (_,i) => i*0.2);
  const line1 = { x, y: x.map(x => 8 - 2*x), mode:'lines', name:'2x + y = 8' };
  const line2 = { x, y: x.map(x => (13 - x)/3), mode:'lines', name:'x + 3y = 13' };
  const sol   = { x: [11/5], y: [18/5], mode:'markers', marker:{size:10, color:'red'}, name:'(2.2,3.6)' };

  Plotly.newPlot('lineIntersect', [line1, line2, sol], {
    margin: { l:30, r:10, t:20, b:30 },
    xaxis: { title: 'x' },
    yaxis: { title: 'y' }
  });
</script>
{% endraw %}



---

**What you’re seeing**  
- Each equation  
  $$
    2x + y = 8
    \quad\Longleftrightarrow\quad
    y = 8 - 2x,
  $$  
  and  
  $$
    x + 3y = 13
    \quad\Longleftrightarrow\quad
    y = \frac{13 - x}{3},
  $$  
  is plotted as a straight line in the $$xy$$‑plane.  

- The **blue** line ($$2x + y = 8$$) has slope $$m_1 = -2$$ and $$y$$‑intercept $$b_1 = 8$$;  
- The **orange** line ($$x + 3y = 13$$) has slope $$m_2 = -\tfrac{1}{3}$$ and $$y$$‑intercept $$b_2 = \tfrac{13}{3}\approx 4.33.$$

**Why intersection = solution**  
- Algebraically, a point $$(x,y)$$ solves *both* equations exactly when it lies on *both* lines simultaneously.  
- Geometrically, that’s their **point of intersection**. In our numeric example, Plotly marks it in **red** at  
  $$
    \Bigl(\tfrac{11}{5},\,\tfrac{18}{5}\Bigr)\;\approx\;(2.2,3.6).
  $$  
  That matches the row‑reduction result.

**Edge cases you can imagine**  
- If the lines were **parallel** (identical slopes, different intercepts), they’d never meet ⇒ **no solution**.  
- If they were **coincident** (same line drawn twice), they’d overlap entirely ⇒ **infinitely many solutions**.  

**Takeaway for ML**  
- Whenever you solve a 2×2 system, you’re literally finding where those two “constraint” lines agree.  
- In higher dimensions you get planes or hyperplanes; their unique intersection is the only consistent parameter set.  

By pairing the algebraic steps of row reduction with this picture, readers build both symbolic and visual intuition for *why* a unique solution exists—and *how* pathologies (no solution, infinite solutions) appear when lines fail to intersect properly.

---



## Solving Systems via Gaussian Elimination

**Generic Algorithmic Steps**  
For any square system  
$$
A\,x = b,\quad A\in\mathbb{R}^{n\times n},\;b\in\mathbb{R}^n,
$$  
you can solve it in-place via Gaussian elimination:

1. **Form the augmented matrix**  
   $$
   \bigl[\;A\;\bigm|\;b\;\bigr]
   \;\in\;\mathbb{R}^{n\times(n+1)}.
   $$
2. **Forward elimination**  
   For each pivot row $$i=1,\dots,n-1$$:  
   - **Partial pivoting:** swap row $$i$$ with a row $$k>i$$ maximizing  
     $$\bigl|\,[A\mid b]_{k,i}\bigr|$$  
     to reduce numerical error.  
   - For each row $$j=i+1,\dots,n$$, compute  
     $$
     \text{factor}
     =\frac{[A\mid b]_{j,i}}{[A\mid b]_{i,i}},
     $$  
     then do  
     $$
     \text{Row}_j
     \;\longrightarrow\;
     \text{Row}_j
     \;-\;
     \text{factor}\times\text{Row}_i
     $$
     to zero out the entry below the pivot.
3. **Zero‑pivot check**  
   If any pivot $$[A\mid b]_{i,i}$$ is (near) zero after pivoting, the system is singular—no unique solution.
4. **Back substitution**  
   Once the matrix is upper‑triangular, solve  
   $$
   [A\mid b]_{n,n}\,x_n = [A\mid b]_{n,n+1},
   $$  
   then substitute upward for $$x_{n-1},\dots,x_1$$.

---

### Numeric Example

We’ll solve the system using Gaussian elimination:

$$
\begin{cases}
3x + 2y - 4z = 3,\\[6pt]
2x + 3y + 3z = 15,\\[6pt]
5x - 3y +  z = 14
\end{cases}
$$

---

**Step 1. Form the augmented matrix:**

$$
\left[
\begin{array}{ccc|c}
3 &  2 & -4 &  3 \\[4pt]
2 &  3 &  3 & 15 \\[4pt]
5 & -3 &  1 & 14
\end{array}
\right]
$$

---

**Step 2. Eliminate** $$x$$ **from rows 2 and 3 using row 1 (pivot = 3):**

Apply:

$$
\text{R}_2 \longrightarrow \text{R}_2 - \tfrac{2}{3}\,\text{R}_1
\qquad\text{and}\qquad
\text{R}_3 \longrightarrow \text{R}_3 - \tfrac{5}{3}\,\text{R}_1
$$

Resulting in:

$$
\left[
\begin{array}{ccc|c}
3 & 2 & -4 & 3 \\[4pt]
0 & 5 & 17 & 39 \\[4pt]
0 & -19 & 23 & 27
\end{array}
\right]
$$

---

**Step 3. Eliminate** $$y$$ **from row 3 using row 2 (pivot = 5):**

Apply:

$$
\text{R}_3 \longrightarrow \text{R}_3 + \tfrac{19}{5}\,\text{R}_2
$$

Resulting in:

$$
\left[
\begin{array}{ccc|c}
3 & 2 & -4 & 3 \\[4pt]
0 & 5 & 17 & 39 \\[4pt]
0 & 0 & 87.6 & 175.2
\end{array}
\right]
$$

---

**Step 4. Back Substitution:**

From row 3:

$$
87.6\,z = 175.2
\quad\Longrightarrow\quad
z = 2
$$

From row 2:

$$
5\,y + 17 \cdot 2 = 39
\quad\Longrightarrow\quad
5y = 5
\quad\Longrightarrow\quad
y = 1
$$

From row 1:

$$
3\,x + 2 \cdot 1 - 4 \cdot 2 = 3
\quad\Longrightarrow\quad
3x = 9
\quad\Longrightarrow\quad
x = 3
$$

---

**Final Solution:**

$$
(x,\,y,\,z) = (3,\;1,\;2)
$$

---



## Solving Systems via Cramer’s Rule

Cramer’s Rule provides an explicit formula for solving a square linear system when a unique solution exists. Given

$$
A\,x = b
$$

where  
- $$A$$ is an $$n \times n$$ matrix of coefficients  
- $$x$$ is the unknown vector  
- $$b$$ is the result vector  

Each variable $$x_i$$ is computed as:

$$
x_i = \frac{\det(A_i)}{\det(A)}
$$

where  
- $$A_i$$ is the matrix formed by replacing the $$i^{\text{th}}$$ column of $$A$$ with $$b$$  
- $$\det(A) \ne 0$$ is required for a unique solution  

---

### General Steps

1. Compute the determinant of the original matrix $$A$$  
2. For each variable $$x_i$$:
   - Replace the $$i^{\text{th}}$$ column of $$A$$ with the vector $$b$$ to form $$A_i$$
   - Compute $$\det(A_i)$$
   - Compute $$x_i = \frac{\det(A_i)}{\det(A)}$$

---

### Numerical Example

Consider the system:

$$
\begin{cases}
x + y + z = 6 \\\\
2x + 3y + z = 14 \\\\
x - y + 2z = 7
\end{cases}
$$

We’ll solve this using Cramer’s Rule.

---

#### Step 1: Define $$A$$ and $$b$$

$$
A =
\begin{bmatrix}
1 & 1 & 1 \\\\
2 & 3 & 1 \\\\
1 & -1 & 2
\end{bmatrix},
\qquad
b =
\begin{bmatrix}
6 \\\\
14 \\\\
7
\end{bmatrix}
$$

---

#### Step 2: Compute $$\det(A)$$

Using cofactor expansion:

$$
\det(A) = 
1 \cdot (3 \cdot 2 - 1 \cdot (-1)) -
1 \cdot (2 \cdot 2 - 1 \cdot 1) +
1 \cdot (2 \cdot (-1) - 3 \cdot 1)
$$

$$
= 1(6 + 1) - 1(4 - 1) + 1(-2 - 3) = 7 - 3 - 5 = -1
$$

---

#### Step 3: Compute $$\det(A_1)$$  
Replace column 1 with $$b$$:

$$
A_1 =
\begin{bmatrix}
6 & 1 & 1 \\\\
14 & 3 & 1 \\\\
7 & -1 & 2
\end{bmatrix}
$$

Then:

$$
\det(A_1) = 6(3 \cdot 2 - 1 \cdot (-1)) - 1(14 \cdot 2 - 1 \cdot 7) + 1(14 \cdot (-1) - 3 \cdot 7)
$$

$$
= 6(6 + 1) - 1(28 - 7) + 1(-14 - 21)
= 6 \cdot 7 - 21 - 35 = 42 - 21 - 35 = -14
$$

---

#### Step 4: Compute $$\det(A_2)$$  
Replace column 2 with $$b$$:

$$
A_2 =
\begin{bmatrix}
1 & 6 & 1 \\\\
2 & 14 & 1 \\\\
1 & 7 & 2
\end{bmatrix}
$$

Then:

$$
\det(A_2) = 1(14 \cdot 2 - 1 \cdot 7) - 6(2 \cdot 2 - 1 \cdot 1) + 1(2 \cdot 7 - 14 \cdot 1)
$$

$$
= 1(28 - 7) - 6(4 - 1) + 1(14 - 14)
= 21 - 18 + 0 = 3
$$

---

#### Step 5: Compute $$\det(A_3)$$  
Replace column 3 with $$b$$:

$$
A_3 =
\begin{bmatrix}
1 & 1 & 6 \\\\
2 & 3 & 14 \\\\
1 & -1 & 7
\end{bmatrix}
$$

Then:

$$
\det(A_3) = 1(3 \cdot 7 - 14 \cdot (-1)) - 1(2 \cdot 7 - 14 \cdot 1) + 6(2 \cdot (-1) - 3 \cdot 1)
$$

$$
= 1(21 + 14) - 1(14 - 14) + 6(-2 - 3)
= 35 - 0 - 30 = 5
$$

---

#### Step 6: Solve

Now compute:

$$
x = \frac{\det(A_1)}{\det(A)} = \frac{-14}{-1} = 14
$$

$$
y = \frac{\det(A_2)}{\det(A)} = \frac{3}{-1} = -3
$$

$$
z = \frac{\det(A_3)}{\det(A)} = \frac{5}{-1} = -5
$$

---

#### Final Answer

$$
(x,\;y,\;z) = (14,\;-3,\;-5)
$$

Cramer’s Rule is elegant and precise for small systems. For larger systems (say $$n > 4$$), it becomes computationally expensive due to the number of determinants required.


---



## A Quick Reality Check: When Things Go Wrong

Before we celebrate a unique solution, let’s peek at two situations that haunt data scientists in the wild.

### 1. No Solution (Inconsistent System)

Suppose your feature matrix accidentally contains conflicting information:

$$
\begin{aligned}
x + y &= 4 \\\\
2x + 2y &= 9
\end{aligned}
$$

The augmented matrix is:

$$
\left[
\begin{array}{cc|c}
1 & 1 & 4 \\\\
2 & 2 & 9
\end{array}
\right]
$$

Row-reducing gives a *row of doom*:

$$
\left[
\begin{array}{cc|c}
1 & 1 & 4 \\\\
0 & 0 & 1
\end{array}
\right]
\quad\Longrightarrow\quad
0 = 1
$$

So the system is **inconsistent**—exactly what you get when two straight lines in $$\mathbb{R}^2$$ are parallel. In machine learning, this often means a bug in the labels or two contradictory constraints in a rule-based engine.

---

### 2. Infinitely Many Solutions (Underdetermined)

Now flip the constants:

$$
\begin{aligned}
x + y &= 4 \\\\
2x + 2y &= 8
\end{aligned}
$$

Row reduction ends with a *free variable*:

$$
\left[
\begin{array}{cc|c}
1 & 1 & 4 \\\\
0 & 0 & 0
\end{array}
\right]
\quad\Longrightarrow\quad
x = 4 - y,\quad y \text{ free.}
$$

In geometric terms, the two equations describe the *same* line. The solution set is that entire line.

This is often a **multicollinearity** issue in data science—where one feature is a linear combination of others, making the model weights not uniquely identifiable.


## Conclusion

Whether you’re fitting a linear regression model or designing a network analysis algorithm, these techniques will inevitably come into play.

Row reduction gives you a practical handle on small systems and helps interpret model behavior. Gaussian elimination scales that power for larger systems and forms the foundation of more advanced solvers. And Cramer’s Rule offers theoretical clarity on how and why solutions behave the way they do.
