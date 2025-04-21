---
layout: post
title: Linear Algebra Basics for ML - Vector Operations, Norms, and Projections
date: 2022-01-13
description: Linear Algebra 1 - Mathematics for Machine Learning
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



<div style="text-align: center; margin: 2rem 0;">
  {% include figure.liquid 
      path="assets/img/linalg1_1.jpg" 
      class="img-fluid rounded shadow-sm" 
      loading="eager" 
      zoomable=true 
      alt="" 
  %}
</div>

<p>In Indian Hindu culture, it’s tradition to begin any auspicious journey with a prayer to <strong>Lord Ganesha</strong>, the remover of obstacles. In the spirit of that tradition, as we embark on our journey through the fascinating world of <strong>Linear Algebra</strong>, we begin with our own humble invocation.</p>

<p>But this time, it’s not to a deity carved in stone — it’s to two giants whose teachings have shaped how we understand the abstract beauty of vectors, matrices, and transformations:</p>

<ul>
  <li><strong>Gilbert Strang</strong> — the legendary MIT professor whose lectures have brought clarity and structure to thousands of learners around the world.</li>
  <li><strong>Grant Sanderson</strong> — the mind behind <em>3Blue1Brown</em>, who gave visual intuition to linear algebra through stunning animations and deep insights.</li>
</ul>

<p>This blog series is a tribute and a continuation — a place where we'll walk step by step through the core concepts, problems, and applications of Linear Algebra with a lens tuned for data science and machine learning.</p>

<p>Let's all whisper the sacred mantra that begins all good things:</p>

<p style="text-align: center; font-size: 1.3rem; margin-top: 1rem; margin-bottom: 1rem;">
  $$ A \mathbf{x} = \mathbf{b} $$
</p>

<p>May your matrices be full rank,<br>
May your basis always span,<br>
And may your inner product bring clarity, not confusion.<br>
Let the journey begin. </p>




---


So, with our mathematical minds tuned, let’s begin—not with the heaviest proofs or grand abstractions, but with the smallest steps. 

Linear algebra, like all things sacred and scientific, starts with the humble: a line, a vector, a movement through space. Whether you're adjusting weights in a neural net or calculating embeddings in NLP, you're moving through vector space—scaling, shifting, aligning.




---

## Vector Addition and Scalar Multiplication

Imagine you're training a neural network. Every time you update its weights using gradient descent, you're really doing something quite basic: taking one vector (the weights), adding another (the gradient scaled by learning rate), and replacing the old with the new. This isn’t just an implementation detail—it’s a foundational operation that sits at the heart of every learning loop.

Let’s break it down.

A vector $$\mathbf{v}$$ in $$\mathbb{R}^n$$ is an ordered list of $$n$$ real numbers:

$$
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
$$

Vectors can represent anything: coordinates in space, model weights, gradient directions, or even raw data. They're the language of modern ML.

To combine two vectors $$\mathbf{u}$$ and $$\mathbf{v}$$, you just add them component-wise:

$$
\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}
$$

{% raw %}
<div style="margin: 2rem auto; max-width: 900px;">
  <div id="vectorOpsPlot" style="height: 600px;"></div>
</div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function () {
  function plotArrow(x0, y0, x1, y1, color, width = 3, dash = "solid", name = '') {
    const headSize = 0.3;
    const angle = Math.atan2(y1 - y0, x1 - x0);
    const xHead1 = x1 - headSize * Math.cos(angle - Math.PI / 6);
    const yHead1 = y1 - headSize * Math.sin(angle - Math.PI / 6);
    const xHead2 = x1 - headSize * Math.cos(angle + Math.PI / 6);
    const yHead2 = y1 - headSize * Math.sin(angle + Math.PI / 6);

    return [
      {
        type: 'scatter',
        mode: 'lines',
        x: [x0, x1],
        y: [y0, y1],
        line: { color: color, width: width, dash: dash },
        name: name,
        showlegend: true,
        hoverinfo: 'name'
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: [x1, xHead1, xHead2, x1],
        y: [y1, yHead1, yHead2, y1],
        fill: 'toself',
        fillcolor: color,
        line: { color: color, width: 1 },
        showlegend: false,
        hoverinfo: 'skip'
      }
    ];
  }

  const u = [2, 1];
  const v = [1, 3];
  const u_plus_v = [u[0] + v[0], u[1] + v[1]];

  const traces = [
    ...plotArrow(0, 0, u[0], u[1], 'blue', 3, 'solid', 'u = [2, 1]'),
    ...plotArrow(0, 0, v[0], v[1], 'red', 3, 'solid', 'v = [1, 3]'),
    ...plotArrow(u[0], u[1], u_plus_v[0], u_plus_v[1], 'red', 2, 'dot'),
    ...plotArrow(0, 0, u_plus_v[0], u_plus_v[1], 'green', 4, 'solid', 'u + v = [3, 4]')
  ];

  const layout = {
    title: 'Vector Addition: u + v',
    width: 700,
    height: 550,
    xaxis: { title: 'X', range: [-1, 6], zeroline: true, gridcolor: '#eee' },
    yaxis: { title: 'Y', range: [-1, 6], zeroline: true, gridcolor: '#eee', scaleanchor: "x", scaleratio: 1 },
    plot_bgcolor: "#fcfcfc",
    paper_bgcolor: "#ffffff",
    margin: { l: 60, r: 40, t: 70, b: 60 },
    showlegend: true,
    font: { family: "Segoe UI", size: 16 }
  };

  Plotly.newPlot('vectorOpsPlot', traces, layout);
});
</script>
{% endraw %}

<p style="text-align: center; font-style: italic; margin-top: 5px;">
  <b>Visualization:</b> Vector <b>v</b> (red, dashed) starts at the tip of <b>u</b> (blue). The resulting vector <b>u + v</b> (green) connects the origin to the end of the chain. This is vector addition in action.
</p>

Now, multiplying a vector by a scalar $$\alpha$$ stretches or shrinks it:

$$
\alpha \mathbf{u} = \begin{bmatrix} \alpha u_1 \\ \alpha u_2 \\ \vdots \\ \alpha u_n \end{bmatrix}
$$

The direction stays the same, but the length changes. If $$\alpha < 0$$, the vector flips around the origin.

This is the backbone of gradient descent:

$$
\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} - \alpha \nabla \mathbf{w}
$$

You're scaling the gradient, flipping it (via subtraction), and combining it with the old weights—exactly the operations we’ve just visualized.

Whether you're training a deep model or tuning a simple linear regression, vector addition and scalar multiplication are the building blocks. The math may be simple—but the impact is profound.



---

## Linear Combinations, Span, Basis, and Dimensionality

Let’s move one level higher in abstraction. Suppose you have a bunch of vectors—what can you build from them? A lot, it turns out. In fact, much of machine learning is built on the idea that complex things can be constructed by combining simple things in clever ways.

This idea is formalized through **linear combinations**. Given a set of vectors $$\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k$$ in $$\mathbb{R}^n$$, we can build a new vector like so:

$$
\mathbf{v} = \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \dots + \alpha_k \mathbf{v}_k
$$

Here, each original vector is scaled by a coefficient, and then all are added together. This is more than a formula—this is how feature engineering works, how word embeddings combine meanings, and how models generalize.

The set of *all* such vectors you can make using these combinations is called the **span**. For example, the span of two non-parallel vectors in 2D is the entire plane. The span tells you how expressive your set of vectors is.

But not all sets of vectors are created equal. Some are redundant. That’s where the concept of a **basis** comes in.

A basis is a minimal set of **linearly independent** vectors whose span still covers the whole space. Every vector in the space can be written *uniquely* as a linear combination of the basis vectors.

The number of vectors in that basis? That’s the **dimension** of the space.

Dimensionality matters deeply in ML. High-dimensional data is everywhere—images, text, genomics—but often, we don’t need all those dimensions. That’s where dimensionality reduction enters.

A classic technique is PCA (Principal Component Analysis). PCA finds new basis vectors—called **principal components**—which are orthogonal and capture the most variance in the data. You can then drop the less useful components and keep only the top ones, shrinking your data’s dimensionality without losing much information.

---

### Numerical Example

Suppose we are given two vectors:
- $$\mathbf{v}_1 = [2, 1]$$
- $$\mathbf{v}_2 = [1, 3]$$

These two vectors are not scalar multiples of each other, so they are linearly independent and span a plane in $$\mathbb{R}^2$$. In other words, **any point in 2D space** can be written as a linear combination of $$\mathbf{v}_1$$ and $$\mathbf{v}_2$$.

Let’s choose coefficients:
- $$\alpha_1 = 2$$
- $$\alpha_2 = -1$$

Then the linear combination becomes:

$$
\mathbf{v} = 2 \cdot [2, 1] + (-1) \cdot [1, 3] = [4, 2] + [-1, -3] = [3, -1]
$$

So, the vector $$[3, -1]$$ lies in the **span** of $$\mathbf{v}_1$$ and $$\mathbf{v}_2$$.

Let’s verify it computationally:

```python
import numpy as np

v1 = np.array([2, 1])
v2 = np.array([1, 3])
coeffs = np.array([2, -1])

new_vector = coeffs[0] * v1 + coeffs[1] * v2
print("New vector (linear combination):", new_vector)
```

Output:
```
New vector (linear combination): [ 3 -1]
```

Now let’s understand how this connects to span, basis, and dimension:

- **Linear Combination**: By scaling and adding the vectors, we reached a new point in space.
- **Span**: All such combinations form the filled-in area (the entire 2D plane in this case) spanned by $$\mathbf{v}_1$$ and $$\mathbf{v}_2$$.
- **Basis**: Since $$\mathbf{v}_1$$ and $$\mathbf{v}_2$$ are linearly independent and span $$\mathbb{R}^2$$, they form a basis.
- **Dimensionality**: The number of vectors in the basis is 2 ⇒ so the space is 2D.

This example shows that you can express any vector in 2D using a basis of two independent vectors. And this same idea is extended in machine learning when we:
- Learn latent embeddings with fewer dimensions
- Use PCA to reduce input feature space
- Combine attention vectors in transformers
- Compress images or signals in autoencoders

---

The same idea scales up: Autoencoders learn compact representations (i.e., a new basis). Sparse coding finds minimal combinations to represent signals. In NLP, transformer models represent words as weighted combinations of basis-like vectors. In genomics and medical imaging, dimensionality reduction helps extract essential patterns from noisy, high-dimensional data.

The key takeaway? Whenever you reduce features, compress signals, or build latent spaces—you’re working with linear combinations and bases, even if you don’t always realize it.


---

## Orthogonality and Projections

Now that we’ve learned how to construct vectors using others, let’s explore how to simplify or extract structure from vectors we already have. This is where projections—and particularly **orthogonal projections**—come into play. They help us reduce complexity while preserving what matters most.

Imagine you have a vector $$\mathbf{u}$$ that represents some data, and you want to understand how much of it aligns with another vector $$\mathbf{v}$$—perhaps a direction that captures maximum variance, or a component of interest in a dataset.

To isolate that portion of $$\mathbf{u}$$, we project it onto $$\mathbf{v}$$. This projection is essentially $$\mathbf{u}$$’s shadow in the direction of $$\mathbf{v}$$, and it's defined by:

$$
\text{proj}_{\mathbf{v}}(\mathbf{u}) = \left( \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \right) \mathbf{v}
$$

This formula gives us a new vector pointing in the direction of $$\mathbf{v}$$, scaled to reflect how much $$\mathbf{u}$$ "leans" toward it.

Why is this useful?

Because orthogonal projections let us decompose vectors. For example, in **Principal Component Analysis (PCA)**, we use projections to compress data: projecting high-dimensional data onto a smaller set of orthogonal axes (principal components) that capture the greatest variance.

Let’s make that concrete with a code snippet:

```python
import numpy as np

data_point = np.array([3, 4])
principal_component = np.array([1, 0])

# Projection of data_point onto principal_component
dot = np.dot(data_point, principal_component)
norm_sq = np.dot(principal_component, principal_component)
projection = (dot / norm_sq) * principal_component
print("Projection:", projection)
```

This returns `[3, 0]`—a shadow of `[3, 4]` on the x-axis. You’ve just performed dimensionality reduction: mapping a 2D point to a 1D axis while preserving the meaningful part.

---

### Numerical Example

Let’s unpack the projection formula with another set of numbers to build a strong geometric intuition.

Suppose:
- $$\mathbf{u} = [2, 3]$$
- $$\mathbf{v} = [4, 0]$$ (aligned with the x-axis)

Let’s compute the projection of $$\mathbf{u}$$ onto $$\mathbf{v}$$:

- Dot product:  
  $$
  \mathbf{u} \cdot \mathbf{v} = 2 \times 4 + 3 \times 0 = 8
  $$

- Magnitude squared of $$\mathbf{v}$$:  
  $$
  \|\mathbf{v}\|^2 = 4^2 + 0^2 = 16
  $$

So the projection is:

$$
\text{proj}_{\mathbf{v}}(\mathbf{u}) = \left( \frac{8}{16} \right) \times [4, 0] = 0.5 \times [4, 0] = [2, 0]
$$

{% raw %}
<div style="display: flex; justify-content: center; overflow: auto; padding: 1rem 0;">
  <div id="projectionExamplePlot" style="width: 100%; max-width: 720px; height: 460px;"></div>
</div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function () {
  const u = [2, 3];
  const v = [4, 0];

  const dot = u[0]*v[0] + u[1]*v[1];
  const vNormSq = v[0]*v[0] + v[1]*v[1];
  const scale = dot / vNormSq;
  const proj = [scale * v[0], scale * v[1]];

  function arrowTrace(x0, y0, x1, y1, color, name, dash = "solid") {
    return {
      type: 'scatter',
      mode: 'lines+markers',
      x: [x0, x1],
      y: [y0, y1],
      name: name,
      line: { color: color, width: 3, dash: dash },
      marker: { size: 7 },
      hoverinfo: 'name'
    };
  }

  const data = [
    arrowTrace(0, 0, u[0], u[1], 'blue', 'u = [2, 3]'),
    arrowTrace(0, 0, v[0], v[1], 'orange', 'v = [4, 0]', 'dot'),
    arrowTrace(0, 0, proj[0], proj[1], 'green', 'proj_v(u) = [2, 0]'),
    arrowTrace(u[0], u[1], proj[0], proj[1], 'gray', 'u - proj_v(u)', 'dash')
  ];

  const layout = {
    title: {
      text: 'Projection of u = [2, 3] onto v = [4, 0]',
      font: { size: 20 },
      xanchor: 'center',
      x: 0.5
    },
    xaxis: { title: 'X', range: [-1, 5], zeroline: true, gridcolor: '#eee' },
    yaxis: { title: 'Y', range: [-1, 5], zeroline: true, gridcolor: '#eee', scaleanchor: 'x', scaleratio: 1 },
    showlegend: true,
    plot_bgcolor: "#fdfdfd",
    paper_bgcolor: "#ffffff",
    margin: { t: 80, l: 60, r: 40, b: 60 },
    font: { family: 'Segoe UI', size: 14 }
  };

  Plotly.newPlot('projectionExamplePlot', data, layout);
});
</script>
{% endraw %}

This tells us that although $$\mathbf{u}$$ points somewhere into the plane, its component along $$\mathbf{v}$$ (the x-direction) is just 2 units. The rest—$$[0, 3]$$—is orthogonal to $$\mathbf{v}$$.

To isolate the orthogonal component:

$$
\mathbf{u}_\perp = \mathbf{u} - \text{proj}_{\mathbf{v}}(\mathbf{u}) = [2, 3] - [2, 0] = [0, 3]
$$

This kind of decomposition is used frequently in machine learning to separate signal from noise, or to isolate the informative component of a data point.

---

You’ll see projections all over the place: in feature extraction for images, orthogonal initialization in deep networks, and latent representations in generative models. Wherever there’s compression or abstraction, projections are probably doing the heavy lifting behind the scenes.

---

## Vector Norms and Model Complexity

You’ve trained a model and it performs well—almost too well. It nails the training data but struggles with new inputs. That’s a red flag: your model might be **overfitting**. A common way to fix this is **regularization**, which penalizes large weights. But what does “large” mean for a vector of weights? Enter **vector norms**.

Vector norms provide a way to measure the **size** or **length** of a vector, giving us a handle to control model complexity.

Let’s walk through the three most common norms:

### L1 Norm: Manhattan Distance

This is the sum of the absolute values of all components:

$$
\|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i|
$$

It’s called “Manhattan” because it mimics city block distances—walking only along gridlines.

### L2 Norm: Euclidean Distance

This is the familiar straight-line distance from the origin:

$$
\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2}
$$

### L∞ Norm: Maximum Norm

This measures the largest single component in the vector:

$$
\|\mathbf{v}\|_\infty = \max_i |v_i|
$$

Each norm is useful in different settings:
- **L1** promotes sparsity (many zeroes), useful in Lasso Regression
- **L2** encourages small but non-zero values, used in Ridge Regression
- **L∞** is useful for constraining peak values, such as in adversarial robustness

Let’s plug in some numbers to make this concrete:

### Numerical Example

Let’s say:

$$
\mathbf{v} = [3, -4, 1]
$$

Then:

- **L1 norm**:

$$
\|\mathbf{v}\|_1 = |3| + |-4| + |1| = 3 + 4 + 1 = 8
$$

- **L2 norm**:

$$
\|\mathbf{v}\|_2 = \sqrt{3^2 + (-4)^2 + 1^2} = \sqrt{9 + 16 + 1} = \sqrt{26} \approx 5.10
$$

- **L∞ norm**:

$$
\|\mathbf{v}\|_\infty = \max(|3|, |4|, |1|) = 4
$$

And here’s the code that calculates these:

```python
import numpy as np

v = np.array([3, -4, 1])

l1 = np.sum(np.abs(v))
l2 = np.sqrt(np.sum(v ** 2))
linf = np.max(np.abs(v))

print("L1 norm:", l1)
print("L2 norm:", l2)
print("L∞ norm:", linf)
```

---

{% raw %}
<div id="normComparePlot" style="width:100%; max-width:700px; margin: 2rem auto;"></div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
const vectors = {
  "v = [3, -4, 1]": [3, -4, 1],
  "v = [1, 2, -2]": [1, 2, -2],
  "v = [-5, 0, 0]": [-5, 0, 0],
  "v = [0, 0, 6]": [0, 0, 6],
};

function calculateNorms(v) {
  const l1 = v.reduce((acc, val) => acc + Math.abs(val), 0);
  const l2 = Math.sqrt(v.reduce((acc, val) => acc + val * val, 0));
  const linf = Math.max(...v.map(val => Math.abs(val)));
  return [l1, l2, linf];
}

const initial = calculateNorms(vectors["v = [3, -4, 1]"]);

const data = [{
  x: ['L1 (Manhattan)', 'L2 (Euclidean)', 'L∞ (Max)'],
  y: initial,
  type: 'bar',
  marker: { color: ['#2a9d8f', '#264653', '#e76f51'] },
  text: initial.map(v => v.toFixed(2)),
  textposition: 'auto'
}];

const layout = {
  title: 'Norm Comparison for Different Vectors',
  yaxis: { title: 'Norm Value', range: [0, 10] },
  plot_bgcolor: '#fafafa',
  paper_bgcolor: '#ffffff',
  margin: { l: 50, r: 20, t: 50, b: 40 },
  updatemenus: [{
    buttons: Object.keys(vectors).map(label => ({
      method: 'restyle',
      args: ['y', [calculateNorms(vectors[label])]],
      label: label
    })),
    direction: 'down',
    showactive: true,
    x: 0.05,
    xanchor: 'left',
    y: 1.2,
    yanchor: 'top'
  }]
};

Plotly.newPlot('normComparePlot', data, layout);
</script>
{% endraw %}

To see how these norms behave for different vectors, here's an interactive plot. Use the dropdown to switch between vectors and observe how the L1, L2, and L∞ norms respond based on vector composition.

---

{% raw %}
<div id="lpBalls" style="width: 100%; max-width: 650px; margin: 2rem auto;"></div>

<script>
function generateLpBall(p, numPoints = 200) {
  const theta = Array.from({ length: numPoints }, (_, i) => (2 * Math.PI * i) / numPoints);
  return theta.map(t => {
    const cosT = Math.cos(t);
    const sinT = Math.sin(t);
    const denom = Math.pow(Math.abs(cosT) ** p + Math.abs(sinT) ** p, 1 / p);
    return [cosT / denom, sinT / denom];
  });
}

const p1 = generateLpBall(1);
const p2 = generateLpBall(2);
const pinf = [[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]];

const trace1 = {
  x: p1.map(p => p[0]), y: p1.map(p => p[1]),
  mode: 'lines', name: 'L1 Norm Ball (p=1)', line: { color: '#e76f51' }
};
const trace2 = {
  x: p2.map(p => p[0]), y: p2.map(p => p[1]),
  mode: 'lines', name: 'L2 Norm Ball (p=2)', line: { color: '#2a9d8f' }
};
const traceInf = {
  x: pinf.map(p => p[0]), y: pinf.map(p => p[1]),
  mode: 'lines', name: 'L∞ Norm Ball (p=∞)', line: { color: '#264653', dash: 'dot' }
};

const layoutBalls = {
  title: 'Lp Norm Balls in 2D',
  xaxis: { title: 'x', range: [-1.5, 1.5] },
  yaxis: { title: 'y', range: [-1.5, 1.5], scaleanchor: 'x', scaleratio: 1 },
  plot_bgcolor: '#fcfcfc',
  paper_bgcolor: '#ffffff',
  margin: { l: 50, r: 20, t: 50, b: 40 }
};

Plotly.newPlot('lpBalls', [trace1, trace2, traceInf], layoutBalls);
</script>
{% endraw %}

Each type of norm defines a different concept of "distance." Here's a geometric view: the L1 norm forms a diamond, L2 a circle, and L∞ a square. These unit norm balls visually explain why different norms behave differently when regularizing or constraining model weights.

---

### Why Norms Matter in ML

Norms are used to **regularize** models—penalizing large weight magnitudes to reduce overfitting:

- L1 regularization adds $$\lambda \|\mathbf{w}\|_1$$ to the loss function
- L2 adds $$\lambda \|\mathbf{w}\|_2^2$$

They’re also used to:

- Clip gradients during training (to avoid exploding gradients)
- Control noise in adversarial training (L\infty perturbation bounds)
- Measure distance in nearest neighbor and anomaly detection models
- Guide similarity in metric learning (e.g. contrastive loss)

Norms offer a mathematical grip on what it means for a model or input to be “large,” and in doing so, help keep our models **efficient**, **robust**, and **generalizable**.

---

## Inner and Outer Products

Let’s say you’re building a recommendation engine or clustering users based on their behavior. One of the first things you’ll need to do is measure how similar two data points are. But how do you quantify "similarity" in a vector space?

That’s where the **inner product** comes in.

Given two vectors $$\mathbf{u}$$ and $$\mathbf{v}$$, the inner product (or dot product) is defined as:

$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i
$$

This value tells us how aligned two vectors are. A large dot product means they’re pointing in the same general direction—like two users with similar preferences. A zero dot product means the vectors are orthogonal, i.e., completely unrelated.

This concept underpins **cosine similarity**, a widely used metric in NLP for comparing word embeddings or document vectors.

<div id="innerProductViz" style="max-width: 650px; height: 480px; margin: 2rem auto;"></div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function () {
  const u = [1, 2];
  const v = [3, 4];
  const dot = u[0]*v[0] + u[1]*v[1];
  const normV2 = v[0]*v[0] + v[1]*v[1];
  const scale = dot / normV2;
  const proj = [scale * v[0], scale * v[1]];

  const data = [
    {
      x: [0, u[0]],
      y: [0, u[1]],
      type: 'scatter',
      mode: 'lines+markers',
      name: 'u',
      line: { color: 'blue', width: 4 }
    },
    {
      x: [0, v[0]],
      y: [0, v[1]],
      type: 'scatter',
      mode: 'lines+markers',
      name: 'v',
      line: { color: 'green', width: 4 }
    },
    {
      x: [0, proj[0]],
      y: [0, proj[1]],
      type: 'scatter',
      mode: 'lines+markers',
      name: 'proj_u_on_v',
      line: { color: 'orange', width: 3, dash: 'dot' }
    },
    {
      x: [u[0], proj[0]],
      y: [u[1], proj[1]],
      type: 'scatter',
      mode: 'lines+markers',
      name: 'residual (u - proj)',
      line: { color: 'gray', width: 2, dash: 'dash' }
    }
  ];

  const layout = {
    title: 'Visualizing Inner Product and Projection',
    xaxis: { title: 'x', range: [-1, 5], zeroline: true },
    yaxis: { title: 'y', range: [-1, 5], zeroline: true, scaleanchor: 'x', scaleratio: 1 },
    plot_bgcolor: "#f9f9f9",
    paper_bgcolor: "#ffffff",
    showlegend: true,
    margin: { l: 40, r: 20, t: 50, b: 40 }
  };

  Plotly.newPlot('innerProductViz', data, layout);
});
</script>

Here’s a visual interpretation of the dot product. The projection of **u** onto **v** represents how much of **u** lies in the direction of **v**. The more aligned they are, the longer the projection—hence a larger dot product.

---

On the other hand, the **outer product** builds a full matrix that captures all pairwise interactions between components of two vectors. If $$\mathbf{u} \in \mathbb{R}^m$$ and $$\mathbf{v} \in \mathbb{R}^n$$, then:

$$
\mathbf{u} \otimes \mathbf{v} =
\begin{bmatrix}
  u_1v_1 & u_1v_2 & \cdots & u_1v_n \\
  u_2v_1 & u_2v_2 & \cdots & u_2v_n \\
  \vdots & \vdots & \ddots & \vdots \\
  u_mv_1 & u_mv_2 & \cdots & u_mv_n
\end{bmatrix}
$$

This matrix can be used to model interactions, correlations, or structure in higher dimensions.

The outer product expands two vectors into a matrix where each entry is the product of one element from **u** and one from **v**. This interaction matrix is fundamental to building things like attention maps or covariance matrices.

### A Numerical Example

Let’s take:  
$$\mathbf{u} = [1, 2] \quad \text{and} \quad \mathbf{v} = [3, 4]$$

- **Inner product**:

$$
\mathbf{u} \cdot \mathbf{v} = 1 \times 3 + 2 \times 4 = 3 + 8 = 11
$$

- **Outer product**:

$$
\mathbf{u} \otimes \mathbf{v} =
\begin{bmatrix}
1 \times 3 & 1 \times 4 \\
2 \times 3 & 2 \times 4
\end{bmatrix} =
\begin{bmatrix}
3 & 4 \\
6 & 8
\end{bmatrix}
$$

And here's how you'd compute both in Python:

```python
import numpy as np

u = np.array([1, 2])
v = np.array([3, 4])

inner = np.dot(u, v)
outer = np.outer(u, v)

print("Inner Product:", inner)
print("Outer Product:\n", outer)
```

### Why These Matter in ML

- The **inner product** appears in similarity search, attention mechanisms, and projection-based reasoning.
- The **outer product** is essential in building covariance matrices, attention weight matrices, and tensor decomposition for multi-modal learning.

In deep learning, attention mechanisms use scaled dot-product attention, which relies on inner products to determine how much focus to place on different inputs.

In kernel methods (like SVMs), the inner product is generalized into a **kernel function**, which lets us work in high-dimensional (or even infinite-dimensional) spaces without computing those spaces explicitly.

Outer products, meanwhile, show up in **bilinear models**, **tensor factorization**, and **multi-head attention**, where interaction between elements of different feature sets needs to be captured.

Together, inner and outer products give us the building blocks for understanding similarity, structure, and interaction in data—and they show up everywhere from NLP to recommender systems to generative models.

---



We’ve covered a lot—maybe more than it first seemed. Starting with simple vector addition and scaling, we built up to ideas like span and basis, saw how projections carve structure out of noise, and explored how norms and products give us tools to measure and compare. On paper, these are just operations. But together, they shape how machine learning models move through data, learn patterns, and ultimately make sense of the world.

And what's striking is that none of this feels outdated. These ideas—linear combinations, dot products, orthogonality—are as relevant in the depths of a transformer model as they were in the early days of signal processing or classical statistics.

If you understand this much, you’re not just doing the math behind machine learning. You’re speaking its native language.
