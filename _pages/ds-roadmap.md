---
layout: default
title: "DS Roadmap"
permalink: /ds-roadmap/
nav: false
toc: false
---

<style>
:root {
  --ds-surface: #fff;
  --ds-surface-glass: rgba(255,255,255,0.64);
  --ds-surface-hover: #f5f7fa;
  --ds-border: #e0e4ea;
  --ds-border-focus: #1976d2;
  --ds-accent: #1976d2;
  --ds-accent-light: #3fa3ff;
  --ds-shadow: 0 2px 24px 0 rgba(20,28,48,0.08);
  --ds-shadow-hover: 0 8px 32px 0 rgba(60,80,170,0.08);
  --ds-text: #1b1d1f;
  --ds-text-muted: #748094;
  --ds-pill-published: #e6f6e9;
  --ds-pill-published-text: #16913b;
  --ds-pill-upcoming: #fff8ea;
  --ds-pill-upcoming-text: #db8600;
}

@media (prefers-color-scheme: dark) {
  :root {
    --ds-surface: #15171b;
    --ds-surface-glass: rgba(21,23,27,0.75);
    --ds-surface-hover: #1d2024;
    --ds-border: #31343c;
    --ds-border-focus: #42a5f5;
    --ds-accent: #42a5f5;
    --ds-accent-light: #3fa3ff;
    --ds-shadow: 0 2px 16px 0 rgba(15,28,44,0.28);
    --ds-shadow-hover: 0 8px 36px 0 rgba(0,40,130,0.18);
    --ds-text: #f7fafd;
    --ds-text-muted: #a2afbe;
    --ds-pill-published: #23392e;
    --ds-pill-published-text: #66e393;
    --ds-pill-upcoming: #33291a;
    --ds-pill-upcoming-text: #ffd88b;
  }
}

/* Support for manual/class-based dark themes */
body.dark, html.dark, [data-theme="dark"], body.dark-mode {
  --ds-surface: #15171b;
  --ds-surface-glass: rgba(21,23,27,0.75);
  --ds-surface-hover: #1d2024;
  --ds-border: #31343c;
  --ds-border-focus: #42a5f5;
  --ds-accent: #42a5f5;
  --ds-accent-light: #3fa3ff;
  --ds-shadow: 0 2px 16px 0 rgba(15,28,44,0.28);
  --ds-shadow-hover: 0 8px 36px 0 rgba(0,40,130,0.18);
  --ds-text: #f7fafd;
  --ds-text-muted: #a2afbe;
  --ds-pill-published: #23392e;
  --ds-pill-published-text: #66e393;
  --ds-pill-upcoming: #33291a;
  --ds-pill-upcoming-text: #ffd88b;
}

.roadmap-header {
  text-align: center;
  margin: 2.8rem 0 3.5rem;
}
.roadmap-header h2 {
  margin: 0 0 0.5rem 0;
  font-size: 2.3rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--ds-text);
}
.roadmap-subtitle {
  margin: 0;
  font-size: 1.03rem;
  font-weight: 400;
  color: var(--ds-text-muted);
  letter-spacing: 0.01em;
  line-height: 1.7;
}

.roadmap {
  max-width: 720px;
  margin: 0 auto;
  padding: 0 1rem;
}

.roadmap details {
  margin: 1.3rem 0;
  border: 1.5px solid var(--ds-border);
  border-radius: 16px;
  background: var(--ds-surface-glass);
  box-shadow: var(--ds-shadow);
  transition: box-shadow .22s cubic-bezier(.4,0,.2,1), border-color .18s;
  overflow: hidden;
  backdrop-filter: blur(3px);
}

.roadmap details:hover {
  box-shadow: var(--ds-shadow-hover);
  border-color: var(--ds-accent);
}

.roadmap details[open] {
  border-color: var(--ds-border-focus);
  box-shadow: var(--ds-shadow-hover);
}

.roadmap summary {
  display: flex;
  align-items: center;
  gap: 1.1rem;
  padding: 1.12rem 2rem;
  list-style: none;
  font-weight: 510;
  font-size: 1.08rem;
  cursor: pointer;
  user-select: none;
  color: var(--ds-text);
  background: transparent;
  transition: background .14s;
  outline: none;
}

.roadmap summary::-webkit-details-marker { display: none; }

/* Fixed dropdown arrow using CSS borders instead of SVG masks */
.roadmap summary::before {
  content: "";
  width: 0;
  height: 0;
  border-left: 6px solid var(--ds-accent);
  border-top: 4px solid transparent;
  border-bottom: 4px solid transparent;
  margin-right: 0.5rem;
  transition: transform .21s cubic-bezier(.4,0,.2,1);
  opacity: 0.8;
}

.roadmap details[open] > summary::before {
  transform: rotate(90deg);
  opacity: 1;
  border-left-color: var(--ds-accent-light);
}

.roadmap summary:hover {
  background: var(--ds-surface-hover);
}

.roadmap details > ul {
  margin: 0;
  padding: 0 2rem 1.5rem;
  background: transparent;
  border-top: 1px solid var(--ds-border);
  font-size: 1.01rem;
}

.roadmap details details {
  margin: 0.75rem 0;
  border: 1px solid var(--ds-border);
  border-radius: 12px;
  background: var(--ds-surface);
  box-shadow: 0 1px 3px rgba(32,44,80,.07);
}
.roadmap details details summary {
  padding: 0.9rem 1.2rem;
  font-weight: 460;
  font-size: 0.97rem;
}
.roadmap details details > ul {
  padding: 0 1.2rem 1rem;
}

/* Modern, clean list items */
.roadmap li {
  padding: 0.26rem 0;
  list-style: none;
  position: relative;
}
.roadmap li:not(:has(details)) {
  padding-left: 1.33rem;
}
.roadmap li:not(:has(details))::before {
  content: "";
  position: absolute;
  left: 0.4rem;
  top: 0.85em;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--ds-accent);
  opacity: 0.14;
}

.roadmap a {
  text-decoration: none;
  color: var(--ds-text);
  font-weight: 500;
  transition: color 0.15s;
  border-bottom: 1.5px solid transparent;
  border-radius: 3px;
  padding: 0 0.1em;
  line-height: 1.6;
}
.roadmap a:hover {
  color: var(--ds-accent);
  border-bottom: 1.5px solid var(--ds-accent-light);
}

.roadmap em {
  color: var(--ds-text-muted);
  font-style: italic;
  font-weight: 400;
  letter-spacing: 0.01em;
}

.status-badge {
  display: inline-block;
  padding: 0.14em 0.9em;
  border-radius: 14px;
  font-size: 0.77rem;
  font-weight: 570;
  margin-left: 0.85rem;
  vertical-align: middle;
  background: var(--ds-pill-upcoming);
  color: var(--ds-pill-upcoming-text);
  border: none;
  letter-spacing: 0.015em;
  box-shadow: 0 1px 2px rgba(255,200,100,0.06);
  user-select: none;
}
.status-published {
  background: var(--ds-pill-published);
  color: var(--ds-pill-published-text);
}
.status-upcoming {
  background: var(--ds-pill-upcoming);
  color: var(--ds-pill-upcoming-text);
}

/* Footer */
.roadmap-footer {
  text-align: center;
  margin-top: 4rem;
  padding: 2rem 1rem 1rem 1rem;
  border-top: 1px solid var(--ds-border);
  background: transparent;
}
.roadmap-footer p {
  margin: 0;
  color: var(--ds-text-muted);
  font-size: 0.94rem;
  letter-spacing: 0.01em;
  line-height: 1.7;
}
.roadmap-footer a {
  color: var(--ds-accent);
  text-decoration: underline dotted;
  font-weight: 510;
  transition: color 0.18s, opacity 0.16s;
}
.roadmap-footer a:hover { opacity: 0.75; color: var(--ds-accent-light); }

/* Responsive */
@media (max-width: 800px) {
  .roadmap-header { margin: 2rem 0 2.5rem;}
  .roadmap-header h2 { font-size: 2rem; }
  .roadmap { padding: 0 0.6rem;}
  .roadmap details { margin: 0.85rem 0;}
  .roadmap summary { padding: 0.88rem 1.2rem;}
  .roadmap details > ul { padding: 0 1.15rem 1.2rem;}
}
</style>

<center>
  <h2>Data Science Blog Series</h2>
</center>

<div class="roadmap">

<!-- 0. Foundations -->
<details>
<summary>0. Foundations & Orientation <span class="status-badge status-published">Live</span></summary>
<ul>
  <li><a href="{{ site.baseurl }}/blog/2024/ds000-introduction/">Introduction to Data Science, ML & AI</a></li>
</ul>
</details>

<!-- 1. Mathematics for ML -->
<details id="math">
<summary>1. Mathematics for Data Science <span class="status-badge status-published">Live</span></summary>
<ul>

  <!-- 1.1 Linear Algebra -->
  <li>
    <details>
    <summary>1.1 Linear Algebra</summary>
      <ul>
        <li><a href="{{ site.baseurl }}/blog/2022/mathforml-linalg1/">Vector Operations, Norms & Projections</a></li>
        <li><a href="{{ site.baseurl }}/blog/2022/mathforml-linalg2/">Matrices & Matrix Operations</a></li>
        <li><a href="{{ site.baseurl }}/blog/2022/mathforml-linalg3/">Systems of Linear Equations</a></li>
        <li><a href="{{ site.baseurl }}/blog/2022/mathforml-linalg4/">Eigenvalues, Eigenvectors & SVD</a></li>
        <li><a href="{{ site.baseurl }}/blog/2022/mathforml-linalg5/">Vector Spaces & Transformations</a></li>
        <li><a href="{{ site.baseurl }}/blog/2022/mathforml-linalg6/">Linear Algebra — Bonus Topics</a></li>
      </ul>
    </details>
  </li>

  <!-- 1.2 Probability & Statistics -->
  <li>
    <details>
    <summary>1.2 Probability & Statistics</summary>
      <ul>
        <li><a href="{{ site.baseurl }}/blog/2022/mathforml-probstat1/">Foundations of Probability</a></li>
        <li><a href="{{ site.baseurl }}/blog/2022/mathforml-probstat2/">Probability Distributions</a></li>
        <li><a href="{{ site.baseurl }}/blog/2022/mathforml-probstat3/">Bayesian Thinking, MLE, MAP & Inference</a></li>
        <li><a href="{{ site.baseurl }}/blog/2022/mathforml-probstat4/">Statistical Inference & Hypothesis Testing</a></li>
        <li><a href="{{ site.baseurl }}/blog/2022/mathforml-probstat5/">Markov Chains & Probabilistic Sequence Models</a></li>
        <li><a href="{{ site.baseurl }}/blog/2022/mathforml-probstat6/">Expectation-Maximization (EM)</a></li>
        <li><a href="{{ site.baseurl }}/blog/2022/mathforml-probstat7/">Information Theory & Statistical Learning</a></li>
        <li><a href="{{ site.baseurl }}/blog/2022/mathforml-probstat8/">Linear Regression Models (Probabilistic View)</a></li>
        <li><a href="{{ site.baseurl }}/blog/2023/mathforml-probstat9/">Applied Probabilistic Modeling</a></li>
      </ul>
    </details>
  </li>

  <!-- 1.3 Calculus & Optimization -->
  <li>
    <details>
    <summary>1.3 Calculus & Optimization</summary>
      <ul>
        <li><a href="{{ site.baseurl }}/blog/2023/mathforml-optmcalc1/">Multivariable Functions & Gradients</a></li>
        <li><a href="{{ site.baseurl }}/blog/2023/mathforml-optmcalc2/">Optimization Foundations</a></li>
        <li><a href="{{ site.baseurl }}/blog/2023/mathforml-optmcalc3/">Taylor Expansions & Second-Order Thinking</a></li>
        <li><a href="{{ site.baseurl }}/blog/2023/mathforml-optmcalc4/">Convexity, Constraints & Convergence Guarantees</a></li>
        <li><a href="{{ site.baseurl }}/blog/2023/mathforml-optmcalc5/">Adaptive Optimization — Adam, RMSProp & Beyond</a></li>
        <li><a href="{{ site.baseurl }}/blog/2023/mathforml-optmcalc6/">Regularization & Generalization in Optimization</a></li>
        <li><a href="{{ site.baseurl }}/blog/2024/mathforml-optmcalc7/">Matrix Calculus for Data Science</a></li>
        <li><a href="{{ site.baseurl }}/blog/2024/mathforml-optmcalc8/">Optimization Strategies in Modern ML (Non-Convexity, Hyper-params, Meta-Learning)</a></li>
      </ul>
    </details>
  </li>

</ul>
</details>

<!-- 2. Preprocessing -->
<details>
<summary>2. Data Preprocessing & Feature Engineering <span class="status-badge status-published">Live</span></summary>
<ul>
  <li><a href="{{ site.baseurl }}/blog/2024/ds001-preprocessing1/">2.1 Data Collection & Understanding</a></li>
  <li><a href="{{ site.baseurl }}/blog/2024/ds002-preprocessing2/">2.2 Data Cleaning</a></li>
  <li><a href="{{ site.baseurl }}/blog/2024/ds003-preprocessing3/">2.3 Data Transformation</a></li>
  <li><a href="{{ site.baseurl }}/blog/2024/ds004-preprocessing4/">2.4 Feature Engineering</a></li>
  <li><a href="{{ site.baseurl }}/blog/2024/ds005-preprocessing5/">2.5 Handling Imbalanced Data</a></li>
  <li><a href="{{ site.baseurl }}/blog/2024/ds006-preprocessing6/">2.6 Dimensionality Reduction</a></li>
</ul>
</details>

<!-- 3. Core ML -->
<details>
<summary>3. Core Machine Learning Algorithms <span class="status-badge status-published">Ongoing</span></summary>
<ul>

  <!-- 3.1 Regression -->
  <li><details><summary>3.1 Regression</summary>
    <ul>
      <li><a href="{{ site.baseurl }}/blog/2025/ds007-heartofml-regression1/">3.1.1 Linear Regression (OLS)</a></li>
      <li><a href="{{ site.baseurl }}/blog/2025/ds008-heartofml-regression2/">3.1.2 Regularized Linear Models</a></li>
      <li><a href="{{ site.baseurl }}/blog/2025/ds009-heartofml-regression3/">3.1.3 Generalized Linear Models</a></li>
      <li><a href="{{ site.baseurl }}/blog/2025/ds010-heartofml-regression4/">3.1.4 Evaluation Metrics</a></li>
      <li><a href="{{ site.baseurl }}/blog/2025/ds011-heartofml-regression5/">3.1.5 Robust Regression</a></li>
      <li><a href="{{ site.baseurl }}/blog/2025/ds012-heartofml-regression6/">3.1.6 Capturing Non-Linearity in Regression</a></li>
      <li><a href="{{ site.baseurl }}/blog/2025/ds013-heartofml-regression7/">3.1.7 Advanced Parametric Regression Techniques</a></li>
      <li><em>3.1.7 Optimization Techniques for Regression</em> <span class="status-badge status-upcoming">Soon</span></li>
    </ul>
  </details></li>

  <!-- 3.2 Classification -->
  <li><details><summary>3.2 Classification</summary>
    <ul>
      <li><em>3.2.1 Foundations & Problem Framing</em> <span class="status-badge status-upcoming">Soon</span></li>
      <li><em>3.2.2 Classical P1 (Logistic, Naïve Bayes)</em> <span class="status-badge status-upcoming">Soon</span></li>
      <li><em>3.2.3 Classical P2 (k-NN, SVM)</em> <span class="status-badge status-upcoming">Soon</span></li>
      <li><em>3.2.4 Advanced P1 (Softmax, LDA)</em> <span class="status-badge status-upcoming">Soon</span></li>
      <li><em>3.2.5 Advanced P2 (QDA, PGMs)</em> <span class="status-badge status-upcoming">Soon</span></li>
      <li><em>3.2.6 Evaluation & Diagnostics</em> <span class="status-badge status-upcoming">Soon</span></li>
      <li><em>3.2.7 Optimization & Training</em> <span class="status-badge status-upcoming">Soon</span></li>
      <li><em>3.2.8 Deployment & Interpretability</em> <span class="status-badge status-upcoming">Soon</span></li>
    </ul>
  </details></li>

  <!-- 3.3 Tree/Ensemble -->
  <li><details><summary>3.3 Tree-Based & Ensemble Methods</summary>
    <ul>
      <li><em>3.3.1 Decision Trees</em> <span class="status-badge status-upcoming">Soon</span></li>
      <li><em>3.3.2 Random Forests</em> <span class="status-badge status-upcoming">Soon</span></li>
      <li><em>3.3.3 Boosting Algorithms</em> <span class="status-badge status-upcoming">Soon</span></li>
    </ul>
  </details></li>

  <!-- 3.4 Unsupervised -->
  <li><details><summary>3.4 Unsupervised Learning</summary>
    <ul>
      <li><em>3.4.1 Clustering</em> <span class="status-badge status-upcoming">Soon</span></li>
      <li><em>3.4.2 Dimensionality Reduction & Embeddings</em> <span class="status-badge status-upcoming">Soon</span></li>
      <li><em>3.4.3 Anomaly Detection</em> <span class="status-badge status-upcoming">Soon</span></li>
      <li><em>3.4.4 Association Rule Learning</em> <span class="status-badge status-upcoming">Soon</span></li>
      <li><em>3.4.5 Recommender Systems (Bridge)</em> <span class="status-badge status-upcoming">Soon</span></li>
    </ul>
  </details></li>

  <!-- 3.5 Probabilistic -->
  <li><details><summary>3.5 Probabilistic & Generative Models</summary>
    <ul>
      <li><em>3.5.1 MLE & MAP</em> <span class="status-badge status-upcoming">Soon</span></li>
      <li><em>3.5.2 Expectation-Maximization</em> <span class="status-badge status-upcoming">Soon</span></li>
      <li><em>3.5.3 Hidden Markov Models</em> <span class="status-badge status-upcoming">Soon</span></li>
      <li><em>3.5.4 Conditional Random Fields</em> <span class="status-badge status-upcoming">Soon</span></li>
    </ul>
  </details></li>

</ul>
</details>

<!-- 4. Evaluation -->
<details>
<summary>4. Model Evaluation, Hyper-parameter Tuning & Experimentation <span class="status-badge status-upcoming">Coming</span></summary>
<ul>
  <li><em>4.1 Cross-Validation Strategies</em></li>
  <li><em>4.2 Metric Selection</em></li>
  <li><em>4.3 Hyper-parameter Optimization</em></li>
  <li><em>4.4 Model Interpretability</em></li>
  <li><em>4.5 A/B Testing & Experimentation</em></li>
  <li><em>4.6 Fairness & Bias Mitigation</em></li>
</ul>
</details>

<!-- 5. Deep Learning -->
<details>
<summary>5. Deep Learning <span class="status-badge status-upcoming">Coming</span></summary>
<ul>

  <!-- 5.1 Fundamentals -->
  <li><details><summary>5.1 Neural Network Fundamentals</summary>
    <ul>
      <li><em>5.1.1 Perceptron & Feed-Forward NNs</em></li>
      <li><em>5.1.2 Training Neural Networks</em></li>
      <li><em>5.1.3 Regularization in NNs</em></li>
      <li><em>5.1.4 Loss Functions (Cross-Entropy, MSE, Focal)</em></li>
      <li><em>5.1.5 Code Example: MNIST MLP</em></li>
    </ul>
  </details></li>

  <!-- 5.2 Architectures -->
  <li><details><summary>5.2 Deep Learning Architectures</summary>
    <ul>
      <li><em>5.2.1 Convolutional Neural Networks (CNNs)</em></li>
      <li><em>5.2.2 Recurrent Neural Networks (RNNs) and Sequence Models</em></li>
      <li><em>5.2.3 Attention Mechanism & Transformers</em></li>
    </ul>
  </details></li>

  <!-- 5.3 Generative DL -->
  <li><details><summary>5.3 Generative Deep Learning</summary>
    <ul>
      <li><em>5.3.1 Autoencoders</em></li>
      <li><em>5.3.2 Variational Autoencoders (VAE)</em></li>
      <li><em>5.3.3 Generative Adversarial Networks (GANs)</em></li>
    </ul>
  </details></li>

  <!-- 5.4 Graph ML -->
  <li><em>5.4 Graph Machine Learning</em></li>

</ul>
</details>

<!-- 6. Specialized -->
<details>
<summary>6. Specialized Topics in Data Science <span class="status-badge status-upcoming">Coming</span></summary>
<ul>
  <li><em>6.1 Recommender Systems</em></li>
  <li><em>6.2 Time-Series & Forecasting</em></li>
  <li><em>6.3 Bayesian & Probabilistic ML</em></li>
  <li><em>6.4 Natural Language Processing</em></li>
  <li><em>6.5 Computer Vision</em></li>
  <li><em>6.6 Reinforcement Learning & Bandits</em></li>
</ul>
</details>

<!-- 7. Production -->
<details>
<summary>7. Production Deployment & Monitoring <span class="status-badge status-upcoming">Coming</span></summary>
<ul>
  <li><em>7.1 From Prototype to Production</em></li>
  <li><em>7.2 Pipelines & Serialization</em></li>
  <li><em>7.3 Deployment Strategies</em></li>
  <li><em>7.4 Model Serving</em></li>
  <li><em>7.5 Monitoring & Maintenance</em></li>
  <li><em>7.6 Scalability & Optimization</em></li>
  <li><em>7.7 Infrastructure Challenges</em></li>
</ul>
</details>

<!-- 8. Case Studies -->
<details>
<summary>8. Applied ML in Practice & Case Studies <span class="status-badge status-upcoming">Coming</span></summary>
<ul>
  <li><em>8.1 End-to-End Project Workflow</em></li>
  <li><em>8.2 Healthcare — Disease Risk Prediction</em></li>
  <li><em>8.3 Finance — Fraud Detection</em></li>
  <li><em>8.4 E-commerce — Recommendation System</em></li>
  <li><em>8.5 Best Practices & Lessons Learned</em></li>
</ul>
</details>

<!-- 9. Responsible AI -->
<details>
<summary>9. Responsible AI, Ethics, Privacy & Governance <span class="status-badge status-upcoming">Coming</span></summary>
<ul>
  <li><em>9.1 Ethical Considerations & Bias</em></li>
  <li><em>9.2 Transparency & Explainability</em></li>
  <li><em>9.3 Accountability</em></li>
  <li><em>9.4 Privacy & Data Protection</em></li>
  <li><em>9.5 Security of ML Systems</em></li>
</ul>
</details>

<!-- 10. Emerging Topics -->
<details>
<summary>10. Emerging & Advanced Topics <span class="status-badge status-upcoming">Coming</span></summary>
<ul>
  <li><em>10.1 Transfer Learning</em></li>
  <li><em>10.2 Federated & Privacy-Preserving ML</em></li>
  <li><em>10.3 Automated Machine Learning (AutoML)</em></li>
</ul>
</details>

</div>

<div class="roadmap-footer">
  <p>
    Looking for strict chronology? Head to the <a href="{{ site.baseurl }}/blog/">full blog index</a>.<br>
    This roadmap updates automatically as new posts go live.
  </p>
</div>