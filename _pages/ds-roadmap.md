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
<li><details><summary>3.1 Regression — Linear, Regularized, Robust & Advanced Parametric Models</summary>
  <ul>
    <li><a href="{{ site.baseurl }}/blog/2025/ds007-heartofml-regression1/">3.1.1 Linear Regression (OLS): Foundations, Assumptions, and Interpretation</a></li>
    <li><a href="{{ site.baseurl }}/blog/2025/ds008-heartofml-regression2/">3.1.2 Regularized Linear Models: Ridge, Lasso, Elastic Net Explained</a></li>
    <li><a href="{{ site.baseurl }}/blog/2025/ds009-heartofml-regression3/">3.1.3 Generalized Linear Models (GLMs): Logistic, Poisson & Beyond</a></li>
    <li><a href="{{ site.baseurl }}/blog/2025/ds010-heartofml-regression4/">3.1.4 Evaluation Metrics for Regression: MAE, RMSE, R², and More</a></li>
    <li><a href="{{ site.baseurl }}/blog/2025/ds011-heartofml-regression5/">3.1.5 Robust Regression: Huber, RANSAC, and Quantile Methods</a></li>
    <li><a href="{{ site.baseurl }}/blog/2025/ds012-heartofml-regression6/">3.1.6 Capturing Non-Linearity: Polynomial, Splines & Basis Expansions</a></li>
    <li><a href="{{ site.baseurl }}/blog/2025/ds013-heartofml-regression7/">3.1.7 Advanced Parametric Regression: Bayesian Models, Mixed Effects, GPs</a></li>
    <li><a href="{{ site.baseurl }}/blog/2025/ds014-heartofml-regression8/">3.1.8 Optimization Techniques for Regression: Gradient Descent, Normal Equations, SGD and Beyond</a></li>
  </ul>
</details></li>


 <!-- 3.2 Classification -->
<li><details><summary>3.2 Classification — Foundations, Algorithms, Evaluation & Deployment</summary>
  <ul>
    <li><em>3.2.1 Foundations of Classification: Problem Framing, Data Prep & Label Encoding</em> <span class="status-badge status-upcoming">Soon</span></li>
    <li><em>3.2.2 Classical Algorithms Part 1: Logistic Regression & Naïve Bayes Classifiers</em> <span class="status-badge status-upcoming">Soon</span></li>
    <li><em>3.2.3 Classical Algorithms Part 2: k-Nearest Neighbors (k-NN) & Support Vector Machines (SVM)</em> <span class="status-badge status-upcoming">Soon</span></li>
    <li><em>3.2.4 Advanced Models Part 1: Softmax Regression & Linear Discriminant Analysis (LDA)</em> <span class="status-badge status-upcoming">Soon</span></li>
    <li><em>3.2.5 Advanced Models Part 2: Quadratic Discriminant Analysis (QDA) & Probabilistic Graphical Models (PGMs)</em> <span class="status-badge status-upcoming">Soon</span></li>
    <li><em>3.2.6 Model Evaluation & Diagnostics: Confusion Matrix, ROC-AUC, PR Curves</em> <span class="status-badge status-upcoming">Soon</span></li>
    <li><em>3.2.7 Optimization & Training: Loss Functions, Gradient Descent, Regularization</em> <span class="status-badge status-upcoming">Soon</span></li>
    <li><em>3.2.8 Real-World Deployment & Interpretability: SHAP, LIME & Model Explainability</em> <span class="status-badge status-upcoming">Soon</span></li>
  </ul>
</details></li>

<!-- 3.3 Tree/Ensemble -->
<li><details><summary>3.3 Tree-Based & Ensemble Learning — Decision Trees, Bagging, Boosting</summary>
  <ul>
    <li><em>3.3.1 Decision Trees Explained: ID3, CART, Gini Index, Entropy</em> <span class="status-badge status-upcoming">Soon</span></li>
    <li><em>3.3.2 Random Forests for Regression and Classification: Ensemble Bagging Techniques</em> <span class="status-badge status-upcoming">Soon</span></li>
    <li><em>3.3.3 Boosting Algorithms: AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost</em> <span class="status-badge status-upcoming">Soon</span></li>
  </ul>
</details></li>

<!-- 3.4 Unsupervised -->
<li><details><summary>3.4 Unsupervised Learning — Clustering, Dimensionality Reduction, Anomaly Detection</summary>
  <ul>
    <li><em>3.4.1 Clustering Algorithms: k-Means, DBSCAN, Hierarchical Clustering</em> <span class="status-badge status-upcoming">Soon</span></li>
    <li><em>3.4.2 Dimensionality Reduction & Embeddings: PCA, t-SNE, UMAP</em> <span class="status-badge status-upcoming">Soon</span></li>
    <li><em>3.4.3 Anomaly Detection Techniques: Statistical, Distance-Based, Isolation Forest</em> <span class="status-badge status-upcoming">Soon</span></li>
    <li><em>3.4.4 Association Rule Learning: Apriori, FP-Growth, Market Basket Analysis</em> <span class="status-badge status-upcoming">Soon</span></li>
    <li><em>3.4.5 Recommender Systems (Bridge Topic): Collaborative Filtering & Hybrid Models</em> <span class="status-badge status-upcoming">Soon</span></li>
  </ul>
</details></li>

<!-- 3.5 Probabilistic -->
<li><details><summary>3.5 Probabilistic & Generative Models — MLE, MAP, EM, HMMs, CRFs</summary>
  <ul>
    <li><em>3.5.1 Maximum Likelihood & MAP Estimation: Foundations of Probabilistic Modeling</em> <span class="status-badge status-upcoming">Soon</span></li>
    <li><em>3.5.2 Expectation-Maximization (EM): Latent Variables & Gaussian Mixture Models</em> <span class="status-badge status-upcoming">Soon</span></li>
    <li><em>3.5.3 Hidden Markov Models (HMM): Sequence Modeling with Forward-Backward Algorithm</em> <span class="status-badge status-upcoming">Soon</span></li>
    <li><em>3.5.4 Conditional Random Fields (CRF): Structured Prediction for NLP & Vision</em> <span class="status-badge status-upcoming">Soon</span></li>
  </ul>
</details></li>

</ul>
</details>

<!-- 4. Evaluation -->
<details>
<summary>4. Model Evaluation, Hyperparameter Tuning & Experimentation — CV, Metrics, Interpretability, A/B Testing <span class="status-badge status-upcoming">Coming</span></summary>
<ul>
  <li><em>4.1 Cross-Validation Strategies: k-Fold, Time-Series CV, and Nested Validation</em></li>
  <li><em>4.2 Metric Selection: Evaluation Metrics for Classification, Regression, Ranking, and More</em></li>
  <li><em>4.3 Hyperparameter Optimization: Grid Search, Bayesian Tuning, AutoML</em></li>
  <li><em>4.4 Model Interpretability: SHAP, LIME, PDP, Feature Attribution</em></li>
  <li><em>4.5 A/B Testing & Experimentation: Design, Power Analysis, CUPED</em></li>
  <li><em>4.6 Fairness & Bias Mitigation: Metrics, Algorithms, and Case Studies</em></li>
</ul>
</details>

<!-- 5. Deep Learning -->
<details>
<summary>5. Deep Learning — Architectures, Generative Models, and Graph ML <span class="status-badge status-upcoming">Coming</span></summary>
<ul>

  <!-- 5.1 Fundamentals -->
  <li><details><summary>5.1 Neural Network Fundamentals — Architecture, Training & Regularization</summary>
    <ul>
      <li><em>5.1.1 Perceptron & Feedforward Neural Networks: Basics of Deep Learning</em></li>
      <li><em>5.1.2 Training Neural Networks: Backpropagation, Optimizers, Initialization</em></li>
      <li><em>5.1.3 Regularization in Neural Networks: Dropout, Weight Decay, BatchNorm</em></li>
      <li><em>5.1.4 Loss Functions: Cross-Entropy, MSE, Focal Loss and Use Cases</em></li>
      <li><em>5.1.5 Code Walkthrough: MNIST Classification with MLP (PyTorch/Keras)</em></li>
    </ul>
  </details></li>

  <!-- 5.2 Architectures -->
  <li><details><summary>5.2 Deep Learning Architectures — CNNs, RNNs, and Transformers</summary>
    <ul>
      <li><em>5.2.1 Convolutional Neural Networks (CNNs): Vision Models & Feature Hierarchies</em></li>
      <li><em>5.2.2 Recurrent Neural Networks (RNNs) and Sequence Models: LSTM, GRU, Bidirectional Models</em></li>
      <li><em>5.2.3 Attention Mechanisms & Transformers: Self-Attention, BERT, GPT</em></li>
    </ul>
  </details></li>

  <!-- 5.3 Generative DL -->
  <li><details><summary>5.3 Generative Deep Learning — Autoencoders, VAEs, and GANs</summary>
    <ul>
      <li><em>5.3.1 Autoencoders: Dimensionality Reduction, Denoising, Anomaly Detection</em></li>
      <li><em>5.3.2 Variational Autoencoders (VAE): Latent Space Learning & Sample Generation</em></li>
      <li><em>5.3.3 Generative Adversarial Networks (GANs): Image Synthesis & Applications</em></li>
    </ul>
  </details></li>

  <!-- 5.4 Graph ML -->
  <li><em>5.4 Graph Machine Learning: GCNs, GraphSAGE, GAT for Networked Data</em></li>

</ul>
</details>

<!-- 6. Specialized -->
<details>
<summary>6. Specialized Topics in Data Science — Recommenders, Time Series, NLP, Vision, RL <span class="status-badge status-upcoming">Coming</span></summary>
<ul>
  <li><em>6.1 Recommender Systems: Collaborative Filtering, Hybrid Models & Ranking Metrics</em></li>
  <li><em>6.2 Time-Series Forecasting: ARIMA, XGBoost, LSTM & Transformer-Based Models</em></li>
  <li><em>6.3 Bayesian & Probabilistic ML: MCMC, Variational Inference, Uncertainty Modeling</em></li>
  <li><em>6.4 Natural Language Processing (NLP): Embeddings, Transformers & Classification</em></li>
  <li><em>6.5 Computer Vision: Detection, Segmentation & Transfer Learning</em></li>
  <li><em>6.6 Reinforcement Learning & Bandits: Q-Learning, PPO, and Exploration Strategies</em></li>
</ul>
</details>

<!-- 7. Production -->
<details>
<summary>7. Production Deployment & Monitoring — Pipelines, Drift, Serving, CI/CD <span class="status-badge status-upcoming">Coming</span></summary>
<ul>
  <li><em>7.1 From Prototype to Production: Packaging & Testing ML Models</em></li>
  <li><em>7.2 Pipelines & Serialization: MLflow, ONNX, Scikit-Learn Pipelines</em></li>
  <li><em>7.3 Deployment Strategies: REST APIs, Docker, Real-Time vs Batch Inference</em></li>
  <li><em>7.4 Model Serving: FastAPI, TorchServe, GCP Vertex, AWS SageMaker</em></li>
  <li><em>7.5 Monitoring & Maintenance: Drift Detection, Retraining, Shadow Testing</em></li>
  <li><em>7.6 Scalability & Optimization: Distributed Inference, Quantization, ONNX Runtime</em></li>
  <li><em>7.7 Infrastructure Challenges: DevOps, Cloud Platforms, MLOps Tooling</em></li>
</ul>
</details>

<!-- 8. Case Studies -->
<details>
<summary>8. Applied ML in Practice — End-to-End Projects in Healthcare, Finance & E-commerce <span class="status-badge status-upcoming">Coming</span></summary>
<ul>
  <li><em>8.1 End-to-End Project Workflow: From Problem Framing to Deployment</em></li>
  <li><em>8.2 Case Study: Healthcare — Disease Risk Prediction Using EHR & Clinical Features</em></li>
  <li><em>8.3 Case Study: Finance — Real-Time Credit Card Fraud Detection System</em></li>
  <li><em>8.4 Case Study: E-commerce — Personalized Product Recommender</em></li>
  <li><em>8.5 Best Practices & Lessons Learned: Simplicity, Interpretability & Domain Knowledge</em></li>
</ul>
</details>

<!-- 9. Responsible AI -->
<details>
<summary>9. Responsible AI — Ethics, Fairness, Privacy, Governance & Safety <span class="status-badge status-upcoming">Coming</span></summary>
<ul>
  <li><em>9.1 Ethical Considerations & Bias: Algorithmic Fairness & Group-Level Equity</em></li>
  <li><em>9.2 Transparency & Explainability: Model Cards, SHAP, LIME</em></li>
  <li><em>9.3 Accountability: Human Oversight, Review Loops & Fail-Safe Systems</em></li>
  <li><em>9.4 Privacy & Data Protection: GDPR, Differential Privacy, Federated Learning</em></li>
  <li><em>9.5 Security of ML Systems: Adversarial Attacks, Data Poisoning & Model Inversion</em></li>
</ul>
</details>

<!-- 10. Emerging Topics -->
<details>
<summary>10. Emerging & Advanced Topics — Transfer Learning, AutoML, Privacy-Preserving AI <span class="status-badge status-upcoming">Coming</span></summary>
<ul>
  <li><em>10.1 Transfer Learning: Fine-Tuning Pretrained Models like BERT & ResNet</em></li>
  <li><em>10.2 Federated & Privacy-Preserving ML: Secure Aggregation & Differential Privacy</em></li>
  <li><em>10.3 Automated Machine Learning (AutoML): Model Search, Tuning, and Deployment</em></li>
</ul>
</details>


</div>

<div class="roadmap-footer">
  <p>
    Looking for strict chronology? Head to the <a href="{{ site.baseurl }}/blogging/">full blog index</a>.<br>
    This roadmap updates automatically as new posts go live.
  </p>
</div>