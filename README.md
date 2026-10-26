# Hidden Markov Model (HMM) Simulation & Algorithmic Suite

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Web Standards](https://img.shields.io/badge/HTML5%20%2F%20ES6-Vanilla-orange.svg?logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-brightgreen.svg?logo=github)](https://thevinit.github.io/ml_algo_simulation/)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(5%2F5)-success.svg)](#-unit-tests--verification)

> **An interactive visual simulator and computational reference implementation addressing the fundamental evaluation and decoding problems of Hidden Markov Models (HMMs).**

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Live Interactive Simulator](#-live-interactive-simulator)
- [Theoretical & Mathematical Framework](#-theoretical--mathematical-framework)
  - [Formal Model Definition](#formal-model-definition)
  - [Problem 1: Evaluation via Forward Algorithm](#problem-1-evaluation-via-forward-algorithm)
  - [Problem 1b: Smoothing & Likelihood via Backward Algorithm](#problem-1b-smoothing--likelihood-via-backward-algorithm)
  - [Problem 2: Decoding via Viterbi Algorithm](#problem-2-decoding-via-viterbi-algorithm)
  - [Theoretical Invariant: Path vs. Total Probability](#theoretical-invariant-path-vs-total-probability)
- [Python Reference Implementation](#-python-reference-implementation)
  - [Architecture & API](#architecture--api)
  - [Running the Benchmarks](#running-the-benchmarks)
  - [Unit Tests & Verification](#unit-tests--verification)
- [Project Directory Structure](#-project-directory-structure)
- [Getting Started Locally](#-getting-started-locally)
- [GitHub Pages Deployment Guide](#-github-pages-deployment-guide)
- [References](#-references)
- [License & Author](#-license--author)

---

## 🌟 Overview

Hidden Markov Models (HMMs) are doubly stochastic processes widely utilized in machine learning, natural language processing, speech recognition, computational biology (gene finding), and financial time-series forecasting.

This repository provides a **two-fold contribution**:
1. **Interactive Visual Trellis Simulator (`index.html`)**: A browser-native, zero-dependency educational tool that animates Forward evaluation and Viterbi decoding across custom state lattices with step-by-step arithmetic breakdowns.
2. **Algorithmic Python Engine (`python/`)**: A clean, modular, and fully tested reference implementation featuring the **Forward**, **Backward**, and **Viterbi** algorithms, complete with stochastic matrix validations and automated unit tests.

---

## ✨ Key Features

- **Interactive Trellis Lattice (SVG)**: Visualizes hidden state dynamics over discrete time steps $t \in \{1, \dots, T\}$ with animated transitions, state activations, and color-coded path backpointers.
- **Dual Problem Solvers**:
  - **Forward Algorithm**: Aggregates all possible precursor paths using sum-product trellis recurrence to calculate total likelihood $P(O \mid \lambda)$.
  - **Viterbi Algorithm**: Computes maximal subpaths and backpointers ($\psi$) using max-product recurrence, tracing the globally optimal state sequence $Q^*$.
- **Real-Time Stochastic Verification**: Interactive matrix editor validates that initial probability vectors ($\pi$), state transition matrices ($A$), and emission matrices ($B$) adhere strictly to stochastic row-sum constraints ($\sum = 1.0$).
- **Benchmark Presets**: Instant one-click loading of canonical datasets:
  - **Weather Problem**: Rainy / Sunny hidden weather states generating Walk / Shop / Clean activities.
  - **Medical Diagnosis**: Healthy / Fever hidden physiological states generating Normal / Cold / Dizzy diagnostic symptoms.
- **Zero External Dependencies**:
  - Web Visualizer runs natively in any modern browser without npm or bundler overhead.
  - Python engine runs on standard Python 3 (zero third-party dependencies required).

---

## 🖥️ Live Interactive Simulator

Experience the interactive simulator directly in your browser:

🔗 **[Launch Live Web Demo](https://thevinit.github.io/ml_algo_simulation/)**

### Interactive Controls
- **Play / Pause / Step Controls**: Animate through the trellis at variable speeds (Slow, Normal, Fast).
- **Detail Granularity**: Toggle between detailed 3-step arithmetic explanations (incoming product $\rightarrow$ summation/argmax $\rightarrow$ emission scaling) and accelerated single-step transitions.
- **Keyboard Navigation**:
  - <kbd>&rarr;</kbd> : Next step
  - <kbd>&larr;</kbd> : Previous step
  - <kbd>Space</kbd> : Play / Pause

---

## 📐 Theoretical & Mathematical Framework

### Formal Model Definition

A discrete Hidden Markov Model is completely characterized by the 5-tuple:

$$\lambda = (S, V, A, B, \pi)$$

Where:
- $S = \{S_1, S_2, \dots, S_N\}$ is the set of $N$ hidden states.
- $V = \{v_1, v_2, \dots, v_M\}$ is the discrete alphabet of $M$ observable symbols.
- $A \in \mathbb{R}^{N \times N}$ is the state transition probability matrix:
  $$A_{ij} = P(q_t = S_j \mid q_{t-1} = S_i), \quad 1 \le i, j \le N$$
  $$\sum_{j=1}^N A_{ij} = 1 \quad \forall i$$
- $B \in \mathbb{R}^{N \times M}$ is the observation emission probability matrix:
  $$B_j(k) = P(O_t = v_k \mid q_t = S_j), \quad 1 \le j \le N, \; 1 \le k \le M$$
  $$\sum_{k=1}^M B_j(k) = 1 \quad \forall j$$
- $\pi \in \mathbb{R}^N$ is the initial state probability distribution vector:
  $$\pi_i = P(q_1 = S_i), \quad 1 \le i \le N, \quad \sum_{i=1}^N \pi_i = 1$$

---

### Problem 1: Evaluation via Forward Algorithm

Given model $\lambda$ and observation sequence $O = (o_1, o_2, \dots, o_T)$, determine the total probability of observation $P(O \mid \lambda)$.

Direct path summation is computationally intractable with complexity $\mathcal{O}(N^T)$. The Forward Algorithm solves this using dynamic programming in $\mathcal{O}(N^2 T)$ time:

1. **Forward Variable Definition**:
   $$\alpha_t(j) = P(o_1, o_2, \dots, o_t, q_t = S_j \mid \lambda)$$

2. **Initialization ($t = 1$)**:
   $$\alpha_1(i) = \pi_i \cdot B_i(o_1), \quad 1 \le i \le N$$

3. **Induction ($t = 2, 3, \dots, T$)**:
   $$\alpha_t(j) = \left[ \sum_{i=1}^N \alpha_{t-1}(i) A_{ij} \right] \cdot B_j(o_t), \quad 1 \le j \le N$$

4. **Termination**:
   $$P(O \mid \lambda) = \sum_{i=1}^N \alpha_T(i)$$

---

### Problem 1b: Smoothing & Likelihood via Backward Algorithm

The Backward Algorithm computes the conditional probability of future observations given the hidden state at step $t$, running **right-to-left** across the lattice:

1. **Backward Variable Definition**:
   $$\beta_t(i) = P(o_{t+1}, o_{t+2}, \dots, o_T \mid q_t = S_i, \lambda)$$

2. **Initialization ($t = T$)**:
   $$\beta_T(i) = 1.0, \quad 1 \le i \le N$$

3. **Induction ($t = T-1, T-2, \dots, 1$)**:
   $$\beta_t(i) = \sum_{j=1}^N A_{ij} \cdot B_j(o_{t+1}) \cdot \beta_{t+1}(j), \quad 1 \le i \le N$$

4. **Termination ($t = 1$)**:
   $$P(O \mid \lambda) = \sum_{i=1}^N \pi_i \cdot B_i(o_1) \cdot \beta_1(i)$$

> **Cross-Validation Invariant**: For any discrete time step $t \in \{1, \dots, T\}$, the total likelihood is invariant:
> $$P(O \mid \lambda) = \sum_{i=1}^N \alpha_t(i) \beta_t(i) = \sum_{i=1}^N \alpha_T(i) = \sum_{i=1}^N \pi_i B_i(o_1) \beta_1(i)$$

---

### Problem 2: Decoding via Viterbi Algorithm

Find the single optimal state sequence $Q^* = (q_1^*, q_2^*, \dots, q_T^*)$ that maximizes the joint conditional probability $P(Q, O \mid \lambda)$:

1. **Viterbi Trellis Variable Definition**:
   $$\delta_t(j) = \max_{q_1, \dots, q_{t-1}} P(q_1, \dots, q_{t-1}, q_t = S_j, o_1, \dots, o_t \mid \lambda)$$

2. **Initialization ($t = 1$)**:
   $$\delta_1(i) = \pi_i \cdot B_i(o_1), \quad 1 \le i \le N$$
   $$\psi_1(i) = 0$$

3. **Recursion ($t = 2, 3, \dots, T$)**:
   $$\delta_t(j) = \left[ \max_{1 \le i \le N} \big(\delta_{t-1}(i) A_{ij}\big) \right] \cdot B_j(o_t), \quad 1 \le j \le N$$
   $$\psi_t(j) = \arg\max_{1 \le i \le N} \big(\delta_{t-1}(i) A_{ij}\big), \quad 1 \le j \le N$$

4. **Termination**:
   $$P^* = \max_{1 \le i \le N} \delta_T(i)$$
   $$q_T^* = \arg\max_{1 \le i \le N} \delta_T(i)$$

5. **Path Backtracking ($t = T-1, T-2, \dots, 1$)**:
   $$q_t^* = \psi_{t+1}(q_{t+1}^*)$$

---

### Theoretical Invariant: Path vs. Total Probability

Because the Viterbi metric identifies the probability of the single maximal path through the state lattice, while the Forward metric integrates over the entire probability mass of all possible paths, the following strict inequality holds:

$$P^* = \max_Q P(Q, O \mid \lambda) \;\le\; P(O \mid \lambda) = \sum_Q P(Q, O \mid \lambda)$$

Both the interactive simulator and the Python engine assert and verify this inequality at runtime.

---

## 🐍 Python Reference Implementation

### Architecture & API

The Python engine is contained inside `python/hmm.py` and provides a clean, object-oriented interface:

```python
from hmm import HiddenMarkovModel

# Initialize model parameters
model = HiddenMarkovModel(
    states=["Rainy", "Sunny"],
    symbols=["Walk", "Shop", "Clean"],
    A=[[0.7, 0.3], [0.4, 0.6]],
    B=[[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]],
    pi=[0.6, 0.4],
)

sequence = ["Walk", "Shop", "Clean"]

# 1. Forward Algorithm (Evaluation)
alpha, p_total = model.forward(sequence)
print(f"Total Sequence Likelihood P(O|lambda): {p_total:.6f}")

# 2. Viterbi Algorithm (Optimal Decoding)
best_path, p_best, delta, psi = model.viterbi(sequence)
print(f"Optimal Sequence Q*: {' -> '.join(best_path)} (P* = {p_best:.6f})")
```

### Running the Benchmarks

To execute the benchmark suite and view structured tabular terminal output:

```bash
python python/examples.py
```
*(On Windows systems where the Python launcher is configured: `py python/examples.py`)*

#### Sample Benchmark Output:
```text
======================================================================
BENCHMARK: Weather & Activity Model
Observation Sequence: Walk -> Shop -> Clean
======================================================================

--- Forward Trellis Variables (alpha) ---
     State | t=1 (Walk) | t=2 (Shop) | t=3 (Clean)
--------------------------------------------------
     Rainy |    0.06000 |    0.05520 |    0.02904
     Sunny |    0.24000 |    0.04860 |    0.00457

=> Total Sequence Likelihood P(O | lambda) = 3.361200e-02 (0.033612)
=> Backward Algorithm Likelihood Verification = 3.361200e-02 (0.033612)

--- Viterbi Trellis Scores (delta) ---
     State | t=1 (Walk) | t=2 (Shop) | t=3 (Clean)
--------------------------------------------------
     Rainy |    0.06000 |    0.03840 |    0.01344
     Sunny |    0.24000 |    0.04320 |    0.00259

=> Optimal Hidden State Path (Viterbi) Q* = Sunny -> Rainy -> Rainy
=> Optimal Path Probability P* = 1.344000e-02 (0.013440)
=> Inequality Check: P* (0.013440) <= P(O|lambda) (0.033612) : True
```

### Unit Tests & Verification

The test suite in `python/test_hmm.py` verifies mathematical and stochastic invariants:
- Stochastic row-sum constraints on matrices $\pi, A, B$.
- Numerical accuracy of alpha and delta variables against analytical solutions.
- Forward vs. Backward likelihood equivalence ($P_{\text{Forward}} = P_{\text{Backward}}$).
- Backpointer trajectory integrity and boundary conditions ($T=1$).

Execute the test suite via:
```bash
python -m unittest python/test_hmm.py
```

---

## 📂 Project Directory Structure

```text
ml_algo_simulation/
├── index.html            # Main web application & interactive Trellis visualizer
├── python/               # Algorithmic Python implementation
│   ├── hmm.py            # Core HiddenMarkovModel class (Forward, Backward, Viterbi)
│   ├── examples.py       # Benchmark demonstration script (Weather & Diagnosis)
│   └── test_hmm.py       # Automated unit test suite
├── .gitignore            # Git exclusion rules for clean repository state
├── LICENSE               # MIT Open Source License
└── README.md             # Comprehensive academic documentation
```

---

## 🚀 Getting Started Locally

### Option A: Interactive Web Visualizer
No compilation or installation is required:
1. Clone the repository:
   ```bash
   git clone https://github.com/TheVinit/ml_algo_simulation.git
   cd ml_algo_simulation
   ```
2. Open `index.html` directly in any web browser, or serve it using Python's built-in static server:
   ```bash
   python -m http.server 8000
   ```
3. Navigate to `http://localhost:8000`.

### Option B: Python Algorithmic Engine
1. Verify Python 3.9+ is installed:
   ```bash
   python --version
   ```
2. Run unit tests:
   ```bash
   python -m unittest python/test_hmm.py
   ```

---

## 🌐 GitHub Pages Deployment Guide

To host the visual simulator online on GitHub Pages:
1. Navigate to your repository on GitHub: `https://github.com/TheVinit/ml_algo_simulation`.
2. Click **Settings** $\rightarrow$ **Pages** (under the "Code and automation" sidebar).
3. Under **Build and deployment** $\rightarrow$ **Branch**:
   - Source: Select **Deploy from a branch**.
   - Branch: Select `main` and folder `/ (root)`.
4. Click **Save**.
5. Within 1–2 minutes, your live simulator will be active at:
   `https://thevinit.github.io/ml_algo_simulation/`

---

## 📚 References

1. **Rabiner, L. R. (1989)**. *A tutorial on hidden Markov models and selected applications in speech recognition*. Proceedings of the IEEE, 77(2), 257–286.
2. **Viterbi, A. (1967)**. *Error bounds for convolutional codes and an asymptotically optimum decoding algorithm*. IEEE Transactions on Information Theory, 13(2), 260–269.
3. **Baum, L. E., & Petrie, T. (1966)**. *Statistical inference for probabilistic functions of finite state Markov chains*. The Annals of Mathematical Statistics, 37(6), 1554–1563.
4. **Jurafsky, D., & Martin, J. H. (2023)**. *Speech and Language Processing (3rd ed. draft)*. Chapter A: Hidden Markov Models.

---

## 📄 License & Author

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for complete terms.

**Author**: Vinit ([GitHub Profile](https://github.com/TheVinit))  
**Repository**: [TheVinit/ml_algo_simulation](https://github.com/TheVinit/ml_algo_simulation)
