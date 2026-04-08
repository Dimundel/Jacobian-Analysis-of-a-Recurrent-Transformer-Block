# Jacobian Analysis of a Recurrent Transformer Block

[![Venue: AINL 2026](https://img.shields.io/badge/Venue-AINL%202026-blue.svg)](link_to_paper_if_available)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)

Official repository for the paper **"Jacobian Analysis of a Recurrent Transformer Block"** (Accepted at AINL 2026).

This repository contains the full source code to reproduce the theoretical bounds validation, asymptotic behavior analysis, and contraction dynamics experiments presented in the paper.

## 📌 Abstract

*Recurrent Transformers are a parameter-efficient approach for modeling deep computations, but their deployment in iterative settings requires controlled and stable dynamics. We
study the Jacobian dynamics of a full Transformer block with Multi-Head Self-Attention
(MSA), Feed-Forward Networks (FFN), and RMS normalization. We derive an explicit
upper bound on the spectral norm of the full-step Jacobian, showing that RMSNorm acts as a
multiplicative damping factor that prevents gradient explosion. Under mild assumptions, the
Jacobian norm remains O(1) in the large-step regime. Our theoretical results, validated on
synthetic and CIFAR-10 data, provide insight into stability properties of recurrent models.*

## 🚀 Quickstart & Reproducibility

### 1. Install Dependencies

Ensure you have `uv` installed, then clone the repo and sync the environment:

```bash
git clone https://github.com/YOUR_USERNAME/jacobian_experiment.git
cd jacobian_experiment
uv sync
```

### 2. Running Experiments

#### Experiment 1: Theoretical Bound Validation (Theorem 3.1)

```bash
uv run python -m scripts.exp1_bounds
```

#### Experiment 2: Asymptotic Behavior Analysis (Theorem 3.2)

```bash
uv run python -m scripts.exp2_asymptotic
```

#### Experiment 3: Contraction Dynamics on Synthetic & CIFAR-10

```bash
uv run python -m scripts.exp3_contraction
```
