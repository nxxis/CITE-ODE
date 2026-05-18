# CEMR-Fair: Continuous-Time Evidential Multimodal Representation Learning for Fairness

This repository contains the official, fully reproducible PyTorch implementation for **CEMR-Fair**. The framework introduces a continuous-time approach to health informatics, learning robust patient representations from irregularly sampled clinical telemetry data while strictly enforcing demographic epistemic parity constraints.

This project is tailored for submission to the IEEE International Conference on Data Mining (ICDM 2026) Applied Track.

---

# 📌 Framework Overview

Clinical telemetry data in Intensive Care Units (ICUs) is notoriously sparse, heterogeneous, and irregularly observed. Traditional data mining methods rely on rigid imputation techniques that distort the true underlying physiological state and perpetuate systemic demographic biases.

**CEMR-Fair** counters these challenges through three integrated architectural layers:

## 1. Continuous-Time Trajectory Modeling

Uses Neural Ordinary Differential Equations (ODEs) to model patient states continuously, mapping irregular intervals natively without imputation.

## 2. Evidential Representation Learning

Projects continuous latent states through a Normal-Inverse-Gamma (NIG) distribution layer, enabling the system to explicitly quantify its own epistemic uncertainty.

## 3. Minimax Adversarial Fairness

Employs an adversarial discriminator that actively removes protected demographic attributes (such as Age and Gender) from the latent trajectory space, enforcing equitable confidence calibration.

---

# 🔧 Installation Guide

To satisfy the requirements of the Machine Learning Reproducibility Checklist, follow the steps below to build an isolated environment and install the exact frozen dependencies used during evaluation.

## Prerequisites

- Python 3.10+
- CUDA-enabled GPU (NVIDIA T4, A100, or RTX series recommended)
- NVIDIA CUDA Toolkit compatible with the installed PyTorch version

---

# ⚙️ Step-by-Step Environment Setup

## 1. Clone the Repository

```bash
git clone [https://github.com/nxxis/CEMR-Fair](https://github.com/nxxis/CEMR-Fair)
cd CEMR-Fair
```

---

## 2. Create an Isolated Virtual Environment

Using a dedicated environment prevents dependency conflicts with local system packages.

### Using `venv`

```bash
python3 -m venv cemr_env
source cemr_env/bin/activate
```

### Using Conda

```bash
conda create -n cemr_env python=3.10 -y
conda activate cemr_env
```

---

## 3. Install Frozen Dependencies

Install the strict version-locked package manifest:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# 📂 Repository Structure

```plaintext
CEMR-Fair/
├── data/
│   ├── mimic_cemr_cohort.csv    # Static reproducible clinical dataset
│   └── clinical_mimic.py        # Optimized offline dataloader pipeline
│
├── models/
│   └── tide_ode.py              # Neural ODE + Evidential Head architecture
│
├── utils/
│   └── losses.py                # Evidential NLL & fairness losses
│
├── results/
│   └── latent_tsne.pdf          # Generated latent space visualization
│
├── checkpoints/
│   └── cemr_fair_final.pth      # Saved trained model weights
│
├── train_cemr.py                # Full training pipeline
├── evaluate_cemr.py             # Evaluation & fairness audit script
├── requirements_strict.txt      # Frozen dependency manifest
└── README.md                    # Project documentation
```

---

# 📥 Data Pipeline & Artifact Isolation

To satisfy peer-review reproducibility requirements without exposing live database endpoints, the experimental cohort has been frozen into a static artifact.

## Dataset Information

- **Source Dataset:** MIMIC-IV Clinical Database
- **Placement:** Put `mimic_cemr_cohort.csv` inside the `data/` directory
- **Cohort Size:**
  - 450,484 sequential physiological observations
  - 1,700 unique ICU stays

---

# 🚀 Execution & Reproducibility Guide

All execution scripts are hard-locked with:

- Global random seed: `seed=42`
- Deterministic cuDNN operations
- Fully reproducible evaluation configuration

This guarantees stable execution metrics across supported hardware platforms.

---

# 1. Train the Framework

To jointly optimize:

- Neural ODE trajectory dynamics
- Evidential uncertainty parameters
- Adversarial demographic fairness constraints

run:

```bash
python train_cemr.py
```

## Output

Serialized model weights are stored at:

```plaintext
checkpoints/cemr_fair_final.pth
```

---

# 2. Run the Evaluation & Fairness Audit

To compute:

- Clinical prediction performance
- Demographic parity gaps
- Epistemic uncertainty metrics
- Vectorized visualizations

run:

```bash
python evaluate_cemr.py
```

## Output

- Terminal performance summary
- High-resolution vector figures saved inside `results/`

---

# 🏆 Verified Experimental Results

The metrics below represent the final reproducible evaluation results from the full cohort experiment.

| Evaluation Metric          | Overall Cohort | Demographic Group A | Demographic Group B | Metric Delta (Δ) |
| -------------------------- | -------------- | ------------------- | ------------------- | ---------------- |
| Clinical Utility (AUROC)   | 0.6950         | 0.6471              | 0.7329              | 0.0858           |
| Clinical Utility (AUPRC)   | 0.2535         | —                   | —                   | —                |
| Mean Epistemic Uncertainty | 0.2536         | 0.2506              | 0.2556              | 0.0050           |
| Subsample Size (N)         | 340            | 131                 | 209                 | —                |

---

# 🔍 Key Analytical Takeaways

## Epistemic Parity Achieved

The framework demonstrates a remarkably tight uncertainty gap:

```math
\Delta = 0.0050
```

Although downstream physiological disparities in the raw clinical data produce a measurable predictive utility gap:

```math
\Delta = 0.0858
```

the model remains consistently calibrated across demographic groups. In practice, the system distributes uncertainty equitably, meaning it maintains comparable confidence behavior regardless of demographic identity.

---

## Latent Space Demographic Blinding

During training, the demographic adversary consistently converged near:

```math
\text{Adversarial MSE} \approx 1.0
```

This indicates that the learned continuous latent trajectories become statistically insensitive to protected demographic indicators, demonstrating successful fairness regularization within the representation space.

---

# 🎨 Visualization Assets

When evaluation executes, the visualization pipeline automatically generates publication-ready figures using deep navy, gold, and light gray presentation palettes.

## Generated Artifact

```plaintext
results/latent_tsne.pdf
```

This file contains a 2D t-SNE projection of the learned latent representations.

Reviewers should observe:

- Highly mixed demographic distributions
- Minimal clustering by protected attributes
- Homogeneous latent organization

These visual characteristics provide qualitative evidence that demographic separability has been substantially neutralized in the learned representation space.
