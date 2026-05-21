# CITE-ODE: Continuous-Time Evidential Neural ODEs for Uncertainty-Aware ICU Risk Prediction

Official PyTorch implementation of **CITE-ODE**, a continuous-time evidential learning framework for uncertainty-aware ICU risk prediction from irregularly sampled clinical telemetry.

---

# Framework Overview

ICU telemetry data is sparse, irregularly sampled, and frequently interrupted by long contiguous gaps caused by monitor disconnects, workflow changes, and missing charting. Traditional sequence models often rely on imputation strategies that distort temporal structure and provide no principled estimate of epistemic uncertainty.

**CITE-ODE** addresses these limitations through three integrated components:

---

## 1. Continuous-Time Trajectory Modeling

A Neural Ordinary Differential Equation (**Neural ODE**) backbone models physiological trajectories continuously in time, directly handling irregular sampling without imputation.

---

## 2. Evidential Representation Learning

Latent trajectories are projected through a **Normal-Inverse-Gamma (NIG)** evidential head that estimates predictive uncertainty in a single forward pass.

Epistemic uncertainty is computed as:

$$
u = \frac{\beta}{\alpha - 1}
$$

---

## 3. Selective Prediction Framework

Predictions are ranked using epistemic uncertainty and selectively filtered at predefined coverage levels:

- 100%
- 90%
- 80%
- 70%

A stratified random control preserving class prevalence isolates the effect of uncertainty ranking from prevalence-induced calibration changes.

---

# Installation Guide

The repository is fully reproducible and follows the Machine Learning Reproducibility Checklist.

---

# Prerequisites

- Python 3.10+
- CUDA-enabled NVIDIA GPU (T4, A100, RTX series recommended)
- CUDA Toolkit compatible with installed PyTorch version

---

# Reproducibility & Environment

All experiments were executed under a tightly controlled environment to ensure reproducibility. We recommend reproducing the environment exactly.

## Supported Runtime

- Python: `3.10` (recommended)

## Key Package Versions

| Package | Version |
|---|---|
| PyTorch | `2.10.0+cu128` |
| torchdiffeq | `0.2.5` |
| NumPy | `2.0.2` |
| Pandas | `2.2.2` |
| Scikit-learn | `1.6.1` |
| Matplotlib | `3.10.0` |
| Seaborn | `0.13.2` |
| Google Cloud BigQuery | `3.41.0` |

## Quick Verification

```bash
python -c "import torch,numpy,pandas; print('torch',torch.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__)"
```

---

# Google Colab Quick Setup

The codebase is compatible with Google Colab.

## Recommended Workflow

1. Upload or clone the repository into Google Drive
2. Open a Colab notebook
3. Mount Google Drive
4. Change into the project directory
5. Install the required dependencies
6. Run training or evaluation scripts

A detailed walkthrough is provided in:

```text
COLAB_SETUP.md
```

---

# Repository Structure

```text
CITE-ODE/
├── checkpoints/
│   ├── cemr_fair_seed42.pth
│   ├── cemr_fair_seed123.pth
│   ├── cemr_fair_seed456.pth
│   ├── cemr_fair_seed789.pth
│   ├── cemr_fair_seed101112.pth
│   ├── baseline_gru_seed42.pth
│   ├── baseline_gru_seed123.pth
│   ├── baseline_gru_seed456.pth
│   ├── baseline_gru_seed789.pth
│   ├── baseline_gru_seed101112.pth
│   ├── transformer_seed42.pth
│   ├── transformer_seed123.pth
│   ├── transformer_seed456.pth
│   ├── transformer_seed789.pth
│   └── transformer_seed101112.pth
│
├── data/
│   ├── clinical_mimic.py
│   └── mimic_cemr_cohort.csv
│
├── models/
│   └── tide_ode.py
│
├── utils/
│   └── metrics.py
│
├── scripts/
│   ├── train_cemr_seed.py
│   ├── run_multiseed_train.py
│   ├── train_gru_seed.py
│   ├── run_multiseed_gru_5.py
│   ├── train_transformer_seed.py
│   ├── run_multiseed_transformer_5.py
│   ├── evaluate_multiseed.py
│   ├── evaluate_multiseed_gru_5.py
│   ├── evaluate_multiseed_transformer_5.py
│   ├── evaluate_selective_multiseed_full.py
│   ├── evaluate_multiseed_subgroups.py
│   └── generate_all_figures.py
│
├── plots/
│   ├── figure1_reliability.pdf
│   ├── figure2_selective_variance.pdf
│   └── figure3_subgroup_scatter.pdf
│
├── archive/
├── README.md
├── COLAB_SETUP.md
└── requirements.txt
```

---

# Dataset & Cohort Information

To support reproducibility without requiring live database access, the evaluation cohort is stored as an external artifact.

## Dataset Details

| Field | Value |
|---|---|
| Source Dataset | MIMIC-IV Clinical Database |
| Cohort Size | 10,000 ICU stays |
| Observation Window | 24 hours |
| Signals | 8 vital signs |
| Target | Binary in-hospital mortality |

---

# Downloading the Cohort

## Recommended Method

```bash
pip install gdown
python scripts/fetch_cohort.py
```

The dataset will be written to:

```text
data/mimic_cemr_cohort.csv
```

## Alternative Direct Download

```bash
gdown --folder "https://drive.google.com/drive/folders/1oupz5CcQIMn-16I8KFWeqlpirY0vBCxg" -O data/
```

## Verify Download

```bash
ls -lh data/mimic_cemr_cohort.csv
```

---

# Training & Evaluation

All experiments use:

- Fixed global random seeds
- Deterministic cuDNN execution
- Frozen evaluation configuration

This ensures reproducibility across supported hardware.

---

# Seeds

## CITE-ODE

```text
42, 123, 456, 789, 101112
```

## GRU Baseline

```text
42, 123, 456, 789, 101112
```

## Transformer Baseline

```text
42, 123, 456, 789, 101112
```

---

# 1. Training

## Train CITE-ODE (5 Seeds)

```bash
python scripts/run_multiseed_train.py
```

Generated checkpoints:

```text
checkpoints/cemr_fair_seed*.pth
```

---

## Train GRU Baseline (5 Seeds)

```bash
python scripts/run_multiseed_gru_5.py
```

Generated checkpoints:

```text
checkpoints/baseline_gru_seed*.pth
```

---

## Train Transformer Baseline (5 Seeds)

```bash
python scripts/run_multiseed_transformer_5.py
```

Generated checkpoints:

```text
checkpoints/transformer_seed*.pth
```

---

# 2. Evaluation

## Global Metrics — CITE-ODE

```bash
python scripts/evaluate_multiseed.py
```

---

## Global Metrics — GRU

```bash
python scripts/evaluate_multiseed_gru_5.py
```

---

## Global Metrics — Transformer

```bash
python scripts/evaluate_multiseed_transformer_5.py
```

---

## Selective Prediction Sweep — CITE-ODE

```bash
python scripts/evaluate_selective_multiseed_full.py
```

Outputs mean ± standard deviation across coverage levels:

- 100%
- 90%
- 80%
- 70%

---

# 3. Figure Generation

```bash
python scripts/generate_all_figures.py
```

Generated figures:

| File | Description |
|---|---|
| `figure1_reliability.pdf` | Reliability diagram under 6-hour telemetry blackout |
| `figure2_selective_variance.pdf` | ECE vs coverage with 1σ variance bands |
| `figure3_subgroup_scatter.pdf` | Subgroup AUROC vs ECE scatter plot |

Figures are exported as:

- Vector PDF
- 300 dpi PNG

---

# Experimental Results

| Metric | CITE-ODE (5 seeds) | GRU (5 seeds) | Transformer (5 seeds) |
|---|---|---|---|
| Global AUROC | `0.797 ± 0.016` | `0.853 ± 0.009` | `0.873 ± 0.015` |
| Global ECE | `0.019 ± 0.007` | `0.017 ± 0.005` | `0.022 ± 0.005` |
| Blackout AUROC | `0.807 ± 0.013` | `0.836 ± 0.013` | `—` |
| Blackout ECE | `0.018 ± 0.010` | `0.016 ± 0.004` | `—` |
| Selective ECE @80% | `0.0085 ± 0.0074` | `N/A` | `N/A` |
| Stratified Control @80% | `0.0195 ± 0.0106` | `N/A` | `N/A` |

Selective prediction improvement is consistent across all five CITE-ODE seeds with no seed-level reversal.

---

# Key Findings

- Uncertainty-guided filtering reduces conditional calibration error by more than 50% at 80% coverage.
- Stratified random controls preserving prevalence do not exhibit equivalent calibration improvements.
- Learned epistemic uncertainty is informative for calibration-aware filtering.
- CITE-ODE intentionally trades some discriminative performance (AUROC) for improved uncertainty reliability under structured missingness.
- Subgroup variability correlates primarily with sample size, with no persistent directional bias observed.

---

# Visualization Assets

The evaluation pipeline automatically generates publication-ready figures directly from final multi-seed outputs.

All figures are stored in:

```text
plots/
```

Available formats:

- PDF (vector)
- PNG (300 dpi raster)

---

# Citation

```bibtex
pending
```

---

# Contact

For questions, issues, or reproducibility concerns, please open a GitHub issue.