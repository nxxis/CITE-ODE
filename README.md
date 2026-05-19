# CITE-ODE: Continuous-Time Evidential Neural ODEs for Uncertainty-Aware ICU Risk Prediction

Official PyTorch implementation of **CITE-ODE**, a continuous-time evidential learning framework for uncertainty-aware ICU risk prediction from irregularly sampled clinical telemetry.

Designed for submission to the **IEEE International Conference on Data Mining (ICDM 2026)**.

---

# 📌 Framework Overview

ICU telemetry data is sparse, irregularly sampled, and frequently interrupted by long contiguous gaps caused by monitor disconnects, workflow changes, and missing charting. Traditional sequence models often rely on imputation strategies that distort temporal structure and provide no principled estimate of epistemic uncertainty.

**CITE-ODE** addresses these limitations through three integrated components:

## 1. Continuous-Time Trajectory Modeling

A Neural Ordinary Differential Equation (Neural ODE) backbone models physiological trajectories continuously in time, directly handling irregular sampling without imputation.

## 2. Evidential Representation Learning

Latent trajectories are projected through a Normal-Inverse-Gamma (NIG) evidential head that estimates predictive uncertainty in a single forward pass.

Epistemic uncertainty is computed as:

\[
u = \frac{\beta}{\alpha - 1}
\]

## 3. Selective Prediction Framework

Predictions are ranked using epistemic uncertainty and selectively filtered at predefined coverage levels (100%, 90%, 80%, 70%).

A stratified random control preserving class prevalence isolates the effect of uncertainty ranking from prevalence-induced calibration changes.

---

# 🔧 Installation Guide

The repository is fully reproducible and follows the Machine Learning Reproducibility Checklist.

## Prerequisites

- Python 3.10+
- CUDA-enabled NVIDIA GPU (T4, A100, RTX series recommended)
- CUDA Toolkit compatible with installed PyTorch version

---

# ⚙️ Reproducibility & Environment

The experiments reported in this repository were executed under a tightly controlled environment to ensure reproducibility. We recommend reproducing the environment exactly when preparing artifacts for review.

Supported runtime

- Python: 3.10 (recommended)

Key package versions used in the paper

- PyTorch: `2.10.0+cu128` (CUDA 12.8 build)
- torchdiffeq: `0.2.5`
- NumPy: `2.0.2`
- Pandas: `2.2.2`
- Scikit-learn: `1.6.1`
- Matplotlib: `3.10.0`
- Seaborn: `0.13.2`
- Google Cloud BigQuery: `3.41.0`

Use Conda or Colab instead of local virtualenvs on Windows when possible (recommended for reviewers). The Conda and Colab instructions below provide the tested installation flow and exact wheel commands for PyTorch.

Conda (recommended on Windows when wheels are problematic)

```powershell
conda create -n cite_env python=3.10 -y
conda activate cite_env
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y
python -m pip install --upgrade pip
pip install -r requirements.txt --no-deps
```

Quick verification

```bash
python -c "import torch,numpy,pandas; print('torch',torch.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__)"
```

Colab

The codebase is compatible with Google Colab; prefer installing the exact PyTorch wheel that matches the session CUDA runtime, then `pip install -r requirements.txt --no-deps`. After setup, run one of the scripts in `scripts/` to validate the environment.

Colab editing tip

When iterating on notebooks in Colab you can write Python modules or scripts directly into the repository area using `%%writefile` at the top of a notebook cell. Example:

```python
%%writefile scripts/generate_all_figures.py
# (paste or export the file contents here)
```

This mirrors the author's workflow: edit code in Colab, save to Drive, and run the project's scripts from the mounted Drive copy.

**Reviewer-friendly: Upload to Google Drive and run in Colab**

If reviewers prefer a frictionless, reproducible flow (the workflow the author used), download the repository ZIP and upload it to your Google Drive, then run everything from a Colab session mounted to Drive. This avoids local wheel/build issues on some Windows/Python combinations.

Steps (reviewer):

1. Download the repository (use the GitHub "Download ZIP" button or `git clone`) and upload the ZIP to your Google Drive.

2. In a Colab notebook, mount Google Drive and extract the archive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!unzip /content/drive/MyDrive/CEMR-Fair.zip -d /content/drive/MyDrive/
%cd /content/drive/MyDrive/CEMR-Fair
```

3. Install the matching PyTorch wheel for the Colab CUDA runtime, then the remaining requirements:

```bash
!pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 torchvision torchaudio
!pip install -r requirements.txt --no-deps
```

4. Optionally save the environment lock:

```bash
!pip freeze > requirements_lock_colab.txt
```

Notes:

- Replace `cu128` with the CUDA version available in the Colab runtime if needed.
- This Drive+Colab flow mirrors the author's reproducible workflow and avoids local binary build failures.

Quick reviewer flow (one-line)

Upload the repository folder (not a ZIP) to Google Drive, or clone the repository directly from GitHub in Colab → open Google Colab and mount Drive → change directory into the project folder on Drive (or the cloned folder) → install the matching PyTorch wheel and `requirements.txt` → run `smoke_test.py` or the scripts in `scripts/` to reproduce results. See [COLAB_SETUP.md](COLAB_SETUP.md) for the exact commands to copy-paste.

---

# 📂 Repository Structure

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
│   └── baseline_grud.pth
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
│   ├── run_multiseed_gru.py
│   ├── evaluate_multiseed.py
│   ├── evaluate_multiseed_gru.py
│   ├── evaluate_multiseed_gru_blackout.py
│   ├── evaluate_selective_multiseed_full.py
│   └── generate_all_figures.py
│
├── plots/
│   ├── figure1_reliability.pdf
│   ├── figure2_selective_variance.pdf
│   └── figure3_subgroup_scatter.pdf
│
├── archive/
├── README.md
└── requirements.txt
```

---

# 📥 Dataset & Cohort Information

To support peer-review reproducibility without requiring live database access, the evaluation cohort is stored as an external artifact rather than in Git history.

## Dataset Details

- **Source Dataset:** MIMIC-IV Clinical Database
- **Cohort Size:** 10,000 ICU stays
- **Observation Window:** 24 hours
- **Signals:** 8 vital signs
- **Target:** Binary in-hospital mortality

Download the cohort into the repository with:

```bash
pip install gdown
python scripts/fetch_cohort.py
```

The script writes the file to:

```text
data/mimic_cemr_cohort.csv
```

Google Drive source folder:

https://drive.google.com/drive/u/2/folders/1oupz5CcQIMn-16I8KFWeqlpirY0vBCxg

Quick download instructions (two options):

1. Use the included Python helper (recommended):

```bash
pip install gdown
python scripts/fetch_cohort.py
```

2. Directly download the Drive folder with `gdown` (requires `gdown` supporting folder downloads):

```bash
# download the entire folder into `data/`
gdown --folder "https://drive.google.com/drive/folders/1oupz5CcQIMn-16I8KFWeqlpirY0vBCxg" -O data/
```

After downloading, verify the file exists:

```bash
ls -lh data/mimic_cemr_cohort.csv
```

---

# 🚀 Training & Evaluation

All experiments use:

- Fixed global random seeds
- Deterministic cuDNN execution
- Frozen evaluation configuration

This ensures reproducibility across supported hardware.

## Seeds

### CITE-ODE

- 42
- 123
- 456
- 789
- 101112

### GRU Baseline

- 42
- 123
- 456

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

## Train GRU Baseline (3 Seeds)

```bash
python scripts/run_multiseed_gru.py
```

Generated checkpoints:

```text
checkpoints/baseline_gru_seed*.pth
```

---

# 2. Evaluation

## Global Metrics — CITE-ODE

```bash
python scripts/evaluate_multiseed.py
```

## Global Metrics — GRU

```bash
python scripts/evaluate_multiseed_gru.py
```

## Blackout Evaluation — GRU

```bash
python scripts/evaluate_multiseed_gru_blackout.py
```

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

| File                             | Description                                         |
| -------------------------------- | --------------------------------------------------- |
| `figure1_reliability.pdf`        | Reliability diagram under 6-hour telemetry blackout |
| `figure2_selective_variance.pdf` | ECE vs coverage with 1σ variance bands              |
| `figure3_subgroup_scatter.pdf`   | Subgroup AUROC vs ECE scatter plot                  |

Figures are exported as:

- Vector PDF
- 300 dpi PNG

---

# 🏆 Experimental Results

| Metric                  | CITE-ODE (5 seeds)  | GRU (3 seeds)     |
| ----------------------- | ------------------- | ----------------- |
| Global AUROC            | 0.797 ± 0.016       | **0.853 ± 0.011** |
| Global ECE              | 0.019 ± 0.007       | 0.018 ± 0.005     |
| Blackout AUROC          | 0.807 ± 0.013       | **0.836 ± 0.013** |
| Blackout ECE            | 0.018 ± 0.010       | 0.016 ± 0.004     |
| Selective ECE @80%      | **0.0085 ± 0.0074** | N/A               |
| Stratified Control @80% | 0.0195 ± 0.0106     | N/A               |

> Selective prediction improvement is consistent across all five CITE-ODE seeds with no seed-level reversal.

---

# 🔍 Key Findings

- Uncertainty-guided filtering reduces conditional calibration error by more than 50% at 80% coverage.
- Stratified random controls preserving prevalence do not exhibit equivalent calibration improvements.
- Learned epistemic uncertainty is therefore informative for calibration-aware filtering.
- CITE-ODE intentionally trades some discriminative performance (AUROC) for improved uncertainty reliability under structured missingness.
- Subgroup variability correlates primarily with sample size, with no persistent directional bias observed.

---

# 🎨 Visualization Assets

The evaluation pipeline automatically generates publication-ready figures directly from final multi-seed outputs.

All figures are stored in:

```text
plots/
```

Available formats:

- PDF (vector)
- PNG (300 dpi raster)

---

# 📝 Citation

```bibtex
pending
```

---

# 📧 Contact

For questions, issues, or reproducibility concerns, please open a GitHub issue.

---

**Last Updated:** May 2026
