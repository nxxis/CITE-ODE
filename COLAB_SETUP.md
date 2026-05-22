# Google Colab Quick Setup

Use this guide inside a Google Colab notebook to prepare the runtime, install the correct PyTorch build for the active CUDA environment, and execute training or evaluation scripts for **CITE-ODE**.

---

# 1. Clone or Upload the Repository

## Option A — Upload to Google Drive (Recommended)

Upload the repository folder directly into your Google Drive and mount Drive inside Colab.

---

## Option B — Clone Directly in Colab

```bash
git clone https://github.com/nxxis/CITE-ODE.git
cd CITE-ODE
```

---

# 2. Install PyTorch Matching the Colab CUDA Runtime

Example for CUDA 12.8 (`cu128`):

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 torchvision torchaudio
```

If unsure which CUDA version your Colab runtime uses, generate the correct install command from:

```text
https://pytorch.org/get-started/locally/
```

---

# 3. Install Remaining Requirements

Install the remaining dependencies without overwriting the PyTorch installation:

```bash
pip install -r requirements.txt --no-deps
```

---

# 4. Optional Environment Verification

```bash
python -c "import torch,numpy,pandas; print('OK', torch.__version__, numpy.__version__, pandas.__version__)"
```

Expected output format:

```text
OK 2.10.0+cu128 2.0.2 2.2.2
```

---

# 5. Optional Exact Environment Snapshot

To save the full Colab environment for reproducibility:

```bash
pip freeze > requirements_lock_colab.txt
```

---

# Notes

- Replace `cu128` with the CUDA version used by your Colab runtime if necessary.
- For CPU-only sessions, install the CPU PyTorch wheel instead.
- Always run commands from the repository root directory.

---

# Quick Run Workflow (Mount + Execute)

## Mount Google Drive

Run the following inside a Colab notebook cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Change Into the Repository Folder

```bash
%cd /content/drive/MyDrive/CITE-ODE
```

Adjust the path if your Drive folder name differs.

---

# Training Commands

## Train CITE-ODE (5 Seeds)

```bash
python scripts/run_multiseed_train.py
```

---

## Train GRU Baseline (5 Seeds)

```bash
python scripts/run_multiseed_gru_5.py
```

---

## Train Transformer Baseline (5 Seeds)

```bash
python scripts/run_multiseed_transformer_5.py
```

---

## Train MC Dropout GRU (5 Seeds)

```bash
python scripts/run_multiseed_gru_mc_dropout.py
```

---

## Train Ablations (ODE+BCE, Evidential GRU)

```bash
python scripts/run_multiseed_ode_bce.py
python scripts/run_multiseed_evidential_gru.py
```

Where:

- `run_multiseed_ode_bce.py` → ODE with binary cross-entropy
- `run_multiseed_evidential_gru.py` → GRU with evidential head

---

# Evaluation Commands

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

---

## MC Dropout GRU Selective Prediction

```bash
python scripts/evaluate_multiseed_gru_mc_dropout.py
```

---

## Ablation Evaluations (Global Metrics)

```bash
python scripts/evaluate_ode_bce.py
python scripts/evaluate_evidential_gru.py
```

Where:

- `evaluate_ode_bce.py` → ODE+BCE
- `evaluate_evidential_gru.py` → Evidential GRU

---

# Subgroup Analysis

```bash
python scripts/evaluate_multiseed_subgroups.py
```

---

# Generate Publication Figures

```bash
python scripts/generate_all_figures.py
```

Generated outputs include:

- Reliability diagram (Figure 1)
- Selective prediction variance plot (Figure 2)
- Subgroup scatter plot (Figure 3)
- Risk-coverage curve (Figure 4)
- Uncertainty trajectory during blackout (Figure 5)

---

# Reproducibility Notebook

A complete Colab notebook that reproduces all paper results is provided:

```text
CITE_ODE_Reproducibility.ipynb
```

It contains:

- Quick evaluation mode (uses pre-trained checkpoints, runs in minutes)
- Full training mode (optional, several hours)
- All evaluation commands and expected result tables

---

# Important Reminder

All executable scripts are located under:

```text
scripts/
```

Ensure all commands are executed from the repository root:

```text
CITE-ODE/
```