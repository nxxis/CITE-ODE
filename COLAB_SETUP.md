# Google Colab Quick Setup

Use this guide inside a Google Colab notebook to prepare the runtime, install the correct PyTorch build for the active CUDA environment, and execute training or evaluation scripts for **CITE-ODE**.

---

# 1. Clone or Upload the Repository

## Option A — Upload to Google Drive (Recommended)

Upload the repository folder directly into your Google Drive and mount Drive inside Colab.

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

## Train GRU Baseline (5 Seeds)

```bash
python scripts/run_multiseed_gru_5.py
```

## Train Transformer Baseline (5 Seeds)

```bash
python scripts/run_multiseed_transformer_5.py
```

---

# Evaluation Commands

## Evaluate CITE-ODE

```bash
python scripts/evaluate_multiseed.py
```

## Evaluate GRU Baseline

```bash
python scripts/evaluate_multiseed_gru_5.py
```

## Evaluate Transformer Baseline

```bash
python scripts/evaluate_multiseed_transformer_5.py
```

## Run Selective Prediction Evaluation

```bash
python scripts/evaluate_selective_multiseed_full.py
```

---

# Generate Publication Figures

```bash
python scripts/generate_all_figures.py
```

Generated outputs include:

- Reliability diagrams
- Selective prediction variance plots
- Subgroup calibration scatter plots

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