# Google Colab quick setup

Use this snippet in a Colab notebook cell to prepare the runtime, install the exact PyTorch wheel that matches the session CUDA, then install the remaining requirements and run a quick verification.

```bash
# Get the repository into Colab / Drive
# Option A — upload the project folder (preferred):
#   Upload the repository folder (not a ZIP) directly into your Google Drive and then mount Drive in Colab (see README).
# Option B — clone in Colab:
git clone https://github.com/nxxis/CITE-ODE.git
cd CITE-ODE

# Install PyTorch wheel matching the Colab CUDA (example: cu128). If unsure, visit https://pytorch.org/get-started/locally/ to generate the right command.
pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 torchvision torchaudio

# Install the rest of the requirements without overwriting torch
pip install -r requirements.txt --no-deps

# Optional quick verification
python -c "import torch,numpy,pandas; print('OK', torch.__version__, numpy.__version__, pandas.__version__)"

# Optional: save Colab environment for exact reproduction
pip freeze > requirements_lock_colab.txt
```

Notes

- Replace `cu128` with the CUDA version used by the Colab runtime if necessary.
- If using a CPU-only session, install the CPU PyTorch wheel instead (see the PyTorch site).

Quick run (mount + execute)

In a Colab notebook cell, run the following sequence to mount Drive, change into the project folder, and execute the run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content/drive/MyDrive/CITE-ODE   # adjust path if your folder name differs
# To train models from scratch (5 seeds for CITE-ODE, 3 seeds for GRU):
# Train CITE-ODE (5 seeds)
python scripts/run_multiseed_train.py

# Train GRU baseline (3 seeds)
python scripts/run_multiseed_gru.py

# To evaluate pre-trained models and reproduce paper results:
python scripts/evaluate_multiseed.py
python scripts/evaluate_multiseed_gru.py
python scripts/evaluate_multiseed_gru_blackout.py
python scripts/evaluate_selective_multiseed_full.py

# Generate paper figures
python scripts/generate_all_figures.py
```

Note: If the repository uses different script names, run the equivalent scripts under `scripts/` (for example `scripts/run_multiseed_train.py` and `scripts/evaluate_multiseed.py`).
