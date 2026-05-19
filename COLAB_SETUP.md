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
pip freeze > requirements.txt
```

Notes

- Replace `cu128` with the CUDA version used by the Colab runtime if necessary.
- If using a CPU-only session, install the CPU PyTorch wheel instead (see the PyTorch site).
