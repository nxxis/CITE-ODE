"""Launcher script for training GRU baselines across multiple seeds.

Runs `train_gru_seed.py` for a short list of seeds and writes checkpoints
into `checkpoints/`. This wrapper is provided for convenience when
reproducing baseline runs locally or in a Colab session.
"""

import subprocess
import os
import sys

os.makedirs("checkpoints", exist_ok=True)

# Use 3 seeds (enough to show variance, less costly)
seeds = [42, 123, 456]
script_dir = os.path.dirname(__file__)
python = sys.executable
for seed in seeds:
    output_path = os.path.join(script_dir, f"checkpoints/baseline_gru_seed{seed}.pth")
    script_path = os.path.join(script_dir, "train_gru_seed.py")
    cmd = [python, script_path, "--seed", str(seed), "--output", output_path]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Finished GRU seed {seed}\n")
