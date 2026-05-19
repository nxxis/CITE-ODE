"""Launcher script for training GRU baselines across multiple seeds.

Runs `train_gru_seed.py` for a short list of seeds and writes checkpoints
into `checkpoints/`. This wrapper is provided for convenience when
reproducing baseline runs locally or in a Colab session.
"""

import subprocess
import os

os.makedirs("checkpoints", exist_ok=True)

# Use 3 seeds (enough to show variance, less costly)
seeds = [42, 123, 456]
for seed in seeds:
    output_path = f"checkpoints/baseline_gru_seed{seed}.pth"
    cmd = f"python train_gru_seed.py --seed {seed} --output {output_path}"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    print(f"Finished GRU seed {seed}\n")
