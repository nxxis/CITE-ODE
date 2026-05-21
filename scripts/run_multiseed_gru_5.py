"""Launcher to train the GRU baseline across the canonical multi-seed set.

This lightweight wrapper mirrors the experimental protocol used for the
CEMR model: it invokes `train_gru_seed.py` once per random seed and writes
checkpoints to the `checkpoints/` directory using a consistent filename
convention.
"""

import subprocess
import os
import sys

# Ensure checkpoint directory exists
os.makedirs("checkpoints", exist_ok=True)

seeds = [42, 123, 456, 789, 101112]
script_dir = os.path.dirname(__file__)
python = sys.executable
for seed in seeds:
    output = os.path.join(script_dir, f"checkpoints/baseline_gru_seed{seed}.pth")
    script_path = os.path.join(script_dir, "train_gru_seed.py")
    cmd = [python, script_path, "--seed", str(seed), "--output", output]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
