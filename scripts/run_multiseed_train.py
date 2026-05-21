"""Launcher script: train CEMR model across multiple random seeds.

This simple wrapper invokes `train_cemr_seed.py` once per seed and stores
each resulting checkpoint under `checkpoints/` following the repository's
naming convention.
"""

import subprocess
import os
import sys

# Ensure checkpoint directory exists irrespective of working directory
os.makedirs("checkpoints", exist_ok=True)

seeds = [42, 123, 456, 789, 101112]
script_dir = os.path.dirname(__file__)
python = sys.executable
for seed in seeds:
    output_path = os.path.join(script_dir, f"checkpoints/cemr_fair_seed{seed}.pth")
    script_path = os.path.join(script_dir, "train_cemr_seed.py")
    cmd = [python, script_path, "--seed", str(seed), "--output", output_path]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Finished seed {seed}\n")
