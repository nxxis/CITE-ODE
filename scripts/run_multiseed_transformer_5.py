import subprocess
import os
import sys

os.makedirs("checkpoints", exist_ok=True)
seeds = [42, 123, 456, 789, 101112]
script_dir = os.path.dirname(__file__)
python = sys.executable
for seed in seeds:
    output = os.path.join(script_dir, f"checkpoints/transformer_seed{seed}.pth")
    script_path = os.path.join(script_dir, "train_transformer_seed.py")
    cmd = [python, script_path, "--seed", str(seed), "--output", output]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
