import subprocess
import os

os.makedirs("checkpoints", exist_ok=True)
seeds = [42, 123, 456, 789, 101112]
for seed in seeds:
    output = f"checkpoints/transformer_seed{seed}.pth"
    cmd = f"python train_transformer_seed.py --seed {seed} --output {output}"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
