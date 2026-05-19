import subprocess
import os

os.makedirs("checkpoints", exist_ok=True)

seeds = [42, 123, 456, 789, 101112]
for seed in seeds:
    output_path = f"checkpoints/cemr_fair_seed{seed}.pth"
    cmd = f"python train_cemr_seed.py --seed {seed} --output {output_path}"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    print(f"Finished seed {seed}\n")
