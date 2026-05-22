import subprocess
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
os.makedirs("checkpoints", exist_ok=True)
seeds = [42, 123, 456, 789, 101112]
for seed in seeds:
    output = f"checkpoints/gru_mc_dropout_seed{seed}.pth"
    cmd = f"python scripts/train_gru_mc_dropout_seed.py --seed {seed} --output {output}"
    logging.info("Running: %s", cmd)
    subprocess.run(cmd, shell=True, check=True)
