"""Train a single-seed Transformer baseline on the MIMIC-style cohort.

This script trains the `TSTransformer` baseline for a single random seed
and writes the resulting checkpoint to the requested output path. It is
intended as a straightforward, reproducible training entrypoint used in
baseline comparisons.
"""

import os
import random
import argparse
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from data.clinical_mimic import get_mimic_dataloader
from models.modern_baselines import TSTransformer
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def seed_everything(seed):
    """Set RNG seeds for Python/NumPy/PyTorch for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()

    model = TSTransformer(input_dim=4, d_model=64, n_heads=4, num_layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(15):
        model.train()
        total_loss = 0.0
        for x, t, c, y, d, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info("Seed %d | Epoch %d/15 | Loss: %.4f", args.seed, epoch+1, total_loss/len(loader))

    torch.save(model.state_dict(), args.output)
    logging.info("Saved %s", args.output)

if __name__ == "__main__":
    main()
