"""Train a single-seed GRU baseline and save the checkpoint.

This script provides a minimal, reproducible training entrypoint for the
discrete GRU baseline used in experiments. It accepts a `--seed` and
`--output` path and writes the trained weights to the requested file.

The implementation is deliberately compact for clarity: the goal is
readability and reproducibility rather than advanced training utilities.
"""

import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data.clinical_mimic import get_mimic_dataloader

class GRUBaselineNet(nn.Module):
    """Small GRU encoder + linear classifier used as a baseline.

    The network returns a single logit per example corresponding to the
    binary supervision used across the MIMIC-style cohort.
    """

    def __init__(self, input_dim=4, hidden_dim=16):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
            mask: Boolean/int mask of valid timesteps, same batch/seq_len

        Returns:
            logits: Tensor of shape (batch,) with raw logits (not sigmoid-ed)
        """
        out, _ = self.gru(x)
        # final valid timestep per-example (handles padding)
        idx_last = mask.sum(dim=1).long() - 1
        final = out[torch.arange(out.size(0)), idx_last]
        return self.classifier(final).squeeze(-1)

def seed_everything(seed):
    """Deterministic seeding helper for reproducible runs.

    Note: exact bitwise determinism depends on platform/CUDA/cuDNN.
    """
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
    model = GRUBaselineNet(input_dim=4, hidden_dim=16).to(device)
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
        print(f"Seed {args.seed} | Epoch {epoch+1}/15 | Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), args.output)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
