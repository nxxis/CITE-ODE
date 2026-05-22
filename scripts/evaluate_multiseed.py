"""Evaluate pre-trained CEMR models across multiple random seeds.

This script loads checkpoints saved at `checkpoints/cemr_fair_seed{seed}.pth`,
runs each model on the canonical dataloader, and computes aggregate
performance metrics (AUROC, AUPRC, ECE, Brier) using a held-out split.

The outputs are printed as mean ± std across evaluated seeds.
"""

import os, numpy as np
import sys
import logging
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import CEMREvidentialODE
from utils.metrics import calculate_ece

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def evaluate_one_model(model_path, device, loader):
    model = CEMREvidentialODE(latent_dim=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, t, mask = x.to(device), t.to(device), mask.to(device)
            full_traj, logits_y, _ = model(x, torch.linspace(0.0, 1.0, steps=x.shape[1], device=device))
            idx_last = mask.sum(dim=1) - 1
            final_logits = logits_y[torch.arange(logits_y.size(0)), idx_last].squeeze(-1)
            probs = torch.sigmoid(final_logits).cpu().numpy()
            all_probs.extend(probs)
            all_y.extend(y.cpu().numpy())
    X = np.array(all_probs).reshape(-1, 1)
    Y = np.array(all_y)
    # Use same test split as in evaluate_rigor.py
    _, y_prob_te, _, y_te = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    # Flatten to 1D arrays for metrics
    y_prob_te = y_prob_te.flatten()
    y_te = y_te.flatten()
    return {
        'auroc': roc_auc_score(y_te, y_prob_te),
        'auprc': average_precision_score(y_te, y_prob_te),
        'ece': calculate_ece(y_te, y_prob_te),
        'brier': brier_score_loss(y_te, y_prob_te)
    }

def main():
    """Evaluate saved checkpoints for a list of seeds and print aggregates.

    The function expects model checkpoints to follow the naming convention
    `checkpoints/cemr_fair_seed{seed}.pth`. Missing checkpoints are skipped
    with a warning.
    """

    seeds = [42, 123, 456, 789, 101112]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()
    results = {k: [] for k in ['auroc', 'auprc', 'ece', 'brier']}
    for seed in seeds:
        model_path = f"checkpoints/cemr_fair_seed{seed}.pth"
        if not os.path.exists(model_path):
            logging.warning("model not found at %s; skipping seed %s.", model_path, seed)
            continue
        metrics = evaluate_one_model(model_path, device, loader)
        for k, v in metrics.items():
            results[k].append(v)
        logging.info("Seed %s results: AUROC=%.4f, ECE=%.4f", seed, metrics['auroc'], metrics['ece'])
    logging.info("Multi-Seed aggregated results:")
    for k, vlist in results.items():
        mean_val = np.mean(vlist)
        std_val = np.std(vlist)
        logging.info("%s: %.4f ± %.4f", k.upper(), mean_val, std_val)

if __name__ == "__main__":
    main()
