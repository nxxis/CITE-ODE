"""Evaluate GRU baseline checkpoints across multiple random seeds.

Loads GRU checkpoints from `checkpoints/baseline_gru_seed{seed}.pth`, runs them
on the canonical dataloader and reports AUROC, ECE, and Brier score aggregates.
"""

import os, numpy as np
import sys
import logging
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clinical_mimic import get_mimic_dataloader
from utils.metrics import calculate_ece
from scripts.train_gru_seed import GRUBaselineNet

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def evaluate_gru_one_model(model_path, device, loader):
    model = GRUBaselineNet(input_dim=4, hidden_dim=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, mask = x.to(device), mask.to(device)
            _, logits = model(x, mask)
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_probs.extend(probs)
            all_y.extend(y.cpu().numpy())
    X = np.array(all_probs).reshape(-1, 1)
    Y = np.array(all_y)
    _, y_prob_te, _, y_te = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    y_prob_te = y_prob_te.flatten()
    y_te = y_te.flatten()
    return {
        'auroc': roc_auc_score(y_te, y_prob_te),
        'ece': calculate_ece(y_te, y_prob_te),
        'brier': brier_score_loss(y_te, y_prob_te)
    }

def main():
    """Evaluate saved GRU checkpoints and print mean ± std across seeds."""

    seeds = [42, 123, 456]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()
    results = {'auroc': [], 'ece': [], 'brier': []}
    for seed in seeds:
        model_path = f"checkpoints/baseline_gru_seed{seed}.pth"
        if not os.path.exists(model_path):
            logging.warning("%s not found, skipping seed %s", model_path, seed)
            continue
        metrics = evaluate_gru_one_model(model_path, device, loader)
        for k, v in metrics.items():
            results[k].append(v)
        logging.info("GRU Seed %s: AUROC=%.4f, ECE=%.4f", seed, metrics['auroc'], metrics['ece'])
    if not results['auroc']:
        logging.warning("No models found.")
        return
    logging.info("GRU Multi‑Seed Results")
    logging.info("AUROC: %.4f ± %.4f", np.mean(results['auroc']), np.std(results['auroc']))
    logging.info("ECE:   %.4f ± %.4f", np.mean(results['ece']), np.std(results['ece']))
    logging.info("Brier: %.4f ± %.4f", np.mean(results['brier']), np.std(results['brier']))

if __name__ == "__main__":
    main()
