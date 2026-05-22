"""Evaluate Transformer baseline checkpoints across five random seeds.

This script mirrors the GRU multi-seed evaluators but loads the
`TSTransformer` baseline checkpoints and reports mean ± std metrics.
"""

import os
import numpy as np
import sys
import logging
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clinical_mimic import get_mimic_dataloader
from models.modern_baselines import TSTransformer
from utils.metrics import calculate_ece

def evaluate_one(model_path, device, loader):
    model = TSTransformer(input_dim=4, d_model=64, n_heads=4, num_layers=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, mask = x.to(device), mask.to(device)
            logits = model(x, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_y.extend(y.cpu().numpy())
    X = np.array(all_probs).reshape(-1,1)
    Y = np.array(all_y)
    _, y_prob_te, _, y_te = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    y_prob_te = y_prob_te.flatten()
    y_te = y_te.flatten()
    return {
        'auroc': roc_auc_score(y_te, y_prob_te),
        'auprc': average_precision_score(y_te, y_prob_te),
        'ece': calculate_ece(y_te, y_prob_te),
        'brier': brier_score_loss(y_te, y_prob_te)
    }

def main():
    seeds = [42, 123, 456, 789, 101112]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()
    results = {k: [] for k in ['auroc','auprc','ece','brier']}
    for seed in seeds:
        model_path = f"checkpoints/transformer_seed{seed}.pth"
        if not os.path.exists(model_path):
            logging.warning("Missing %s, skipping seed %s", model_path, seed)
            continue
        metrics = evaluate_one(model_path, device, loader)
        for k, v in metrics.items():
            results[k].append(v)
        logging.info("Transformer seed %s: AUROC=%.4f, ECE=%.4f", seed, metrics['auroc'], metrics['ece'])
    logging.info("TRANSFORMER (5 seeds) mean ± std:")
    for k, vlist in results.items():
        logging.info("%s: %.4f ± %.4f", k.upper(), np.mean(vlist), np.std(vlist))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

if __name__ == "__main__":
    main()
