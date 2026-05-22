"""Train and evaluate the discrete GRU baseline on the MIMIC-style cohort.

This module contains a compact training loop for the GRU baseline used
in comparison experiments. After training, it runs a bootstrap-based
audit (AUROC, AUPRC, ECE, Brier) on the held-out test split and saves
the trained weights to `checkpoints/baseline_gru.pth`.
"""

import os
import random
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clinical_mimic import get_mimic_dataloader
from utils.metrics import calculate_ece, run_bootstrap_audit
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class GRUBaselineNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16):
        super(GRUBaselineNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        out, _ = self.gru(x)
        # Extract the final valid sequential step before padding boundary zones
        idx_last = mask.sum(dim=1) - 1
        final_hidden = out[torch.arange(out.size(0)), idx_last]
        logits = self.classifier(final_hidden)
        return final_hidden, logits


def main():
    """Train and evaluate a discrete GRU baseline on the MIMIC-style cohort.

    This script trains a small GRU classifier on the cohort provided by
    `data.clinical_mimic.get_mimic_dataloader`. After training it runs a
    bootstrap-based evaluation (AUROC, AUPRC, ECE, Brier) on the held-out
    test split and prints summary statistics. Model weights are saved to
    `checkpoints/baseline_gru.pth`.
    """

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = get_mimic_dataloader()
    model = GRUBaselineNet(input_dim=4, hidden_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    logging.info("Optimizing discrete GRU baseline on 10k longitudinal footprint...")
    for epoch in range(15):
        model.train()
        total_loss = 0.0
        for x, t, c, y, d, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            optimizer.zero_grad()
            _, logits = model(x, mask)
            loss = criterion(logits.squeeze(-1), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logging.info("Baseline Epoch [%02d/15] | Supervised BCE Loss: %.4f", epoch+1, total_loss/len(loader))

    model.eval()
    all_probs, all_y, all_d = [], [], []

    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, mask = x.to(device), mask.to(device)
            _, logits = model(x, mask)
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_probs.append(probs)
            all_y.append(y.cpu().numpy())
            all_d.append(d.cpu().numpy())

    Y_prob = np.concatenate(all_probs)
    Y_test = np.concatenate(all_y)
    D_test = np.vstack(all_d)

    # Isolate strictly onto the identical holdout test split footprint
    _, y_prob_te, _, y_te, _, d_te = train_test_split(
        Y_prob, Y_test, D_test, test_size=0.2, random_state=42, stratify=Y_test
    )

    bounds = run_bootstrap_audit(y_te, y_prob_te, n_resamples=1000)

    logging.info("%s", "=" * 85)
    logging.info("Discrete recurrent baseline audit (variant A: no ODE - 10k cohort)")
    logging.info("%s", "=" * 85)
    logging.info(
        "Global predictive AUROC: %.4f | 95%% CI: (%.4f, %.4f)",
        roc_auc_score(y_te, y_prob_te),
        bounds["auroc_ci"][0],
        bounds["auroc_ci"][1],
    )
    logging.info(
        "Global predictive AUPRC: %.4f | 95%% CI: (%.4f, %.4f)",
        average_precision_score(y_te, y_prob_te),
        bounds["auprc_ci"][0],
        bounds["auprc_ci"][1],
    )
    logging.info(
        "Expected calibration error (ECE): %.4f | 95%% CI: (%.4f, %.4f)",
        calculate_ece(y_te, y_prob_te),
        bounds["ece_ci"][0],
        bounds["ece_ci"][1],
    )
    logging.info(
        "Total clinical Brier score:  %.4f | 95%% CI: (%.4f, %.4f)",
        brier_score_loss(y_te, y_prob_te),
        bounds["brier_ci"][0],
        bounds["brier_ci"][1],
    )

    logging.info("Post-hoc subgroup disparity matrix")
    logging.info("%s", "-" * 85)
    groups = {
        "Female": d_te[:, 1] == 0,
        "Male": d_te[:, 1] == 1,
        "Younger": d_te[:, 0] < np.median(d_te[:, 0]),
        "Older": d_te[:, 0] >= np.median(d_te[:, 0]),
        "White": d_te[:, 2] == 0,
        "Black": d_te[:, 2] == 1,
        "Hispanic": d_te[:, 2] == 2,
        "Asian": d_te[:, 2] == 3,
    }
    for name, mask_g in groups.items():
        if sum(mask_g) > 5 and len(np.unique(y_te[mask_g])) > 1:
            g_auc = roc_auc_score(y_te[mask_g], y_prob_te[mask_g])
            g_ece = calculate_ece(y_te[mask_g], y_prob_te[mask_g])
            logging.info("%s | AUROC: %.4f | ECE: %.4f", name, g_auc, g_ece)
        else:
            logging.info("%s | Insufficient data density.", name)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/baseline_gru.pth")
    logging.info("10k GRU baseline weights saved to checkpoints/baseline_gru.pth")


if __name__ == "__main__":
    main()
