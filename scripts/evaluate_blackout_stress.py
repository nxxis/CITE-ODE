"""Stress-test evaluation comparing CEMR, GRU, and GRU-D under blackout.

This module applies a contiguous window blackout to input trajectories
to simulate telemetry loss and then evaluates three models' predictive
performance (AUROC and calibration/ECE) on the stressed trajectories.
"""

import os
import numpy as np
import sys
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import CEMREvidentialODE
from scripts.run_baseline_gru import GRUBaselineNet
from scripts.run_baseline_grud import GRUDBaselineNet
from utils.metrics import calculate_ece


def apply_contiguous_blackout(x, window_len=15):
    """Simulates a continuous telemetry lead disconnection by zeroing a window."""
    x_stressed = x.clone()
    batch_size, seq_len, _ = x.shape
    start_idx = seq_len // 2 - window_len // 2
    end_idx = start_idx + window_len
    x_stressed[:, start_idx:end_idx, :] = 0.0
    return x_stressed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()

    # Load all models trained across the matching 10k workspace
    ode_model = CEMREvidentialODE(latent_dim=16).to(device)
    ode_model.load_state_dict(torch.load("checkpoints/cemr_fair_final.pth", map_location=device))
    ode_model.eval()

    gru_model = GRUBaselineNet(input_dim=4, hidden_dim=16).to(device)
    gru_model.load_state_dict(torch.load("checkpoints/baseline_gru.pth", map_location=device))
    gru_model.eval()

    grud_model = GRUDBaselineNet(input_dim=4, hidden_dim=16).to(device)
    grud_model.load_state_dict(torch.load("checkpoints/baseline_grud.pth", map_location=device))
    grud_model.eval()

    all_ode, all_gru, all_grud, all_y = [], [], [], []

    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, t, y, mask = x.to(device), t.to(device), y.to(device), mask.to(device)

            stressed_x = apply_contiguous_blackout(x, window_len=15)

            # 1. CITE-ODE Framework
            _, logits_ode, _ = ode_model(stressed_x, torch.linspace(0.0, 1.0, steps=x.shape[1], device=device))
            idx_last = mask.sum(dim=1) - 1
            ode_probs = torch.sigmoid(logits_ode[torch.arange(logits_ode.size(0)), idx_last].squeeze(-1)).cpu().numpy()

            # 2. Vanilla GRU
            _, logits_gru = gru_model(stressed_x, mask)
            gru_probs = torch.sigmoid(logits_gru.squeeze(-1)).cpu().numpy()

            # 3. Irregular GRU-D
            _, logits_grud = grud_model(stressed_x, t, mask)
            grud_probs = torch.sigmoid(logits_grud.squeeze(-1)).cpu().numpy()

            all_ode.append(ode_probs)
            all_gru.append(gru_probs)
            all_grud.append(grud_probs)
            all_y.append(y.cpu().numpy())

    Y_true = np.concatenate(all_y)
    _, ode_te, _, y_te = train_test_split(np.concatenate(all_ode), Y_true, test_size=0.2, random_state=42, stratify=Y_true)
    _, gru_te, _, _ = train_test_split(np.concatenate(all_gru), Y_true, test_size=0.2, random_state=42, stratify=Y_true)
    _, grud_te, _, _ = train_test_split(np.concatenate(all_grud), Y_true, test_size=0.2, random_state=42, stratify=Y_true)

    print("\n=== Clinical stress test: 6-hour contiguous telemetry blackout ===")
    print(f"Vanilla GRU baseline       | AUROC: {roc_auc_score(y_te, gru_te):.4f} | ECE: {calculate_ece(y_te, gru_te):.4f}")
    print(f"Irregular GRU-D            | AUROC: {roc_auc_score(y_te, grud_te):.4f} | ECE: {calculate_ece(y_te, grud_te):.4f}")
    print(f"Proposed CITE-ODE          | AUROC: {roc_auc_score(y_te, ode_te):.4f} | ECE: {calculate_ece(y_te, ode_te):.4f}")


if __name__ == "__main__":
    main()
