"""Train and evaluate the GRU-D baseline (handles irregular sampling).

This module implements a GRU-D variant where inputs and hidden states
are decayed according to learned exponential factors capturing irregular
time gaps. The script trains the model, evaluates with bootstrap CIs,
and saves the checkpoint to `checkpoints/baseline_grud.pth`.
"""

import os
import random
import numpy as np
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


class GRUDCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUDCell, self).__init__()
        self.hidden_dim = hidden_dim
        # Trainable decay parameters for temporal gaps
        self.w_dg_x = nn.Parameter(torch.zeros(input_dim))
        self.b_dg_x = nn.Parameter(torch.zeros(input_dim))
        self.w_dg_h = nn.Linear(input_dim, hidden_dim)

        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, x, h, delta_t):
        # Calculate exponential decay multipliers based on time deltas
        gamma_x = torch.exp(-torch.clamp(self.w_dg_x * delta_t + self.b_dg_x, min=0.0))
        gamma_h = torch.exp(-torch.clamp(self.w_dg_h(delta_t), min=0.0))

        # Decay input features and hidden states toward historical expectations
        x_decayed = gamma_x * x
        h_decayed = gamma_h * h

        h_next = self.gru_cell(x_decayed, h_decayed)
        return h_next


class GRUDBaselineNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16):
        super(GRUDBaselineNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.grud_cell = GRUDCell(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, t, mask):
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        h = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Sequentially map steps while incorporating time-delta intervals
        for step in range(seq_len):
            if step == 0:
                delta_t = torch.zeros(batch_size, input_dim, device=device)
            else:
                dt_step = (t[:, step] - t[:, step - 1]).unsqueeze(-1).expand(-1, input_dim)
                delta_t = torch.clamp(dt_step, min=1e-5)

            h = self.grud_cell(x[:, step], h, delta_t)

        logits = self.classifier(h)
        return h, logits


def main():
    """Train and evaluate a GRU-D baseline that models irregular sampling.

    The GRU-D variant accounts for irregular time gaps between observations
    by applying learned exponential decay factors to inputs and hidden
    states. Training uses the dataloader from `data.clinical_mimic` and
    evaluates final predictive performance with bootstrap confidence
    intervals. The trained checkpoint is saved to
    `checkpoints/baseline_grud.pth`.
    """
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()

    model = GRUDBaselineNet(input_dim=4, hidden_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    logging.info("Optimizing irregular GRU-D baseline network on 10k cohort...")
    for epoch in range(15):
        model.train()
        total_loss = 0.0
        for x, t, c, y, d, mask in loader:
            x, t, y, mask = x.to(device), t.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            _, logits = model(x, t, mask)
            loss = criterion(logits.squeeze(-1), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logging.info("GRU-D Epoch [%02d/15] | Temporal Cross-Entropy Loss: %.4f", epoch+1, total_loss/len(loader))

    model.eval()
    all_probs, all_y, all_d = [], [], []
    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, t, mask = x.to(device), t.to(device), mask.to(device)
            _, logits = model(x, t, mask)
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_probs.append(probs)
            all_y.append(y.cpu().numpy())
            all_d.append(d.cpu().numpy())

    Y_prob, Y_test, D_test = np.concatenate(all_probs), np.concatenate(all_y), np.vstack(all_d)
    _, y_prob_te, _, y_te, _, d_te = train_test_split(Y_prob, Y_test, D_test, test_size=0.2, random_state=42, stratify=Y_test)
    bounds = run_bootstrap_audit(y_te, y_prob_te, n_resamples=1000)

    logging.info("%s", "=" * 85)
    logging.info("Irregular decay GRU-D baseline audit results (10k cohort)")
    logging.info("%s", "=" * 85)
    logging.info(
        "Global predictive AUROC: %.4f | 95%% CI: (%.4f, %.4f)",
        roc_auc_score(y_te, y_prob_te),
        bounds["auroc_ci"][0],
        bounds["auroc_ci"][1],
    )
    logging.info(
        "Expected calibration error (ECE): %.4f | 95%% CI: (%.4f, %.4f)",
        calculate_ece(y_te, y_prob_te),
        bounds["ece_ci"][0],
        bounds["ece_ci"][1],
    )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/baseline_grud.pth")
    logging.info("GRU-D weights saved to checkpoints/baseline_grud.pth")


if __name__ == "__main__":
    main()
