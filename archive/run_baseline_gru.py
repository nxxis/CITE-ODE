import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from data.clinical_mimic import get_mimic_dataloader
from utils.metrics import calculate_ece, run_bootstrap_audit

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
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loader = get_mimic_dataloader()
    model = GRUBaselineNet(input_dim=4, hidden_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    print("🚀 Optimizing Discrete GRU Baseline on 10k Longitudinal Footprint...")
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
            
        print(f"Baseline Epoch [{epoch+1:02d}/15] | Supervised BCE Loss: {total_loss/len(loader):.4f}")

    # ─────────────────────────────────────────────────────────────
    # POST-TRAINING RIGOROUS VALIDATION & DEMOGRAPHIC STRATIFICATION
    # ─────────────────────────────────────────────────────────────
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

    print("\n=====================================================================================")
    print("📋 DISCRETE RECURRENT BASELINE AUDIT (VARIANT A: NO ODE - 10K COHORT)")
    print("=====================================================================================")
    print(f"🏥 Global Predictive AUROC:    {roc_auc_score(y_te, y_prob_te):.4f} | 95% CI: ({bounds['auroc_ci'][0]:.4f}, {bounds['auroc_ci'][1]:.4f})")
    print(f"🏥 Global Predictive AUPRC:    {average_precision_score(y_te, y_prob_te):.4f} | 95% CI: ({bounds['auprc_ci'][0]:.4f}, {bounds['auprc_ci'][1]:.4f})")
    print(f"🎯 Expected Calibration Error: {calculate_ece(y_te, y_prob_te):.4f} | 95% CI: ({bounds['ece_ci'][0]:.4f}, {bounds['ece_ci'][1]:.4f})")
    print(f"📉 Total Clinical Brier Score:  {brier_score_loss(y_te, y_prob_te):.4f} | 95% CI: ({bounds['brier_ci'][0]:.4f}, {bounds['brier_ci'][1]:.4f})")
    
    print("\n⚖️ POST-HOC SUBGROUP DISPARITY MATRIX")
    print("-" * 85)
    groups = {
        "Female": d_te[:, 1] == 0, 
        "Male": d_te[:, 1] == 1, 
        "Younger": d_te[:, 0] < np.median(d_te[:, 0]), 
        "Older": d_te[:, 0] >= np.median(d_te[:, 0]), 
        "White": d_te[:, 2] == 0, 
        "Black": d_te[:, 2] == 1,
        "Hispanic": d_te[:, 2] == 2, 
        "Asian": d_te[:, 2] == 3
    }
    for name, mask in groups.items():
        if sum(mask) > 5 and len(np.unique(y_te[mask])) > 1:
            g_auc = roc_auc_score(y_te[mask], y_prob_te[mask])
            g_ece = calculate_ece(y_te[mask], y_prob_te[mask])
            print(f"👤 {name:<12} | AUROC: {g_auc:.4f} | ECE: {g_ece:.4f}")
        else:
            print(f"👤 {name:<12} | Insufficient data density.")

    torch.save(model.state_dict(), "checkpoints/baseline_gru.pth")
    print("\n💾 10k GRU Baseline weights saved cleanly to Drive.")

if __name__ == "__main__":
    main()
