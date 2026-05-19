import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import CEMREvidentialODE
from utils.metrics import calculate_ece, run_bootstrap_audit

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CEMREvidentialODE(latent_dim=16).to(device)
    model.load_state_dict(torch.load("checkpoints/cemr_ablation_no_adv.pth", map_location=device))
    model.eval()
    
    loader = get_mimic_dataloader()
    all_probs, all_y, all_d, all_unc = [], [], [], []
    
    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, t, mask = x.to(device), t.to(device), mask.to(device)
            full_traj, logits_y, params = model(x, torch.linspace(0.0, 1.0, steps=x.shape[1], device=device))
            
            idx_last = mask.sum(dim=1) - 1
            final_logits = logits_y[torch.arange(logits_y.size(0)), idx_last].squeeze(-1)
            probs = torch.sigmoid(final_logits).cpu().numpy()
            
            alpha, beta = params[2], params[3]
            epistemic_unc = torch.mean(beta / (alpha - 1 + 1e-6), dim=-1)
            final_unc = epistemic_unc[torch.arange(epistemic_unc.size(0)), idx_last].cpu().numpy()
            
            all_probs.append(probs)
            all_y.append(y.cpu().numpy())
            all_d.append(d.cpu().numpy())
            all_unc.append(final_unc)
            
    Y_prob = np.concatenate(all_probs)
    Y_test = np.concatenate(all_y)
    D_test = np.vstack(all_d)
    Unc_test = np.concatenate(all_unc)
    
    _, y_prob_te, _, y_te, _, d_te, _, unc_te = train_test_split(
        Y_prob, Y_test, D_test, Unc_test, test_size=0.2, random_state=42, stratify=Y_test
    )
    
    bounds = run_bootstrap_audit(y_te, y_prob_te, n_resamples=1000)
    
    print("\n=== Contribution-Isolation Audit: Variant C (no adversary) ===")
    print(f"Global Predictive AUROC: {roc_auc_score(y_te, y_prob_te):.4f} | 95% CI: ({bounds['auroc_ci'][0]:.4f}, {bounds['auroc_ci'][1]:.4f})")
    print(f"Global Predictive AUPRC: {average_precision_score(y_te, y_prob_te):.4f} | 95% CI: ({bounds['auprc_ci'][0]:.4f}, {bounds['auprc_ci'][1]:.4f})")
    print(f"Expected Calibration Error: {calculate_ece(y_te, y_prob_te):.4f} | 95% CI: ({bounds['ece_ci'][0]:.4f}, {bounds['ece_ci'][1]:.4f})")
    print(f"Clinical Brier Score: {brier_score_loss(y_te, y_prob_te):.4f} | 95% CI: ({bounds['brier_ci'][0]:.4f}, {bounds['brier_ci'][1]:.4f})")

    print("\nPost-hoc subgroup disparity summary:")
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
            print(f"{name:<12} | AUROC: {roc_auc_score(y_te[mask], y_prob_te[mask]):.4f} | ECE: {calculate_ece(y_te[mask], y_prob_te[mask]):.4f} | Epistemic Unc: {np.mean(unc_te[mask]):.4f}")

if __name__ == "__main__":
    main()
