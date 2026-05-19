import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import CEMREvidentialODE
from run_baseline_gru import GRUBaselineNet
from utils.metrics import calculate_ece

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def apply_stress_mask(x, dropout_rate):
    if dropout_rate == 0.0:
        return x
    # Generate a binomial missingness field replicating real-world clinical data dropouts
    mask = torch.rand(*x.shape, device=x.device) > dropout_rate
    return x * mask

def evaluate_models_under_stress(dropout_rate, device, loader, ode_model, gru_model):
    all_ode_probs, all_gru_probs, all_y, all_ode_unc = [], [], [], []
    
    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, t, y, mask = x.to(device), t.to(device), y.to(device), mask.to(device)
            
            # Inject random telemetry dropout directly onto the current processing inputs
            stressed_x = apply_stress_mask(x, dropout_rate)
            
            # Target A: Latent Neural ODE Continuous Path Evaluation
            full_traj, logits_y, params = ode_model(stressed_x, torch.linspace(0.0, 1.0, steps=x.shape[1], device=device))
            idx_last = mask.sum(dim=1) - 1
            ode_logits = logits_y[torch.arange(logits_y.size(0)), idx_last].squeeze(-1)
            ode_probs = torch.sigmoid(ode_logits).cpu().numpy()
            
            alpha, beta = params[2], params[3]
            epistemic_unc = torch.mean(beta / (alpha - 1 + 1e-6), dim=-1)
            final_ode_unc = epistemic_unc[torch.arange(epistemic_unc.size(0)), idx_last].cpu().numpy()
            
            # Target B: Standard Discrete Recurrent Baseline Evaluation
            _, gru_logits = gru_model(stressed_x, mask)
            gru_probs = torch.sigmoid(gru_logits.squeeze(-1)).cpu().numpy()
            
            all_ode_probs.append(ode_probs)
            all_gru_probs.append(gru_probs)
            all_y.append(y.cpu().numpy())
            all_ode_unc.append(final_ode_unc)
            
    Y_ode = np.concatenate(all_ode_probs)
    Y_gru = np.concatenate(all_gru_probs)
    Y_true = np.concatenate(all_y)
    Unc_ode = np.concatenate(all_ode_unc)
    
    # Isolate onto the identical validation split index allocation
    _, ode_probs_te, _, y_te = train_test_split(Y_ode, Y_true, test_size=0.2, random_state=42, stratify=Y_true)
    _, gru_probs_te, _, _ = train_test_split(Y_gru, Y_true, test_size=0.2, random_state=42, stratify=Y_true)
    _, unc_te, _, _ = train_test_split(Unc_ode, Y_true, test_size=0.2, random_state=42, stratify=Y_true)
    
    ode_auc = roc_auc_score(y_te, ode_probs_te)
    gru_auc = roc_auc_score(y_te, gru_probs_te)
    ode_ece = calculate_ece(y_te, ode_probs_te)
    gru_ece = calculate_ece(y_te, gru_probs_te)
    mean_unc = np.mean(unc_te)
    
    return ode_auc, gru_auc, ode_ece, gru_ece, mean_unc

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()
    
    # Load frozen optimized models
    ode_model = CEMREvidentialODE(latent_dim=16).to(device)
    ode_model.load_state_dict(torch.load("checkpoints/cemr_fair_final.pth", map_location=device))
    ode_model.eval()
    
    gru_model = GRUBaselineNet(input_dim=4, hidden_dim=16).to(device)
    gru_model.load_state_dict(torch.load("checkpoints/baseline_gru.pth", map_location=device))
    gru_model.eval()
    
    stress_levels = [0.0, 0.30, 0.50, 0.75]
    
    print("\n=====================================================================================")
    print("Clinical telemetry sparsity robustness stress test")
    print("=====================================================================================")
    print(f"{'Data Dropout Rate':<20} | {'ODE AUROC':<10} | {'GRU AUROC':<10} | {'ODE ECE':<8} | {'GRU ECE':<8} | {'ODE Epistemic Unc'}")
    print("-" * 90)
    
    for level in stress_levels:
        ode_auc, gru_auc, ode_ece, gru_ece, mean_unc = evaluate_models_under_stress(
            level, device, loader, ode_model, gru_model
        )
        print(f"{int(level*100):>3}% Random Dropout   | {ode_auc:.4f}    | {gru_auc:.4f}    | {ode_ece:.4f}  | {gru_ece:.4f}  | {mean_unc:.4f}")

if __name__ == "__main__":
    main()
