import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import CEMREvidentialODE
from utils.metrics import calculate_ece

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def apply_contiguous_blackout(x, window_len=15):
    x_stressed = x.clone()
    batch_size, seq_len, _ = x.shape
    start_idx = seq_len // 2 - window_len // 2
    end_idx = start_idx + window_len
    x_stressed[:, start_idx:end_idx, :] = 0.0
    return x_stressed

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()
    
    model = CEMREvidentialODE(latent_dim=16).to(device)
    model.load_state_dict(torch.load("checkpoints/cemr_fair_final.pth", map_location=device))
    model.eval()
    
    all_probs, all_y, all_unc = [], [], []
    
    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, t, y, mask = x.to(device), t.to(device), y.to(device), mask.to(device)
            stressed_x = apply_contiguous_blackout(x, window_len=15)
            
            full_traj, logits_y, params = model(stressed_x, torch.linspace(0.0, 1.0, steps=x.shape[1], device=device))
            idx_last = mask.sum(dim=1) - 1
            
            final_logits = logits_y[torch.arange(logits_y.size(0)), idx_last].squeeze(-1)
            probs = torch.sigmoid(final_logits).cpu().numpy()
            
            alpha, beta = params[2], params[3]
            epistemic_unc = torch.mean(beta / (alpha - 1 + 1e-6), dim=-1)
            final_unc = epistemic_unc[torch.arange(epistemic_unc.size(0)), idx_last].cpu().numpy()
            
            all_probs.append(probs)
            all_y.append(y.cpu().numpy())
            all_unc.append(final_unc)
            
    Y_prob = np.concatenate(all_probs)
    Y_true = np.concatenate(all_y)
    Unc_val = np.concatenate(all_unc)
    
    # Isolate onto identical validation footprint splits
    _, y_prob_te, _, y_te, _, unc_te = train_test_split(
        Y_prob, Y_true, Unc_val, test_size=0.2, random_state=42, stratify=Y_true
    )
    
    print("\n=====================================================================================")
    print("CITE-ODE selective prediction validation (6-hour blackout stress)")
    print("=====================================================================================")
    print(f"{'Abstention Rate (Dropped)':<25} | {'Retained Sample N':<18} | {'Retained AUROC':<15} | {'Retained ECE'}")
    print("-" * 85)
    
    # Sweep progressive decision coverage thresholds
    abstention_thresholds = [0.0, 0.10, 0.20, 0.30]
    total_samples = len(y_te)
    
    for t in abstention_thresholds:
        cutoff = np.quantile(unc_te, 1.0 - t)
        retained_mask = unc_te <= cutoff
        
        retained_y = y_te[retained_mask]
        retained_probs = y_prob_te[retained_mask]
        
        r_auc = roc_auc_score(retained_y, retained_probs)
        r_ece = calculate_ece(retained_y, retained_probs)
        
        print(f"{int(t*100):>2}% Most Uncertain Dropped   | {len(retained_y):<18} | {r_auc:.4f}          | {r_ece:.4f}")

if __name__ == "__main__":
    main()
