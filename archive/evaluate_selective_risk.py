import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import brier_score_loss

from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import CEMREvidentialODE
from utils.metrics import calculate_ece, calculate_brier_skill_score

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

    ode_model = CEMREvidentialODE(latent_dim=16).to(device)
    ode_model.load_state_dict(torch.load("checkpoints/cemr_fair_final.pth", map_location=device))
    ode_model.eval()

    all_probs, all_y, all_unc = [], [], []

    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, t, mask = x.to(device), t.to(device), mask.to(device)
            stressed_x = apply_contiguous_blackout(x, window_len=15)

            full_traj, logits_y, params = ode_model(stressed_x, torch.linspace(0.0, 1.0, steps=x.shape[1], device=device))
            idx_last = mask.sum(dim=1) - 1

            final_logits = logits_y[torch.arange(logits_y.size(0)), idx_last].squeeze(-1)
            probs = torch.sigmoid(final_logits).cpu().numpy()

            alpha, beta = params[2], params[3]
            epistemic_unc = torch.mean(beta / (alpha - 1 + 1e-6), dim=-1)
            final_unc = epistemic_unc[torch.arange(epistemic_unc.size(0)), idx_last].cpu().numpy()

            all_probs.append(probs)
            all_y.append(y.cpu().numpy())
            all_unc.append(final_unc)

    Y_prob, Y_true, Unc_val = np.concatenate(all_probs), np.concatenate(all_y), np.concatenate(all_unc)
    _, y_prob_te, _, y_te, _, unc_te = train_test_split(Y_prob, Y_true, Unc_val, test_size=0.2, random_state=42, stratify=Y_true)

    print("\n=========================================================================================================")
    print("Table III: selective prediction and stratified random control (6-hour blackout)")
    print("=========================================================================================================")
    print(f"{'Decision Coverage':<18} | {'Retained N':<12} | {'Prevalence':<12} | {'CITE-ODE ECE':<14} | {'Control ECE':<14} | {'BSS (Skill)'}")
    print("-" * 105)

    coverage_tiers = [1.0, 0.90, 0.80, 0.70]
    for cov in coverage_tiers:
        # CITE-ODE Uncertainty Filtering
        cutoff = np.quantile(unc_te, cov)
        retained_mask = unc_te <= cutoff
        r_y = y_te[retained_mask]
        r_probs = y_prob_te[retained_mask]
        
        cond_ece = calculate_ece(r_y, r_probs)
        obs_mortality = np.mean(r_y)
        bss = calculate_brier_skill_score(r_y, r_probs)
        
        # Stratified Random Control (matching original 12.07% prevalence)
        if cov == 1.0:
            rand_ece = cond_ece
        else:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=len(r_y), random_state=42)
            for retain_idx, _ in sss.split(y_te, y_te):
                rand_ece = calculate_ece(y_te[retain_idx], y_prob_te[retain_idx])

        print(f"{int(cov*100):>3}% Coverage       | {len(r_y):<12} | {obs_mortality:.2%}       | {cond_ece:.4f}         | {rand_ece:.4f}         | {bss:.4f}")

if __name__ == "__main__":
    main()
