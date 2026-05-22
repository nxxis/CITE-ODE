"""Compute subgroup performance and epistemic uncertainty under blackout.

This script evaluates CEMR checkpoints across multiple seeds, applies a
contiguous blackout stress to inputs, and reports subgroup AUROC, ECE,
and mean epistemic uncertainty per demographic subgroup.
Results are saved to `results/multiseed_subgroups.csv`.
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
from utils.metrics import calculate_ece

def apply_contiguous_blackout(x, window_len=15):
    """Zero a contiguous window in each input trajectory to simulate
    telemetry blackout.

    Args:
        x: Tensor shaped (batch, seq_len, features)
        window_len: Length of blackout window in timesteps
    """
    x_stressed = x.clone()
    batch_size, seq_len, _ = x.shape
    start_idx = seq_len // 2 - window_len // 2
    end_idx = start_idx + window_len
    x_stressed[:, start_idx:end_idx, :] = 0.0
    return x_stressed

# ------------------------------------------------------------
def get_probs_unc_subgroups(model, device, loader):
    """Collect probs, targets, epistemic uncertainty, and demographics.

    Returns arrays: (probs, y_true, unc, gender, age, race)
    """
    model.eval()
    all_probs, all_y, all_unc = [], [], []
    all_gender, all_age, all_race = [], [], []
    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, t, d, mask = x.to(device), t.to(device), d.to(device), mask.to(device)
            stressed_x = apply_contiguous_blackout(x, window_len=15)
            full_traj, logits_y, params = model(stressed_x, torch.linspace(0.0, 1.0, steps=x.shape[1], device=device))
            idx_last = mask.sum(dim=1) - 1
            final_logits = logits_y[torch.arange(logits_y.size(0)), idx_last].squeeze(-1)
            probs = torch.sigmoid(final_logits).cpu().numpy()
            alpha, beta = params[2], params[3]
            epistemic_unc = torch.mean(beta / (alpha - 1 + 1e-6), dim=-1)
            final_unc = epistemic_unc[torch.arange(epistemic_unc.size(0)), idx_last].cpu().numpy()
            all_probs.extend(probs)
            all_y.extend(y.cpu().numpy())
            all_unc.extend(final_unc)
            all_gender.extend(d[:,1].cpu().numpy())
            all_age.extend(d[:,0].cpu().numpy())
            all_race.extend(d[:,2].cpu().numpy())
    return (np.array(all_probs), np.array(all_y), np.array(all_unc),
            np.array(all_gender), np.array(all_age), np.array(all_race))

def evaluate_one_seed(model_path, device, loader):
    model = CEMREvidentialODE(latent_dim=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    probs, y_true, unc, gender, age, race = get_probs_unc_subgroups(model, device, loader)
    # Train/test split (same as evaluate_rigor.py)
    _, probs_te, _, y_te, _, unc_te, _, gender_te, _, age_te, _, race_te = train_test_split(
        probs.reshape(-1,1), y_true, unc.reshape(-1,1), gender, age, race,
        test_size=0.2, random_state=42, stratify=y_true
    )
    probs_te = probs_te.flatten()
    y_te = y_te.flatten()
    unc_te = unc_te.flatten()
    gender_te = gender_te.flatten()
    age_te = age_te.flatten()
    race_te = race_te.flatten()
    median_age = np.median(age_te)
    subgroups = {
        "Female": gender_te == 0,
        "Male": gender_te == 1,
        "Younger": age_te < median_age,
        "Older": age_te >= median_age,
        "White": race_te == 0,
        "Black": race_te == 1,
        "Hispanic": race_te == 2,
        "Asian": race_te == 3
    }
    results = {}
    for name, mask in subgroups.items():
        if np.sum(mask) > 5 and len(np.unique(y_te[mask])) > 1:
            au = roc_auc_score(y_te[mask], probs_te[mask])
            ece = calculate_ece(y_te[mask], probs_te[mask])
            unc_mean = np.mean(unc_te[mask])
            results[name] = {"AUROC": au, "ECE": ece, "Unc": unc_mean, "N": int(np.sum(mask))}
        else:
            results[name] = {"AUROC": np.nan, "ECE": np.nan, "Unc": np.nan, "N": int(np.sum(mask))}
    return results

def main():
    seeds = [42, 123, 456, 789, 101112]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()
    all_results = {name: [] for name in ["Female","Male","Younger","Older","White","Black","Hispanic","Asian"]}
    for seed in seeds:
        model_path = f"checkpoints/cemr_fair_seed{seed}.pth"
        if not os.path.exists(model_path):
            import logging
            logging.warning("%s not found, skipping seed %s", model_path, seed)
            continue
        res = evaluate_one_seed(model_path, device, loader)
        for name in all_results:
            all_results[name].append(res[name])
    import logging
    logging.info("Multi-seed subgroup results (mean ± std across seeds):")
    logging.info("%s", f"{'Subgroup':<10} {'N (avg)':<10} {'AUROC':<14} {'ECE':<14} {'Epistemic Unc':<14}")
    for name in all_results:
        n_vals = [r["N"] for r in all_results[name] if not np.isnan(r["AUROC"])]
        au_vals = [r["AUROC"] for r in all_results[name] if not np.isnan(r["AUROC"])]
        ece_vals = [r["ECE"] for r in all_results[name] if not np.isnan(r["ECE"])]
        unc_vals = [r["Unc"] for r in all_results[name] if not np.isnan(r["Unc"])]
        if au_vals:
            au_mean, au_std = np.mean(au_vals), np.std(au_vals)
            ece_mean, ece_std = np.mean(ece_vals), np.std(ece_vals)
            unc_mean, unc_std = np.mean(unc_vals), np.std(unc_vals)
            n_mean = int(np.mean(n_vals))
            logging.info(
                "%s %s %s %s %s",
                f"{name:<10}",
                f"{n_mean:<10}",
                f"{au_mean:.3f}±{au_std:.3f}",
                f"{ece_mean:.3f}±{ece_std:.3f}",
                f"{unc_mean:.3f}±{unc_std:.3f}",
            )
        else:
            logging.info(f"{name:<10} {'N/A':<10} N/A         N/A         N/A")
    os.makedirs("results", exist_ok=True)
    with open("results/multiseed_subgroups.csv", "w") as f:
        f.write("Seed,Subgroup,AUROC,ECE,Unc,N\n")
        for i, seed in enumerate(seeds):
            for name in all_results:
                r = all_results[name][i]
                if not np.isnan(r["AUROC"]):
                    f.write(f"{seed},{name},{r['AUROC']:.4f},{r['ECE']:.4f},{r['Unc']:.4f},{r['N']}\n")
    logging.info("Per-seed results saved to results/multiseed_subgroups.csv")

if __name__ == "__main__":
    main()
