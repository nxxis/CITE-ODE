import os, numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import CEMREvidentialODE
from utils.metrics import calculate_ece

def apply_contiguous_blackout(x, window_len=15):
    x_stressed = x.clone()
    batch_size, seq_len, _ = x.shape
    start_idx = seq_len // 2 - window_len // 2
    end_idx = start_idx + window_len
    x_stressed[:, start_idx:end_idx, :] = 0.0
    return x_stressed

def get_probs_and_unc(model, device, loader):
    """Return probabilities and epistemic uncertainty for all samples."""
    all_probs, all_y, all_unc = [], [], []
    model.eval()
    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, t, mask = x.to(device), t.to(device), mask.to(device)
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
    return np.array(all_probs), np.array(all_y), np.array(all_unc)

def evaluate_selective_one_model(model_path, device, loader, coverage=0.8):
    model = CEMREvidentialODE(latent_dim=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    probs, y_true, unc = get_probs_and_unc(model, device, loader)
    # Split into train/test (same as before)
    _, probs_te, _, y_te, _, unc_te = train_test_split(
        probs.reshape(-1,1), y_true, unc, test_size=0.2, random_state=42, stratify=y_true
    )
    probs_te = probs_te.flatten()
    y_te = y_te.flatten()
    unc_te = unc_te.flatten()
    
    # CITE-ODE selective ECE at given coverage
    cutoff = np.quantile(unc_te, coverage)
    retained_mask = unc_te <= cutoff
    cite_ece = calculate_ece(y_te[retained_mask], probs_te[retained_mask])
    
    # Stratified random control (preserve prevalence)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=int(len(y_te)*coverage), random_state=42)
    for train_idx, _ in sss.split(y_te, y_te):
        control_ece = calculate_ece(y_te[train_idx], probs_te[train_idx])
    
    return cite_ece, control_ece

def main():
    seeds = [42, 123, 456, 789, 101112]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()
    cite_eces = []
    control_eces = []
    for seed in seeds:
        model_path = f"checkpoints/cemr_fair_seed{seed}.pth"
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found, skipping seed {seed}")
            continue
        cite_ece, control_ece = evaluate_selective_one_model(model_path, device, loader, coverage=0.8)
        cite_eces.append(cite_ece)
        control_eces.append(control_ece)
        print(f"Seed {seed}: CITE-ECE={cite_ece:.4f}, Control ECE={control_ece:.4f}")
    
    if len(cite_eces) == 0:
        print("No models found. Exiting.")
        return
    
    print("\n===== Multi‑Seed Selective Prediction (80% Coverage) =====")
    print(f"CITE-ODE ECE: {np.mean(cite_eces):.4f} ± {np.std(cite_eces):.4f}")
    print(f"Stratified Control ECE: {np.mean(control_eces):.4f} ± {np.std(control_eces):.4f}")

if __name__ == "__main__":
    main()
