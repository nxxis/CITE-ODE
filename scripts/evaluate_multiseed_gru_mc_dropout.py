import os
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.metrics import calculate_ece
from evaluate_blackout_stress import apply_contiguous_blackout
from data.clinical_mimic import get_mimic_dataloader
from scripts.train_gru_mc_dropout_seed import GRUMCDropoutNet

def get_mc_predictions(model, device, loader, num_passes=30):
    model.train()  # keep dropout active
    all_means, all_vars, all_y = [], [], []
    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, t, y, mask = x.to(device), t.to(device), y.to(device), mask.to(device)
            stressed_x = apply_contiguous_blackout(x, window_len=15)
            probs_list = []
            for _ in range(num_passes):
                logits = model(stressed_x, mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                probs_list.append(probs)
            probs_stack = np.stack(probs_list, axis=0)
            mean_probs = np.mean(probs_stack, axis=0)
            var_probs = np.var(probs_stack, axis=0)
            all_means.extend(mean_probs)
            all_vars.extend(var_probs)
            all_y.extend(y.cpu().numpy())
    return np.array(all_means), np.array(all_vars), np.array(all_y)

def evaluate_one_seed(model_path, device, loader, num_passes=30):
    model = GRUMCDropoutNet(input_dim=4, hidden_dim=16, dropout=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    probs, unc, y_true = get_mc_predictions(model, device, loader, num_passes)
    # Train/test split
    _, probs_te, _, y_te, _, unc_te = train_test_split(
        probs.reshape(-1,1), y_true, unc, test_size=0.2, random_state=42, stratify=y_true
    )
    probs_te = probs_te.flatten()
    y_te = y_te.flatten()
    unc_te = unc_te.flatten()
    coverages = [1.0, 0.9, 0.8, 0.7]
    ece_dict = {}
    for cov in coverages:
        cutoff = np.quantile(unc_te, cov)
        mask = unc_te <= cutoff
        ece_dict[cov] = calculate_ece(y_te[mask], probs_te[mask])
    return ece_dict

def main():
    seeds = [42, 123, 456, 789, 101112]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()
    results = {cov: [] for cov in [1.0, 0.9, 0.8, 0.7]}
    for seed in seeds:
        model_path = f"checkpoints/gru_mc_dropout_seed{seed}.pth"
        if not os.path.exists(model_path):
            print(f"Missing {model_path}, skipping seed {seed}")
            continue
        ece_dict = evaluate_one_seed(model_path, device, loader)
        for cov, ece in ece_dict.items():
            results[cov].append(ece)
        print(f"MC Dropout GRU seed {seed}: ECEc@80% = {ece_dict[0.8]:.4f}")
    print("\n===== MC Dropout GRU Multi‑Seed Selective Prediction =====")
    for cov in [1.0, 0.9, 0.8, 0.7]:
        mean_ece = np.mean(results[cov])
        std_ece = np.std(results[cov])
        print(f"Coverage {cov:.0%}: ECEc = {mean_ece:.4f} ± {std_ece:.4f}")

if __name__ == "__main__":
    main()
