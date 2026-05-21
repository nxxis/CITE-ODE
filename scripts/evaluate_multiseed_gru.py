"""Evaluate GRU baseline checkpoints across multiple random seeds.

Loads GRU checkpoints from `checkpoints/baseline_gru_seed{seed}.pth`, runs them
on the canonical dataloader and reports AUROC, ECE, and Brier score aggregates.
"""

import os, numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from data.clinical_mimic import get_mimic_dataloader
from utils.metrics import calculate_ece
from scripts.train_gru_seed import GRUBaselineNet

def evaluate_gru_one_model(model_path, device, loader):
    model = GRUBaselineNet(input_dim=4, hidden_dim=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, mask = x.to(device), mask.to(device)
            _, logits = model(x, mask)
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_probs.extend(probs)
            all_y.extend(y.cpu().numpy())
    X = np.array(all_probs).reshape(-1, 1)
    Y = np.array(all_y)
    _, y_prob_te, _, y_te = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    y_prob_te = y_prob_te.flatten()
    y_te = y_te.flatten()
    return {
        'auroc': roc_auc_score(y_te, y_prob_te),
        'ece': calculate_ece(y_te, y_prob_te),
        'brier': brier_score_loss(y_te, y_prob_te)
    }

def main():
    """Evaluate saved GRU checkpoints and print mean ± std across seeds."""

    seeds = [42, 123, 456]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()
    results = {'auroc': [], 'ece': [], 'brier': []}
    for seed in seeds:
        model_path = f"checkpoints/baseline_gru_seed{seed}.pth"
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found, skipping seed {seed}")
            continue
        metrics = evaluate_gru_one_model(model_path, device, loader)
        for k, v in metrics.items():
            results[k].append(v)
        print(f"GRU Seed {seed}: AUROC={metrics['auroc']:.4f}, ECE={metrics['ece']:.4f}")
    if not results['auroc']:
        print("No models found.")
        return
    print("\n===== GRU Multi‑Seed Results =====")
    print(f"AUROC: {np.mean(results['auroc']):.4f} ± {np.std(results['auroc']):.4f}")
    print(f"ECE:   {np.mean(results['ece']):.4f} ± {np.std(results['ece']):.4f}")
    print(f"Brier: {np.mean(results['brier']):.4f} ± {np.std(results['brier']):.4f}")

if __name__ == "__main__":
    main()
