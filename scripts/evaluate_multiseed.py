import os, numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import CEMREvidentialODE
from utils.metrics import calculate_ece

def evaluate_one_model(model_path, device, loader):
    model = CEMREvidentialODE(latent_dim=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, t, mask = x.to(device), t.to(device), mask.to(device)
            full_traj, logits_y, _ = model(x, torch.linspace(0.0, 1.0, steps=x.shape[1], device=device))
            idx_last = mask.sum(dim=1) - 1
            final_logits = logits_y[torch.arange(logits_y.size(0)), idx_last].squeeze(-1)
            probs = torch.sigmoid(final_logits).cpu().numpy()
            all_probs.extend(probs)
            all_y.extend(y.cpu().numpy())
    X = np.array(all_probs).reshape(-1, 1)
    Y = np.array(all_y)
    # Use same test split as in evaluate_rigor.py
    _, y_prob_te, _, y_te = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    # Flatten to 1D arrays for metrics
    y_prob_te = y_prob_te.flatten()
    y_te = y_te.flatten()
    return {
        'auroc': roc_auc_score(y_te, y_prob_te),
        'auprc': average_precision_score(y_te, y_prob_te),
        'ece': calculate_ece(y_te, y_prob_te),
        'brier': brier_score_loss(y_te, y_prob_te)
    }

def main():
    seeds = [42, 123, 456, 789, 101112]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()
    results = {k: [] for k in ['auroc', 'auprc', 'ece', 'brier']}
    for seed in seeds:
        model_path = f"checkpoints/cemr_fair_seed{seed}.pth"
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found, skipping seed {seed}")
            continue
        metrics = evaluate_one_model(model_path, device, loader)
        for k, v in metrics.items():
            results[k].append(v)
        print(f"Seed {seed}: AUROC={metrics['auroc']:.4f}, ECE={metrics['ece']:.4f}")

    print("\n===== Multi‑Seed Aggregated Results =====")
    for k, vlist in results.items():
        mean_val = np.mean(vlist)
        std_val = np.std(vlist)
        print(f"{k.upper()}: {mean_val:.4f} ± {std_val:.4f}")

if __name__ == "__main__":
    main()
