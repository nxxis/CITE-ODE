import os, sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clinical_mimic import get_mimic_dataloader
from scripts.train_ode_bce_seed import ODEBCEModel
from utils.metrics import calculate_ece

def evaluate_one_seed(model_path, device, loader):
    model = ODEBCEModel(latent_dim=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, t, y, mask = x.to(device), t.to(device), y.to(device), mask.to(device)
            t_eval = torch.linspace(0.0, 1.0, steps=x.shape[1], device=device)
            logits = model(x, t_eval)
            idx_last = mask.sum(dim=1) - 1
            final_logits = logits[torch.arange(logits.size(0)), idx_last].squeeze(-1)
            probs = torch.sigmoid(final_logits).cpu().numpy()
            all_probs.extend(probs)
            all_y.extend(y.cpu().numpy())
    X = np.array(all_probs).reshape(-1,1)
    Y = np.array(all_y)
    _, y_prob_te, _, y_te = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
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
    results = {k: [] for k in ['auroc','auprc','ece','brier']}
    for seed in seeds:
        path = f"checkpoints/ode_bce_seed{seed}.pth"
        if not os.path.exists(path):
            print(f"Missing {path}, skipping seed {seed}")
            continue
        metrics = evaluate_one_seed(path, device, loader)
        for k,v in metrics.items():
            results[k].append(v)
        print(f"ODE+BCE seed {seed}: AUROC={metrics['auroc']:.4f}, ECE={metrics['ece']:.4f}")
    print("\nODE+BCE (5 seeds) mean ± std:")
    for k,v in results.items():
        print(f"{k.upper()}: {np.mean(v):.4f} ± {np.std(v):.4f}")

if __name__ == "__main__":
    main()
