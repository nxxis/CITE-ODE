import os, numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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

def evaluate_blackout_one_model(model_path, device, loader):
    model = CEMREvidentialODE(latent_dim=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, t, y, mask = x.to(device), t.to(device), y.to(device), mask.to(device)
            stressed_x = apply_contiguous_blackout(x, window_len=15)
            _, logits_y, _ = model(stressed_x, torch.linspace(0.0, 1.0, steps=x.shape[1], device=device))
            idx_last = mask.sum(dim=1) - 1
            probs = torch.sigmoid(logits_y[torch.arange(logits_y.size(0)), idx_last].squeeze(-1)).cpu().numpy()
            all_probs.extend(probs)
            all_y.extend(y.cpu().numpy())
    X = np.array(all_probs).reshape(-1, 1)
    Y = np.array(all_y)
    _, y_prob_te, _, y_te = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    # Flatten to 1D for metrics
    y_prob_te = y_prob_te.flatten()
    y_te = y_te.flatten()
    return {'auroc': roc_auc_score(y_te, y_prob_te), 'ece': calculate_ece(y_te, y_prob_te)}

def main():
    seeds = [42, 123, 456, 789, 101112]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()
    results = {'auroc': [], 'ece': []}
    for seed in seeds:
        model_path = f"checkpoints/cemr_fair_seed{seed}.pth"
        if not os.path.exists(model_path):
            print(f"Warning: model not found at {model_path}; skipping seed {seed}.")
            continue
        metrics = evaluate_blackout_one_model(model_path, device, loader)
        results['auroc'].append(metrics['auroc'])
        results['ece'].append(metrics['ece'])
        print(f"Seed {seed} results: AUROC={metrics['auroc']:.4f}, ECE={metrics['ece']:.4f}")
    print("\nBlackout multi-seed results:")
    print(f"AUROC: {np.mean(results['auroc']):.4f} ± {np.std(results['auroc']):.4f}")
    print(f"ECE:   {np.mean(results['ece']):.4f} ± {np.std(results['ece']):.4f}")

if __name__ == "__main__":
    main()
