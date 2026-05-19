import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.calibration import calibration_curve
from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import CEMREvidentialODE
from run_baseline_gru import GRUBaselineNet
from run_baseline_grud import GRUDBaselineNet
from evaluate_blackout_stress import apply_contiguous_blackout

def get_probs_blackout(model, loader, device, is_ode=True):
    all_probs, all_y = [], []
    model.eval()
    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, t, y, mask = x.to(device), t.to(device), y.to(device), mask.to(device)
            stressed_x = apply_contiguous_blackout(x, window_len=15)
            if is_ode:
                _, logits, _ = model(stressed_x, torch.linspace(0.0, 1.0, steps=x.shape[1], device=device))
                idx_last = mask.sum(dim=1) - 1
                logits = logits[torch.arange(logits.size(0)), idx_last].squeeze(-1)
            else:
                # For GRU-D, we need to pass time tensor as well
                if hasattr(model, 'grud_cell'):  # GRU-D model
                    _, logits = model(stressed_x, t, mask)
                else:  # Vanilla GRU
                    _, logits = model(stressed_x, mask)
                logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_y.extend(y.cpu().numpy())
    return np.array(all_probs), np.array(all_y)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()

    # Load models (use seed 42 for consistency)
    cite_model = CEMREvidentialODE(latent_dim=16).to(device)
    cite_model.load_state_dict(torch.load("checkpoints/cemr_fair_seed42.pth", map_location=device))

    gru_model = GRUBaselineNet(input_dim=4, hidden_dim=16).to(device)
    gru_model.load_state_dict(torch.load("checkpoints/baseline_gru_seed42.pth", map_location=device))

    grud_model = GRUDBaselineNet(input_dim=4, hidden_dim=16).to(device)
    grud_model.load_state_dict(torch.load("checkpoints/baseline_grud.pth", map_location=device))

    cite_probs, cite_y = get_probs_blackout(cite_model, loader, device, is_ode=True)
    gru_probs, gru_y = get_probs_blackout(gru_model, loader, device, is_ode=False)
    grud_probs, grud_y = get_probs_blackout(grud_model, loader, device, is_ode=False)

    # Compute calibration curves
    n_bins = 10
    cite_frac_pos, cite_mean_pred = calibration_curve(cite_y, cite_probs, n_bins=n_bins, strategy='uniform')
    gru_frac_pos, gru_mean_pred = calibration_curve(gru_y, gru_probs, n_bins=n_bins, strategy='uniform')
    grud_frac_pos, grud_mean_pred = calibration_curve(grud_y, grud_probs, n_bins=n_bins, strategy='uniform')

    # ECE values (from earlier multi‑seed, we can annotate with means)
    cite_ece = 0.018   # from multi‑seed blackout ECE
    gru_ece = 0.013
    grud_ece = 0.014

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.7)
    plt.plot(cite_mean_pred, cite_frac_pos, 'o-', color='#1b7a43', lw=2, ms=6, label=f'CITE-ODE (ECE={cite_ece:.3f})')
    plt.plot(gru_mean_pred, gru_frac_pos, 's-', color='#d95f02', lw=2, ms=6, label=f'GRU (ECE={gru_ece:.3f})')
    plt.plot(grud_mean_pred, grud_frac_pos, '^-', color='#7570b3', lw=2, ms=6, label=f'GRU-D (ECE={grud_ece:.3f})')

    plt.xlabel('Mean predicted probability')
    plt.ylabel('Observed mortality fraction')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    plt.title('Reliability diagram (6‑hour blackout)')
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig('plots/figure1_reliability.pdf', dpi=300)
    plt.savefig('plots/figure1_reliability.png', dpi=300)
    print("Figure 1 saved: plots/figure1_reliability.pdf")

if __name__ == "__main__":
    main()
