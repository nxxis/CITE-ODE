import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.calibration import calibration_curve
from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import CEMREvidentialODE
from run_baseline_gru import GRUBaselineNet
from run_baseline_grud import GRUDBaselineNet
from evaluate_blackout_stress import apply_contiguous_blackout
import os

os.makedirs("plots", exist_ok=True)

# ------------------------------
# Figure 1: Reliability Diagram (6‑hour blackout, single‑seed representative)
# ------------------------------
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
                if hasattr(model, 'grud_cell'):  # GRU-D
                    _, logits = model(stressed_x, t, mask)
                else:  # Vanilla GRU
                    _, logits = model(stressed_x, mask)
                logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_y.extend(y.cpu().numpy())
    return np.array(all_probs), np.array(all_y)

def generate_figure1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()

    cite_model = CEMREvidentialODE(latent_dim=16).to(device)
    cite_model.load_state_dict(torch.load("checkpoints/cemr_fair_seed42.pth", map_location=device))
    gru_model = GRUBaselineNet(input_dim=4, hidden_dim=16).to(device)
    gru_model.load_state_dict(torch.load("checkpoints/baseline_gru_seed42.pth", map_location=device))
    grud_model = GRUDBaselineNet(input_dim=4, hidden_dim=16).to(device)
    grud_model.load_state_dict(torch.load("checkpoints/baseline_grud.pth", map_location=device))

    cite_probs, cite_y = get_probs_blackout(cite_model, loader, device, is_ode=True)
    gru_probs, gru_y = get_probs_blackout(gru_model, loader, device, is_ode=False)
    grud_probs, grud_y = get_probs_blackout(grud_model, loader, device, is_ode=False)

    n_bins = 10
    cite_frac, cite_pred = calibration_curve(cite_y, cite_probs, n_bins=n_bins, strategy='uniform')
    gru_frac, gru_pred = calibration_curve(gru_y, gru_probs, n_bins=n_bins, strategy='uniform')
    grud_frac, grud_pred = calibration_curve(grud_y, grud_probs, n_bins=n_bins, strategy='uniform')

    # Multi‑seed blackout ECE values (from your final runs)
    cite_ece = 0.018
    gru_ece = 0.016
    grud_ece = 0.014

    plt.figure(figsize=(5,5))
    plt.plot([0,1], [0,1], 'k--', label='Perfect calibration', alpha=0.7)
    plt.plot(cite_pred, cite_frac, 'o-', color='#1b7a43', lw=2, ms=6, label=f'CITE-ODE (ECE={cite_ece:.3f})')
    plt.plot(gru_pred, gru_frac, 's-', color='#d95f02', lw=2, ms=6, label=f'GRU (ECE={gru_ece:.3f})')
    plt.plot(grud_pred, grud_frac, '^-', color='#7570b3', lw=2, ms=6, label=f'GRU-D (ECE={grud_ece:.3f})')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Observed mortality fraction')
    plt.xlim(0,1); plt.ylim(0,1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.title('Reliability diagram (6‑hour blackout)')
    plt.tight_layout()
    plt.savefig('plots/figure1_reliability.pdf', dpi=300)
    plt.savefig('plots/figure1_reliability.png', dpi=300)
    print("✅ Figure 1 saved (reliability diagram).")

# ------------------------------
# Figure 2: Selective Prediction (Multi‑Seed, with variance bands)
# ------------------------------
def generate_figure2():
    coverage = np.array([100, 90, 80, 70])
    # From your multi‑seed sweep results
    cite_mean = np.array([0.0177, 0.0087, 0.0085, 0.0081])
    cite_std  = np.array([0.0074, 0.0064, 0.0074, 0.0054])
    ctrl_mean = np.array([0.0177, 0.0194, 0.0195, 0.0174])
    ctrl_std  = np.array([0.0074, 0.0104, 0.0100, 0.0044])
    prevalence = np.array([12.07, 6.84, 5.39, 4.72])

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 300, 'font.family': 'serif'})

    fig, ax1 = plt.subplots(figsize=(8,5.5))
    ax1.plot(coverage, cite_mean, 'o-', color='#2563eb', lw=2.5, label='CITE-ODE (uncertainty filter)')
    ax1.fill_between(coverage, cite_mean - cite_std, cite_mean + cite_std, color='#2563eb', alpha=0.2)
    ax1.plot(coverage, ctrl_mean, 's--', color='#64748b', lw=2, label='Stratified random control')
    ax1.fill_between(coverage, ctrl_mean - ctrl_std, ctrl_mean + ctrl_std, color='#64748b', alpha=0.2)
    ax1.invert_xaxis()
    ax1.set_xlabel('Decision coverage (%)')
    ax1.set_ylabel('Conditional Expected Calibration Error (ECE)')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(coverage, prevalence, '^-', color='#dc2626', alpha=0.7, label='Observed mortality (%)')
    ax2.set_ylabel('Mortality prevalence (%)', color='#dc2626')
    ax2.tick_params(axis='y', labelcolor='#dc2626')
    ax2.grid(False)

    plt.title('Selective prediction (5‑seed mean ± 1σ)')
    plt.tight_layout()
    plt.savefig('plots/figure2_selective_variance.pdf', bbox_inches='tight')
    plt.savefig('plots/figure2_selective_variance.png', dpi=300)
    print("✅ Figure 2 saved (selective prediction with variance bands).")

# ------------------------------
# Figure 3: Subgroup Scatter Plot (Trade‑off between AUROC and ECE)
# ------------------------------
def generate_figure3():
    groups = ['Female', 'Male', 'White', 'Black', 'Hispanic', 'Asian']
    n_counts = [900, 1100, 1300, 500, 240, 180]
    auroc = [0.843, 0.822, 0.825, 0.848, 0.814, 0.875]
    ece   = [0.024, 0.019, 0.017, 0.052, 0.031, 0.053]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 300, 'font.family': 'serif'})
    
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sizes = [n * 0.8 for n in n_counts]   # scale marker size by N
    
    scatter = ax.scatter(auroc, ece, s=sizes, color='#1b7a43', alpha=0.7, edgecolors='k', linewidth=1.5)
    
    for i, group in enumerate(groups):
        ax.annotate(f"{group}\n(N={n_counts[i]})",
                    (auroc[i], ece[i]),
                    xytext=(10, -5), textcoords='offset points',
                    fontsize=10, ha='left', va='center')
    
    ax.set_xlabel('Discriminative Performance (AUROC $\\rightarrow$)')
    ax.set_ylabel('Calibration Error (ECE, $\\leftarrow$ lower is better)')
    ax.set_title('Subgroup Performance Trade-offs (CITE-ODE)')
    # Optional reference lines for "ideal" zone (high AUROC, low ECE)
    ax.axhline(0.02, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.83, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/figure3_subgroup_scatter.pdf', dpi=300)
    plt.savefig('plots/figure3_subgroup_scatter.png', dpi=300)
    print("✅ Figure 3 saved (subgroup scatter plot).")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    generate_figure1()
    generate_figure2()
    generate_figure3()
    print("\n✅ All figures saved to 'plots/' directory.")
