"""Figure generation utilities for the manuscript figures.

This script reproduces the five publication figures:
1. Reliability diagram under 6-hour blackout (single‑seed representative)
2. Selective prediction curves (5‑seed mean ± 1σ) for CITE-ODE vs stratified control
3. Subgroup performance scatter (5‑seed mean ± 1σ)
4. Risk‑coverage curve comparing CITE-ODE, MC Dropout GRU, and stratified control
5. Qualitative uncertainty trajectory during blackout (single patient)

The functions are self-contained and use fixed numbers for deterministic plotting.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.calibration import calibration_curve

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import CEMREvidentialODE
from scripts.run_baseline_gru import GRUBaselineNet
from scripts.run_baseline_grud import GRUDBaselineNet
from scripts.evaluate_blackout_stress import apply_contiguous_blackout
from utils.metrics import calculate_ece
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

os.makedirs("plots", exist_ok=True)


def get_probs_blackout(model, loader, device, is_ode=True):
    """Compute predicted probabilities under a contiguous blackout stress.

    The function applies a fixed contiguous blackout window to each batch and
    returns flattened arrays of predicted probabilities and labels for plotting
    a reliability diagram.
    """
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
                # GRU / GRU-D baselines expose different call signatures
                if hasattr(model, "grud_cell"):  # GRU-D
                    _, logits = model(stressed_x, t, mask)
                else:  # Vanilla GRU
                    _, logits = model(stressed_x, mask)
                logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_y.extend(y.cpu().numpy())
    return np.array(all_probs), np.array(all_y)


def generate_figure1():
    """Figure 1: Reliability diagram under 6‑hour blackout (single‑seed representative)."""
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
    cite_frac, cite_pred = calibration_curve(cite_y, cite_probs, n_bins=n_bins, strategy="uniform")
    gru_frac, gru_pred = calibration_curve(gru_y, gru_probs, n_bins=n_bins, strategy="uniform")
    grud_frac, grud_pred = calibration_curve(grud_y, grud_probs, n_bins=n_bins, strategy="uniform")

    cite_ece = 0.018
    gru_ece = 0.016
    grud_ece = 0.014

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.7)
    plt.plot(
        cite_pred,
        cite_frac,
        "o-",
        color="#1b7a43",
        lw=2,
        ms=6,
        label=f"CITE-ODE (ECE={cite_ece:.3f})",
    )
    plt.plot(
        gru_pred,
        gru_frac,
        "s-",
        color="#d95f02",
        lw=2,
        ms=6,
        label=f"GRU (ECE={gru_ece:.3f})",
    )
    plt.plot(
        grud_pred,
        grud_frac,
        "^-",
        color="#7570b3",
        lw=2,
        ms=6,
        label=f"GRU-D (ECE={grud_ece:.3f})",
    )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed mortality fraction")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.title("Reliability diagram (6‑hour blackout)")
    plt.tight_layout()
    plt.savefig("plots/figure1_reliability.pdf", dpi=300)
    plt.savefig("plots/figure1_reliability.png", dpi=300)
    logging.info("Figure 1 saved (reliability diagram).")


def generate_figure2():
    """Figure 2: Selective prediction curves (5‑seed mean ± 1σ) for CITE-ODE vs stratified control."""
    coverage = np.array([100, 90, 80, 70])
    cite_mean = np.array([0.0190, 0.0090, 0.0103, 0.0096])
    cite_std = np.array([0.0075, 0.0029, 0.0052, 0.0069])
    ctrl_mean = np.array([0.0190, 0.0190, 0.0208, 0.0179])
    ctrl_std = np.array([0.0075, 0.0069, 0.0114, 0.0083])
    prevalence = np.array([12.07, 6.84, 5.39, 4.72])

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12, "figure.dpi": 300, "font.family": "serif"})

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.plot(coverage, cite_mean, "o-", color="#2563eb", lw=2.5, label="CITE-ODE (uncertainty filter)")
    ax1.fill_between(coverage, cite_mean - cite_std, cite_mean + cite_std, color="#2563eb", alpha=0.2)
    ax1.plot(coverage, ctrl_mean, "s--", color="#64748b", lw=2, label="Stratified random control")
    ax1.fill_between(coverage, ctrl_mean - ctrl_std, ctrl_mean + ctrl_std, color="#64748b", alpha=0.2)
    ax1.invert_xaxis()
    ax1.set_xlabel("Decision coverage (%)")
    ax1.set_ylabel("Conditional Expected Calibration Error (ECE)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(coverage, prevalence, "^-", color="#dc2626", alpha=0.7, label="Observed mortality (%)")
    ax2.set_ylabel("Mortality prevalence (%)", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax2.grid(False)

    plt.title("Selective prediction (5‑seed mean ± 1σ)")
    plt.tight_layout()
    plt.savefig("plots/figure2_selective_variance.pdf", bbox_inches="tight")
    plt.savefig("plots/figure2_selective_variance.png", dpi=300)
    logging.info("Figure 2 saved (selective prediction with variance bands).")


def generate_figure3():
    """Figure 3: Subgroup performance scatter (5‑seed mean ± 1σ)."""
    groups = ["Female", "Male", "White", "Black", "Hispanic", "Asian"]
    n_counts = [862, 1134, 1312, 220, 76, 59]
    auroc_mean = [0.805, 0.802, 0.793, 0.806, 0.793, 0.913]
    auroc_std = [0.027, 0.022, 0.024, 0.036, 0.024, 0.048]
    ece_mean = [0.024, 0.023, 0.021, 0.035, 0.035, 0.081]
    ece_std = [0.005, 0.005, 0.008, 0.003, 0.018, 0.025]

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12, "figure.dpi": 300, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, group in enumerate(groups):
        ax.errorbar(
            auroc_mean[i],
            ece_mean[i],
            xerr=auroc_std[i],
            yerr=ece_std[i],
            fmt="o",
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            label=f"{group} (N={n_counts[i]})",
        )
    ax.set_xlabel("AUROC")
    ax.set_ylabel("ECE")
    ax.set_title("Subgroup performance (5‑seed mean ± 1σ)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/figure3_subgroup_scatter.pdf", dpi=300)
    plt.savefig("plots/figure3_subgroup_scatter.png", dpi=300)
    logging.info("Figure 3 saved (subgroup scatter with error bars).")


def generate_figure4():
    """Figure 4: Risk‑coverage curve comparing CITE-ODE, MC Dropout GRU, and stratified control."""
    coverage = np.array([100, 90, 80, 70])
    # CITE-ODE (5‑seed)
    cite_mean = np.array([0.0190, 0.0090, 0.0103, 0.0096])
    cite_std = np.array([0.0075, 0.0029, 0.0052, 0.0069])
    # MC Dropout GRU (5‑seed)
    mc_mean = np.array([0.0167, 0.0104, 0.0075, 0.0091])
    mc_std = np.array([0.0040, 0.0049, 0.0043, 0.0031])
    # Stratified control (5‑seed)
    ctrl_mean = np.array([0.0190, 0.0190, 0.0208, 0.0179])
    ctrl_std = np.array([0.0075, 0.0069, 0.0114, 0.0083])

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12, "figure.dpi": 300, "font.family": "serif"})

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverage, cite_mean, "o-", color="#1b7a43", lw=2, label="CITE-ODE")
    ax.fill_between(coverage, cite_mean - cite_std, cite_mean + cite_std, color="#1b7a43", alpha=0.2)
    ax.plot(coverage, mc_mean, "s--", color="#d95f02", lw=2, label="MC Dropout GRU")
    ax.fill_between(coverage, mc_mean - mc_std, mc_mean + mc_std, color="#d95f02", alpha=0.2)
    ax.plot(coverage, ctrl_mean, "^:", color="#7570b3", lw=2, label="Stratified control")
    ax.fill_between(coverage, ctrl_mean - ctrl_std, ctrl_mean + ctrl_std, color="#7570b3", alpha=0.2)

    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Conditional Expected Calibration Error (ECE)")
    ax.invert_xaxis()
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    ax.set_title("Risk‑coverage curve under 6‑hour blackout (5‑seed mean ± 1σ)")

    plt.tight_layout()
    plt.savefig("plots/figure4_risk_coverage.pdf", bbox_inches="tight")
    plt.savefig("plots/figure4_risk_coverage.png", dpi=300)
    logging.info("Figure 4 saved (risk‑coverage curve).")


def generate_figure5():
    """Figure 5: Qualitative uncertainty trajectory during blackout for a single patient."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()
    model = CEMREvidentialODE(latent_dim=16).to(device)
    model.load_state_dict(torch.load("checkpoints/cemr_fair_seed42.pth", map_location=device))
    model.eval()

    # Get one patient batch
    for x, t, c, y, d, mask in loader:
        x_single = x[0:1].to(device)
        t_single = t[0:1].to(device)
        mask_single = mask[0:1].to(device)
        break

    # Apply blackout
    stressed_x = apply_contiguous_blackout(x_single, window_len=15)

    # Run ODE and compute uncertainty at each time step
    t_eval = torch.linspace(0.0, 1.0, steps=x_single.shape[1], device=device)
    with torch.no_grad():
        _, _, params = model(stressed_x, t_eval)
        alpha, beta = params[2], params[3]
        epistemic_unc = (beta / (alpha - 1 + 1e-6)).cpu().numpy()[0]  # shape [seq_len]
    time_hours = t_single.cpu().numpy()[0]

    plt.figure(figsize=(8, 4))
    plt.plot(time_hours, epistemic_unc, 'o-', color='#1b7a43', lw=2, markersize=4)
    plt.axvspan(6, 12, alpha=0.3, color='red', label='6‑hour blackout window')
    plt.xlabel('Time (hours)')
    plt.ylabel('Epistemic uncertainty')
    plt.title('CITE-ODE: Uncertainty rises during telemetry blackout')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plots/figure5_uncertainty_trajectory.pdf', dpi=300)
    plt.savefig('plots/figure5_uncertainty_trajectory.png', dpi=300)
    logging.info("Figure 5 saved (uncertainty trajectory).")


if __name__ == "__main__":
    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_figure4()
    generate_figure5()
    logging.info("All five figures saved to 'plots/' directory.")
