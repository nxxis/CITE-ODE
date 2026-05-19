import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Multi‑seed selective results from your runs (80% coverage)
    # For full curves, we use the single‑seed sweep (seed=42) and add error bars at 80% from multi‑seed.
    # Values from evaluate_selective_risk.py output:
    # coverage [100,90,80,70]; cite_ece [0.0183,0.0084,0.0034,0.0056]; brier [0.072,0.062,0.050,0.045]; prev [12.07,6.84,5.39,4.72].
    # Control ECE (single‑seed) [0.0183,0.0172,0.0148,0.0169].

    coverage = np.array([100, 90, 80, 70])[::-1]  # descending
    cite_ece = np.array([0.0183, 0.0084, 0.0034, 0.0056])[::-1]
    control_ece = np.array([0.0183, 0.0172, 0.0148, 0.0169])[::-1]
    brier = np.array([0.072, 0.062, 0.050, 0.045])[::-1]
    prev = np.array([12.07, 6.84, 5.39, 4.72])[::-1]

    # Multi‑seed std at 80% coverage (from your multi‑seed selective evaluation)
    cite_80_mean, cite_80_std = 0.0080, 0.0055
    control_80_mean, control_80_std = 0.0222, 0.0106

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Conditional ECE
    ax1.plot(coverage, cite_ece, 'o-', color='#1b7a43', lw=2, label='CITE-ODE (uncertainty-guided)')
    ax1.plot(coverage, control_ece, 's--', color='#7570b3', lw=2, label='Stratified random control')
    # Add error bar at 80% coverage
    ax1.errorbar(80, cite_80_mean, yerr=cite_80_std, fmt='o', color='#1b7a43', capsize=5, elinewidth=2, markeredgewidth=2)
    ax1.errorbar(80, control_80_mean, yerr=control_80_std, fmt='s', color='#7570b3', capsize=5, elinewidth=2)
    ax1.set_xlabel('Decision coverage (%)')
    ax1.set_ylabel('Conditional ECE')
    ax1.set_xlim(105, 65)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='upper right')

    # Right: Brier score and prevalence (dual axis)
    ax2.plot(coverage, brier, 'o-', color='#e7298a', lw=2, label='Retained Brier score')
    ax2.set_xlabel('Decision coverage (%)')
    ax2.set_ylabel('Brier score', color='#e7298a')
    ax2.tick_params(axis='y', labelcolor='#e7298a')
    ax2.set_xlim(105, 65)
    ax2.grid(True, linestyle=':', alpha=0.5)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(coverage, prev, 'd-', color='#d95f02', lw=2, label='Observed mortality prevalence')
    ax2_twin.set_ylabel('Mortality prevalence (%)', color='#d95f02')
    ax2_twin.tick_params(axis='y', labelcolor='#d95f02')

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2, loc='upper right')

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig('plots/figure3_selective.pdf', dpi=300)
    plt.savefig('plots/figure3_selective.png', dpi=300)
    print("✅ Figure 3 saved: plots/figure3_selective.pdf")

if __name__ == "__main__":
    main()
