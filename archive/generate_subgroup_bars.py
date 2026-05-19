import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    groups = ['Female', 'Male', 'White', 'Black', 'Hispanic', 'Asian']
    auroc = [0.843, 0.822, 0.825, 0.848, 0.814, 0.875]
    ece = [0.024, 0.019, 0.017, 0.052, 0.031, 0.053]
    unc = [0.168, 0.107, 0.140, 0.097, 0.068, 0.128]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8,5))
    bars1 = ax1.bar(x - width/2, auroc, width, label='AUROC', color='#1b7a43', alpha=0.8)
    ax1.set_ylabel('AUROC')
    ax1.set_ylim(0.7, 0.9)
    ax1.tick_params(axis='y', labelcolor='#1b7a43')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, ece, width, label='ECE', color='#d95f02', alpha=0.8)
    ax2.set_ylabel('ECE', color='#d95f02')
    ax2.tick_params(axis='y', labelcolor='#d95f02')
    ax2.set_ylim(0, 0.08)

    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, rotation=45, ha='right')
    ax1.set_title('Subgroup performance (CITE-ODE, single seed)')

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='upper left')

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig('plots/figure4_subgroup.pdf', dpi=300)
    print("Figure 4 saved: plots/figure4_subgroup.pdf")

if __name__ == "__main__":
    main()
