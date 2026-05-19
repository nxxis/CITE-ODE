import numpy as np
import matplotlib.pyplot as plt

coverage = np.array([100, 90, 80, 70])

# CITE-ODE means and stds (from your multi‑seed results)
cite_mean = np.array([0.0177, 0.0087, 0.0085, 0.0081])
cite_std  = np.array([0.0074, 0.0064, 0.0074, 0.0054])

# Stratified control means and stds
ctrl_mean = np.array([0.0177, 0.0194, 0.0195, 0.0174])
ctrl_std  = np.array([0.0074, 0.0104, 0.0100, 0.0044])

# Observed mortality prevalence (from earlier runs)
prevalence = np.array([12.07, 6.84, 5.39, 4.72])

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300, 'font.family': 'serif'})

fig, ax1 = plt.subplots(figsize=(8, 5.5))

# CITE-ODE
ax1.plot(coverage, cite_mean, 'o-', color='#2563eb', lw=2.5, label='CITE-ODE (uncertainty filter)')
ax1.fill_between(coverage, cite_mean - cite_std, cite_mean + cite_std,
                 color='#2563eb', alpha=0.2)

# Stratified control
ax1.plot(coverage, ctrl_mean, 's--', color='#64748b', lw=2, label='Stratified random control')
ax1.fill_between(coverage, ctrl_mean - ctrl_std, ctrl_mean + ctrl_std,
                 color='#64748b', alpha=0.2)

ax1.invert_xaxis()
ax1.set_xlabel('Decision coverage (%)')
ax1.set_ylabel('Conditional Expected Calibration Error (ECE)')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle=':', alpha=0.6)

# Secondary axis for prevalence
ax2 = ax1.twinx()
ax2.plot(coverage, prevalence, '^-', color='#dc2626', alpha=0.7, label='Observed mortality (%)')
ax2.set_ylabel('Mortality prevalence (%)', color='#dc2626')
ax2.tick_params(axis='y', labelcolor='#dc2626')
ax2.grid(False)

plt.title('Selective prediction: CITE-ODE vs stratified random control (5‑seed mean ± 1σ)')
plt.tight_layout()
plt.savefig('plots/figure2_selective_variance.pdf', bbox_inches='tight')
plt.savefig('plots/figure2_selective_variance.png', dpi=300)
print("✅ Final selective prediction figure saved.")
