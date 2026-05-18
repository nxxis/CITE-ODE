"""Evaluation utilities for the CEMR-Evidential ODE.

This module runs a frozen CEMR model to extract final latent states
and a per-patient epistemic uncertainty score. It also performs
downstream mortality prediction and a demographic parity audit.

The dataloader is expected to yield (x, t, c, y, d, mask) where `d`
contains static demographic attributes (e.g., age, gender).
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import CEMREvidentialODE

import os
import random
import numpy as np
import torch

def seed_everything(seed=42):
    """Forces strict determinism for perfect reproducibility."""
    print(f"Locking random seeds to {seed} for reproducibility...")
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU
    
    # Force cuDNN to behave deterministically
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_cemr_features(loader, model, device):
    """Run model inference and return final state, labels, demographics, and uncertainty.

    Returns
    -------
    Z : ndarray, shape (N, latent_dim)
        Final latent representation per patient.
    Y : ndarray, shape (N,)
        Binary labels.
    D : ndarray, shape (N, num_demographics)
        Demographic attributes (e.g., age, gender).
    Unc : ndarray, shape (N,)
        Patient-level epistemic uncertainty computed as
        mean(beta / (alpha - 1)) across vitals with a small epsilon.
    """
    model.eval()
    all_z, all_y, all_d, all_uncertainty = [], [], [], []
    
    print("Extracting evidential trajectories and epistemic uncertainty...")
    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, t, d = x.to(device), t.to(device), d.to(device)
            
            seq_len = x.shape[1]
            t_eval_1d = torch.linspace(0.0, 1.0, steps=seq_len, device=device)
            x_seq_first = x.permute(1, 0, 2)
            
            if x_seq_first.shape[-1] != model.dynamics.net[0].in_features:
                 padding_dim = model.dynamics.net[0].in_features - x_seq_first.shape[-1]
                 x_seq_first = torch.cat([x_seq_first, torch.zeros(*x_seq_first.shape[:-1], padding_dim, device=device)], dim=-1)
            
            full_traj_raw, _, _, evidential_params = model(x_seq_first, t_eval_1d)
            
            # Align trajectory shapes
            if full_traj_raw.ndim == 3 and full_traj_raw.shape[0] == seq_len:
                full_traj = full_traj_raw.permute(1, 0, 2)
            else:
                full_traj = full_traj_raw
                
            # Align evidential parameters
            _, _, alpha_raw, beta_raw = evidential_params
            alpha = alpha_raw.permute(1, 0, 2)
            beta = beta_raw.permute(1, 0, 2)
            
            # Calculate Epistemic Uncertainty per timestep and vital:
            #   epistemic = beta / (alpha - 1)
            # We average across the `num_vitals` dimension to obtain
            # a scalar epistemic score per timestep, and use an epsilon
            # for numerical stability when alpha is close to 1.
            epistemic_unc = torch.mean(beta / (alpha - 1 + 1e-6), dim=-1)
            
            # Extract the LAST valid hidden state and uncertainty for each patient
            lengths = mask.sum(dim=1) - 1
            
            last_states = full_traj[torch.arange(full_traj.size(0)), lengths]
            last_unc = epistemic_unc[torch.arange(epistemic_unc.size(0)), lengths]
            
            all_z.append(last_states.cpu().numpy())
            all_y.append(y.cpu().numpy())
            all_d.append(d.cpu().numpy()) # Demographics (Age, Gender)
            all_uncertainty.append(last_unc.cpu().numpy())
            
    return np.vstack(all_z), np.concatenate(all_y), np.vstack(all_d), np.concatenate(all_uncertainty)

def plot_latent_tsne(Z, D):
    """Generates an A* quality t-SNE scatter plot of the latent space."""
    print("Generating latent space t-SNE plot...")
    
    # Palette (Navy, Gold, Light Gray)
    NAVY = '#0B2046'
    GOLD = '#D4AF37'
    LIGHT_GRAY = '#F5F5F7'

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.facecolor': LIGHT_GRAY,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.format': 'pdf'
    })
    
    # Run t-SNE dimensionality reduction (from 16D down to 2D)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    z_2d = tsne.fit_transform(Z)
    
    # Split by gender threshold
    gender_0_mask = D[:, 1] < 0
    gender_1_mask = D[:, 1] >= 0
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot Group A (Navy)
    ax.scatter(z_2d[gender_0_mask, 0], z_2d[gender_0_mask, 1], 
               color=NAVY, alpha=0.7, s=40, edgecolor='white', lw=0.5, label='Group A')
    
    # Plot Group B (Gold)
    ax.scatter(z_2d[gender_1_mask, 0], z_2d[gender_1_mask, 1], 
               color=GOLD, alpha=0.7, s=40, edgecolor='white', lw=0.5, label='Group B')
    
    ax.set_xlabel('t-SNE Dimension 1', fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontweight='bold')
    ax.set_title('Latent Space Representation (Z)', fontweight='bold', pad=15)
    
    ax.grid(True, linestyle='--', alpha=0.6, color='white')
    ax.legend(frameon=True, facecolor='white', edgecolor='black')
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/latent_tsne.pdf')
    print("Saved latent_tsne.pdf to results/ directory.")
    
    plt.close()

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading trained CEMR-Fair weights...")
    model = CEMREvidentialODE(latent_dim=16, num_vitals=4, num_confounders=2).to(device)
    
    try:
        model.load_state_dict(torch.load("checkpoints/cemr_fair_final.pth", map_location=device))
        print("Weights loaded successfully.")
    except FileNotFoundError:
        print("Error: 'checkpoints/cemr_fair_final.pth' not found.")
        return

    loader = get_mimic_dataloader(batch_size=32)
    Z, Y, D, Unc = extract_cemr_features(loader, model, device)
    
    # Use stratified split to ensure the minority mortality class appears
    # in both train and test folds for reliable AUROC/AUPRC estimates.
    Z_train, Z_test, Y_train, Y_test, D_train, D_test, Unc_train, Unc_test = train_test_split(
        Z, Y, D, Unc, test_size=0.2, random_state=42, stratify=Y
    )
    
    print("\n" + "="*50)
    print("CEMR-FAIR DOWNSTREAM EVALUATION")
    print("="*50)
    
    # 1. Overall Clinical Utility
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(Z_train, Y_train)
    y_probs = clf.predict_proba(Z_test)[:, 1]
    
    print(f"Overall Mortality AUROC:  {roc_auc_score(Y_test, y_probs):.4f}")
    print(f"Overall Mortality AUPRC:  {average_precision_score(Y_test, y_probs):.4f}")
    print(f"Mean Epistemic Uncertainty: {np.mean(Unc_test):.4f}")
    
    # 2. Demographic Fairness Audit (Gender Parity)
    # D_test[:, 1] is Gender. In our normalization, values > 0 map to one gender, < 0 to the other.
    gender_0_mask = D_test[:, 1] < 0
    gender_1_mask = D_test[:, 1] >= 0
    
    print("\nDEMOGRAPHIC PARITY AUDIT (Gender)")
    print("-" * 50)
    
    if sum(gender_0_mask) > 0 and sum(gender_1_mask) > 0:
        auroc_0 = roc_auc_score(Y_test[gender_0_mask], y_probs[gender_0_mask])
        auroc_1 = roc_auc_score(Y_test[gender_1_mask], y_probs[gender_1_mask])
        
        unc_0 = np.mean(Unc_test[gender_0_mask])
        unc_1 = np.mean(Unc_test[gender_1_mask])
        
        print(f"Group A - AUROC: {auroc_0:.4f} | Mean Uncertainty: {unc_0:.4f} | Size: {sum(gender_0_mask)}")
        print(f"Group B - AUROC: {auroc_1:.4f} | Mean Uncertainty: {unc_1:.4f} | Size: {sum(gender_1_mask)}")
        # Report utility and trust gaps between demographic groups
        print(f"Δ AUROC (Utility Gap):     {abs(auroc_0 - auroc_1):.4f}")
        print(f"Δ Uncertainty (Trust Gap): {abs(unc_0 - unc_1):.4f}")
        # Pass the Test Set representations to the plotter
        plot_latent_tsne(Z_test, D_test)
    else:
        print("Not enough samples in one demographic group to calculate parity.")

if __name__ == "__main__":
    main()