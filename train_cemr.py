"""Train the CEMR-Fair evidential continuous-time model.

This training harness implements the recipe used for our ICDM 2026
submission: a continuous-time latent ODE that outputs both trajectories
and per-vital evidential parameters. The loop follows a minimax-style
procedure where a discriminator attempts to predict demographics from
latent states and the generator minimizes task loss plus a drift
regularizer while adversarially reducing confounder leakage.

The code is intentionally explicit and readable to ease reproducibility
during peer review.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os

from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import CEMREvidentialODE
from utils.losses import tide_drift_loss, evidential_regression_loss

import os
import random
import numpy as np
import torch

def seed_everything(seed=42):
    """Fix random seeds and deterministic flags for reproducible runs.

    Reproducibility is central for reviewers: this helper pins Python,
    NumPy and PyTorch RNGs and configures cuDNN to deterministic mode.
    """
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


def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader = get_mimic_dataloader(batch_size=32)

    print("\nInitializing CEMR-Fair engine...")
    # num_confounders=2 now represents Age and Gender
    model = CEMREvidentialODE(latent_dim=16, num_vitals=4, num_confounders=2).to(device)
    
    opt_ode = optim.Adam(model.dynamics.parameters(), lr=1e-3)
    opt_head = optim.Adam(model.evidential_head.parameters(), lr=1e-3)
    opt_d = optim.Adam(model.discriminator.parameters(), lr=1e-4)

    epochs = 200
    print("\n" + "="*50)
    print("TRAINING CEMR-FAIR WITH EVIDENTIAL OBJECTIVE")
    print("="*50)

    for epoch in range(epochs):
        model.train()
        total_nll_loss, total_drift_loss, total_d_loss = 0, 0, 0
        batches = 0
        
                # The dataloader yields `d` containing static demographics
                # (e.g., age, gender). We pass these to the discriminator.
        for x, t, c, y, d, mask in loader:
            x, t, d, mask = x.to(device), t.to(device), d.to(device), mask.to(device)
            
            seq_len = x.shape[1] 
            t_eval_1d = torch.linspace(0.0, 1.0, steps=seq_len, device=device)
            x_seq_first = x.permute(1, 0, 2)
            
            if x_seq_first.shape[-1] != model.dynamics.net[0].in_features:
                 padding_dim = model.dynamics.net[0].in_features - x_seq_first.shape[-1]
                 x_seq_first = torch.cat([x_seq_first, torch.zeros(*x_seq_first.shape[:-1], padding_dim, device=device)], dim=-1)
            
            full_traj_raw, z_actual_raw, z_pred_raw, evidential_params = model(x_seq_first, t_eval_1d) 
            
            if full_traj_raw.ndim == 3 and full_traj_raw.shape[0] == seq_len:
                full_traj = full_traj_raw.permute(1, 0, 2)
                z_actual = z_actual_raw.permute(1, 0, 2)
                z_pred = z_pred_raw.permute(1, 0, 2)
            else:
                full_traj, z_actual, z_pred = full_traj_raw, z_actual_raw, z_pred_raw
                
            # The ODE outputs [Seq, Batch, Vitals]. Permute to [Batch, Seq, Vitals]
            # to match the collate `mask` which is [batch, seq_len].
            gamma_raw, v_raw, alpha_raw, beta_raw = evidential_params
            gamma = gamma_raw.permute(1, 0, 2)
            v = v_raw.permute(1, 0, 2)
            alpha = alpha_raw.permute(1, 0, 2)
            beta = beta_raw.permute(1, 0, 2)
            
            valid_traj = full_traj[mask]
            
            # --- Expand demographics 'd' to match the unmasked trajectory steps ---
            # 'd' is [Batch, num_demographics]. We expand it to
            # [Batch, Seq, num_demographics] and then mask to get the
            # flattened [Total_Valid_Steps, num_demographics] tensor
            # used by the discriminator loss.
            d_expanded = d.unsqueeze(1).expand(-1, seq_len, -1)[mask]
            
            # ---------------------------------------------------------
            # PHASE 1: Optimize Demographic Discriminator
            # ---------------------------------------------------------
            opt_d.zero_grad()
            predicted_demographics = model.discriminator(valid_traj.detach())
            loss_d = nn.MSELoss()(predicted_demographics, d_expanded)
            loss_d.backward()
            opt_d.step()
            
            # ---------------------------------------------------------
            # PHASE 2: Optimize Evidential Generator
            # ---------------------------------------------------------
            opt_ode.zero_grad()
            opt_head.zero_grad()
            
            # Evidential NLL Task Loss computed only on valid (unmasked)
            # timesteps. This uses a Normal-Inverse-Gamma negative
            # log-likelihood plus a regularizer that penalizes low
            # uncertainty for high errors.
            loss_nll = evidential_regression_loss(gamma, v, alpha, beta, x, mask)
            
            # Continuous Drift Stability
            if z_actual.ndim == 3 and z_actual.shape[1] == mask.shape[1] - 1:
                loss_drift = tide_drift_loss(z_actual[mask[:, 1:]], z_pred[mask[:, 1:]])
            else:
                loss_drift = tide_drift_loss(z_actual, z_pred)
                
            # Minimax Fairness Penalty (Hide Age and Gender)
            adv_preds = model.discriminator(full_traj[mask])
            loss_adv = nn.MSELoss()(adv_preds, d_expanded)
            
            total_loss = loss_nll + 1.0 * loss_drift - 0.5 * loss_adv
            total_loss.backward()
            
            opt_ode.step()
            opt_head.step()
            
            total_nll_loss += loss_nll.item()
            total_drift_loss += loss_drift.item()
            total_d_loss += loss_d.item()
            batches += 1
            
        print(f"Epoch {epoch:03d}/{epochs} | Evidential NLL: {total_nll_loss/batches:.4f} | Drift: {total_drift_loss/batches:.4f} | Demographic Adv MSE: {total_d_loss/batches:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/cemr_fair_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nCEMR-Fair training complete. Model saved to: {save_path}")

if __name__ == "__main__":
    main()