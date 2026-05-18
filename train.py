"""Phase 1 training: TIDE Neural ODE (minimax adversarial training).

This module implements the Phase 1 training used in our experiments.
We alternate discriminator updates (to predict confounders from latent
states) with generator updates that minimize reconstruction error,
stabilize continuous drift, and reduce adversarial leakage. The
implementation relies on mask-based indexing so losses are computed
only on valid (unpadded) timesteps.

The code is written to be clear and reproducible for reviewers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import TIDENeuralODE
from utils.losses import tide_drift_loss
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    loader = get_mimic_dataloader(batch_size=32)

    # 2. Initialize Model
    # Note: The training alternates between discriminator updates
    # and generator (ODE) updates in a classical adversarial setup.
    print("\nInitializing TIDE minimax engine...")
    model = TIDENeuralODE(latent_dim=16, num_confounders=2).to(device)
    opt_ode = optim.Adam(model.dynamics.parameters(), lr=1e-3)
    opt_d = optim.Adam(model.discriminator.parameters(), lr=1e-4)

    epochs = 200
    print("\n" + "="*50)
    print("TRAINING TIDE ON MIMIC-IV TRAJECTORIES")
    print("="*50)

    for epoch in range(epochs):
        model.train()
        total_task_loss, total_drift_loss, total_d_loss = 0, 0, 0
        batches = 0
        
        for x, t, c, y, mask in loader:
            x, t, c, y, mask = x.to(device), t.to(device), c.to(device), y.to(device), mask.to(device)
            
            seq_len = x.shape[1] 
            t_eval_1d = torch.linspace(0.0, 1.0, steps=seq_len, device=device)
            x_seq_first = x.permute(1, 0, 2)
            
            if x_seq_first.shape[-1] != model.dynamics.net[0].in_features:
                 padding_dim = model.dynamics.net[0].in_features - x_seq_first.shape[-1]
                 x_seq_first = torch.cat([x_seq_first, torch.zeros(*x_seq_first.shape[:-1], padding_dim, device=device)], dim=-1)
            
            full_traj_raw, z_actual_raw, z_pred_raw = model(x_seq_first, t_eval_1d) 
            
            if full_traj_raw.ndim == 3 and full_traj_raw.shape[0] == seq_len:
                full_traj = full_traj_raw.permute(1, 0, 2)
                z_actual = z_actual_raw.permute(1, 0, 2)
                z_pred = z_pred_raw.permute(1, 0, 2)
            else:
                full_traj, z_actual, z_pred = full_traj_raw, z_actual_raw, z_pred_raw
                
            # `mask` is boolean with shape [batch, seq_len]; indexing
            # `full_traj[mask]` returns a flat tensor of only the valid
            # timesteps across the batch which matches the confounder
            # tensor when similarly masked.
            valid_traj = full_traj[mask]
            valid_confounders = c[mask]
            
            # PHASE 1: Optimize Discriminator
            opt_d.zero_grad()
            predicted_confounders = model.discriminator(valid_traj.detach())
            loss_d = nn.MSELoss()(predicted_confounders, valid_confounders)
            loss_d.backward()
            opt_d.step()
            
            # PHASE 2: Optimize Generator
            opt_ode.zero_grad()
            loss_task = nn.MSELoss()(full_traj[mask][:, :4], x[mask])
            
            if z_actual.ndim == 3 and z_actual.shape[1] == mask.shape[1] - 1:
                loss_drift = tide_drift_loss(z_actual[mask[:, 1:]], z_pred[mask[:, 1:]])
            else:
                loss_drift = tide_drift_loss(z_actual, z_pred)
                
            adv_preds = model.discriminator(full_traj[mask])
            loss_adv = nn.MSELoss()(adv_preds, valid_confounders)
            
            # Loss composition: reconstruction + drift regularizer - adv
            # The negative weight on `loss_adv` implements the minimax
            # objective: make discriminator predictions worse.
            total_loss = loss_task + 1.0 * loss_drift - 0.5 * loss_adv
            total_loss.backward()
            opt_ode.step()
            
            total_task_loss += loss_task.item()
            total_drift_loss += loss_drift.item()
            total_d_loss += loss_d.item()
            batches += 1
            
            print(f"Epoch {epoch:03d}/{epochs} | Task MSE: {total_task_loss/batches:.4f} | Drift Loss: {total_drift_loss/batches:.4f} | Adversarial MSE: {total_d_loss/batches:.4f}")

    # 3. Pipeline Freeze: Save the Weights
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/tide_mimic_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nPhase 1 complete. Model saved to: {save_path}")

if __name__ == "__main__":
    main()