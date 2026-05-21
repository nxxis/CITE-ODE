"""Training entrypoint for a single-seed CEMR run.

This script encapsulates the minimal training loop used to produce the
checkpoints reported in the paper. It is intentionally compact: the goal is
readability and reproducibility rather than training speed or advanced
instrumentation.
"""

import os
import random
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import CEMREvidentialODE


def seed_everything(seed):
    """Set deterministic seeds for reproducible experiments.

    Note: full determinism also depends on CUDA/cuDNN settings and PyTorch
    configuration. This function covers the common RNG sources used in code.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def evidential_regression_loss(gamma, v, alpha, beta, targets, mask, lambda_reg=0.01):
    """Negative log-likelihood for the Normal-Inverse-Gamma evidential head.

    The implementation follows the parameterization used in the paper: the
    head predicts (gamma, v, alpha, beta) per prediction and the NLL is
    computed only over valid (unmasked) timesteps. A small regularizer that
    encourages agreement between mean prediction and target is included.
    """

    gamma, v, alpha, beta = gamma[mask], v[mask], alpha[mask], beta[mask]
    y = targets[mask]
    v, beta = torch.clamp(v, min=1e-6), torch.clamp(beta, min=1e-6)
    alpha = torch.clamp(alpha, min=1.0 + 1e-6)
    omg = 2 * beta * (1 + v)
    nll = (
        0.5 * torch.log(torch.pi / v)
        - alpha * torch.log(omg)
        + (2 * alpha + 1) * 0.5 * torch.log(v * (y - gamma) ** 2 + omg)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return torch.mean(nll) + lambda_reg * torch.mean(torch.abs(y - gamma) * (2 * v + alpha))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--output", type=str, required=True, help="Output checkpoint path")
    args = parser.parse_args()

    # Training loop: alternating discriminator and model updates per batch.
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training with seed {args.seed} on {device}")

    loader = get_mimic_dataloader()
    model = CEMREvidentialODE(latent_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    opt_d = optim.Adam(model.discriminator.parameters(), lr=1e-3)
    criterion_task = nn.BCEWithLogitsLoss()

    # Small epoch count by default (keeps CI/debug runs fast); experiments use
    # longer schedules as described in the README/colab notes.
    for epoch in range(15):
        model.train()
        for x, t, c, y, d, mask in loader:
            # Move tensors to device
            x, t, y, d, mask = x.to(device), t.to(device), y.to(device), d.to(device), mask.to(device)

            # Evaluate model on a normalized time grid matching the padded length
            full_traj, logits_y, params = model(x, torch.linspace(0.0, 1.0, steps=x.shape[1], device=device))
            gamma, v, alpha, beta = params

            # Extract the final valid timestep logits per example
            idx_last = mask.sum(dim=1) - 1
            final_logits = logits_y[torch.arange(logits_y.size(0)), idx_last].squeeze(-1)

            # ----------------- Discriminator update -----------------
            opt_d.zero_grad()
            # We pass only the valid latent states into the adversary
            p_gender = model.discriminator(full_traj[mask].detach())
            # Expand the per-stay gender label to per-timestep and mask
            gender_expanded = d[:, 1].unsqueeze(1).expand(-1, x.shape[1])[mask]
            loss_d = nn.BCEWithLogitsLoss()(p_gender.squeeze(-1), gender_expanded)
            loss_d.backward()
            opt_d.step()

            # ----------------- Model update -----------------
            optimizer.zero_grad()
            loss_nll = evidential_regression_loss(gamma, v, alpha, beta, x, mask)
            loss_task = criterion_task(final_logits, y)
            p_gender_g = model.discriminator(full_traj[mask])
            loss_adv = nn.BCEWithLogitsLoss()(p_gender_g.squeeze(-1), gender_expanded)
            # Composite objective: evidential NLL + task loss - adversary
            (loss_nll + 5.0 * loss_task - 5.0 * loss_adv).backward()
            optimizer.step()

        print(
            f"Seed {args.seed} | Epoch {epoch+1:02d}/15 | NLL: {loss_nll.item():.4f} | Task: {loss_task.item():.4f} | Adv: {loss_d.item():.4f}"
        )

    torch.save(model.state_dict(), args.output)
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
