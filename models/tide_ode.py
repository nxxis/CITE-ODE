"""CEMR model components: latent ODE dynamics, evidential head, and adversary.

This module implements a compact Continuous-time Evidential framework used in the
paper. The architecture follows three responsibilities:

- Encode an observed irregularly-sampled trajectory into a latent initial state
- Integrate a latent ODE to produce a continuous trajectory
- Decode evidential parameters (NIG-style) and a task logit at each timepoint

The implementation aims to be clear and minimal while matching the training
scripts in `scripts/`.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    """Right-hand side f(t, y) for the latent ODE.

    A small MLP maps the current latent state to its time-derivative. Kept simple
    (one hidden layer, Tanh activation) to match the experiments in the paper.
    """

    def __init__(self, latent_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, t, y):
        # odeint expects signature (t, y)
        return self.net(y)


class DemographicAdversary(nn.Module):
    """Lightweight adversary to predict sensitive attributes from latents.

    Used during adversarial training to encourage demographic invariance in the
    learned latent representations. Returns a single logit per example.
    """

    def __init__(self, latent_dim=16):
        super(DemographicAdversary, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, z):
        # z: [N, latent_dim] -> returns [N, 1] logits
        return self.net(z)


class CEMREvidentialODE(nn.Module):
    """End-to-end CITE-ODE model combining encoder, ODE and evidential head.

    Args:
        latent_dim (int): dimensionality of the latent ODE state.
        num_vitals (int): number of input vital-sign channels.

    Forward inputs:
        x: tensor [B, T, num_vitals] observed trajectories (padded)
        t_eval: 1D tensor of timepoints at which to evaluate the ODE

    Returns:
        full_trajectory: [B, T, latent_dim] continuous latent states
        logits_y: [B, T, 1] task logits at each timepoint
        evidential_params: tuple of four tensors corresponding to the NIG-style
            parameters (gamma, v, alpha, beta) suitable for loss computation.
    """

    def __init__(self, latent_dim=16, num_vitals=4):
        super(CEMREvidentialODE, self).__init__()
        self.latent_dim = latent_dim
        self.num_vitals = num_vitals

        # Recognition network: map observed multivariate sequence -> initial latent z0
        self.encoder = nn.GRU(num_vitals, latent_dim, batch_first=True)

        # Latent dynamics (ODE RHS)
        self.dynamics = ODEFunc(latent_dim)

        # Evidential head: produces 4 parameters per vital channel per timepoint
        # Output shape before reshaping: [B, T, num_vitals * 4]
        self.evidential_head = nn.Linear(latent_dim, num_vitals * 4)

        # Adversary and downstream classifier
        self.discriminator = DemographicAdversary(latent_dim)
        self.task_classifier = nn.Linear(latent_dim, 1)

    def forward(self, x, t_eval):
        # Encode full observed sequence into a single initial latent state z_0
        _, h_n = self.encoder(x)
        z_0 = h_n.squeeze(0)

        # Integrate the latent ODE to obtain states at t_eval.
        # odeint returns [len(t_eval), B, latent_dim] -> permute to [B, T, H]
        full_trajectory = odeint(self.dynamics, z_0, t_eval, method="rk4")
        full_trajectory = full_trajectory.permute(1, 0, 2)  # [B, T, H]

        # Evidential parameters for each vital-sign: reshape to [B, T, V, 4]
        raw_params = self.evidential_head(full_trajectory).view(
            full_trajectory.shape[0], full_trajectory.shape[1], self.num_vitals, 4
        )

        # Task classifier produces a single logit per timepoint
        logits_y = self.task_classifier(full_trajectory)

        # Apply safe transforms to produce strictly positive scales where needed.
        gamma = raw_params[..., 0]
        v = torch.exp(raw_params[..., 1]) + 1e-6
        alpha = torch.exp(raw_params[..., 2]) + 1.0 + 1e-6
        beta = torch.exp(raw_params[..., 3]) + 1e-6

        return full_trajectory, logits_y, (gamma, v, alpha, beta)
