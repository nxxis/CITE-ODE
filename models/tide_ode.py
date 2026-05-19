import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, latent_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 32), nn.Tanh(), nn.Linear(32, latent_dim))
    def forward(self, t, y): return self.net(y)

class DemographicAdversary(nn.Module):
    def __init__(self, latent_dim=16):
        super(DemographicAdversary, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, z): return self.net(z)

class CEMREvidentialODE(nn.Module):
    def __init__(self, latent_dim=16, num_vitals=4):
        super(CEMREvidentialODE, self).__init__()
        self.latent_dim = latent_dim
        self.num_vitals = num_vitals
        
        # Recognition network to encode longitudinal trajectory context into z_0
        self.encoder = nn.GRU(num_vitals, latent_dim, batch_first=True)
        self.dynamics = ODEFunc(latent_dim)
        self.evidential_head = nn.Linear(latent_dim, num_vitals * 4)
        self.discriminator = DemographicAdversary(latent_dim)
        self.task_classifier = nn.Linear(latent_dim, 1)

    def forward(self, x, t_eval):
        # Compute starting latent vector from full timeline sequence
        _, h_n = self.encoder(x)
        z_0 = h_n.squeeze(0)
        
        # Continuous differential integration
        full_trajectory = odeint(self.dynamics, z_0, t_eval, method="rk4")
        full_trajectory = full_trajectory.permute(1, 0, 2) # [Batch, Seq, Hidden]
        
        raw_params = self.evidential_head(full_trajectory).view(full_trajectory.shape[0], full_trajectory.shape[1], self.num_vitals, 4)
        logits_y = self.task_classifier(full_trajectory)
        
        return full_trajectory, logits_y, (raw_params[...,0], torch.exp(raw_params[...,1])+1e-6, torch.exp(raw_params[...,2])+1.0+1e-6, torch.exp(raw_params[...,3])+1e-6)
