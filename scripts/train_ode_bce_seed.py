import os, sys, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import ODEFunc

class ODEBCEModel(nn.Module):
    def __init__(self, latent_dim=16, num_vitals=4):
        super().__init__()
        self.encoder = nn.GRU(num_vitals, latent_dim, batch_first=True)
        self.dynamics = ODEFunc(latent_dim)
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, x, t_eval):
        _, h_n = self.encoder(x)
        z_0 = h_n.squeeze(0)
        full_trajectory = odeint(self.dynamics, z_0, t_eval, method="rk4")
        full_trajectory = full_trajectory.permute(1, 0, 2)
        logits = self.classifier(full_trajectory)
        return logits

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()

    model = ODEBCEModel(latent_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(15):
        model.train()
        total_loss = 0.0
        for x, t, c, y, d, mask in loader:
            x, t, y, mask = x.to(device), t.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            t_eval = torch.linspace(0.0, 1.0, steps=x.shape[1], device=device)
            logits = model(x, t_eval)
            idx_last = mask.sum(dim=1) - 1
            final_logits = logits[torch.arange(logits.size(0)), idx_last].squeeze(-1)
            loss = criterion(final_logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Seed {args.seed} | Epoch {epoch+1}/15 | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
