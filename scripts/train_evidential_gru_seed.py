import os, sys, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clinical_mimic import get_mimic_dataloader
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class EvidentialGRU(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.evidential_head = nn.Linear(hidden_dim, 4)  # outputs (gamma, v, alpha, beta)

    def forward(self, x, mask):
        out, _ = self.gru(x)
        idx_last = mask.sum(dim=1).long() - 1
        final = out[torch.arange(out.size(0)), idx_last]
        ev_params = self.evidential_head(final)  # [B, 4]
        gamma = ev_params[:, 0]
        v = torch.exp(ev_params[:, 1]) + 1e-6
        alpha = torch.exp(ev_params[:, 2]) + 1.0 + 1e-6
        beta = torch.exp(ev_params[:, 3]) + 1e-6
        return gamma, v, alpha, beta

def evidential_regression_loss(gamma, v, alpha, beta, targets, lambda_reg=0.01):
    y = targets
    v = torch.clamp(v, min=1e-6)
    beta = torch.clamp(beta, min=1e-6)
    alpha = torch.clamp(alpha, min=1.0 + 1e-6)
    omg = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(torch.pi / v) - alpha * torch.log(omg) + (2 * alpha + 1) * 0.5 * torch.log(v * (y - gamma)**2 + omg) + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    nll = torch.mean(nll)
    reg = torch.mean(torch.abs(y - gamma) * (2 * v + alpha))
    return nll + lambda_reg * reg

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

    model = EvidentialGRU(input_dim=4, hidden_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(15):
        model.train()
        total_loss = 0.0
        for x, t, c, y, d, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            gamma, v, alpha, beta = model(x, mask)
            loss = evidential_regression_loss(gamma, v, alpha, beta, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info("Seed %d | Epoch %d/15 | Loss: %.4f", args.seed, epoch+1, total_loss/len(loader))

    torch.save(model.state_dict(), args.output)
    logging.info("Saved %s", args.output)

if __name__ == "__main__":
    main()
