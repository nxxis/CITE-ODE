import os, random, torch, numpy as np
import torch.nn as nn, torch.optim as optim
from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import CEMREvidentialODE

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def evidential_regression_loss(gamma, v, alpha, beta, targets, mask, lambda_reg=0.01):
    gamma, v, alpha, beta = gamma[mask], v[mask], alpha[mask], beta[mask]
    y = targets[mask]
    v, beta = torch.clamp(v, min=1e-6), torch.clamp(beta, min=1e-6)
    alpha = torch.clamp(alpha, min=1.0 + 1e-6)
    omg = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(torch.pi / v) - alpha * torch.log(omg) + (2 * alpha + 1) * 0.5 * torch.log(v * (y - gamma)**2 + omg) + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return torch.mean(nll) + lambda_reg * torch.mean(torch.abs(y - gamma) * (2 * v + alpha))

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()
    
    model = CEMREvidentialODE(latent_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion_task = nn.BCEWithLogitsLoss()
    
    print("🧪 Optimizing Ablation Variant C (Latent ODE No Adversary) over 10k Cohort...")
    for epoch in range(15):
        model.train()
        for x, t, c, y, d, mask in loader:
            x, t, y, mask = x.to(device), t.to(device), y.to(device), mask.to(device)
            
            full_traj, logits_y, params = model(x, torch.linspace(0.0, 1.0, steps=x.shape[1], device=device))
            gamma, v, alpha, beta = params
            
            idx_last = mask.sum(dim=1) - 1
            final_logits = logits_y[torch.arange(logits_y.size(0)), idx_last].squeeze(-1)
            
            # Pure Predictive Training: Minimax Adversarial head completely disconnected
            optimizer.zero_grad()
            loss_nll = evidential_regression_loss(gamma, v, alpha, beta, x, mask)
            loss_task = criterion_task(final_logits, y)
            
            (loss_nll + 5.0 * loss_task).backward()
            optimizer.step()
            
        print(f"Ablation Epoch [{epoch+1:02d}/15] | Evidential NLL: {loss_nll.item():.4f} | Task BCE: {loss_task.item():.4f}")
        
    torch.save(model.state_dict(), "checkpoints/cemr_ablation_no_adv.pth")
    print("💾 Ablation weights successfully saved to Google Drive.")

if __name__ == "__main__": main()
