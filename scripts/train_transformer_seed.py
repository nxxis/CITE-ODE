import os, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data.clinical_mimic import get_mimic_dataloader
from models.modern_baselines import TSTransformer

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

    model = TSTransformer(input_dim=4, d_model=64, n_heads=4, num_layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(15):
        model.train()
        total_loss = 0.0
        for x, t, c, y, d, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Seed {args.seed} | Epoch {epoch+1}/15 | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
