import os, random, numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data.clinical_mimic import get_mimic_dataloader

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class GRUBaselineNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16):
        super(GRUBaselineNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        out, _ = self.gru(x)
        idx_last = mask.sum(dim=1) - 1
        final_hidden = out[torch.arange(out.size(0)), idx_last]
        logits = self.classifier(final_hidden)
        return final_hidden, logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training GRU with seed {args.seed} on {device}")

    loader = get_mimic_dataloader()
    model = GRUBaselineNet(input_dim=4, hidden_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(15):
        model.train()
        total_loss = 0.0
        for x, t, c, y, d, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            _, logits = model(x, mask)
            loss = criterion(logits.squeeze(-1), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Seed {args.seed} | Epoch {epoch+1:02d}/15 | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"Saved GRU checkpoint to {args.output}")

if __name__ == "__main__":
    main()
