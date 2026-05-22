import os, sys, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clinical_mimic import get_mimic_dataloader
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class GRUMCDropoutNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, mask):
        out, _ = self.gru(x)
        idx_last = mask.sum(dim=1).long() - 1
        final = out[torch.arange(out.size(0)), idx_last]
        final = self.dropout(final)
        logits = self.classifier(final).squeeze(-1)
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

    model = GRUMCDropoutNet(input_dim=4, hidden_dim=16, dropout=0.3).to(device)
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
        logging.info("Seed %d | Epoch %d/15 | Loss: %.4f", args.seed, epoch+1, total_loss/len(loader))

    torch.save(model.state_dict(), args.output)
    logging.info("Saved %s", args.output)

if __name__ == "__main__":
    main()
