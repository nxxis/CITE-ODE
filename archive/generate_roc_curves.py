import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import CEMREvidentialODE
from run_baseline_gru import GRUBaselineNet
from evaluate_blackout_stress import apply_contiguous_blackout

def get_probs(model, loader, device, is_ode, blackout=False):
    all_probs, all_y = [], []
    model.eval()
    with torch.no_grad():
        for x, t, c, y, d, mask in loader:
            x, t, y, mask = x.to(device), t.to(device), y.to(device), mask.to(device)
            if blackout:
                x = apply_contiguous_blackout(x, window_len=15)
            if is_ode:
                _, logits, _ = model(x, torch.linspace(0.0, 1.0, steps=x.shape[1], device=device))
                idx_last = mask.sum(dim=1) - 1
                logits = logits[torch.arange(logits.size(0)), idx_last].squeeze(-1)
            else:
                _, logits = model(x, mask)
                logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_y.extend(y.cpu().numpy())
    return np.array(all_probs), np.array(all_y)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_mimic_dataloader()

    cite_model = CEMREvidentialODE(latent_dim=16).to(device)
    cite_model.load_state_dict(torch.load("checkpoints/cemr_fair_seed42.pth", map_location=device))
    gru_model = GRUBaselineNet(input_dim=4, hidden_dim=16).to(device)
    gru_model.load_state_dict(torch.load("checkpoints/baseline_gru_seed42.pth", map_location=device))

    # Clean
    cite_probs, cite_y = get_probs(cite_model, loader, device, is_ode=True, blackout=False)
    gru_probs, gru_y = get_probs(gru_model, loader, device, is_ode=False, blackout=False)

    cite_fpr, cite_tpr, _ = roc_curve(cite_y, cite_probs)
    gru_fpr, gru_tpr, _ = roc_curve(gru_y, gru_probs)
    cite_auc = auc(cite_fpr, cite_tpr)
    gru_auc = auc(gru_fpr, gru_tpr)

    # Blackout
    cite_probs_b, cite_y_b = get_probs(cite_model, loader, device, is_ode=True, blackout=True)
    gru_probs_b, gru_y_b = get_probs(gru_model, loader, device, is_ode=False, blackout=True)
    cite_fpr_b, cite_tpr_b, _ = roc_curve(cite_y_b, cite_probs_b)
    gru_fpr_b, gru_tpr_b, _ = roc_curve(gru_y_b, gru_probs_b)
    cite_auc_b = auc(cite_fpr_b, cite_tpr_b)
    gru_auc_b = auc(gru_fpr_b, gru_tpr_b)

    plt.figure(figsize=(5,5))
    plt.plot(cite_fpr, cite_tpr, lw=2, label=f'CITE-ODE clean (AUC={cite_auc:.3f})')
    plt.plot(gru_fpr, gru_tpr, lw=2, label=f'GRU clean (AUC={gru_auc:.3f})')
    plt.plot(cite_fpr_b, cite_tpr_b, '--', lw=2, label=f'CITE-ODE blackout (AUC={cite_auc_b:.3f})')
    plt.plot(gru_fpr_b, gru_tpr_b, '--', lw=2, label=f'GRU blackout (AUC={gru_auc_b:.3f})')
    plt.plot([0,1],[0,1], 'k--', alpha=0.5)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('plots/figure2_roc.pdf', dpi=300)
    print("✅ Figure 2 saved: plots/figure2_roc.pdf")

if __name__ == "__main__":
    main()
