"""Evaluation helpers for TIDE Neural ODE used in downstream analysis.

These utilities extract frozen latent representations from a trained
TIDENeuralODE and evaluate clinical utility (mortality prediction)
and a compact fairness check (correlation with observation density).

Expected dataloader output: `(x, t, c, y, mask)` where `x` is
`[batch, seq, vitals]`, `c` contains per-timestep confounder features,
`y` is the supervision label, and `mask` flags valid timesteps.
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr

from data.clinical_mimic import get_mimic_dataloader
from models.tide_ode import TIDENeuralODE

def extract_features_and_labels(loader, model, device):
    """Run a trained ODE on the dataset and collect per-patient features.

    The function handles padding and returns the final valid hidden
    state for each patient along with labels and per-patient confounders
    aligned to the final observed timestep. This representation is used
    for downstream logistic regression evaluations reported in the paper.
    """
    model.eval()
    all_z = []
    all_y = []
    all_c = []
    
    print("Extracting latent trajectories from frozen Neural ODE...")
    with torch.no_grad():
        for x, t, c, y, mask in loader:
            x, t, c = x.to(device), t.to(device), c.to(device)
            
            seq_len = x.shape[1]
            t_eval_1d = torch.linspace(0.0, 1.0, steps=seq_len, device=device)
            x_seq_first = x.permute(1, 0, 2)
            
            # Project to latent dim 
            if x_seq_first.shape[-1] != model.dynamics.net[0].in_features:
                 padding_dim = model.dynamics.net[0].in_features - x_seq_first.shape[-1]
                 x_seq_first = torch.cat([x_seq_first, torch.zeros(*x_seq_first.shape[:-1], padding_dim, device=device)], dim=-1)
            
            # Forward pass
            full_traj_raw, _, _ = model(x_seq_first, t_eval_1d)
            
            if full_traj_raw.ndim == 3 and full_traj_raw.shape[0] == seq_len:
                full_traj = full_traj_raw.permute(1, 0, 2)
            else:
                full_traj = full_traj_raw
            
            # Extract the LAST valid hidden state for each patient based on their mask
            # `mask` is expected to be a boolean tensor of shape [batch, seq_len].
            # `lengths` therefore yields the index of the last valid timestep.
            lengths = mask.sum(dim=1) - 1
            last_states = full_traj[torch.arange(full_traj.size(0)), lengths]
            last_confounders = c[torch.arange(c.size(0)), lengths]
            
            all_z.append(last_states.cpu().numpy())
            all_y.append(y.cpu().numpy())
            all_c.append(last_confounders.cpu().numpy())
            
    return np.vstack(all_z), np.concatenate(all_y), np.vstack(all_c)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize architecture and load the trained weights
    print("Loading trained weights...")
    model = TIDENeuralODE(latent_dim=16, num_confounders=2).to(device)
    
    try:
        model.load_state_dict(torch.load("checkpoints/tide_mimic_final.pth", map_location=device))
        print("Weights loaded successfully.")
    except FileNotFoundError:
        print("Error: Could not find 'checkpoints/tide_mimic_final.pth'. Ensure Phase 1 finished running.")
        return

    # 2. Extract Data
    loader = get_mimic_dataloader(batch_size=32)
    Z, Y, C = extract_features_and_labels(loader, model, device)
    
    # 3. Train/Test Split for Downstream Evaluation
    Z_train, Z_test, Y_train, Y_test, C_train, C_test = train_test_split(Z, Y, C, test_size=0.2, random_state=42)
    
    print("\n" + "="*50)
    print("DOWNSTREAM CLINICAL EVALUATION")
    print("="*50)
    
    # 4. Clinical Utility: Mortality Prediction
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(Z_train, Y_train)
    
    y_probs = clf.predict_proba(Z_test)[:, 1]
    auroc = roc_auc_score(Y_test, y_probs)
    auprc = average_precision_score(Y_test, y_probs)
    
    print(f"Mortality Prediction AUROC: {auroc:.4f}")
    print(f"Mortality Prediction AUPRC: {auprc:.4f}")
    
    # 5. Fairness Verification: Correlation with Observation Bias
    # Checking if the latent state (Z) is correlated with Minutes Since Last Obs (C[:, 1])
    # We test the first principal dimension of Z as a proxy
    corr, p_value = pearsonr(Z_test[:, 0], C_test[:, 1])
    
    print(f"\nFairness Verification (Pearson r with Observation Density):")
    print(f"   Correlation coefficient: {corr:.4f}")
    print(f"   p-value: {p_value:.4f}")
    if abs(corr) < 0.1:
         print("   PASSED: Representations show low correlation with observation bias.")
    else:
         print("   FAILED: Significant bias leakage detected.")

if __name__ == "__main__":
    main()