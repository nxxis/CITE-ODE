import torch
import torch.nn as nn

def tide_drift_loss(actual, predicted):
  """Mean-squared drift loss that stabilizes the learned ODE dynamics.

  A simple MSE between actual and predicted subsequences encourages
  the continuous ODE solution to follow the discrete observed
  subsequences, improving numerical stability and representation
  fidelity.
  """
    return nn.MSELoss()(actual, predicted)


def evidential_regression_loss(gamma, v, alpha, beta, targets, mask, lambda_reg=0.01):
    """Negative log-likelihood for Normal-Inverse-Gamma + epistemic regularizer.

    The loss combines a probabilistically-principled NLL under a
    Normal-Inverse-Gamma predictive family with a lightweight
    regularizer that penalizes large errors occurring with low
    epistemic evidence. The `mask` argument removes padded timesteps
    from the computation so training is stable on variable-length
    sequences.

    Arguments are expected to be aligned such that masking produces
    flattened [TotalValidSteps, num_vitals] tensors for all terms.
    """
    # Filter down to unpadded timesteps
    gamma = gamma[mask]
    v = v[mask]
    alpha = alpha[mask]
    beta = beta[mask]
    y = targets[mask]

    # 1. Negative Log-Likelihood of Normal-Inverse-Gamma
    # Remap parameters to guarantee valid statistical boundaries
    v = torch.clamp(v, min=1e-6)
    alpha = torch.clamp(alpha, min=1.0 + 1e-6)
    beta = torch.clamp(beta, min=1e-6)

    omg = 2 * beta * (1 + v)
    
    # Standard continuous EDL log-likelihood formulation
    nll = 0.5 * torch.log(torch.pi / v) \
          - alpha * torch.log(omg) \
          + (2 * alpha + 1) * 0.5 * torch.log(v * (y - gamma)**2 + omg) \
          + torch.lgamma(alpha) \
          - torch.lgamma(alpha + 0.5)

    # 2. Epistemic Uncertainty Regularizer (penalizes high error with low uncertainty)
    error = torch.abs(y - gamma)
    evidence = 2 * v + alpha
    reg = error * evidence

    return torch.mean(nll) + lambda_reg * torch.mean(reg)