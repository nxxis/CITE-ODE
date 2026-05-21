"""Evaluation metrics utilities used by the analysis scripts.

This module provides standard calibration and discrimination metrics used in
the manuscript: Expected Calibration Error (ECE), Adaptive Calibration Error
(ACE), Brier Skill Score (BSS), and a simple bootstrap audit utility that
returns percentile-based confidence intervals.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def calculate_ece(y_true, y_prob, n_bins=10):
    """Compute (non-adaptive) Expected Calibration Error.

    The dataset is partitioned into `n_bins` equal-width probability bins. For
    each bin we compute the difference between average predicted probability
    and empirical frequency, weighted by the bin prevalence.

    Args:
        y_true: array-like of binary labels (0/1)
        y_prob: array-like of predicted probabilities in [0, 1]
        n_bins: number of equal-width bins

    Returns:
        ece (float): expected calibration error
    """

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            ece += prop_in_bin * np.abs(np.mean(y_prob[in_bin]) - np.mean(y_true[in_bin]))
    return ece


def calculate_ace(y_true, y_prob, n_bins=15):
    """Adaptive Calibration Error (ACE) using equal-frequency binning.

    ACE partitions predictions into equal-sized bins (by count) and averages
    absolute differences between predicted probability and empirical frequency.
    """

    sorted_indices = np.argsort(y_prob)
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]

    bins_true = np.array_split(y_true_sorted, n_bins)
    bins_prob = np.array_split(y_prob_sorted, n_bins)

    ace = 0.0
    for b_true, b_prob in zip(bins_true, bins_prob):
        if len(b_true) == 0:
            continue
        ace += np.abs(np.mean(b_true) - np.mean(b_prob)) / n_bins
    return ace


def calculate_brier_skill_score(y_true, y_prob):
    """Compute the Brier Skill Score relative to a climatology baseline.

    BSS = 1 - (Brier_model / Brier_reference), where the reference forecast
    predicts the dataset-wide prevalence for every sample.
    """

    global_prevalence = np.mean(y_true)
    reference_forecast = np.full_like(y_true, global_prevalence)
    brier_ref = brier_score_loss(y_true, reference_forecast)
    brier_model = brier_score_loss(y_true, y_prob)

    if brier_ref == 0:
        return 0.0
    return 1.0 - (brier_model / brier_ref)


def run_bootstrap_audit(y_true, y_prob, n_resamples=1000, alpha=0.05):
    """Bootstrap confidence intervals for common metrics.

    Returns percentile-based confidence intervals for AUROC, AUPRC, ECE and
    Brier score. The bootstrap uses a fixed RNG seed for reproducibility.
    """

    boot_aurocs, boot_auprcs, boot_eces, boot_briers = [], [], [], []
    n_samples = len(y_true)
    rng = np.random.default_rng(seed=42)
    for _ in range(n_resamples):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(y_true[boot_idx])) > 1:
            boot_aurocs.append(roc_auc_score(y_true[boot_idx], y_prob[boot_idx]))
            boot_auprcs.append(average_precision_score(y_true[boot_idx], y_prob[boot_idx]))
            boot_eces.append(calculate_ece(y_true[boot_idx], y_prob[boot_idx]))
            boot_briers.append(brier_score_loss(y_true[boot_idx], y_prob[boot_idx]))
    lower_p, upper_p = (alpha / 2.0) * 100, (1.0 - alpha / 2.0) * 100
    return {
        "auroc_ci": (np.percentile(boot_aurocs, lower_p), np.percentile(boot_aurocs, upper_p)),
        "auprc_ci": (np.percentile(boot_auprcs, lower_p), np.percentile(boot_auprcs, upper_p)),
        "ece_ci": (np.percentile(boot_eces, lower_p), np.percentile(boot_eces, upper_p)),
        "brier_ci": (np.percentile(boot_briers, lower_p), np.percentile(boot_briers, upper_p)),
    }
