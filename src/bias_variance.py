"""
Rigorous Bias-Variance Decomposition on Face Recognition across 30 Resampled Fits.
Implements identity-stratified sampling with replacement, computes:
  Bias^2 = (1 / N) sum_i || p_bar(x_i) - e_i ||_2^2
  Variance = (1 / (N * R)) sum_{i, r} || p_r(x_i) - p_bar(x_i) ||_2^2
"""

import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score


def resample_within_identity(X_train, y_train, n_classes=40, seed=42):
    """
    Creates a new training set by sampling with replacement 6 faces from
    each identity's 6 training faces.
    Returns:
        X_resampled: (240, 4096)
        y_resampled: (240,)
    """
    rng = np.random.default_rng(seed)
    X_resampled, y_resampled = [], []

    for k in range(n_classes):
        idx = np.where(y_train == k)[0]
        resampled_idx = rng.choice(idx, size=len(idx), replace=True)
        X_resampled.append(X_train[resampled_idx])
        y_resampled.append(y_train[resampled_idx])

    return np.vstack(X_resampled), np.concatenate(y_resampled)


def compute_bias_variance(p_r_matrix, y_true, n_classes=40):
    """
    p_r_matrix: shape (R, N_test, n_classes)
    y_true: shape (N_test,)
    Returns:
        bias_squared: float
        variance: float
    """
    R, N, K = p_r_matrix.shape
    # e_i is the one-hot target
    e_i = label_binarize(y_true, classes=np.arange(K))  # shape (N, K)

    # p_bar(x_i) is the mean over repetitions
    p_bar = np.mean(p_r_matrix, axis=0)  # shape (N, K)

    # Bias^2 = (1 / N) sum_i || p_bar(x_i) - e_i ||_2^2
    bias_squared = float(np.mean(np.sum((p_bar - e_i) ** 2, axis=1)))

    # Variance = (1 / (N * R)) sum_{i, r} || p_r(x_i) - p_bar(x_i) ||_2^2
    variance = float(np.mean(np.sum((p_r_matrix - p_bar) ** 2, axis=2)))

    return bias_squared, variance
