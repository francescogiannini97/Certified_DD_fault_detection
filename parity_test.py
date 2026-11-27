import numpy as np
from scipy import stats

from weighting import weighted_windows


def parity_test(model, w_calibration_noisy, alpha, n_windows):
    residual = model["Ut2"].T @ weighted_windows(model, w_calibration_noisy)
    return dict(Sigma_inv=np.linalg.inv(np.cov(residual)),
                threshold=stats.chi2.ppf(1 - alpha / n_windows, residual.shape[0]))


def parity_statistic(model, test, w_bar):
    residual = model["Ut2"].T @ w_bar
    return np.einsum("ij,ik,kj->j", residual, test["Sigma_inv"], residual)
