"""Data covariance Cd estimated from pre-event noise (BEAT-style).

Two structures (mirroring BEAT):
  - 'variance'    : diagonal Cd = var * I  -> weight is 1/sqrt(var) (cheap, robust)
  - 'exponential' : Toeplitz Cd_ij = var * exp(-|i-j|*dt / T0) -> full Cholesky-inverse weight

Each returns a weight W and logdet(Cd) such that the likelihood evaluates
r^T Cd^-1 r = ||W r||^2.  T0 (correlation time) is estimated from the lag-1
autocorrelation of the pre-event noise. A small diagonal jitter keeps the
Cholesky stable (BEAT uses a QR fallback for the same reason).
"""
import numpy as np


def estimate_noise_variance(noise):
    return float(np.var(noise, ddof=1)) if noise.size > 1 else float(np.var(noise) + 1e-30)


def _correlation_time(noise, deltat):
    """T0 from lag-1 autocorrelation rho1: T0 = -deltat / ln(rho1)."""
    x = noise - noise.mean()
    denom = np.sum(x * x)
    if denom <= 0:
        return deltat
    rho1 = np.sum(x[:-1] * x[1:]) / denom
    rho1 = min(max(rho1, 1e-3), 0.999)
    return -deltat / np.log(rho1)


def variance_weight(noise, n_win):
    """Diagonal weight (n_win,) and logdet for a variance (white-noise) covariance."""
    var = estimate_noise_variance(noise)
    var = max(var, 1e-30)
    w = np.full(n_win, 1.0 / np.sqrt(var))
    logdet = n_win * np.log(var)
    return w, logdet


def exponential_weight(noise, n_win, deltat, jitter_rel=1e-6):
    """Full lower-triangular Cholesky-inverse weight (n_win, n_win) and logdet
    for an exponentially-correlated (Toeplitz) covariance."""
    var = max(estimate_noise_variance(noise), 1e-30)
    T0 = _correlation_time(noise, deltat)
    lags = np.abs(np.subtract.outer(np.arange(n_win), np.arange(n_win))) * deltat
    Cd = var * np.exp(-lags / T0)
    Cd[np.diag_indices(n_win)] += jitter_rel * var
    L = np.linalg.cholesky(Cd)                 # Cd = L L^T
    from scipy.linalg import solve_triangular
    W = solve_triangular(L, np.eye(n_win), lower=True)   # W = L^-1, so ||W r||^2 = r^T Cd^-1 r
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return W, logdet


def build_weights(noise_list, n_win, deltat, structure="variance"):
    """Stack per-target weights and logdets.

    noise_list : list of pre-event noise arrays (one per target)
    Returns (weights, logdets) where weights is (T, N) for 'variance'
    or (T, N, N) for 'exponential', logdets is (T,).
    """
    weights, logdets = [], []
    for noise in noise_list:
        if structure == "variance":
            w, ld = variance_weight(noise, n_win)
        elif structure == "exponential":
            w, ld = exponential_weight(noise, n_win, deltat)
        else:
            raise ValueError(f"unknown covariance structure: {structure}")
        weights.append(w)
        logdets.append(ld)
    return np.asarray(weights), np.asarray(logdets)
