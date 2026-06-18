"""BEAT-style Gaussian waveform likelihood (JAX).

For each waveform group (station-channel) with residual r = d - synth, data
covariance Cd, and a hierarchical log-scale noise hyperparameter hp:

    logL = -0.5 * [ logdet(Cd) + M*(2*hp + log 2pi) + exp(-2*hp) * r^T Cd^-1 r ]

where M is the window length and hp scales the noise standard deviation
(sigma = exp(hp)). The quadratic form r^T Cd^-1 r is evaluated through the
Cholesky-inverse weight W (so r^T Cd^-1 r = ||W r||^2), supplied by covariance.py.
This mirrors BEAT's ``multivariate_normal_chol``.
"""
import jax.numpy as jnp

LOG2PI = jnp.log(2.0 * jnp.pi)


def synth_waveforms(m6, basis):
    """m6: (6,) NED moment tensor; basis: (T, 6, N) -> synthetics (T, N)."""
    return jnp.einsum("k,tkn->tn", m6, basis)


def group_quadratic(residual, weight):
    """r^T Cd^-1 r per group. weight is diagonal (T, N) or full lower-tri inverse (T, N, N)."""
    if weight.ndim == 3:
        wr = jnp.einsum("tij,tj->ti", weight, residual)
    else:
        wr = weight * residual
    return jnp.sum(wr ** 2, axis=1)


def _assemble(quad, data_len, logdet, hp, hp_index):
    hp_t = hp[hp_index]                            # (T,)
    ll = -0.5 * (logdet + data_len * (2.0 * hp_t + LOG2PI) + jnp.exp(-2.0 * hp_t) * quad)
    return jnp.sum(ll)


def waveform_loglike(m6, basis, data, weight, logdet, hp, hp_index):
    """Total Gaussian waveform log-likelihood.

    m6        : (6,)            moment tensor in NED
    basis     : (T, 6, N)       processed MT basis per target
    data      : (T, N)          processed observed waveforms
    weight    : (T, N) or (T, N, N)  Cholesky-inverse noise weights
    logdet    : (T,)            log-determinant of Cd per target
    hp        : (G,)            log-scale noise hyperparameters
    hp_index  : (T,) int        maps each target to its hyperparameter group
    """
    synth = synth_waveforms(m6, basis)
    quad = group_quadratic(data - synth, weight)   # (T,)
    return _assemble(quad, data.shape[1], logdet, hp, hp_index)


def waveform_loglike_autoshift(m6, basis, data_shifts, weight, logdet, hp, hp_index,
                               lag_penalty):
    """Like ``waveform_loglike`` but with a per-trace cross-correlation time shift.

    For each trace the synthetic is compared against time-shifted copies of the
    observed waveform (``data_shifts`` is precomputed and constant), and the shift
    minimizing the quadratic misfit is chosen -- grond-style autoshift that absorbs
    velocity-model / pick timing errors. ``lag_penalty`` is (L,): a small quadratic
    penalty per lag discouraging large shifts.

    data_shifts : (L, T, N)   observed waveforms shifted by each candidate lag
    """
    synth = synth_waveforms(m6, basis)             # (T, N)
    r = data_shifts - synth[None]                  # (L, T, N)
    if weight.ndim == 3:
        wr = jnp.einsum("tij,ltj->lti", weight, r)
    else:
        wr = weight[None] * r
    quad_lag = jnp.sum(wr ** 2, axis=2) + lag_penalty[:, None]   # (L, T)
    quad = jnp.min(quad_lag, axis=0)               # (T,)  best alignment per trace
    return _assemble(quad, data_shifts.shape[2], logdet, hp, hp_index)
