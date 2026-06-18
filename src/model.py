"""Assemble the posterior: priors, waveform likelihood, particle init, extraction."""
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from . import parameterization as P
from .likelihood import waveform_loglike, waveform_loglike_autoshift


def make_data_shifts(data, lags):
    """Precompute observed waveforms shifted by each integer lag (zero-filled). (L,T,N)."""
    T, N = data.shape
    out = np.zeros((len(lags), T, N))
    for i, lag in enumerate(lags):
        if lag == 0:
            out[i] = data
        elif lag > 0:
            out[i, :, lag:] = data[:, :N - lag]
        else:
            out[i, :, :N + lag] = data[:, -lag:]
    return out


def build_logdensities(dataset, mw_bounds=(4.0, 8.0), hp_bounds=(-2.0, 6.0), dc=False,
                       max_shift=0, lag_penalty_coef=0.0,
                       gamma_beta=(3.0, 3.0), delta_beta=(3.0, 3.0)):
    """Return (logprior_fn, loglikelihood_fn) over unconstrained particle dicts.

    max_shift>0 enables grond-style per-trace cross-correlation time-shift (in samples).
    gamma_beta/delta_beta are the (alpha, beta) of the Beta priors on the lune
    coordinates; symmetric values >1 regularise toward double couple (gamma=delta=0).
    """
    basis = jnp.asarray(dataset.basis)        # (T, 6, N)
    weight = jnp.asarray(dataset.weights)     # (T, N) or (T, N, N)
    logdet = jnp.asarray(dataset.logdet)      # (T,)
    hp_index = jnp.asarray(dataset.hp_index)  # (T,) int

    if max_shift and max_shift > 0:
        lags = np.arange(-int(max_shift), int(max_shift) + 1)
        data_shifts = jnp.asarray(make_data_shifts(np.asarray(dataset.data), lags))
        lag_penalty = jnp.asarray(lag_penalty_coef * (lags.astype(float) ** 2))

        def loglikelihood_fn(position):
            m6, _ = P.m6_physical(position, mw_bounds, dc=dc)
            hp = P.to_physical(position["hp"], *hp_bounds)
            return waveform_loglike_autoshift(m6, basis, data_shifts, weight, logdet,
                                              hp, hp_index, lag_penalty)
    else:
        data = jnp.asarray(dataset.data)

        def loglikelihood_fn(position):
            m6, _ = P.m6_physical(position, mw_bounds, dc=dc)
            hp = P.to_physical(position["hp"], *hp_bounds)   # (G,)
            return waveform_loglike(m6, basis, data, weight, logdet, hp, hp_index)

    def logprior_fn(position):
        lp = 0.0
        if not dc:
            # Beta priors on the lune coordinates (regularise toward double couple)
            lp += P.beta_log_prior(position["gamma"], *gamma_beta)
            lp += P.beta_log_prior(position["delta"], *delta_beta)
        for k in ("kappa", "h", "sigma", "mag"):
            lp += P.log_jacobian(position[k])
        lp += jnp.sum(P.log_jacobian(position["hp"]))
        return lp

    return logprior_fn, loglikelihood_fn


def init_particles(rng_key, n_particles, n_groups, dc=False,
                   gamma_beta=(3.0, 3.0), delta_beta=(3.0, 3.0)):
    """Sample initial particles from the prior (logit of uniform; Beta for gamma/delta)."""
    keys = random.split(rng_key, 8)

    def logit_uniform(key, shape):
        x = random.uniform(key, shape=shape, minval=1e-4, maxval=1.0 - 1e-4, dtype=jnp.float64)
        return jnp.log(x) - jnp.log1p(-x)

    def logit_beta(key, a, b, shape):
        x = random.beta(key, a, b, shape=shape, dtype=jnp.float64)
        x = jnp.clip(x, 1e-4, 1.0 - 1e-4)
        return jnp.log(x) - jnp.log1p(-x)

    parts = {
        "kappa": logit_uniform(keys[2], (n_particles,)),
        "h": logit_uniform(keys[3], (n_particles,)),
        "sigma": logit_uniform(keys[4], (n_particles,)),
        "mag": logit_uniform(keys[5], (n_particles,)),
        "hp": logit_uniform(keys[6], (n_particles, n_groups)),
    }
    if not dc:
        parts["gamma"] = logit_beta(keys[0], gamma_beta[0], gamma_beta[1], (n_particles,))
        parts["delta"] = logit_beta(keys[1], delta_beta[0], delta_beta[1], (n_particles,))
    return parts


def extract_posterior(particles, weights, mw_bounds, hp_bounds, dc=False):
    """Map unconstrained particles to physical Tape params + NED MT6 (vectorized)."""
    def one(position):
        m6, phys = P.m6_physical(position, mw_bounds, dc=dc)
        return m6, phys

    m6, phys = jax.vmap(one)(particles)
    hp = P.to_physical(particles["hp"], *hp_bounds)
    out = {k: np.asarray(v) for k, v in phys.items()}
    out["m6"] = np.asarray(m6)            # (n, 6) NED
    out["hp"] = np.asarray(hp)            # (n, G)
    out["weights"] = np.asarray(weights)
    return out
