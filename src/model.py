"""Assemble the posterior: priors, waveform likelihood, particle init, extraction."""
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from . import parameterization as P
from .likelihood import waveform_loglike, waveform_loglike_autoshift, polarity_loglike


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
                       gamma_beta=(3.0, 3.0), delta_beta=(3.0, 3.0), polarity=None,
                       location=None):
    """Return (logprior_fn, loglikelihood_fn) over unconstrained particle dicts.

    max_shift>0 enables grond-style per-trace cross-correlation time-shift (in samples).
    gamma_beta/delta_beta are the (alpha, beta) of the Beta priors on the lune
    coordinates; symmetric values >1 regularise toward double couple (gamma=delta=0).
    polarity (optional) adds a first-motion (Pz) probit term; a dict with keys
    a_pol (P,6), inc (P,), sigma (float), weight (float).
    location (optional) enables discrete source-location sampling over the green
    cloud: a dict with basis_all (P,T,6,N), a_pol_all (P,Npol,6) or None,
    axes (3 coord arrays, km), pid_lut (nx,ny,nz) int, lo (3,), hi (3,).
    The particle gains a "loc" field (3,) mapped to xyz in [lo,hi] (uniform
    prior) and snapped to the nearest grid node inside the likelihood.
    """
    basis = jnp.asarray(dataset.basis)        # (T, 6, N)
    weight = jnp.asarray(dataset.weights)     # (T, N) or (T, N, N)
    logdet = jnp.asarray(dataset.logdet)      # (T,)
    hp_index = jnp.asarray(dataset.hp_index)  # (T,) int

    a_pol = None
    if polarity is not None:
        a_pol = jnp.asarray(polarity["a_pol"])
        pol_inc = jnp.asarray(polarity["inc"])
        pol_sigma = float(polarity["sigma"])
        pol_w = float(polarity["weight"])

    if location is not None:
        basis_all = jnp.asarray(location["basis_all"])      # (P, T, 6, N)
        apol_all = (jnp.asarray(location["a_pol_all"])
                    if polarity is not None else None)
        loc_axes = [jnp.asarray(a) for a in location["axes"]]
        pid_lut = jnp.asarray(location["pid_lut"])
        loc_lo = jnp.asarray(location["lo"])
        loc_hi = jnp.asarray(location["hi"])

        def resolve(position):
            xyz = P.to_physical(position["loc"], loc_lo, loc_hi)
            i, j, k = (jnp.argmin(jnp.abs(ax - xyz[q]))
                       for q, ax in enumerate(loc_axes))
            pid = pid_lut[i, j, k]
            return basis_all[pid], (apol_all[pid] if apol_all is not None else None)
    else:
        def resolve(position):
            return basis, a_pol

    def add_polarity(ll, m6, ap):
        if polarity is None:
            return ll
        return ll + pol_w * polarity_loglike(m6, ap, pol_sigma, pol_inc)

    if max_shift and max_shift > 0:
        lags = np.arange(-int(max_shift), int(max_shift) + 1)
        data_shifts = jnp.asarray(make_data_shifts(np.asarray(dataset.data), lags))
        lag_penalty = jnp.asarray(lag_penalty_coef * (lags.astype(float) ** 2))

        def loglikelihood_fn(position):
            m6, _ = P.m6_physical(position, mw_bounds, dc=dc)
            hp = P.to_physical(position["hp"], *hp_bounds)
            basis_p, ap = resolve(position)
            ll = waveform_loglike_autoshift(m6, basis_p, data_shifts, weight, logdet,
                                            hp, hp_index, lag_penalty)
            return add_polarity(ll, m6, ap)
    else:
        data = jnp.asarray(dataset.data)

        def loglikelihood_fn(position):
            m6, _ = P.m6_physical(position, mw_bounds, dc=dc)
            hp = P.to_physical(position["hp"], *hp_bounds)   # (G,)
            basis_p, ap = resolve(position)
            ll = waveform_loglike(m6, basis_p, data, weight, logdet, hp, hp_index)
            return add_polarity(ll, m6, ap)

    def logprior_fn(position):
        lp = 0.0
        if not dc:
            # Beta priors on the lune coordinates (regularise toward double couple)
            lp += P.beta_log_prior(position["gamma"], *gamma_beta)
            lp += P.beta_log_prior(position["delta"], *delta_beta)
        for k in ("kappa", "h", "sigma", "mag"):
            lp += P.log_jacobian(position[k])
        lp += jnp.sum(P.log_jacobian(position["hp"]))
        if location is not None:
            lp += jnp.sum(P.log_jacobian(position["loc"]))
        return lp

    return logprior_fn, loglikelihood_fn


def init_particles(rng_key, n_particles, n_groups, dc=False,
                   gamma_beta=(3.0, 3.0), delta_beta=(3.0, 3.0),
                   sample_location=False):
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
    if sample_location:
        parts["loc"] = logit_uniform(keys[7], (n_particles, 3))
    return parts


def extract_posterior(particles, weights, mw_bounds, hp_bounds, dc=False,
                      loc_bounds=None):
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
    if loc_bounds is not None and "loc" in particles:
        out["loc_xyz"] = np.asarray(P.to_physical(
            particles["loc"], jnp.asarray(loc_bounds[0]), jnp.asarray(loc_bounds[1])))
    return out
