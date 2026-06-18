"""Parameter transforms: unconstrained sampling space <-> physical source params.

Free physical parameters (Tape & Tape 2015 lune + magnitude):
    gamma  in [-pi/6, pi/6]   lune longitude (source type)
    delta  in [-pi/2, pi/2]   lune latitude  (source type)
    kappa  in [0, 2pi]        strike
    h      in (0, 1)          cos(dip)
    sigma  in [-pi/2, pi/2]   rake
    Mw     in [mw_lo, mw_hi]   moment magnitude (linear amplitude of the MT)

Each bounded parameter is sampled on the real line via an inverse-sigmoid
(logit) transform; a uniform prior on the physical parameter contributes the
log-Jacobian ``log sigma(u) + log(1-sigma(u))``. Setting gamma=delta=0 gives a
pure double couple.
"""
import jax
import jax.numpy as jnp
import jax.scipy.special as jspecial

from .smti_bridge import jax_Tape_MT33

# physical bounds
GAMMA = (-jnp.pi / 6.0, jnp.pi / 6.0)
DELTA = (-jnp.pi / 2.0, jnp.pi / 2.0)
KAPPA = (0.0, 2.0 * jnp.pi)
H = (1e-3, 1.0 - 1e-3)          # cos(dip), kept off the boundary (arccos gradient)
SIGMA = (-jnp.pi / 2.0, jnp.pi / 2.0)

TAPE_KEYS = ("gamma", "delta", "kappa", "h", "sigma")


def to_physical(u, lo, hi):
    return lo + (hi - lo) * jax.nn.sigmoid(u)


def log_jacobian(u):
    """log|d(physical)/du| up to the constant log(hi-lo): log sig(u)+log(1-sig(u))."""
    return -(jax.nn.softplus(u) + jax.nn.softplus(-u))


def beta_log_prior(u, a, b):
    """Log-density in unconstrained ``u`` of a Beta(a, b) prior on ``s = sigmoid(u)``.

    The logit Jacobian is folded in, so the contribution is ``a*log s + b*log(1-s)
    - betaln(a, b)``; with ``a=b=1`` this reduces exactly to ``log_jacobian`` (a
    uniform prior on the physical parameter). A symmetric Beta(a, a) with a > 1
    centres ``s`` at 0.5 -- i.e. the physical parameter at its midpoint (gamma=delta=0,
    pure double couple) -- gently regularising the non-DC source components (SMTI-style).
    """
    log_s = -jax.nn.softplus(-u)      # log sigmoid(u)
    log_1ms = -jax.nn.softplus(u)     # log(1 - sigmoid(u))
    return a * log_s + b * log_1ms - jspecial.betaln(a, b)


def mw_to_m0(mw):
    """Moment magnitude -> scalar moment M0 [N*m] (Hanks & Kanamori)."""
    return 10.0 ** (1.5 * mw + 9.1)


def tape_to_m6_ned(gamma, delta, kappa, h, sigma):
    """Tape params -> unit-norm moment tensor as NED 6-vector [mnn,mee,mdd,mne,mnd,med]."""
    M = jax_Tape_MT33(gamma, delta, kappa, h, sigma)   # (3,3), Frobenius norm 1, NED
    return jnp.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]])


def m6_physical(position, mw_bounds, dc=False):
    """Map an unconstrained particle to a physical NED moment tensor [N*m]."""
    if dc:
        gamma = jnp.array(0.0)
        delta = jnp.array(0.0)
    else:
        gamma = to_physical(position["gamma"], *GAMMA)
        delta = to_physical(position["delta"], *DELTA)
    kappa = to_physical(position["kappa"], *KAPPA)
    h = to_physical(position["h"], *H)
    sigma = to_physical(position["sigma"], *SIGMA)
    mw = to_physical(position["mag"], *mw_bounds)

    m6_unit = tape_to_m6_ned(gamma, delta, kappa, h, sigma)
    m0 = mw_to_m0(mw)
    # scalar moment M0 = ||M||_F / sqrt(2); unit MT has ||.||_F = 1, so scale by sqrt(2)*M0
    m6 = jnp.sqrt(2.0) * m0 * m6_unit
    physical = {"gamma": gamma, "delta": delta, "kappa": kappa, "h": h,
                "sigma": sigma, "mw": mw, "m0": m0}
    return m6, physical
