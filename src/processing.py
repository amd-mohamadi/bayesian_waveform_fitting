"""Shared trace processing applied IDENTICALLY to observed and synthetic waveforms.

Bandpass + cosine taper + resampling onto a fixed window grid are all LINEAR
operations, so processing each of the 6 MT basis traces and then linearly
combining them is identical to processing the full synthetic. This is what lets
the forward stay an exact matmul in the sampler.
"""
import numpy as np


def cosine_taper(n, n_fade):
    """Cosine taper: 1.0 in the interior, smoothly fading to 0 over n_fade samples at each end."""
    w = np.ones(n)
    if n_fade > 0:
        n_fade = min(n_fade, n // 2)
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_fade) / n_fade))
        w[:n_fade] = ramp
        w[n - n_fade:] = ramp[::-1]
    return w


def process_trace(tr, t_grid, fmin, fmax, order, tfade, demean=True):
    """Bandpass a pyrocko Trace and sample it onto ``t_grid`` (sample times in the
    trace's OWN time frame: source-relative for synthetics, absolute for observed),
    applying a cosine taper of width ``tfade`` seconds at both window ends.

    Returns a float array of length ``len(t_grid)``.
    """
    t = tr.copy()
    y = t.get_ydata().astype(float)
    if demean:
        y = y - y.mean()
    t.set_ydata(y)
    if fmin is not None and fmax is not None:
        t.bandpass(order, fmin, fmax)
    elif fmax is not None:
        t.lowpass(order, fmax)
    elif fmin is not None:
        t.highpass(order, fmin)

    t_tr = t.tmin + np.arange(t.data_len()) * t.deltat
    out = np.interp(t_grid, t_tr, t.get_ydata(), left=0.0, right=0.0)

    deltat = t_grid[1] - t_grid[0]
    n_fade = int(round(tfade / deltat))
    return out * cosine_taper(len(out), n_fade)


def window_grid(t_anchor, t_pre, t_post, deltat):
    """Sample-time grid for a window spanning [t_anchor - t_pre, t_anchor + t_post]."""
    n = int(round((t_pre + t_post) / deltat)) + 1
    return (t_anchor - t_pre) + np.arange(n) * deltat
