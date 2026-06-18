"""Assemble a WaveformDataset: window + process basis/obs, estimate covariance.

Both observed and synthetic paths produce the SAME structure so the model/sampler
are identical for a synthetic-recovery test and a real-data fit. The P-anchored
window has a fixed length across stations (so all targets share N and the
likelihood vectorizes), spanning [tP - t_pre, tP + t_post] in source-relative time.
"""
from dataclasses import dataclass

import numpy as np

from .processing import process_trace, window_grid
from .covariance import build_weights


@dataclass
class WaveformDataset:
    basis: np.ndarray     # (T, 6, N) processed unit-component MT basis (NED)
    data: np.ndarray      # (T, N) processed observed displacement
    weights: np.ndarray   # (T, N) diagonal or (T, N, N) full Cholesky-inverse weights
    logdet: np.ndarray    # (T,) log det Cd
    hp_index: np.ndarray  # (T,) int -> hyperparameter group
    n_groups: int
    meta: list            # per-target dicts (nslc, channel, ...)


def _assign_groups(meta, group_by):
    if group_by == "global":
        idx = [0] * len(meta)
        return np.array(idx), 1
    if group_by == "channel":
        chans = sorted({m["channel"] for m in meta})
        lut = {c: i for i, c in enumerate(chans)}
        return np.array([lut[m["channel"]] for m in meta]), len(chans)
    if group_by == "station":
        stas = sorted({m["nslc"][1] for m in meta})
        lut = {s: i for i, s in enumerate(stas)}
        return np.array([lut[m["nslc"][1]] for m in meta]), len(stas)
    if group_by == "trace":
        return np.arange(len(meta)), len(meta)
    raise ValueError(f"unknown group_by: {group_by}")


def _process_basis(forward, t_pre, t_post, filt):
    """Per target: processed basis (6, N) on the P-anchored source-relative window."""
    basis_traces = forward.basis_pyrocko_traces()
    arrivals = forward.arrivals()
    deltat = forward.deltat
    grids, B_list = [], []
    for jt, trs6 in enumerate(basis_traces):
        wg = window_grid(arrivals[jt]["tp"], t_pre, t_post, deltat)  # source-relative
        B = np.stack([process_trace(trs6[i], wg, **filt) for i in range(6)])  # (6, N)
        grids.append(wg)
        B_list.append(B)
    return B_list, grids, arrivals, deltat


def build_dataset_synthetic(forward, m6_true, t_pre, t_post, filt,
                            snr=5.0, seed=0, structure="variance", group_by="channel"):
    """Generate observed = synthetic(m6_true) + noise (per-target SNR), and estimate Cd."""
    B_list, grids, _, deltat = _process_basis(forward, t_pre, t_post, filt)
    rng = np.random.default_rng(seed)
    data, noise_list = [], []
    for B in B_list:
        synth = m6_true @ B
        rms = np.sqrt(np.mean(synth ** 2)) + 1e-30
        sigma = rms / snr
        data.append(synth + rng.normal(0.0, sigma, size=B.shape[1]))
        noise_list.append(rng.normal(0.0, sigma, size=B.shape[1]))  # independent realization for Cd
    n_win = B_list[0].shape[1]
    weights, logdet = build_weights(noise_list, n_win, deltat, structure)
    hp_index, n_groups = _assign_groups(forward.meta, group_by)
    return WaveformDataset(np.stack(B_list), np.stack(data), weights, logdet,
                           hp_index, n_groups, list(forward.meta))


def build_dataset_real(forward, observed, event_time, t_pre, t_post, filt,
                       noise_len=60.0, noise_gap=5.0, structure="variance", group_by="channel"):
    """Window/process real observed traces and estimate Cd from pre-P noise."""
    B_list, grids, arrivals, deltat = _process_basis(forward, t_pre, t_post, filt)
    data, noise_list, keep, meta = [], [], [], []
    for jt, B in enumerate(B_list):
        nslc = forward.meta[jt]["nslc"]
        tr = observed.get(nslc)
        if tr is None:
            continue
        # signal window: same source-relative grid shifted to absolute time
        d = process_trace(tr, event_time + grids[jt], **filt)
        # pre-P noise window (absolute), length noise_len ending noise_gap before P
        tp_abs = event_time + arrivals[jt]["tp"]
        n_noise = int(round(noise_len / deltat)) + 1
        noise_grid = (tp_abs - noise_gap - noise_len) + np.arange(n_noise) * deltat
        noise = process_trace(tr, noise_grid, **filt)
        data.append(d)
        noise_list.append(noise)
        keep.append(jt)
        meta.append(forward.meta[jt])
    if not keep:
        raise RuntimeError("no observed traces matched the forward targets")
    basis = np.stack([B_list[jt] for jt in keep])
    n_win = basis.shape[2]
    weights, logdet = build_weights(noise_list, n_win, deltat, structure)
    hp_index, n_groups = _assign_groups(meta, group_by)
    return WaveformDataset(basis, np.stack(data), weights, logdet, hp_index, n_groups, meta)


# --- pick-driven windowing -------------------------------------------------
# These mirror the builders above but anchor each target's window on a
# per-target source-relative pick (e.g. P for vertical, S for horizontals)
# instead of the GF-store traveltime. This is for events where observed phase
# picks (not a velocity model) define the windows. The total window length
# (t_pre + t_post) MUST be equal across targets so all share N and vectorize.


def _process_basis_picks(forward, anchors, wins, filts):
    """Per target: processed basis (6, N) windowed at a per-target source-relative anchor.

    anchors : (T,) source-relative window-anchor time per target (pick - origin)
    wins    : list of (t_pre, t_post) per target
    filts   : list of filt dicts per target
    """
    basis_traces = forward.basis_pyrocko_traces()
    deltat = forward.deltat
    grids, B_list = [], []
    for jt, trs6 in enumerate(basis_traces):
        t_pre, t_post = wins[jt]
        wg = window_grid(anchors[jt], t_pre, t_post, deltat)  # source-relative
        B = np.stack([process_trace(trs6[i], wg, **filts[jt]) for i in range(6)])  # (6, N)
        grids.append(wg)
        B_list.append(B)
    sizes = {B.shape[1] for B in B_list}
    if len(sizes) != 1:
        raise ValueError(f"pick windows produced differing sample counts {sizes}; "
                         "keep (t_pre + t_post) equal across all phases/targets")
    return B_list, grids, deltat


def build_dataset_synthetic_picks(forward, m6_true, anchors, wins, filts,
                                  snr=5.0, seed=0, structure="variance", group_by="trace"):
    """Synthetic recovery test on pick-anchored windows (observed = synth + noise)."""
    B_list, grids, deltat = _process_basis_picks(forward, anchors, wins, filts)
    rng = np.random.default_rng(seed)
    data, noise_list = [], []
    for B in B_list:
        synth = m6_true @ B
        rms = np.sqrt(np.mean(synth ** 2)) + 1e-30
        sigma = rms / snr
        data.append(synth + rng.normal(0.0, sigma, size=B.shape[1]))
        noise_list.append(rng.normal(0.0, sigma, size=B.shape[1]))
    n_win = B_list[0].shape[1]
    weights, logdet = build_weights(noise_list, n_win, deltat, structure)
    hp_index, n_groups = _assign_groups(forward.meta, group_by)
    return WaveformDataset(np.stack(B_list), np.stack(data), weights, logdet,
                           hp_index, n_groups, list(forward.meta))


def build_dataset_real_picks(forward, observed, event_time, anchors, wins, filts,
                             noise_anchors, obs_anchors=None, noise_len=0.8, noise_gap=0.2,
                             structure="variance", group_by="trace"):
    """Window/process real observed traces; estimate Cd from a pre-P noise window
    (anchored on ``noise_anchors``, the P-pick offset).

    ``anchors`` window the synthetic basis; ``obs_anchors`` (default = ``anchors``)
    window the observed trace. Using the model traveltime for ``anchors`` and the
    pick for ``obs_anchors`` aligns each model arrival with its observed pick, so the
    likelihood compares wavetrain shapes (only a small residual time shift remains).

    Targets without a matching observed trace are dropped, so excluded picks can be
    removed simply by omitting them from ``observed``.
    """
    B_list, grids, deltat = _process_basis_picks(forward, anchors, wins, filts)
    if obs_anchors is None:
        obs_anchors = anchors
    data, noise_list, keep, meta = [], [], [], []
    for jt, B in enumerate(B_list):
        nslc = forward.meta[jt]["nslc"]
        tr = observed.get(nslc)
        if tr is None:
            continue
        t_pre, t_post = wins[jt]
        obs_grid = window_grid(obs_anchors[jt], t_pre, t_post, deltat)
        d = process_trace(tr, event_time + obs_grid, **filts[jt])
        tp_abs = event_time + noise_anchors[jt]
        n_noise = int(round(noise_len / deltat)) + 1
        noise_grid = (tp_abs - noise_gap - noise_len) + np.arange(n_noise) * deltat
        noise = process_trace(tr, noise_grid, **filts[jt])
        data.append(d)
        noise_list.append(noise)
        keep.append(jt)
        meta.append(forward.meta[jt])
    if not keep:
        raise RuntimeError("no observed traces matched the forward targets")
    basis = np.stack([B_list[jt] for jt in keep])
    n_win = basis.shape[2]
    weights, logdet = build_weights(noise_list, n_win, deltat, structure)
    hp_index, n_groups = _assign_groups(meta, group_by)
    return WaveformDataset(basis, np.stack(data), weights, logdet, hp_index, n_groups, meta)
