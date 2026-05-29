import argparse
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.special import betaln
from matplotlib.lines import Line2D
from obspy import UTCDateTime
from tqdm import tqdm

from run_simulated_inversion import (
    FastSynthesizer,
    L2Likelihood,
    OpenSWPCGFSynthesizer,
    Tape_MT6,
    Tape_MT33,
    calculate_arrival_time,
    load_stations_from_xml,
    load_velocity_model,
)
from synthetic_inversion_softdtw_test import (
    GSOTLikelihood,
    SoftDTWLikelihood,
    decode_unit_to_physical,
    plot_amplitude_beachball,
    weighted_posterior_medoid,
)
from src_smc_mti.data_loader import read_data
from src_smc_mti.data_prep import polarity_matrix
from src_smc_mti.likelihoods import polarity_ln_pdf


UNIT_LOWER = np.zeros(5, dtype=float)
UNIT_UPPER = np.ones(5, dtype=float)
COMP_TOKEN_TO_IDX = {"PZ": 0, "SN": 1, "SE": 2}
COMP_TOKEN_TO_LABEL = {"PZ": "Z (P-wave)", "SN": "N (S-wave)", "SE": "E (S-wave)"}


def _set_seed(seed: int):
    np.random.seed(seed)


def _extract_centered_window(
    x: np.ndarray, start_idx: int, n_samples: int
) -> np.ndarray:
    out = np.zeros(n_samples, dtype=np.float32)
    a0 = max(0, start_idx)
    a1 = min(x.shape[0], start_idx + n_samples)
    if a1 <= a0:
        return out
    b0 = max(0, -start_idx)
    b1 = b0 + (a1 - a0)
    out[b0:b1] = x[a0:a1]
    return out


def _apply_preprocess(x: np.ndarray, fs: float, args) -> np.ndarray:
    return _apply_preprocess_with_band(
        x, fs, args, bp_low=args.bp_low, bp_high=args.bp_high
    )


def _apply_bandpass(
    y: np.ndarray, fs: float, bp_low: float | None, bp_high: float | None, bp_order: int
) -> np.ndarray:
    if bp_low is None or bp_high is None or bp_high <= bp_low:
        return y
    nyq = 0.5 * float(fs)
    low = max(float(bp_low) / nyq, 1e-6)
    high = min(float(bp_high) / nyq, 0.999)
    if high <= low:
        return y
    b, a = scipy.signal.butter(int(bp_order), [low, high], btype="bandpass")
    return scipy.signal.filtfilt(b, a, y, axis=-1)


def _apply_preprocess_with_band(
    x: np.ndarray,
    fs: float,
    args,
    bp_low: float | None,
    bp_high: float | None,
) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    if args.detrend:
        y = scipy.signal.detrend(y, type="linear")
    if args.demean:
        y = y - np.mean(y)
    y = _apply_bandpass(
        y, fs=fs, bp_low=bp_low, bp_high=bp_high, bp_order=args.bp_order
    )
    if args.taper_frac > 0:
        y = y * scipy.signal.windows.tukey(
            len(y), alpha=min(1.0, 2.0 * args.taper_frac)
        )
    return y.astype(np.float32, copy=False)


def parse_components(components_arg: str):
    txt = components_arg.strip()
    if txt.startswith("[") and txt.endswith("]"):
        txt = txt[1:-1]
    toks = [t.strip().upper() for t in txt.split(",") if t.strip()]
    if not toks:
        raise ValueError("--components is empty")
    bad = [t for t in toks if t not in COMP_TOKEN_TO_IDX]
    if bad:
        raise ValueError(
            f"Unknown component token(s): {bad}. Allowed: {list(COMP_TOKEN_TO_IDX.keys())}"
        )
    idx = [COMP_TOKEN_TO_IDX[t] for t in toks]
    labels = [COMP_TOKEN_TO_LABEL[t] for t in toks]
    return toks, idx, labels


def parse_beta_prior(text: str, name: str) -> tuple[float, float]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must be in format 'alpha,beta'")
    alpha = float(parts[0])
    beta = float(parts[1])
    if alpha <= 0 or beta <= 0:
        raise ValueError(f"{name} alpha and beta must be > 0")
    return alpha, beta


def _normalize_trace_weights(weights: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.clip(w, 0.0, None)
    wsum = float(np.sum(w))
    if wsum <= eps:
        raise ValueError("Trace weights must contain positive values")
    w = w / float(np.mean(w))
    return np.asarray(w, dtype=np.float32)


def _build_manual_trace_weights(
    weights_cfg: dict,
    station_ids: list[str],
    comp_tokens: list[str],
) -> np.ndarray:
    nsta = len(station_ids)
    ncomp = len(comp_tokens)
    out = np.ones(nsta * ncomp, dtype=np.float64)

    default_entry = weights_cfg.get("default", 1.0)

    def _entry_weight(entry, token: str):
        if isinstance(entry, (int, float)):
            return float(entry)
        if isinstance(entry, dict):
            if token in entry:
                return float(entry[token])
            if "all" in entry:
                return float(entry["all"])
        return None

    for i, sid in enumerate(station_ids):
        parts = sid.split(".")
        sta_code = parts[1] if len(parts) > 1 else sid
        candidates = [sid, sta_code]

        station_entry = None
        for key in candidates:
            if key in weights_cfg:
                station_entry = weights_cfg[key]
                break

        for j, tok in enumerate(comp_tokens):
            w = None
            if station_entry is not None:
                w = _entry_weight(station_entry, tok)
            if w is None:
                w = _entry_weight(default_entry, tok)
            if w is None:
                w = 1.0
            out[i * ncomp + j] = float(w)

    return _normalize_trace_weights(out)


def _adaptive_weights_from_trace_cost(
    trace_cost: np.ndarray,
    alpha: float,
    floor_frac: float,
    min_cost: float,
    zero_below: float = 0.0,
    min_active: int = 1,
) -> np.ndarray:
    c = np.asarray(trace_cost, dtype=np.float64).reshape(-1)
    c = np.maximum(c, float(min_cost))
    raw = 1.0 / np.power(c, float(alpha))
    if floor_frac > 0.0:
        floor_val = float(floor_frac) * float(np.max(raw))
        raw = np.maximum(raw, floor_val)
    w = _normalize_trace_weights(raw)

    thr = float(zero_below)
    if thr > 0.0:
        keep = w >= thr
        min_active = max(1, int(min_active))
        if int(np.sum(keep)) < min_active:
            order = np.argsort(w)[::-1]
            keep = np.zeros_like(w, dtype=bool)
            keep[order[:min_active]] = True
        w = np.where(keep, w, 0.0)
        w = _normalize_trace_weights(w)
    return w


def _print_trace_weight_summary(
    trace_weights: np.ndarray,
    station_ids: list[str],
    comp_tokens: list[str],
    label: str,
    top_n: int = 8,
):
    ncomp = len(comp_tokens)
    rows = []
    for i, sid in enumerate(station_ids):
        for j, tok in enumerate(comp_tokens):
            rows.append((float(trace_weights[i * ncomp + j]), sid, tok))
    rows_sorted = sorted(rows, key=lambda x: x[0], reverse=True)
    nshow = max(1, min(int(top_n), len(rows_sorted)))
    print(f"{label} (mean-normalized):")
    print("  Highest-weight traces:")
    for w, sid, tok in rows_sorted[:nshow]:
        print(f"    {sid:>16s} {tok:>2s}: {w:.3f}")
    print("  Lowest-weight traces:")
    for w, sid, tok in rows_sorted[-nshow:]:
        print(f"    {sid:>16s} {tok:>2s}: {w:.3f}")


def dc_log_prior_beta(
    particles: np.ndarray,
    gamma_prior: tuple[float, float],
    delta_prior: tuple[float, float],
    eps: float = 1e-9,
) -> np.ndarray:
    gamma = particles[:, 0]
    delta = particles[:, 1]

    gamma_u = (gamma + np.pi / 6.0) / (np.pi / 3.0)
    delta_u = (delta + np.pi / 2.0) / np.pi
    gamma_u = np.clip(gamma_u, eps, 1.0 - eps)
    delta_u = np.clip(delta_u, eps, 1.0 - eps)

    ag, bg = gamma_prior
    ad, bd = delta_prior

    lp_gamma = (
        (ag - 1.0) * np.log(gamma_u)
        + (bg - 1.0) * np.log(1.0 - gamma_u)
        - betaln(ag, bg)
    )
    lp_delta = (
        (ad - 1.0) * np.log(delta_u)
        + (bd - 1.0) * np.log(1.0 - delta_u)
        - betaln(ad, bd)
    )
    return lp_gamma + lp_delta


def build_phase_window_cache_dual(
    stations,
    source_loc,
    velocity_model,
    duration,
    npts,
    p_window_len,
    s_window_len,
    source_delay=0.1,
    time_steps=70,
    p_bp_low=None,
    p_bp_high=None,
    s_bp_low=None,
    s_bp_high=None,
    bp_order=4,
):
    nsta = stations.shape[0]
    dt = duration / float(npts)
    p_win_samples = int(round(float(p_window_len) / dt))
    s_win_samples = int(round(float(s_window_len) / dt))
    p_half = 0.5 * float(p_window_len)
    s_half = 0.5 * float(s_window_len)
    sx_ev, sy_ev, sz_ev = source_loc

    p_idx = np.zeros(nsta, dtype=np.int32)
    s_idx = np.zeros(nsta, dtype=np.int32)
    for i in range(nsta):
        st_x, st_y, st_z = stations[i, 1], stations[i, 2], stations[i, 3]
        dist_m = np.hypot(st_x - sx_ev, st_y - sy_ev)
        t_p = calculate_arrival_time(sz_ev, st_z, dist_m, velocity_model, phase="P")
        t_s = calculate_arrival_time(sz_ev, st_z, dist_m, velocity_model, phase="S")
        p_idx[i] = int(np.floor((t_p + source_delay - p_half) / dt))
        s_idx[i] = int(np.floor((t_s + source_delay - s_half) / dt))

    return {
        "p_idx": p_idx,
        "s_idx": s_idx,
        "npts": int(npts),
        "time_steps": int(time_steps),
        "p_win_samples": int(max(1, p_win_samples)),
        "s_win_samples": int(max(1, s_win_samples)),
        "taper": np.hanning(int(time_steps)).astype(np.float32),
        "p_window_len": float(p_window_len),
        "s_window_len": float(s_window_len),
        "p_bp_low": p_bp_low,
        "p_bp_high": p_bp_high,
        "s_bp_low": s_bp_low,
        "s_bp_high": s_bp_high,
        "bp_order": int(bp_order),
    }


def extract_phase_windows_batch_cached_dual(waveforms, cache):
    bsz, nsta, _, npts = waveforms.shape
    if npts != cache["npts"]:
        raise ValueError(
            f"Waveform length mismatch: got {npts}, expected {cache['npts']} from cache"
        )

    time_steps = cache["time_steps"]
    out = np.zeros((bsz, nsta, 3, time_steps), dtype=np.float32)

    def _extract_component(slice_data, idx, win_samples, win_len, bp_low, bp_high):
        start = max(0, idx)
        end = min(npts, idx + win_samples)
        if end <= start:
            return np.zeros((bsz, time_steps), dtype=np.float32)

        seg = slice_data[:, start:end]
        pad_left = max(0, -idx)
        pad_right = max(0, (idx + win_samples) - npts)
        if pad_left > 0 or pad_right > 0:
            seg = np.pad(seg, ((0, 0), (pad_left, pad_right)), mode="constant")
        if seg.shape[1] != time_steps:
            seg = scipy.signal.resample(seg, time_steps, axis=1)
        fs_win = float(time_steps) / max(float(win_len), 1e-9)
        seg = _apply_bandpass(
            seg,
            fs=fs_win,
            bp_low=bp_low,
            bp_high=bp_high,
            bp_order=int(cache["bp_order"]),
        )
        return np.asarray(seg, dtype=np.float32)

    for i in range(nsta):
        idx_p = int(cache["p_idx"][i])
        idx_s = int(cache["s_idx"][i])
        out[:, i, 0, :] = _extract_component(
            waveforms[:, i, 0, :],
            idx_p,
            int(cache["p_win_samples"]),
            float(cache["p_window_len"]),
            cache["p_bp_low"],
            cache["p_bp_high"],
        )
        out[:, i, 1, :] = _extract_component(
            waveforms[:, i, 1, :],
            idx_s,
            int(cache["s_win_samples"]),
            float(cache["s_window_len"]),
            cache["s_bp_low"],
            cache["s_bp_high"],
        )
        out[:, i, 2, :] = _extract_component(
            waveforms[:, i, 2, :],
            idx_s,
            int(cache["s_win_samples"]),
            float(cache["s_window_len"]),
            cache["s_bp_low"],
            cache["s_bp_high"],
        )

    out *= cache["taper"][None, None, None, :]
    max_vals = np.max(np.abs(out), axis=(2, 3), keepdims=True)
    out = np.divide(out, max_vals, out=np.zeros_like(out), where=max_vals > 1e-9)
    return out


def build_observation_windows(invdata: dict, args):
    station_ids = invdata["station_ids"]
    picks = invdata["picks"]
    waveforms = invdata["waveforms"]

    nsta = len(station_ids)
    tsteps = int(args.time_steps)
    obs = np.zeros((nsta, 3, tsteps), dtype=np.float32)
    keep_ids = []

    p_len = float(args.phase_window_p_len)
    s_len = float(args.phase_window_s_len)

    for sid in station_ids:
        if sid not in picks or sid not in waveforms:
            continue
        wf = waveforms[sid]
        comps = wf["components"]
        if not all(k in comps for k in ("E", "N", "Z")):
            continue

        fs = float(wf["sampling_rate"])
        dt = float(wf["delta"])
        t0 = UTCDateTime(wf["starttime"])

        z = _apply_preprocess_with_band(
            comps["Z"], fs, args, bp_low=args.bp_p_low, bp_high=args.bp_p_high
        )
        n = _apply_preprocess_with_band(
            comps["N"], fs, args, bp_low=args.bp_s_low, bp_high=args.bp_s_high
        )
        e = _apply_preprocess_with_band(
            comps["E"], fs, args, bp_low=args.bp_s_low, bp_high=args.bp_s_high
        )

        p_time = UTCDateTime(picks[sid]["P"])
        s_time = UTCDateTime(picks[sid]["S"])

        p_samples = int(round(p_len / dt))
        s_samples = int(round(s_len / dt))
        p_start = int(np.floor((p_time - t0 - 0.5 * p_len) / dt))
        s_start = int(np.floor((s_time - t0 - 0.5 * s_len) / dt))

        z_win = _extract_centered_window(z, p_start, p_samples)
        n_win = _extract_centered_window(n, s_start, s_samples)
        e_win = _extract_centered_window(e, s_start, s_samples)

        if p_samples != tsteps:
            z_win = scipy.signal.resample(z_win, tsteps).astype(np.float32)
        if s_samples != tsteps:
            n_win = scipy.signal.resample(n_win, tsteps).astype(np.float32)
            e_win = scipy.signal.resample(e_win, tsteps).astype(np.float32)

        keep_ids.append(sid)
        i = len(keep_ids) - 1
        obs[i, 0] = z_win
        obs[i, 1] = n_win
        obs[i, 2] = e_win

    if not keep_ids:
        raise RuntimeError("No valid station windows were built from invdata")

    obs = obs[: len(keep_ids)]
    if args.apply_window_taper:
        taper = np.hanning(tsteps).astype(np.float32)
        obs *= taper[None, None, :]
    if args.normalize_per_station:
        max_vals = np.max(np.abs(obs), axis=(1, 2), keepdims=True)
        obs = np.divide(obs, max_vals, out=np.zeros_like(obs), where=max_vals > 1e-9)

    return obs, keep_ids


def select_station_geometry(stations_all, codes_all, station_ids):
    code_to_idx = {str(c): i for i, c in enumerate(codes_all)}
    rows = []
    out_ids = []
    for sid in station_ids:
        parts = sid.split(".")
        code = parts[1] if len(parts) > 1 else sid
        if code not in code_to_idx:
            continue
        src = stations_all[code_to_idx[code]]
        rows.append([len(rows) + 1, float(src[1]), float(src[2]), float(src[3])])
        out_ids.append(sid)
    if not rows:
        raise RuntimeError("No station geometry matches invdata station_ids")
    return np.asarray(rows, dtype=float), out_ids


def _shift_1d_with_zeros(x: np.ndarray, shift: int) -> np.ndarray:
    y = np.zeros_like(x)
    if shift == 0:
        y[:] = x
    elif shift > 0:
        y[shift:] = x[:-shift]
    else:
        k = -shift
        y[:-k] = x[k:]
    return y


def _best_lag_xcorr(
    obs_1d: np.ndarray, syn_1d: np.ndarray, max_shift: int
) -> tuple[int, float]:
    obs = np.asarray(obs_1d, dtype=np.float64)
    syn = np.asarray(syn_1d, dtype=np.float64)
    obs = obs - np.mean(obs)
    syn = syn - np.mean(syn)

    best_shift = 0
    best_corr = -np.inf
    denom_obs = np.linalg.norm(obs)
    if denom_obs <= 1e-12:
        return 0, 0.0

    for shift in range(-int(max_shift), int(max_shift) + 1):
        syn_s = _shift_1d_with_zeros(syn, shift)
        denom = denom_obs * np.linalg.norm(syn_s)
        if denom <= 1e-12:
            corr = -1.0
        else:
            corr = float(np.dot(obs, syn_s) / denom)
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
    return int(best_shift), float(best_corr)


def align_synthetics_for_plot(
    obs: np.ndarray, syn: np.ndarray, max_shift: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nsta, ncomp, _ = obs.shape
    shifted = np.zeros_like(syn)
    lag_samples = np.zeros((nsta, ncomp), dtype=np.int32)
    lag_corr = np.zeros((nsta, ncomp), dtype=np.float32)

    for i in range(nsta):
        for c in range(ncomp):
            lag, corr = _best_lag_xcorr(obs[i, c], syn[i, c], max_shift=max_shift)
            shifted[i, c] = _shift_1d_with_zeros(syn[i, c], lag)
            lag_samples[i, c] = lag
            lag_corr[i, c] = corr

    return shifted, lag_samples, lag_corr


def plot_obs_vs_best(
    obs,
    best,
    station_ids,
    out_path,
    p_len,
    s_len,
    comp_tokens,
    comp_labels,
    include_mask=None,
    best_label="Best fit",
):
    nsta, ncomp, npts = obs.shape
    t_p = np.linspace(0.0, float(p_len), npts)
    t_s = np.linspace(0.0, float(s_len), npts)
    if include_mask is None:
        include_mask = np.ones((nsta, ncomp), dtype=bool)
    else:
        include_mask = np.asarray(include_mask, dtype=bool)
        if include_mask.shape != (nsta, ncomp):
            raise ValueError(
                f"include_mask shape mismatch: got {include_mask.shape}, expected {(nsta, ncomp)}"
            )

    fig, axes = plt.subplots(
        nsta, ncomp, figsize=(5 * ncomp, 1.8 * nsta), squeeze=False
    )
    has_included = bool(np.any(include_mask))
    has_excluded = bool(np.any(~include_mask))
    legend_handles = [Line2D([0], [0], color="b", lw=1.0, label="Observed")]
    if has_included:
        legend_handles.append(
            Line2D(
                [0], [0], color="r", lw=1.3, linestyle="--", label="Best fit (included)"
            )
        )
    if has_excluded:
        legend_handles.append(
            Line2D(
                [0], [0], color="g", lw=1.3, linestyle="--", label="Best fit (excluded)"
            )
        )

    for i in range(nsta):
        for c in range(ncomp):
            ax = axes[i, c]
            tt = t_p if comp_tokens[c] == "PZ" else t_s
            ax.plot(tt, obs[i, c], "b-", lw=1.0)
            color = "r" if include_mask[i, c] else "g"
            ax.plot(
                tt, best[i, c], linestyle="--", color=color, lw=1.3, label=best_label
            )
            if i == 0:
                ax.set_title(comp_labels[c])
            if c == 0:
                ax.set_ylabel(station_ids[i])
            if i == 0 and c == 0:
                ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_likelihood_model(args):
    if args.likelihood == "softdtw":
        model = SoftDTWLikelihood(
            sigma=args.sigma,
            gamma=args.softdtw_gamma,
            device=args.device,
            downsample_to=args.softdtw_downsample,
            backend=args.softdtw_backend,
            warp_radius=args.softdtw_warp_radius,
            l2_weight=args.softdtw_l2_weight,
            corr_weight=args.softdtw_corr_weight,
            amp_weight=args.softdtw_amp_weight,
            aggregation=args.softdtw_aggregation,
            normalize_mode=args.softdtw_normalize,
        )
        print(f"Using Soft-DTW likelihood backend={args.softdtw_backend}")
    elif args.likelihood == "gsot":
        model = GSOTLikelihood(
            sigma=args.sigma,
            epsilon=args.gsot_epsilon,
            device=args.device,
            downsample_to=args.gsot_downsample,
            n_iters=args.gsot_iters,
            time_weight=args.gsot_time_weight,
            amp_weight=args.gsot_amp_weight,
            slope_weight=args.gsot_slope_weight,
            l2_weight=args.gsot_l2_weight,
            corr_weight=args.gsot_corr_weight,
            amp_penalty_weight=args.gsot_amp_penalty_weight,
            aggregation=args.gsot_aggregation,
            normalize_mode=args.gsot_normalize,
        )
        print("Using GSOT likelihood")
    else:
        model = L2Likelihood(sigma=args.sigma)
        print("Using L2 likelihood")

    anneal_model = (
        model if isinstance(model, (SoftDTWLikelihood, GSOTLikelihood)) else None
    )
    return model, anneal_model


def build_observed_ratio_targets(
    invdata: dict, station_ids: list[str], eps: float = 1e-9
):
    targets = np.full((len(station_ids), 3), np.nan, dtype=np.float32)
    picks = invdata.get("picks", {})
    for i, sid in enumerate(station_ids):
        amp = picks.get(sid, {}).get("amplitude", {})
        pz = amp.get("Pz")
        sh = amp.get("Sh")
        sv = amp.get("Sv")
        if pz is None or sh is None or sv is None:
            continue
        pz = float(abs(pz))
        sh = float(abs(sh))
        sv = float(abs(sv))
        if pz <= 0 or sh <= 0 or sv <= 0:
            continue
        targets[i, 0] = np.log((pz + eps) / (sh + eps))
        targets[i, 1] = np.log((pz + eps) / (sv + eps))
        targets[i, 2] = np.log((sh + eps) / (sv + eps))
    valid = np.isfinite(targets)
    return targets, valid


def gsot_ratio_loglike(
    processed: np.ndarray,
    ratio_targets: np.ndarray,
    valid_mask: np.ndarray,
    sigma: float,
    weight: float,
    eps: float = 1e-9,
) -> np.ndarray:
    if weight <= 0:
        return np.zeros(processed.shape[0], dtype=np.float32)
    if not np.any(valid_mask):
        return np.zeros(processed.shape[0], dtype=np.float32)

    amp = np.sqrt(np.mean(processed**2, axis=3) + eps)  # (B,N,3)
    p = amp[:, :, 0]
    sh = amp[:, :, 1]
    sv = amp[:, :, 2]

    syn = np.stack(
        [
            np.log((p + eps) / (sh + eps)),
            np.log((p + eps) / (sv + eps)),
            np.log((sh + eps) / (sv + eps)),
        ],
        axis=2,
    )

    diff2 = (syn - ratio_targets[None, :, :]) ** 2
    mask = valid_mask[None, :, :]
    sum2 = np.sum(np.where(mask, diff2, 0.0), axis=(1, 2))
    cnt = np.sum(mask, axis=(1, 2))
    cnt = np.maximum(cnt, 1)
    mean2 = sum2 / cnt
    return -0.5 * float(weight) * (mean2 / (float(sigma) ** 2))


def ratio_residual_summary(
    best_synthetic: np.ndarray,
    ratio_targets: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-9,
):
    if not np.any(valid_mask):
        return None
    amp = np.sqrt(np.mean(best_synthetic**2, axis=2) + eps)  # (N,3)
    p = amp[:, 0]
    sh = amp[:, 1]
    sv = amp[:, 2]
    syn = np.stack(
        [
            np.log((p + eps) / (sh + eps)),
            np.log((p + eps) / (sv + eps)),
            np.log((sh + eps) / (sv + eps)),
        ],
        axis=1,
    )
    d = np.abs(syn - ratio_targets)
    d = np.where(valid_mask, d, np.nan)
    med = np.nanmedian(d, axis=0)
    return {"P_SH": float(med[0]), "P_SV": float(med[1]), "SH_SV": float(med[2])}


def l2_normalized_loglike(
    synthetics: np.ndarray,
    observation: np.ndarray,
    sigma: float,
    weight: float,
    aggregation: str = "median",
    eps: float = 1e-9,
    trace_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Scale-invariant L2 penalty.

    Computes per-trace relative L2:
      mean((syn-obs)^2) / (mean(obs^2) + eps)
    then aggregates across traces per particle.
    """
    if weight <= 0.0:
        return np.zeros(synthetics.shape[0], dtype=np.float32)

    bsz, nsta, ncomp, npts = synthetics.shape
    ntr = nsta * ncomp
    syn = synthetics.reshape(bsz, ntr, npts)
    obs = observation.reshape(1, ntr, npts)

    num = np.mean((syn - obs) ** 2, axis=2)
    den = np.mean(obs**2, axis=2) + float(eps)
    rel = num / den

    if trace_weights is not None:
        w = np.asarray(trace_weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != ntr:
            raise ValueError(
                f"trace_weights length mismatch for L2 norm: got {w.shape[0]}, expected {ntr}"
            )
        w = np.clip(w, 0.0, None)
        wsum = float(np.sum(w))
        if wsum <= eps:
            raise ValueError("trace_weights must contain positive values")
        cost = np.sum(rel * w[None, :], axis=1) / wsum
    elif aggregation == "mean":
        cost = np.mean(rel, axis=1)
    else:
        cost = np.median(rel, axis=1)
    return -0.5 * float(weight) * (cost / (float(sigma) ** 2))


def prepare_ray_polarity_constraints(args, event_dir: Path):
    if not args.use_ray_polarity:
        return None

    pol_file = Path(args.polarity_data_file)
    if not pol_file.is_absolute():
        pol_file = event_dir / pol_file
    if not pol_file.exists():
        raise FileNotFoundError(f"Polarity data file not found: {pol_file}")

    phase_tokens = [
        p.strip().upper() for p in args.polarity_phases.split(",") if p.strip()
    ]
    inversion_options = []
    if "PZ" in phase_tokens:
        inversion_options.append("PPolarity")
    if "SH" in phase_tokens:
        inversion_options.append("SHPolarity")
    if not inversion_options:
        raise ValueError("--polarity-phases must include at least one of PZ,SH")

    pol_data = read_data(
        str(pol_file),
        min_p_phase_score=float(args.min_p_phase_score),
        min_s_phase_score=float(args.min_s_phase_score),
        min_ppl_score=float(args.min_ppl_score),
        min_shpl_score=float(args.min_shpl_score),
        min_svpl_score=float(args.min_svpl_score),
        p_polarity_phase=args.p_polarity_phase,
        inversion_options=inversion_options,
        velocity_model_path=args.polarity_velocity_model,
    )
    a_pol, err_pol, inc_pol = polarity_matrix(pol_data, location_samples=False)
    if isinstance(a_pol, bool) or isinstance(err_pol, bool):
        return None

    return {
        "a": np.asarray(a_pol, dtype=float),
        "sigma": np.asarray(err_pol, dtype=float).reshape(-1),
        "incorrect": inc_pol,
        "weight": float(args.polarity_weight),
        "options": inversion_options,
    }


def polarity_loglike_from_particles(particles: np.ndarray, pol_cfg: dict) -> np.ndarray:
    mt6 = Tape_MT6(
        particles[:, 0],
        particles[:, 1],
        particles[:, 2],
        particles[:, 3],
        particles[:, 4],
    )
    mt6 = np.asarray(mt6, dtype=float)
    if mt6.ndim == 1:
        mt6 = mt6.reshape(6, 1)
    ln_pol = polarity_ln_pdf(
        pol_cfg["a"],
        mt6,
        pol_cfg["sigma"],
        pol_cfg["incorrect"],
    )
    ln_pol = np.asarray(ln_pol, dtype=float).reshape(-1)
    return pol_cfg["weight"] * ln_pol


def main():
    parser = argparse.ArgumentParser(
        description="Run waveform inversion from prepared invdata pickle"
    )
    parser.add_argument("--event-dir", type=str, required=True)
    parser.add_argument("--invdata", type=str, default=None)
    parser.add_argument("--stations-dir", type=str, default="stationxml")
    parser.add_argument("--velocity-model", type=str, default="forge.tvel")
    parser.add_argument(
        "--synthetic-backend",
        type=str,
        choices=["axitra", "openswpc_gf"],
        default="axitra",
        help="Forward-model backend: axitra (1D) or openswpc_gf (precomputed GF library)",
    )
    parser.add_argument(
        "--openswpc-gf-file",
        type=str,
        default=None,
        help="Path to packed OpenSWPC GF NPZ file (required for --synthetic-backend openswpc_gf)",
    )
    parser.add_argument("--sampler", type=str, choices=["smc", "cmaes"], default="smc")
    parser.add_argument(
        "--likelihood", type=str, choices=["softdtw", "gsot", "l2"], default="gsot"
    )
    parser.add_argument("--n-particles", type=int, default=160)
    parser.add_argument("--n-stages", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--sigma-end", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--fmax", type=float, default=500.0)
    parser.add_argument(
        "--source-target-freq-hz",
        type=float,
        default=30.0,
        help="Ricker source central frequency used for synthetics (Hz)",
    )
    parser.add_argument(
        "--source-target-freq-hz-list",
        type=str,
        default=None,
        help="Comma-separated sweep list, e.g. '40,60,80,100'",
    )
    parser.add_argument(
        "--source-t0",
        type=float,
        default=None,
        help="Override Ricker rise time t0 directly (seconds). If set, ignores --source-target-freq-hz",
    )
    parser.add_argument("--source-delay", type=float, default=0.1)

    parser.add_argument("--phase-window-len", type=float, default=0.14)
    parser.add_argument("--phase-window-p-len", type=float, default=None)
    parser.add_argument("--phase-window-s-len", type=float, default=None)
    parser.add_argument(
        "--synthetic-phase-window-p-len",
        type=float,
        default=None,
        help="Synthetic-only P window length (s). If unset, uses --phase-window-p-len",
    )
    parser.add_argument(
        "--synthetic-phase-window-s-len",
        type=float,
        default=None,
        help="Synthetic-only S window length (s). If unset, uses --phase-window-s-len",
    )
    parser.add_argument("--time-steps", type=int, default=70)
    parser.add_argument(
        "--auto-time-steps",
        action="store_true",
        help="Auto-scale time_steps using invdata preprocess_hint.upsample_factor",
    )
    parser.add_argument("--bp-low", type=float, default=8.0)
    parser.add_argument("--bp-high", type=float, default=150.0)
    parser.add_argument(
        "--bp-p-low",
        type=float,
        default=None,
        help="Bandpass low-cut for P-phase/Z channel (defaults to --bp-low)",
    )
    parser.add_argument(
        "--bp-p-high",
        type=float,
        default=None,
        help="Bandpass high-cut for P-phase/Z channel (defaults to --bp-high)",
    )
    parser.add_argument(
        "--bp-s-low",
        type=float,
        default=None,
        help="Bandpass low-cut for S-phase/N,E channels (defaults to --bp-low)",
    )
    parser.add_argument(
        "--bp-s-high",
        type=float,
        default=None,
        help="Bandpass high-cut for S-phase/N,E channels (defaults to --bp-high)",
    )
    parser.add_argument("--bp-order", type=int, default=4)
    parser.add_argument("--taper-frac", type=float, default=0.05)
    parser.add_argument("--detrend", action="store_true", default=True)
    parser.add_argument("--demean", action="store_true", default=True)
    parser.add_argument("--apply-window-taper", action="store_true", default=True)
    parser.add_argument("--normalize-per-station", action="store_true", default=True)
    parser.add_argument(
        "--lag-max-shift-samples",
        type=int,
        default=15,
        help="Max sample shift (+/-) for lag-aligned plot diagnostics",
    )
    parser.add_argument(
        "--plot-align-lag",
        action="store_true",
        default=True,
        help="Estimate per-trace lag and shift synthetics before plotting",
    )
    parser.add_argument(
        "--no-plot-align-lag",
        action="store_false",
        dest="plot_align_lag",
        help="Disable lag-aligned plotting and use raw best-fit synthetics",
    )
    parser.add_argument(
        "--components",
        type=str,
        default="PZ,SN,SE",
        help="Comma-separated components to use, e.g. PZ,SN,SE or PZ",
    )
    parser.add_argument(
        "--use-ray-polarity",
        action="store_true",
        help="Add ray-pattern polarity likelihood term from picks.dat (Pz/SH)",
    )
    parser.add_argument(
        "--polarity-phases",
        type=str,
        default="PZ,SH",
        help="Comma-separated phase polarities for ray-pattern term: PZ,SH",
    )
    parser.add_argument(
        "--polarity-data-file",
        type=str,
        default="picks.dat",
        help="Polarity pick file path (relative to event-dir if not absolute)",
    )
    parser.add_argument(
        "--polarity-weight",
        type=float,
        default=1.0,
        help="Weight for ray-pattern polarity log-likelihood",
    )
    parser.add_argument("--p-polarity-phase", type=str, default="Pz")
    parser.add_argument("--min-p-phase-score", type=float, default=0.55)
    parser.add_argument("--min-s-phase-score", type=float, default=0.55)
    parser.add_argument("--min-ppl-score", type=float, default=0.5)
    parser.add_argument("--min-shpl-score", type=float, default=0.5)
    parser.add_argument("--min-svpl-score", type=float, default=0.5)
    parser.add_argument(
        "--polarity-velocity-model",
        type=str,
        default=None,
        help="Velocity model path used by TauP in polarity read_data",
    )

    parser.add_argument("--softdtw-gamma", type=float, default=0.05)
    parser.add_argument("--softdtw-gamma-end", type=float, default=0.01)
    parser.add_argument("--softdtw-downsample", type=int, default=48)
    parser.add_argument(
        "--softdtw-backend", type=str, choices=["torch", "tslearn"], default="tslearn"
    )
    parser.add_argument("--softdtw-warp-radius", type=int, default=8)
    parser.add_argument("--softdtw-l2-weight", type=float, default=0.15)
    parser.add_argument("--softdtw-corr-weight", type=float, default=0.10)
    parser.add_argument("--softdtw-amp-weight", type=float, default=0.12)
    parser.add_argument(
        "--softdtw-normalize",
        type=str,
        choices=["none", "demean", "zscore"],
        default="demean",
    )
    parser.add_argument(
        "--softdtw-aggregation", type=str, choices=["mean", "median"], default="median"
    )

    parser.add_argument("--gsot-epsilon", type=float, default=0.03)
    parser.add_argument("--gsot-epsilon-end", type=float, default=0.015)
    parser.add_argument("--gsot-downsample", type=int, default=48)
    parser.add_argument("--gsot-iters", type=int, default=40)
    parser.add_argument("--gsot-time-weight", type=float, default=0.30)
    parser.add_argument("--gsot-amp-weight", type=float, default=1.0)
    parser.add_argument("--gsot-slope-weight", type=float, default=0.35)
    parser.add_argument("--gsot-l2-weight", type=float, default=0.08)
    parser.add_argument("--gsot-corr-weight", type=float, default=0.06)
    parser.add_argument("--gsot-amp-penalty-weight", type=float, default=0.08)
    parser.add_argument(
        "--gsot-ratio-weight",
        type=float,
        default=0.25,
        help="Weight for observed pick amplitude-ratio penalty (P/SH, P/SV, SH/SV)",
    )
    parser.add_argument(
        "--gsot-ratio-sigma",
        type=float,
        default=0.5,
        help="Sigma for log-amplitude-ratio residuals",
    )
    parser.add_argument(
        "--gsot-normalize",
        type=str,
        choices=["none", "demean", "zscore"],
        default="demean",
    )
    parser.add_argument(
        "--gsot-aggregation", type=str, choices=["mean", "median"], default="median"
    )
    parser.add_argument(
        "--l2norm-weight",
        type=float,
        default=0.0,
        help="Additional normalized-L2 log-likelihood weight (added to base likelihood)",
    )
    parser.add_argument(
        "--l2norm-sigma",
        type=float,
        default=0.25,
        help="Sigma for normalized-L2 penalty",
    )
    parser.add_argument(
        "--l2norm-aggregation",
        type=str,
        choices=["mean", "median"],
        default="median",
        help="Aggregation across traces for normalized-L2 penalty",
    )
    parser.add_argument(
        "--station-channel-weights-file",
        type=str,
        default=None,
        help=(
            "Optional JSON file with manual per station-channel weights. "
            "Supports keys by full station_id or station code; values can be scalar or {PZ,SN,SE/all}."
        ),
    )
    parser.add_argument(
        "--adaptive-gsot-trace-weighting",
        action="store_true",
        help=(
            "Enable adaptive per station-channel weighting from GSOT per-trace cost "
            "(currently supported for --sampler smc with --likelihood gsot)."
        ),
    )
    parser.add_argument(
        "--adaptive-gsot-warmup-stages",
        type=int,
        default=5,
        help="Number of initial SMC stages with uniform trace weights before adaptive GSOT weighting starts",
    )
    parser.add_argument(
        "--adaptive-gsot-update-interval",
        type=int,
        default=5,
        help="Recompute adaptive GSOT trace weights every N stages after warmup",
    )
    parser.add_argument(
        "--adaptive-gsot-weight-alpha",
        type=float,
        default=1.0,
        help="Power alpha in adaptive weight rule w ~ 1 / cost^alpha",
    )
    parser.add_argument(
        "--adaptive-gsot-weight-floor",
        type=float,
        default=0.10,
        help="Floor fraction of max raw weight before normalization (keeps weak traces non-zero)",
    )
    parser.add_argument(
        "--adaptive-gsot-min-cost",
        type=float,
        default=1e-6,
        help="Minimum per-trace GSOT cost used in adaptive weighting for numerical stability",
    )
    parser.add_argument(
        "--adaptive-gsot-zero-below",
        type=float,
        default=0.0,
        help=(
            "Zero out trace weights below this threshold after normalization "
            "(mean-normalized units; 0 disables exclusion)"
        ),
    )
    parser.add_argument(
        "--adaptive-gsot-min-active-traces",
        type=int,
        default=1,
        help="Minimum number of active traces to keep when zero-below exclusion is used",
    )

    parser.add_argument("--cmaes-sigma0", type=float, default=0.22)
    parser.add_argument(
        "--dc-only",
        action="store_true",
        help="Hard DC mode: fix gamma=delta=0",
    )
    parser.add_argument(
        "--gamma-beta-prior",
        type=str,
        default="1,1",
        help="Soft DC prior for gamma (unit space) as alpha,beta",
    )
    parser.add_argument(
        "--delta-beta-prior",
        type=str,
        default="1,1",
        help="Soft DC prior for delta (unit space) as alpha,beta",
    )
    parser.add_argument(
        "--dc-prior-weight",
        type=float,
        default=1.0,
        help="Weight multiplier for soft DC beta-prior",
    )
    parser.add_argument("--_sweep-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    gamma_beta_prior = parse_beta_prior(args.gamma_beta_prior, "--gamma-beta-prior")
    delta_beta_prior = parse_beta_prior(args.delta_beta_prior, "--delta-beta-prior")
    if args.polarity_velocity_model is None:
        args.polarity_velocity_model = args.velocity_model

    if args.source_target_freq_hz_list and not args._sweep_child:
        if args.source_t0 is not None:
            raise ValueError(
                "--source-target-freq-hz-list cannot be used together with --source-t0"
            )
        freqs = [
            float(x.strip())
            for x in str(args.source_target_freq_hz_list).split(",")
            if x.strip()
        ]
        if not freqs:
            raise ValueError("Empty --source-target-freq-hz-list")

        raw_args = sys.argv[1:]
        filtered = []
        skip_next = False
        for i, tok in enumerate(raw_args):
            if skip_next:
                skip_next = False
                continue
            if tok == "--source-target-freq-hz-list":
                skip_next = True
                continue
            if tok.startswith("--source-target-freq-hz-list="):
                continue
            if tok == "--source-target-freq-hz":
                skip_next = True
                continue
            if tok.startswith("--source-target-freq-hz="):
                continue
            filtered.append(tok)

        scores = []
        print(f"Running source-frequency sweep: {freqs}")
        for f_hz in freqs:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                *filtered,
                "--source-target-freq-hz",
                str(f_hz),
                "--_sweep-child",
            ]
            print(f"\n--- Sweep run: f0={f_hz:.3f} Hz ---")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.stdout:
                print(proc.stdout)
            if proc.returncode != 0:
                if proc.stderr:
                    print(proc.stderr)
                raise RuntimeError(f"Sweep run failed for f0={f_hz:.3f} Hz")

            m = re.search(r"WAVEFORM_FIT_SCORE=([-+eE0-9\.]+)", proc.stdout)
            if m is None:
                raise RuntimeError(
                    f"Sweep run for f0={f_hz:.3f} Hz did not report WAVEFORM_FIT_SCORE"
                )
            score = float(m.group(1))
            scores.append((f_hz, score))

        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
        print("\nSource-frequency sweep results (higher is better):")
        for f_hz, sc in scores_sorted:
            print(f"  f0={f_hz:8.3f} Hz | score={sc:.6e}")
        best_f, best_s = scores_sorted[0]
        print(f"Best source frequency: {best_f:.3f} Hz (score={best_s:.6e})")
        return

    _set_seed(args.seed)

    event_dir = Path(args.event_dir)
    inv_path = Path(args.invdata) if args.invdata else (event_dir / "invdata.pkl")
    if not inv_path.exists():
        raise FileNotFoundError(f"invdata pickle not found: {inv_path}")

    polarity_cfg = prepare_ray_polarity_constraints(args, event_dir)
    if polarity_cfg is not None:
        print(
            "Ray-pattern polarity enabled: "
            f"options={polarity_cfg['options']}, weight={polarity_cfg['weight']}, "
            f"n_obs={polarity_cfg['a'].shape[0]}"
        )

    with inv_path.open("rb") as f:
        invdata = pickle.load(f)

    if args.auto_time_steps:
        upsample_factor = float(
            invdata.get("preprocess_hint", {}).get("upsample_factor", 1.0)
        )
        if upsample_factor > 0:
            auto_steps = max(4, int(round(float(args.time_steps) * upsample_factor)))
            if auto_steps != args.time_steps:
                print(
                    f"Auto time-steps enabled: base={args.time_steps}, "
                    f"upsample_factor={upsample_factor:.3f}, using={auto_steps}"
                )
            else:
                print(
                    f"Auto time-steps enabled: upsample_factor={upsample_factor:.3f}, "
                    f"time_steps remains {args.time_steps}"
                )
            args.time_steps = auto_steps

    if args.phase_window_p_len is None:
        args.phase_window_p_len = float(args.phase_window_len)
    if args.phase_window_s_len is None:
        args.phase_window_s_len = float(args.phase_window_len)
    if args.synthetic_phase_window_p_len is None:
        args.synthetic_phase_window_p_len = float(args.phase_window_p_len)
    if args.synthetic_phase_window_s_len is None:
        args.synthetic_phase_window_s_len = float(args.phase_window_s_len)
    if args.bp_p_low is None:
        args.bp_p_low = args.bp_low
    if args.bp_p_high is None:
        args.bp_p_high = args.bp_high
    if args.bp_s_low is None:
        args.bp_s_low = args.bp_low
    if args.bp_s_high is None:
        args.bp_s_high = args.bp_high

    print(
        f"Observed phase windows: P={args.phase_window_p_len:.3f}s, "
        f"S={args.phase_window_s_len:.3f}s"
    )
    print(
        f"Synthetic phase windows: P={args.synthetic_phase_window_p_len:.3f}s, "
        f"S={args.synthetic_phase_window_s_len:.3f}s"
    )
    print(
        "Bandpass by phase: "
        f"P=[{args.bp_p_low:.3f}, {args.bp_p_high:.3f}] Hz, "
        f"S=[{args.bp_s_low:.3f}, {args.bp_s_high:.3f}] Hz"
    )

    comp_tokens, comp_indices, comp_labels = parse_components(args.components)
    print(f"Using inversion components: {comp_tokens}")
    if args.dc_only:
        print("DC mode: hard constraint enabled (gamma=delta=0)")
    else:
        print(
            "DC mode: soft prior "
            f"gamma_beta={gamma_beta_prior}, delta_beta={delta_beta_prior}, "
            f"weight={args.dc_prior_weight}"
        )

    source = invdata["source"]
    source_loc = (float(source["x"]), float(source["y"]), float(source["z"]))
    m0 = 10 ** (1.5 * float(source["mw"]) + 9.1)

    obs, obs_station_ids = build_observation_windows(invdata, args)
    stations_all, codes_all, _, _ = load_stations_from_xml(args.stations_dir)
    stations_sel, geo_station_ids = select_station_geometry(
        stations_all, codes_all, obs_station_ids
    )

    idx_map = [obs_station_ids.index(sid) for sid in geo_station_ids]
    obs = obs[idx_map]
    obs = obs[:, comp_indices, :]
    station_ids = geo_station_ids

    ratio_targets, ratio_valid = build_observed_ratio_targets(invdata, station_ids)
    n_ratio_valid = int(np.sum(np.all(ratio_valid, axis=1)))
    ratio_compatible = set(comp_tokens) == {"PZ", "SN", "SE"}
    if args.gsot_ratio_weight > 0 and ratio_compatible:
        print(
            f"GSOT ratio penalty enabled: weight={args.gsot_ratio_weight}, "
            f"sigma={args.gsot_ratio_sigma}, valid_stations={n_ratio_valid}/{len(station_ids)}"
        )
    elif args.gsot_ratio_weight > 0 and not ratio_compatible:
        print("GSOT ratio penalty disabled: requires components exactly {PZ,SN,SE}")
    if args.l2norm_weight > 0:
        print(
            "Normalized-L2 penalty enabled: "
            f"weight={args.l2norm_weight}, sigma={args.l2norm_sigma}, "
            f"aggregation={args.l2norm_aggregation}"
        )

    manual_trace_weights = None
    if args.station_channel_weights_file is not None:
        if not (args.sampler == "smc" and args.likelihood == "gsot"):
            raise ValueError(
                "--station-channel-weights-file is currently supported only with --sampler smc and --likelihood gsot"
            )
        weights_path = Path(args.station_channel_weights_file)
        if not weights_path.is_absolute():
            weights_path = event_dir / weights_path
        if not weights_path.exists():
            raise FileNotFoundError(f"Manual weights file not found: {weights_path}")
        with weights_path.open("r", encoding="utf-8") as f:
            weights_cfg = json.load(f)
        if not isinstance(weights_cfg, dict):
            raise ValueError("Manual weights JSON must be an object/dict")
        manual_trace_weights = _build_manual_trace_weights(
            weights_cfg=weights_cfg,
            station_ids=station_ids,
            comp_tokens=comp_tokens,
        )
        print(f"Manual station-channel weights loaded from {weights_path}")
        _print_trace_weight_summary(
            manual_trace_weights,
            station_ids,
            comp_tokens,
            label="Manual trace weights",
            top_n=6,
        )

    adaptive_trace_weighting = bool(args.adaptive_gsot_trace_weighting)
    if adaptive_trace_weighting and not (
        args.sampler == "smc" and args.likelihood == "gsot"
    ):
        print(
            "Adaptive GSOT trace weighting requested but disabled: "
            "currently only supported for sampler=smc and likelihood=gsot"
        )
        adaptive_trace_weighting = False
    if adaptive_trace_weighting and manual_trace_weights is not None:
        print(
            "Adaptive GSOT trace weighting disabled because manual weights are provided"
        )
        adaptive_trace_weighting = False
    if adaptive_trace_weighting:
        if args.adaptive_gsot_warmup_stages < 1:
            raise ValueError("--adaptive-gsot-warmup-stages must be >= 1")
        if args.adaptive_gsot_update_interval < 1:
            raise ValueError("--adaptive-gsot-update-interval must be >= 1")
        if args.adaptive_gsot_weight_alpha <= 0:
            raise ValueError("--adaptive-gsot-weight-alpha must be > 0")
        if args.adaptive_gsot_weight_floor < 0:
            raise ValueError("--adaptive-gsot-weight-floor must be >= 0")
        if args.adaptive_gsot_min_cost <= 0:
            raise ValueError("--adaptive-gsot-min-cost must be > 0")
        if args.adaptive_gsot_zero_below < 0:
            raise ValueError("--adaptive-gsot-zero-below must be >= 0")
        if args.adaptive_gsot_min_active_traces < 1:
            raise ValueError("--adaptive-gsot-min-active-traces must be >= 1")
        print(
            "Adaptive GSOT trace weighting enabled: "
            f"warmup={args.adaptive_gsot_warmup_stages}, "
            f"update_interval={args.adaptive_gsot_update_interval}, "
            f"alpha={args.adaptive_gsot_weight_alpha}, "
            f"floor={args.adaptive_gsot_weight_floor}, "
            f"min_cost={args.adaptive_gsot_min_cost}, "
            f"zero_below={args.adaptive_gsot_zero_below}, "
            f"min_active={args.adaptive_gsot_min_active_traces}"
        )

    velocity_model = load_velocity_model(args.velocity_model)

    if args.source_t0 is not None:
        synth_t0 = float(args.source_t0)
    else:
        if args.source_target_freq_hz <= 0:
            raise ValueError("--source-target-freq-hz must be > 0")
        synth_t0 = 1.0 / (np.pi * float(args.source_target_freq_hz))

    print(
        f"Synthetic source settings: target_freq={args.source_target_freq_hz:.3f} Hz, "
        f"t0={synth_t0:.6f} s, delay={args.source_delay:.4f} s"
    )

    if args.likelihood == "softdtw":
        likelihood_model = SoftDTWLikelihood(
            sigma=args.sigma,
            gamma=args.softdtw_gamma,
            device=args.device,
            downsample_to=args.softdtw_downsample,
            backend=args.softdtw_backend,
            warp_radius=args.softdtw_warp_radius,
            l2_weight=args.softdtw_l2_weight,
            corr_weight=args.softdtw_corr_weight,
            amp_weight=args.softdtw_amp_weight,
            aggregation=args.softdtw_aggregation,
            normalize_mode=args.softdtw_normalize,
        )
        print(f"Using Soft-DTW likelihood backend={args.softdtw_backend}")
    elif args.likelihood == "gsot":
        likelihood_model = GSOTLikelihood(
            sigma=args.sigma,
            epsilon=args.gsot_epsilon,
            device=args.device,
            downsample_to=args.gsot_downsample,
            n_iters=args.gsot_iters,
            time_weight=args.gsot_time_weight,
            amp_weight=args.gsot_amp_weight,
            slope_weight=args.gsot_slope_weight,
            l2_weight=args.gsot_l2_weight,
            corr_weight=args.gsot_corr_weight,
            amp_penalty_weight=args.gsot_amp_penalty_weight,
            aggregation=args.gsot_aggregation,
            normalize_mode=args.gsot_normalize,
        )
        print("Using GSOT likelihood")
    else:
        likelihood_model = L2Likelihood(sigma=args.sigma)
        print("Using L2 likelihood")

    anneal_model = (
        likelihood_model
        if isinstance(likelihood_model, (SoftDTWLikelihood, GSOTLikelihood))
        else None
    )

    if args.synthetic_backend == "axitra":
        synthesizer = FastSynthesizer(velocity_model, stations_sel, source_loc)
        synthesizer.duration = float(args.duration)
        synthesizer.fmax = float(args.fmax)
        synthesizer.t0 = float(synth_t0)
        synthesizer.setup()
    else:
        if not args.openswpc_gf_file:
            raise ValueError(
                "--openswpc-gf-file is required when --synthetic-backend openswpc_gf"
            )
        synthesizer = OpenSWPCGFSynthesizer(
            args.openswpc_gf_file,
            stations_sel,
            source_loc=source_loc,
            duration=float(args.duration),
            fmax=float(args.fmax),
            t0=float(synth_t0),
        )
        synthesizer.setup()
        print(f"Using OpenSWPC GF backend: {args.openswpc_gf_file}")
    if synthesizer.ap is None:
        raise RuntimeError("Failed to initialize synthesizer")
    phase_cache = build_phase_window_cache_dual(
        stations=stations_sel,
        source_loc=source_loc,
        velocity_model=velocity_model,
        duration=float(args.duration),
        npts=int(synthesizer.ap.npt),
        p_window_len=float(args.synthetic_phase_window_p_len),
        s_window_len=float(args.synthetic_phase_window_s_len),
        source_delay=float(args.source_delay),
        time_steps=args.time_steps,
        p_bp_low=args.bp_p_low,
        p_bp_high=args.bp_p_high,
        s_bp_low=args.bp_s_low,
        s_bp_high=args.bp_s_high,
        bp_order=args.bp_order,
    )

    n_particles = args.n_particles
    best_ll = -np.inf
    best_particle = None
    best_synthetic = None
    history = []
    last_stage_weights = None
    last_stage_particles = None
    last_stage_log_weights = None
    trace_weights = (
        manual_trace_weights.copy() if manual_trace_weights is not None else None
    )

    if args.sampler == "smc":
        particles = np.zeros((n_particles, 5))
        particles[:, 0] = np.random.uniform(-np.pi / 6, np.pi / 6, n_particles)
        particles[:, 1] = np.random.uniform(-np.pi / 2, np.pi / 2, n_particles)
        particles[:, 2] = np.random.uniform(0, 2 * np.pi, n_particles)
        particles[:, 3] = np.random.uniform(0, 1, n_particles)
        particles[:, 4] = np.random.uniform(-np.pi / 2, np.pi / 2, n_particles)
        if args.dc_only:
            particles[:, 0] = 0.0
            particles[:, 1] = 0.0

        for stage in tqdm(range(args.n_stages), desc="SMC stages"):
            if anneal_model is not None and args.n_stages > 1:
                frac = stage / float(args.n_stages - 1)
                anneal_model.sigma = args.sigma * (
                    (args.sigma_end / args.sigma) ** frac
                )
                if isinstance(anneal_model, SoftDTWLikelihood):
                    anneal_model.gamma = args.softdtw_gamma * (
                        (args.softdtw_gamma_end / args.softdtw_gamma) ** frac
                    )
                elif isinstance(anneal_model, GSOTLikelihood):
                    anneal_model.epsilon = args.gsot_epsilon * (
                        (args.gsot_epsilon_end / args.gsot_epsilon) ** frac
                    )

            raw_wfs = synthesizer.synthesize_batch(
                particles, m0, source_delay=float(args.source_delay)
            )
            processed_full = extract_phase_windows_batch_cached_dual(
                raw_wfs, phase_cache
            )
            processed = processed_full[:, :, comp_indices, :]
            if args.likelihood == "gsot":
                log_weights = likelihood_model.compute_log_likelihood(
                    processed, obs, trace_weights=trace_weights
                )
            else:
                log_weights = likelihood_model.compute_log_likelihood(processed, obs)
            if polarity_cfg is not None and polarity_cfg["weight"] != 0.0:
                log_weights = log_weights + polarity_loglike_from_particles(
                    particles, polarity_cfg
                )
            if (not args.dc_only) and args.dc_prior_weight != 0.0:
                log_weights = log_weights + float(
                    args.dc_prior_weight
                ) * dc_log_prior_beta(particles, gamma_beta_prior, delta_beta_prior)
            if (
                args.likelihood == "gsot"
                and args.gsot_ratio_weight > 0
                and ratio_compatible
            ):
                log_weights = log_weights + gsot_ratio_loglike(
                    processed_full,
                    ratio_targets,
                    ratio_valid,
                    sigma=args.gsot_ratio_sigma,
                    weight=args.gsot_ratio_weight,
                )
            if args.l2norm_weight > 0:
                log_weights = log_weights + l2_normalized_loglike(
                    processed,
                    obs,
                    sigma=args.l2norm_sigma,
                    weight=args.l2norm_weight,
                    aggregation=args.l2norm_aggregation,
                    trace_weights=trace_weights,
                )

            bi = int(np.argmax(log_weights))
            if log_weights[bi] > best_ll:
                best_ll = float(log_weights[bi])
                best_particle = particles[bi].copy()
                best_synthetic = processed[bi].copy()

            weights = np.exp(log_weights - np.max(log_weights))
            weights /= np.sum(weights)
            ess = 1.0 / np.sum(weights**2)

            if adaptive_trace_weighting and args.likelihood == "gsot":
                stage_one_based = stage + 1
                if stage_one_based >= int(args.adaptive_gsot_warmup_stages):
                    offset = stage_one_based - int(args.adaptive_gsot_warmup_stages)
                    if offset % int(args.adaptive_gsot_update_interval) == 0:
                        per_trace_cost = likelihood_model.compute_per_trace_cost(
                            processed, obs
                        )
                        stage_trace_cost = np.sum(
                            per_trace_cost * weights[:, None], axis=0
                        )
                        trace_weights = _adaptive_weights_from_trace_cost(
                            stage_trace_cost,
                            alpha=float(args.adaptive_gsot_weight_alpha),
                            floor_frac=float(args.adaptive_gsot_weight_floor),
                            min_cost=float(args.adaptive_gsot_min_cost),
                            zero_below=float(args.adaptive_gsot_zero_below),
                            min_active=int(args.adaptive_gsot_min_active_traces),
                        )
                        n_active = int(np.sum(trace_weights > 0.0))
                        print(
                            f"Adaptive GSOT active traces: {n_active}/{trace_weights.size}"
                        )
                        _print_trace_weight_summary(
                            trace_weights,
                            station_ids,
                            comp_tokens,
                            label=f"Adaptive GSOT trace weights @ stage {stage_one_based}",
                            top_n=6,
                        )

            history.append(np.sum(particles * weights[:, None], axis=0))
            last_stage_weights = weights.copy()
            last_stage_particles = particles.copy()
            last_stage_log_weights = log_weights.copy()

            print(
                f"Stage {stage + 1:02d}/{args.n_stages}: maxLL={np.max(log_weights):.3f}, ESS={ess:.1f}"
            )

            idx = np.random.choice(n_particles, size=n_particles, p=weights)
            particles = particles[idx]
            step_size = 0.1 * (0.9**stage)
            particles += np.random.normal(0, step_size, size=particles.shape)
            particles[:, 0] = np.clip(particles[:, 0], -np.pi / 6, np.pi / 6)
            particles[:, 1] = np.clip(particles[:, 1], -np.pi / 2, np.pi / 2)
            particles[:, 2] = np.mod(particles[:, 2], 2 * np.pi)
            particles[:, 3] = np.clip(particles[:, 3], 0, 1)
            particles[:, 4] = np.clip(particles[:, 4], -np.pi / 2, np.pi / 2)
            if args.dc_only:
                particles[:, 0] = 0.0
                particles[:, 1] = 0.0
    else:
        n_dim = 5
        lam = n_particles
        mu = lam // 2
        rank = np.arange(1, mu + 1)
        w = np.log(mu + 0.5) - np.log(rank)
        w /= np.sum(w)
        mueff = (np.sum(w) ** 2) / np.sum(w**2)

        cc = (4 + mueff / n_dim) / (n_dim + 4 + 2 * mueff / n_dim)
        cs = (mueff + 2) / (n_dim + mueff + 5)
        c1 = 2 / ((n_dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n_dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n_dim + 1)) - 1) + cs
        chi_n = np.sqrt(n_dim) * (1 - 1 / (4 * n_dim) + 1 / (21 * n_dim**2))

        mean_u = np.full(n_dim, 0.5, dtype=float)
        sigma_cma = float(args.cmaes_sigma0)
        c = np.eye(n_dim)
        p_c = np.zeros(n_dim)
        p_s = np.zeros(n_dim)
        b = np.eye(n_dim)
        d = np.ones(n_dim)
        invsqrt_c = np.eye(n_dim)
        eigeneval = 0
        counteval = 0

        for gen in tqdm(range(args.n_stages), desc="CMA-ES generations"):
            if anneal_model is not None and args.n_stages > 1:
                frac = gen / float(args.n_stages - 1)
                anneal_model.sigma = args.sigma * (
                    (args.sigma_end / args.sigma) ** frac
                )
                if isinstance(anneal_model, SoftDTWLikelihood):
                    anneal_model.gamma = args.softdtw_gamma * (
                        (args.softdtw_gamma_end / args.softdtw_gamma) ** frac
                    )
                elif isinstance(anneal_model, GSOTLikelihood):
                    anneal_model.epsilon = args.gsot_epsilon * (
                        (args.gsot_epsilon_end / args.gsot_epsilon) ** frac
                    )

            arz = np.random.randn(n_dim, lam)
            ary = b @ (d[:, None] * arz)
            arx = mean_u[:, None] + sigma_cma * ary
            arx = np.clip(arx, UNIT_LOWER[:, None], UNIT_UPPER[:, None])
            particles = decode_unit_to_physical(arx.T)
            if args.dc_only:
                particles[:, 0] = 0.0
                particles[:, 1] = 0.0

            raw_wfs = synthesizer.synthesize_batch(
                particles, m0, source_delay=float(args.source_delay)
            )
            processed_full = extract_phase_windows_batch_cached_dual(
                raw_wfs, phase_cache
            )
            processed = processed_full[:, :, comp_indices, :]
            log_like = likelihood_model.compute_log_likelihood(processed, obs)
            if polarity_cfg is not None and polarity_cfg["weight"] != 0.0:
                log_like = log_like + polarity_loglike_from_particles(
                    particles, polarity_cfg
                )
            if (not args.dc_only) and args.dc_prior_weight != 0.0:
                log_like = log_like + float(args.dc_prior_weight) * dc_log_prior_beta(
                    particles, gamma_beta_prior, delta_beta_prior
                )
            if (
                args.likelihood == "gsot"
                and args.gsot_ratio_weight > 0
                and ratio_compatible
            ):
                log_like = log_like + gsot_ratio_loglike(
                    processed_full,
                    ratio_targets,
                    ratio_valid,
                    sigma=args.gsot_ratio_sigma,
                    weight=args.gsot_ratio_weight,
                )
            if args.l2norm_weight > 0:
                log_like = log_like + l2_normalized_loglike(
                    processed,
                    obs,
                    sigma=args.l2norm_sigma,
                    weight=args.l2norm_weight,
                    aggregation=args.l2norm_aggregation,
                )
            costs = -log_like
            counteval += lam

            order = np.argsort(costs)
            bi = int(order[0])
            if log_like[bi] > best_ll:
                best_ll = float(log_like[bi])
                best_particle = particles[bi].copy()
                best_synthetic = processed[bi].copy()

            sel = order[:mu]
            x_sel = arx.T[sel]
            old_mean = mean_u.copy()
            mean_u = np.sum(x_sel * w[:, None], axis=0)

            y = (mean_u - old_mean) / sigma_cma
            z = invsqrt_c @ y
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * z
            norm_ps = np.linalg.norm(p_s)
            hsig = (
                1.0
                if norm_ps / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))
                < (1.4 + 2 / (n_dim + 1)) * chi_n
                else 0.0
            )

            p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mueff) * y
            y_sel = (x_sel - old_mean[None, :]) / sigma_cma
            rank_mu = np.zeros((n_dim, n_dim), dtype=float)
            for i in range(mu):
                rank_mu += w[i] * np.outer(y_sel[i], y_sel[i])

            c = (
                (1 - c1 - cmu) * c
                + c1 * (np.outer(p_c, p_c) + (1 - hsig) * cc * (2 - cc) * c)
                + cmu * rank_mu
            )
            sigma_cma *= np.exp((cs / damps) * (norm_ps / chi_n - 1))
            sigma_cma = float(np.clip(sigma_cma, 1e-3, 0.5))

            if counteval - eigeneval > lam / (c1 + cmu) / n_dim / 10:
                eigeneval = counteval
                c = np.triu(c) + np.triu(c, 1).T
                eigvals, b = np.linalg.eigh(c)
                eigvals = np.maximum(eigvals, 1e-12)
                d = np.sqrt(eigvals)
                invsqrt_c = b @ np.diag(1.0 / d) @ b.T

            weights = np.exp(log_like - np.max(log_like))
            weights /= np.sum(weights)
            ess = 1.0 / np.sum(weights**2)
            history.append(np.sum(particles * weights[:, None], axis=0))
            last_stage_weights = weights.copy()
            last_stage_particles = particles.copy()
            last_stage_log_weights = log_like.copy()
            print(
                f"Gen {gen + 1:02d}/{args.n_stages}: bestLL={np.max(log_like):.3f}, ESS={ess:.1f}"
            )

    synthesizer.cleanup()

    if (
        last_stage_particles is None
        or last_stage_weights is None
        or last_stage_log_weights is None
    ):
        raise RuntimeError("Inversion failed")
    if best_particle is None or best_synthetic is None:
        raise RuntimeError("No best solution found")

    mean_params = np.sum(last_stage_particles * last_stage_weights[:, None], axis=0)
    map_idx = int(np.argmax(last_stage_log_weights))
    map_particle = last_stage_particles[map_idx].copy()
    map_loglike = float(last_stage_log_weights[map_idx])
    medoid_idx, medoid_obj = weighted_posterior_medoid(
        last_stage_particles, last_stage_weights
    )
    medoid_particle = last_stage_particles[medoid_idx].copy()

    print("\n" + "=" * 54)
    print(f"INVERSION COMPLETE ({args.likelihood.upper()} | {args.sampler.upper()})")
    print("=" * 54)
    print("Posterior weighted mean:")
    print(f"  Gamma: {mean_params[0]:.4f}")
    print(f"  Delta: {mean_params[1]:.4f}")
    print(f"  Kappa: {mean_params[2]:.4f}")
    print(f"  h    : {mean_params[3]:.4f}")
    print(f"  Sigma: {mean_params[4]:.4f}")
    print("\nRepresentative solutions:")
    print(f"  MAP particle index: {map_idx} | logL={map_loglike:.4f}")
    print(
        f"    [gamma, delta, kappa, h, sigma] = [{map_particle[0]:.4f}, {map_particle[1]:.4f}, {map_particle[2]:.4f}, {map_particle[3]:.4f}, {map_particle[4]:.4f}]"
    )
    print(f"  Posterior medoid index: {medoid_idx} | objective={medoid_obj:.4e}")
    print(
        f"    [gamma, delta, kappa, h, sigma] = [{medoid_particle[0]:.4f}, {medoid_particle[1]:.4f}, {medoid_particle[2]:.4f}, {medoid_particle[3]:.4f}, {medoid_particle[4]:.4f}]"
    )

    out_prefix = (
        f"{invdata['event_id']}_{args.likelihood}_{args.sampler}"
        f"_f{float(args.source_target_freq_hz):g}"
    )
    wf_plot = f"waveform_fit_{out_prefix}.png"
    best_for_plot = best_synthetic
    include_mask_plot = None
    if trace_weights is not None:
        include_mask_plot = (
            np.asarray(trace_weights, dtype=np.float64).reshape(
                len(station_ids), len(comp_tokens)
            )
            > 0.0
        )
        n_active_plot = int(np.sum(include_mask_plot))
        n_total_plot = int(include_mask_plot.size)
        if n_active_plot < n_total_plot:
            print(
                f"Plot styling: included traces={n_active_plot}/{n_total_plot}, excluded traces={n_total_plot - n_active_plot}"
            )
    if args.plot_align_lag and int(args.lag_max_shift_samples) > 0:
        best_for_plot, lag_samples, lag_corr = align_synthetics_for_plot(
            obs, best_synthetic, max_shift=int(args.lag_max_shift_samples)
        )
        npts_plot = int(obs.shape[2])
        dt_p = float(args.phase_window_p_len) / float(max(1, npts_plot - 1))
        dt_s = float(args.phase_window_s_len) / float(max(1, npts_plot - 1))
        print(
            "Lag-aligned plotting enabled: "
            f"max_shift=+/-{int(args.lag_max_shift_samples)} samples"
        )
        for j, tok in enumerate(comp_tokens):
            dt = dt_p if tok == "PZ" else dt_s
            med_lag = int(np.median(lag_samples[:, j]))
            med_corr = float(np.median(lag_corr[:, j]))
            print(
                f"  {tok}: median lag={med_lag:+d} samples "
                f"({med_lag * dt:+.4f} s), median corr={med_corr:.3f}"
            )

    plot_obs_vs_best(
        obs,
        best_for_plot,
        station_ids,
        wf_plot,
        args.phase_window_p_len,
        args.phase_window_s_len,
        comp_tokens,
        comp_labels,
        include_mask=include_mask_plot,
        best_label=("Best fit (lag-shifted)" if args.plot_align_lag else "Best fit"),
    )
    print(f"Saved waveform fit plot to {wf_plot}")

    mt_map = Tape_MT33(
        map_particle[0],
        map_particle[1],
        map_particle[2],
        map_particle[3],
        map_particle[4],
    )
    mt_med = Tape_MT33(
        medoid_particle[0],
        medoid_particle[1],
        medoid_particle[2],
        medoid_particle[3],
        medoid_particle[4],
    )
    bb_map = f"bb_map_{out_prefix}.png"
    bb_med = f"bb_medoid_{out_prefix}.png"
    plot_amplitude_beachball(mt_map, bb_map, title="MAP")
    plot_amplitude_beachball(mt_med, bb_med, title="Medoid")
    print(f"Saved beachballs to {bb_map}, {bb_med}")

    ratio_diag = (
        ratio_residual_summary(best_synthetic, ratio_targets, ratio_valid)
        if ratio_compatible
        else None
    )
    if ratio_diag is not None:
        print("Observed ratio residuals (median |delta log-ratio|):")
        print(f"  P/SH: {ratio_diag['P_SH']:.4f}")
        print(f"  P/SV: {ratio_diag['P_SV']:.4f}")
        print(f"  SH/SV: {ratio_diag['SH_SV']:.4f}")

    fit_mse = float(np.mean((obs - best_synthetic) ** 2))
    fit_score = -fit_mse
    print(f"Waveform fit MSE: {fit_mse:.6e}")
    print(f"WAVEFORM_FIT_SCORE={fit_score:.12e}")


if __name__ == "__main__":
    main()
