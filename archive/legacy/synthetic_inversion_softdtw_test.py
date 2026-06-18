"""
Synthetic inversion test with noisy and shifted observations.

LEGACY SCRIPT:
- Active GSOT/Soft-DTW/posterior/beachball helpers have been extracted to
  src package modules.
- Keep this file only as an old synthetic demo/test workflow until it is moved
  to tests/examples or archived.

This mirrors synthetic_inversion_test.py, but replaces the Siamese likelihood
with a normalized Soft-DTW likelihood.
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from tslearn.metrics import soft_dtw as tslearn_soft_dtw
    from tslearn.metrics.soft_dtw_fast import (
        _njit_soft_dtw_batch as tslearn_njit_soft_dtw_batch,
    )
except Exception:
    tslearn_soft_dtw = None
    tslearn_njit_soft_dtw_batch = None


sys.path.append(str(Path(__file__).parent))
SMCMTI_SRC = Path(__file__).parent / "src"
if SMCMTI_SRC.exists():
    sys.path.append(str(SMCMTI_SRC))

try:
    from src.plot.plot_classes import _AmplitudePlot
except Exception:
    try:
        from src.plot.plot_classes import _AmplitudePlot
    except Exception:
        _AmplitudePlot = None

from src.forward import FastSynthesizer, calculate_arrival_time  # noqa: E402
from src.io import (  # noqa: E402
    load_observation,
    load_stations_from_xml,
    load_velocity_model,
)
from src.moment_metrics import kagan_angle_deg  # noqa: E402
from src.tape import MT33_MT6, Tape_MT33  # noqa: E402
from src.waveform_likelihoods import L2Likelihood  # noqa: E402


def add_noise_and_shifts(observation, snr_min=5.0, snr_max=10.0, max_shift=12):
    """
    Add random noise (uniform SNR range) and random phase-window shifts.

    Args:
        observation: (N_stations, 3, T)
    Returns:
        noisy_observation: (N_stations, 3, T)
    """
    nsta, ncomp, npts = observation.shape
    noisy_obs = np.zeros_like(observation)

    for i in range(nsta):
        shift_p = np.random.randint(-max_shift, max_shift + 1)
        noisy_obs[i, 0] = np.roll(observation[i, 0], shift_p)
        if shift_p > 0:
            noisy_obs[i, 0, :shift_p] = 0
        elif shift_p < 0:
            noisy_obs[i, 0, shift_p:] = 0

        shift_s = np.random.randint(-max_shift, max_shift + 1)
        for c in (1, 2):
            noisy_obs[i, c] = np.roll(observation[i, c], shift_s)
            if shift_s > 0:
                noisy_obs[i, c, :shift_s] = 0
            elif shift_s < 0:
                noisy_obs[i, c, shift_s:] = 0

    for i in range(nsta):
        for c in range(ncomp):
            sig = noisy_obs[i, c]
            p_sig = np.mean(sig**2)
            if p_sig < 1e-15:
                continue
            target_snr = np.random.uniform(snr_min, snr_max)
            p_noise = p_sig / (10 ** (target_snr / 10.0))
            noisy_obs[i, c] += np.random.normal(0, np.sqrt(p_noise), size=npts)

    return noisy_obs


def plot_noisy_comparison(
    obs_clean,
    obs_noisy,
    best_synthetic,
    output_path="waveform_comparison_noisy_softdtw.png",
):
    """Plot clean vs noisy target vs best-fit synthetic."""
    nsta, _, npts = obs_clean.shape
    fig, axes = plt.subplots(nsta, 3, figsize=(15, 1.8 * nsta))
    component_names = ["Z (P-wave)", "N (S-wave)", "E (S-wave)"]
    time_axis = np.linspace(0, 0.14, npts)

    for i in range(nsta):
        for c in range(3):
            ax = axes[i, c] if nsta > 1 else axes[c]
            ax.plot(
                time_axis,
                obs_clean[i, c],
                "k-",
                label="Clean",
                linewidth=0.8,
                alpha=0.5,
            )
            ax.plot(
                time_axis,
                obs_noisy[i, c],
                "b-",
                label="Noisy (Target)",
                linewidth=1.0,
                alpha=0.8,
            )
            ax.plot(
                time_axis, best_synthetic[i, c], "r--", label="Best Fit", linewidth=1.5
            )

            if i == 0:
                ax.set_title(component_names[c])
            if c == 0:
                ax.set_ylabel(f"Station {i + 1}")
            if i == 0 and c == 0:
                ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved waveform comparison to {output_path}")
    plt.close()


class SoftDTWLikelihood:
    """
    Normalized Soft-DTW likelihood:
      d_gamma(x, y) = sdtw(x, y) - 0.5*sdtw(x, x) - 0.5*sdtw(y, y)
    and log-likelihood
      ll = -0.5 * mean_trace_cost / sigma^2

    Inputs follow the same shape convention as L2Likelihood:
      synthetics: (B, N, C, T)
      observation: (N, C, T)
    """

    def __init__(
        self,
        sigma=0.05,
        gamma=0.05,
        device="cpu",
        downsample_to=48,
        warp_radius=None,
        l2_weight=0.15,
        corr_weight=0.10,
        amp_weight=0.12,
        aggregation="median",
        normalize_mode="demean",
        backend="torch",
        eps=1e-8,
    ):
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.downsample_to = downsample_to
        self.warp_radius = None if warp_radius is None else int(warp_radius)
        self.l2_weight = float(l2_weight)
        self.corr_weight = float(corr_weight)
        self.amp_weight = float(amp_weight)
        if backend not in ("torch", "tslearn"):
            raise ValueError("backend must be 'torch' or 'tslearn'")
        if backend == "tslearn" and tslearn_soft_dtw is None:
            raise RuntimeError(
                "Soft-DTW backend 'tslearn' requested but tslearn is unavailable."
            )
        self.backend = backend
        if aggregation not in ("mean", "median"):
            raise ValueError("aggregation must be 'mean' or 'median'")
        self.aggregation = aggregation
        if normalize_mode not in ("none", "demean", "zscore"):
            raise ValueError("normalize_mode must be 'none', 'demean', or 'zscore'")
        self.normalize_mode = normalize_mode
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)

        if self.backend == "tslearn" and tslearn_njit_soft_dtw_batch is not None:
            self._warmup_tslearn_njit()

    def _zscore(self, x):
        mu = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(self.eps)
        return (x - mu) / std

    def _soft_dtw_batch(self, x, y):
        """
        Compute Soft-DTW for a batch of 1D traces.
        x, y: (B, T)
        returns: (B,)
        """
        bsz, npts = x.shape
        dmat = (x[:, :, None] - y[:, None, :]) ** 2

        if self.warp_radius is not None:
            ii = torch.arange(npts, device=x.device)
            jj = torch.arange(npts, device=x.device)
            outside = (ii[:, None] - jj[None, :]).abs() > self.warp_radius
            dmat = dmat + outside.to(dmat.dtype)[None, :, :] * 1e6

        r = torch.full(
            (bsz, npts + 1, npts + 1),
            float("inf"),
            device=x.device,
            dtype=x.dtype,
        )
        r[:, 0, 0] = 0.0

        g = self.gamma
        for i in range(1, npts + 1):
            for j in range(1, npts + 1):
                a = r[:, i - 1, j - 1]
                b = r[:, i - 1, j]
                c = r[:, i, j - 1]
                stacked = torch.stack((-a / g, -b / g, -c / g), dim=1)
                softmin = -g * torch.logsumexp(stacked, dim=1)
                r[:, i, j] = dmat[:, i - 1, j - 1] + softmin

        return r[:, npts, npts]

    def _soft_dtw_batch_tslearn(self, x, y):
        """
        Compute Soft-DTW for a batch of 1D traces using tslearn.
        x, y: (B, T) torch tensors
        returns: (B,) torch tensor
        """
        if tslearn_soft_dtw is None:
            raise RuntimeError("tslearn Soft-DTW backend is unavailable")

        x_np = np.asarray(x.detach().cpu().numpy(), dtype=np.float64)
        y_np = np.asarray(y.detach().cpu().numpy(), dtype=np.float64)
        bsz, npts = x_np.shape

        if tslearn_njit_soft_dtw_batch is not None:
            dmat = (x_np[:, :, None] - y_np[:, None, :]) ** 2
            if self.warp_radius is not None:
                ii = np.arange(npts)
                jj = np.arange(npts)
                outside = np.abs(ii[:, None] - jj[None, :]) > self.warp_radius
                dmat += outside[None, :, :] * 1e6
            r = np.empty((bsz, npts + 2, npts + 2), dtype=np.float64)
            tslearn_njit_soft_dtw_batch(dmat, r, float(self.gamma))
            out = r[:, npts, npts].astype(np.float32)
        else:
            out = np.empty(bsz, dtype=np.float32)
            for i in range(bsz):
                out[i] = float(tslearn_soft_dtw(x_np[i], y_np[i], gamma=self.gamma))

        return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)

    def _warmup_tslearn_njit(self):
        if tslearn_njit_soft_dtw_batch is None:
            return
        d = np.zeros((1, 2, 2), dtype=np.float64)
        r = np.empty((1, 4, 4), dtype=np.float64)
        tslearn_njit_soft_dtw_batch(d, r, float(max(self.gamma, 1e-6)))

    def _soft_dtw_dispatch(self, x, y):
        if self.backend == "tslearn":
            return self._soft_dtw_batch_tslearn(x, y)
        return self._soft_dtw_batch(x, y)

    def compute_log_likelihood(self, synthetics, observation):
        if synthetics.ndim == 3:
            synthetics = synthetics[np.newaxis, ...]

        bsz, nsta, ncomp, npts = synthetics.shape
        ntr = nsta * ncomp

        syn = torch.from_numpy(synthetics).float().to(self.device)
        obs = torch.from_numpy(observation).float().to(self.device)

        if self.downsample_to is not None and self.downsample_to < npts:
            target = int(self.downsample_to)
            syn = F.interpolate(
                syn.reshape(bsz * ntr, 1, npts),
                size=target,
                mode="linear",
                align_corners=False,
            ).reshape(bsz, ntr, target)
            obs = F.interpolate(
                obs.reshape(ntr, 1, npts),
                size=target,
                mode="linear",
                align_corners=False,
            ).reshape(ntr, target)
            npts_eff = target
        else:
            syn = syn.reshape(bsz, ntr, npts)
            obs = obs.reshape(ntr, npts)
            npts_eff = npts

        syn_raw_flat = syn.reshape(bsz * ntr, npts_eff)
        obs_raw_rep = obs.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * ntr, npts_eff)
        obs_raw = obs

        if self.normalize_mode == "zscore":
            syn_flat = self._zscore(syn_raw_flat)
            obs_rep = self._zscore(obs_raw_rep)
            obs_norm = self._zscore(obs_raw)
        elif self.normalize_mode == "demean":
            syn_flat = syn_raw_flat - syn_raw_flat.mean(dim=1, keepdim=True)
            obs_rep = obs_raw_rep - obs_raw_rep.mean(dim=1, keepdim=True)
            obs_norm = obs_raw - obs_raw.mean(dim=1, keepdim=True)
        else:
            syn_flat = syn_raw_flat
            obs_rep = obs_raw_rep
            obs_norm = obs_raw

        with torch.no_grad():
            s_xy = self._soft_dtw_dispatch(syn_flat, obs_rep)
            s_xx = self._soft_dtw_dispatch(syn_flat, syn_flat)
            s_yy = self._soft_dtw_dispatch(obs_norm, obs_norm)

            s_yy_rep = s_yy.unsqueeze(0).expand(bsz, -1).reshape(bsz * ntr)
            div = s_xy - 0.5 * s_xx - 0.5 * s_yy_rep
            div = torch.clamp(div, min=0.0)

            mse = ((syn_flat - obs_rep) ** 2).mean(dim=1)
            corr = (syn_flat * obs_rep).mean(dim=1)
            corr_cost = 1.0 - corr
            syn_rms = torch.sqrt((syn_raw_flat**2).mean(dim=1) + self.eps)
            obs_rms = torch.sqrt((obs_raw_rep**2).mean(dim=1) + self.eps)
            amp_cost = torch.abs(torch.log(syn_rms) - torch.log(obs_rms))

            trace_cost = (
                div
                + self.l2_weight * mse
                + self.corr_weight * corr_cost
                + self.amp_weight * amp_cost
            )

            per_particle = trace_cost.reshape(bsz, ntr)
            if self.aggregation == "median":
                cost = torch.median(per_particle, dim=1).values
            else:
                cost = per_particle.mean(dim=1)
            ll = -0.5 * cost / (self.sigma**2)

        return ll.cpu().numpy()


def build_phase_window_cache(
    stations,
    source_loc,
    velocity_model,
    duration,
    npts,
    phase_window_len=0.14,
    source_delay=0.1,
    time_steps=70,
):
    """
    Precompute station-wise phase window indices once for fixed geometry.
    """
    nsta = stations.shape[0]
    dt = duration / float(npts)
    win_samples = int(phase_window_len / dt)
    half_window = phase_window_len / 2.0
    sx_ev, sy_ev, sz_ev = source_loc

    p_idx = np.zeros(nsta, dtype=np.int32)
    s_idx = np.zeros(nsta, dtype=np.int32)
    for i in range(nsta):
        st_x, st_y, st_z = stations[i, 1], stations[i, 2], stations[i, 3]
        dist_m = np.hypot(st_x - sx_ev, st_y - sy_ev)
        t_p = calculate_arrival_time(sz_ev, st_z, dist_m, velocity_model, phase="P")
        t_s = calculate_arrival_time(sz_ev, st_z, dist_m, velocity_model, phase="S")
        p_idx[i] = int(np.floor((t_p + source_delay - half_window) / dt))
        s_idx[i] = int(np.floor((t_s + source_delay - half_window) / dt))

    return {
        "p_idx": p_idx,
        "s_idx": s_idx,
        "npts": int(npts),
        "time_steps": int(time_steps),
        "win_samples": int(win_samples),
        "taper": np.hanning(time_steps).astype(np.float32),
    }


def extract_phase_windows_batch_cached(waveforms, cache):
    """
    Extract phase windows using precomputed station window indices.

    Args:
        waveforms: (B, N, C, T)
    Returns:
        processed: (B, N, C, time_steps)
    """
    bsz, nsta, _, npts = waveforms.shape
    if npts != cache["npts"]:
        raise ValueError(
            f"Waveform length mismatch: got {npts}, expected {cache['npts']} from cache"
        )

    processed = np.zeros((bsz, nsta, 3, cache["time_steps"]), dtype=np.float32)
    win_samples = cache["win_samples"]
    time_steps = cache["time_steps"]

    for i in range(nsta):
        idx_p = int(cache["p_idx"][i])
        idx_s = int(cache["s_idx"][i])

        for c, idx in ((0, idx_p), (1, idx_s), (2, idx_s)):
            start = max(0, idx)
            end = min(npts, idx + win_samples)
            if end <= start:
                continue

            slice_data = waveforms[:, i, c, start:end]
            pad_left = max(0, -idx)
            pad_right = max(0, (idx + win_samples) - npts)
            if pad_left > 0 or pad_right > 0:
                slice_data = np.pad(
                    slice_data,
                    ((0, 0), (pad_left, pad_right)),
                    mode="constant",
                )

            if slice_data.shape[1] != time_steps:
                slice_data = scipy.signal.resample(slice_data, time_steps, axis=1)
            processed[:, i, c, :] = np.asarray(slice_data, dtype=np.float32)

    processed *= cache["taper"][None, None, None, :]
    max_vals = np.max(np.abs(processed), axis=(2, 3), keepdims=True)
    processed = np.divide(
        processed,
        max_vals,
        out=np.zeros_like(processed),
        where=max_vals > 1e-9,
    )
    return processed


class GSOTLikelihood:
    """
    Graph-space OT likelihood using Sinkhorn divergence on per-trace features.

    Each trace is embedded as feature points over time samples:
      [w_t * t, w_a * amp, w_s * d(amp)/dt]
    with mass proportional to |amp| + mass_floor.

    Inputs:
      synthetics: (B, N, C, T)
      observation: (N, C, T)
    """

    def __init__(
        self,
        sigma=0.05,
        epsilon=0.03,
        device="cpu",
        downsample_to=48,
        n_iters=40,
        time_weight=0.30,
        amp_weight=1.00,
        slope_weight=0.35,
        l2_weight=0.08,
        corr_weight=0.06,
        amp_penalty_weight=0.08,
        aggregation="median",
        normalize_mode="demean",
        mass_floor=1e-3,
        eps=1e-8,
    ):
        self.sigma = float(sigma)
        self.epsilon = float(epsilon)
        self.eps = float(eps)
        self.downsample_to = downsample_to
        self.n_iters = int(n_iters)
        self.time_weight = float(time_weight)
        self.amp_weight = float(amp_weight)
        self.slope_weight = float(slope_weight)
        self.l2_weight = float(l2_weight)
        self.corr_weight = float(corr_weight)
        self.amp_penalty_weight = float(amp_penalty_weight)
        self.mass_floor = float(mass_floor)
        if aggregation not in ("mean", "median"):
            raise ValueError("aggregation must be 'mean' or 'median'")
        self.aggregation = aggregation
        if normalize_mode not in ("none", "demean", "zscore"):
            raise ValueError("normalize_mode must be 'none', 'demean', or 'zscore'")
        self.normalize_mode = normalize_mode
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)

    def _zscore(self, x):
        mu = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(self.eps)
        return (x - mu) / std

    def _signal_features_and_mass(self, x):
        """
        x: (B, T)
        returns:
          feat: (B, T, D)
          mass: (B, T)
        """
        bsz, npts = x.shape
        time = torch.linspace(-0.5, 0.5, npts, device=x.device, dtype=x.dtype)
        time = time[None, :].expand(bsz, -1)

        dx = torch.zeros_like(x)
        if npts > 1:
            dx[:, 1:] = x[:, 1:] - x[:, :-1]

        feat = torch.stack(
            (
                self.time_weight * time,
                self.amp_weight * x,
                self.slope_weight * dx,
            ),
            dim=2,
        )

        mass = torch.abs(x) + self.mass_floor
        mass = mass / mass.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return feat, mass

    def _sinkhorn_cost_batch(self, a, b, c):
        """
        Entropic OT cost via Sinkhorn iterations.

        a, b: (B, T) masses
        c: (B, T, T) squared distance cost matrix
        returns: (B,)
        """
        k = torch.exp(-c / self.epsilon).clamp_min(self.eps)
        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(self.n_iters):
            kv = torch.bmm(k, v.unsqueeze(2)).squeeze(2).clamp_min(self.eps)
            u = a / kv
            ktu = (
                torch.bmm(k.transpose(1, 2), u.unsqueeze(2))
                .squeeze(2)
                .clamp_min(self.eps)
            )
            v = b / ktu

        transport = u.unsqueeze(2) * k * v.unsqueeze(1)
        return (transport * c).sum(dim=(1, 2))

    def compute_per_trace_cost(self, synthetics, observation):
        if synthetics.ndim == 3:
            synthetics = synthetics[np.newaxis, ...]

        bsz, nsta, ncomp, npts = synthetics.shape
        ntr = nsta * ncomp

        syn = torch.from_numpy(synthetics).float().to(self.device)
        obs = torch.from_numpy(observation).float().to(self.device)

        if self.downsample_to is not None and self.downsample_to < npts:
            target = int(self.downsample_to)
            syn = F.interpolate(
                syn.reshape(bsz * ntr, 1, npts),
                size=target,
                mode="linear",
                align_corners=False,
            ).reshape(bsz, ntr, target)
            obs = F.interpolate(
                obs.reshape(ntr, 1, npts),
                size=target,
                mode="linear",
                align_corners=False,
            ).reshape(ntr, target)
            npts_eff = target
        else:
            syn = syn.reshape(bsz, ntr, npts)
            obs = obs.reshape(ntr, npts)
            npts_eff = npts

        syn_raw_flat = syn.reshape(bsz * ntr, npts_eff)
        obs_raw_rep = obs.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * ntr, npts_eff)
        obs_raw = obs

        if self.normalize_mode == "zscore":
            syn_flat = self._zscore(syn_raw_flat)
            obs_rep = self._zscore(obs_raw_rep)
            obs_norm = self._zscore(obs_raw)
        elif self.normalize_mode == "demean":
            syn_flat = syn_raw_flat - syn_raw_flat.mean(dim=1, keepdim=True)
            obs_rep = obs_raw_rep - obs_raw_rep.mean(dim=1, keepdim=True)
            obs_norm = obs_raw - obs_raw.mean(dim=1, keepdim=True)
        else:
            syn_flat = syn_raw_flat
            obs_rep = obs_raw_rep
            obs_norm = obs_raw

        with torch.no_grad():
            syn_feat, syn_mass = self._signal_features_and_mass(syn_flat)
            obs_feat_rep, obs_mass_rep = self._signal_features_and_mass(obs_rep)
            obs_feat, obs_mass = self._signal_features_and_mass(obs_norm)

            c_xy = torch.cdist(syn_feat, obs_feat_rep, p=2) ** 2
            c_xx = torch.cdist(syn_feat, syn_feat, p=2) ** 2
            c_yy = torch.cdist(obs_feat, obs_feat, p=2) ** 2

            ot_xy = self._sinkhorn_cost_batch(syn_mass, obs_mass_rep, c_xy)
            ot_xx = self._sinkhorn_cost_batch(syn_mass, syn_mass, c_xx)
            ot_yy = self._sinkhorn_cost_batch(obs_mass, obs_mass, c_yy)

            ot_yy_rep = ot_yy.unsqueeze(0).expand(bsz, -1).reshape(bsz * ntr)
            div = torch.clamp(ot_xy - 0.5 * ot_xx - 0.5 * ot_yy_rep, min=0.0)

            mse = ((syn_flat - obs_rep) ** 2).mean(dim=1)
            corr = (syn_flat * obs_rep).mean(dim=1)
            corr_cost = 1.0 - corr
            syn_rms = torch.sqrt((syn_raw_flat**2).mean(dim=1) + self.eps)
            obs_rms = torch.sqrt((obs_raw_rep**2).mean(dim=1) + self.eps)
            amp_cost = torch.abs(torch.log(syn_rms) - torch.log(obs_rms))

            trace_cost = (
                div
                + self.l2_weight * mse
                + self.corr_weight * corr_cost
                + self.amp_penalty_weight * amp_cost
            )

            per_particle = trace_cost.reshape(bsz, ntr)

        return per_particle.cpu().numpy()

    def compute_log_likelihood(self, synthetics, observation, trace_weights=None):
        per_particle = self.compute_per_trace_cost(synthetics, observation)

        if trace_weights is None:
            if self.aggregation == "median":
                cost = np.median(per_particle, axis=1)
            else:
                cost = np.mean(per_particle, axis=1)
        else:
            w = np.asarray(trace_weights, dtype=np.float64).reshape(-1)
            if w.shape[0] != per_particle.shape[1]:
                raise ValueError(
                    "trace_weights length mismatch: "
                    f"got {w.shape[0]}, expected {per_particle.shape[1]}"
                )
            w = np.clip(w, 0.0, None)
            wsum = float(np.sum(w))
            if wsum <= self.eps:
                raise ValueError("trace_weights must contain positive values")
            cost = np.sum(per_particle * w[None, :], axis=1) / wsum

        ll = -0.5 * cost / (self.sigma**2)
        return np.asarray(ll, dtype=np.float64)


def _tpb2q(t, p, b):
    eps = 0.001
    tqw = 1.0 + t[0] + p[1] + b[2]
    tqx = 1.0 + t[0] - p[1] - b[2]
    tqy = 1.0 - t[0] + p[1] - b[2]
    tqz = 1.0 - t[0] - p[1] + b[2]

    q = np.zeros(4, dtype=float)
    if tqw > eps:
        q[0] = 0.5 * math.sqrt(tqw)
        q[1:] = (p[2] - b[1], b[0] - t[2], t[1] - p[0])
    elif tqx > eps:
        q[0] = 0.5 * math.sqrt(tqx)
        q[1:] = (p[2] - b[1], p[0] + t[1], b[0] + t[2])
    elif tqy > eps:
        q[0] = 0.5 * math.sqrt(tqy)
        q[1:] = (b[0] - t[2], p[0] + t[1], b[1] + p[2])
    elif tqz > eps:
        q[0] = 0.5 * math.sqrt(tqz)
        q[1:] = (t[1] - p[0], b[0] + t[2], b[1] + p[2])
    else:
        raise RuntimeError("Kagan angle computation failed: invalid quaternion.")

    q[1:] /= 4.0 * q[0]
    q /= math.sqrt(float(np.sum(q * q)))
    return q


def kagan_angle_deg_local(mt1, mt2):
    pbt2tpb = np.array(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    ai = pbt2tpb @ mt1.T
    aj = pbt2tpb @ mt2.T
    u = ai @ aj.T
    tk, pk, bk = u.tolist()
    qk = _tpb2q(tk, pk, bk)
    return float(2.0 * (180.0 / np.pi) * math.acos(np.max(np.abs(qk))))


def weighted_posterior_medoid(particles, weights):
    """
    Weighted medoid in MT6 space.

    The medoid is the particle i minimizing sum_j w_j * d(i, j), where
    d(i,j) = 1 - dot(mt6_i, mt6_j).
    """
    n = particles.shape[0]
    w = np.asarray(weights, dtype=float)
    w /= np.sum(w)

    mt6 = np.zeros((n, 6), dtype=float)
    for i in range(n):
        g, d, k, h, s = particles[i]
        mt33 = Tape_MT33(g, d, k, h, s)
        v = MT33_MT6(mt33)
        nv = np.linalg.norm(v)
        if nv > 0:
            v = v / nv
        mt6[i] = v

    sim = np.clip(mt6 @ mt6.T, -1.0, 1.0)
    dist = 1.0 - sim
    objective = dist @ w
    idx = int(np.argmin(objective))
    return idx, float(objective[idx])


UNIT_LOWER = np.zeros(5, dtype=float)
UNIT_UPPER = np.ones(5, dtype=float)
PHYS_LOWER = np.array([-np.pi / 6, -np.pi / 2, 0.0, 0.0, -np.pi / 2], dtype=float)
PHYS_UPPER = np.array([np.pi / 6, np.pi / 2, 2 * np.pi, 1.0, np.pi / 2], dtype=float)


def decode_unit_to_physical(unit_particles):
    p = PHYS_LOWER[None, :] + unit_particles * (PHYS_UPPER - PHYS_LOWER)[None, :]
    p[:, 2] = np.mod(p[:, 2], 2 * np.pi)
    p[:, 0] = np.clip(p[:, 0], PHYS_LOWER[0], PHYS_UPPER[0])
    p[:, 1] = np.clip(p[:, 1], PHYS_LOWER[1], PHYS_UPPER[1])
    p[:, 3] = np.clip(p[:, 3], PHYS_LOWER[3], PHYS_UPPER[3])
    p[:, 4] = np.clip(p[:, 4], PHYS_LOWER[4], PHYS_UPPER[4])
    return p


def plot_amplitude_beachball(mt33, output_path, title=None):
    """Plot a single MT beachball using the CAPE-style _AmplitudePlot."""
    if _AmplitudePlot is None:
        print("Warning: _AmplitudePlot unavailable; skipping beachball plot.")
        return

    mt6 = MT33_MT6(mt33).reshape(6, 1)
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    amp_plot = _AmplitudePlot(
        None,
        fig,
        mt6,
        phase="P",
        projection="equalarea",
        lower=True,
        full_sphere=False,
        colormap="bwr",
        axis_lines=False,
        fault_plane=True,
        nodal_line=False,
        TNP=False,
        text=False,
        show=False,
        resolution=200,
    )
    amp_plot.plot()
    ax = amp_plot.ax
    ax.set_axis_off()
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default="data_phase.h5")
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--stations-dir", type=str, default="stationxml")
    parser.add_argument("--velocity-model", type=str, default="forge.tvel")
    parser.add_argument(
        "--sampler", type=str, choices=["smc", "cmaes"], default="cmaes"
    )
    parser.add_argument("--n-particles", type=int, default=1000)
    parser.add_argument("--n-stages", type=int, default=10)
    parser.add_argument("--cmaes-sigma0", type=float, default=0.22)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--likelihood", type=str, choices=["softdtw", "gsot", "l2"], default="softdtw"
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
        "--softdtw-aggregation",
        type=str,
        choices=["mean", "median"],
        default="mean",
    )
    parser.add_argument("--gsot-epsilon", type=float, default=0.03)
    parser.add_argument("--gsot-epsilon-end", type=float, default=0.015)
    parser.add_argument("--gsot-downsample", type=int, default=48)
    parser.add_argument("--gsot-iters", type=int, default=40)
    parser.add_argument("--gsot-time-weight", type=float, default=0.30)
    parser.add_argument("--gsot-amp-weight", type=float, default=1.00)
    parser.add_argument("--gsot-slope-weight", type=float, default=0.35)
    parser.add_argument("--gsot-l2-weight", type=float, default=0.08)
    parser.add_argument("--gsot-corr-weight", type=float, default=0.06)
    parser.add_argument("--gsot-amp-penalty-weight", type=float, default=0.08)
    parser.add_argument(
        "--gsot-normalize",
        type=str,
        choices=["none", "demean", "zscore"],
        default="demean",
    )
    parser.add_argument(
        "--gsot-aggregation",
        type=str,
        choices=["mean", "median"],
        default="mean",
    )
    parser.add_argument("--sigma-end", type=float, default=0.02)
    parser.add_argument("--snr-min", type=float, default=10.0)
    parser.add_argument("--snr-max", type=float, default=15.0)
    parser.add_argument("--max-shift", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading Observation #{args.event_index} from {args.data_file}...")
    gt_params, obs_clean = load_observation(args.data_file, args.event_index)

    print(
        f"\nApplying noise SNR [{args.snr_min}, {args.snr_max}] dB "
        f"and random shifts +/- {args.max_shift} samples..."
    )
    obs_noisy = add_noise_and_shifts(
        obs_clean,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        max_shift=args.max_shift,
    )

    stations, _, _, _ = load_stations_from_xml(args.stations_dir)
    velocity_model = load_velocity_model(args.velocity_model)
    source_loc = (gt_params["x"], gt_params["y"], gt_params["z"])
    m0 = 10 ** (1.5 * gt_params["mw"] + 9.1)

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
        print(
            "Using Soft-DTW likelihood "
            f"(gamma={args.softdtw_gamma}->{args.softdtw_gamma_end}, "
            f"sigma={args.sigma}->{args.sigma_end}, downsample={args.softdtw_downsample}, "
            f"backend={args.softdtw_backend}, "
            f"warp_radius={args.softdtw_warp_radius}, l2_w={args.softdtw_l2_weight}, "
            f"corr_w={args.softdtw_corr_weight}, amp_w={args.softdtw_amp_weight}, "
            f"norm={args.softdtw_normalize}, agg={args.softdtw_aggregation})"
        )
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
        print(
            "Using GSOT likelihood "
            f"(epsilon={args.gsot_epsilon}->{args.gsot_epsilon_end}, "
            f"sigma={args.sigma}->{args.sigma_end}, downsample={args.gsot_downsample}, "
            f"iters={args.gsot_iters}, tw={args.gsot_time_weight}, aw={args.gsot_amp_weight}, "
            f"sw={args.gsot_slope_weight}, l2_w={args.gsot_l2_weight}, "
            f"corr_w={args.gsot_corr_weight}, amp_w={args.gsot_amp_penalty_weight}, "
            f"norm={args.gsot_normalize}, agg={args.gsot_aggregation})"
        )
    else:
        likelihood_model = L2Likelihood(sigma=args.sigma)
        print(f"Using L2 likelihood (sigma={args.sigma})")

    anneal_model = None
    if isinstance(likelihood_model, SoftDTWLikelihood):
        anneal_model = likelihood_model
    elif isinstance(likelihood_model, GSOTLikelihood):
        anneal_model = likelihood_model

    synthesizer = FastSynthesizer(velocity_model, stations, source_loc)
    synthesizer.setup()
    if synthesizer.ap is None:
        raise RuntimeError(
            "Synthesizer setup failed: Green's functions are unavailable."
        )
    phase_cache = build_phase_window_cache(
        stations=stations,
        source_loc=source_loc,
        velocity_model=velocity_model,
        duration=1.0,
        npts=int(synthesizer.ap.npt),
        phase_window_len=0.14,
        source_delay=0.1,
        time_steps=70,
    )

    print(f"\nStarting {args.sampler.upper()} inversion...")
    n_particles = args.n_particles

    best_ll = -np.inf
    best_synthetic = None
    best_particle = None
    history = []
    last_stage_weights = None
    last_stage_particles = None
    last_stage_log_weights = None

    if args.sampler == "smc":
        particles = np.zeros((n_particles, 5))
        particles[:, 0] = np.random.uniform(-np.pi / 6, np.pi / 6, n_particles)
        particles[:, 1] = np.random.uniform(-np.pi / 2, np.pi / 2, n_particles)
        particles[:, 2] = np.random.uniform(0, 2 * np.pi, n_particles)
        particles[:, 3] = np.random.uniform(0, 1, n_particles)
        particles[:, 4] = np.random.uniform(-np.pi / 2, np.pi / 2, n_particles)

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

            raw_wfs = synthesizer.synthesize_batch(particles, m0)
            processed = extract_phase_windows_batch_cached(raw_wfs, phase_cache)
            log_weights = likelihood_model.compute_log_likelihood(processed, obs_noisy)

            best_idx = int(np.argmax(log_weights))
            if log_weights[best_idx] > best_ll:
                best_ll = float(log_weights[best_idx])
                best_particle = particles[best_idx].copy()
                best_synthetic = processed[best_idx].copy()

            weights = np.exp(log_weights - np.max(log_weights))
            weights /= np.sum(weights)
            ess = 1.0 / np.sum(weights**2)

            weighted_mean = np.sum(particles * weights[:, None], axis=0)
            history.append(weighted_mean)

            last_stage_weights = weights.copy()
            last_stage_particles = particles.copy()
            last_stage_log_weights = log_weights.copy()

            print(
                f"Stage {stage + 1:02d}/{args.n_stages}: "
                f"maxLL={np.max(log_weights):.3f}, minLL={np.min(log_weights):.3f}, ESS={ess:.1f}"
            )

            indices = np.random.choice(n_particles, size=n_particles, p=weights)
            particles = particles[indices]

            step_size = 0.1 * (0.9**stage)
            particles += np.random.normal(0, step_size, size=particles.shape)
            particles[:, 0] = np.clip(particles[:, 0], -np.pi / 6, np.pi / 6)
            particles[:, 1] = np.clip(particles[:, 1], -np.pi / 2, np.pi / 2)
            particles[:, 2] = np.mod(particles[:, 2], 2 * np.pi)
            particles[:, 3] = np.clip(particles[:, 3], 0, 1)
            particles[:, 4] = np.clip(particles[:, 4], -np.pi / 2, np.pi / 2)
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
        cmu = min(
            1 - c1,
            2 * (mueff - 2 + 1 / mueff) / ((n_dim + 2) ** 2 + mueff),
        )
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
            unit_pop = arx.T
            particles = decode_unit_to_physical(unit_pop)

            raw_wfs = synthesizer.synthesize_batch(particles, m0)
            processed = extract_phase_windows_batch_cached(raw_wfs, phase_cache)
            log_like = likelihood_model.compute_log_likelihood(processed, obs_noisy)
            costs = -log_like
            counteval += lam

            order = np.argsort(costs)
            best_idx = int(order[0])
            if log_like[best_idx] > best_ll:
                best_ll = float(log_like[best_idx])
                best_particle = particles[best_idx].copy()
                best_synthetic = processed[best_idx].copy()

            sel = order[:mu]
            x_sel = unit_pop[sel]
            old_mean = mean_u.copy()
            mean_u = np.sum(x_sel * w[:, None], axis=0)

            y = (mean_u - old_mean) / sigma_cma
            z = invsqrt_c @ y
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * z
            norm_ps = np.linalg.norm(p_s)
            hsig_cond = norm_ps / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))
            hsig = 1.0 if hsig_cond < (1.4 + 2 / (n_dim + 1)) * chi_n else 0.0

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
                f"Gen {gen + 1:02d}/{args.n_stages}: "
                f"bestLL={np.max(log_like):.3f}, meanLL={np.mean(log_like):.3f}, "
                f"ESS={ess:.1f}, sigma_cma={sigma_cma:.4f}"
            )

    synthesizer.cleanup()

    if (
        last_stage_particles is None
        or last_stage_weights is None
        or last_stage_log_weights is None
    ):
        raise RuntimeError("Inversion failed: missing last-stage posterior state.")
    assert last_stage_particles is not None
    assert last_stage_weights is not None
    assert last_stage_log_weights is not None

    mean_params = np.sum(last_stage_particles * last_stage_weights[:, None], axis=0)
    map_idx = int(np.argmax(last_stage_log_weights))
    map_particle = last_stage_particles[map_idx].copy()
    map_loglike = float(last_stage_log_weights[map_idx])

    medoid_idx, medoid_obj = weighted_posterior_medoid(
        last_stage_particles, last_stage_weights
    )
    medoid_particle = last_stage_particles[medoid_idx].copy()

    print("\n" + "=" * 50)
    print(
        f"NOISY INVERSION COMPLETE ({args.likelihood.upper()} | {args.sampler.upper()})"
    )
    print("=" * 50)
    print("Posterior (last stage) weighted mean:")
    print(f"  Gamma: {mean_params[0]:.4f}  vs True: {gt_params['gamma']:.4f}")
    print(f"  Delta: {mean_params[1]:.4f}  vs True: {gt_params['delta']:.4f}")
    print(f"  Kappa: {mean_params[2]:.4f}  vs True: {gt_params['kappa']:.4f}")
    print(f"  h    : {mean_params[3]:.4f}  vs True: {gt_params['h']:.4f}")
    print(f"  Sigma: {mean_params[4]:.4f}  vs True: {gt_params['sigma']:.4f}")

    print("\nRepresentative solutions:")
    print(f"  MAP particle index (last stage): {map_idx} | logL={map_loglike:.4f}")
    print(
        f"    [gamma, delta, kappa, h, sigma] = "
        f"[{map_particle[0]:.4f}, {map_particle[1]:.4f}, {map_particle[2]:.4f}, {map_particle[3]:.4f}, {map_particle[4]:.4f}]"
    )
    print(
        f"  Posterior medoid index: {medoid_idx} | objective={medoid_obj:.4e}\n"
        f"    [gamma, delta, kappa, h, sigma] = "
        f"[{medoid_particle[0]:.4f}, {medoid_particle[1]:.4f}, {medoid_particle[2]:.4f}, {medoid_particle[3]:.4f}, {medoid_particle[4]:.4f}]"
    )

    if best_particle is None or best_synthetic is None:
        raise RuntimeError("Inversion failed: no valid best particle was found.")
    assert best_particle is not None
    assert best_synthetic is not None

    mt_true = None
    mt_best = None
    mt_map = None
    mt_medoid = None
    mt_mean = None

    try:
        mt_true = Tape_MT33(
            gt_params["gamma"],
            gt_params["delta"],
            gt_params["kappa"],
            gt_params["h"],
            gt_params["sigma"],
        )
        mt_best = Tape_MT33(
            best_particle[0],
            best_particle[1],
            best_particle[2],
            best_particle[3],
            best_particle[4],
        )
        mt_map = Tape_MT33(
            map_particle[0],
            map_particle[1],
            map_particle[2],
            map_particle[3],
            map_particle[4],
        )
        mt_medoid = Tape_MT33(
            medoid_particle[0],
            medoid_particle[1],
            medoid_particle[2],
            medoid_particle[3],
            medoid_particle[4],
        )
        mt_mean = Tape_MT33(
            mean_params[0],
            mean_params[1],
            mean_params[2],
            mean_params[3],
            mean_params[4],
        )

        k_best = kagan_angle_deg(mt_true, mt_best)
        k_map = kagan_angle_deg(mt_true, mt_map)
        k_medoid = kagan_angle_deg(mt_true, mt_medoid)
        k_mean = kagan_angle_deg(mt_true, mt_mean)

        print("\nMT-space similarity (lower Kagan is better):")
        print(f"  Kagan(best-global,true): {k_best:.2f} deg")
        print(f"  Kagan(MAP,true): {k_map:.2f} deg")
        print(f"  Kagan(medoid,true): {k_medoid:.2f} deg")
        print(f"  Kagan(mean,true): {k_mean:.2f} deg")
    except Exception:
        try:
            if mt_true is not None and mt_best is not None and mt_mean is not None:
                k_best = kagan_angle_deg_local(mt_true, mt_best)
                k_mean = kagan_angle_deg_local(mt_true, mt_mean)
                print("\nMT-space similarity (lower Kagan is better):")
                print(f"  Kagan(best,true): {k_best:.2f} deg")
                print(f"  Kagan(mean,true): {k_mean:.2f} deg")
        except Exception:
            pass

    plot_noisy_comparison(
        obs_clean,
        obs_noisy,
        best_synthetic,
        output_path=f"noisy_inversion_result_{args.likelihood}_{args.sampler}.png",
    )

    if mt_true is not None and mt_map is not None and mt_medoid is not None:
        bb_true = f"bb_true_{args.likelihood}_{args.sampler}.png"
        bb_map = f"bb_map_{args.likelihood}_{args.sampler}.png"
        bb_medoid = f"bb_medoid_{args.likelihood}_{args.sampler}.png"
        plot_amplitude_beachball(mt_true, bb_true, title="True")
        plot_amplitude_beachball(mt_map, bb_map, title="MAP")
        plot_amplitude_beachball(mt_medoid, bb_medoid, title="Medoid")
        print(f"Saved beachball plots to {bb_true}, {bb_map}, {bb_medoid}")

    hist_arr = np.array(history)
    fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
    labels = ["Gamma", "Delta", "Kappa", "h", "Sigma"]
    true_vals = [
        gt_params["gamma"],
        gt_params["delta"],
        gt_params["kappa"],
        gt_params["h"],
        gt_params["sigma"],
    ]
    for i in range(5):
        axes[i].plot(hist_arr[:, i], label="Inversion Mean")
        axes[i].axhline(true_vals[i], color="r", linestyle="--", label="Ground Truth")
        axes[i].set_ylabel(labels[i])
        axes[i].legend()
    plt.savefig(f"noisy_convergence_{args.likelihood}_{args.sampler}.png")
    print(
        f"Saved parameter convergence to noisy_convergence_{args.likelihood}_{args.sampler}.png"
    )


if __name__ == "__main__":
    main()
