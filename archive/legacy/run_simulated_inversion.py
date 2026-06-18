"""
Simulated Inversion using Pre-trained Siamese Model + Adaptive SMC.

LEGACY SCRIPT:
- Active reusable forward-model utilities have been extracted to src.
- Keep this file only for the old standalone synthetic/Siamese workflow until
  that workflow is archived or removed.

DEBUG VERSION:
- Option for L2 (least-squares) likelihood for comparison
- Plots best-fit waveform vs observation
"""

import numpy as np
import h5py
import torch
import argparse
import json
import sys
import os
import tempfile
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import scipy.signal
import math
from types import SimpleNamespace
from typing import List, Tuple, Union
import glob
import obspy
import pyproj

AXITRA_SRC = Path(__file__).resolve().parents[1] / "axitra" / "MOMENT_DISP_F90_OPENMP" / "src"
if str(AXITRA_SRC) not in sys.path:
    sys.path.append(str(AXITRA_SRC))

try:
    from axitra import Axitra, moment
except Exception:
    Axitra = None
    moment = None

from src.tape import Tape_MT6, Tape_MT33, MT33_MT6


def _station_id_from_xml(xml_file: str, net_code: str, sta_code: str, cha) -> str:
    stem = Path(xml_file).stem
    parts = stem.split(".")
    if len(parts) >= 4 and parts[0] == net_code and parts[1] == sta_code:
        return ".".join(parts[:4])

    loc = getattr(cha, "location_code", "")
    chan = getattr(cha, "code", "")
    band = chan[:2] if len(chan) >= 2 else chan
    return f"{net_code}.{sta_code}.{loc}.{band}"


def load_stations_from_xml(
    station_xml_dir: Union[str, Path],
) -> Tuple[np.ndarray, List[str], float, float]:
    """
    Parse stationxml files and return:
      - stations array [idx, x_m, y_m, depth_m]
      - station ids corresponding to rows
      - centroid x, centroid y
    """
    station_xml_dir = Path(station_xml_dir)
    easting_ref = 335697.57
    northing_ref = 4263390.77
    transformer = pyproj.Proj("EPSG:26912", preserve_units=False)

    xml_files = sorted(glob.glob(str(station_xml_dir / "*.xml")))
    stations = []
    codes = []
    processed = set()
    idx = 1
    for xml_file in xml_files:
        try:
            inv = obspy.read_inventory(xml_file)
        except Exception:
            continue
        for net in inv:
            for sta in net:
                if len(sta) == 0:
                    continue
                cha = sta[0]
                station_id = _station_id_from_xml(xml_file, net.code, sta.code, cha)
                if station_id in processed:
                    continue
                try:
                    easting, northing = transformer(cha.longitude, cha.latitude)
                except Exception:
                    continue
                stations.append(
                    [
                        idx,
                        float(easting - easting_ref),
                        float(northing - northing_ref),
                        float(cha.depth),
                    ]
                )
                codes.append(station_id)
                processed.add(station_id)
                idx += 1

    if not stations:
        raise ValueError(f"No stations found in {station_xml_dir}")

    arr = np.array(stations, dtype=float)
    cx, cy = float(arr[:, 1].mean()), float(arr[:, 2].mean())
    return arr, codes, cx, cy


def load_velocity_model(
    path: Union[str, Path],
    max_depth_m: float = 3000.0,
    fict_layer_dz: float = 500.0,
    target_depth: float = 3000.0,
) -> np.ndarray:
    """
    Load a layered forge.tvel-like model as rows:
    [thickness_m, Vp, Vs, rho, Qp, Qs].
    """
    path = Path(path)
    boundaries_m = []
    vp = []
    vs = []
    rho = []
    default_qp = 600.0
    default_qs = 300.0

    with path.open() as f:
        for line in f:
            if not line.strip() or line[0].isalpha():
                continue
            vals = line.split()
            if len(vals) < 4:
                continue
            depth_m = float(vals[0]) * 1000.0
            if depth_m > max_depth_m:
                break
            boundaries_m.append(depth_m)
            vp.append(float(vals[1]) * 1000.0)
            vs.append(float(vals[2]) * 1000.0)
            rho.append(float(vals[3]) * 1000.0)

    if not boundaries_m:
        raise ValueError(f"No layers parsed from {path}")

    cleaned = []
    for i, d in enumerate(boundaries_m):
        if i > 0 and math.isclose(d, cleaned[-1][0], rel_tol=0.0, abs_tol=1e-6):
            continue
        cleaned.append([d, vp[i], vs[i], rho[i]])

    layers = []
    for i, (top_m, vp_m, vs_m, rho_m) in enumerate(cleaned):
        thick_m = max(cleaned[i + 1][0] - top_m, 0.0) if i < len(cleaned) - 1 else 0.0
        layers.append([thick_m, vp_m, vs_m, rho_m, default_qp, default_qs])

    layers_arr = np.array(layers, dtype=float)
    total_depth = float(layers_arr[:-1, 0].sum()) if len(layers_arr) > 1 else 0.0
    target_depth = max(total_depth, target_depth)

    if total_depth < target_depth:
        vp_last, vs_last, rho_last, qp_last, qs_last = layers_arr[-1, 1:]
        extra_layers = []
        current_depth = total_depth
        while current_depth < target_depth:
            dz = min(fict_layer_dz, target_depth - current_depth)
            extra_layers.append([dz, vp_last, vs_last, rho_last, qp_last, qs_last])
            current_depth += dz
        layers_arr = np.vstack(
            [layers_arr[:-1], np.array(extra_layers, dtype=float), layers_arr[-1:]]
        )

    return layers_arr


def calculate_arrival_time(
    src_depth_m: float,
    rcv_depth_m: float,
    dist_m: float,
    model: np.ndarray,
    phase: str = "P",
) -> float:
    """Approximate first-arrival time for a ray in a 1D layered model."""
    if src_depth_m < rcv_depth_m:
        src_depth_m, rcv_depth_m = rcv_depth_m, src_depth_m

    current_depth = 0.0
    active_layers = []
    v_idx = 1 if phase.upper() == "P" else 2

    for row in model:
        thick = float(row[0])
        vel = float(row[v_idx])
        layer_top = current_depth
        layer_bot = current_depth + thick
        overlap_top = max(layer_top, rcv_depth_m)
        overlap_bot = min(layer_bot, src_depth_m)
        if overlap_bot > overlap_top:
            active_layers.append((overlap_bot - overlap_top, vel))
        current_depth += thick
        if current_depth >= src_depth_m:
            break

    if current_depth < src_depth_m:
        h = src_depth_m - max(current_depth, rcv_depth_m)
        if h > 0:
            active_layers.append((h, float(model[-1, v_idx])))

    if not active_layers:
        vel = float(model[0, v_idx])
        return dist_m / max(vel, 1e-6)

    v_max = max(v for _, v in active_layers)
    p_limit = 1.0 / v_max

    def horizontal_distance(p: float) -> float:
        x = 0.0
        for h, v in active_layers:
            if p * v >= 1.0:
                return float("inf")
            tan_theta = (p * v) / np.sqrt(1.0 - (p * v) ** 2)
            x += h * tan_theta
        return x

    p_min = 0.0
    p_max = p_limit * 0.999999
    for _ in range(50):
        p_mid = 0.5 * (p_min + p_max)
        if horizontal_distance(p_mid) < dist_m:
            p_min = p_mid
        else:
            p_max = p_mid
    p_sol = 0.5 * (p_min + p_max)

    t_travel = 0.0
    for h, v in active_layers:
        cos_theta = np.sqrt(1.0 - (p_sol * v) ** 2)
        t_travel += h / (v * cos_theta)

    return float(t_travel)


def load_observation(h5_path, index):
    """Load Ground Truth Observation from HDF5."""
    with h5py.File(h5_path, "r") as f:
        gt_params = {}
        for key in f["params"].keys():
            gt_params[key] = f["params"][key][index]
        observation = f["waveforms"][index]
    return gt_params, observation


def extract_phase_windows_batch(
    waveforms: np.ndarray,
    stations: np.ndarray,
    source_loc: tuple,
    velocity_model: np.ndarray,
    duration: float,
    phase_window_len: float = 0.14,
    source_delay: float = 0.1,
    time_steps: int = 70,
) -> np.ndarray:
    """Vectorized phase window extraction for a batch of waveforms."""
    B, N, C, T = waveforms.shape
    dt = duration / float(T)
    processed = np.zeros((B, N, C, time_steps), dtype=waveforms.dtype)

    win_samples = int(phase_window_len / dt)
    half_window = phase_window_len / 2.0

    sx_ev, sy_ev, sz_ev = source_loc

    # Pre-compute window indices for each station
    p_indices = []
    s_indices = []

    for i in range(N):
        st_x, st_y, st_z = stations[i, 1], stations[i, 2], stations[i, 3]
        dist_m = np.hypot(st_x - sx_ev, st_y - sy_ev)

        t_p = calculate_arrival_time(sz_ev, st_z, dist_m, velocity_model, phase="P")
        t_s = calculate_arrival_time(sz_ev, st_z, dist_m, velocity_model, phase="S")

        t_p += source_delay
        t_s += source_delay

        idx_p = int(np.floor((t_p - half_window) / dt))
        idx_s = int(np.floor((t_s - half_window) / dt))

        p_indices.append(idx_p)
        s_indices.append(idx_s)

    for i in range(N):
        idx_p = p_indices[i]
        idx_s = s_indices[i]

        # P Window (Channel 0)
        start = max(0, idx_p)
        end = min(T, idx_p + win_samples)
        pad_left = max(0, -idx_p)
        pad_right = max(0, (idx_p + win_samples) - T)

        if end > start:
            slice_data = waveforms[:, i, 0, start:end]
            if pad_left > 0 or pad_right > 0:
                slice_data = np.pad(
                    slice_data, ((0, 0), (pad_left, pad_right)), mode="constant"
                )
            if slice_data.shape[1] != time_steps:
                for b in range(B):
                    processed[b, i, 0] = scipy.signal.resample(
                        slice_data[b], time_steps
                    )
            else:
                processed[:, i, 0] = slice_data

        # S Window (Channels 1, 2)
        start = max(0, idx_s)
        end = min(T, idx_s + win_samples)
        pad_left = max(0, -idx_s)
        pad_right = max(0, (idx_s + win_samples) - T)

        for c in [1, 2]:
            if end > start:
                slice_data = waveforms[:, i, c, start:end]
                if pad_left > 0 or pad_right > 0:
                    slice_data = np.pad(
                        slice_data, ((0, 0), (pad_left, pad_right)), mode="constant"
                    )
                if slice_data.shape[1] != time_steps:
                    for b in range(B):
                        processed[b, i, c] = scipy.signal.resample(
                            slice_data[b], time_steps
                        )
                else:
                    processed[:, i, c] = slice_data

    # Apply Taper
    taper = np.hanning(time_steps)
    processed *= taper[np.newaxis, np.newaxis, np.newaxis, :]

    # Per-station normalization
    max_vals = np.max(np.abs(processed), axis=(2, 3), keepdims=True)
    processed = np.divide(
        processed, max_vals, out=np.zeros_like(processed), where=max_vals > 1e-9
    )

    return processed


def _tpb2q(t: list[float], p: list[float], b: list[float]) -> np.ndarray:
    """Convert T/P/B axes to a unit quaternion (helper for Kagan angle)."""
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


def kagan_angle_deg(mt1: np.ndarray, mt2: np.ndarray) -> float:
    """
    Given two 3x3 moment tensors, return the Kagan angle in degrees.

    After Kagan (1991) and Tape & Tape (2012).
    """
    pbt2tpb = np.array(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    ai = pbt2tpb @ mt1.T
    aj = pbt2tpb @ mt2.T
    u = ai @ aj.T
    tk, pk, bk = u.tolist()
    qk = _tpb2q(tk, pk, bk)
    return float(2.0 * (180.0 / np.pi) * math.acos(np.max(np.abs(qk))))


def station_misfit_summary(observation: np.ndarray, synthetics: np.ndarray) -> dict:
    """
    Compute simple per-station/per-component RMS misfit in the processed space.

    Args:
        observation: (N, 3, T)
        synthetics: (N, 3, T)
    Returns:
        dict with per-station RMS and per-component RMS arrays.
    """
    resid = synthetics - observation
    # RMS per station across components+time
    rms_station = np.sqrt(np.mean(resid**2, axis=(1, 2)))
    # RMS per station/component across time
    rms_station_comp = np.sqrt(np.mean(resid**2, axis=2))  # (N, 3)
    # RMS per component across stations+time
    rms_comp = np.sqrt(np.mean(resid**2, axis=(0, 2)))  # (3,)
    return {
        "rms_station": rms_station,
        "rms_station_comp": rms_station_comp,
        "rms_comp": rms_comp,
    }


class L2Likelihood:
    """Simple L2 (least-squares) likelihood for debugging."""

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def compute_log_likelihood(
        self, synthetics: np.ndarray, observation: np.ndarray
    ) -> np.ndarray:
        """
        Compute L2 log-likelihood: -0.5 * sum((syn - obs)^2) / sigma^2

        Args:
            synthetics: (B, N, C, T)
            observation: (N, C, T)
        Returns:
            log_likelihoods: (B,)
        """
        B = synthetics.shape[0]

        # Broadcast observation
        obs = observation[np.newaxis, ...]  # (1, N, C, T)

        # Compute squared residuals
        residuals = synthetics - obs  # (B, N, C, T)

        # Sum over stations, channels, time
        ss_res = np.sum(residuals**2, axis=(1, 2, 3))  # (B,)

        # Log-likelihood (Gaussian)
        ll = -0.5 * ss_res / (self.sigma**2)

        return ll


class FastSynthesizer:
    """Optimized synthesizer that pre-computes Green's functions once."""

    def __init__(
        self,
        velocity_model,
        stations,
        source_loc,
        duration=1.0,
        fmax=500.0,
        t0=0.01,
        work_dir=None,
        cache_id=2387,
        generate_if_missing=True,
    ):
        self.velocity_model = velocity_model
        self.stations = stations
        self.source_loc = source_loc
        self.duration = duration
        self.fmax = fmax
        self.t0 = t0
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.cache_id = int(cache_id)
        self.generate_if_missing = bool(generate_if_missing)
        self.axpath = str(AXITRA_SRC)

        self.ap = None
        self._temp_dir = None
        self._orig_dir = None

    def _cache_metadata(self, sources: np.ndarray) -> dict:
        return {
            "backend": "axitra",
            "cache_id": self.cache_id,
            "duration": float(self.duration),
            "fmax": float(self.fmax),
            "t0": float(self.t0),
            "axpath": self.axpath,
            "velocity_model": np.asarray(self.velocity_model, dtype=float).tolist(),
            "stations": np.asarray(self.stations, dtype=float).tolist(),
            "sources": np.asarray(sources, dtype=float).tolist(),
        }

    def _cache_status(self, metadata_path: Path, metadata: dict) -> tuple[bool, str]:
        if not metadata_path.exists():
            return False, f"missing metadata file {metadata_path}"
        data_file = metadata_path.parent / f"axi_{self.cache_id}.data"
        res_file = metadata_path.parent / f"axi_{self.cache_id}.res"
        if (
            not data_file.exists()
            or not res_file.exists()
            or data_file.stat().st_size == 0
            or res_file.stat().st_size == 0
        ):
            return False, (
                f"missing or empty Axitra cache files axi_{self.cache_id}.data/"
                f"axi_{self.cache_id}.res in {metadata_path.parent}"
            )
        with metadata_path.open("r", encoding="utf-8") as f:
            old_metadata = json.load(f)
        if old_metadata != metadata:
            return False, f"metadata in {metadata_path} does not match this run"
        return True, "ok"

    def _cache_is_valid(self, metadata_path: Path, metadata: dict) -> bool:
        valid, _ = self._cache_status(metadata_path, metadata)
        return valid

    def _cache_result_exists(self, run_dir: Path) -> bool:
        data_file = run_dir / f"axi_{self.cache_id}.data"
        res_file = run_dir / f"axi_{self.cache_id}.res"
        return (
            data_file.exists()
            and res_file.exists()
            and data_file.stat().st_size > 0
            and res_file.stat().st_size > 0
        )

    def setup(self):
        """Compute Green's functions (run once)."""
        if Axitra is None or moment is None:
            raise RuntimeError(
                "Axitra backend requested, but the axitra Python module is not "
                "available. Use --synthetic-backend openswpc_gf or install/build "
                "Axitra and make its Python module importable."
            )

        sources = np.array(
            [[1, self.source_loc[0], self.source_loc[1], self.source_loc[2]]],
            dtype=float,
        )

        if self.work_dir is None:
            self._temp_dir = tempfile.mkdtemp()
            run_dir = Path(self._temp_dir)
        else:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            run_dir = self.work_dir
        self._orig_dir = os.getcwd()
        os.chdir(run_dir)

        metadata = self._cache_metadata(sources)
        metadata_path = run_dir / "axitra_greens_metadata.json"
        if self.work_dir is not None:
            cache_valid, cache_reason = self._cache_status(metadata_path, metadata)
        else:
            cache_valid, cache_reason = False, "no persistent work_dir"

        if self.work_dir is not None and cache_valid:
            print(f"Loading Axitra Green's functions from {run_dir}")
            self.ap = Axitra.read(suffix=str(self.cache_id), axpath=self.axpath)
            if self.ap is None:
                raise RuntimeError(f"Failed to load Axitra Green's functions from {run_dir}")
            self.ap.id = self.cache_id
            self.ap.sid = f"axi_{self.cache_id}"
            return

        if self.work_dir is not None and not self.generate_if_missing:
            raise RuntimeError(
                "Required Axitra Green's functions are missing or stale: "
                f"{cache_reason}. Run generate_cape_axitra_greens.py for this event "
                f"or pass a matching --axitra-greens-dir."
            )

        print("Computing Green's functions (one-time cost)...")
        t0 = time.time()

        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)

        try:
            self.ap = Axitra(
                self.velocity_model,
                self.stations,
                sources,
                fmax=self.fmax,
                duration=self.duration,
                xl=0.0,
                latlon=False,
                axpath=self.axpath,
                id=self.cache_id if self.work_dir is not None else None,
            )
            self.ap = moment.green(self.ap)
            if self.work_dir is not None and not self._cache_result_exists(run_dir):
                raise RuntimeError(
                    f"Axitra Green-function generation failed in {run_dir}; "
                    f"missing or empty axi_{self.cache_id}.res"
                )
            if self.work_dir is not None:
                with metadata_path.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, sort_keys=True)
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
            os.close(devnull)

        print(f"Green's functions computed in {time.time() - t0:.2f}s")

    def synthesize_batch(
        self, mt_params: np.ndarray, m0: float, source_delay: float = 0.1
    ) -> np.ndarray:
        """Synthesize waveforms for a batch of MT params."""
        B = mt_params.shape[0]
        N = self.ap.nstation
        T = self.ap.npt

        waveforms = np.zeros((B, N, 3, T), dtype=np.float32)

        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)

        try:
            for i in range(B):
                gamma, delta, kappa, h, sigma = mt_params[i]
                mt6_norm = Tape_MT6(gamma, delta, kappa, h, sigma)

                mxx = mt6_norm[0] * m0
                myy = mt6_norm[1] * m0
                mzz = mt6_norm[2] * m0
                mxy = (mt6_norm[3] / np.sqrt(2.0)) * m0
                mxz = (mt6_norm[4] / np.sqrt(2.0)) * m0
                myz = (mt6_norm[5] / np.sqrt(2.0)) * m0

                hist_mt = np.array(
                    [[1, mxx, mxy, mxz, myy, myz, mzz, source_delay]], dtype=float
                )

                t, ux, uy, uz = moment.conv_tensor(
                    self.ap, hist_mt, source_type=1, t0=self.t0, unit=2
                )

                waveforms[i] = np.stack([uz, ux, uy], axis=1)
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
            os.close(devnull)

        return waveforms

    def cleanup(self):
        """Clean up temp directory."""
        if self._orig_dir:
            os.chdir(self._orig_dir)
        if self._temp_dir:
            import shutil

            shutil.rmtree(self._temp_dir, ignore_errors=True)


class OpenSWPCGFSynthesizer:
    """Fast synthesizer from precomputed OpenSWPC MT-basis Green's functions."""

    def __init__(
        self, gf_file, stations, source_loc=None, duration=1.0, fmax=500.0, t0=0.01
    ):
        self.gf_file = str(gf_file)
        self.stations = np.asarray(stations, dtype=float)
        self.source_loc = source_loc
        self.duration = duration
        self.fmax = fmax
        self.t0 = t0

        self._gf = None
        self._dt = None
        self.ap = None

    def setup(self):
        data = np.load(self.gf_file, allow_pickle=False)
        gf_basis = np.asarray(data["gf_basis"], dtype=np.float32)
        station_coords = np.asarray(data["station_coords"], dtype=float)
        dt = float(np.asarray(data["dt"]).reshape(()))
        basis_order = (
            [str(x) for x in np.asarray(data["basis_order"])]
            if "basis_order" in data.files
            else []
        )
        component_order = (
            [str(x) for x in np.asarray(data["component_order"])]
            if "component_order" in data.files
            else []
        )
        quantity = (
            str(np.asarray(data["quantity"]).reshape(()))
            if "quantity" in data.files
            else ""
        )

        if gf_basis.ndim != 4 or gf_basis.shape[0] != 6 or gf_basis.shape[2] != 3:
            raise ValueError(
                f"Invalid gf_basis shape: {gf_basis.shape}, expected (6,N,3,T)"
            )
        if basis_order != ["Mxx", "Myy", "Mzz", "Mxy", "Mxz", "Myz"]:
            raise ValueError(f"Unexpected OpenSWPC GF basis_order: {basis_order}")
        if component_order != ["Z", "N", "E"]:
            raise ValueError(
                f"Unexpected OpenSWPC GF component_order: {component_order}"
            )
        if quantity.upper() != "V":
            raise ValueError(
                f"OpenSWPC GF library must contain velocity, got {quantity}"
            )

        req_xyz = self.stations[:, 1:4]
        gf_xyz = station_coords[:, 1:4]
        idx = []
        for p in req_xyz:
            d2 = np.sum((gf_xyz - p[None, :]) ** 2, axis=1)
            idx.append(int(np.argmin(d2)))

        self._gf = gf_basis[:, np.asarray(idx, dtype=int), :, :]
        self._dt = dt
        self.duration = dt * float(self._gf.shape[3])
        self.ap = SimpleNamespace(nstation=self._gf.shape[1], npt=self._gf.shape[3])

    def synthesize_batch(
        self, mt_params: np.ndarray, m0: float, source_delay: float = 0.1
    ) -> np.ndarray:
        if self._gf is None:
            raise RuntimeError(
                "OpenSWPC GF synthesizer not initialized. Call setup() first."
            )

        B = mt_params.shape[0]
        coeffs = np.zeros((B, 6), dtype=np.float64)
        for i in range(B):
            gamma, delta, kappa, h, sigma = mt_params[i]
            mt6 = Tape_MT6(gamma, delta, kappa, h, sigma)
            coeffs[i, 0] = mt6[0] * m0
            coeffs[i, 1] = mt6[1] * m0
            coeffs[i, 2] = mt6[2] * m0
            coeffs[i, 3] = (mt6[3] / np.sqrt(2.0)) * m0
            coeffs[i, 4] = (mt6[4] / np.sqrt(2.0)) * m0
            coeffs[i, 5] = (mt6[5] / np.sqrt(2.0)) * m0

        out = np.einsum("bq,qnct->bnct", coeffs, self._gf, optimize=True).astype(
            np.float32
        )

        return out

    def cleanup(self):
        return


def plot_waveform_comparison(
    observation, best_synthetic, stations, output_path="waveform_comparison.png"
):
    """Plot observation vs best-fit synthetic waveform."""
    N, C, T = observation.shape
    n_stations_to_plot = N  # Show ALL stations

    fig, axes = plt.subplots(
        n_stations_to_plot, 3, figsize=(15, 1.8 * n_stations_to_plot)
    )
    component_names = ["Z (P-wave)", "N (S-wave)", "E (S-wave)"]

    time_axis = np.linspace(0, 0.14, T)  # phase_window_len = 0.14s

    for i in range(n_stations_to_plot):
        for c in range(3):
            ax = axes[i, c] if n_stations_to_plot > 1 else axes[c]

            ax.plot(
                time_axis, observation[i, c], "b-", label="Observation", linewidth=1.5
            )
            ax.plot(
                time_axis, best_synthetic[i, c], "r--", label="Best Fit", linewidth=1.5
            )

            if i == 0:
                ax.set_title(component_names[c])
            if c == 0:
                ax.set_ylabel(f"Station {i + 1}")
            if i == n_stations_to_plot - 1:
                ax.set_xlabel("Time (s)")

            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved waveform comparison to {output_path}")
    plt.close()


def find_best_checkpoint(base_dir="lightning_logs_phase_sep/lightning_logs"):
    """
    Find the best checkpoint from the latest version in the lightning_logs directory.

    Strategy:
    1. Find latest 'version_X' directory.
    2. Look for .ckpt files in 'checkpoints' subdir.
    3. Sort by validation loss if present in filename ('loss=0.xx'), else use latest modified.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        # Try fallback if relative path might be different
        base_path = Path("lightning_logs_phase_sep")
        if not base_path.exists():
            raise FileNotFoundError(
                f"Could not find lightning logs directory at {base_dir} or lightning_logs_phase_sep"
            )
        # If we found lightning_logs_phase_sep but it doesn't have an inner lightning_logs,
        # maybe versions are at root?
        if (base_path / "lightning_logs").exists():
            base_path = base_path / "lightning_logs"

    # 1. Find all version directories
    version_dirs = [
        d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("version_")
    ]
    if not version_dirs:
        raise FileNotFoundError(f"No version directories found in {base_path}")

    # Sort by version number
    version_dirs.sort(key=lambda d: int(d.name.split("_")[1]))
    latest_version = version_dirs[-1]
    print(f"Latest run found: {latest_version}")

    # 2. Look in checkpoints
    ckpt_dir = latest_version / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found in {latest_version}")

    # Search recursively for .ckpt files because they might be in subdirectories
    ckpts = list(ckpt_dir.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir} (recursively)")

    # 3. Find best checkpoint
    # Try to parse loss from filename "epoch=xx-val_loss=0.xx.ckpt" or similar
    # PyTorch Lightning default: "epoch=0-step=100.ckpt" or user defined
    # User example: "best-checkpoint-epoch=08-val/loss=0.45.ckpt"
    # Wait, user path was "best-checkpoint-epoch=08-val/loss=0.45.ckpt" -> likely "/" is part of dir name or just weird filename char?
    # Linux allows almost anything. if "val/loss" is in name, it might be a subdir or just a name.
    # Let's assume filename contains "loss=".

    best_ckpt = None
    min_loss = float("inf")

    import re

    for ckpt in ckpts:
        name = ckpt.name
        # Check for loss in name
        match = re.search(r"loss=([0-9.]+)", name)
        if match:
            try:
                loss = float(match.group(1))
                if loss < min_loss:
                    min_loss = loss
                    best_ckpt = ckpt
            except Exception:
                pass  # parsing failed, ignore

    if best_ckpt:
        print(f"Selected best checkpoint based on loss ({min_loss}): {best_ckpt.name}")
        return best_ckpt

    # Fallback: Sort by modification time (newest first)
    ckpts.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    print(
        f"No loss info found in filenames. Selected most recent checkpoint: {ckpts[0].name}"
    )
    return ckpts[0]


def run_inversion(args):
    """Main Inversion Loop."""

    # 1. Load Data
    print(f"Loading Observation #{args.event_index} from {args.data_file}...")
    gt_params, observation = load_observation(args.data_file, args.event_index)

    print("\nGround Truth Params:")
    for k, v in gt_params.items():
        print(f"  {k}: {v:.4f}")

    # 2. Setup Geometry
    stations, _, _, _ = load_stations_from_xml(args.stations_dir)
    velocity_model = load_velocity_model(args.velocity_model)
    source_loc = (gt_params["x"], gt_params["y"], gt_params["z"])
    m0 = 10 ** (1.5 * gt_params["mw"] + 9.1)

    # 3. Setup Likelihood
    if args.likelihood == "siamese":
        raise RuntimeError(
            "The clean bayesian_waveform_fitting copy does not include the old "
            "Siamese/deep-learning likelihood. Use --likelihood l2 here, or use "
            "run_inversion.py with --likelihood gsot/softdtw/l2."
        )
    else:
        print(f"\nUsing L2 (least-squares) likelihood with sigma={args.sigma}")
        likelihood_model = L2Likelihood(sigma=args.sigma)

    # 4. Setup Fast Synthesizer
    synthesizer = FastSynthesizer(
        velocity_model=velocity_model,
        stations=stations,
        source_loc=source_loc,
        duration=1.0,
        fmax=500.0,
        t0=1.0 / (30.0 * np.pi),
    )
    synthesizer.setup()

    # 5. SMC Inversion
    print("\nStarting Adaptive SMC Inversion...")
    n_particles = args.n_particles
    n_stages = args.n_stages

    # Initialize Particles
    particles = np.zeros((n_particles, 5))
    particles[:, 0] = np.random.uniform(-np.pi / 6, np.pi / 6, n_particles)
    particles[:, 1] = np.random.uniform(-np.pi / 2, np.pi / 2, n_particles)
    particles[:, 2] = np.random.uniform(0, 2 * np.pi, n_particles)
    particles[:, 3] = np.random.uniform(0, 1, n_particles)
    particles[:, 4] = np.random.uniform(-np.pi / 2, np.pi / 2, n_particles)

    history = []
    best_particle = None
    best_ll = -np.inf
    best_synthetic = None

    # Track last-stage (pre-resample) posterior approximation.
    last_stage_particles = None
    last_stage_weights = None
    last_stage_log_weights = None

    for stage in tqdm(range(n_stages), desc="SMC Stages"):
        # A. Synthesize
        raw_waveforms = synthesizer.synthesize_batch(particles, m0, source_delay=0.1)

        # B. Phase Windowing
        processed = extract_phase_windows_batch(
            waveforms=raw_waveforms,
            stations=stations,
            source_loc=source_loc,
            velocity_model=velocity_model,
            duration=1.0,
            phase_window_len=0.14,
            source_delay=0.1,
            time_steps=70,
        )

        # C. Compute Likelihoods
        log_weights = likelihood_model.compute_log_likelihood(processed, observation)

        # Track best particle
        best_idx = np.argmax(log_weights)
        if log_weights[best_idx] > best_ll:
            best_ll = log_weights[best_idx]
            best_particle = particles[best_idx].copy()
            best_synthetic = processed[best_idx].copy()

        # D. Selection / Resampling
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= np.sum(weights)
        ess = 1.0 / np.sum(weights**2)

        # Weighted mean BEFORE resampling/mutation is a better summary of the
        # current posterior approximation than the unweighted mean after mutation.
        weighted_mean = np.sum(particles * weights[:, None], axis=0)
        history.append(weighted_mean)

        # Keep last-stage posterior approximation for reporting.
        last_stage_particles = particles.copy()
        last_stage_weights = weights.copy()
        last_stage_log_weights = log_weights.copy()

        print(
            f"Stage {stage + 1}/{n_stages}: MaxLL={np.max(log_weights):.2f}, MinLL={np.min(log_weights):.2f}, ESS={ess:.1f}"
        )

        indices = np.random.choice(n_particles, size=n_particles, p=weights)
        particles = particles[indices]

        # E. Mutation
        step_size = 0.1 * (0.9**stage)
        noise = np.random.normal(0, step_size, size=particles.shape)
        particles = particles + noise

        particles[:, 0] = np.clip(particles[:, 0], -np.pi / 6, np.pi / 6)
        particles[:, 1] = np.clip(particles[:, 1], -np.pi / 2, np.pi / 2)
        particles[:, 3] = np.clip(particles[:, 3], 0, 1)

        # Note: history is tracked above (weighted mean pre-resample).

    # Cleanup
    synthesizer.cleanup()

    # Final Result
    if last_stage_particles is None or last_stage_weights is None:
        raise RuntimeError("Internal error: missing last-stage posterior state.")

    # Posterior weighted mean (last stage, before resampling/mutation).
    mean_params = np.sum(last_stage_particles * last_stage_weights[:, None], axis=0)

    print("\n" + "=" * 50)
    print(f"INVERSION COMPLETE ({args.likelihood.upper()} Likelihood)")
    print("=" * 50)
    print("Posterior (last stage) weighted mean:")
    print(f"  Gamma: {mean_params[0]:.4f}  vs True: {gt_params['gamma']:.4f}")
    print(f"  Delta: {mean_params[1]:.4f}  vs True: {gt_params['delta']:.4f}")
    print(f"  Kappa: {mean_params[2]:.4f}  vs True: {gt_params['kappa']:.4f}")
    print(f"  h    : {mean_params[3]:.4f}  vs True: {gt_params['h']:.4f}")
    print(f"  Sigma: {mean_params[4]:.4f}  vs True: {gt_params['sigma']:.4f}")

    print(f"\nBest Particle (LL={best_ll:.4f}):")
    print(f"  Gamma: {best_particle[0]:.4f}")
    print(f"  Delta: {best_particle[1]:.4f}")
    print(f"  Kappa: {best_particle[2]:.4f}")
    print(f"  h    : {best_particle[3]:.4f}")
    print(f"  Sigma: {best_particle[4]:.4f}")

    # MT-space similarity diagnostics: Tape params are not a great distance metric.
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
        mt_mean = Tape_MT33(
            mean_params[0],
            mean_params[1],
            mean_params[2],
            mean_params[3],
            mean_params[4],
        )

        k_best = kagan_angle_deg(mt_true, mt_best)
        k_mean = kagan_angle_deg(mt_true, mt_mean)

        mt6_true = MT33_MT6(mt_true)
        mt6_best = MT33_MT6(mt_best)
        mt6_mean = MT33_MT6(mt_mean)
        dot_best = float(np.dot(mt6_true, mt6_best))
        dot_mean = float(np.dot(mt6_true, mt6_mean))

        print("\nMT-space similarity (lower Kagan is better):")
        print(
            f"  Kagan(best,true): {k_best:.2f} deg | MT6 dot(best,true): {dot_best:.3f}"
        )
        print(
            f"  Kagan(mean,true): {k_mean:.2f} deg | MT6 dot(mean,true): {dot_mean:.3f}"
        )
    except Exception as exc:
        print(f"\nWarning: failed to compute MT-space diagnostics: {exc}")

    # Plot waveform comparison
    if best_synthetic is not None:
        plot_waveform_comparison(
            observation,
            best_synthetic,
            stations,
            output_path=f"waveform_comparison_{args.likelihood}.png",
        )
        misfit = station_misfit_summary(observation, best_synthetic)
        worst = int(np.argmax(misfit["rms_station"]))
        rms_station = misfit["rms_station"]
        print("\nWaveform misfit (processed space):")
        print(
            f"  RMS per component [Z,N,E]: {misfit['rms_comp'][0]:.4e}, {misfit['rms_comp'][1]:.4e}, {misfit['rms_comp'][2]:.4e}"
        )
        print(f"  Worst station (1-based): {worst + 1} | RMS: {rms_station[worst]:.4e}")
        # Also show top-3 worst stations for quick debugging.
        top3 = np.argsort(-rms_station)[:3]
        top3_str = ", ".join([f"{i + 1}:{rms_station[i]:.2e}" for i in top3])
        print(f"  Top-3 worst stations (idx:rms): {top3_str}")

    # Plot parameter convergence
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

    plt.xlabel("Stage")
    plt.suptitle(f"Inversion Convergence ({args.likelihood.upper()} Likelihood)")
    plt.tight_layout()
    plt.savefig(f"inversion_result_{args.likelihood}.png")
    print(f"Saved result plot to inversion_result_{args.likelihood}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default="data_phase.h5")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to Siamese checkpoint. If None, auto-selects best from lightning_logs.",
    )
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--stations-dir", type=str, default="stationxml")
    parser.add_argument("--velocity-model", type=str, default="forge.tvel")
    parser.add_argument("--n-particles", type=int, default=100)
    parser.add_argument("--n-stages", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--likelihood",
        type=str,
        choices=["l2"],
        default="l2",
        help="Legacy synthetic-test likelihood. The clean copy keeps only L2 here.",
    )

    args = parser.parse_args()

    # removed conditional check requiring checkpoint since it's now optional

    run_inversion(args)
