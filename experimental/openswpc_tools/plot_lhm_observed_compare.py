import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime, read
from scipy.signal import butter, sosfiltfilt


SAC_TO_COMPONENT = {"Vz": "Z", "Vy": "N", "Vx": "E"}


def _bandpass(x, fs, low, high):
    nyq = 0.5 * fs
    lo = max(low / nyq, 1e-5)
    hi = min(high / nyq, 0.999)
    if not 0 < lo < hi < 1:
        return np.asarray(x, dtype=float)
    sos = butter(3, [lo, hi], btype="bandpass", output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=float))


def _safe_scale(obs_t, obs_y, syn_t, syn_y):
    obs_i = np.interp(syn_t, obs_t, obs_y, left=0.0, right=0.0)
    denom = float(np.dot(syn_y, syn_y))
    if denom <= 0.0:
        return 1.0
    return float(np.dot(obs_i, syn_y) / denom)


def main():
    parser = argparse.ArgumentParser(
        description="Overlay OpenSWPC LHM synthetic velocity with observed CAPE velocity."
    )
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--invdata", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--suffix", default=".lhm_observed_compare")
    parser.add_argument("--filter-low", type=float, default=0.5)
    parser.add_argument("--filter-high", type=float, default=8.0)
    parser.add_argument("--scale-mode", choices=["peak", "lsq", "none"], default="peak")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=7.0)
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    metadata = json.loads((case_dir / "case_metadata.json").read_text(encoding="utf-8"))
    inv_path = Path(args.invdata) if args.invdata else case_dir.parent / "invdata.pkl"
    with inv_path.open("rb") as f:
        invdata = pickle.load(f)

    origin = UTCDateTime(invdata["source"]["origin_time"])
    stations = metadata["stations"]
    output_arg = Path(args.output) if args.output else None
    for station in stations:
        sid = station["station_id"]
        label = station["label"]
        obs = invdata["waveforms"][sid]
        picks = invdata.get("picks", {}).get(sid, {})

        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        scale_report = []
        for ax, comp in zip(axes, ["Z", "N", "E"]):
            obs_y = np.asarray(obs["components"][comp], dtype=float)
            obs_fs = float(obs["sampling_rate"])
            obs_t0 = float(UTCDateTime(obs["starttime"]) - origin)
            obs_t = obs_t0 + np.arange(obs_y.size) / obs_fs
            obs_f = _bandpass(obs_y, obs_fs, args.filter_low, args.filter_high)

            sac_comp = next(k for k, v in SAC_TO_COMPONENT.items() if v == comp)
            sac_path = next((case_dir / "out" / "wav").glob(f"*.{label}.{sac_comp}.sac"))
            tr = read(str(sac_path))[0]
            syn_t = np.arange(tr.stats.npts) * float(tr.stats.delta)
            syn_f = _bandpass(
                tr.data.astype(float), tr.stats.sampling_rate, args.filter_low, args.filter_high
            )
            if args.scale_mode == "lsq":
                scale = _safe_scale(obs_t, obs_f, syn_t, syn_f)
            elif args.scale_mode == "peak":
                obs_peak = float(np.max(np.abs(obs_f)))
                syn_peak = float(np.max(np.abs(syn_f)))
                sign = np.sign(_safe_scale(obs_t, obs_f, syn_t, syn_f)) or 1.0
                scale = sign * obs_peak / syn_peak if syn_peak > 0.0 else 1.0
            else:
                scale = 1.0
            syn_scaled = syn_f * scale

            ax.plot(obs_t, obs_f, color="black", lw=1.1, label="observed velocity")
            ax.plot(syn_t, syn_scaled, color="#d55e00", lw=1.0, label="OpenSWPC 1D synthetic")
            ax.axhline(0.0, color="0.8", lw=0.6)
            ax.set_ylabel(comp)
            peak_t = float(syn_t[int(np.argmax(np.abs(syn_scaled)))])
            scale_report.append((comp, scale, peak_t))

        for ax in axes:
            if picks.get("P"):
                ax.axvline(float(UTCDateTime(picks["P"]) - origin), color="#0072b2", ls="--", lw=1.0)
            if picks.get("S"):
                ax.axvline(float(UTCDateTime(picks["S"]) - origin), color="#009e73", ls="--", lw=1.0)
            ax.set_xlim(args.tmin, args.tmax)

        axes[0].legend(loc="upper right", frameon=False)
        axes[-1].set_xlabel("Time from origin (s)")
        fig.suptitle(
            f"{sid} eq02387 OpenSWPC LHM vs observed, {args.filter_low:g}-{args.filter_high:g} Hz"
        )
        fig.tight_layout()

        if output_arg and (output_arg.exists() and output_arg.is_dir()):
            output = output_arg / f"{sid}{args.suffix}.png"
        elif output_arg and len(stations) == 1:
            output = output_arg
        elif output_arg:
            output_arg.mkdir(parents=True, exist_ok=True)
            output = output_arg / f"{sid}{args.suffix}.png"
        else:
            output = case_dir / f"{sid}{args.suffix}.png"
        fig.savefig(output, dpi=160)
        plt.close(fig)

        print(f"Wrote {output}")
        for comp, scale, peak_t in scale_report:
            print(f"{sid} {comp}: {args.scale_mode} scale={scale:.3e}, synthetic peak t={peak_t:.3f}s")


if __name__ == "__main__":
    main()
