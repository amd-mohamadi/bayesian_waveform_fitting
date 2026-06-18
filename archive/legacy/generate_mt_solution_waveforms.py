"""Archived legacy script; not part of the active Axitra/GSOT workflow."""

import argparse
import csv
import pickle
from pathlib import Path
from types import SimpleNamespace

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from run_inversion import (
    _apply_trace_mask_to_batch,
    align_synthetics_for_plot,
    build_observation_windows,
    build_phase_window_cache_dual,
    build_phase_window_cache_from_picks,
    extract_phase_windows_batch_cached_dual,
    parse_components,
    select_station_geometry,
)
from src.forward import FastSynthesizer
from src.forward.axitra import moment
from src.io import load_stations_from_xml, load_velocity_model


def load_mt6_from_results(path: Path, solution: str) -> np.ndarray:
    with path.open("r", newline="") as f:
        row = next(csv.DictReader(f))
    mt6 = np.array([float(row[f"{solution}_mt6_{i}"]) for i in range(6)], dtype=float)
    norm = float(np.linalg.norm(mt6))
    if norm <= 0.0:
        raise ValueError(f"{solution} MT6 in {path} has zero norm")
    return mt6 / norm


def synthesize_mt6(synth: FastSynthesizer, mt6_norm: np.ndarray, m0: float, source_delay: float) -> np.ndarray:
    if moment is None:
        raise RuntimeError("Axitra moment module is unavailable")

    mt6_norm = np.asarray(mt6_norm, dtype=float).reshape(6)
    nsta = synth.ap.nstation
    npts = synth.ap.npt
    waveforms = np.zeros((nsta, 3, npts), dtype=np.float32)

    # Project MT6 [Mxx, Myy, Mzz, sqrt(2)Mxy, sqrt(2)Mxz, sqrt(2)Myz] to Axitra's tensor history.
    mxx = mt6_norm[0] * m0
    myy = mt6_norm[1] * m0
    mzz = mt6_norm[2] * m0
    mxy = (mt6_norm[3] / np.sqrt(2.0)) * m0
    mxz = (mt6_norm[4] / np.sqrt(2.0)) * m0
    myz = (mt6_norm[5] / np.sqrt(2.0)) * m0
    hist_mt = np.array([[1, mxx, mxy, mxz, myy, myz, mzz, source_delay]], dtype=float)

    _, ux, uy, uz = moment.conv_tensor(synth.ap, hist_mt, source_type=1, t0=synth.t0, unit=2)
    waveforms[:] = np.stack([uz, uy, ux], axis=1)
    return waveforms


def plot_station(t: np.ndarray, y: np.ndarray, station_id: str, solution: str, mt6: np.ndarray, out: Path) -> None:
    labels = ["Z", "N", "E"]
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    scale = peak if peak > 0.0 else 1.0

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, y[i] / scale, color="tab:red", lw=1.0)
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"{station_id} Axitra MT {solution} | peak={peak:.3e}\n"
        f"mt6=[{', '.join(f'{x:.3f}' for x in mt6)}]",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_phase_window_overlay(
    obs: np.ndarray,
    syn: np.ndarray,
    station_ids: list[str],
    out_path: Path,
    p_len: float,
    s_len: float,
    comp_tokens: list[str],
    comp_labels: list[str],
    include_mask: np.ndarray,
    syn_label: str,
) -> None:
    nsta, ncomp, npts = obs.shape
    t_p = np.linspace(0.0, float(p_len), npts)
    t_s = np.linspace(0.0, float(s_len), npts)
    include_mask = np.asarray(include_mask, dtype=bool)

    fig, axes = plt.subplots(
        nsta, ncomp, figsize=(5 * ncomp, 1.8 * nsta), squeeze=False
    )
    legend_handles = [
        Line2D([0], [0], color="b", lw=1.0, label="Observed"),
        Line2D([0], [0], color="r", lw=1.3, linestyle="--", label=syn_label),
    ]
    if np.any(~include_mask):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="g",
                lw=1.3,
                linestyle="--",
                label=f"{syn_label} (missing trace)",
            )
        )

    for i in range(nsta):
        for c in range(ncomp):
            ax = axes[i, c]
            tt = t_p if comp_tokens[c] == "PZ" else t_s
            color = "r" if include_mask[i, c] else "g"
            ax.plot(tt, obs[i, c], "b-", lw=1.0)
            ax.plot(tt, syn[i, c], linestyle="--", color=color, lw=1.3)
            if i == 0:
                ax.set_title(comp_labels[c])
            if c == 0:
                ax.set_ylabel(station_ids[i])
            if i == 0 and c == 0:
                ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_peak_summary(path: Path, station_ids: list[str], waveforms: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["station_id", "peak_abs", "peak_z", "peak_n", "peak_e"])
        for sid, y in zip(station_ids, waveforms):
            peaks = np.max(np.abs(y), axis=1)
            writer.writerow(
                [
                    sid,
                    f"{float(np.max(peaks)):.8e}",
                    f"{float(peaks[0]):.8e}",
                    f"{float(peaks[1]):.8e}",
                    f"{float(peaks[2]):.8e}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Axitra synthetics from a saved MT6 solution"
    )
    parser.add_argument("--event-dir", default="cape_events/eq02387")
    parser.add_argument("--stations-dir", default="stations")
    parser.add_argument("--velocity-model", default="cape_pathavg_eq02387_600m.tvel")
    parser.add_argument(
        "--axitra-greens-dir",
        default="cape_events/eq02387/axitra_greens_pathavg_600m",
    )
    parser.add_argument(
        "--mt-results",
        default="eq02387_PPolarity_SHPolarity_PSHAmplitudeRatio_PSVAmplitudeRatio_mt/mt_results.csv",
    )
    parser.add_argument("--solution", choices=("mode", "median"), default="mode")
    parser.add_argument("--output-dir", default="waveforms/eq02387_mt_solution")
    parser.add_argument("--duration", type=float, default=7.0)
    parser.add_argument("--fmax", type=float, default=30.0)
    parser.add_argument("--source-target-freq-hz", type=float, default=10.0)
    parser.add_argument("--source-delay", type=float, default=0.0)
    parser.add_argument("--components", default="PZ,SN,SE")
    parser.add_argument("--phase-window-p-len", type=float, default=0.50)
    parser.add_argument("--phase-window-s-len", type=float, default=0.70)
    parser.add_argument("--synthetic-phase-window-p-len", type=float, default=None)
    parser.add_argument("--synthetic-phase-window-s-len", type=float, default=None)
    parser.add_argument(
        "--synthetic-window-source",
        choices=("velocity", "picks"),
        default="velocity",
    )
    parser.add_argument("--time-steps", type=int, default=100)
    parser.add_argument("--bp-p-low", type=float, default=2.0)
    parser.add_argument("--bp-p-high", type=float, default=25.0)
    parser.add_argument("--bp-s-low", type=float, default=1.0)
    parser.add_argument("--bp-s-high", type=float, default=17.0)
    parser.add_argument("--bp-order", type=int, default=4)
    parser.add_argument("--taper-frac", type=float, default=0.05)
    parser.add_argument("--detrend", action="store_true", default=True)
    parser.add_argument("--demean", action="store_true", default=True)
    parser.add_argument("--apply-window-taper", action="store_true", default=True)
    parser.add_argument(
        "--normalize-per-station",
        dest="normalize_per_station",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-normalize-per-station",
        dest="normalize_per_station",
        action="store_false",
    )
    parser.add_argument("--lag-max-shift-samples", type=int, default=15)
    parser.add_argument("--plot-align-lag", action="store_true", default=True)
    parser.add_argument(
        "--no-plot-align-lag",
        action="store_false",
        dest="plot_align_lag",
    )
    parser.add_argument("--overlay-plot", default=None)
    args = parser.parse_args()
    if args.synthetic_phase_window_p_len is None:
        args.synthetic_phase_window_p_len = float(args.phase_window_p_len)
    if args.synthetic_phase_window_s_len is None:
        args.synthetic_phase_window_s_len = float(args.phase_window_s_len)

    event_dir = Path(args.event_dir)
    with (event_dir / "invdata.pkl").open("rb") as f:
        invdata = pickle.load(f)

    source = invdata["source"]
    source_loc = (float(source["x"]), float(source["y"]), float(source["z"]))
    m0 = 10.0 ** (1.5 * float(source["mw"]) + 9.1)

    comp_tokens, comp_indices, comp_labels = parse_components(args.components)
    obs_args = SimpleNamespace(
        time_steps=int(args.time_steps),
        phase_window_p_len=float(args.phase_window_p_len),
        phase_window_s_len=float(args.phase_window_s_len),
        bp_p_low=float(args.bp_p_low),
        bp_p_high=float(args.bp_p_high),
        bp_s_low=float(args.bp_s_low),
        bp_s_high=float(args.bp_s_high),
        bp_order=int(args.bp_order),
        taper_frac=float(args.taper_frac),
        detrend=bool(args.detrend),
        demean=bool(args.demean),
        apply_window_taper=bool(args.apply_window_taper),
        normalize_per_station=bool(args.normalize_per_station),
    )
    obs, obs_station_ids, obs_trace_mask = build_observation_windows(invdata, obs_args)

    stations_all, codes_all, _, _ = load_stations_from_xml(args.stations_dir)
    stations, station_ids = select_station_geometry(
        stations_all, codes_all, obs_station_ids
    )
    idx_map = [obs_station_ids.index(sid) for sid in station_ids]
    obs = obs[idx_map][:, comp_indices, :]
    obs_trace_mask = obs_trace_mask[idx_map][:, comp_indices]

    velocity_model = load_velocity_model(args.velocity_model)
    mt6 = load_mt6_from_results(Path(args.mt_results), args.solution)

    synth = FastSynthesizer(
        velocity_model,
        stations,
        source_loc,
        duration=float(args.duration),
        fmax=float(args.fmax),
        t0=1.0 / (np.pi * float(args.source_target_freq_hz)),
        work_dir=args.axitra_greens_dir,
        cache_id=1,
        generate_if_missing=False,
    )
    synth.setup()
    try:
        waveforms = synthesize_mt6(
            synth,
            mt6,
            m0=m0,
            source_delay=float(args.source_delay),
        )
        if args.synthetic_window_source == "picks":
            phase_cache = build_phase_window_cache_from_picks(
                invdata=invdata,
                station_ids=station_ids,
                duration=float(args.duration),
                npts=int(synth.ap.npt),
                p_window_len=float(args.synthetic_phase_window_p_len),
                s_window_len=float(args.synthetic_phase_window_s_len),
                time_steps=int(args.time_steps),
                p_bp_low=float(args.bp_p_low),
                p_bp_high=float(args.bp_p_high),
                s_bp_low=float(args.bp_s_low),
                s_bp_high=float(args.bp_s_high),
                bp_order=int(args.bp_order),
                normalize_per_station=bool(args.normalize_per_station),
            )
        else:
            phase_cache = build_phase_window_cache_dual(
                stations=stations,
                source_loc=source_loc,
                velocity_model=velocity_model,
                duration=float(args.duration),
                npts=int(synth.ap.npt),
                p_window_len=float(args.synthetic_phase_window_p_len),
                s_window_len=float(args.synthetic_phase_window_s_len),
                source_delay=float(args.source_delay),
                time_steps=int(args.time_steps),
                p_bp_low=float(args.bp_p_low),
                p_bp_high=float(args.bp_p_high),
                s_bp_low=float(args.bp_s_low),
                s_bp_high=float(args.bp_s_high),
                bp_order=int(args.bp_order),
                normalize_per_station=bool(args.normalize_per_station),
            )
        windowed = extract_phase_windows_batch_cached_dual(
            waveforms[np.newaxis, :, :, :], phase_cache
        )
        windowed = windowed[:, :, comp_indices, :]
        windowed = _apply_trace_mask_to_batch(windowed, obs_trace_mask)[0]
    finally:
        synth.cleanup()

    npts = waveforms.shape[-1]
    t = np.linspace(0.0, float(args.duration), npts, endpoint=False)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{invdata['event_id']}_mt_{args.solution}_axitra_f{float(args.source_target_freq_hz):g}"
    npz_path = out_dir / f"{stem}.npz"
    np.savez(
        npz_path,
        waveforms=waveforms,
        time=t,
        mt6=mt6,
        station_ids=np.array(station_ids, dtype=str),
        component_order=np.array(["Z", "N", "E"], dtype=str),
        m0=np.array(m0),
        source_target_freq_hz=np.array(float(args.source_target_freq_hz)),
        source_delay=np.array(float(args.source_delay)),
    )

    peak_summary = out_dir / f"{stem}_peaks.csv"
    write_peak_summary(peak_summary, station_ids, waveforms)

    for i, sid in enumerate(station_ids):
        plot_station(
            t,
            waveforms[i],
            sid,
            args.solution,
            mt6,
            out_dir / f"{sid}.mt_{args.solution}_axitra_f{float(args.source_target_freq_hz):g}.png",
        )

    overlay = (
        Path(args.overlay_plot)
        if args.overlay_plot is not None
        else out_dir / f"{stem}_phase_window_overlay.png"
    )
    syn_for_plot = windowed
    syn_label = f"MT {args.solution}"
    if args.plot_align_lag and int(args.lag_max_shift_samples) > 0:
        syn_for_plot, lag_samples, lag_corr = align_synthetics_for_plot(
            obs, windowed, max_shift=int(args.lag_max_shift_samples)
        )
        syn_label = f"MT {args.solution} (lag-shifted)"
        for j, tok in enumerate(comp_tokens):
            dt = (
                float(args.phase_window_p_len)
                if tok == "PZ"
                else float(args.phase_window_s_len)
            ) / float(max(1, int(args.time_steps) - 1))
            med_lag = int(np.median(lag_samples[:, j]))
            med_corr = float(np.median(lag_corr[:, j]))
            print(
                f"{tok}: median lag={med_lag:+d} samples "
                f"({med_lag * dt:+.4f} s), median corr={med_corr:.3f}"
            )
    plot_phase_window_overlay(
        obs,
        syn_for_plot,
        station_ids,
        overlay,
        p_len=float(args.phase_window_p_len),
        s_len=float(args.phase_window_s_len),
        comp_tokens=comp_tokens,
        comp_labels=comp_labels,
        include_mask=obs_trace_mask,
        syn_label=syn_label,
    )

    print(f"Solution: {args.solution}")
    print("MT6:", " ".join(f"{x:.8g}" for x in mt6))
    print(f"M0: {m0:.6e}")
    print(f"Axitra output shape: {waveforms.shape} (station, component, time)")
    print(f"Saved waveforms: {npz_path}")
    print(f"Saved peak summary: {peak_summary}")
    print(f"Saved station plots in: {out_dir}")
    print(f"Saved phase-window overlay: {overlay}")


if __name__ == "__main__":
    main()
