import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from obspy import UTCDateTime, read
from scipy.signal import butter, correlate, correlation_lags, sosfiltfilt


SAC_TO_COMPONENT = {"Vz": "Z", "Vy": "N", "Vx": "E"}


def _bandpass(x, fs, low, high):
    nyq = 0.5 * fs
    lo = max(low / nyq, 1e-5)
    hi = min(high / nyq, 0.999)
    if not 0 < lo < hi < 1:
        return np.asarray(x, dtype=float)
    sos = butter(3, [lo, hi], btype="bandpass", output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=float))


def _window(t, y, t0, before, after):
    mask = (t >= t0 - before) & (t <= t0 + after)
    return t[mask], y[mask]


def _peak_time(t, y):
    if t.size == 0 or np.max(np.abs(y)) <= 0.0:
        return np.nan
    return float(t[int(np.argmax(np.abs(y)))])


def _interp_window(obs_t, obs_y, syn_t, syn_y, t0, before, after):
    t, syn = _window(syn_t, syn_y, t0, before, after)
    if t.size < 5:
        return t, np.array([]), np.array([])
    obs = np.interp(t, obs_t, obs_y, left=0.0, right=0.0)
    return t, obs, syn


def _max_abs_corr(obs, syn):
    if obs.size < 5 or syn.size < 5:
        return np.nan, np.nan
    obs = obs - np.mean(obs)
    syn = syn - np.mean(syn)
    denom = np.linalg.norm(obs) * np.linalg.norm(syn)
    if denom <= 0.0:
        return np.nan, np.nan
    c = correlate(obs, syn, mode="full") / denom
    lags = correlation_lags(obs.size, syn.size, mode="full")
    idx = int(np.argmax(np.abs(c)))
    return float(c[idx]), int(lags[idx])


def _load_synthetics(case_dir, label, filter_low, filter_high):
    out = {}
    for sac_comp, comp in SAC_TO_COMPONENT.items():
        sac_path = next((case_dir / "out" / "wav").glob(f"*.{label}.{sac_comp}.sac"), None)
        if sac_path is None:
            continue
        tr = read(str(sac_path))[0]
        t = np.arange(tr.stats.npts) * float(tr.stats.delta)
        y = _bandpass(tr.data.astype(float), tr.stats.sampling_rate, filter_low, filter_high)
        out[comp] = (t, y)
    return out


def _horizontal(tn, yn, te, ye):
    if tn.size <= te.size:
        t = tn
        e = np.interp(t, te, ye, left=0.0, right=0.0)
        n = yn
    else:
        t = te
        n = np.interp(t, tn, yn, left=0.0, right=0.0)
        e = ye
    return t, np.sqrt(n * n + e * e)


def main():
    parser = argparse.ArgumentParser(
        description="Score OpenSWPC LHM synthetic timing against observed CAPE picks."
    )
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--invdata", default=None)
    parser.add_argument("--filter-low", type=float, default=0.5)
    parser.add_argument("--filter-high", type=float, default=8.0)
    parser.add_argument("--p-before", type=float, default=0.45)
    parser.add_argument("--p-after", type=float, default=0.55)
    parser.add_argument("--s-before", type=float, default=0.55)
    parser.add_argument("--s-after", type=float, default=0.75)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    metadata = json.loads((case_dir / "case_metadata.json").read_text(encoding="utf-8"))
    inv_path = Path(args.invdata) if args.invdata else case_dir.parent / "invdata.pkl"
    with inv_path.open("rb") as f:
        invdata = pickle.load(f)

    origin = UTCDateTime(invdata["source"]["origin_time"])
    rows = []
    p_lags = []
    s_lags = []
    corrs = []

    for station in metadata["stations"]:
        sid = station["station_id"]
        label = station["label"]
        obs = invdata["waveforms"][sid]
        picks = invdata.get("picks", {}).get(sid, {})
        syn = _load_synthetics(case_dir, label, args.filter_low, args.filter_high)
        if set(["Z", "N", "E"]) - set(syn):
            continue

        obs_series = {}
        for comp in ["Z", "N", "E"]:
            y = np.asarray(obs["components"][comp], dtype=float)
            fs = float(obs["sampling_rate"])
            t0 = float(UTCDateTime(obs["starttime"]) - origin)
            t = t0 + np.arange(y.size) / fs
            obs_series[comp] = (t, _bandpass(y, fs, args.filter_low, args.filter_high))

        row = {"station_id": sid}
        if picks.get("P"):
            p_pick = float(UTCDateTime(picks["P"]) - origin)
            syn_t, syn_z = syn["Z"]
            obs_t, obs_z = obs_series["Z"]
            st, sy = _window(syn_t, syn_z, p_pick, args.p_before, args.p_after)
            ot, oy = _window(obs_t, obs_z, p_pick, args.p_before, args.p_after)
            syn_peak = _peak_time(st, sy)
            obs_peak = _peak_time(ot, oy)
            _, obs_i, syn_i = _interp_window(
                obs_t, obs_z, syn_t, syn_z, p_pick, args.p_before, args.p_after
            )
            corr, lag_samples = _max_abs_corr(obs_i, syn_i)
            lag_dt = lag_samples * (syn_t[1] - syn_t[0]) if np.isfinite(corr) else np.nan
            row.update(
                {
                    "p_pick": p_pick,
                    "p_syn_peak": syn_peak,
                    "p_obs_peak": obs_peak,
                    "p_pick_lag": syn_peak - p_pick if np.isfinite(syn_peak) else np.nan,
                    "p_obs_peak_lag": syn_peak - obs_peak
                    if np.isfinite(syn_peak) and np.isfinite(obs_peak)
                    else np.nan,
                    "p_corr": corr,
                    "p_corr_lag": lag_dt,
                }
            )
            if np.isfinite(row["p_pick_lag"]):
                p_lags.append(abs(row["p_pick_lag"]))
            if np.isfinite(corr):
                corrs.append(abs(corr))

        if picks.get("S"):
            s_pick = float(UTCDateTime(picks["S"]) - origin)
            syn_h_t, syn_h = _horizontal(*syn["N"], *syn["E"])
            obs_h_t, obs_h = _horizontal(*obs_series["N"], *obs_series["E"])
            st, sy = _window(syn_h_t, syn_h, s_pick, args.s_before, args.s_after)
            ot, oy = _window(obs_h_t, obs_h, s_pick, args.s_before, args.s_after)
            syn_peak = _peak_time(st, sy)
            obs_peak = _peak_time(ot, oy)
            _, obs_i, syn_i = _interp_window(
                obs_h_t, obs_h, syn_h_t, syn_h, s_pick, args.s_before, args.s_after
            )
            corr, lag_samples = _max_abs_corr(obs_i, syn_i)
            lag_dt = lag_samples * (syn_h_t[1] - syn_h_t[0]) if np.isfinite(corr) else np.nan
            row.update(
                {
                    "s_pick": s_pick,
                    "s_syn_peak": syn_peak,
                    "s_obs_peak": obs_peak,
                    "s_pick_lag": syn_peak - s_pick if np.isfinite(syn_peak) else np.nan,
                    "s_obs_peak_lag": syn_peak - obs_peak
                    if np.isfinite(syn_peak) and np.isfinite(obs_peak)
                    else np.nan,
                    "s_corr": corr,
                    "s_corr_lag": lag_dt,
                }
            )
            if np.isfinite(row["s_pick_lag"]):
                s_lags.append(abs(row["s_pick_lag"]))
            if np.isfinite(corr):
                corrs.append(abs(corr))

        rows.append(row)

    if args.output_csv:
        fields = sorted({key for row in rows for key in row})
        with Path(args.output_csv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    p_mean = float(np.mean(p_lags)) if p_lags else np.nan
    s_mean = float(np.mean(s_lags)) if s_lags else np.nan
    corr_mean = float(np.mean(corrs)) if corrs else np.nan
    score = np.nanmean([p_mean, s_mean, 1.0 - corr_mean if np.isfinite(corr_mean) else np.nan])
    print(f"case={case_dir}")
    print(f"stations_scored={len(rows)}")
    print(f"mean_abs_p_pick_lag={p_mean:.4f}s")
    print(f"mean_abs_s_pick_lag={s_mean:.4f}s")
    print(f"mean_abs_corr={corr_mean:.4f}")
    print(f"summary_score={score:.4f}")
    for row in rows:
        print(
            row["station_id"],
            f"P_lag={row.get('p_pick_lag', np.nan): .3f}",
            f"S_lag={row.get('s_pick_lag', np.nan): .3f}",
            f"P_corr={row.get('p_corr', np.nan): .3f}",
            f"S_corr={row.get('s_corr', np.nan): .3f}",
        )


if __name__ == "__main__":
    main()
