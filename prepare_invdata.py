import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read, read_inventory
from scipy.signal import resample


EASTING_REF = 335697.57
NORTHING_REF = 4263390.77


def _parse_event_id(event_dir: Path, event_id_arg: str | None) -> str:
    if event_id_arg is not None:
        return str(event_id_arg)
    return event_dir.name


def _load_catalog_row(catalog_path: Path, event_id: str) -> dict:
    cat = pd.read_csv(catalog_path)
    id_col = "event_id" if "event_id" in cat.columns else cat.columns[0]
    row = cat[cat[id_col].astype(str) == str(event_id)]
    if row.empty:
        raise ValueError(f"Event {event_id} not found in {catalog_path}")
    r = row.iloc[0]
    easting = float(r["easting"])
    northing = float(r["northing"])
    depth_km = float(r["depth_km"])
    magnitude = float(r["magnitude"])
    return {
        "event_id": str(event_id),
        "easting": easting,
        "northing": northing,
        "depth_km": depth_km,
        "magnitude": magnitude,
        "x": easting - EASTING_REF,
        "y": northing - NORTHING_REF,
        "z": depth_km * 1000.0,
        "mw": magnitude,
    }


def _pick_phase_time(
    rows: pd.DataFrame, phase_names: tuple[str, ...], min_score: float
):
    sub = rows[rows["phase_type"].isin(phase_names)]
    if sub.empty:
        return None, None
    sub = sub[sub["phase_score"] >= min_score]
    if sub.empty:
        return None, None
    best = sub.sort_values("phase_score", ascending=False).iloc[0]
    return str(best["phase_time"]), float(best["phase_score"])


def _pick_phase_amp(
    rows: pd.DataFrame, phase_name: str, min_score: float
) -> tuple[float | None, float | None]:
    sub = rows[rows["phase_type"] == phase_name]
    if sub.empty:
        return None, None
    sub = sub[sub["phase_score"] >= min_score]
    if sub.empty:
        return None, None
    best = sub.sort_values("phase_score", ascending=False).iloc[0]
    amp = float(best.get("phase_amplitude", np.nan))
    if not np.isfinite(amp):
        return None, None
    return float(abs(amp)), float(best["phase_score"])


def _find_stationxml(stations_dir: Path, station_id: str) -> Path | None:
    p = stations_dir / f"{station_id}.xml"
    if p.exists():
        return p
    candidates = list(stations_dir.glob(f"{station_id}*.xml"))
    return candidates[0] if candidates else None


def _load_rotate_to_enz(mseed_path: Path, inv_path: Path, upsample_factor: float = 1.0):
    st = read(str(mseed_path))
    if len(st) == 0:
        raise ValueError(f"Empty stream: {mseed_path}")

    st = st.merge(method=1, fill_value="interpolate")
    t0 = max(tr.stats.starttime for tr in st)
    t1 = min(tr.stats.endtime for tr in st)
    if t1 <= t0:
        raise ValueError(f"No common time overlap in {mseed_path}")
    st = st.trim(t0, t1, pad=False)

    inv = read_inventory(str(inv_path))
    st = st.rotate(method="->ZNE", inventory=inv)

    z = st.select(channel="*Z")
    n = st.select(channel="*N")
    e = st.select(channel="*E")
    if len(z) == 0 or len(n) == 0 or len(e) == 0:
        raise ValueError(f"Missing Z/N/E after rotation for {mseed_path.name}")

    z = z[0]
    n = n[0]
    e = e[0]
    npts = min(len(z.data), len(n.data), len(e.data))
    if npts <= 0:
        raise ValueError(f"No samples after rotation for {mseed_path.name}")

    z_data = np.asarray(z.data[:npts], dtype=np.float32)
    n_data = np.asarray(n.data[:npts], dtype=np.float32)
    e_data = np.asarray(e.data[:npts], dtype=np.float32)

    fs = float(z.stats.sampling_rate)
    if upsample_factor > 1.0:
        n_up = max(npts + 1, int(round(npts * upsample_factor)))
        z_data = resample(z_data, n_up).astype(np.float32)
        n_data = resample(n_data, n_up).astype(np.float32)
        e_data = resample(e_data, n_up).astype(np.float32)
        npts = int(n_up)
        fs = fs * float(upsample_factor)

    return {
        "starttime": str(z.stats.starttime),
        "sampling_rate": fs,
        "delta": 1.0 / fs,
        "npts": int(npts),
        "components": {
            "E": e_data,
            "N": n_data,
            "Z": z_data,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare real-event inversion input from event folder"
    )
    parser.add_argument("--event-dir", type=str, required=True)
    parser.add_argument("--catalog", type=str, default="FORGE_catalog.csv")
    parser.add_argument("--stations-dir", type=str, default="stationxml")
    parser.add_argument("--event-id", type=str, default=None)
    parser.add_argument("--picks-file", type=str, default="picks.dat")
    parser.add_argument("--min-p-score", type=float, default=0.55)
    parser.add_argument("--min-s-score", type=float, default=0.55)
    parser.add_argument(
        "--upsample-factor",
        type=float,
        default=1.0,
        help="Upsample factor applied after rotation to ENZ (1.0 disables upsampling)",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    event_dir = Path(args.event_dir)
    if not event_dir.exists():
        raise FileNotFoundError(f"Event dir not found: {event_dir}")

    event_id = _parse_event_id(event_dir, args.event_id)
    picks_path = Path(args.picks_file)
    if not picks_path.is_absolute():
        picks_path = event_dir / picks_path
    if not picks_path.exists():
        raise FileNotFoundError(f"Picks file not found: {picks_path}")

    source = _load_catalog_row(Path(args.catalog), event_id)

    df = pd.read_csv(picks_path, sep="\t")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    if "event_index" in df.columns:
        df = df[df["event_index"].astype(str) == str(event_id)]
    if df.empty:
        raise ValueError(f"No picks found for event {event_id} in {picks_path}")

    stations_dir = Path(args.stations_dir)
    waveform_data = {}
    picks_data = {}
    used = []

    station_ids = sorted(df["station_id"].dropna().astype(str).unique().tolist())
    for sid in station_ids:
        rows = df[df["station_id"].astype(str) == sid]
        p_time, p_score = _pick_phase_time(rows, ("P", "Pz"), args.min_p_score)
        s_time, s_score = _pick_phase_time(rows, ("S", "Sh", "Sv"), args.min_s_score)
        if p_time is None or s_time is None:
            continue

        mseed_path = event_dir / f"{event_id}_{sid}.mseed"
        if not mseed_path.exists():
            candidates = list(event_dir.glob(f"*_{sid}.mseed"))
            if not candidates:
                continue
            mseed_path = candidates[0]

        inv_path = _find_stationxml(stations_dir, sid)
        if inv_path is None:
            continue

        try:
            wf = _load_rotate_to_enz(
                mseed_path,
                inv_path,
                upsample_factor=float(args.upsample_factor),
            )
            waveform_data[sid] = wf
            amp_pz, amp_pz_score = _pick_phase_amp(rows, "Pz", args.min_p_score)
            amp_sh, amp_sh_score = _pick_phase_amp(rows, "Sh", args.min_s_score)
            amp_sv, amp_sv_score = _pick_phase_amp(rows, "Sv", args.min_s_score)
            picks_data[sid] = {
                "P": str(UTCDateTime(p_time)),
                "S": str(UTCDateTime(s_time)),
                "p_score": float(p_score),
                "s_score": float(s_score),
                "amplitude": {
                    "Pz": amp_pz,
                    "Sh": amp_sh,
                    "Sv": amp_sv,
                    "Pz_score": amp_pz_score,
                    "Sh_score": amp_sh_score,
                    "Sv_score": amp_sv_score,
                },
            }
            used.append(sid)
        except Exception as exc:
            print(f"Skip {sid}: {exc}")

    if not used:
        raise RuntimeError(
            "No stations could be prepared from picks + waveforms + stationxml"
        )

    out_path = Path(args.out) if args.out else (event_dir / "invdata.pkl")
    payload = {
        "event_id": str(event_id),
        "source": source,
        "station_ids": sorted(used),
        "component_order": "ENZ",
        "waveforms": waveform_data,
        "picks": picks_data,
        "paths": {
            "event_dir": str(event_dir.resolve()),
            "catalog": str(Path(args.catalog).resolve()),
            "stations_dir": str(stations_dir.resolve()),
            "picks_file": str(picks_path.resolve()),
        },
        "preprocess_hint": {
            "upsample_factor": float(args.upsample_factor),
        },
    }

    with out_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Saved inversion input pickle: {out_path}")
    print(f"Prepared stations: {len(used)}")


if __name__ == "__main__":
    main()
