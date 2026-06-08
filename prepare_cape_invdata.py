import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read, read_inventory
from scipy.signal import resample


EASTING_REF = 335697.57
NORTHING_REF = 4263390.77


def _resolve_existing(path_text: str, fallback: str | None = None) -> Path:
    path = Path(path_text)
    if path.exists() or fallback is None:
        return path
    fb = Path(fallback)
    return fb if fb.exists() else path


def _load_catalog(catalog_path: Path) -> pd.DataFrame:
    cat = pd.read_csv(catalog_path)
    required = {"event_index", "Easting", "Northing", "depth", "ges_magnitude"}
    missing = sorted(required - set(cat.columns))
    if missing:
        raise ValueError(f"Catalog missing required columns: {missing}")
    cat["event_index"] = cat["event_index"].astype(str).str.strip()
    return cat


def _catalog_source(row: pd.Series) -> dict:
    easting = float(row["Easting"])
    northing = float(row["Northing"])
    depth_km = float(row["depth"])
    mw = float(row["ges_magnitude"])
    return {
        "event_id": str(row["event_index"]),
        "origin_time": str(row.get("origin_time", "")),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "easting": easting,
        "northing": northing,
        "depth_km": depth_km,
        "magnitude": mw,
        "mw": mw,
        "x": easting - EASTING_REF,
        "y": northing - NORTHING_REF,
        "z": depth_km * 1000.0,
    }


def _find_stationxml(stations_dir: Path, station_id: str) -> Path | None:
    exact = stations_dir / f"{station_id}.xml"
    if exact.exists():
        return exact
    matches = sorted(stations_dir.glob(f"{station_id}*.xml"))
    return matches[0] if matches else None


def _station_metadata(inv_path: Path) -> dict:
    inv = read_inventory(str(inv_path))
    for net in inv:
        for sta in net:
            if len(sta) == 0:
                continue
            cha = sta[0]
            return {
                "network": net.code,
                "station": sta.code,
                "latitude": float(cha.latitude),
                "longitude": float(cha.longitude),
                "depth_km": float(cha.depth) / 1000.0,
            }
    raise ValueError(f"No station channels found in {inv_path}")


def _parse_pre_filt(text: str) -> tuple[float, float, float, float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) != 4:
        raise ValueError("--response-pre-filt must contain four comma-separated values")
    if not (0.0 < vals[0] < vals[1] < vals[2] < vals[3]):
        raise ValueError("--response-pre-filt must satisfy f1 < f2 < f3 < f4")
    return tuple(vals)


def _load_rotate_to_enz(
    mseed_path: Path,
    inv_path: Path,
    upsample_factor: float = 1.0,
    remove_response: bool = True,
    response_output: str = "VEL",
    response_pre_filt: tuple[float, float, float, float] = (1.0, 2.0, 80.0, 95.0),
    response_water_level: float = 60.0,
):
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
    if remove_response:
        st = st.copy()
        st.detrend("linear")
        st.detrend("demean")
        st.remove_response(
            inventory=inv,
            output=response_output,
            pre_filt=response_pre_filt,
            water_level=float(response_water_level),
        )

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
        "components": {"E": e_data, "N": n_data, "Z": z_data},
    }


def _best_phase(rows: pd.DataFrame, phase_names: tuple[str, ...], min_score: float):
    sub = rows[rows["phase_type"].isin(phase_names)].copy()
    sub = sub[sub["phase_score"] >= float(min_score)]
    if sub.empty:
        return None
    return sub.sort_values("phase_score", ascending=False).iloc[0]


def _best_amp(rows: pd.DataFrame, phase_name: str, min_score: float):
    sub = rows[rows["phase_type"] == phase_name].copy()
    sub = sub[sub["phase_score"] >= float(min_score)]
    if sub.empty:
        return None, None, None
    best = sub.sort_values("phase_score", ascending=False).iloc[0]
    amp = float(best.get("phase_amplitude", np.nan))
    if not np.isfinite(amp):
        return None, None, None
    return float(abs(amp)), float(best["phase_score"]), float(best.get("phase_polarity", np.nan))


def _read_relevant_picks(
    picks_file: Path,
    event_ids: set[str],
    waveform_stations: dict[str, set[str]],
    chunksize: int = 500000,
) -> pd.DataFrame:
    rows = []
    for chunk in pd.read_csv(picks_file, chunksize=chunksize):
        chunk["event_index"] = chunk["event_index"].astype(str).str.strip()
        chunk["station_id"] = chunk["station_id"].astype(str).str.strip()
        mask = chunk["event_index"].isin(event_ids)
        if not mask.any():
            continue
        keep = np.zeros(len(chunk), dtype=bool)
        for event_id, stations in waveform_stations.items():
            keep |= (chunk["event_index"] == event_id).to_numpy() & chunk[
                "station_id"
            ].isin(stations).to_numpy()
        part = chunk[mask & keep].copy()
        if not part.empty:
            rows.append(part)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def _waveform_stations(waveform_root: Path, event_id: str) -> dict[str, Path]:
    event_wf_dir = waveform_root / event_id
    if not event_wf_dir.exists():
        raise FileNotFoundError(f"Waveform event directory not found: {event_wf_dir}")
    out = {}
    for mseed in sorted(event_wf_dir.glob(f"{event_id}_*.mseed")):
        sid = mseed.name[: -len(".mseed")].replace(f"{event_id}_", "", 1)
        out[sid] = mseed
    return out


def _write_picks_dat(path: Path, rows: pd.DataFrame, source: dict, station_meta: dict):
    enriched = rows.copy()
    enriched["latitude_pick"] = float(source["latitude"])
    enriched["longitude_pick"] = float(source["longitude"])
    enriched["depth_km_pick"] = float(source["depth_km"])
    enriched["latitude_station"] = enriched["station_id"].map(
        lambda sid: station_meta[sid]["latitude"]
    )
    enriched["longitude_station"] = enriched["station_id"].map(
        lambda sid: station_meta[sid]["longitude"]
    )
    enriched["depth_km_station"] = enriched["station_id"].map(
        lambda sid: station_meta[sid]["depth_km"]
    )
    enriched.to_csv(path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description="Prepare CAPE event inversion inputs")
    parser.add_argument("--catalog", default="FebMarch_catalog.csv")
    parser.add_argument("--waveform-root", default="waveforms")
    parser.add_argument("--stations-dir", default="stations")
    parser.add_argument("--picks-file", default="picks_amp.csv")
    parser.add_argument("--output-root", default="cape_events")
    parser.add_argument("--min-p-score", type=float, default=0.55)
    parser.add_argument("--min-s-score", type=float, default=0.35)
    parser.add_argument("--upsample-factor", type=float, default=1.0)
    parser.add_argument(
        "--no-remove-response",
        action="store_true",
        help="Keep waveform samples as raw counts instead of removing StationXML response",
    )
    parser.add_argument(
        "--response-output",
        default="VEL",
        choices=("DISP", "VEL", "ACC"),
        help="ObsPy remove_response output unit",
    )
    parser.add_argument(
        "--response-pre-filt",
        default="1.0,2.0,80.0,95.0",
        help="Four comma-separated frequency corners for response removal",
    )
    parser.add_argument("--response-water-level", type=float, default=60.0)
    args = parser.parse_args()
    response_pre_filt = _parse_pre_filt(args.response_pre_filt)

    catalog = _resolve_existing(
        args.catalog, f"bayesian_waveform_fitting/{args.catalog}"
    )
    waveform_root = _resolve_existing(
        args.waveform_root, f"bayesian_waveform_fitting/{args.waveform_root}"
    )
    stations_dir = _resolve_existing(
        args.stations_dir, f"bayesian_waveform_fitting/{args.stations_dir}"
    )
    picks_fallback = Path(f"bayesian_waveform_fitting/{args.picks_file}")
    picks_file = _resolve_existing(args.picks_file, str(picks_fallback))
    output_root = Path(args.output_root)

    cat = _load_catalog(catalog)
    event_ids = set(cat["event_index"].astype(str))
    waveform_paths = {
        event_id: _waveform_stations(waveform_root, event_id) for event_id in event_ids
    }
    waveform_station_ids = {
        event_id: set(paths.keys()) for event_id, paths in waveform_paths.items()
    }
    picks = _read_relevant_picks(picks_file, event_ids, waveform_station_ids)
    if picks.empty and picks_fallback.exists() and picks_fallback.resolve() != picks_file.resolve():
        picks_file = picks_fallback
        picks = _read_relevant_picks(picks_file, event_ids, waveform_station_ids)
    if picks.empty:
        raise RuntimeError("No picks matched the catalog events and waveform stations")

    output_root.mkdir(parents=True, exist_ok=True)
    for _, row in cat.sort_values("event_index").iterrows():
        event_id = str(row["event_index"])
        source = _catalog_source(row)
        event_dir = output_root / event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        event_picks = picks[picks["event_index"] == event_id].copy()
        waveforms = {}
        picks_data = {}
        station_meta = {}
        used = []
        p_count = 0
        p_threshold_count = 0
        s_count = 0

        for sid, mseed_path in sorted(waveform_paths[event_id].items()):
            rows = event_picks[event_picks["station_id"] == sid]
            if rows.empty:
                continue
            p_pick = _best_phase(rows, ("P", "Pz"), args.min_p_score)
            p_meets_threshold = p_pick is not None
            if p_pick is None:
                p_pick = _best_phase(rows, ("P", "Pz"), -np.inf)
            if p_pick is None:
                continue
            s_pick = _best_phase(rows, ("S", "Sh", "Sv"), args.min_s_score)

            inv_path = _find_stationxml(stations_dir, sid)
            if inv_path is None:
                print(f"Skip {event_id}/{sid}: station XML not found")
                continue

            try:
                wf = _load_rotate_to_enz(
                    mseed_path,
                    inv_path,
                    upsample_factor=float(args.upsample_factor),
                    remove_response=not bool(args.no_remove_response),
                    response_output=str(args.response_output),
                    response_pre_filt=response_pre_filt,
                    response_water_level=float(args.response_water_level),
                )
                meta = _station_metadata(inv_path)
            except Exception as exc:
                print(f"Skip {event_id}/{sid}: {exc}")
                continue

            amp_pz, amp_pz_score, pol_pz = _best_amp(rows, "Pz", args.min_p_score)
            amp_sh, amp_sh_score, pol_sh = _best_amp(rows, "Sh", args.min_s_score)
            amp_sv, amp_sv_score, pol_sv = _best_amp(rows, "Sv", args.min_s_score)

            waveforms[sid] = wf
            station_meta[sid] = meta
            picks_data[sid] = {
                "P": str(UTCDateTime(p_pick["phase_time"])),
                "S": None if s_pick is None else str(UTCDateTime(s_pick["phase_time"])),
                "p_score": float(p_pick["phase_score"]),
                "p_meets_threshold": bool(p_meets_threshold),
                "s_score": None if s_pick is None else float(s_pick["phase_score"]),
                "amplitude": {
                    "Pz": amp_pz,
                    "Sh": amp_sh,
                    "Sv": amp_sv,
                    "Pz_score": amp_pz_score,
                    "Sh_score": amp_sh_score,
                    "Sv_score": amp_sv_score,
                    "Pz_polarity": pol_pz,
                    "Sh_polarity": pol_sh,
                    "Sv_polarity": pol_sv,
                },
            }
            used.append(sid)
            p_count += 1
            if p_meets_threshold:
                p_threshold_count += 1
            if s_pick is not None:
                s_count += 1

        if not used:
            raise RuntimeError(f"No stations prepared for {event_id}")

        used = sorted(used)
        payload = {
            "event_id": event_id,
            "source": source,
            "station_ids": used,
            "component_order": "ENZ",
            "waveforms": {sid: waveforms[sid] for sid in used},
            "picks": {sid: picks_data[sid] for sid in used},
            "station_metadata": {sid: station_meta[sid] for sid in used},
            "paths": {
                "event_dir": str(event_dir.resolve()),
                "catalog": str(catalog.resolve()),
                "waveform_root": str(waveform_root.resolve()),
                "stations_dir": str(stations_dir.resolve()),
                "picks_file": str(picks_file.resolve()),
            },
            "preprocess_hint": {
                "upsample_factor": float(args.upsample_factor),
                "remove_response": not bool(args.no_remove_response),
                "response_output": str(args.response_output),
                "response_pre_filt": response_pre_filt,
                "response_water_level": float(args.response_water_level),
            },
        }

        with (event_dir / "invdata.pkl").open("wb") as f:
            pickle.dump(payload, f)

        picks_out = event_picks[event_picks["station_id"].isin(used)].copy()
        _write_picks_dat(event_dir / "picks.dat", picks_out, source, station_meta)

        n_missing_s = len(used) - s_count
        print(f"Saved {event_dir / 'invdata.pkl'}")
        print(
            f"  stations={len(used)}, P={p_count} ({p_threshold_count} >= threshold), "
            f"S={s_count}, "
            f"masked_missing_S_traces={2 * n_missing_s}"
        )


if __name__ == "__main__":
    main()
