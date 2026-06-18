"""Prepare Grond's regional CMT example for the local inversion workflow.

The Grond example is a regional event with ordinary geographic coordinates.
The current Axitra workflow in this repository expects event-centered metric
coordinates and reads station geometry through the CAPE StationXML loader.
This script keeps the waveform data and original station metadata from Grond,
but writes a small StationXML geometry directory whose coordinates map through
the existing loader into event-centered east/north offsets.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pyproj
from obspy import UTCDateTime, read, read_inventory
from obspy.core.inventory import Channel, Inventory, Network, Site, Station
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.taup import TauPyModel


EASTING_REF = 335697.57
NORTHING_REF = 4263390.77


def _parse_event(path: Path) -> dict:
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("-") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        data[key] = value

    required = {"name", "time", "latitude", "longitude", "magnitude", "depth"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Event file missing required fields: {missing}")

    depth_m = float(data["depth"])
    source = {
        "event_id": data["name"],
        "origin_time": str(UTCDateTime(data["time"])),
        "latitude": float(data["latitude"]),
        "longitude": float(data["longitude"]),
        "depth_km": depth_m / 1000.0,
        "magnitude": float(data["magnitude"]),
        "mw": float(data["magnitude"]),
        "x": 0.0,
        "y": 0.0,
        "z": depth_m,
    }

    for key in ("moment", "mnn", "mee", "mdd", "mne", "mnd", "med"):
        if key in data:
            source[key] = float(data[key])
    for key in ("strike1", "dip1", "rake1", "strike2", "dip2", "rake2"):
        if key in data:
            source[key] = float(data[key])

    return source


def _station_groups(raw_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in sorted(raw_dir.glob("*.mseed")):
        parts = path.name.split("_")
        if len(parts) < 2:
            continue
        nslc = parts[1].split(".")
        if len(nslc) < 4:
            continue
        net, sta, loc, cha = nslc[:4]
        band = cha[:2]
        station_id = f"{net}.{sta}.{loc}.{band}"
        groups.setdefault(station_id, []).append(path)
    return groups


def _inventory_station_metadata(inv, station_id: str) -> dict:
    net_code, sta_code, loc_code, _ = station_id.split(".")
    for net in inv:
        if net.code != net_code:
            continue
        for sta in net:
            if sta.code != sta_code:
                continue
            channels = [cha for cha in sta if cha.location_code == loc_code]
            if not channels:
                channels = list(sta)
            if not channels:
                break
            cha = channels[0]
            return {
                "network": net.code,
                "station": sta.code,
                "location": cha.location_code,
                "latitude": float(cha.latitude),
                "longitude": float(cha.longitude),
                "depth_km": float(cha.depth) / 1000.0,
                "elevation_m": float(cha.elevation),
            }
    raise ValueError(f"Station {station_id} not found in inventory")


def _local_offsets_m(source: dict, station_meta: dict) -> tuple[float, float, float]:
    dist_m, azimuth, _ = gps2dist_azimuth(
        float(source["latitude"]),
        float(source["longitude"]),
        float(station_meta["latitude"]),
        float(station_meta["longitude"]),
    )
    azimuth_rad = np.deg2rad(float(azimuth))
    east_m = float(dist_m * np.sin(azimuth_rad))
    north_m = float(dist_m * np.cos(azimuth_rad))
    depth_m = float(station_meta["depth_km"]) * 1000.0
    return east_m, north_m, depth_m


def _write_geometry_xml(
    out_dir: Path,
    station_id: str,
    local_xyz: tuple[float, float, float],
    original_meta: dict,
) -> None:
    net_code, sta_code, loc_code, band_code = station_id.split(".")
    transformer = pyproj.Transformer.from_crs(
        "EPSG:26912", "EPSG:4326", always_xy=True
    )
    east_m, north_m, depth_m = local_xyz
    lon, lat = transformer.transform(EASTING_REF + east_m, NORTHING_REF + north_m)

    channels = []
    for suffix, azimuth, dip in (("E", 90.0, 0.0), ("N", 0.0, 0.0), ("Z", 0.0, -90.0)):
        channels.append(
            Channel(
                code=f"{band_code}{suffix}",
                location_code=loc_code,
                latitude=float(lat),
                longitude=float(lon),
                elevation=float(original_meta.get("elevation_m", 0.0)),
                depth=float(depth_m),
                azimuth=azimuth,
                dip=dip,
                sample_rate=1.0,
            )
        )

    inv = Inventory(
        networks=[
            Network(
                code=net_code,
                stations=[
                    Station(
                        code=sta_code,
                        latitude=float(lat),
                        longitude=float(lon),
                        elevation=float(original_meta.get("elevation_m", 0.0)),
                        site=Site(name=sta_code),
                        channels=channels,
                    )
                ],
            )
        ],
        source="prepare_grond_regional_invdata.py",
    )
    inv.write(str(out_dir / f"{station_id}.xml"), format="STATIONXML")


def _load_velocity_enz(
    mseed_paths: list[Path],
    inventory_path: Path,
    response_pre_filt: tuple[float, float, float, float],
    response_water_level: float,
) -> dict:
    stream = None
    for path in mseed_paths:
        part = read(str(path))
        stream = part if stream is None else stream + part
    if stream is None or len(stream) == 0:
        raise ValueError("Empty waveform stream")

    stream.merge(method=1, fill_value="interpolate")
    t0 = max(tr.stats.starttime for tr in stream)
    t1 = min(tr.stats.endtime for tr in stream)
    if t1 <= t0:
        raise ValueError("No common waveform overlap")
    stream.trim(t0, t1, pad=False)

    inv = read_inventory(str(inventory_path))
    stream = stream.copy()
    stream.detrend("linear")
    stream.detrend("demean")
    stream.remove_response(
        inventory=inv,
        output="VEL",
        pre_filt=response_pre_filt,
        water_level=float(response_water_level),
    )
    stream.rotate(method="->ZNE", inventory=inv)

    traces = {
        "Z": stream.select(channel="*Z"),
        "N": stream.select(channel="*N"),
        "E": stream.select(channel="*E"),
    }
    if any(len(value) == 0 for value in traces.values()):
        raise ValueError("Missing Z/N/E after response removal and rotation")

    z = traces["Z"][0]
    n = traces["N"][0]
    e = traces["E"][0]
    npts = min(len(z.data), len(n.data), len(e.data))
    if npts <= 0:
        raise ValueError("No waveform samples after trimming")

    fs = float(z.stats.sampling_rate)
    return {
        "starttime": str(z.stats.starttime),
        "sampling_rate": fs,
        "delta": 1.0 / fs,
        "npts": int(npts),
        "components": {
            "E": np.asarray(e.data[:npts], dtype=np.float32),
            "N": np.asarray(n.data[:npts], dtype=np.float32),
            "Z": np.asarray(z.data[:npts], dtype=np.float32),
        },
    }


def _theoretical_picks(source: dict, station_meta: dict, model: TauPyModel) -> dict:
    origin = UTCDateTime(source["origin_time"])
    distance_deg = locations2degrees(
        float(source["latitude"]),
        float(source["longitude"]),
        float(station_meta["latitude"]),
        float(station_meta["longitude"]),
    )
    depth_km = float(source["depth_km"])

    def first_arrival(phases: list[str]) -> float:
        arrivals = model.get_travel_times(
            source_depth_in_km=depth_km,
            distance_in_degree=float(distance_deg),
            phase_list=phases,
        )
        if not arrivals:
            raise ValueError(f"No theoretical arrival for phases {phases}")
        return float(arrivals[0].time)

    p_offset = first_arrival(["p", "P", "Pn", "Pg"])
    s_offset = first_arrival(["s", "S", "Sn", "Sg"])
    return {
        "P": str(origin + p_offset),
        "S": str(origin + s_offset),
        "p_score": 1.0,
        "p_meets_threshold": True,
        "s_score": 1.0,
        "pick_source": "iasp91_theoretical",
    }


def _parse_pre_filt(text: str) -> tuple[float, float, float, float]:
    values = tuple(float(v.strip()) for v in text.split(",") if v.strip())
    if len(values) != 4:
        raise ValueError("--response-pre-filt requires four comma-separated values")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Grond example_regional_cmt data to invdata.pkl"
    )
    parser.add_argument(
        "--grond-event-dir",
        default="grond/src/data/examples/example_regional_cmt/data/events/gfz2018pmjk",
    )
    parser.add_argument("--output-root", default="cape_events")
    parser.add_argument("--stations-out", default="stations_grond_regional")
    parser.add_argument("--event-id", default=None)
    parser.add_argument("--max-stations", type=int, default=4)
    parser.add_argument(
        "--exclude",
        default="GE.UGM,GE.PLAI",
        help="Comma-separated NET.STA codes to skip; Grond's config excludes GE.UGM and GE.PLAI.",
    )
    parser.add_argument("--response-pre-filt", default="0.005,0.01,0.4,0.8")
    parser.add_argument("--response-water-level", type=float, default=60.0)
    args = parser.parse_args()

    grond_event_dir = Path(args.grond_event_dir)
    event_path = grond_event_dir / "event.txt"
    waveform_dir = grond_event_dir / "waveforms"
    raw_dir = waveform_dir / "raw"
    inventory_path = waveform_dir / "stations.geofon.xml"
    if not event_path.exists():
        raise FileNotFoundError(event_path)
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)
    if not inventory_path.exists():
        raise FileNotFoundError(inventory_path)

    source = _parse_event(event_path)
    event_id = str(args.event_id or source["event_id"])
    source["event_id"] = event_id
    response_pre_filt = _parse_pre_filt(args.response_pre_filt)
    exclude = {item.strip() for item in args.exclude.split(",") if item.strip()}

    inventory = read_inventory(str(inventory_path))
    taup_model = TauPyModel(model="iasp91")
    station_groups = _station_groups(raw_dir)
    if not station_groups:
        raise RuntimeError(f"No MiniSEED station groups found in {raw_dir}")

    out_event_dir = Path(args.output_root) / event_id
    stations_out = Path(args.stations_out)
    out_event_dir.mkdir(parents=True, exist_ok=True)
    stations_out.mkdir(parents=True, exist_ok=True)

    waveforms = {}
    picks = {}
    station_metadata = {}
    station_geometry = {}
    used = []

    for station_id, mseed_paths in sorted(station_groups.items()):
        net_code, sta_code, _, _ = station_id.split(".")
        if f"{net_code}.{sta_code}" in exclude:
            continue
        if args.max_stations > 0 and len(used) >= int(args.max_stations):
            break

        try:
            meta = _inventory_station_metadata(inventory, station_id)
            local_xyz = _local_offsets_m(source, meta)
            waveform = _load_velocity_enz(
                mseed_paths,
                inventory_path,
                response_pre_filt=response_pre_filt,
                response_water_level=float(args.response_water_level),
            )
            pick_data = _theoretical_picks(source, meta, taup_model)
            _write_geometry_xml(stations_out, station_id, local_xyz, meta)
        except Exception as exc:
            print(f"Skip {station_id}: {exc}")
            continue

        waveforms[station_id] = waveform
        picks[station_id] = pick_data
        station_metadata[station_id] = meta
        station_geometry[station_id] = {
            "x": local_xyz[0],
            "y": local_xyz[1],
            "z": local_xyz[2],
        }
        used.append(station_id)

    if not used:
        raise RuntimeError("No stations could be converted")

    payload = {
        "event_id": event_id,
        "source": source,
        "station_ids": used,
        "component_order": "ENZ",
        "waveforms": {sid: waveforms[sid] for sid in used},
        "picks": {sid: picks[sid] for sid in used},
        "station_metadata": {sid: station_metadata[sid] for sid in used},
        "station_geometry": {sid: station_geometry[sid] for sid in used},
        "paths": {
            "grond_event_dir": str(grond_event_dir.resolve()),
            "raw_waveforms": str(raw_dir.resolve()),
            "original_stationxml": str(inventory_path.resolve()),
            "stations_dir": str(stations_out.resolve()),
        },
        "preprocess_hint": {
            "remove_response": True,
            "response_output": "VEL",
            "response_pre_filt": response_pre_filt,
            "response_water_level": float(args.response_water_level),
            "upsample_factor": 1.0,
            "pick_source": "iasp91_theoretical",
            "geometry_note": "Station XML coordinates are synthetic local geometry for the CAPE loader.",
        },
    }

    with (out_event_dir / "invdata.pkl").open("wb") as f:
        pickle.dump(payload, f)

    print(f"Saved {out_event_dir / 'invdata.pkl'}")
    print(f"Saved station geometry XMLs in {stations_out}")
    print(f"Prepared stations: {len(used)}")
    for sid in used:
        geom = station_geometry[sid]
        print(f"  {sid}: x={geom['x']:.1f} m y={geom['y']:.1f} m")


if __name__ == "__main__":
    main()
