import argparse
import pickle
from pathlib import Path

import numpy as np

from run_cape_inversion import CAPE_DEFAULTS, validate_event_inputs


def default_event_dir(event_id: str) -> Path:
    return Path("cape_events") / event_id


def build_synthesizer(args: argparse.Namespace, event_dir: Path):
    from run_inversion import select_station_geometry
    from src_smc_mti.forward import FastSynthesizer
    from src_smc_mti.io import load_stations_from_xml, load_velocity_model

    invdata = validate_event_inputs(event_dir)
    source = invdata["source"]
    source_loc = (float(source["x"]), float(source["y"]), float(source["z"]))
    if args.source_target_freq_hz <= 0:
        raise ValueError("--source-target-freq-hz must be > 0")

    stations_all, codes_all, _, _ = load_stations_from_xml(args.stations_dir)
    stations_sel, station_ids = select_station_geometry(
        stations_all,
        codes_all,
        [str(sid) for sid in invdata["station_ids"]],
    )
    if len(station_ids) != len(invdata["station_ids"]):
        missing = sorted(
            set(str(sid) for sid in invdata["station_ids"]) - set(station_ids)
        )
        raise RuntimeError(f"Missing station geometry for invdata stations: {missing}")

    velocity_model = load_velocity_model(args.velocity_model)
    out_dir = Path(args.out_dir) if args.out_dir else event_dir / "axitra_greens"
    return FastSynthesizer(
        velocity_model,
        stations_sel,
        source_loc,
        duration=float(args.duration),
        fmax=float(args.fmax),
        t0=1.0 / (np.pi * float(args.source_target_freq_hz)),
        work_dir=out_dir,
        cache_id=1,
        generate_if_missing=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or refresh persistent Axitra Green's functions for a CAPE event"
    )
    parser.add_argument("event_id", choices=("eq02387", "eq02883"))
    parser.add_argument("--event-dir", default=None)
    parser.add_argument("--stations-dir", default="stations")
    parser.add_argument("--velocity-model", default="cape.tvel")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <event-dir>/axitra_greens.",
    )
    parser.add_argument("--duration", type=float, default=CAPE_DEFAULTS["duration"])
    parser.add_argument("--fmax", type=float, default=CAPE_DEFAULTS["fmax"])
    parser.add_argument("--source-target-freq-hz", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    event_dir = Path(args.event_dir) if args.event_dir else default_event_dir(args.event_id)
    out_dir = Path(args.out_dir) if args.out_dir else event_dir / "axitra_greens"
    inv_path = event_dir / "invdata.pkl"
    if not inv_path.exists():
        raise FileNotFoundError(f"Missing CAPE invdata: {inv_path}")

    with inv_path.open("rb") as f:
        invdata = pickle.load(f)

    print(f"event: {args.event_id}")
    print(f"event_dir: {event_dir}")
    print(f"invdata: {inv_path}")
    print(f"stations: {len(invdata.get('station_ids', []))}")
    print(f"velocity_model: {args.velocity_model}")
    print(f"out_dir: {out_dir}")
    print(f"duration: {float(args.duration):g} s")
    print(f"fmax: {float(args.fmax):g} Hz")
    print(f"source_target_freq_hz: {float(args.source_target_freq_hz):g} Hz")
    if args.dry_run:
        return

    synthesizer = build_synthesizer(args, event_dir)
    try:
        synthesizer.setup()
    finally:
        synthesizer.cleanup()
    print(f"Axitra Green's functions are ready in {out_dir}")


if __name__ == "__main__":
    main()
