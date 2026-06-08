import argparse
import json
import re
from pathlib import Path
import sys

import numpy as np
from obspy import read

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FNAME_RE = re.compile(
    r"^.+__(?P<green_id>\d+)__(?P<station>st\d+)__(?P<component>[xyz])__(?P<basis>m(?:xx|yy|zz|xy|xz|yz))__\.sac$",
    re.IGNORECASE,
)

BASIS_ORDER = ["Mxx", "Myy", "Mzz", "Mxy", "Mxz", "Myz"]
BASIS_INDEX = {basis.lower(): i for i, basis in enumerate(BASIS_ORDER)}
COMPONENT_ORDER = ["Z", "N", "E"]
COMPONENT_INDEX = {"z": 0, "y": 1, "x": 2}


def _read_metadata(case_dir: Path, metadata_path: Path | None) -> dict:
    path = metadata_path if metadata_path is not None else case_dir / "station_order.json"
    if not path.exists():
        raise FileNotFoundError(f"Native green metadata not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("mode") != "openswpc_native_green":
        raise ValueError(f"{path} is not native-green metadata")
    return metadata


def _source_points(metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    points = metadata.get("source_grid", {}).get("points")
    if not points:
        source = metadata["source"]
        points = [
            {
                "green_id": 1,
                "x_m": float(source["x"]),
                "y_m": float(source["y"]),
                "z_m": float(source["z"]),
            }
        ]
    points = sorted(points, key=lambda item: int(item["green_id"]))
    source_ids = np.asarray([int(item["green_id"]) for item in points], dtype=np.int32)
    coords = np.asarray(
        [[float(item["x_m"]), float(item["y_m"]), float(item["z_m"])] for item in points],
        dtype=np.float64,
    )
    return source_ids, coords


def _case_sac_files(case_dir: Path, station_label: str, component: str) -> list[Path]:
    green_dir = case_dir / station_label / component / "out" / "green"
    if not green_dir.exists():
        raise FileNotFoundError(f"Missing native green output directory: {green_dir}")
    files = sorted(green_dir.rglob("*.sac"))
    if not files:
        raise RuntimeError(f"No SAC files found in {green_dir}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack native OpenSWPC green-mode grid SAC outputs into an NPZ library"
    )
    parser.add_argument("--case-dir", required=True, help="Native green case directory")
    parser.add_argument("--output", required=True, help="Output NPZ file path")
    parser.add_argument(
        "--station-metadata",
        default=None,
        help="Path to station_order.json; defaults to <case-dir>/station_order.json",
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    metadata_path = Path(args.station_metadata) if args.station_metadata else None
    metadata = _read_metadata(case_dir, metadata_path)

    stations = metadata.get("stations", [])
    if not stations:
        raise ValueError("Metadata does not list any stations")
    station_labels = [str(item["label"]).lower() for item in stations]
    station_codes = [str(item["station_id"]) for item in stations]
    station_coords = np.asarray([item["coord"] for item in stations], dtype=np.float64)
    source_ids, source_coords = _source_points(metadata)
    source_index = {int(green_id): i for i, green_id in enumerate(source_ids)}

    n_source = len(source_ids)
    n_station = len(station_labels)
    gf_basis = None
    dt_ref = None
    npts_ref = None
    seen: set[tuple[int, int, int, int]] = set()

    for station_i, station_label in enumerate(station_labels):
        for component in ("z", "y", "x"):
            for fp in _case_sac_files(case_dir, station_label, component):
                match = FNAME_RE.match(fp.name)
                if not match:
                    continue
                green_id = int(match.group("green_id"))
                if green_id not in source_index:
                    continue
                file_station = match.group("station").lower()
                file_component = match.group("component").lower()
                basis = match.group("basis").lower()
                if file_station != station_label or file_component != component:
                    continue
                source_i = source_index[green_id]
                basis_i = BASIS_INDEX[basis]
                component_i = COMPONENT_INDEX[file_component]

                tr = read(str(fp))[0]
                y = np.asarray(tr.data, dtype=np.float32)
                dt = float(tr.stats.delta)
                npts = int(tr.stats.npts)
                if gf_basis is None:
                    dt_ref = dt
                    npts_ref = npts
                    gf_basis = np.zeros(
                        (n_source, len(BASIS_ORDER), n_station, len(COMPONENT_ORDER), npts),
                        dtype=np.float32,
                    )
                else:
                    if abs(dt - dt_ref) > 1e-9:
                        raise ValueError(f"Inconsistent dt in {fp}: {dt} vs {dt_ref}")
                    if npts != npts_ref:
                        raise ValueError(f"Inconsistent npts in {fp}: {npts} vs {npts_ref}")

                key = (source_i, basis_i, station_i, component_i)
                if key in seen:
                    raise ValueError(f"Duplicate native green trace for {key}: {fp}")
                gf_basis[source_i, basis_i, station_i, component_i, :] = y
                seen.add(key)

    if gf_basis is None:
        raise RuntimeError(f"No matching native green SAC files found under {case_dir}")

    expected = n_source * len(BASIS_ORDER) * n_station * len(COMPONENT_ORDER)
    if len(seen) != expected:
        raise RuntimeError(f"Packed {len(seen)} traces, expected {expected}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        gf_basis=gf_basis,
        source_coords=source_coords,
        source_ids=source_ids,
        basis_order=np.asarray(BASIS_ORDER),
        station_labels=np.asarray(station_labels),
        station_codes=np.asarray(station_codes),
        station_coords=station_coords,
        dt=np.float32(dt_ref),
        npts=np.int32(npts_ref),
        quantity=np.asarray("V"),
        component_order=np.asarray(COMPONENT_ORDER),
    )
    print(f"Saved native green grid library: {out}")
    print(
        "Shape gf_basis: "
        f"{gf_basis.shape} (source, basis, station, component, time)"
    )


if __name__ == "__main__":
    main()
