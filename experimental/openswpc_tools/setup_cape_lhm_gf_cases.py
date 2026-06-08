import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experimental.openswpc_tools.run_cape_lhm_forward import (
    DEFAULT_LHM,
    _build_input_text,
    _grid_for_case,
    _select_stations,
)


BASIS_ORDER = ["Mxx", "Myy", "Mzz", "Mxy", "Mxz", "Myz"]

# OpenSWPC xym0ij source-file order is mxx myy mzz myz mxz mxy.
BASIS_TO_SOURCE = {
    "Mxx": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "Myy": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "Mzz": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "Mxy": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    "Mxz": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    "Myz": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
}


def _basis_list(text: str) -> list[str]:
    if text.lower() == "all":
        return list(BASIS_ORDER)
    basis = [b.strip() for b in text.split(",") if b.strip()]
    bad = [b for b in basis if b not in BASIS_TO_SOURCE]
    if bad:
        raise ValueError(f"Unknown basis values: {bad}")
    return basis


def _write_station_file(out_dir: Path, stations: list[dict]) -> list[dict]:
    station_meta = []
    with (out_dir / "stloc.xy").open("w", encoding="utf-8") as f:
        f.write("# x(km) y(km) z(km) stnm zsw\n")
        for i, sta in enumerate(stations):
            f.write(
                f"{sta['x_m'] / 1000.0:10.5f} {sta['y_m'] / 1000.0:10.5f} "
                f"{sta['z_m'] / 1000.0:10.5f} {sta['label']:>10s} dep\n"
            )
            station_meta.append(
                {
                    "label": sta["label"],
                    "station_id": sta["station_id"],
                    "coord": [
                        int(i + 1),
                        float(sta["x_m"]),
                        float(sta["y_m"]),
                        float(sta["z_m"]),
                    ],
                    "distance_km": float(sta["distance_km"]),
                    "horizontal_distance_km": float(sta["horizontal_distance_km"]),
                    "p_seconds": sta["p_seconds"],
                    "s_seconds": sta["s_seconds"],
                }
            )
    return station_meta


def _write_lhm_file(args, out_dir: Path) -> Path:
    lhm_path = out_dir / "cape_simple_lhm.dat"
    if args.lhm_file:
        lhm_path.write_text(
            Path(args.lhm_file).read_text(encoding="utf-8"), encoding="utf-8"
        )
    else:
        lhm_path.write_text(DEFAULT_LHM, encoding="utf-8")
    return lhm_path


def _run_command(args) -> list[str]:
    swpc_bin = str(Path(args.swpc_bin).resolve())
    nproc = int(args.nproc_x) * int(args.nproc_y)
    if nproc == 1:
        return [swpc_bin]
    return ["mpirun", "-np", str(nproc), swpc_bin]


def _write_run_script(
    out_dir: Path, basis_list: list[str], run_cmd: list[str]
) -> Path:
    run_sh = out_dir / "run_all_basis.sh"
    with run_sh.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        f.write(f"cd {out_dir.resolve().as_posix()}\n")
        for basis in basis_list:
            f.write(f"echo 'Running {basis}'\n")
            f.write(f"cd {basis}\n")
            f.write(" ".join(run_cmd) + "\n")
            f.write("cd ..\n")
    run_sh.chmod(0o755)
    return run_sh


def _write_cases(
    args, invdata: dict, stations: list[dict]
) -> tuple[Path, dict, list[str], list[str]]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    basis_list = _basis_list(args.basis)
    grid = _grid_for_case(args, invdata["source"], stations)
    station_meta = _write_station_file(out_dir, stations)
    _write_lhm_file(args, out_dir)

    metadata = {
        "event_id": invdata.get("event_id"),
        "source": invdata["source"],
        "stations": station_meta,
        "component_order": ["Z", "N", "E"],
        "openswpc_components": {"Vz": "Z", "Vx": "E", "Vy": "N"},
        "basis_order": BASIS_ORDER,
        "basis_source_order": ["mxx", "myy", "mzz", "myz", "mxz", "mxy"],
        "quantity": "V",
        "grid": grid,
        "parameters": {
            "duration": float(args.duration),
            "dt": float(args.dt),
            "dx": float(args.dx),
            "fmax": float(args.fmax),
            "fq_min": float(args.fq_min),
            "fq_ref": float(args.fq_ref),
            "trise": float(args.trise),
            "pml_width": int(args.pml_width),
            "vcut": float(args.vcut),
            "basis_moment": float(args.basis_moment),
            "lhm_file": str(args.lhm_file) if args.lhm_file else None,
        },
    }
    (out_dir / "station_order.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    source = invdata["source"]
    sx_km = float(source["x"]) / 1000.0
    sy_km = float(source["y"]) / 1000.0
    sz_km = float(source["z"]) / 1000.0

    for basis in basis_list:
        basis_dir = out_dir / basis
        basis_dir.mkdir(parents=True, exist_ok=True)
        mxx, myy, mzz, myz, mxz, mxy = BASIS_TO_SOURCE[basis]

        source_file = basis_dir / f"source_{basis}.dat"
        with source_file.open("w", encoding="utf-8") as f:
            f.write("# x y z tbeg trise mo mxx myy mzz myz mxz mxy\n")
            f.write(
                f" {sx_km:.6f} {sy_km:.6f} {sz_km:.6f} "
                f"0.0 {args.trise:.6f} {args.basis_moment:.6e} "
                f"{mxx:.1f} {myy:.1f} {mzz:.1f} {myz:.1f} {mxz:.1f} {mxy:.1f}\n"
            )

        input_text = _build_input_text(
            args,
            grid,
            Path("../cape_simple_lhm.dat"),
            title=f"cape_lhm_{basis}",
            station_file="../stloc.xy",
            source_file=f"source_{basis}.dat",
        )
        (basis_dir / f"input_{basis}.inf").write_text(input_text, encoding="utf-8")
        in_dir = basis_dir / "in"
        in_dir.mkdir(exist_ok=True)
        (in_dir / "input.inf").write_text(input_text, encoding="utf-8")

    run_cmd = _run_command(args)
    _write_run_script(out_dir, basis_list, run_cmd)
    return out_dir, grid, basis_list, run_cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and optionally run six CAPE LHM OpenSWPC MT-basis GF cases."
    )
    parser.add_argument("--event-dir", default="cape_events/eq02387")
    parser.add_argument("--stations-dir", default="stations")
    parser.add_argument(
        "--out-dir",
        default="cape_events/eq02387/openswpc_lhm_gf_all7_mediumvs_f67_dx60",
    )
    parser.add_argument(
        "--station-ids",
        default="all",
        help="Comma-separated station IDs, 'all', or empty with --max-stations.",
    )
    parser.add_argument("--max-stations", type=int, default=7)
    parser.add_argument(
        "--basis", default="all", help="all or comma-separated basis names"
    )
    parser.add_argument(
        "--lhm-file",
        default="experimental/openswpc_tools/cape_lhm_medium_vs.dat",
        help="Layered CAPE model copied into the case directory.",
    )
    parser.add_argument("--duration", type=float, default=7.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--dx", type=float, default=0.06, help="Grid spacing in km")
    parser.add_argument("--margin-m", type=float, default=0.0)
    parser.add_argument("--interior-margin-m", type=float, default=500.0)
    parser.add_argument("--z-air-m", type=float, default=200.0)
    parser.add_argument("--z-extra-m", type=float, default=0.0)
    parser.add_argument("--bottom-margin-m", type=float, default=800.0)
    parser.add_argument("--fmax", type=float, default=6.7)
    parser.add_argument("--fq-min", type=float, default=0.2)
    parser.add_argument("--fq-ref", type=float, default=6.7)
    parser.add_argument("--trise", type=float, default=0.30)
    parser.add_argument("--vcut", type=float, default=0.5)
    parser.add_argument("--pml-width", type=int, default=8)
    parser.add_argument("--basis-moment", type=float, default=1.0)
    parser.add_argument("--write-displacement", action="store_true")
    parser.add_argument("--nproc-x", type=int, default=1)
    parser.add_argument("--nproc-y", type=int, default=1)
    parser.add_argument(
        "--swpc-bin",
        default="../openswpc/bin/swpc_3d.x",
        help="Path to swpc_3d.x from bayesian_waveform_fitting.",
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if args.trise < args.dt:
        raise ValueError("--trise must be >= --dt")

    event_dir = Path(args.event_dir)
    inv_path = event_dir / "invdata.pkl"
    if not inv_path.exists():
        raise FileNotFoundError(f"invdata not found: {inv_path}")
    with inv_path.open("rb") as f:
        invdata = pickle.load(f)

    stations = _select_stations(args, invdata)
    out_dir, grid, basis_list, run_cmd = _write_cases(args, invdata, stations)

    print(f"GF cases written: {out_dir}")
    print(f"Basis order: {BASIS_ORDER}")
    print(
        "Grid: "
        f"nx={grid['nx']}, ny={grid['ny']}, nz={grid['nz']}, "
        f"dx={args.dx:.3f} km, dt={args.dt:.4f} s, nt={grid['nt']}"
    )
    print(f"Stations: {len(stations)}")
    print("Run command:", " ".join(run_cmd))
    print(
        "Expected packed gf_basis shape: "
        f"({len(BASIS_ORDER)}, {len(stations)}, 3, {grid['nt']})"
    )

    if args.run:
        for basis in basis_list:
            print(f"Running {basis}...")
            subprocess.run(run_cmd, cwd=out_dir / basis, check=True)
        print(f"Basis runs complete: {out_dir}")


if __name__ == "__main__":
    main()
