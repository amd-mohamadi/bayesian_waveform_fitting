import argparse
import concurrent.futures
import json
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experimental.openswpc_tools.run_cape_lhm_forward import (
    DEFAULT_LHM,
    _build_input_text,
    _grid_for_case,
    _select_stations,
)


COMPONENTS = ("x", "y", "z")
BASIS_ORDER = ["Mxx", "Myy", "Mzz", "Mxy", "Mxz", "Myz"]


def _component_list(text: str) -> list[str]:
    if text.lower() == "all":
        return list(COMPONENTS)
    out = [c.strip().lower() for c in text.split(",") if c.strip()]
    bad = [c for c in out if c not in COMPONENTS]
    if bad:
        raise ValueError(f"Unknown green components: {bad}")
    return out


def _write_station_file(out_dir: Path, stations: list[dict]) -> list[dict]:
    station_meta = []
    with (out_dir / "stloc.xy").open("w", encoding="utf-8") as f:
        f.write("# x(km) y(km) z(km) stnm zsw\n")
        for sta in stations:
            f.write(
                f"{sta['x_m'] / 1000.0:10.5f} {sta['y_m'] / 1000.0:10.5f} "
                f"{sta['z_m'] / 1000.0:10.5f} {sta['label']:>10s} dep\n"
            )
            station_meta.append(
                {
                    "label": sta["label"],
                    "station_id": sta["station_id"],
                    "coord": [
                        int(sta["label"].replace("st", "")),
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


def _axis_values(center: float, halfwidth: float, spacing: float) -> list[float]:
    if halfwidth < 0:
        raise ValueError("Source halfwidths must be non-negative")
    if spacing <= 0:
        raise ValueError("Source spacings must be positive")
    n_each_side = int(round(halfwidth / spacing))
    if abs(n_each_side * spacing - halfwidth) > 1e-6:
        raise ValueError(
            f"Source halfwidth {halfwidth:g} m is not an integer multiple of spacing {spacing:g} m"
        )
    return [center + i * spacing for i in range(-n_each_side, n_each_side + 1)]


def _source_grid(args, source: dict) -> list[dict]:
    xs = _axis_values(
        float(source["x"]), float(args.source_halfwidth_x_m), float(args.source_spacing_x_m)
    )
    ys = _axis_values(
        float(source["y"]), float(args.source_halfwidth_y_m), float(args.source_spacing_y_m)
    )
    zs = _axis_values(
        float(source["z"]), float(args.source_halfwidth_z_m), float(args.source_spacing_z_m)
    )
    points = []
    green_id = 1
    for x in xs:
        for y in ys:
            for z in zs:
                points.append(
                    {
                        "green_id": green_id,
                        "x_m": float(x),
                        "y_m": float(y),
                        "z_m": float(z),
                        "is_catalog_source": (
                            abs(x - float(source["x"])) < 1e-9
                            and abs(y - float(source["y"])) < 1e-9
                            and abs(z - float(source["z"])) < 1e-9
                        ),
                    }
                )
                green_id += 1
    if not any(p["is_catalog_source"] for p in points):
        raise RuntimeError("Catalog source was not included in the source grid")
    return points


def _write_green_list(out_dir: Path, source_points: list[dict]) -> Path:
    green_list = out_dir / "green_points.xy"
    with green_list.open("w", encoding="utf-8") as f:
        f.write("# x(km) y(km) z(km) green_id\n")
        for point in source_points:
            f.write(
                f"{point['x_m'] / 1000.0:.6f} "
                f"{point['y_m'] / 1000.0:.6f} "
                f"{point['z_m'] / 1000.0:.6f} {point['green_id']}\n"
            )
    return green_list


def _run_command(args) -> list[str]:
    swpc_bin = str(Path(args.swpc_bin).resolve())
    nproc = int(args.nproc_x) * int(args.nproc_y)
    if nproc == 1:
        return [swpc_bin]
    return ["mpirun", "-np", str(nproc), swpc_bin]


def _case_args(args, station_label: str, component: str):
    ns = argparse.Namespace(**vars(args))
    ns.green_mode = True
    ns.green_stnm = station_label
    ns.green_cmp = component
    ns.green_list = "../../green_points.xy"
    ns.green_fmt = "xyz"
    return ns


def _write_cases(args, invdata: dict, stations: list[dict]):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    components = _component_list(args.components)
    grid = _grid_for_case(args, invdata["source"], stations)
    station_meta = _write_station_file(out_dir, stations)
    source_points = _source_grid(args, invdata["source"])
    green_list = _write_green_list(out_dir, source_points)

    lhm_path = out_dir / "cape_simple_lhm.dat"
    if args.lhm_file:
        lhm_path.write_text(
            Path(args.lhm_file).read_text(encoding="utf-8"), encoding="utf-8"
        )
    else:
        lhm_path.write_text(DEFAULT_LHM, encoding="utf-8")

    metadata = {
        "mode": "openswpc_native_green",
        "event_id": invdata.get("event_id"),
        "source": invdata["source"],
        "stations": station_meta,
        "green_components": components,
        "green_list": green_list.name,
        "green_bforce": bool(args.green_bforce),
        "green_output_order": ["mxx", "myy", "mzz", "myz", "mxz", "mxy"],
        "basis_order": BASIS_ORDER,
        "component_order": ["Z", "N", "E"],
        "component_map": {"x": "E", "y": "N", "z": "Z"},
        "source_grid": {
            "count": len(source_points),
            "halfwidth_m": {
                "x": float(args.source_halfwidth_x_m),
                "y": float(args.source_halfwidth_y_m),
                "z": float(args.source_halfwidth_z_m),
            },
            "spacing_m": {
                "x": float(args.source_spacing_x_m),
                "y": float(args.source_spacing_y_m),
                "z": float(args.source_spacing_z_m),
            },
            "points": source_points,
        },
        "grid": grid,
        "parameters": {
            "duration": float(args.duration),
            "dt": float(args.dt),
            "dx": float(args.dx),
            "dy": float(getattr(args, "dy", args.dx)),
            "dz": float(getattr(args, "dz", args.dx)),
            "fmax": float(args.fmax),
            "fq_min": float(args.fq_min),
            "fq_ref": float(args.fq_ref),
            "trise": float(args.trise),
            "pml_width": int(args.pml_width),
            "vcut": float(args.vcut),
            "lhm_file": str(args.lhm_file) if args.lhm_file else None,
        },
    }
    (out_dir / "station_order.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    case_dirs = []
    for sta in stations:
        label = sta["label"]
        for comp in components:
            case_dir = out_dir / label / comp
            case_dir.mkdir(parents=True, exist_ok=True)
            cargs = _case_args(args, label, comp)
            input_text = _build_input_text(
                cargs,
                grid,
                Path("../../cape_simple_lhm.dat"),
                title=f"green_{label}_{comp}",
                station_file="../../stloc.xy",
                source_file="unused_source.dat",
            )
            (case_dir / "input_green.inf").write_text(input_text, encoding="utf-8")
            in_dir = case_dir / "in"
            in_dir.mkdir(exist_ok=True)
            (in_dir / "input.inf").write_text(input_text, encoding="utf-8")
            case_dirs.append((label, comp, case_dir))

    run_cmd = _run_command(args)
    run_script = out_dir / "run_all_green.sh"
    with run_script.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        f.write(f"cd {out_dir.resolve().as_posix()}\n")
        for label, comp, _ in case_dirs:
            f.write(f"echo 'Running native green {label} {comp}'\n")
            f.write(f"cd {label}/{comp}\n")
            f.write(" ".join(run_cmd) + "\n")
            f.write("cd ../..\n")
    run_script.chmod(0o755)
    return out_dir, grid, case_dirs, run_cmd


def _sac_count(case_dir: Path) -> int:
    out_dir = case_dir / "out" / "green"
    if not out_dir.exists():
        return 0
    return sum(1 for _ in out_dir.rglob("*.sac"))


def _run_one_case(run_cmd: list[str], label: str, comp: str, case_dir: Path) -> dict:
    start = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    result = subprocess.run(run_cmd, cwd=case_dir, check=False)
    end = datetime.now(timezone.utc)
    return {
        "station": label,
        "component": comp,
        "case_dir": str(case_dir),
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "wall_seconds": time.perf_counter() - t0,
        "return_code": int(result.returncode),
        "output_count": _sac_count(case_dir),
    }


def _run_cases_parallel(out_dir: Path, case_dirs: list[tuple[str, str, Path]], run_cmd: list[str], parallel_jobs: int) -> list[dict]:
    if parallel_jobs < 1:
        raise ValueError("--parallel-jobs must be >= 1")
    timings = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
        futures = {
            executor.submit(_run_one_case, run_cmd, label, comp, case_dir): (label, comp)
            for label, comp, case_dir in case_dirs
        }
        for future in concurrent.futures.as_completed(futures):
            label, comp = futures[future]
            report = future.result()
            timings.append(report)
            print(
                f"Runtime {label} {comp}: {report['wall_seconds']:.2f} s, "
                f"rc={report['return_code']}, outputs={report['output_count']}"
            )
    timings.sort(key=lambda item: (item["station"], item["component"]))
    runtime_path = out_dir / "runtime_native_green.json"
    runtime_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and optionally run native OpenSWPC green-mode CAPE cases."
    )
    parser.add_argument("--event-dir", default="cape_events/eq02387")
    parser.add_argument("--stations-dir", default="stations")
    parser.add_argument(
        "--out-dir", default="cape_events/eq02387/openswpc_native_green_test"
    )
    parser.add_argument("--station-ids", default="UU.FSB2.01.HH")
    parser.add_argument("--max-stations", type=int, default=1)
    parser.add_argument("--components", default="z", help="'all' or comma list of x,y,z")
    parser.add_argument(
        "--lhm-file", default="experimental/openswpc_tools/cape_lhm_medium_vs.dat"
    )
    parser.add_argument("--duration", type=float, default=7.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--dx", type=float, default=0.06, help="Grid spacing in km")
    parser.add_argument("--dy", type=float, default=None, help="Grid spacing in y in km")
    parser.add_argument("--dz", type=float, default=None, help="Grid spacing in z in km")
    parser.add_argument("--source-halfwidth-x-m", type=float, default=300.0)
    parser.add_argument("--source-halfwidth-y-m", type=float, default=300.0)
    parser.add_argument("--source-halfwidth-z-m", type=float, default=300.0)
    parser.add_argument("--source-spacing-x-m", type=float, default=150.0)
    parser.add_argument("--source-spacing-y-m", type=float, default=150.0)
    parser.add_argument("--source-spacing-z-m", type=float, default=150.0)
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
    parser.add_argument("--green-bforce", action="store_true")
    parser.add_argument("--green-maxdist", type=float, default=550.0)
    parser.add_argument("--write-displacement", action="store_true")
    parser.add_argument("--nproc-x", type=int, default=1)
    parser.add_argument("--nproc-y", type=int, default=1)
    parser.add_argument("--swpc-bin", default="../openswpc/bin/swpc_3d.x")
    parser.add_argument("--parallel-jobs", type=int, default=7)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.dy is None:
        args.dy = args.dx
    if args.dz is None:
        args.dz = args.dx

    if args.trise < args.dt:
        raise ValueError("--trise must be >= --dt")

    inv_path = Path(args.event_dir) / "invdata.pkl"
    if not inv_path.exists():
        raise FileNotFoundError(f"invdata not found: {inv_path}")
    with inv_path.open("rb") as f:
        invdata = pickle.load(f)

    stations = _select_stations(args, invdata)
    out_dir, grid, case_dirs, run_cmd = _write_cases(args, invdata, stations)

    print(f"Native green cases written: {out_dir}")
    print(
        "Grid: "
        f"nx={grid['nx']}, ny={grid['ny']}, nz={grid['nz']}, "
        f"dx={args.dx:.3f} km, dy={args.dy:.3f} km, dz={args.dz:.3f} km, "
        f"dt={args.dt:.4f} s, nt={grid['nt']}"
    )
    print(
        f"Runs: {len(case_dirs)} "
        f"({len(stations)} station(s) x {len(_component_list(args.components))} component(s))"
    )
    print("Run command:", " ".join(run_cmd))

    if args.run:
        print(f"Running up to {args.parallel_jobs} native green case(s) in parallel...")
        timings = _run_cases_parallel(out_dir, case_dirs, run_cmd, args.parallel_jobs)
        failures = [item for item in timings if item["return_code"] != 0]
        if failures:
            raise RuntimeError(f"{len(failures)} native green run(s) failed")
        runtime_path = out_dir / "runtime_native_green.json"
        print(f"Wrote runtime report: {runtime_path}")


if __name__ == "__main__":
    main()
