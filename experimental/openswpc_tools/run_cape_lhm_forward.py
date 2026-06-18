import argparse
import json
import math
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from obspy import UTCDateTime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io import load_stations_from_xml


DEFAULT_LHM = """# depth(km) rho(g/cm^3) vp(km/s) vs(km/s) Qp Qs
# Simple CAPE diagnostic model. Keep this low-frequency until timing is sane.
0.0   2.14  2.65  1.53  300  150
0.9   2.53  4.95  2.86  500  250
1.5   2.59  5.70  3.29  600  300
8.0   2.69  6.00  3.47  800  400
21.0  2.82  6.40  3.70  800  400
"""


def _round_down(x: float, step: float) -> float:
    return float(np.floor(x / step) * step)


def _round_up(x: float, step: float) -> float:
    return float(np.ceil(x / step) * step)


def _random_mt_components(seed: int) -> tuple[float, float, float, float, float, float]:
    rng = np.random.default_rng(seed)
    mt = rng.normal(size=6).astype(float)
    mt /= np.max(np.abs(mt))
    return tuple(float(x) for x in mt)


def _station_lookup(stations_dir: str):
    stations_all, codes_all, _, _ = load_stations_from_xml(stations_dir)
    by_id = {str(code): i for i, code in enumerate(codes_all)}
    by_station = {}
    for i, code in enumerate(codes_all):
        parts = str(code).split(".")
        if len(parts) > 1:
            by_station.setdefault(parts[1], i)
    return stations_all, by_id, by_station


def _pick_time_seconds(pick, origin):
    if not pick:
        return None
    return float(UTCDateTime(pick) - origin)


def _selected_station_ids(args, invdata):
    requested = [s.strip() for s in args.station_ids.split(",") if s.strip()]
    if requested == ["all"]:
        return [str(s) for s in invdata["station_ids"]]
    if requested:
        return requested
    return [str(s) for s in invdata["station_ids"][: args.max_stations]]


def _select_stations(args, invdata):
    stations_all, by_id, by_station = _station_lookup(args.stations_dir)
    source = invdata["source"]
    sx_m = float(source["x"])
    sy_m = float(source["y"])
    sz_m = float(source["z"])
    origin = UTCDateTime(source["origin_time"])

    selected = []
    missing = []
    for sid in _selected_station_ids(args, invdata):
        parts = sid.split(".")
        sta = parts[1] if len(parts) > 1 else sid
        idx = by_id.get(sid, by_station.get(sta))
        if idx is None:
            missing.append(sid)
            continue
        row = stations_all[idx]
        x_m = float(row[1])
        y_m = float(row[2])
        z_m = float(row[3])
        distance_km = math.sqrt((x_m - sx_m) ** 2 + (y_m - sy_m) ** 2 + (z_m - sz_m) ** 2) / 1000.0
        horizontal_km = math.sqrt((x_m - sx_m) ** 2 + (y_m - sy_m) ** 2) / 1000.0
        picks = invdata.get("picks", {}).get(sid, {})
        p_sec = _pick_time_seconds(picks.get("P"), origin)
        s_sec = _pick_time_seconds(picks.get("S"), origin)
        selected.append(
            {
                "station_id": sid,
                "label": f"st{len(selected) + 1:03d}",
                "x_m": x_m,
                "y_m": y_m,
                "z_m": z_m,
                "distance_km": distance_km,
                "horizontal_distance_km": horizontal_km,
                "p_seconds": p_sec,
                "s_seconds": s_sec,
            }
        )

    if missing:
        raise RuntimeError(f"StationXML missing selected station(s): {missing}")
    if not selected:
        raise RuntimeError("No stations selected")
    return selected


def _build_input_text(
    args,
    grid,
    lhm_file: Path,
    title: str = "cape_lhm_random_mt",
    station_file: str = "stloc.xy",
    source_file: str = "source_random.dat",
):
    dx = float(args.dx)
    dy = float(getattr(args, "dy", dx))
    dz = float(getattr(args, "dz", dx))
    sw_wav_u = ".true." if args.write_displacement else ".false."
    green_mode = ".true." if getattr(args, "green_mode", False) else ".false."
    green_stnm = getattr(args, "green_stnm", "st01")
    green_cmp = getattr(args, "green_cmp", "z")
    green_bforce = ".true." if getattr(args, "green_bforce", False) else ".false."
    green_maxdist = float(getattr(args, "green_maxdist", 550.0))
    green_fmt = getattr(args, "green_fmt", "xyz")
    green_list = getattr(args, "green_list", "example/green/green.lst")
    return (
        f"""
  title            = '{title}'
  odir             = 'out'
  ntdec_r          = 50
  strict_mode      = .false.

  nproc_x          = {args.nproc_x}
  nproc_y          = {args.nproc_y}
  nx               = {grid["nx"]}
  ny               = {grid["ny"]}
  nz               = {grid["nz"]}
  nt               = {grid["nt"]}

  dx               = {dx:.6f}
  dy               = {dy:.6f}
  dz               = {dz:.6f}
  dt               = {args.dt:.6f}

  vcut             = {args.vcut:.2f}

  xbeg             = {grid["x_min_m"] / 1000.0:.6f}
  ybeg             = {grid["y_min_m"] / 1000.0:.6f}
  zbeg             = {grid["z_min_m"] / 1000.0:.6f}
  tbeg             = 0.0

  clon             = -112.8963897
  clat             = 38.50402147
  phi              = 0.0

  fq_min           = {args.fq_min:.3f}
  fq_max           = {args.fmax:.3f}
  fq_ref           = {args.fq_ref:.3f}

  snp_format       = 'netcdf'
  xy_ps%sw         = .false.
  xz_ps%sw         = .false.
  yz_ps%sw         = .false.
  fs_ps%sw         = .false.
  ob_ps%sw         = .false.
  xy_v%sw          = .false.
  xz_v%sw          = .false.
  yz_v%sw          = .false.
  fs_v%sw          = .false.
  ob_v%sw          = .false.
  xy_u%sw          = .false.
  xz_u%sw          = .false.
  yz_u%sw          = .false.
  fs_u%sw          = .false.
  ob_u%sw          = .false.
  ntdec_s          = 10
  idec             = 2
  jdec             = 2
  kdec             = 2

  sw_wav_v         = .true.
  sw_wav_u         = {sw_wav_u}
  sw_wav_stress    = .false.
  sw_wav_strain    = .false.
  ntdec_w          = 1
  st_format        = 'xy'
  fn_stloc         = '{station_file}'
  wav_format       = 'sac'
  ntdec_w_prg      = 0

  stf_format       = 'xym0ij'
  stftype          = 'cosine'
  fn_stf           = '{source_file}'
  sdep_fit         = 'asis'
  bf_mode          = .false.
  pw_mode          = .false.

  abc_type         = 'pml'
  na               = {args.pml_width}
  stabilize_pml    = .false.

  vmodel_type      = 'lhm'
  is_ocean         = .false.
  topo_flatten     = .true.
  munk_profile     = .false.
  earth_flattening = .false.

  vp0              = 5.0
  vs0              = 3.0
  rho0             = 2.7
  qp0              = 500.0
  qs0              = 250.0
  topo0            = 0.0
  fn_lhm           = '{lhm_file.as_posix()}'

  is_ckp           = .false.
  ckpdir           = './out/ckp'
  ckp_interval     = 1000000
  ckp_time         = 1000000.0
  ckp_seq          = .true.

  green_mode       = {green_mode}
  green_stnm       = '{green_stnm}'
  green_cmp        = '{green_cmp}'
  green_trise      = {args.trise:.6f}
  green_bforce     = {green_bforce}
  green_maxdist    = {green_maxdist:.6f}
  green_fmt        = '{green_fmt}'
  fn_glst          = '{green_list}'

  stopwatch_mode   = .false.
  benchmark_mode   = .false.

  ipad             = 0
  jpad             = 0
  kpad             = 0
""".strip()
        + "\n"
    )


def _grid_for_case(args, source, stations):
    dx_m = float(args.dx) * 1000.0
    dy_m = float(getattr(args, "dy", args.dx)) * 1000.0
    dz_m = float(getattr(args, "dz", args.dx)) * 1000.0
    horizontal_step_m = max(dx_m, dy_m)
    vertical_step_m = dz_m
    pml_margin_m = args.pml_width * horizontal_step_m + args.interior_margin_m
    margin_m = max(args.margin_m, pml_margin_m)
    vertical_extra_m = max(args.z_extra_m, args.pml_width * vertical_step_m + args.bottom_margin_m)

    xs = [float(source["x"])] + [s["x_m"] for s in stations]
    ys = [float(source["y"])] + [s["y_m"] for s in stations]
    zs = [float(source["z"])] + [s["z_m"] for s in stations]

    x_min_m = _round_down(min(xs) - margin_m, dx_m)
    x_max_m = _round_up(max(xs) + margin_m, dx_m)
    y_min_m = _round_down(min(ys) - margin_m, dy_m)
    y_max_m = _round_up(max(ys) + margin_m, dy_m)
    z_min_m = -float(args.z_air_m)
    z_max_m = _round_up(max(zs) + vertical_extra_m, dz_m)

    nx = int(round((x_max_m - x_min_m) / dx_m))
    ny = int(round((y_max_m - y_min_m) / dy_m))
    nz = int(round((z_max_m - z_min_m) / dz_m))
    nt = int(math.ceil(args.duration / args.dt))

    if nx <= 2 * args.pml_width + 4 or ny <= 2 * args.pml_width + 4:
        raise ValueError(
            "Grid too small after PML. Increase --margin-m or reduce --pml-width."
        )
    if nz <= args.pml_width + 6:
        raise ValueError(
            "Vertical grid too small after bottom PML. Increase --z-extra-m or reduce --pml-width."
        )

    return {
        "dx_m": dx_m,
        "dy_m": dy_m,
        "dz_m": dz_m,
        "margin_m": margin_m,
        "vertical_extra_m": vertical_extra_m,
        "x_min_m": x_min_m,
        "x_max_m": x_max_m,
        "y_min_m": y_min_m,
        "y_max_m": y_max_m,
        "z_min_m": z_min_m,
        "z_max_m": z_max_m,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "nt": nt,
    }


def _write_case(args, invdata, stations):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = invdata["source"]
    grid = _grid_for_case(args, source, stations)
    m0 = 10 ** (1.5 * float(source["mw"]) + 9.1) * float(args.moment_scale)
    mxx, myy, mzz, myz, mxz, mxy = _random_mt_components(args.seed)

    lhm_path = out_dir / "cape_simple_lhm.dat"
    if args.lhm_file:
        lhm_path.write_text(Path(args.lhm_file).read_text(encoding="utf-8"), encoding="utf-8")
    else:
        lhm_path.write_text(DEFAULT_LHM, encoding="utf-8")

    with (out_dir / "stloc.xy").open("w", encoding="utf-8") as f:
        f.write("# x(km) y(km) z(km) stnm zsw\n")
        for sta in stations:
            f.write(
                f"{sta['x_m'] / 1000.0:10.5f} {sta['y_m'] / 1000.0:10.5f} "
                f"{sta['z_m'] / 1000.0:10.5f} {sta['label']:>10s} dep\n"
            )

    with (out_dir / "source_random.dat").open("w", encoding="utf-8") as f:
        f.write("# x y z tbeg trise mo mxx myy mzz myz mxz mxy\n")
        f.write(
            f" {float(source['x']) / 1000.0:.6f} {float(source['y']) / 1000.0:.6f} "
            f"{float(source['z']) / 1000.0:.6f} 0.0 {args.trise:.6f} {m0:.6e} "
            f"{mxx:.6f} {myy:.6f} {mzz:.6f} {myz:.6f} {mxz:.6f} {mxy:.6f}\n"
        )

    input_text = _build_input_text(args, grid, Path("cape_simple_lhm.dat"))
    (out_dir / "input_lhm.inf").write_text(input_text, encoding="utf-8")
    in_dir = out_dir / "in"
    in_dir.mkdir(exist_ok=True)
    (in_dir / "input.inf").write_text(input_text, encoding="utf-8")

    metadata = {
        "event_id": invdata.get("event_id"),
        "source": source,
        "stations": stations,
        "component_order": ["Z", "N", "E"],
        "openswpc_components": {"Vz": "Z", "Vx": "E", "Vy": "N"},
        "grid": grid,
        "random_mt": {
            "seed": args.seed,
            "m0": m0,
            "mxx": mxx,
            "myy": myy,
            "mzz": mzz,
            "myz": myz,
            "mxz": mxz,
            "mxy": mxy,
        },
        "parameters": {
            "duration": args.duration,
            "dt": args.dt,
            "dx": args.dx,
            "fmax": args.fmax,
            "source_fmax_estimate": 2.0 / args.trise,
            "fq_min": args.fq_min,
            "fq_ref": args.fq_ref,
            "trise": args.trise,
            "pml_width": args.pml_width,
            "vcut": args.vcut,
        },
    }
    (out_dir / "case_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    nproc = args.nproc_x * args.nproc_y
    if nproc == 1:
        run_cmd = [str(Path(args.swpc_bin).resolve())]
    else:
        run_cmd = ["mpirun", "-np", str(nproc), str(Path(args.swpc_bin).resolve())]
    run_script = out_dir / "run_case.sh"
    run_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {out_dir.resolve().as_posix()}\n"
        + " ".join(run_cmd)
        + "\n",
        encoding="utf-8",
    )
    run_script.chmod(0o755)

    return out_dir, grid, run_cmd


def main():
    parser = argparse.ArgumentParser(
        description="Create and optionally run a fast 1D-layer OpenSWPC diagnostic for CAPE."
    )
    parser.add_argument("--event-dir", default="cape_events/eq02387")
    parser.add_argument("--stations-dir", default="stations")
    parser.add_argument("--out-dir", default="cape_events/eq02387/openswpc_lhm_fsb2_f8_dx60")
    parser.add_argument(
        "--station-ids",
        default="UU.FSB2.01.HH",
        help="Comma-separated station IDs, 'all', or empty with --max-stations.",
    )
    parser.add_argument("--max-stations", type=int, default=1)
    parser.add_argument("--lhm-file", default=None)
    parser.add_argument("--seed", type=int, default=2387)
    parser.add_argument("--duration", type=float, default=7.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--dx", type=float, default=0.06, help="Grid spacing in km")
    parser.add_argument("--margin-m", type=float, default=0.0)
    parser.add_argument("--interior-margin-m", type=float, default=500.0)
    parser.add_argument("--z-air-m", type=float, default=200.0)
    parser.add_argument("--z-extra-m", type=float, default=0.0)
    parser.add_argument("--bottom-margin-m", type=float, default=800.0)
    parser.add_argument("--fmax", type=float, default=8.0)
    parser.add_argument("--fq-min", type=float, default=0.2)
    parser.add_argument("--fq-ref", type=float, default=8.0)
    parser.add_argument("--trise", type=float, default=0.25)
    parser.add_argument("--vcut", type=float, default=2.5)
    parser.add_argument("--pml-width", type=int, default=8)
    parser.add_argument("--moment-scale", type=float, default=1.0)
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
        raise ValueError("--trise must be >= --dt; smaller values can produce bad/zero output.")
    if args.station_ids == "":
        args.station_ids = ",".join([])

    event_dir = Path(args.event_dir)
    inv_path = event_dir / "invdata.pkl"
    if not inv_path.exists():
        raise FileNotFoundError(f"invdata not found: {inv_path}")
    with inv_path.open("rb") as f:
        invdata = pickle.load(f)

    stations = _select_stations(args, invdata)
    out_dir, grid, run_cmd = _write_case(args, invdata, stations)

    print(f"Case written: {out_dir}")
    print(
        "Grid: "
        f"nx={grid['nx']}, ny={grid['ny']}, nz={grid['nz']}, "
        f"dx={args.dx:.3f} km, dt={args.dt:.4f} s, nt={grid['nt']}"
    )
    source_fmax = 2.0 / args.trise
    print(
        f"Frequency target: fq_max={args.fmax:.2f} Hz, "
        f"trise={args.trise:.3f} s, source fmax~{source_fmax:.2f} Hz"
    )
    wavelength_ratio = max(args.vcut, 1.53) / (source_fmax * args.dx)
    print(f"Approx. minimum wavelength / dx from source fmax: {wavelength_ratio:.2f}")
    print("Stations:")
    for sta in stations:
        p_text = "NA" if sta["p_seconds"] is None else f"{sta['p_seconds']:.3f}s"
        s_text = "NA" if sta["s_seconds"] is None else f"{sta['s_seconds']:.3f}s"
        print(
            f"  {sta['label']} {sta['station_id']} "
            f"r3={sta['distance_km']:.2f} km rh={sta['horizontal_distance_km']:.2f} km "
            f"P={p_text} S={s_text}"
        )
    print("Run command:", " ".join(run_cmd))

    if args.run:
        subprocess.run(run_cmd, cwd=out_dir, check=True)
        print(f"Run complete: {out_dir / 'out'}")


if __name__ == "__main__":
    main()
