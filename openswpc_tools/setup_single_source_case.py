import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_smc_mti.io import load_stations_from_xml


BASIS_TO_MT = {
    "Mxx": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "Myy": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "Mzz": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "Mxy": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    "Mxz": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    "Myz": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
}


def _round_down(x: float, step: float) -> float:
    return np.floor(x / step) * step


def _round_up(x: float, step: float) -> float:
    return np.ceil(x / step) * step


def _build_input_text(
    title: str,
    odir: Path,
    station_file: Path,
    source_file: Path,
    model_nc: Path,
    ref_e: float,
    ref_n: float,
    ref_h: float,
    nproc_x: int,
    nproc_y: int,
    nx: int,
    ny: int,
    nz: int,
    nt: int,
    dx_km: float,
    dy_km: float,
    dz_km: float,
    dt_s: float,
    xbeg_km: float,
    ybeg_km: float,
    zbeg_km: float,
    fq_max: float,
    trise: float,
    vcut: float,
    write_displacement: bool,
):
    sw_wav_u = ".true." if write_displacement else ".false."
    return (
        f"""
  title            = '{title}'
  odir             = '{odir.as_posix()}'
  ntdec_r          = 50
  strict_mode      = .false.

  nproc_x          = {nproc_x}
  nproc_y          = {nproc_y}
  nx               = {nx}
  ny               = {ny}
  nz               = {nz}
  nt               = {nt}

  dx               = {dx_km:.6f}
  dy               = {dy_km:.6f}
  dz               = {dz_km:.6f}
  dt               = {dt_s:.6f}

  vcut             = {vcut:.2f}

  xbeg             = {xbeg_km:.6f}
  ybeg             = {ybeg_km:.6f}
  zbeg             = {zbeg_km:.6f}
  tbeg             = 0.0

  clon             = -112.8963897
  clat             = 38.50402147
  phi              = 0.0

  fq_min           = 0.5
  fq_max           = {fq_max:.3f}
  fq_ref           = 30.0

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
  fn_stloc         = '{station_file.as_posix()}'
  wav_format       = 'sac'
  ntdec_w_prg      = 0

  stf_format       = 'xym0ij'
  stftype          = 'cosine'
  fn_stf           = '{source_file.as_posix()}'
  sdep_fit         = 'asis'
  bf_mode          = .false.
  pw_mode          = .false.

  abc_type         = 'pml'
  na               = 20
  stabilize_pml    = .false.

  vmodel_type      = 'user'
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

  user_model_nc        = '{model_nc.as_posix()}'
  user_ref_easting     = {ref_e:.4f}
  user_ref_northing    = {ref_n:.4f}
  user_ref_elevation   = {ref_h:.4f}

  is_ckp           = .false.
  ckpdir           = './out/ckp'
  ckp_interval     = 1000000
  ckp_time         = 1000000.0
  ckp_seq          = .true.

  green_mode       = .false.
  green_stnm       = 'st01'
  green_cmp        = 'z'
  green_trise      = {trise:.6f}
  green_bforce     = .false.
  green_maxdist    = 550.0
  green_fmt        = 'xyz'
  fn_glst          = 'example/green/green.lst'

  stopwatch_mode   = .false.
  benchmark_mode   = .false.

  ipad             = 0
  jpad             = 0
  kpad             = 0
""".strip()
        + "\n"
    )


def _select_stations(stations_all, codes_all, station_ids):
    by_id = {str(code): i for i, code in enumerate(codes_all)}
    by_station = {}
    for i, code in enumerate(codes_all):
        parts = str(code).split(".")
        if len(parts) > 1:
            by_station.setdefault(parts[1], i)

    rows = []
    ids = []
    missing = []
    for sid in station_ids:
        parts = str(sid).split(".")
        sta = parts[1] if len(parts) > 1 else str(sid)
        idx = by_id.get(str(sid), by_station.get(sta))
        if idx is None:
            missing.append(str(sid))
            continue
        src = stations_all[idx]
        rows.append([len(rows) + 1, float(src[1]), float(src[2]), float(src[3])])
        ids.append(str(sid))

    if missing:
        raise RuntimeError(f"StationXML missing event station(s): {missing}")
    if not rows:
        raise RuntimeError("No event stations matched StationXML files")
    return np.asarray(rows, dtype=float), ids


def main():
    parser = argparse.ArgumentParser(
        description="Create OpenSWPC single-source run inputs for MT basis simulations"
    )
    parser.add_argument("--event-dir", required=True)
    parser.add_argument("--stations-dir", required=True)
    parser.add_argument("--model-nc", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--basis", default="all", help="all or comma-separated basis names"
    )
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--dx", type=float, default=0.05, help="Grid spacing in km")
    parser.add_argument("--margin-m", type=float, default=1200.0)
    parser.add_argument("--z-air-m", type=float, default=200.0)
    parser.add_argument("--z-extra-m", type=float, default=1200.0)
    parser.add_argument("--fmax", type=float, default=200.0)
    parser.add_argument("--trise", type=float, default=0.001)
    parser.add_argument("--vcut", type=float, default=1.5)
    parser.add_argument(
        "--write-displacement",
        action="store_true",
        help="Also write displacement SAC traces. CAPE velocity-GF runs leave this off.",
    )
    parser.add_argument("--nproc-x", type=int, default=2)
    parser.add_argument("--nproc-y", type=int, default=2)
    parser.add_argument("--ref-elevation", type=float, default=1650.0249)
    args = parser.parse_args()

    event_dir = Path(args.event_dir)
    out_dir = Path(args.out_dir).resolve()
    model_nc = Path(args.model_nc).resolve()
    inv_path = event_dir / "invdata.pkl"
    if not inv_path.exists():
        raise FileNotFoundError(f"invdata not found: {inv_path}")

    with inv_path.open("rb") as f:
        invdata = pickle.load(f)

    source = invdata["source"]
    sx_m = float(source["x"])
    sy_m = float(source["y"])
    sz_m = float(source["z"])
    ref_e = float(source.get("easting", 334641.1891)) - sx_m
    ref_n = float(source.get("northing", 4263443.693)) - sy_m

    stations_all, codes_all, _, _ = load_stations_from_xml(args.stations_dir)
    stations, station_ids = _select_stations(
        stations_all, codes_all, invdata["station_ids"]
    )
    x_sta = stations[:, 1].astype(float)
    y_sta = stations[:, 2].astype(float)
    z_sta = stations[:, 3].astype(float)

    x_all = np.concatenate([x_sta, np.array([sx_m])])
    y_all = np.concatenate([y_sta, np.array([sy_m])])

    dx_m = args.dx * 1000.0
    x_min = _round_down(float(np.min(x_all) - args.margin_m), dx_m)
    x_max = _round_up(float(np.max(x_all) + args.margin_m), dx_m)
    y_min = _round_down(float(np.min(y_all) - args.margin_m), dx_m)
    y_max = _round_up(float(np.max(y_all) + args.margin_m), dx_m)

    z_min_m = -float(args.z_air_m)
    z_max_m = _round_up(max(sz_m, float(np.max(z_sta))) + args.z_extra_m, dx_m)

    nx = int(round((x_max - x_min) / dx_m)) + 1
    ny = int(round((y_max - y_min) / dx_m)) + 1
    nz = int(round((z_max_m - z_min_m) / dx_m)) + 1
    nt = int(round(args.duration / args.dt))

    basis_list = (
        list(BASIS_TO_MT.keys())
        if args.basis.lower() == "all"
        else [b.strip() for b in args.basis.split(",") if b.strip()]
    )
    bad = [b for b in basis_list if b not in BASIS_TO_MT]
    if bad:
        raise ValueError(f"Unknown basis values: {bad}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model_link = out_dir / "model.nc"
    if model_link.exists() or model_link.is_symlink():
        if model_link.is_symlink():
            model_link.unlink()
    if not model_link.exists():
        try:
            model_link.symlink_to(model_nc)
        except OSError:
            model_link = model_nc

    st_file = out_dir / "stloc.xy"
    station_meta = []
    with st_file.open("w", encoding="utf-8") as f:
        f.write("# x(km) y(km) z(km) stnm zsw\n")
        for i in range(stations.shape[0]):
            stnm = f"st{i + 1:03d}"
            f.write(
                f"{x_sta[i] / 1000.0:10.5f} {y_sta[i] / 1000.0:10.5f} {z_sta[i] / 1000.0:10.5f} {stnm:>10s} dep\n"
            )
            station_meta.append(
                {
                    "label": stnm,
                    "station_id": station_ids[i],
                    "coord": [
                        int(i + 1),
                        float(x_sta[i]),
                        float(y_sta[i]),
                        float(z_sta[i]),
                    ],
                }
            )

    metadata_file = out_dir / "station_order.json"
    metadata = {
        "station_file": str(st_file.resolve()),
        "stations": station_meta,
        "component_order": ["Z", "N", "E"],
        "basis_order": list(BASIS_TO_MT.keys()),
        "event_id": str(invdata.get("event_id", event_dir.name)),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for basis in basis_list:
        basis_dir = out_dir / basis
        basis_dir.mkdir(parents=True, exist_ok=True)

        mxx, myy, mzz, myz, mxz, mxy = BASIS_TO_MT[basis]
        src_file = basis_dir / f"source_{basis}.dat"
        with src_file.open("w", encoding="utf-8") as f:
            f.write("# x y z tbeg trise mo mxx myy mzz myz mxz mxy\n")
            f.write(
                f" {sx_m / 1000.0:.6f} {sy_m / 1000.0:.6f} {sz_m / 1000.0:.6f} 0.0 {args.trise:.6f} 1.0 "
                f"{mxx:.1f} {myy:.1f} {mzz:.1f} {myz:.1f} {mxz:.1f} {mxy:.1f}\n"
            )

        input_file = basis_dir / f"input_{basis}.inf"
        input_text = _build_input_text(
            title=f"forge_{basis}",
            odir=Path("out"),
            station_file=Path("../stloc.xy"),
            source_file=Path(f"source_{basis}.dat"),
            model_nc=Path("../model.nc") if model_link != model_nc else model_nc,
            ref_e=ref_e,
            ref_n=ref_n,
            ref_h=args.ref_elevation,
            nproc_x=args.nproc_x,
            nproc_y=args.nproc_y,
            nx=nx,
            ny=ny,
            nz=nz,
            nt=nt,
            dx_km=args.dx,
            dy_km=args.dx,
            dz_km=args.dx,
            dt_s=args.dt,
            xbeg_km=x_min / 1000.0,
            ybeg_km=y_min / 1000.0,
            zbeg_km=z_min_m / 1000.0,
            fq_max=args.fmax,
            trise=args.trise,
            vcut=args.vcut,
            write_displacement=bool(args.write_displacement),
        )
        input_file.write_text(input_text, encoding="utf-8")

        in_dir = basis_dir / "in"
        in_dir.mkdir(exist_ok=True)
        (in_dir / "input.inf").write_text(input_text, encoding="utf-8")

    run_sh = out_dir / "run_all_basis.sh"
    with run_sh.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        f.write(
            'SWPC_BIN="/home/a-mohamdi/Projects/focal_inversion/openswpc/bin/swpc_3d.x"\n'
        )
        for basis in basis_list:
            basis_path = (out_dir / basis).as_posix()
            f.write(f"echo 'Running {basis}'\n")
            f.write(
                f'(cd {basis_path} && mpirun -np {args.nproc_x * args.nproc_y} "$SWPC_BIN")\n'
            )

    run_sh.chmod(0o755)

    print(f"Wrote OpenSWPC case directory: {out_dir}")
    print(f"Basis simulations: {basis_list}")
    print(f"Stations: {len(station_ids)} written to {st_file}")
    print(f"Station metadata: {metadata_file}")
    print(
        "Grid: "
        f"nx={nx}, ny={ny}, nz={nz}, dx={args.dx:.3f} km, dt={args.dt:.4f} s, nt={nt}"
    )
    print(f"Runner script: {run_sh}")


if __name__ == "__main__":
    main()
