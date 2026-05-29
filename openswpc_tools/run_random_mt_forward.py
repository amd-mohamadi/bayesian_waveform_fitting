import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_simulated_inversion import load_stations_from_xml


def _round_down(x: float, step: float) -> float:
    return np.floor(x / step) * step


def _round_up(x: float, step: float) -> float:
    return np.ceil(x / step) * step


def _random_mt_components(seed: int) -> tuple[float, float, float, float, float, float]:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=6).astype(float)
    v /= np.max(np.abs(v))
    return float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4]), float(v[5])


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
    dt_s: float,
    xbeg_km: float,
    ybeg_km: float,
    zbeg_km: float,
    fq_max: float,
    trise: float,
    vcut: float,
):
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
  dy               = {dx_km:.6f}
  dz               = {dx_km:.6f}
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
  sw_wav_u         = .true.
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


def main():
    parser = argparse.ArgumentParser(
        description="Run one random-MT forward simulation with OpenSWPC"
    )
    parser.add_argument("--event-dir", required=True)
    parser.add_argument("--stations-dir", required=True)
    parser.add_argument("--model-nc", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--dx", type=float, default=0.05, help="km")
    parser.add_argument("--margin-m", type=float, default=1200.0)
    parser.add_argument("--z-air-m", type=float, default=200.0)
    parser.add_argument("--z-extra-m", type=float, default=1200.0)
    parser.add_argument("--fmax", type=float, default=100.0)
    parser.add_argument("--trise", type=float, default=0.006)
    parser.add_argument("--vcut", type=float, default=1.5)
    parser.add_argument("--nproc-x", type=int, default=2)
    parser.add_argument("--nproc-y", type=int, default=2)
    parser.add_argument("--ref-elevation", type=float, default=1650.0249)
    parser.add_argument(
        "--swpc-bin",
        type=str,
        default="swpc_3d.x",
        help="OpenSWPC swpc_3d.x executable used when --run is set",
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    event_dir = Path(args.event_dir)
    out_dir = Path(args.out_dir)
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
    mw = float(source["mw"])
    m0 = 10 ** (1.5 * mw + 9.1)

    ref_e = float(source.get("easting", 334641.1891)) - sx_m
    ref_n = float(source.get("northing", 4263443.693)) - sy_m

    stations, _, _, _ = load_stations_from_xml(args.stations_dir)
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

    case_dir = out_dir
    case_dir.mkdir(parents=True, exist_ok=True)

    st_file = case_dir / "stloc.xy"
    with st_file.open("w", encoding="utf-8") as f:
        f.write("# x(km) y(km) z(km) stnm zsw\n")
        for i in range(stations.shape[0]):
            stnm = f"st{i + 1:03d}"
            f.write(
                f"{x_sta[i] / 1000.0:10.5f} {y_sta[i] / 1000.0:10.5f} {z_sta[i] / 1000.0:10.5f} {stnm:>10s} dep\n"
            )

    mxx, myy, mzz, myz, mxz, mxy = _random_mt_components(args.seed)
    src_file = case_dir / "source_random.dat"
    with src_file.open("w", encoding="utf-8") as f:
        f.write("# x y z tbeg trise mo mxx myy mzz myz mxz mxy\n")
        f.write(
            f" {sx_m / 1000.0:.6f} {sy_m / 1000.0:.6f} {sz_m / 1000.0:.6f} 0.0 {args.trise:.6f} {m0:.6e} "
            f"{mxx:.6f} {myy:.6f} {mzz:.6f} {myz:.6f} {mxz:.6f} {mxy:.6f}\n"
        )

    input_text = _build_input_text(
        title="forge_random_mt_f100",
        odir=(case_dir / "out"),
        station_file=st_file,
        source_file=src_file,
        model_nc=model_nc,
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
        dt_s=args.dt,
        xbeg_km=x_min / 1000.0,
        ybeg_km=y_min / 1000.0,
        zbeg_km=z_min_m / 1000.0,
        fq_max=args.fmax,
        trise=args.trise,
        vcut=args.vcut,
    )

    (case_dir / "input_random.inf").write_text(input_text, encoding="utf-8")
    in_dir = case_dir / "in"
    in_dir.mkdir(exist_ok=True)
    (in_dir / "input.inf").write_text(input_text, encoding="utf-8")

    print(f"Case written: {case_dir}")
    print(
        f"Random MT (mxx,myy,mzz,myz,mxz,mxy)=({mxx:.4f}, {myy:.4f}, {mzz:.4f}, {myz:.4f}, {mxz:.4f}, {mxy:.4f})"
    )
    print(f"M0 = {m0:.6e}")
    print(
        f"Grid nx,ny,nz = {nx},{ny},{nz}; dt={args.dt:.4f}s nt={nt}; fmax={args.fmax:.1f}Hz"
    )

    if args.run:
        cmd = [
            "mpirun",
            "-np",
            str(args.nproc_x * args.nproc_y),
            str(args.swpc_bin),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=case_dir, check=True)
        print("Run complete. Output directory:", case_dir / "out")


if __name__ == "__main__":
    main()
