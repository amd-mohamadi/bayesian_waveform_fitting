import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np


CAPE_DEFAULTS = {
    "duration": 7.0,
    "p_window": 0.30,
    "s_window": 0.60,
    "time_steps": 100,
    "bp_p_low": 2.0,
    "bp_p_high": 30.0,
    "bp_s_low": 2.0,
    "bp_s_high": 30.0,
    "fmax": 30.0,
}


def _default_gf(event_id: str) -> Path:
    if event_id == "eq02387":
        return (
            Path("cape_events")
            / event_id
            / "gf_library_v_lhm_mediumvs_f67_dx60.npz"
        )
    return Path("cape_events") / event_id / "gf_library_v.npz"


def build_command(args: argparse.Namespace) -> list[str]:
    here = Path(__file__).resolve().parent
    event_dir = Path(args.event_dir) if args.event_dir else Path("cape_events") / args.event_id

    cmd = [
        sys.executable,
        str(here / "run_inversion.py"),
        "--event-dir",
        str(event_dir),
        "--stations-dir",
        str(Path(args.stations_dir)),
        "--velocity-model",
        str(Path(args.velocity_model)),
        "--synthetic-backend",
        args.synthetic_backend,
        "--likelihood",
        args.likelihood,
        "--sampler",
        args.sampler,
        "--n-particles",
        str(args.n_particles),
        "--n-stages",
        str(args.n_stages),
        "--components",
        "PZ,SN,SE",
        "--duration",
        str(args.duration),
        "--fmax",
        str(args.fmax),
        "--source-delay",
        str(args.source_delay),
        "--synthetic-window-source",
        "picks",
        "--phase-window-p-len",
        str(args.p_window),
        "--phase-window-s-len",
        str(args.s_window),
        "--synthetic-phase-window-p-len",
        str(args.p_window),
        "--synthetic-phase-window-s-len",
        str(args.s_window),
        "--time-steps",
        str(args.time_steps),
        "--bp-p-low",
        str(args.bp_p_low),
        "--bp-p-high",
        str(args.bp_p_high),
        "--bp-s-low",
        str(args.bp_s_low),
        "--bp-s-high",
        str(args.bp_s_high),
        "--source-target-freq-hz",
        str(args.source_target_freq_hz),
    ]
    if args.synthetic_backend == "openswpc_gf":
        gf_file = (
            Path(args.openswpc_gf_file)
            if args.openswpc_gf_file
            else _default_gf(args.event_id)
        )
        cmd.extend(["--openswpc-gf-file", str(gf_file)])
    else:
        greens_dir = (
            Path(args.axitra_greens_dir)
            if args.axitra_greens_dir
            else event_dir / "axitra_greens"
        )
        cmd.extend(["--axitra-greens-dir", str(greens_dir)])
    if args.dc_only:
        cmd.append("--dc-only")
    if args.extra_args:
        cmd.extend(args.extra_args)
    return cmd


def validate_event_inputs(event_dir: Path) -> dict:
    inv_path = event_dir / "invdata.pkl"
    if not inv_path.exists():
        raise FileNotFoundError(f"Missing CAPE invdata: {inv_path}")

    with inv_path.open("rb") as f:
        invdata = pickle.load(f)
    hint = invdata.get("preprocess_hint", {})
    if not hint.get("remove_response", False) or hint.get("response_output") != "VEL":
        raise ValueError(
            f"{inv_path} is not velocity-prepared. Re-run prepare_cape_invdata.py "
            "with response removal enabled and --response-output VEL."
        )
    if invdata.get("component_order") != "ENZ":
        raise ValueError(f"{inv_path} does not store ENZ waveform components")
    return invdata


def validate_inputs(event_dir: Path, gf_file: Path) -> None:
    invdata = validate_event_inputs(event_dir)
    if not gf_file.exists():
        raise FileNotFoundError(f"Missing OpenSWPC velocity GF library: {gf_file}")

    with np.load(gf_file, allow_pickle=False) as gf:
        gf_shape = tuple(np.asarray(gf["gf_basis"]).shape)
        basis_order = (
            [str(x) for x in np.asarray(gf["basis_order"])]
            if "basis_order" in gf.files
            else []
        )
        quantity = (
            str(np.asarray(gf["quantity"]).reshape(()))
            if "quantity" in gf.files
            else "unknown"
        )
        comp_order = (
            [str(x) for x in np.asarray(gf["component_order"])]
            if "component_order" in gf.files
            else []
        )
    if quantity.upper() != "V":
        raise ValueError(f"{gf_file} is not a velocity GF library: quantity={quantity}")
    if gf_shape[0] != 6 or gf_shape[2] != 3:
        raise ValueError(f"{gf_file} has invalid gf_basis shape={gf_shape}")
    if gf_shape[1] != len(invdata.get("station_ids", [])):
        raise ValueError(
            f"{gf_file} station count {gf_shape[1]} does not match invdata "
            f"station count {len(invdata.get('station_ids', []))}"
        )
    if basis_order != ["Mxx", "Myy", "Mzz", "Mxy", "Mxz", "Myz"]:
        raise ValueError(f"{gf_file} has unexpected basis_order={basis_order}")
    if comp_order != ["Z", "N", "E"]:
        raise ValueError(f"{gf_file} has unexpected component_order={comp_order}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CAPE waveform inversion with local defaults"
    )
    parser.add_argument("event_id", choices=("eq02387", "eq02883"))
    parser.add_argument("--event-dir", default=None)
    parser.add_argument("--stations-dir", default="stations")
    parser.add_argument("--velocity-model", default="cape.tvel")
    parser.add_argument(
        "--synthetic-backend",
        choices=("axitra", "openswpc_gf"),
        default="axitra",
        help="Synthetic backend. Axitra is the default; OpenSWPC GF is retained as a reference option.",
    )
    parser.add_argument("--openswpc-gf-file", default=None)
    parser.add_argument(
        "--axitra-greens-dir",
        default=None,
        help="Directory for persistent Axitra Green's functions. Defaults to <event-dir>/axitra_greens.",
    )
    parser.add_argument("--likelihood", choices=("gsot", "softdtw", "l2"), default="gsot")
    parser.add_argument("--sampler", choices=("smc", "cmaes"), default="smc")
    parser.add_argument("--n-particles", type=int, default=500)
    parser.add_argument("--n-stages", type=int, default=20)
    parser.add_argument("--duration", type=float, default=CAPE_DEFAULTS["duration"])
    parser.add_argument("--p-window", type=float, default=CAPE_DEFAULTS["p_window"])
    parser.add_argument("--s-window", type=float, default=CAPE_DEFAULTS["s_window"])
    parser.add_argument("--time-steps", type=int, default=CAPE_DEFAULTS["time_steps"])
    parser.add_argument("--fmax", type=float, default=CAPE_DEFAULTS["fmax"])
    parser.add_argument("--bp-p-low", type=float, default=CAPE_DEFAULTS["bp_p_low"])
    parser.add_argument("--bp-p-high", type=float, default=CAPE_DEFAULTS["bp_p_high"])
    parser.add_argument("--bp-s-low", type=float, default=CAPE_DEFAULTS["bp_s_low"])
    parser.add_argument("--bp-s-high", type=float, default=CAPE_DEFAULTS["bp_s_high"])
    parser.add_argument("--source-delay", type=float, default=0.0)
    parser.add_argument(
        "--source-target-freq-hz",
        type=float,
        default=30.0,
        help="Used by Axitra only; OpenSWPC GF frequency is fixed by the packed GF",
    )
    parser.add_argument("--dc-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra_args = parser.parse_known_args()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    if "--prepare-greens-only" in extra_args:
        parser.error(
            "--prepare-greens-only was removed; use generate_cape_axitra_greens.py "
            f"{args.event_id} before running inversion"
        )
    args.extra_args = extra_args

    event_dir = Path(args.event_dir) if args.event_dir else Path("cape_events") / args.event_id

    cmd = build_command(args)
    if args.dry_run:
        print(" ".join(cmd))
        return
    if args.synthetic_backend == "openswpc_gf":
        gf_file = (
            Path(args.openswpc_gf_file)
            if args.openswpc_gf_file
            else _default_gf(args.event_id)
        )
        validate_inputs(event_dir, gf_file)
    else:
        validate_event_inputs(event_dir)
    subprocess.run(cmd, cwd=Path(__file__).resolve().parent, check=True)


if __name__ == "__main__":
    main()
