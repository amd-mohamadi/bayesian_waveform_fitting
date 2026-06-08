import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy import read


FNAME_RE = re.compile(
    r"^(?P<title>.+)\.3d\.(?P<station>st\d+)\.(?P<quantity>[UV])(?P<comp>[xyz])\.sac$",
    re.IGNORECASE,
)
BASIS_ORDER = ["Mxx", "Myy", "Mzz", "Mxy", "Mxz", "Myz"]
COMPONENT_ORDER = ["Z", "N", "E"]


def _station_sort_key(st: str) -> int:
    return int(st[2:])


def _load_velocity_sac_array(
    wav_dir: Path, station_labels: list[str]
) -> tuple[np.ndarray, float]:
    grouped: dict[str, dict[str, np.ndarray]] = {}
    dt_ref = None
    npts_ref = None
    for fp in sorted(wav_dir.glob("*.sac")):
        m = FNAME_RE.match(fp.name)
        if not m or m.group("quantity").upper() != "V":
            continue
        st = m.group("station").lower()
        comp = m.group("comp").lower()
        tr = read(str(fp))[0]
        y = np.asarray(tr.data, dtype=np.float32)
        dt = float(tr.stats.delta)
        npts = int(tr.stats.npts)
        if dt_ref is None:
            dt_ref = dt
            npts_ref = npts
        elif abs(dt - dt_ref) > 1e-9 or npts != npts_ref:
            raise ValueError(f"Inconsistent SAC metadata in {fp}")
        grouped.setdefault(st, {})[comp] = y

    labels_found = sorted(grouped, key=_station_sort_key)
    if labels_found != station_labels:
        raise ValueError(f"Station labels mismatch: {labels_found} vs {station_labels}")

    out = np.zeros((len(station_labels), 3, int(npts_ref)), dtype=np.float32)
    for i, st in enumerate(station_labels):
        comps = grouped[st]
        for need in ("z", "y", "x"):
            if need not in comps:
                raise ValueError(f"Missing V{need} for {st} in {wav_dir}")
        out[i, 0] = comps["z"]
        out[i, 1] = comps["y"]
        out[i, 2] = comps["x"]
    return out, float(dt_ref)


def _coefficients_from_random_metadata(metadata_path: Path) -> np.ndarray:
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    mt = metadata["random_mt"]
    m0 = float(mt["m0"])
    coeffs = np.asarray(
        [
            float(mt["mxx"]),
            float(mt["myy"]),
            float(mt["mzz"]),
            float(mt["mxy"]),
            float(mt["mxz"]),
            float(mt["myz"]),
        ],
        dtype=np.float64,
    )
    return coeffs * m0


def _plot_station(
    out_file: Path,
    station_code: str,
    direct: np.ndarray,
    reconstructed: np.ndarray,
    dt: float,
) -> None:
    t = np.arange(direct.shape[-1], dtype=float) * dt
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i, comp in enumerate(COMPONENT_ORDER):
        axes[i].plot(t, direct[i], color="black", lw=1.0, label="direct")
        axes[i].plot(
            t, reconstructed[i], color="#d55e00", lw=0.9, ls="--", label="GF sum"
        )
        axes[i].set_ylabel(comp)
        axes[i].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time since source (s)")
    fig.suptitle(f"{station_code} direct random MT vs GF reconstruction")
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a CAPE LHM GF library by reconstructing a direct random-MT run."
    )
    parser.add_argument("--gf-file", required=True)
    parser.add_argument("--random-case-dir", required=True)
    parser.add_argument(
        "--random-metadata",
        default=None,
        help="Defaults to <random-case-dir>/case_metadata.json",
    )
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--fail-rel-max", type=float, default=1e-4)
    args = parser.parse_args()

    gf_file = Path(args.gf_file)
    random_case_dir = Path(args.random_case_dir)
    metadata_path = (
        Path(args.random_metadata)
        if args.random_metadata
        else random_case_dir / "case_metadata.json"
    )

    with np.load(gf_file, allow_pickle=False) as data:
        gf_basis = np.asarray(data["gf_basis"], dtype=np.float32)
        basis_order = [str(x) for x in np.asarray(data["basis_order"])]
        component_order = [str(x) for x in np.asarray(data["component_order"])]
        station_labels = [str(x) for x in np.asarray(data["station_labels"])]
        station_codes = [str(x) for x in np.asarray(data["station_codes"])]
        quantity = str(np.asarray(data["quantity"]).reshape(()))
        dt = float(np.asarray(data["dt"]).reshape(()))

    if basis_order != BASIS_ORDER:
        raise ValueError(f"Unexpected basis_order={basis_order}")
    if component_order != COMPONENT_ORDER:
        raise ValueError(f"Unexpected component_order={component_order}")
    if quantity.upper() != "V":
        raise ValueError(f"GF library is not velocity: quantity={quantity}")
    if gf_basis.shape[:3] != (6, len(station_labels), 3):
        raise ValueError(f"Invalid GF shape: {gf_basis.shape}")

    direct, direct_dt = _load_velocity_sac_array(
        random_case_dir / "out" / "wav", station_labels
    )
    if direct.shape != gf_basis.shape[1:]:
        raise ValueError(f"Shape mismatch: direct={direct.shape}, gf={gf_basis.shape}")
    if abs(direct_dt - dt) > 1e-9:
        raise ValueError(f"dt mismatch: direct={direct_dt}, gf={dt}")

    coeffs = _coefficients_from_random_metadata(metadata_path)
    reconstructed = np.einsum("q,qnct->nct", coeffs, gf_basis, optimize=True).astype(
        np.float32
    )

    diff = reconstructed - direct
    max_direct = float(np.max(np.abs(direct)))
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
    rel_max = max_abs / max(max_direct, 1e-30)
    rel_rms = rms / max(
        float(np.sqrt(np.mean(direct.astype(np.float64) ** 2))), 1e-30
    )

    print(f"GF file: {gf_file}")
    print(f"gf_basis.shape: {gf_basis.shape}")
    print(f"basis_order: {basis_order}")
    print(f"component_order: {component_order}")
    print(f"quantity: {quantity}, dt={dt:.6f}, npts={gf_basis.shape[-1]}")
    print(f"stations: {station_codes}")
    print(f"max_abs_error: {max_abs:.6e}")
    print(f"rel_max_error: {rel_max:.6e}")
    print(f"rel_rms_error: {rel_rms:.6e}")

    if rel_max > float(args.fail_rel_max):
        raise RuntimeError(
            f"GF reconstruction failed: rel_max_error={rel_max:.6e} > {args.fail_rel_max:.6e}"
        )

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        for i, code in enumerate(station_codes):
            out_file = plot_dir / f"{code}.openswpc_lhm_gf_reconstruction.png"
            _plot_station(out_file, code, direct[i], reconstructed[i], dt)
        print(f"Wrote reconstruction plots to {plot_dir}")


if __name__ == "__main__":
    main()
