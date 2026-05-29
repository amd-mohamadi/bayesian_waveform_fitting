import argparse
import re
from pathlib import Path
import sys

import numpy as np
from obspy import read

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_simulated_inversion import load_stations_from_xml


FNAME_RE = re.compile(
    r"^(?P<title>.+)\.3d\.(?P<station>st\d+)\.(?P<quantity>[UV])(?P<comp>[xyz])\.sac$",
    re.IGNORECASE,
)

BASIS_ORDER = ["Mxx", "Myy", "Mzz", "Mxy", "Mxz", "Myz"]


def _station_sort_key(st: str) -> int:
    return int(st[2:])


def _load_basis_waveforms(case_dir: Path, basis: str, quantity: str) -> dict:
    wav_dir = case_dir / basis / "out" / "wav"
    if not wav_dir.exists():
        raise FileNotFoundError(f"Missing waveform directory: {wav_dir}")

    q = quantity.upper()
    grouped: dict[str, dict[str, np.ndarray]] = {}
    dt_ref = None
    npts_ref = None

    for fp in sorted(wav_dir.glob("*.sac")):
        m = FNAME_RE.match(fp.name)
        if not m:
            continue
        if m.group("quantity").upper() != q:
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
        else:
            if abs(dt - dt_ref) > 1e-9:
                raise ValueError(f"Inconsistent dt in {fp}: {dt} vs {dt_ref}")
            if npts != npts_ref:
                raise ValueError(f"Inconsistent npts in {fp}: {npts} vs {npts_ref}")

        grouped.setdefault(st, {})[comp] = y

    if not grouped:
        raise RuntimeError(f"No {q} SAC traces found in {wav_dir}")

    # convert to array in station order, component order [Z, N, E] = [z, y, x]
    stations = sorted(grouped.keys(), key=_station_sort_key)
    arr = np.zeros((len(stations), 3, npts_ref), dtype=np.float32)
    for i, st in enumerate(stations):
        comp_dict = grouped[st]
        for need in ("x", "y", "z"):
            if need not in comp_dict:
                raise ValueError(f"Missing component {need} for {basis}/{st}")
        arr[i, 0, :] = comp_dict["z"]
        arr[i, 1, :] = comp_dict["y"]
        arr[i, 2, :] = comp_dict["x"]

    return {
        "stations": stations,
        "array": arr,
        "dt": dt_ref,
        "npts": npts_ref,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack OpenSWPC MT basis SAC outputs into NPZ GF library"
    )
    parser.add_argument(
        "--case-dir", required=True, help="OpenSWPC case directory with Mxx/.../Mxy"
    )
    parser.add_argument(
        "--stations-dir", required=True, help="StationXML directory used for inversion"
    )
    parser.add_argument("--output", required=True, help="Output NPZ file path")
    parser.add_argument(
        "--quantity",
        default="V",
        choices=["U", "V", "u", "v"],
        help="Use displacement (U) or velocity (V) SAC traces",
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    basis_arrays = []
    station_labels_ref = None
    dt_ref = None
    npts_ref = None
    for basis in BASIS_ORDER:
        loaded = _load_basis_waveforms(case_dir, basis, args.quantity)
        if station_labels_ref is None:
            station_labels_ref = loaded["stations"]
            dt_ref = loaded["dt"]
            npts_ref = loaded["npts"]
        else:
            if loaded["stations"] != station_labels_ref:
                raise ValueError(f"Station mismatch in basis {basis}")
            if abs(loaded["dt"] - dt_ref) > 1e-9:
                raise ValueError(f"dt mismatch in basis {basis}")
            if int(loaded["npts"]) != int(npts_ref):
                raise ValueError(f"npts mismatch in basis {basis}")
        basis_arrays.append(loaded["array"])

    gf_basis = np.stack(basis_arrays, axis=0)  # (6, N, 3, T)

    stations_all, codes_all, _, _ = load_stations_from_xml(args.stations_dir)
    stations_all = np.asarray(stations_all, dtype=np.float64)
    if stations_all.shape[0] < len(station_labels_ref):
        raise ValueError("StationXML count is smaller than OpenSWPC stXXX count")

    # st001..stNN follow the order used to write stloc.xy in setup script,
    # which uses load_stations_from_xml order.
    n = len(station_labels_ref)
    gf_station_coords = stations_all[:n, :]
    gf_station_codes = np.asarray(codes_all[:n])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        gf_basis=gf_basis,
        basis_order=np.asarray(BASIS_ORDER),
        station_labels=np.asarray(station_labels_ref),
        station_codes=gf_station_codes,
        station_coords=gf_station_coords,
        dt=np.float32(dt_ref),
        npts=np.int32(npts_ref),
        quantity=np.asarray(args.quantity.upper()),
        component_order=np.asarray(["Z", "N", "E"]),
    )
    print(f"Saved GF library: {out}")
    print(f"Shape gf_basis: {gf_basis.shape} (basis, station, comp, time)")


if __name__ == "__main__":
    main()
