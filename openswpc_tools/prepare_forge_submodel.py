import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def _fill_nans(arr: np.ndarray) -> np.ndarray:
    """Fill NaNs in a 3D array with nearest-neighbor values."""
    out = arr.copy()
    mask = np.isnan(out)
    if not np.any(mask):
        return out

    valid_idx = np.argwhere(~mask)
    if valid_idx.size == 0:
        raise ValueError("Input array contains only NaNs")

    nan_idx = np.argwhere(mask)
    for i, j, k in nan_idx:
        d2 = np.sum((valid_idx - np.array([i, j, k])) ** 2, axis=1)
        nearest = valid_idx[int(np.argmin(d2))]
        out[i, j, k] = out[tuple(nearest)]
    return out


def _brocher_density_gcc(vp_kms: np.ndarray) -> np.ndarray:
    """Brocher (2005) empirical density relation, vp in km/s, output g/cc."""
    rho = (
        1.6612 * vp_kms
        - 0.4721 * vp_kms**2
        + 0.0671 * vp_kms**3
        - 0.0043 * vp_kms**4
        + 0.000106 * vp_kms**5
    )
    return np.clip(rho, 1.0, 3.5)


def main():
    parser = argparse.ArgumentParser(
        description="Crop FORGE 3D NetCDF model to a compact OpenSWPC submodel"
    )
    parser.add_argument("--input-nc", required=True, help="Input FORGE NetCDF model")
    parser.add_argument(
        "--output-nc", required=True, help="Output cropped NetCDF model"
    )
    parser.add_argument(
        "--output-meta",
        default=None,
        help="Optional JSON metadata output path (defaults to output-nc with .json)",
    )
    parser.add_argument(
        "--ref-easting",
        type=float,
        default=334641.1891,
        help="Reference easting for local x=0 (meters)",
    )
    parser.add_argument(
        "--ref-northing",
        type=float,
        default=4263443.693,
        help="Reference northing for local y=0 (meters)",
    )
    parser.add_argument(
        "--ref-elevation",
        type=float,
        default=1650.0249,
        help="Reference elevation for local z=0 (meters, positive up)",
    )
    parser.add_argument(
        "--x-halfwidth-m",
        type=float,
        default=2000.0,
        help="Half-width around ref easting to keep (meters)",
    )
    parser.add_argument(
        "--y-halfwidth-m",
        type=float,
        default=2000.0,
        help="Half-width around ref northing to keep (meters)",
    )
    parser.add_argument(
        "--z-min-m",
        type=float,
        default=-200.0,
        help="Minimum local z to keep in meters (positive down; can be negative for air)",
    )
    parser.add_argument(
        "--z-max-m",
        type=float,
        default=4500.0,
        help="Maximum local z to keep in meters (positive down)",
    )
    args = parser.parse_args()

    in_nc = Path(args.input_nc)
    out_nc = Path(args.output_nc)
    out_meta = (
        Path(args.output_meta) if args.output_meta else out_nc.with_suffix(".json")
    )

    ds = xr.open_dataset(in_nc)
    if not {"Vp", "Vs", "easting", "northing", "elevation"}.issubset(ds.variables):
        raise ValueError(
            "Input model must contain Vp, Vs, easting, northing, elevation"
        )

    e_min = args.ref_easting - args.x_halfwidth_m
    e_max = args.ref_easting + args.x_halfwidth_m
    n_min = args.ref_northing - args.y_halfwidth_m
    n_max = args.ref_northing + args.y_halfwidth_m

    elev_min = args.ref_elevation - args.z_max_m
    elev_max = args.ref_elevation - args.z_min_m

    ds_sub = ds.sel(
        easting=slice(e_min, e_max),
        northing=slice(n_min, n_max),
        elevation=slice(elev_min, elev_max),
    )

    if ds_sub.easting.size < 2 or ds_sub.northing.size < 2 or ds_sub.elevation.size < 2:
        raise ValueError("Selected submodel is too small. Increase crop bounds.")

    vp = ds_sub["Vp"].values.astype(np.float32)
    vs = ds_sub["Vs"].values.astype(np.float32)

    vp = _fill_nans(vp)
    vs = _fill_nans(vs)
    rho = _brocher_density_gcc(vp).astype(np.float32)

    ds_out = xr.Dataset(
        data_vars={
            "Vp": (("northing", "easting", "elevation"), vp),
            "Vs": (("northing", "easting", "elevation"), vs),
            "Rho": (("northing", "easting", "elevation"), rho),
        },
        coords={
            "easting": ds_sub["easting"].values,
            "northing": ds_sub["northing"].values,
            "elevation": ds_sub["elevation"].values,
        },
        attrs={
            "description": "Cropped FORGE model for OpenSWPC user-model mode",
            "vp_unit": "km/s",
            "vs_unit": "km/s",
            "rho_unit": "g/cc",
            "reference_easting_m": float(args.ref_easting),
            "reference_northing_m": float(args.ref_northing),
            "reference_elevation_m": float(args.ref_elevation),
            "local_x_m": "easting - reference_easting_m",
            "local_y_m": "northing - reference_northing_m",
            "local_z_m": "reference_elevation_m - elevation",
        },
    )

    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_nc)

    meta = {
        "input_nc": str(in_nc),
        "output_nc": str(out_nc),
        "reference": {
            "easting_m": float(args.ref_easting),
            "northing_m": float(args.ref_northing),
            "elevation_m": float(args.ref_elevation),
        },
        "crop": {
            "easting_m": [float(e_min), float(e_max)],
            "northing_m": [float(n_min), float(n_max)],
            "elevation_m": [float(elev_min), float(elev_max)],
            "local_z_m": [float(args.z_min_m), float(args.z_max_m)],
        },
        "shape": {
            "northing": int(ds_out.sizes["northing"]),
            "easting": int(ds_out.sizes["easting"]),
            "elevation": int(ds_out.sizes["elevation"]),
        },
        "spacing_m": {
            "de": float(ds_out.easting.values[1] - ds_out.easting.values[0]),
            "dn": float(ds_out.northing.values[1] - ds_out.northing.values[0]),
            "dz_elevation": float(
                ds_out.elevation.values[1] - ds_out.elevation.values[0]
            ),
        },
    }

    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote submodel: {out_nc}")
    print(f"Wrote metadata: {out_meta}")
    print(
        "Shape (northing,easting,elevation) = "
        f"({meta['shape']['northing']}, {meta['shape']['easting']}, {meta['shape']['elevation']})"
    )


if __name__ == "__main__":
    main()
