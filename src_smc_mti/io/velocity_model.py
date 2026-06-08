from pathlib import Path
from typing import Union
import math

import numpy as np


def load_velocity_model(
    path: Union[str, Path],
    max_depth_m: float = 3000.0,
    fict_layer_dz: float = 500.0,
    target_depth: float = 3000.0,
) -> np.ndarray:
    """
    Load a layered forge.tvel-like model as rows:
    [thickness_m, Vp, Vs, rho, Qp, Qs].
    """
    path = Path(path)
    boundaries_m = []
    vp = []
    vs = []
    rho = []
    default_qp = 600.0
    default_qs = 300.0

    with path.open() as f:
        for line in f:
            if not line.strip() or line[0].isalpha():
                continue
            vals = line.split()
            if len(vals) < 4:
                continue
            depth_m = float(vals[0]) * 1000.0
            if depth_m > max_depth_m:
                break
            boundaries_m.append(depth_m)
            vp.append(float(vals[1]) * 1000.0)
            vs.append(float(vals[2]) * 1000.0)
            rho.append(float(vals[3]) * 1000.0)

    if not boundaries_m:
        raise ValueError(f"No layers parsed from {path}")

    cleaned = []
    for i, d in enumerate(boundaries_m):
        if i > 0 and math.isclose(d, cleaned[-1][0], rel_tol=0.0, abs_tol=1e-6):
            continue
        cleaned.append([d, vp[i], vs[i], rho[i]])

    layers = []
    for i, (top_m, vp_m, vs_m, rho_m) in enumerate(cleaned):
        thick_m = max(cleaned[i + 1][0] - top_m, 0.0) if i < len(cleaned) - 1 else 0.0
        layers.append([thick_m, vp_m, vs_m, rho_m, default_qp, default_qs])

    layers_arr = np.array(layers, dtype=float)
    total_depth = float(layers_arr[:-1, 0].sum()) if len(layers_arr) > 1 else 0.0
    target_depth = max(total_depth, target_depth)

    if total_depth < target_depth:
        vp_last, vs_last, rho_last, qp_last, qs_last = layers_arr[-1, 1:]
        extra_layers = []
        current_depth = total_depth
        while current_depth < target_depth:
            dz = min(fict_layer_dz, target_depth - current_depth)
            extra_layers.append([dz, vp_last, vs_last, rho_last, qp_last, qs_last])
            current_depth += dz
        layers_arr = np.vstack(
            [layers_arr[:-1], np.array(extra_layers, dtype=float), layers_arr[-1:]]
        )

    return layers_arr
