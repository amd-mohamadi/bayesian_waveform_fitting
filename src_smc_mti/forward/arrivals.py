import numpy as np


def calculate_arrival_time(
    src_depth_m: float,
    rcv_depth_m: float,
    dist_m: float,
    model: np.ndarray,
    phase: str = "P",
) -> float:
    """Approximate first-arrival time for a ray in a 1D layered model."""
    if src_depth_m < rcv_depth_m:
        src_depth_m, rcv_depth_m = rcv_depth_m, src_depth_m

    current_depth = 0.0
    active_layers = []
    v_idx = 1 if phase.upper() == "P" else 2

    for row in model:
        thick = float(row[0])
        vel = float(row[v_idx])
        layer_top = current_depth
        layer_bot = current_depth + thick
        overlap_top = max(layer_top, rcv_depth_m)
        overlap_bot = min(layer_bot, src_depth_m)
        if overlap_bot > overlap_top:
            active_layers.append((overlap_bot - overlap_top, vel))
        current_depth += thick
        if current_depth >= src_depth_m:
            break

    if current_depth < src_depth_m:
        h = src_depth_m - max(current_depth, rcv_depth_m)
        if h > 0:
            active_layers.append((h, float(model[-1, v_idx])))

    if not active_layers:
        vel = float(model[0, v_idx])
        return dist_m / max(vel, 1e-6)

    v_max = max(v for _, v in active_layers)
    p_limit = 1.0 / v_max

    def horizontal_distance(p: float) -> float:
        x = 0.0
        for h, v in active_layers:
            if p * v >= 1.0:
                return float("inf")
            tan_theta = (p * v) / np.sqrt(1.0 - (p * v) ** 2)
            x += h * tan_theta
        return x

    p_min = 0.0
    p_max = p_limit * 0.999999
    for _ in range(50):
        p_mid = 0.5 * (p_min + p_max)
        if horizontal_distance(p_mid) < dist_m:
            p_min = p_mid
        else:
            p_max = p_mid
    p_sol = 0.5 * (p_min + p_max)

    t_travel = 0.0
    for h, v in active_layers:
        cos_theta = np.sqrt(1.0 - (p_sol * v) ** 2)
        t_travel += h / (v * cos_theta)

    return float(t_travel)
