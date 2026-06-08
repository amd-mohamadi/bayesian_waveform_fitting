import math

import numpy as np


def _tpb2q(t: list[float], p: list[float], b: list[float]) -> np.ndarray:
    """Convert T/P/B axes to a unit quaternion for Kagan angle."""
    eps = 0.001
    tqw = 1.0 + t[0] + p[1] + b[2]
    tqx = 1.0 + t[0] - p[1] - b[2]
    tqy = 1.0 - t[0] + p[1] - b[2]
    tqz = 1.0 - t[0] - p[1] + b[2]

    q = np.zeros(4, dtype=float)
    if tqw > eps:
        q[0] = 0.5 * math.sqrt(tqw)
        q[1:] = (p[2] - b[1], b[0] - t[2], t[1] - p[0])
    elif tqx > eps:
        q[0] = 0.5 * math.sqrt(tqx)
        q[1:] = (p[2] - b[1], p[0] + t[1], b[0] + t[2])
    elif tqy > eps:
        q[0] = 0.5 * math.sqrt(tqy)
        q[1:] = (b[0] - t[2], p[0] + t[1], b[1] + p[2])
    elif tqz > eps:
        q[0] = 0.5 * math.sqrt(tqz)
        q[1:] = (t[1] - p[0], b[0] + t[2], b[1] + p[2])
    else:
        raise RuntimeError("Kagan angle computation failed: invalid quaternion.")

    q[1:] /= 4.0 * q[0]
    q /= math.sqrt(float(np.sum(q * q)))
    return q


def kagan_angle_deg(mt1: np.ndarray, mt2: np.ndarray) -> float:
    """
    Given two 3x3 moment tensors, return the Kagan angle in degrees.

    After Kagan (1991) and Tape & Tape (2012).
    """
    pbt2tpb = np.array(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    ai = pbt2tpb @ mt1.T
    aj = pbt2tpb @ mt2.T
    u = ai @ aj.T
    tk, pk, bk = u.tolist()
    qk = _tpb2q(tk, pk, bk)
    return float(2.0 * (180.0 / np.pi) * math.acos(np.max(np.abs(qk))))
