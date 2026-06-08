import numpy as np

from src_smc_mti.tape import MT33_MT6, Tape_MT33


PHYS_LOWER = np.array([-np.pi / 6, -np.pi / 2, 0.0, 0.0, -np.pi / 2], dtype=float)
PHYS_UPPER = np.array([np.pi / 6, np.pi / 2, 2 * np.pi, 1.0, np.pi / 2], dtype=float)


def weighted_posterior_medoid(particles, weights):
    """
    Weighted medoid in MT6 space.

    The medoid is the particle i minimizing sum_j w_j * d(i, j), where
    d(i,j) = 1 - dot(mt6_i, mt6_j).
    """
    n = particles.shape[0]
    w = np.asarray(weights, dtype=float)
    w /= np.sum(w)

    mt6 = np.zeros((n, 6), dtype=float)
    for i in range(n):
        g, d, k, h, s = particles[i]
        mt33 = Tape_MT33(g, d, k, h, s)
        v = MT33_MT6(mt33)
        nv = np.linalg.norm(v)
        if nv > 0:
            v = v / nv
        mt6[i] = v

    sim = np.clip(mt6 @ mt6.T, -1.0, 1.0)
    dist = 1.0 - sim
    objective = dist @ w
    idx = int(np.argmin(objective))
    return idx, float(objective[idx])


def decode_unit_to_physical(unit_particles):
    p = PHYS_LOWER[None, :] + unit_particles * (PHYS_UPPER - PHYS_LOWER)[None, :]
    p[:, 2] = np.mod(p[:, 2], 2 * np.pi)
    p[:, 0] = np.clip(p[:, 0], PHYS_LOWER[0], PHYS_UPPER[0])
    p[:, 1] = np.clip(p[:, 1], PHYS_LOWER[1], PHYS_UPPER[1])
    p[:, 3] = np.clip(p[:, 3], PHYS_LOWER[3], PHYS_UPPER[3])
    p[:, 4] = np.clip(p[:, 4], PHYS_LOWER[4], PHYS_UPPER[4])
    return p
