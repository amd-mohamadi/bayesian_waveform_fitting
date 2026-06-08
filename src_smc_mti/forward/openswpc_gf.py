from types import SimpleNamespace

import numpy as np

from src_smc_mti.tape import Tape_MT6


class OpenSWPCGFSynthesizer:
    """Fast synthesizer from precomputed OpenSWPC MT-basis Green's functions."""

    def __init__(
        self, gf_file, stations, source_loc=None, duration=1.0, fmax=500.0, t0=0.01
    ):
        self.gf_file = str(gf_file)
        self.stations = np.asarray(stations, dtype=float)
        self.source_loc = source_loc
        self.duration = duration
        self.fmax = fmax
        self.t0 = t0

        self._gf = None
        self._dt = None
        self.ap = None

    def setup(self):
        data = np.load(self.gf_file, allow_pickle=False)
        gf_basis = np.asarray(data["gf_basis"], dtype=np.float32)
        station_coords = np.asarray(data["station_coords"], dtype=float)
        dt = float(np.asarray(data["dt"]).reshape(()))
        basis_order = (
            [str(x) for x in np.asarray(data["basis_order"])]
            if "basis_order" in data.files
            else []
        )
        component_order = (
            [str(x) for x in np.asarray(data["component_order"])]
            if "component_order" in data.files
            else []
        )
        quantity = (
            str(np.asarray(data["quantity"]).reshape(()))
            if "quantity" in data.files
            else ""
        )

        if gf_basis.ndim != 4 or gf_basis.shape[0] != 6 or gf_basis.shape[2] != 3:
            raise ValueError(
                f"Invalid gf_basis shape: {gf_basis.shape}, expected (6,N,3,T)"
            )
        if basis_order != ["Mxx", "Myy", "Mzz", "Mxy", "Mxz", "Myz"]:
            raise ValueError(f"Unexpected OpenSWPC GF basis_order: {basis_order}")
        if component_order != ["Z", "N", "E"]:
            raise ValueError(
                f"Unexpected OpenSWPC GF component_order: {component_order}"
            )
        if quantity.upper() != "V":
            raise ValueError(
                f"OpenSWPC GF library must contain velocity, got {quantity}"
            )

        req_xyz = self.stations[:, 1:4]
        gf_xyz = station_coords[:, 1:4]
        idx = []
        for p in req_xyz:
            d2 = np.sum((gf_xyz - p[None, :]) ** 2, axis=1)
            idx.append(int(np.argmin(d2)))

        self._gf = gf_basis[:, np.asarray(idx, dtype=int), :, :]
        self._dt = dt
        self.duration = dt * float(self._gf.shape[3])
        self.ap = SimpleNamespace(nstation=self._gf.shape[1], npt=self._gf.shape[3])

    def synthesize_batch(
        self, mt_params: np.ndarray, m0: float, source_delay: float = 0.1
    ) -> np.ndarray:
        if self._gf is None:
            raise RuntimeError(
                "OpenSWPC GF synthesizer not initialized. Call setup() first."
            )

        B = mt_params.shape[0]
        coeffs = np.zeros((B, 6), dtype=np.float64)
        for i in range(B):
            gamma, delta, kappa, h, sigma = mt_params[i]
            mt6 = Tape_MT6(gamma, delta, kappa, h, sigma)
            coeffs[i, 0] = mt6[0] * m0
            coeffs[i, 1] = mt6[1] * m0
            coeffs[i, 2] = mt6[2] * m0
            coeffs[i, 3] = (mt6[3] / np.sqrt(2.0)) * m0
            coeffs[i, 4] = (mt6[4] / np.sqrt(2.0)) * m0
            coeffs[i, 5] = (mt6[5] / np.sqrt(2.0)) * m0

        return np.einsum("bq,qnct->bnct", coeffs, self._gf, optimize=True).astype(
            np.float32
        )

    def cleanup(self):
        return
