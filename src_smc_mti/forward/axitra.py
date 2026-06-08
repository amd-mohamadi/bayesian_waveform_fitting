import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from src_smc_mti.tape import Tape_MT6


AXITRA_SRC = (
    Path(__file__).resolve().parents[3]
    / "axitra"
    / "MOMENT_DISP_F90_OPENMP"
    / "src"
)
if str(AXITRA_SRC) not in sys.path:
    sys.path.append(str(AXITRA_SRC))

try:
    from axitra import Axitra, moment
except Exception:
    Axitra = None
    moment = None


class FastSynthesizer:
    """Axitra synthesizer with optional persistent Green-function cache."""

    def __init__(
        self,
        velocity_model,
        stations,
        source_loc,
        duration=1.0,
        fmax=500.0,
        t0=0.01,
        work_dir=None,
        cache_id=2387,
        generate_if_missing=True,
    ):
        self.velocity_model = velocity_model
        self.stations = stations
        self.source_loc = source_loc
        self.duration = duration
        self.fmax = fmax
        self.t0 = t0
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.cache_id = int(cache_id)
        self.generate_if_missing = bool(generate_if_missing)
        self.axpath = str(AXITRA_SRC)

        self.ap = None
        self._temp_dir = None
        self._orig_dir = None

    def _cache_metadata(self, sources: np.ndarray) -> dict:
        return {
            "backend": "axitra",
            "cache_id": self.cache_id,
            "duration": float(self.duration),
            "fmax": float(self.fmax),
            "t0": float(self.t0),
            "axpath": self.axpath,
            "velocity_model": np.asarray(self.velocity_model, dtype=float).tolist(),
            "stations": np.asarray(self.stations, dtype=float).tolist(),
            "sources": np.asarray(sources, dtype=float).tolist(),
        }

    def _cache_status(self, metadata_path: Path, metadata: dict) -> tuple[bool, str]:
        if not metadata_path.exists():
            return False, f"missing metadata file {metadata_path}"
        data_file = metadata_path.parent / f"axi_{self.cache_id}.data"
        res_file = metadata_path.parent / f"axi_{self.cache_id}.res"
        if (
            not data_file.exists()
            or not res_file.exists()
            or data_file.stat().st_size == 0
            or res_file.stat().st_size == 0
        ):
            return False, (
                f"missing or empty Axitra cache files axi_{self.cache_id}.data/"
                f"axi_{self.cache_id}.res in {metadata_path.parent}"
            )
        with metadata_path.open("r", encoding="utf-8") as f:
            old_metadata = json.load(f)
        if old_metadata != metadata:
            return False, f"metadata in {metadata_path} does not match this run"
        return True, "ok"

    def _cache_is_valid(self, metadata_path: Path, metadata: dict) -> bool:
        valid, _ = self._cache_status(metadata_path, metadata)
        return valid

    def _cache_result_exists(self, run_dir: Path) -> bool:
        data_file = run_dir / f"axi_{self.cache_id}.data"
        res_file = run_dir / f"axi_{self.cache_id}.res"
        return (
            data_file.exists()
            and res_file.exists()
            and data_file.stat().st_size > 0
            and res_file.stat().st_size > 0
        )

    def setup(self):
        """Compute or load Green's functions."""
        if Axitra is None or moment is None:
            raise RuntimeError(
                "Axitra backend requested, but the axitra Python module is not "
                "available. Use --synthetic-backend openswpc_gf or install/build "
                "Axitra and make its Python module importable."
            )

        sources = np.array(
            [[1, self.source_loc[0], self.source_loc[1], self.source_loc[2]]],
            dtype=float,
        )

        if self.work_dir is None:
            self._temp_dir = tempfile.mkdtemp()
            run_dir = Path(self._temp_dir)
        else:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            run_dir = self.work_dir
        self._orig_dir = os.getcwd()
        os.chdir(run_dir)

        metadata = self._cache_metadata(sources)
        metadata_path = run_dir / "axitra_greens_metadata.json"
        if self.work_dir is not None:
            cache_valid, cache_reason = self._cache_status(metadata_path, metadata)
        else:
            cache_valid, cache_reason = False, "no persistent work_dir"

        if self.work_dir is not None and cache_valid:
            print(f"Loading Axitra Green's functions from {run_dir}")
            self.ap = Axitra.read(suffix=str(self.cache_id), axpath=self.axpath)
            if self.ap is None:
                raise RuntimeError(f"Failed to load Axitra Green's functions from {run_dir}")
            self.ap.id = self.cache_id
            self.ap.sid = f"axi_{self.cache_id}"
            return

        if self.work_dir is not None and not self.generate_if_missing:
            raise RuntimeError(
                "Required Axitra Green's functions are missing or stale: "
                f"{cache_reason}. Run generate_cape_axitra_greens.py for this event "
                f"or pass a matching --axitra-greens-dir."
            )

        print("Computing Green's functions (one-time cost)...")
        t0 = time.time()

        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)

        try:
            self.ap = Axitra(
                self.velocity_model,
                self.stations,
                sources,
                fmax=self.fmax,
                duration=self.duration,
                xl=0.0,
                latlon=False,
                axpath=self.axpath,
                id=self.cache_id if self.work_dir is not None else None,
            )
            self.ap = moment.green(self.ap)
            if self.work_dir is not None and not self._cache_result_exists(run_dir):
                raise RuntimeError(
                    f"Axitra Green-function generation failed in {run_dir}; "
                    f"missing or empty axi_{self.cache_id}.res"
                )
            if self.work_dir is not None:
                with metadata_path.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, sort_keys=True)
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
            os.close(devnull)

        print(f"Green's functions computed in {time.time() - t0:.2f}s")

    def synthesize_batch(
        self, mt_params: np.ndarray, m0: float, source_delay: float = 0.1
    ) -> np.ndarray:
        """Synthesize waveforms for a batch of Tape moment-tensor parameters."""
        B = mt_params.shape[0]
        N = self.ap.nstation
        T = self.ap.npt

        waveforms = np.zeros((B, N, 3, T), dtype=np.float32)

        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)

        try:
            for i in range(B):
                gamma, delta, kappa, h, sigma = mt_params[i]
                mt6_norm = Tape_MT6(gamma, delta, kappa, h, sigma)

                mxx = mt6_norm[0] * m0
                myy = mt6_norm[1] * m0
                mzz = mt6_norm[2] * m0
                mxy = (mt6_norm[3] / np.sqrt(2.0)) * m0
                mxz = (mt6_norm[4] / np.sqrt(2.0)) * m0
                myz = (mt6_norm[5] / np.sqrt(2.0)) * m0

                hist_mt = np.array(
                    [[1, mxx, mxy, mxz, myy, myz, mzz, source_delay]], dtype=float
                )

                _, ux, uy, uz = moment.conv_tensor(
                    self.ap, hist_mt, source_type=1, t0=self.t0, unit=2
                )

                waveforms[i] = np.stack([uz, ux, uy], axis=1)
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
            os.close(devnull)

        return waveforms

    def cleanup(self):
        """Clean up temporary Axitra work directory."""
        if self._orig_dir:
            os.chdir(self._orig_dir)
        if self._temp_dir:
            import shutil

            shutil.rmtree(self._temp_dir, ignore_errors=True)
