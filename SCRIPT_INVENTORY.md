# Bayesian Waveform Fitting Script Inventory

This inventory records the current waveform-fitting pieces found in the mixed
`focal_inversion` workspace. It is intended as the starting checklist for
copying a clean, continued project into `bayesian_waveform_fitting`.

## Current workflow

The active workflow appears to be:

1. Prepare real event data from `MSEED/<event_id>/`, `picks.dat`,
   `FORGE_catalog.csv`, and `stationxml/`.
2. Run `run_cape_inversion.py` with the default Axitra backend for fixed-source
   CAPE moment-tensor fitting.
3. Use OpenSWPC GF tools only for optional/reference waveform checks.

## Primary scripts to copy

These are the main project scripts for the current waveform fitting workflow.

| Source path | Role |
| --- | --- |
| `prepare_invdata.py` | Builds `invdata.pkl` for a real event from mseed waveforms, EQNet picks, catalog metadata, and StationXML. |
| `run_cape_inversion.py` | CAPE wrapper with local defaults. Uses Axitra by default and keeps `openswpc_gf` as an optional backend. |
| `run_inversion.py` | Main inversion driver. Supports `--synthetic-backend axitra/openswpc_gf`, `--sampler smc`, `--likelihood gsot/softdtw/l2`, ray-polarity terms, amplitude-ratio terms, and adaptive GSOT trace weighting. |
| `src_smc_mti/forward/` | Provides active Axitra/OpenSWPC-GF synthesizers and arrival-time helpers used by `run_inversion.py`. |
| `src_smc_mti/io/` | Provides active station, velocity-model, and observation loaders. |
| `src_smc_mti/waveform_likelihoods.py` | Provides active `SoftDTWLikelihood`, `GSOTLikelihood`, and `L2Likelihood`. |
| `src_smc_mti/inversion/posterior.py` | Provides posterior medoid and unit-parameter decode helpers. |
| `src_smc_mti/plot/beachball.py` | Provides active amplitude beachball plotting helper. |
| `generate_axitra_random_waveforms.py` | Quick Axitra random-MT waveform plotting diagnostic for CAPE events. |
| `archive/legacy/run_simulated_inversion.py` | Legacy standalone synthetic/Siamese script. Active shared utilities have been extracted to `src_smc_mti`. |
| `archive/legacy/synthetic_inversion_softdtw_test.py` | Legacy synthetic Soft-DTW/GSOT demo script. Active likelihood/posterior/plot helpers have been extracted to `src_smc_mti`. |

## Optional OpenSWPC support scripts

Copy these only if continuing with OpenSWPC reference synthetics.

| Source path | Role |
| --- | --- |
| `openswpc_tools/prepare_forge_submodel.py` | Crops the FORGE 3D NetCDF velocity model and writes density for OpenSWPC user-model input. |
| `openswpc_tools/setup_single_source_case.py` | Creates fixed-source OpenSWPC MT-basis case directories and `run_all_basis.sh`. |
| `openswpc_tools/pack_basis_to_npz.py` | Packs six OpenSWPC MT-basis SAC outputs into `gf_library*.npz` for fast inversion. |
| `openswpc_tools/plot_openswpc_waveforms.py` | Quick plotting utility for OpenSWPC SAC waveform outputs. |
| `openswpc_tools/run_random_mt_forward.py` | Diagnostic single random-MT OpenSWPC forward run. Useful for sanity checks, not required for inversion. |
| `openswpc_tools/__init__.py` | Package marker. |

## Experimental CAPE OpenSWPC scripts

These are retained for historical/reference LHM and native-Green experiments,
but they are outside the active Axitra inversion path.

| Source path | Role |
| --- | --- |
| `experimental/openswpc_tools/run_cape_lhm_forward.py` | CAPE LHM OpenSWPC forward experiment. |
| `experimental/openswpc_tools/setup_cape_lhm_gf_cases.py` | CAPE LHM Green-function case generator. |
| `experimental/openswpc_tools/setup_cape_lhm_native_green_cases.py` | CAPE native-Green experiment generator. |
| `experimental/openswpc_tools/pack_native_green_grid_to_npz.py` | Packs native Green grid outputs. |
| `experimental/openswpc_tools/verify_lhm_gf_reconstruction.py` | Verifies LHM GF reconstruction. |
| `experimental/openswpc_tools/score_lhm_case.py` | Scores LHM OpenSWPC cases. |
| `experimental/openswpc_tools/plot_lhm_observed_compare.py` | Diagnostic observed/synthetic comparison plot. |
| `experimental/openswpc_tools/plot_random_mt_station_waveforms.py` | Diagnostic random-MT waveform plotting utility. |
| `experimental/openswpc_tools/cape_lhm_*_vs.dat` | LHM velocity-model variants for reference experiments. |

## Local SMCMTI helpers

`run_inversion.py` imports these directly for the ray-pattern polarity term.
The active workflow now imports Tape conversion and forward helpers directly
from `src_smc_mti` package modules.

| Source path | Role |
| --- | --- |
| `src_smc_mti/data_loader.py` | Reads event/pick data for polarity constraints. |
| `src_smc_mti/data_prep.py` | Builds polarity matrices. |
| `src_smc_mti/likelihoods.py` | Provides polarity and amplitude-ratio likelihood helpers. |
| `src_smc_mti/tape.py` | Tape & Tape moment tensor conversion functions. |
| `src_smc_mti/moment_tesnor_conversion.py` | Larger moment tensor conversion library used by plotting/utilities. Filename typo is existing code. |
| `src_smc_mti/plot/` | Focal-sphere, beachball, and waveform plotting support. |

## Documentation to carry over

| Source path | Role |
| --- | --- |
| `INVERSION_REAL_EVENT.md` | Current real-event preparation and inversion workflow. |
| `OPENSWPC_SINGLE_SOURCE_SETUP.md` | Best record of the OpenSWPC single-source and MT-basis workflow, including current GSOT+SMC command examples. |
| `SPECFEM3D_OPENSWPC.md` | Architecture notes comparing SPECFEM3D/OpenSWPC GF generation. Useful background, not required to run. |
| `Soft-DTW.md` | Concept notes for the Soft-DTW objective. |

## Inputs and generated artifacts

These are not scripts but are needed to reproduce the current event workflow.

| Path | Role |
| --- | --- |
| `MSEED/1111911135/` | Real event waveforms, `picks.dat`, and existing `invdata.pkl`. |
| `stationxml/` | Station metadata used for waveform rotation and geometry. |
| `FORGE_catalog.csv` | Event source metadata used by `prepare_invdata.py`. |
| `cape.tvel` | CAPE 1D model used for Axitra synthetics, arrival-time windows, and polarity calculations. |
| `../axitra/MOMENT_DISP_F90_OPENMP/src` | External Axitra build used by the default CAPE synthetic backend. |
| `openswpc_model/` | Cropped OpenSWPC model outputs. |
| `openswpc_cases/` and `o20/` | Generated OpenSWPC basis runs and packed GF libraries. Needed only for optional OpenSWPC GF checks. |

## Likely legacy or secondary code

These paths are related but should not be first-pass copied into the clean
Bayesian waveform-fitting project.

| Path | Reason |
| --- | --- |
| `prepare_waveform.py` | Older exploratory waveform-preparation script. It has hard-coded external paths and an apparent `stid` typo. Superseded by `prepare_invdata.py`. |
| `synthetic_inversion_test.py` | Older Siamese/L2 synthetic inversion test. |
| `archive/legacy/run_simulated_inversion.py` main CLI | Older Siamese-oriented standalone workflow; active shared utilities have been extracted. |
| `SMCMTI/` | Older package copy. Prefer `src_smc_mti/` for the current workflow after cleaning imports. |
| `multi_station_simclr/`, `ssl_src/`, `lightning_*`, `generate_data*.py` | Deep-learning and contrastive-learning workflow. Useful historical context, not core for current GSOT/SoftDTW SMC inversion. |
| `axisem/`, `instaseis/`, `beat/`, `Gisola/`, `SPECFEM3D/`, `openswpc/` | Vendored or external solver/package trees. Do not copy wholesale into the clean project. |
| `synthetic_data_axitra/`, `lightning_logs*/`, `logs*/`, `inversion_result/` | Generated data/log/output directories. |

## Cleanup notes before copying

- Active shared utilities have been extracted from the old `run_simulated_inversion.py`
  and `synthetic_inversion_softdtw_test.py` scripts into `src_smc_mti` package
  modules. The old files now live in `archive/legacy/` as standalone
  synthetic/demo workflows.
- `openswpc_tools/setup_single_source_case.py` writes an absolute OpenSWPC
  binary path in the original tree; in this copied project it accepts
  `--swpc-bin`.
- Keep generated solver outputs separate from source code. Copy only selected
  OpenSWPC GF libraries and enough metadata to reproduce them when that
  optional backend is needed.
