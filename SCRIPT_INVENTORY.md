# Bayesian Waveform Fitting Script Inventory

This inventory records the current waveform-fitting pieces found in the mixed
`focal_inversion` workspace. It is intended as the starting checklist for
copying a clean, continued project into `bayesian_waveform_fitting`.

## Current workflow

The active workflow appears to be:

1. Prepare real event data from `MSEED/<event_id>/`, `picks.dat`,
   `FORGE_catalog.csv`, and `stationxml/`.
2. Prepare or run OpenSWPC MT-basis simulations for the same fixed source.
3. Pack OpenSWPC basis SAC files into an NPZ Green's-function library.
4. Run `run_inversion.py` with the OpenSWPC GF backend, SMC sampler, and
   `gsot` or `softdtw` likelihood.

## Primary scripts to copy

These are the main project scripts for the current waveform fitting workflow.

| Source path | Role |
| --- | --- |
| `prepare_invdata.py` | Builds `invdata.pkl` for a real event from mseed waveforms, EQNet picks, catalog metadata, and StationXML. |
| `run_inversion.py` | Main inversion driver. Supports `--synthetic-backend openswpc_gf`, `--sampler smc`, `--likelihood gsot/softdtw/l2`, ray-polarity terms, amplitude-ratio terms, and adaptive GSOT trace weighting. |
| `run_simulated_inversion.py` | Provides shared forward-model utilities used by `run_inversion.py`, including `FastSynthesizer`, `OpenSWPCGFSynthesizer`, `L2Likelihood`, station loading, velocity-model loading, Tape MT imports, and plotting helpers. |
| `synthetic_inversion_softdtw_test.py` | Provides `SoftDTWLikelihood`, `GSOTLikelihood`, posterior medoid helpers, unit-parameter decoding, and beachball plotting used by `run_inversion.py`. Despite the test-like name, it contains active likelihood code. |

## OpenSWPC support scripts

Copy these with the main scripts if continuing with OpenSWPC synthetics.

| Source path | Role |
| --- | --- |
| `openswpc_tools/prepare_forge_submodel.py` | Crops the FORGE 3D NetCDF velocity model and writes density for OpenSWPC user-model input. |
| `openswpc_tools/setup_single_source_case.py` | Creates fixed-source OpenSWPC MT-basis case directories and `run_all_basis.sh`. |
| `openswpc_tools/pack_basis_to_npz.py` | Packs six OpenSWPC MT-basis SAC outputs into `gf_library*.npz` for fast inversion. |
| `openswpc_tools/plot_openswpc_waveforms.py` | Quick plotting utility for OpenSWPC SAC waveform outputs. |
| `openswpc_tools/run_random_mt_forward.py` | Diagnostic single random-MT OpenSWPC forward run. Useful for sanity checks, not required for inversion. |
| `openswpc_tools/__init__.py` | Package marker. |

## Local SMCMTI helpers

`run_inversion.py` imports these directly for the ray-pattern polarity term.
The Tape conversion functions are also needed through `run_simulated_inversion.py`
unless that import is cleaned up during refactor.

| Source path | Role |
| --- | --- |
| `src_smc_mti/data_loader.py` | Reads event/pick data for polarity constraints. |
| `src_smc_mti/data_prep.py` | Builds polarity matrices. |
| `src_smc_mti/likelihoods.py` | Provides polarity and amplitude-ratio likelihood helpers. |
| `src_smc_mti/tape.py` | Tape & Tape moment tensor conversion functions. |
| `src_smc_mti/moment_tesnor_conversion.py` | Larger moment tensor conversion library used by plotting/utilities. Filename typo is existing code. |
| `src_smc_mti/plot/` | Focal-sphere/beachball plotting support imported by `synthetic_inversion_softdtw_test.py`. |

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
| `forge.tvel` | 1D model used for arrival-time and polarity calculations. |
| `openswpc_model/` | Cropped OpenSWPC model outputs. |
| `openswpc_cases/` and `o20/` | Generated OpenSWPC basis runs and packed GF libraries. Copy only selected GF libraries, not all run outputs. |

## Likely legacy or secondary code

These paths are related but should not be first-pass copied into the clean
Bayesian waveform-fitting project.

| Path | Reason |
| --- | --- |
| `prepare_waveform.py` | Older exploratory waveform-preparation script. It has hard-coded external paths and an apparent `stid` typo. Superseded by `prepare_invdata.py`. |
| `synthetic_inversion_test.py` | Older Siamese/L2 synthetic inversion test. |
| `run_simulated_inversion.py` main CLI | The file has useful shared utilities, but its main script path is older and Siamese-oriented. |
| `SMCMTI/` | Older package copy. Prefer `src_smc_mti/` for the current workflow after cleaning imports. |
| `multi_station_simclr/`, `ssl_src/`, `lightning_*`, `generate_data*.py` | Deep-learning and contrastive-learning workflow. Useful historical context, not core for current GSOT/SoftDTW SMC inversion. |
| `axisem/`, `instaseis/`, `beat/`, `Gisola/`, `SPECFEM3D/`, `openswpc/` | Vendored or external solver/package trees. Do not copy wholesale into the clean project. |
| `synthetic_data_axitra/`, `lightning_logs*/`, `logs*/`, `inversion_result/` | Generated data/log/output directories. |

## Cleanup notes before copying

- `run_inversion.py` imports active likelihoods from
  `synthetic_inversion_softdtw_test.py`; rename/extract that module during the
  cleanup pass.
- In this copied project, `run_simulated_inversion.py` has been patched to use
  `src_smc_mti` and local station/velocity/arrival helpers instead of the old
  `SMCMTI` and `multi_station_simclr` trees.
- `openswpc_tools/setup_single_source_case.py` writes an absolute OpenSWPC
  binary path in the original tree; in this copied project it accepts
  `--swpc-bin`.
- Keep generated OpenSWPC outputs separate from source code. Copy only the
  selected packed GF library and enough metadata to reproduce it.
