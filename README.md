# Bayesian Waveform Fitting

Moment tensor waveform fitting for FORGE/CAPE microseismic events using
prepared real-event waveforms, Axitra synthetics, and SMC/CMA-ES search with
GSOT, Soft-DTW, or L2 waveform likelihoods.

## Project goal and current stage

The long-term goal is to build a general joint Bayesian inversion code that
estimates moment tensor, event location, magnitude, and source-time parameters
together from waveform data.

The current stage is focused on fast enough synthetic waveforms for Bayesian
inversion. OpenSWPC remains available for reference Green's-function studies,
but the active CAPE workflow now uses Axitra because direct OpenSWPC runs are
too slow for the target iteration loop.

CAPE is the current application project and data set used to develop and test
this workflow. CAPE-specific scripts and documentation record the current case
setup, but the intended inversion framework is not limited to CAPE.

## What is included

- `prepare_invdata.py` prepares `invdata.pkl` from mseed waveforms, picks,
  catalog metadata, and StationXML.
- `run_inversion.py` is the main inversion script.
- `src_smc_mti/forward/` contains Axitra/OpenSWPC-GF synthesizers and
  arrival-time helpers.
- `src_smc_mti/io/` contains station, velocity-model, and observation loaders.
- `src_smc_mti/waveform_likelihoods.py` contains the active `GSOTLikelihood`,
  `SoftDTWLikelihood`, and `L2Likelihood` implementations.
- `archive/legacy/run_simulated_inversion.py` and
  `archive/legacy/synthetic_inversion_softdtw_test.py` preserve legacy
  synthetic/demo workflows outside the active top-level scripts.
- `generate_axitra_random_waveforms.py` provides a quick Axitra synthetic
  plotting diagnostic for CAPE events.
- `openswpc_tools/` contains optional OpenSWPC model/case/GF-library utilities.
- `src_smc_mti/` contains local Tape moment tensor, polarity, likelihood, and
  plotting helpers.
- `docs/` contains copied workflow notes from the original mixed project.

Generated data and solver outputs are intentionally not included.

## Expected external inputs

Keep these outside Git or add them locally when running:

- `MSEED/<event_id>/` with `*.mseed`, `picks.dat`, and optionally `invdata.pkl`
- `stationxml/`
- `FORGE_catalog.csv`
- `cape.tvel`
- Axitra built at `../axitra/MOMENT_DISP_F90_OPENMP/src`
- packed OpenSWPC GF library only when using `--synthetic-backend openswpc_gf`
- OpenSWPC executable and model files only if regenerating the GF library

## Prepare event data

```bash
python prepare_invdata.py \
  --event-dir MSEED/1111911135 \
  --catalog FORGE_catalog.csv \
  --stations-dir stationxml \
  --upsample-factor 2.0
```

This writes `MSEED/1111911135/invdata.pkl`.

## Run CAPE Inversion With Axitra

First prepare the CAPE waveform data if needed:

```bash
python prepare_cape_invdata.py eq02387
```

Then generate or refresh reusable Axitra Green functions:

```bash
python generate_cape_axitra_greens.py eq02387
```

The default Green-function directory is
`cape_events/eq02387/axitra_greens/`. Inversion runs require that directory to
exist and match the current stations, source, velocity model, duration, `fmax`,
and source time.

```bash
python run_cape_inversion.py eq02387 \
  --n-particles 500 \
  --n-stages 20
```

The CAPE wrapper defaults to Axitra, all three `ZNE` components, pick-centered
synthetic windows, `duration=7 s`, `fmax=30 Hz`, and `2-30 Hz` CAPE filters.
For a quick smoke test, use `--n-particles 30 --n-stages 2`.
More detailed CAPE notes are in `docs/CAPE_WORKFLOW.md`.

The wrapper expands to `run_inversion.py --synthetic-backend axitra`. A direct
equivalent is:

```bash
python run_inversion.py \
  --event-dir cape_events/eq02387 \
  --stations-dir stations \
  --velocity-model cape.tvel \
  --synthetic-backend axitra \
  --axitra-greens-dir cape_events/eq02387/axitra_greens \
  --likelihood gsot \
  --sampler smc \
  --n-particles 500 \
  --n-stages 20 \
  --components PZ,SN,SE \
  --duration 7.0 \
  --fmax 30.0 \
  --source-delay 0.0 \
  --synthetic-window-source picks \
  --phase-window-p-len 0.30 \
  --phase-window-s-len 0.60 \
  --synthetic-phase-window-p-len 0.30 \
  --synthetic-phase-window-s-len 0.60 \
  --time-steps 100 \
  --bp-p-low 2 --bp-p-high 30 \
  --bp-s-low 2 --bp-s-high 30 \
  --source-target-freq-hz 30
```

Expected output files include:

- `waveform_fit_<event>_<likelihood>_<sampler>_f<freq>.png`
- `bb_map_<event>_<likelihood>_<sampler>_f<freq>.png`
- `bb_medoid_<event>_<likelihood>_<sampler>_f<freq>.png`

## Optional OpenSWPC GF Workflow

OpenSWPC GF generation is retained as a reference path, but it is no longer the
default CAPE workflow.

1. Crop the FORGE model:

```bash
python openswpc_tools/prepare_forge_submodel.py \
  --input-nc path/to/FORGE_model.nc \
  --output-nc openswpc_model/forge_submodel_1111911135.nc
```

2. Create six MT-basis OpenSWPC cases:

```bash
python openswpc_tools/setup_single_source_case.py \
  --event-dir MSEED/1111911135 \
  --stations-dir stationxml \
  --model-nc openswpc_model/forge_submodel_1111911135.nc \
  --out-dir openswpc_cases/1111911135_basis_f100_dx20 \
  --duration 1.0 --dt 0.001 --dx 0.02 --fmax 100 \
  --swpc-bin /path/to/swpc_3d.x
```

3. Run `run_all_basis.sh`, then pack SAC files:

```bash
python openswpc_tools/pack_basis_to_npz.py \
  --case-dir openswpc_cases/1111911135_basis_f100_dx20 \
  --stations-dir stationxml \
  --output o20/gf_library_f100_dx20.npz \
  --quantity V
```

More detailed notes are in `docs/OPENSWPC_SINGLE_SOURCE_SETUP.md`.
