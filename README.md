# Bayesian Waveform Fitting

Moment tensor waveform fitting for FORGE microseismic events using prepared
real-event waveforms, OpenSWPC Green's-function libraries, and SMC/CMA-ES
search with GSOT, Soft-DTW, or L2 waveform likelihoods.

## What is included

- `prepare_invdata.py` prepares `invdata.pkl` from mseed waveforms, picks,
  catalog metadata, and StationXML.
- `run_inversion.py` is the main inversion script.
- `run_simulated_inversion.py` contains shared synthesizer, station, velocity,
  and arrival-time helpers.
- `synthetic_inversion_softdtw_test.py` contains the active `GSOTLikelihood`
  and `SoftDTWLikelihood` implementations used by `run_inversion.py`.
- `openswpc_tools/` contains OpenSWPC model/case/GF-library utilities.
- `src_smc_mti/` contains local Tape moment tensor, polarity, likelihood, and
  plotting helpers.
- `docs/` contains copied workflow notes from the original mixed project.

Generated data and solver outputs are intentionally not included.

## Expected external inputs

Keep these outside Git or add them locally when running:

- `MSEED/<event_id>/` with `*.mseed`, `picks.dat`, and optionally `invdata.pkl`
- `stationxml/`
- `FORGE_catalog.csv`
- `forge.tvel`
- packed OpenSWPC GF library, for example `o20/gf_library_f100_dx20.npz`
- OpenSWPC executable and model files if regenerating the GF library

## Prepare event data

```bash
python prepare_invdata.py \
  --event-dir MSEED/1111911135 \
  --catalog FORGE_catalog.csv \
  --stations-dir stationxml \
  --upsample-factor 2.0
```

This writes `MSEED/1111911135/invdata.pkl`.

## Run inversion with OpenSWPC GF backend

```bash
python run_inversion.py \
  --event-dir MSEED/1111911135 \
  --stations-dir stationxml \
  --velocity-model forge.tvel \
  --synthetic-backend openswpc_gf \
  --openswpc-gf-file o20/gf_library_f100_dx20.npz \
  --likelihood gsot \
  --sampler smc \
  --n-particles 500 \
  --n-stages 20 \
  --phase-window-p-len 0.05 \
  --phase-window-s-len 0.05 \
  --synthetic-phase-window-p-len 0.12 \
  --synthetic-phase-window-s-len 0.2 \
  --auto-time-steps \
  --time-steps 70 \
  --source-target-freq-hz 100 \
  --use-ray-polarity \
  --polarity-phases PZ \
  --polarity-weight 1.0 \
  --gsot-ratio-weight 0 \
  --bp-p-low 10 --bp-p-high 240 \
  --bp-s-low 10 --bp-s-high 200 \
  --l2norm-weight 0.01 \
  --dc-only
```

Expected output files include:

- `waveform_fit_<event>_<likelihood>_<sampler>_f<freq>.png`
- `bb_map_<event>_<likelihood>_<sampler>_f<freq>.png`
- `bb_medoid_<event>_<likelihood>_<sampler>_f<freq>.png`

For a quick smoke test, use `--n-particles 30 --n-stages 2`.

## OpenSWPC GF workflow

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
