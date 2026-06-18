# OpenSWPC Single-Source Setup (FORGE)

This document records the current working setup to run OpenSWPC for one fixed source location and generate MT-basis synthetics.

## What is implemented

- OpenSWPC compiled locally in `openswpc/bin`.
- Build helper script: `openswpc/build_local.sh`.
- FORGE model crop tool: `openswpc_tools/prepare_forge_submodel.py`.
- OpenSWPC case generator for one event/source: `openswpc_tools/setup_single_source_case.py`.
- OpenSWPC user-model routine (`vmodel_type='user'`) customized to read 3D NetCDF model directly:
  - `openswpc/src/swpc_3d/m_vmodel_user.f90`


## 1) Build OpenSWPC

```bash
cd /home/a-mohamdi/Projects/focal_inversion/openswpc
./build_local.sh
```


## 2) Prepare cropped FORGE 3D model

```bash
/home/a-mohamdi/anaconda3/envs/seisbench/bin/python \
  /home/a-mohamdi/Projects/focal_inversion/openswpc_tools/prepare_forge_submodel.py \
  --input-nc /home/a-mohamdi/Projects/focal_inversion/forge_velocity_models/CapeEGSandUtahFORGE_Empirical3DSeismicVelocityModel/Cape-model_50m_grid_sonic-logs-only_stretched_3-1-24.nc \
  --output-nc /home/a-mohamdi/Projects/focal_inversion/openswpc_model/forge_submodel_1111911135.nc \
  --x-halfwidth-m 2000 --y-halfwidth-m 2000 \
  --z-min-m -200 --z-max-m 4500
```

Outputs:

- `openswpc_model/forge_submodel_1111911135.nc`
- `openswpc_model/forge_submodel_1111911135.json`


## 3) Generate single-source OpenSWPC run case

```bash
/home/a-mohamdi/anaconda3/envs/seisbench/bin/python \
  /home/a-mohamdi/Projects/focal_inversion/openswpc_tools/setup_single_source_case.py \
  --event-dir /home/a-mohamdi/Projects/focal_inversion/MSEED/1111911135 \
  --stations-dir /home/a-mohamdi/Projects/focal_inversion/stationxml \
  --model-nc /home/a-mohamdi/Projects/focal_inversion/openswpc_model/forge_submodel_1111911135.nc \
  --out-dir /home/a-mohamdi/Projects/focal_inversion/openswpc_cases/1111911135 \
  --duration 1.0 --dt 0.001 --dx 0.05 --fmax 200 --vcut 1.5
```

Generated case layout:

```text
openswpc_cases/1111911135/
  stloc.xy
  run_all_basis.sh
  Mxx/
    source_Mxx.dat
    input_Mxx.inf
    in/input.inf
  Myy/
  Mzz/
  Myz/
  Mxz/
  Mxy/
```


## 4) Run one basis simulation

```bash
cd /home/a-mohamdi/Projects/focal_inversion/openswpc_cases/1111911135/Mxx
mpirun -np 4 /home/a-mohamdi/Projects/focal_inversion/openswpc/bin/swpc_3d.x
```

Or run all basis simulations:

```bash
/home/a-mohamdi/Projects/focal_inversion/openswpc_cases/1111911135/run_all_basis.sh
```


## 5) Pack basis runs into a GF library for inversion

```bash
/home/a-mohamdi/anaconda3/envs/seisbench/bin/python \
  /home/a-mohamdi/Projects/focal_inversion/openswpc_tools/pack_basis_to_npz.py \
  --case-dir /home/a-mohamdi/Projects/focal_inversion/openswpc_cases/1111911135_basis_f100 \
  --stations-dir /home/a-mohamdi/Projects/focal_inversion/stationxml \
  --output /home/a-mohamdi/Projects/focal_inversion/openswpc_cases/1111911135_basis_f100/gf_library_f100.npz \
  --quantity V
```


## 6) Run inversion with OpenSWPC GF backend

`run_inversion.py` now supports:

- `--synthetic-backend axitra|openswpc_gf`
- `--openswpc-gf-file <path-to-npz>`

Example smoke test:

```bash
/home/a-mohamdi/anaconda3/envs/seisbench/bin/python \
  /home/a-mohamdi/Projects/focal_inversion/run_inversion.py \
  --event-dir /home/a-mohamdi/Projects/focal_inversion/MSEED/1111911135 \
  --stations-dir /home/a-mohamdi/Projects/focal_inversion/stationxml \
  --velocity-model /home/a-mohamdi/Projects/focal_inversion/forge.tvel \
  --synthetic-backend openswpc_gf \
  --openswpc-gf-file /home/a-mohamdi/Projects/focal_inversion/openswpc_cases/1111911135_basis_f100/gf_library_f100.npz \
  --likelihood gsot --sampler smc --n-particles 30 --n-stages 2 \
  --phase-window-p-len 0.035 --phase-window-s-len 0.05 --time-steps 70 \
  --source-target-freq-hz 100 --bp-low 10 --bp-high 220 \
  --dc-only --use-ray-polarity --polarity-phases PZ --polarity-weight 1.0 \
  --gsot-ratio-weight 0
```


## 7) Current best-progress run (20 m mesh, GSOT+SMC)

Observation from recent tuning: increasing SMC search size (`n-particles`, `n-stages`) improves fit quality enough that only a very small normalized-L2 weight is needed.

Command used:

```bash
python /home/a-mohamdi/Projects/focal_inversion/run_inversion.py \
  --event-dir /home/a-mohamdi/Projects/focal_inversion/MSEED/1111911135 \
  --stations-dir /home/a-mohamdi/Projects/focal_inversion/stationxml \
  --velocity-model /home/a-mohamdi/Projects/focal_inversion/forge.tvel \
  --synthetic-backend openswpc_gf \
  --openswpc-gf-file /home/a-mohamdi/Projects/focal_inversion/o20/gf_library_f100_dx20.npz \
  --likelihood gsot \
  --sampler smc \
  --n-particles 2000 \
  --n-stages 30 \
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
  --l2norm-weight 0.01
```

Practical takeaway:

- Prioritize larger GSOT+SMC search (`n-particles`, `n-stages`) before increasing `--l2norm-weight`.
- Keep `--l2norm-weight` small (e.g., `0.01`) as a weak stabilizer instead of a dominant term.
- Use phase-specific bands to better balance P and S behavior.


## 8) Adaptive GSOT per-trace weighting (SMC+GSOT)

### What was implemented

The GSOT and L2 likelihoods previously aggregated per-trace costs with a flat mean or median across all `nsta * ncomp` traces. Poorly-fitting station-channels (due to 3D velocity model inaccuracies, site effects, or instrument issues) dragged the posterior away from the true solution.

New features added to `GSOTLikelihood` (`src/waveform_likelihoods.py`) and `run_inversion.py`:

- `GSOTLikelihood.compute_per_trace_cost()` — returns per-trace GSOT cost `(B, ntr)` before aggregation.
- `GSOTLikelihood.compute_log_likelihood(..., trace_weights=)` — optional per-trace weight vector; when provided, uses weighted-mean aggregation instead of flat mean/median.
- `l2_normalized_loglike(..., trace_weights=)` — same weighting propagated to the normalized-L2 penalty.
- Adaptive warm-up reweighting in the SMC loop: runs `warmup_stages` with uniform weights, then computes posterior-weighted per-trace GSOT costs and converts to weights via `w ~ 1 / cost^alpha` with a floor and optional zero-below exclusion threshold.
- Manual station-channel weights via `--station-channel-weights-file` (JSON).
- Waveform-fit plot now shows included traces as red dashed and excluded traces as green dashed.

### New CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--adaptive-gsot-trace-weighting` | off | Enable adaptive per station-channel weighting |
| `--adaptive-gsot-warmup-stages` | `5` | Uniform-weight stages before first weight computation |
| `--adaptive-gsot-update-interval` | `5` | Recompute weights every N stages after warmup |
| `--adaptive-gsot-weight-alpha` | `1.0` | Power in `w ~ 1 / cost^alpha` (higher = more aggressive) |
| `--adaptive-gsot-weight-floor` | `0.10` | Floor as fraction of max raw weight |
| `--adaptive-gsot-zero-below` | `0` | Zero out traces with mean-normalized weight below this threshold |
| `--adaptive-gsot-min-active-traces` | `1` | Safety: keep at least N traces active when zero-below is used |
| `--adaptive-gsot-min-cost` | `1e-6` | Numerical floor on per-trace cost |
| `--station-channel-weights-file` | none | JSON file with manual per station-channel weights (overrides adaptive) |

### First test run with adaptive weighting

```bash
python /home/a-mohamdi/Projects/focal_inversion/run_inversion.py \
  --event-dir /home/a-mohamdi/Projects/focal_inversion/MSEED/1111911135 \
  --stations-dir /home/a-mohamdi/Projects/focal_inversion/stationxml \
  --velocity-model /home/a-mohamdi/Projects/focal_inversion/forge.tvel \
  --synthetic-backend openswpc_gf \
  --openswpc-gf-file /home/a-mohamdi/Projects/focal_inversion/o20/gf_library_f100_dx20.npz \
  --likelihood gsot \
  --sampler smc \
  --n-particles 500 \
  --n-stages 50 \
  --phase-window-p-len 0.05 \
  --phase-window-s-len 0.05 \
  --synthetic-phase-window-p-len 0.12 \
  --synthetic-phase-window-s-len 0.2 \
  --auto-time-steps \
  --time-steps 70 \
  --source-target-freq-hz 100 \
  --use-ray-polarity \
  --polarity-phases PZ \
  --polarity-weight 0.5 \
  --gsot-ratio-weight 0 \
  --bp-p-low 10 --bp-p-high 240 \
  --bp-s-low 10 --bp-s-high 200 \
  --l2norm-weight 0.35 \
  --dc-only \
  --adaptive-gsot-trace-weighting \
  --adaptive-gsot-warmup-stages 10 \
  --adaptive-gsot-update-interval 25 \
  --adaptive-gsot-weight-alpha 0.7 \
  --adaptive-gsot-weight-floor 0.15 \
  --adaptive-gsot-zero-below 1.15 \
  --adaptive-gsot-min-active-traces 12
```

### Observations and current limits of the weighting

The adaptive weighting mechanism produces reasonable results overall but has a known issue: the current GSOT cost-based weight sometimes **excludes visually good fits and includes visually bad fits**. This happens because:

1. The GSOT per-trace cost is computed from the posterior-weighted ensemble average, not from the best-fit particle alone. Ensemble averaging can wash out per-trace quality differences.
2. The weight update happens only at fixed intervals (e.g., every 25 stages), so early-stage cost estimates may not reflect the final fit quality.
3. The `zero-below` threshold operates on mean-normalized weights, which is a relative measure. A trace can have a moderate absolute cost but still fall below the threshold if other traces happen to have much lower costs.
4. S-wave components (N, E) tend to have systematically higher GSOT cost than P-wave (Z) due to stronger 3D path effects, wider frequency content mismatch, and less constrained polarization. The current weighting does not distinguish between "inherently harder to fit" and "genuinely bad data/model".

Next steps to improve the weighting:

- Consider per-phase normalization of costs (normalize P and S costs separately before computing weights) so S-wave traces are not systematically penalized.
- Explore using the best-fit particle's per-trace cost instead of the ensemble average for weight computation.
- Add a waveform cross-correlation sanity check on top of the GSOT cost to catch cases where GSOT cost is low but the waveform shape match is poor.
- Allow user-guided hybrid approach: adaptive weights as starting point, with manual overrides for specific station-channels via JSON.


## Notes and current limits

- The setup is currently fixed-source (single location from `invdata.pkl`).
- OpenSWPC warns about wavelength condition if `dx` is too coarse for very high frequency and low near-surface velocity.
- For realistic stability/dispersion tradeoffs, you may need to lower effective modeled high-frequency content or refine the mesh.
- `run_inversion.py` now includes: synthetic-only window lengths, phase-specific P/S bandpass controls, normalized-L2 additive penalty, and lag-aligned plotting diagnostics.


## Quick waveform visualization (ObsPy)

Use the helper script:

```bash
/home/a-mohamdi/anaconda3/envs/seisbench/bin/python \
  /home/a-mohamdi/Projects/focal_inversion/openswpc_tools/plot_openswpc_waveforms.py \
  --wav-dir /home/a-mohamdi/Projects/focal_inversion/openswpc_cases/1111911135_random_mt_f100/out/wav \
  --quantity V --normalize
```

Output example:

- `openswpc_cases/1111911135_random_mt_f100/out/wav/openswpc_waveforms_V.png`
