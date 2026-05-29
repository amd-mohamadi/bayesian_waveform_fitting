# Real-Event Inversion Workflow

This document describes the current real-event inversion workflow implemented in:

- `prepare_invdata.py`
- `run_inversion.py`

It covers data preparation, preprocessing, inversion options, and practical tuning.

## 1) End-to-end flow

1. Prepare event data from an event folder (`MSEED/<event_id>/`) into one pickle.
2. Run inversion from that pickle with configurable preprocessing, likelihood, sampler, and source settings.

The main design is:

- Raw waveforms are saved in pickle after rotating to `ENZ`.
- Windowing/filtering/tapering and component selection happen in `run_inversion.py`.

## 2) Prepare data (`prepare_invdata.py`)

## Inputs

- Event folder: `MSEED/<event_id>/`
  - `*.mseed` waveforms
  - `picks.dat` (tab-separated EQNet picks)
- `FORGE_catalog.csv` (event location and magnitude)
- `stationxml/*.xml` (station metadata for rotation)

## What it does

- Parses event metadata from catalog (x, y, z, Mw).
- Reads picks and filters by score thresholds.
- Reads each station mseed and rotates to `ZNE` using station XML inventory.
- Stores rotated components in pickle as `ENZ` arrays.
- Optional upsampling before saving.
- Stores pick amplitudes (if available) from `picks.dat`:
  - `Pz`, `Sh`, `Sv`

## Command

```bash
/home/a-mohamdi/anaconda3/envs/seisbench/bin/python prepare_invdata.py \
  --event-dir /home/a-mohamdi/Projects/focal_inversion/MSEED/1111911135 \
  --catalog /home/a-mohamdi/Projects/focal_inversion/FORGE_catalog.csv \
  --stations-dir /home/a-mohamdi/Projects/focal_inversion/stationxml \
  --upsample-factor 2.0
```

## Key args

- `--event-dir`: event folder.
- `--picks-file`: default `picks.dat` inside event dir.
- `--min-p-score`, `--min-s-score`: pick quality thresholds.
- `--upsample-factor`: `1.0` disables upsampling.
- `--out`: output pickle path (default: `<event_dir>/invdata.pkl`).

## Output pickle content

- `source`: event metadata and local coordinates.
- `station_ids`: stations used.
- `waveforms`: raw rotated `ENZ` traces per station.
- `picks`: phase times and scores, plus amplitude dict (`Pz`, `Sh`, `Sv`) when present.
- `preprocess_hint.upsample_factor`.

## 3) Run inversion (`run_inversion.py`)

## Supported likelihoods and samplers

- Likelihoods: `gsot`, `softdtw`, `l2`
- Samplers: `smc`, `cmaes`

## Core command template

```bash
/home/a-mohamdi/anaconda3/envs/seisbench/bin/python run_inversion.py \
  --event-dir /home/a-mohamdi/Projects/focal_inversion/MSEED/1111911135 \
  --stations-dir /home/a-mohamdi/Projects/focal_inversion/stationxml \
  --velocity-model /home/a-mohamdi/Projects/focal_inversion/forge.tvel \
  --likelihood gsot --sampler smc
```

## Preprocessing controls

- Bandpass: `--bp-low`, `--bp-high`, `--bp-order`
- Detrend/demean: `--detrend`, `--demean`
- Taper: `--taper-frac`
- Window taper/normalization: `--apply-window-taper`, `--normalize-per-station`

## Windowing controls

- `--phase-window-p-len`: P-window length in seconds.
- `--phase-window-s-len`: S-window length in seconds.
- `--time-steps`: output samples per window.
- `--auto-time-steps`: scales `time_steps` by `invdata` upsample factor.

Example:

```bash
--phase-window-p-len 0.05 --phase-window-s-len 0.08 --auto-time-steps --time-steps 70
```

## Component selection

Use `--components` with tokens:

- `PZ` = Z component in P window
- `SN` = N component in S window
- `SE` = E component in S window

Examples:

- All components: `--components PZ,SN,SE`
- P-only inversion: `--components PZ`

## Source-time controls

- `--source-target-freq-hz`: Ricker central frequency (higher = sharper).
- `--source-t0`: direct rise-time override; if set, ignores target freq.
- `--source-delay`: global source delay used in synthetic generation/window picks.
- `--duration`, `--fmax`: synthesizer runtime parameters.

## Source sweep mode

You can sweep source frequency in one command:

```bash
--source-target-freq-hz-list "40,60,80,100"
```

The script runs child jobs and reports the best `WAVEFORM_FIT_SCORE`.

## GSOT amplitude-ratio penalty

When using GSOT and all components (`PZ,SN,SE`), observed pick amplitudes from `picks.dat` can be used as ratio constraints:

- Ratios: `P/SH`, `P/SV`, `SH/SV`
- Controls:
  - `--gsot-ratio-weight`
  - `--gsot-ratio-sigma`

If component set is not exactly `{PZ,SN,SE}`, this ratio penalty is disabled automatically.

To explicitly disable ratio enforcement in all cases:

```bash
--gsot-ratio-weight 0
```

## Ray-pattern polarity term (Pz/SH)

`run_inversion.py` can add a CAPE-style polarity likelihood term based on ray-pattern radiation coefficients (from `src_smc_mti`) using `picks.dat`:

- Internals: `read_data` + `polarity_matrix` + `polarity_ln_pdf`
- This is independent from waveform-sign matching.

Enable it with:

```bash
--use-ray-polarity --polarity-phases PZ,SH
```

Useful controls:

- `--polarity-weight` (strength of polarity term)
- `--polarity-data-file` (default `picks.dat`, relative to `--event-dir`)
- `--p-polarity-phase` (default `Pz`)
- `--min-p-phase-score`, `--min-s-phase-score`
- `--min-ppl-score`, `--min-shpl-score`, `--min-svpl-score`
- `--polarity-velocity-model` (TauP model input; defaults to `--velocity-model`)

Example:

```bash
python run_inversion.py ... \
  --components PZ \
  --use-ray-polarity \
  --polarity-phases PZ,SH \
  --polarity-weight 1.0
```

## 4) DC control options

Two modes are available:

## Hard DC mode

```bash
--dc-only
```

Forces `gamma = 0` and `delta = 0` during inversion.

## Soft DC prior mode

Uses Beta priors in unit space for `gamma` and `delta`:

- `--gamma-beta-prior alpha,beta`
- `--delta-beta-prior alpha,beta`
- `--dc-prior-weight`

Defaults are neutral and backward-compatible:

- `--gamma-beta-prior 1,1`
- `--delta-beta-prior 1,1`

Mild DC-favoring example:

```bash
--gamma-beta-prior 3,3 --delta-beta-prior 3,3 --dc-prior-weight 1.0
```

## 5) Recommended starting points

## A) P-only microevent fit

```bash
python run_inversion.py \
  --event-dir /home/a-mohamdi/Projects/focal_inversion/MSEED/1111911135 \
  --stations-dir /home/a-mohamdi/Projects/focal_inversion/stationxml \
  --velocity-model /home/a-mohamdi/Projects/focal_inversion/forge.tvel \
  --likelihood gsot --sampler smc \
  --components PZ \
  --phase-window-p-len 0.05 --phase-window-s-len 0.08 \
  --auto-time-steps --time-steps 70 \
  --source-target-freq-hz 100
```

## B) Strict DC + P-only

```bash
python run_inversion.py ... --components PZ --dc-only
```

## C) Full-component GSOT with ratio control

```bash
python run_inversion.py ... \
  --components PZ,SN,SE \
  --gsot-ratio-weight 0.25 \
  --gsot-ratio-sigma 0.5
```

## D) P-only + ray-pattern polarity + strict DC

```bash
python run_inversion.py \
  --event-dir /home/a-mohamdi/Projects/focal_inversion/MSEED/1111911135 \
  --stations-dir /home/a-mohamdi/Projects/focal_inversion/stationxml \
  --velocity-model /home/a-mohamdi/Projects/focal_inversion/forge.tvel \
  --likelihood gsot --sampler smc \
  --components PZ \
  --phase-window-p-len 0.05 --phase-window-s-len 0.08 \
  --auto-time-steps --time-steps 70 \
  --source-target-freq-hz 100 \
  --dc-only \
  --use-ray-polarity --polarity-phases PZ,SH \
  --polarity-weight 1.0 \
  --gsot-ratio-weight 0
```

## 6) Outputs

`run_inversion.py` writes:

- waveform fit figure: `waveform_fit_<event>_<likelihood>_<sampler>_f<freq>.png`
- beachballs: `bb_map_...png`, `bb_medoid_...png`
- terminal summary (MAP, medoid, weighted mean, fit score)

It also prints:

- `Waveform fit MSE`
- `WAVEFORM_FIT_SCORE` (`-MSE`, higher is better)
