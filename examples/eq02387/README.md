# Example: eq02387 (CAPE/FORGE, Ml 2.64)

Reference input/output for `run_event_cmt.py`, so a new case can be set up by
matching these file formats.

## input/

| File | Role |
|------|------|
| `invdata.pkl` | Prepared observed data: pyrocko event, stations, ENZ velocity traces (rotated to ZRT by the driver), P/S picks (relative to event time). This is what `PATHS["invdata"]` points to. |
| `pz_polarity.csv` | Per-station P first-motion polarity cache (`station_id, polarity, phase_type, score`), sign = up/down, |.| = picker confidence. `PATHS["polarity_cache"]`. Built once from the picker CSV (`load_pz_polarity`); can also be written by hand. Note: FORU was corrected manually to +1.0. |
| `FebMarch_catalog.csv` | Catalog row for the event — the `magnitude` column (local magnitude Ml, manual) is the magnitude reference used in plots/README comparisons. |

Not included here (too large / rebuilt per event):

- 3-D reciprocity GF store npz (`--gf-npz`) — built with `gf_pipeline/`
  (see `gf_pipeline/EQ02387_RECIPROCITY_STORE_PLAN.md`); for eq02387 it is an
  8x8x8 green-point cloud @ 150 m around the hypocenter, dx=30 m FDM.
- 1-D qseis store (`gf_stores/forge_eq02387_qseis_pathavg_600m`) — shipped at
  the repo root; used for takeoff-angle ray tracing in the station beachball
  plots (and for 1-D-mode runs).
- SMTI polarity+amplitude-ratio reference posterior — shipped in
  `reference_solutions/eq02387_smti_blackjax_ppol_ampratio_mt6.npz`.

## output/

Produced by the recommended full-MT ZRT run (see "ZRT recipe" in the top-level
README):

```bash
conda run -n pymc python run_event_cmt.py \
  --mode real --components Z,R,T --sample-location --loc-radius-m 250 \
  --gf-npz <path>/gf_store_eq02387_green_dx30.npz \
  --p-fmin 2 --p-fmax 20 --s-fmin 2 --s-fmax 12 \
  --max-shift-sec 0.05 --lag-penalty 2.0 \
  --num-particles 2000 --mcmc-steps 20 --polarity-weight 50 \
  --out runs/eq02387_zrt_lp.npz --outdir report/eq02387_zrt_lp
```

- `posterior.npz` — posterior samples (m6, Tape parameters, mag, hp, loc_xyz,
  weights) + everything needed to re-plot without re-inversion.
- `report/` — all figures: fuzzy beachball vs reference, waveform fits (true
  amplitude), full traces with fit windows, station-annotated median/mode
  beachballs, Kaverina + Hudson HDI, canonical strike/dip/rake + lune
  marginals, location posterior, arviz summary CSV.

Result: strike/dip/rake 258.5/84.8/+17.6, Kagan 5.0 deg vs the SMTI
polarity+amplitude-ratio reference, 5/5 polarities, Mw 2.47.
