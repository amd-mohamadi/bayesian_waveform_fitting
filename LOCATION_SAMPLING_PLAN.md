# Plan: joint location + MT sampling over the 512-point green cloud

> **Status: Variant A (discrete) implemented** — `--sample-location` in
> `run_eq02387_cmt.py` (+ `--synth-pid` for location-recovery tests).
> Validated: synthetic recovery from an off-centre point lands within one
> grid cell (MAP node +0/−150/+0 m, mean ±40 m, Kagan 6.8° at smoke-test
> sampler settings); 1-D path and single-point 3-D path regressions
> bit-identical. Variant B (trilinear) not implemented.

Goal: stop collapsing the 3D store to the single nearest green point in
`NpzGFForward`; instead sample the source location over the 8x8x8 (150 m
spacing, +/-525 m) cloud jointly with the moment tensor. Two variants --
**A: discrete** (snap to nearest grid node) and **B: trilinear interpolation**
(continuous xyz between nodes). Both share the same infrastructure; B is A
plus an interpolation layer.

## What both variants share

The current pipeline is: `NpzGFForward` (1 point) -> window basis at
basis-detected onsets -> `WaveformDataset.basis (T, 6, N)` -> JAX likelihood
`m6 @ basis`. Location sampling generalizes the basis to a per-point stack
`(P, T, 6, N)` (P = n green points) and makes the likelihood select/blend
along the P axis. The observed side (pick-anchored windows, noise covariance,
autoshift `data_shifts`) is location-independent and unchanged.

1. **`src/forward.py` -- multi-point mode for `NpzGFForward`.**
   Keep the existing single-point behavior as default. Add
   `n_points`-aware accessors:
   - store the full `gf` array and the point grid (`points_xyz_km`, plus the
     8x8x8 shape and spacing read off the coordinates);
   - `raw_basis_point(pid)` -> per-target `(6, N_raw)` NED basis for point
     `pid` (same rotation/scaling code as now, refactored into a helper so
     the single-point path reuses it).
   Memory: the full store is (512, 6, 5, 3, 2000) float32 ~ 368 MB loaded
   once in numpy; the windowed product handed to JAX is
   (512, 15, 6, ~200) ~ 37 MB float32 -- fine.

2. **Per-point synthetic anchors + windowed basis stack.**
   New builder (in `src/dataset.py`, mirroring `_process_basis_picks`):
   for each point `pid`, detect onsets with the existing
   `basis_onset_anchors` (search window = pick +/- 0.35 s covers the
   +/-525 m moveout of ~ +/-0.2 s), then cut/filter the basis at those
   onsets. Produces `basis_all (P, T, 6, N)` and `anchors_all (P, T)`.
   Setup cost: 512 x 15 x 6 filtered windowings, ~1-3 min once per run
   (numpy). Print a progress line every ~50 points.

3. **Timing information -- explicit design decision.**
   Cutting each point's synthetic window at its *own* onset re-centres the
   arrival for every candidate location, and the per-trace autoshift
   (+/-40-100 ms) absorbs what remains. Consequence: **location is
   constrained by waveform shape + relative amplitudes across
   stations/components, not by arrival times.** (The 150 m grid moveout is
   ~25 ms/station -- smaller than the autoshift anyway, so this information
   is already discarded by the current design.) This matches the CAP
   philosophy and is what we implement. If timing-based location is ever
   wanted, that is a separate change: anchor all points in a common time
   frame and shrink the autoshift.

4. **Likelihood (`src/model.py` / `src/likelihood.py`).**
   `build_logdensities` accepts an optional `basis_all (P, T, 6, N)` (and
   `a_pol_all (P, Npol, 6)` for polarity). The particle gains location
   fields (variant-specific, below); the likelihood resolves them to a
   per-particle basis `(T, 6, N)` and proceeds exactly as now
   (`waveform_loglike[_autoshift]`). No change to the covariance/hp
   machinery.

5. **Polarity per point.** `build_polarity_coeffs` already takes the
   forward; run it per point (reusing the per-point onsets from step 2) to
   get `a_pol_all (P, Npol, 6)`; the likelihood looks it up the same way as
   the basis.

6. **Sampler.** No changes needed: `build_blocks_from_keys` in
   `src/mwg_kernel.py` puts unknown keys in their own RMH block, so a new
   `"loc"` field gets its own Metropolis block with adaptive scale for free.
   (Verify the block actually appears; add `"loc"` to the mechanism-blocks
   list only if mixing demands it.)

7. **Driver (`run_eq02387_cmt.py`).** New flag `--sample-location
   {off,discrete,interp}` (default `off` = exactly today's behavior).
   When on: build the stack (steps 2, 5), pass it to `build_logdensities`
   and `init_particles`, and extend the report/outputs:
   - posterior mean/std and MAP of (x, y, z) offset from the catalog
     hypocenter in meters;
   - save location samples in the output npz;
   - new plot `real_location.png`: 3 marginal histograms (x, y, z) plus an
     x-y scatter colored by posterior weight, catalog location marked.

8. **Validation (cheap first, as always).**
   a. *Synthetic recovery with a location offset*: generate synthetic data
      from a green point ~2 nodes off-centre (known m6), invert with
      location sampling on. Pass: recovered node/xyz within one grid cell,
      Kagan < a few deg, Mw exact.
   b. *Real Z-only run* vs the fixed-location result: mechanism should not
      degrade; location posterior should include the DAS location.
   c. Regression: `--sample-location off` reproduces the current result
      bit-for-bit (same seed).

## Variant A: discrete (snap to nearest node)

Location parameter: 3 continuous unconstrained fields `loc = (ux, uy, uz)`
mapped through the existing logit transform to physical (x, y, z) inside the
cloud bounding box (uniform prior + the standard Jacobian terms, exactly like
`mag`). Inside the likelihood the point is **snapped to the nearest grid
node**:

```
i = round((x - x0)/dx_grid), j = ..., k = ...   ->  pid = flat index
basis = basis_all[pid]                          # jnp.take, jit-safe
```

- Piecewise-constant likelihood in (x,y,z); the RMH proposal hops nodes when
  the step crosses a cell boundary. With 150 m cells and adaptive proposal
  scales this mixes fine in SMC (tempering does the heavy lifting).
- Posterior on location = histogram over nodes (150 m resolution floor).
- Physically exact at every node -- no waveform blending artifacts.
- Estimated size: ~40 lines model/likelihood, ~30 lines forward/dataset,
  ~40 lines driver/report/plot.

## Variant B: trilinear interpolation (continuous xyz)

Same 3 location fields, but the basis at (x, y, z) is blended from the 8
surrounding nodes with trilinear weights:

```
cx = (x - x0)/dx_grid; i0 = floor(cx); fx = cx - i0   (same for y, z)
basis = sum over 8 corners of w_corner * basis_all[pid_corner]
```

- **Pre-alignment is what makes this valid**: because step 2 cuts every
  point's basis around its own onset, the 8 corner bases are already
  arrival-aligned, so blending interpolates amplitude/shape instead of
  smearing time-shifted pulses (the classic trilinear-GF artifact; corner
  arrivals differ by up to ~25 ms ~ half a period at 20 Hz).
  Residual intra-cell shape change is what the interpolation linearizes.
- Smooth likelihood in (x,y,z) -> better RMH mixing, and a location
  posterior in meters (sub-cell), not a node histogram.
- Polarity coefficients are blended with the same weights.
- Costs on top of A: the 8-corner gather/blend in the jitted likelihood
  (8x the basis-lookup FLOPs -- negligible next to the residual quadratic),
  regular-grid bookkeeping, and validation that the blend is benign
  (check: interpolated basis at a node == that node's basis; at a cell
  midpoint, compare blend vs the exact store... which we don't have off-node,
  so instead compare blend vs nearest-node on a synthetic recovery -- B must
  do no worse than A).
- Estimated extra: ~40 lines over A.

## Recommendation

Start with **A**: ~half the code, no interpolation-validity questions, and
the 150 m node resolution matches what the data can resolve given that
timing is autoshifted away (amplitude/shape constrains location only
weakly at these scales). If A's posterior piles up on one or two nodes and
sub-cell resolution actually matters, add B on top -- A's infrastructure
(steps 1-8) is 100% reused, B only swaps the lookup for the blend.
