# Bayesian Waveform Fitting for Moment Tensor Inversion

A waveform-fitting moment tensor (MT) inversion built on the **SMTI BlackJAX SMC sampler**,
with a **waveform likelihood adapted from Grond / BEAT** and Green's functions generated
with the **Grond/Pyrocko QSEIS backend**.

## Goal

Build an **efficient and stable** waveform-fitting MT inversion that works for
**small-magnitude events**.

The motivating problem: existing waveform-fitting codes are **not stable for full MT
inversion** — full-MT posteriors are wide and multimodal, and at small magnitude the low
SNR makes least-squares waveform fits collapse to non-physical or local solutions. This
project addresses that by combining a robust sampler (tempered SMC), a physically valid MT
parameterization (Tape & Tape lune coordinates), and a noise-aware waveform likelihood.

## Strategy

The new inversion is assembled from three proven pieces rather than written from scratch:

1. **Sampler — reuse SMTI's BlackJAX SMC.** SMTI already runs a stable adaptive tempered
   SMC sampler for MT inversion from polarity + amplitude-ratio data. Its likelihood
   interface is decoupled from the MT parameterization, so the polarity/amplitude-ratio
   likelihood can be **swapped for a waveform likelihood** without touching the SMC machinery.
2. **Likelihood — adapt from Grond or BEAT.** Both are mature waveform MT codes vendored in
   this repo (see *Design notes* for which pieces to take from each).
3. **Green's functions — Grond/Pyrocko QSEIS backend.** Synthetics come from pre-computed
   Pyrocko `fomosto` GF stores built with **QSEIS** (Wang's layered-halfspace code). QSEIS
   stores for the reference event are already built (see *Green's functions* below).

Why this combination should be more stable than a plain least-squares waveform fit:

- **Tempered SMC** marches the posterior from prior (β=0) to target (β=1) adaptively by
  effective sample size, so it does not get trapped in the local minima that defeat gradient
  / least-squares full-MT fits.
- **Tape & Tape (2015) lune parameterization** keeps every sample a valid moment tensor and
  makes the full-MT ⇄ deviatoric ⇄ DC restriction a matter of fixing parameters, not
  enforcing awkward constraints.
- **Noise-aware likelihood** (data covariance + hierarchical noise scaling, from BEAT)
  prevents a few noisy stations from dominating the fit — the key failure mode at small
  magnitude.

## Components (vendored references)

| Path | Role in this project |
|------|----------------------|
| `SMTI/` | The sampler we build on. BlackJAX adaptive tempered SMC with NUTS / Metropolis-within-Gibbs rejuvenation; Tape & Tape lune parameterization; a likelihood interface designed to be swapped. |
| `grond/` | Waveform signal-processing + misfit reference (filtering, tapering, time-domain/spectral/envelope/CC misfit, auto time-shift). Also the QSEIS GF store tooling. |
| `beat/` | Probabilistic waveform-likelihood reference: Gaussian likelihood with an estimated data covariance `Cd` and hierarchical noise hyperparameters — the most natural fit for an SMC sampler. |
| `src/` | **New project code lives here** (currently empty). |

### Key SMTI pieces to reuse (`SMTI/src/`)

- `inversion_blackjax.py` — SMC orchestration (adaptive tempered / persistent SMC, ESS-based
  resampling, NUTS/MWG kernels, multi-chain + R-hat). **Likelihood-agnostic**; the new
  waveform log-likelihood is substituted in place of `loglikelihood_fn`.
- `tape.py` / `tape_jax.py` — Tape & Tape (2015) lune parameterization: 5 free parameters
  `γ` (source-type lon), `δ` (lat), `κ` (strike), `h = cos(dip)`, `σ` (slip), plus magnitude.
  Fix `γ = δ = 0` for a pure double-couple. JAX version is JIT/GPU-ready.
- `mwg_kernel.py`, `blockwise_rmh.py` — block-wise Metropolis-within-Gibbs with per-block
  adaptive scaling (source-type / mechanism / noise blocks).
- `likelihoods.py` — the current polarity + amplitude-ratio likelihoods. **Pattern to follow**
  for the new waveform likelihood (and useful as an additional constraint term).

## Green's functions (QSEIS)

Synthetics are pre-computed Pyrocko `fomosto` GF stores. The QSEIS backend is built/queried
from the **`grond`** conda env. Existing stores in `gf_stores/`:

```
forge_eq02387_qseis_pathavg_600m
forge_eq02387_qseis_pathavg_600m_d100m
forge_eq02387_qseis_pathavg_600m_eventgrid
```

Stores are addressed by `store_id` through a Pyrocko `LocalEngine`; both Grond and BEAT
query them the same way. Build new stores with `fomosto` (QSEIS backend) for a given 1-D
velocity model (`*.tvel`).

## Conda environments

| Env | Purpose |
|-----|---------|
| `grond` | Grond inversion runs **and** QSEIS / `fomosto` Green's-function generation. |
| `beatenv` | BEAT installed binaries and BEAT inversion runs. |
| `pymc` | BlackJAX, JAX, and the rest of the SMTI sampler stack — **the env for the new `src/` code**. |

## Source layout (`src/`)

All new code lives in **`src/`**. Vendored upstreams (`SMTI/`, `grond/`, `beat/`) are
references, not edited.

| Module | Role |
|--------|------|
| `parameterization.py` | Tape & Tape (2015) lune (γ,δ,κ,h,σ) + magnitude → NED moment tensor; unconstrained↔physical transforms. Reuses SMTI's `jax_Tape_MT33` via `smti_bridge.py`. |
| `forward.py` | pyrocko `LocalEngine` on the QSEIS store; precomputes the 6 elementary MT basis seismograms per station-channel (exact, verified to 2e-7). |
| `processing.py` | Shared bandpass + cosine taper + window-to-grid, applied identically to observed and synthetic (all linear, so basis stays a matmul). |
| `covariance.py` | BEAT-style data covariance `Cd` from pre-event noise (`variance` / `exponential` Toeplitz) → Cholesky-inverse weights + log-det. |
| `likelihood.py` | BEAT Gaussian waveform likelihood (JAX), with optional grond-style per-trace cross-correlation **autoshift**. |
| `dataset.py`, `data.py` | Load the grond example (event, stations, displacement R/T/Z); window/process basis+obs; estimate `Cd`; build a `WaveformDataset`. |
| `model.py` | Assemble `logprior` + `loglikelihood` over particles; uniform-in-physical priors; per-trace/-channel/-station noise hyperparameter grouping. |
| `sampler.py` | Adaptive tempered SMC (BlackJAX), reusing SMTI's pattern; RMH/NUTS rejuvenation, β=1 rejuvenation + final resample to an equally-weighted posterior. |
| `run_regional_cmt.py` | End-to-end driver / test on the grond regional example. |

### Usage

```bash
# synthetic recovery test (generate obs from event.txt MT, recover it)
conda run -n pymc python -m src.run_regional_cmt --mode synthetic

# fit the real prepared GE-network waveforms (enables autoshift + per-trace noise)
conda run -n pymc python -m src.run_regional_cmt --mode real

# visualize a saved posterior: fuzzy beachball + waveform-fit overlays -> report/
conda run -n pymc python -m src.plot_results --mode real --posterior runs/real_adapt.npz
```

`plotting.py` / `plot_results.py` produce a **fuzzy (posterior-ensemble) beachball** vs the
true mechanism and a **per-trace observed-vs-synthetic waveform overlay** (with variance
reduction and autoshift annotated), reconstructing the dataset from the saved `.npz` (no
re-inversion).

### Status / results

Validated on `grond/examples/example_regional_cmt` (event gfz2018pmjk, M5.9, store
`crust2_j3`, fmin/fmax = 0.01/0.05 Hz):

- **Synthetic recovery:** Kagan angle **0.0°**, Mw 5.86 vs 5.89, with a well-mixed
  posterior (600/600 unique particles) — confirms the forward + likelihood + sampler are
  correct end-to-end.
- **Real data:** Kagan 87° (no time-shift handling) → 31° (autoshift) → **≈23°**, Mw 5.74
  vs 5.89, with grond-style per-trace autoshift + BEAT per-trace noise hyperparameters +
  the adaptive sampler — a reasonable real-data fit given a generic 1-D crust2.0 model and
  fixed depth.
- **Sampler:** population-adaptive RMH (proposal scale ← particle-population std each
  tempering stage) was essential — fixed-scale RMH collapsed to ~3 effective particles;
  adaptive scaling restores full diversity (800/800 unique) and an honest posterior.

Known limitations / next steps: invert depth (currently fixed at the catalog value),
per-station static weighting/QC for bad traces, optional `exponential` noise covariance,
and validation on small-magnitude events (the design target; the test fixture is M5.9).

## Design notes (open decisions)

Recorded so they don't have to be re-litigated each session:

- **Which likelihood to adopt.** BEAT and Grond solve different halves of the problem and are
  complementary:
  - *BEAT* gives the **probabilistic core**: a Gaussian likelihood
    `−½ residualᵀ Cd⁻¹ residual` using a Cholesky-decomposed inverse covariance, with `Cd`
    estimated from **pre-event noise** and a **hierarchical noise scaling** hyperparameter per
    station/channel. This is what makes a *Bayesian* waveform fit well-posed and is the main
    lever for small-magnitude stability.
  - *Grond* gives the **signal-processing machinery**: bandpass filtering with tapered
    transitions, cosine tapers around phase windows (`{stored:any_P}`-style timings),
    time/spectral/envelope/cross-correlation misfit domains, and **automatic time-shift**
    search to absorb velocity-model / pick errors.
  - **Working plan:** build a BEAT-style Gaussian-with-`Cd` likelihood and borrow Grond's
    windowing / filtering / time-shift handling for preprocessing and alignment.
- **Forward model gap.** SMTI's current forward model only produces ray-path radiation
  coefficients (no wave propagation). The waveform forward step is **replaced** by querying
  the QSEIS GF store via Pyrocko's engine to synthesize full waveforms for a candidate MT.
- **Stability levers for small magnitude** (from the BEAT/Grond survey): use ≥5 s pre-P noise
  windows to estimate `Cd`; prefer exponential / non-Toeplitz covariance over diagonal for
  short-period channels; keep hierarchical noise hyperparameters bounded to avoid overfitting
  single noisy stations; consider separate frequency bands / phases as independent waveform
  groups with their own noise scaling.
