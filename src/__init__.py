"""Bayesian waveform-fitting moment tensor inversion.

Pipeline:
  parameterization  Tape & Tape (2015) lune coords -> moment tensor (NED)
  forward           pyrocko/QSEIS Green's functions -> linear 6-component MT basis
  processing        shared filter/taper/window applied identically to obs + synth
  covariance        BEAT-style data covariance Cd from pre-event noise
  likelihood        BEAT-style Gaussian waveform likelihood with noise hyperparameters
  sampler           SMTI BlackJAX SMC

Runs in the `pymc` conda env (pyrocko + jax + blackjax).
"""

# Moment-tensor math and likelihood need float64; enable before any jax.numpy use.
import jax
jax.config.update("jax_enable_x64", True)
