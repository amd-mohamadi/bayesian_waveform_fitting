"""Adaptive tempered SMC engine (BlackJAX), reusing SMTI's sampler pattern.

Problem-agnostic: takes per-particle ``logprior_fn`` and ``loglikelihood_fn``
(JAX, operating on a dict-of-scalars position) plus initial particles, and runs
BlackJAX ``adaptive_tempered_smc`` with a rejuvenation kernel. The tempering
parameter marches 0 -> 1 adaptively by effective sample size.

Two ingredients keep the posterior non-degenerate and make tempering converge in
few stages on large (many-station) datasets -- both ported from the tuned
``bjsi_blackjax`` SMC recipe:

* **Waste-free SMC** (``waste_free=True``, default): keep every intermediate
  rejuvenation state instead of only the last one (Dau & Chopin 2022). This turns
  ``n_particles`` x ``num_mcmc_steps`` mutations into that many retained samples
  at no extra likelihood cost, cutting the number of tempering stages needed.
  It is kernel-agnostic and layers on top of any mutation kernel below.

* **Population-adaptive mutation**: the proposal/mass is re-derived from the
  particle population every stage (via ``inner_kernel_tuning``), so acceptance
  does not collapse as the posterior tightens.

Mutation kernels:

* ``rmh`` (default): random-walk Metropolis whose per-parameter proposal scale is
  a fraction of the population std -- gradient-free, robust to the autoshift
  ``min`` non-smoothness.
* ``hmc``: gradient-based HMC with a covariance-preconditioned diagonal mass
  matrix (re-derived from the particle population each stage) and a damped
  step-size adaptation toward ``hmc_target_accept``. Mixes the tightly-peaked
  large-data posterior much better than ``rmh`` when the target is differentiable.
* ``nuts``: for fully differentiable models (no autoshift).
"""
import time

import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
import numpy as np

import blackjax
from blackjax.smc.resampling import systematic
from blackjax.smc import inner_kernel_tuning
import blackjax.mcmc.random_walk as blackjax_rw

from .mwg_kernel import (build_mwg_kernel, build_blocks_from_keys,
                         default_inner_steps, block_scale_overrides)


def _unwrap(state):
    return state.sampler_state if hasattr(state, "sampler_state") else state


def _param_dim(particles):
    d = 0
    for leaf in jax.tree_util.tree_leaves(particles):
        d += int(np.prod(leaf.shape[1:])) if leaf.ndim > 1 else 1
    return d


def run_smc(
    logprior_fn,
    loglikelihood_fn,
    initial_particles,
    rng_key,
    kernel="rmh",
    adapt_proposal=True,
    proposal_factor=None,
    num_mcmc_steps=10,
    mwg_mechanism_steps=None,
    proposal_scale=0.1,
    nuts_step_size=1e-2,
    nuts_max_doublings=10,
    target_ess=0.6,
    max_iterations=300,
    final_rejuvenation=5,
    waste_free=True,
    hmc_step_size=0.1,
    hmc_num_integration_steps=8,
    adapt_hmc_step_size=True,
    hmc_target_accept=0.8,
    hmc_step_adaptation_rate=0.3,
    hmc_step_min=1e-3,
    hmc_step_max=0.5,
    verbose=True,
):
    n_particles = jax.tree_util.tree_leaves(initial_particles)[0].shape[0]

    def logdensity_fn(position):
        val = logprior_fn(position) + loglikelihood_fn(position)
        return jnp.nan_to_num(val, nan=-1e6, posinf=-1e6, neginf=-1e6)

    # Waste-free SMC needs n_particles divisible by the per-particle step count;
    # otherwise fall back to standard SMC so arbitrary particle/step combos still run.
    use_waste_free = bool(waste_free)
    if use_waste_free and n_particles % int(num_mcmc_steps) != 0:
        if verbose:
            print(f"  [waste-free off] n_particles ({n_particles}) not divisible by "
                  f"num_mcmc_steps ({num_mcmc_steps}); using standard SMC.", flush=True)
        use_waste_free = False

    if use_waste_free:
        from blackjax.smc.waste_free import waste_free_smc
        # Waste-free supplies p = num_mcmc_steps internally; the SMC algo must then
        # receive num_mcmc_steps=None.
        smc_extra = {"update_strategy": waste_free_smc(int(n_particles), int(num_mcmc_steps))}
        smc_num_mcmc_steps = None
    else:
        smc_extra = {}
        smc_num_mcmc_steps = int(num_mcmc_steps)

    tuned = False
    adapt_step = False
    if kernel == "nuts":
        init0 = jax.tree.map(lambda x: x[0], initial_particles)
        dim = ravel_pytree(init0)[0].shape[0]
        mcmc_parameters = blackjax.smc.extend_params({
            "step_size": jnp.asarray(nuts_step_size),
            "inverse_mass_matrix": jnp.ones((dim,)),
        })
        nuts_kernel = blackjax.nuts.build_kernel()

        def mcmc_step_fn(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix):
            return nuts_kernel(rng_key, state, logdensity_fn, step_size,
                               inverse_mass_matrix, max_num_doublings=nuts_max_doublings)

        mcmc_init_fn = blackjax.nuts.init

    elif kernel == "hmc":
        # Gradient-based HMC mutation with a covariance-preconditioned diagonal mass
        # matrix (re-derived from the particle population each stage) and a damped
        # step-size adaptation toward hmc_target_accept -- the bjsi_blackjax recipe
        # for fast mixing of tightly-peaked, large-data posteriors.
        tuned = True
        adapt_step = bool(adapt_hmc_step_size)
        hmc_kernel = blackjax.hmc.build_kernel()
        mcmc_init_fn = blackjax.hmc.init

        def mcmc_step_fn(rng_key, state, logdensity_fn, step_size,
                         inverse_mass_matrix, num_integration_steps):
            return hmc_kernel(rng_key, state, logdensity_fn, step_size,
                              inverse_mass_matrix, num_integration_steps)

        def hmc_params(particles, step):
            # Inverse mass = diagonal population variance, raveled to a flat (d,)
            # vector aligned with the position's ravel order (HMC metric convention).
            var = jax.tree.map(lambda x: jnp.maximum(jnp.var(x, axis=0), 1e-8), particles)
            mass_flat, _ = ravel_pytree(var)
            return {
                "step_size": jnp.asarray([float(step)], dtype=mass_flat.dtype),
                "inverse_mass_matrix": mass_flat[None, :],
                "num_integration_steps": jnp.asarray([int(hmc_num_integration_steps)]),
            }

        # The step size from this update is immediately overwritten by the
        # Python-loop adaptation below; only the mass matrix is consumed from here.
        def mcmc_parameter_update_fn(key, state, info):
            return hmc_params(state.particles, hmc_step_size)

        init_param = hmc_params(initial_particles, hmc_step_size)

    elif kernel == "rmh":
        rmh_kernel = blackjax_rw.build_additive_step()
        mcmc_init_fn = blackjax.rmh.init

        if adapt_proposal:
            tuned = True
            d = _param_dim(initial_particles)
            factor = proposal_factor if proposal_factor is not None else 2.38 / np.sqrt(d)

            # Per-parameter RW scale = factor * population std, raveled to a flat (d,)
            # vector aligned with the position's ravel order (what blackjax_rw.normal wants).
            def pop_scale(particles):
                per_key = jax.tree.map(
                    lambda x: factor * jnp.maximum(jnp.std(x, axis=0), 1e-4), particles)
                flat, _ = ravel_pytree(per_key)
                return blackjax.smc.extend_params({"sigma": flat})

            def mcmc_step_fn(rng_key, state, logdensity_fn, sigma):
                return rmh_kernel(rng_key, state, logdensity_fn, blackjax_rw.normal(sigma))

            def mcmc_parameter_update_fn(key, state, info):
                return pop_scale(state.particles)

            init_param = pop_scale(initial_particles)
        else:
            def mcmc_step_fn(rng_key, state, logdensity_fn, proposal_scale):
                return rmh_kernel(rng_key, state, logdensity_fn, blackjax_rw.normal(proposal_scale))

            mcmc_parameters = blackjax.smc.extend_params(
                {"proposal_scale": jnp.asarray(proposal_scale, dtype=jnp.float64)})

    elif kernel == "mwg":
        # Metropolis-within-Gibbs: cycle parameter blocks, each with its own
        # additive-RMH proposal + inner-step count. Slots into the same
        # inner_kernel_tuning / waste-free path as the joint RMH below; only the
        # step fn and the per-block scale-update fn differ.
        tuned = True
        mcmc_init_fn = blackjax.rmh.init
        field_names = list(initial_particles.keys())
        blocks = build_blocks_from_keys(field_names)
        inner_steps = default_inner_steps(blocks, mechanism_steps=mwg_mechanism_steps)
        mcmc_step_fn = build_mwg_kernel(blocks, inner_steps)

        initial_stds = {n: jnp.std(initial_particles[n]) for n in field_names}

        def block_scales(particles, block_acc=None):
            cur = {n: jnp.std(particles[n]) for n in field_names}
            # min_ratio floors the proposal scale at a fraction of the PRIOR
            # width; the eq02387 waveform posterior is ~60x narrower than the
            # prior, so the default 0.1 floor pinned acceptance at ~0.03
            # (proposals 16x too big to ever be accepted). 0.005 keeps the
            # anti-collapse guard while letting std-tracking reach the true
            # posterior scale.
            sc = block_scale_overrides(blocks, initial_stds, cur, min_ratio=0.005,
                                       block_acceptance_rate=block_acc)
            return blackjax.smc.extend_params(sc)

        def mcmc_parameter_update_fn(key, state, info):
            # Increment 2: drive each block's proposal toward its target acceptance.
            # block_acceptance_rates has the block axis LAST and an unknown number of
            # leading axes (waste-free stacks scan x particles), so reduce everything
            # but the last axis to a per-block scalar. Falls back to pure std-tracking
            # if the field doesn't survive the SMC info wrapping.
            upd = getattr(info, "update_info", info)
            arr = getattr(upd, "block_acceptance_rates", None)
            block_acc = None
            if arr is not None:
                arr = jnp.asarray(arr)
                m = jnp.mean(arr.reshape(-1, arr.shape[-1]), axis=0)  # (n_blocks,)
                block_acc = {b.name: m[i] for i, b in enumerate(blocks)}
            return block_scales(state.particles, block_acc)

        init_param = block_scales(initial_particles)
    else:
        raise ValueError(f"unknown kernel: {kernel}")

    if tuned:
        smc = inner_kernel_tuning.as_top_level_api(
            smc_algorithm=blackjax.adaptive_tempered_smc,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            mcmc_step_fn=mcmc_step_fn,
            mcmc_init_fn=mcmc_init_fn,
            resampling_fn=systematic,
            mcmc_parameter_update_fn=mcmc_parameter_update_fn,
            initial_parameter_value=init_param,
            num_mcmc_steps=smc_num_mcmc_steps,
            target_ess=target_ess,
            **smc_extra,
        )
    else:
        smc = blackjax.adaptive_tempered_smc(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            mcmc_step_fn=mcmc_step_fn,
            mcmc_init_fn=mcmc_init_fn,
            mcmc_parameters=mcmc_parameters,
            resampling_fn=systematic,
            target_ess=target_ess,
            num_mcmc_steps=smc_num_mcmc_steps,
            **smc_extra,
        )

    state = smc.init(initial_particles)
    step = jax.jit(smc.step)

    # Step-size adaptation needs the inner_kernel_tuning state form so we can
    # mutate state.parameter_override["step_size"] between iterations.
    can_adapt_step = adapt_step and hasattr(state, "parameter_override")
    cur_step = float(hmc_step_size)

    t0 = time.time()
    it = 0
    for it in range(max_iterations):
        if float(_unwrap(state).tempering_param) >= 1.0 - 1e-6:
            break
        prev_beta = float(_unwrap(state).tempering_param)
        rng_key, subkey = random.split(rng_key)
        state, info = step(subkey, state)
        ss = _unwrap(state)
        beta = float(ss.tempering_param)
        delta_beta = beta - prev_beta
        if not np.isfinite(beta):
            raise RuntimeError("SMC produced non-finite tempering parameter (check log-density).")
        acc = _acceptance(info)
        if can_adapt_step and acc is not None and np.isfinite(acc):
            # Damped multiplicative update on log step toward target acceptance;
            # rate < 1 absorbs the early-iter accept~=1 transient.
            cur_step = float(np.clip(
                cur_step * np.exp(hmc_step_adaptation_rate * (acc - hmc_target_accept)),
                hmc_step_min, hmc_step_max))
            ex = state.parameter_override["step_size"]
            state = state._replace(parameter_override={
                **state.parameter_override,
                "step_size": jnp.asarray([cur_step], dtype=ex.dtype).reshape(ex.shape),
            })
        if verbose:
            ess = float(1.0 / jnp.sum(ss.weights ** 2))
            msg = f"  stage {it + 1}: beta={beta:.6g} dβ={delta_beta:.2e} ESS={ess:.0f}/{n_particles}"
            if acc is not None:
                msg += f" acc={acc:.3f}"
            if can_adapt_step:
                msg += f" eps={cur_step:.4f}"
            print(msg, flush=True)

    # Rejuvenate at beta=1: each extra step reweights with delta_beta=0 (uniform weights)
    # then resamples + moves, curing degenerate final-increment weights and decorrelating.
    for _ in range(final_rejuvenation):
        rng_key, subkey = random.split(rng_key)
        state, info = step(subkey, state)

    ss = _unwrap(state)
    rng_key, rkey = random.split(rng_key)
    idx = systematic(rkey, ss.weights, n_particles)
    particles = jax.tree.map(lambda x: x[idx], ss.particles)

    elapsed = time.time() - t0
    final_ess = float(1.0 / jnp.sum(ss.weights ** 2))
    n_unique = int(np.unique(np.asarray(idx)).size)
    if verbose:
        print(f"SMC done: {it + 1} stages (+{final_rejuvenation} rejuv), "
              f"beta={float(ss.tempering_param):.6g}, ESS_final={final_ess:.0f}, "
              f"unique={n_unique}/{n_particles}, {elapsed:.1f}s", flush=True)

    return {
        "particles": particles,
        "weights": np.full(n_particles, 1.0 / n_particles),
        "final_beta": float(ss.tempering_param),
        "final_ess": final_ess,
        "n_unique": n_unique,
        "n_stages": it + 1,
        "elapsed": elapsed,
    }


def _acceptance(info):
    upd = getattr(info, "update_info", None)
    for obj in (upd, info):
        if obj is not None and hasattr(obj, "acceptance_rate"):
            try:
                return float(np.asarray(obj.acceptance_rate).mean())
            except Exception:
                pass
    return None
