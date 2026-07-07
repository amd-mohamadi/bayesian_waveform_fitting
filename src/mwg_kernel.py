"""Metropolis-within-Gibbs (MWG) mutation kernel for the tempered SMC sampler.

Ported (self-contained) from SMTI's ``mwg_kernel.py`` + ``blockwise_rmh.py``. The
single joint random-walk used by ``run_smc(kernel="rmh")`` moves every parameter
together; on the tightly-peaked small-event posterior it mixes poorly and the
source-type lune coordinates (gamma, delta) are never explored on their own, so
the ensemble collapses into a non-double-couple pocket. MWG instead cycles
through parameter *blocks*, updating each with its own additive-RMH proposal (and
its own number of inner moves) while the other blocks are held fixed -- restoring
particle diversity and letting the source-type block be explored independently.

The kernel operates on the unconstrained (logit / real-line) particle dict, so no
boundary reflection is needed; physical bounds are applied later in
``parameterization.m6_physical``. The proposal scale per block is a single scalar
broadcast over that block's fields, derived from the population std the same way
the existing ``pop_scale`` derives the joint RMH scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, NamedTuple, Sequence

import jax.numpy as jnp
from jax import random

import blackjax.mcmc.random_walk as blackjax_rw


@dataclass(frozen=True)
class RMHBlock:
    name: str
    fields: tuple


class MWGInfo(NamedTuple):
    """Info returned by one MWG rejuvenation step."""
    acceptance_rate: jnp.ndarray          # overall (mean across blocks)
    is_accepted: jnp.ndarray              # from the last block (BlackJAX compat)
    proposal: jnp.ndarray                 # from the last block
    block_acceptance_rates: jnp.ndarray   # (n_blocks,)


def _build_block_proposal(block: RMHBlock, scale):
    """Additive proposal that perturbs only ``block``'s fields (scalar scale)."""

    def random_step(rng_key, position):
        keys = random.split(rng_key, len(block.fields))
        move = {name: jnp.zeros_like(val) for name, val in position.items()}
        for key, field in zip(keys, block.fields):
            move[field] = scale * random.normal(
                key, shape=position[field].shape, dtype=position[field].dtype)
        return move

    return random_step


def build_mwg_kernel(blocks: Sequence[RMHBlock], inner_steps_per_block=None):
    """Metropolis-within-Gibbs kernel compatible with BlackJAX SMC.

    Returns ``mwg_step_fn(rng_key, state, logdensity_fn, **block_scales)`` where
    each ``block_scales[name]`` is that block's scalar proposal std. The block
    names must match the kwarg names supplied by ``inner_kernel_tuning`` (i.e. the
    keys of the parameter-override dict).
    """
    rmh_kernel = blackjax_rw.build_additive_step()
    inner_steps_per_block = inner_steps_per_block or {}

    def mwg_step_fn(rng_key, state, logdensity_fn, **block_scales):
        current_state = state
        block_keys = random.split(rng_key, len(blocks))
        block_acc_rates = []
        last_info = None
        for block_key, block in zip(block_keys, blocks):
            n_inner = inner_steps_per_block.get(block.name, 1)
            scale = block_scales[block.name]
            inner_keys = random.split(block_key, n_inner)
            block_acc_sum = jnp.asarray(0.0)
            for step_idx in range(n_inner):
                step_fn = _build_block_proposal(block, scale)
                current_state, info = rmh_kernel(
                    inner_keys[step_idx], current_state, logdensity_fn, step_fn)
                block_acc_sum = block_acc_sum + info.acceptance_rate
                last_info = info
            block_acc_rates.append(block_acc_sum / n_inner)
        return current_state, MWGInfo(
            acceptance_rate=jnp.mean(jnp.stack(block_acc_rates)),
            is_accepted=last_info.is_accepted,
            proposal=last_info.proposal,
            block_acceptance_rates=jnp.stack(block_acc_rates),
        )

    return mwg_step_fn


# --- project-specific block layout / scaling ------------------------------

# Inner RMH moves per block per SMC rejuvenation. The source-type block gets
# extra moves because (gamma, delta) drive the DC/non-DC tradeoff this kernel is
# meant to fix; the mechanism block gets a few because (kappa, h, sigma) mix
# slowly. magnitude/noise are near-Gaussian and need only one.
DEFAULT_INNER_STEPS = {"source_type": 2, "mechanism": 3, "magnitude": 1, "noise": 1,
                       "location": 2}

# Per-block target acceptance for the scale adaptation (Roberts/Rosenthal optimal
# rates by block dimension): ~0.44 for low-d blocks, ~0.30 for the d=3 mechanism.
DEFAULT_BLOCK_TARGET_ACCEPTANCE = {
    "source_type": 0.44, "mechanism": 0.30, "magnitude": 0.44, "noise": 0.44,
    "location": 0.30}


def build_blocks_from_keys(particle_keys):
    """Block set for this project's particle dict (source_type only if present)."""
    keys = set(particle_keys)
    blocks = []
    if "gamma" in keys and "delta" in keys:
        blocks.append(RMHBlock("source_type", ("gamma", "delta")))
    blocks.append(RMHBlock("mechanism", ("kappa", "h", "sigma")))
    blocks.append(RMHBlock("magnitude", ("mag",)))
    blocks.append(RMHBlock("noise", ("hp",)))
    if "loc" in keys:
        blocks.append(RMHBlock("location", ("loc",)))
    return tuple(blocks)


def default_inner_steps(blocks, mechanism_steps=None):
    """Inner-step dict for ``blocks``; ``mechanism_steps`` overrides the default."""
    steps = dict(DEFAULT_INNER_STEPS)
    if mechanism_steps is not None:
        steps["mechanism"] = int(mechanism_steps)
    return {b.name: steps.get(b.name, 1) for b in blocks}


def block_scale_overrides(blocks, initial_stds, current_stds, min_ratio=0.1,
                          block_acceptance_rate=None, adaptation_rate=1.5,
                          accept_mult_bounds=(0.5, 4.0)):
    """Per-block scalar RW scale = (2.38/sqrt(d)) * mean(field stds) * accept_mult.

    Mirrors SMTI's std-tracking rule. ``initial_stds`` supplies a floor so a block
    scale never collapses below ``min_ratio * optimal_factor * mean(initial stds)``.

    If ``block_acceptance_rate`` (a {block_name: rate} dict) is given, each scale is
    multiplied by ``exp(adaptation_rate * (acc - target))`` (clipped) to drive the
    block's acceptance toward ``DEFAULT_BLOCK_TARGET_ACCEPTANCE`` -- increment 2.
    Passing ``None`` recovers the pure std-tracking behaviour (increment 1).
    """
    lo, hi = accept_mult_bounds
    scales = {}
    for b in blocks:
        d = len(b.fields)
        optimal = 2.38 / jnp.sqrt(jnp.asarray(d, dtype=jnp.float64))
        cur = jnp.mean(jnp.asarray([current_stds[f] for f in b.fields]))
        ini = jnp.mean(jnp.asarray([initial_stds[f] for f in b.fields]))
        scale = jnp.maximum(optimal * cur, min_ratio * optimal * ini)
        if block_acceptance_rate is not None and b.name in block_acceptance_rate:
            target = DEFAULT_BLOCK_TARGET_ACCEPTANCE.get(b.name, 0.30)
            mult = jnp.exp(adaptation_rate * (block_acceptance_rate[b.name] - target))
            scale = scale * jnp.clip(mult, lo, hi)
        scales[b.name] = scale
    return scales
