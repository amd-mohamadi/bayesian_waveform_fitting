"""Bridge to the proven SMTI Tape parameterization (single source of truth).

SMTI's ``tape_jax`` has no relative imports, so we import it directly by adding
SMTI/src to sys.path. We deliberately do NOT import SMTI's ``inversion_blackjax``
(it is coupled to the polarity/amplitude problem and uses relative imports); the
SMC sampler is reimplemented in sampler.py following the same blackjax pattern.
"""
import os
import sys

_SMTI_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SMTI", "src"))
if _SMTI_SRC not in sys.path:
    sys.path.insert(0, _SMTI_SRC)

import tape_jax  # noqa: E402  (also enables jax x64)

jax_Tape_MT33 = tape_jax.jax_Tape_MT33  # (gamma, delta, kappa, h, sigma) -> unit-norm 3x3 MT in NED
