"""V-Trace off-policy correction + PPO surrogate (the paper's RL core).

The ByteDance Hearthstone paper (arXiv:2303.05197 SS5) trains a distributed
actor-learner with **V-Trace** (Espeholt et al. 2018, IMPALA) for off-policy
correction and a **PPO** clipped surrogate on top. The current Phase-5d stack used
plain REINFORCE; this module is the faithful replacement.

It is **pure numpy** (no torch, no ``cg``) so the whole correction is unit-testable
on the host -- the learner (:class:`~src.net.lit_vtrace.LitVtracePPO`) detaches its
behaviour log-probs / target log-probs / values to numpy, calls :func:`vtrace`, and
feeds the returned (detached) targets back as a value-regression target and a
policy-gradient advantage. V-Trace targets are *targets*, never differentiated
through, so computing them outside the autograd graph is exactly right.

Sequences are batch-first ``(B, T)`` and right-padded; a ``valid`` mask marks the
real steps. Episodes here are complete (the game ends), so the bootstrap value is
typically ``0`` and the only non-zero reward is the terminal ±1.

Conventions (IMPALA, with the paper's ρ lower-clip "improved technique"):

- ``ratio_t = π(a_t|s_t) / μ(a_t|s_t)`` (from log-probs),
- ``ρ_t = clip(ratio_t, rho_min, clip_rho)`` -- the lower clip stops the trace
  vanishing when π drifts far below μ; the upper clip ``ρ̄`` caps variance,
- ``c_t = min(ratio_t, clip_c)`` -- the trace-cutting coefficient,
- ``δ_t = ρ_t (r_t + γ V_{t+1} − V_t)``,
- ``v_t = V_t + δ_t + γ c_t (v_{t+1} − V_{t+1})`` (backward recursion,
  ``v`` past the last real step ``= bootstrap``),
- PG advantage ``Â_t = ρ_t (r_t + γ v_{t+1} − V_t)``.

On-policy (``ratio ≡ 1``, ``γ = 1``, terminal-only reward) every ``v_t`` equals the
Monte-Carlo return -- :func:`tests.test_vtrace` pins this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class VTraceReturns:
    """Output of :func:`vtrace` (all ``(B, T)``, zero on padded steps).

    - ``vs``: the V-Trace value targets ``v_t`` (regress the value head onto these).
    - ``pg_advantages``: ``Â_t`` for the policy-gradient / PPO surrogate.
    - ``clipped_rho``: ``ρ_t`` (returned for logging / diagnostics).
    """

    vs: Array
    pg_advantages: Array
    clipped_rho: Array


def _shift_back(values: Array, valid: NDArray[np.bool_], bootstrap: Array) -> Array:
    """``next[:, t] = values[:, t+1]``, but ``= bootstrap`` at each row's last step.

    Right-padding means the column after the last real step holds garbage, so we
    place ``bootstrap`` exactly at the last valid index (where ``valid`` is True and
    the next step is not) rather than at ``T-1``.
    """
    shifted = np.concatenate([values[:, 1:], np.zeros_like(values[:, :1])], axis=1)
    next_valid = np.concatenate(
        [valid[:, 1:], np.zeros_like(valid[:, :1])], axis=1,
    )
    is_last = valid & ~next_valid
    return np.where(is_last, bootstrap[:, None], shifted)


def vtrace(  # noqa: PLR0913 - the V-Trace inputs are irreducibly several
    behaviour_logp: Array,
    target_logp: Array,
    values: Array,
    rewards: Array,
    valid: NDArray[np.bool_],
    *,
    bootstrap_value: Array | None = None,
    gamma: float = 1.0,
    clip_rho: float = 1.0,
    clip_c: float = 1.0,
    rho_min: float = 0.0,
) -> VTraceReturns:
    """Compute V-Trace value targets and policy-gradient advantages.

    All sequence arrays are ``(B, T)`` and right-padded; ``valid`` is the ``(B, T)``
    bool mask of real steps. ``bootstrap_value`` is ``(B,)`` (default zeros -- the
    episodes here are complete, so there is no successor to bootstrap from).
    """
    behaviour_logp = np.asarray(behaviour_logp, dtype=np.float64)
    target_logp = np.asarray(target_logp, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64) * valid
    rewards = np.asarray(rewards, dtype=np.float64) * valid
    bsz = values.shape[0]
    boot = (
        np.zeros(bsz, dtype=np.float64)
        if bootstrap_value is None
        else np.asarray(bootstrap_value, dtype=np.float64)
    )

    ratio = np.exp(np.where(valid, target_logp - behaviour_logp, 0.0))
    rho = np.clip(ratio, rho_min, clip_rho) * valid
    c = np.minimum(ratio, clip_c) * valid

    v_next = _shift_back(values, valid, boot)
    delta = rho * (rewards + gamma * v_next - values)

    # Backward recursion for the V-Trace targets. ``carry`` is v_{t+1}; it starts at
    # the bootstrap and only advances on real steps, so trailing pad steps (visited
    # first, right-padding) leave it untouched until the last real step.
    t_len = values.shape[1]
    vs = np.zeros_like(values)
    carry = boot.copy()
    for t in range(t_len - 1, -1, -1):
        vt = values[:, t] + delta[:, t] + gamma * c[:, t] * (carry - v_next[:, t])
        step_valid = valid[:, t]
        vs[:, t] = np.where(step_valid, vt, 0.0)
        carry = np.where(step_valid, vs[:, t], carry)

    vs_next = _shift_back(vs, valid, boot)
    pg_adv = rho * (rewards + gamma * vs_next - values) * valid
    return VTraceReturns(vs=vs, pg_advantages=pg_adv, clipped_rho=rho)


def ppo_policy_loss(
    target_logp: Array,
    behaviour_logp: Array,
    advantages: Array,
    valid: NDArray[np.bool_],
    *,
    clip_eps: float = 0.2,
) -> float:
    """PPO clipped surrogate **loss** (to minimise), averaged over valid steps.

    ``L = − E[min(r·Â, clip(r, 1±ε)·Â)]`` with ``r = π/μ`` and the (detached)
    V-Trace advantage ``Â``. This numpy version mirrors the torch one in the learner
    and exists so the surrogate's sign/clip behaviour is unit-tested independently.
    """
    valid_f = valid.astype(np.float64)
    ratio = np.exp(np.where(valid, target_logp - behaviour_logp, 0.0))
    unclipped = ratio * advantages
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    surrogate = np.minimum(unclipped, clipped) * valid_f
    denom = max(float(valid_f.sum()), 1.0)
    return float(-surrogate.sum() / denom)
