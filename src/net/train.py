"""Numpy SGD for the policy path -- the Phase-3 learning-wiring sanity.

Phase 3's last exit criterion is "the net can imitate moves to some degree (the
learning wiring works)". This module is that wiring: a hand-written forward +
backward + SGD step that trains the trunk and policy head to predict a target
option index from ``(state, options)`` batches. If backprop were wrong the loss
would not fall, so "loss decreases / accuracy beats chance" is itself the check.

The batched forward fixes the option count ``K`` per batch (so the
options tensor is a clean ``(batch, K, option_dim)``); the same primitives
generalise to the variable-length, engine-logged batches the Phase-4 behaviour-
cloning trainer will use. Value and CB heads are left untouched here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.net.nn import (
    linear_backward,
    linear_forward,
    relu_backward,
    relu_forward,
    softmax,
    softmax_ce,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.net.model import PolicyValueNet

# Parameters updated by the policy SGD step (value / CB heads are frozen here).
_TRAINED = (
    "trunk_w1", "trunk_b1", "trunk_w2", "trunk_b2",
    "policy_w1", "policy_b1", "policy_w2", "policy_b2",
)


def _policy_forward(
    net: PolicyValueNet,
    states: NDArray[np.float64],
    options: NDArray[np.float64],
) -> tuple[NDArray[np.float64], dict]:
    """Batched policy forward; returns ``(B, K)`` logits and a backward cache."""
    p = net.params
    batch, k, _ = options.shape

    a1, c1 = linear_forward(states, p["trunk_w1"], p["trunk_b1"])
    h1, m1 = relu_forward(a1)
    a2, c2 = linear_forward(h1, p["trunk_w2"], p["trunk_b2"])
    h, m2 = relu_forward(a2)  # (B, hidden)

    h_rep = np.repeat(h[:, None, :], k, axis=1)  # (B, K, hidden)
    joint = np.concatenate([h_rep, options], axis=2).reshape(batch * k, -1)
    z_pre, cp1 = linear_forward(joint, p["policy_w1"], p["policy_b1"])
    z, mp = relu_forward(z_pre)
    logit_flat, cp2 = linear_forward(z, p["policy_w2"], p["policy_b2"])
    logits = logit_flat.reshape(batch, k)

    cache = {
        "shape": (batch, k),
        "c1": c1, "m1": m1, "c2": c2, "m2": m2,
        "cp1": cp1, "mp": mp, "cp2": cp2,
        "hidden": h.shape[1],
    }
    return logits, cache


def _policy_backward(
    cache: dict,
    dlogits: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Gradients for the trunk + policy params given ``dL/dlogits`` ``(B, K)``."""
    batch, k = cache["shape"]
    hidden = cache["hidden"]

    dz, g_pw2, g_pb2 = linear_backward(dlogits.reshape(batch * k, 1), cache["cp2"])
    dz = relu_backward(dz, cache["mp"])
    djoint, g_pw1, g_pb1 = linear_backward(dz, cache["cp1"])

    # Only the hidden-state slice of the joint input carries trunk gradient; the
    # option-feature slice is a constant input. Sum the K copies of h back to one.
    dh = djoint.reshape(batch, k, -1)[:, :, :hidden].sum(axis=1)

    da2 = relu_backward(dh, cache["m2"])
    dh1, g_tw2, g_tb2 = linear_backward(da2, cache["c2"])
    da1 = relu_backward(dh1, cache["m1"])
    _, g_tw1, g_tb1 = linear_backward(da1, cache["c1"])

    return {
        "trunk_w1": g_tw1, "trunk_b1": g_tb1,
        "trunk_w2": g_tw2, "trunk_b2": g_tb2,
        "policy_w1": g_pw1, "policy_b1": g_pb1,
        "policy_w2": g_pw2, "policy_b2": g_pb2,
    }


def policy_sgd_step(
    net: PolicyValueNet,
    states: NDArray[np.float64],
    options: NDArray[np.float64],
    targets: NDArray[np.int_],
    lr: float,
) -> float:
    """One full-batch SGD step on the policy path; returns the loss before it."""
    logits, cache = _policy_forward(net, states, options)
    loss, dlogits = softmax_ce(logits, targets)
    grads = _policy_backward(cache, dlogits)
    for name in _TRAINED:
        net.params[name] = net.params[name] - lr * grads[name]
    return loss


def train_policy(  # noqa: PLR0913 - a trainer threads data + hyperparameters
    net: PolicyValueNet,
    states: NDArray[np.float64],
    options: NDArray[np.float64],
    targets: NDArray[np.int_],
    *,
    lr: float = 0.05,
    steps: int = 200,
) -> list[float]:
    """Run ``steps`` full-batch SGD steps; returns the per-step loss history."""
    return [
        policy_sgd_step(net, states, options, targets, lr) for _ in range(steps)
    ]


def policy_accuracy(
    net: PolicyValueNet,
    states: NDArray[np.float64],
    options: NDArray[np.float64],
    targets: NDArray[np.int_],
) -> float:
    """Fraction of samples whose argmax option matches the target."""
    logits, _ = _policy_forward(net, states, options)
    return float(np.mean(softmax(logits).argmax(axis=1) == targets))
