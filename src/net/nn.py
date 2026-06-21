"""Numpy helpers for the inference-side net forward.

The serving forward (:mod:`src.net.model`) is pure numpy so the submission stays
a light, numpy-only bundle (plan SS D). Training is done in torch + Lightning
(:mod:`src.net.torch_model`, :mod:`src.net.lit`) and the trained weights are
exported back to this numpy net; a parity test keeps the two forwards in lock-
step. These two helpers are all the numpy forward needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def he_init(
    rng: np.random.Generator,
    n_in: int,
    n_out: int,
) -> NDArray[np.float64]:
    """He-normal weight matrix ``(n_in, n_out)`` for ReLU layers."""
    scale = np.sqrt(2.0 / n_in)
    return rng.standard_normal((n_in, n_out)) * scale


def softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise softmax, numerically stabilised."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Logistic sigmoid, overflow-safe for large |x| (LSTM gate activation)."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out
