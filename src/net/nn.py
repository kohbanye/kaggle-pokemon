"""Minimal numpy neural-net primitives (forward + backward).

Just enough to build the Phase-3 net and run the learning-wiring sanity (and
later the Phase-4 BC trainer) without a deep-learning framework: dense layers,
ReLU and a masked-softmax cross-entropy. Each ``*_forward`` returns a cache that
the matching ``*_backward`` consumes, so a small training loop can be assembled
by hand. Pure numpy -- numpy is the project's one declared runtime dependency.
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


def linear_forward(
    x: NDArray[np.float64],
    w: NDArray[np.float64],
    b: NDArray[np.float64],
) -> tuple[NDArray[np.float64], tuple]:
    """Affine map ``y = x @ w + b`` with a cache for the backward pass."""
    return x @ w + b, (x, w)


def linear_backward(
    dy: NDArray[np.float64],
    cache: tuple,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Gradients ``(dx, dw, db)`` for :func:`linear_forward`."""
    x, w = cache
    dx = dy @ w.T
    dw = x.T @ dy
    db = dy.sum(axis=0)
    return dx, dw, db


def relu_forward(
    x: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """ReLU activation; the cache is the positive-input mask."""
    mask = x > 0
    return x * mask, mask


def relu_backward(
    dy: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Gradient for :func:`relu_forward`."""
    return dy * mask


def softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise softmax, numerically stabilised."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def softmax_ce(
    logits: NDArray[np.float64],
    targets: NDArray[np.int_],
) -> tuple[float, NDArray[np.float64]]:
    """Mean softmax cross-entropy over a ``(batch, n_class)`` logit matrix.

    Returns the scalar loss and the gradient ``dlogits`` (already averaged over
    the batch), i.e. ``(probs - onehot(targets)) / batch``.
    """
    probs = softmax(logits)
    batch = logits.shape[0]
    rows = np.arange(batch)
    loss = float(-np.mean(np.log(probs[rows, targets] + 1e-12)))
    grad = probs.copy()
    grad[rows, targets] -= 1.0
    grad /= batch
    return loss, grad
