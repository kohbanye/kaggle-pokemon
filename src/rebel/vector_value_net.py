"""Infostate VALUE-VECTOR net for ReBeL (proper form) -- numpy serving.

The scalar `ValueNet` gives ONE value for a determinized leaf (a belief-averaged point
estimate). Proper ReBeL evaluates a Public Belief State to a VECTOR of counterfactual
values -- one per infostate in the belief range. Here the "infostates" are the fixed
archetype hypotheses, so the net maps the public (acting-player) features to K values
and the leaf in world k (drawn from hypothesis h_k) reads output[h_k]. CFR then gets the
correct per-world counterfactual value instead of a shared scalar (removes the
belief-averaging approximation). Trained on the per-world CFR root values from a solve.

Train torch, serve numpy, exact bridge + parity test (like the scalar ValueNet split).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.net.nn import he_init
from src.rebel.value_net import VALUE_IN_DIM

if TYPE_CHECKING:
    from numpy.typing import NDArray

_DEFAULT_HIDDEN = (64,)


class VectorValueNet:
    """N-layer numpy MLP: features -> K values in [-1, 1] (ReLU hidden, tanh out).

    Params ``w1,b1,...,wL,bL``; the last layer's output width is K (the #hypotheses)."""

    def __init__(self, params: dict[str, NDArray[np.float64]]) -> None:
        self.params = params

    @property
    def in_dim(self) -> int:
        return int(self.params["w1"].shape[0])

    @property
    def out_dim(self) -> int:
        return int(self.params[f"w{self.n_layers}"].shape[1])

    @property
    def n_layers(self) -> int:
        return sum(1 for k in self.params if k.startswith("w"))

    @classmethod
    def random(
        cls, rng: np.random.Generator, out_dim: int, in_dim: int = VALUE_IN_DIM,
        hidden: tuple[int, ...] = _DEFAULT_HIDDEN,
    ) -> VectorValueNet:
        params: dict[str, NDArray[np.float64]] = {}
        dims = [in_dim, *hidden]
        for i in range(1, len(dims)):
            params[f"w{i}"] = he_init(rng, dims[i - 1], dims[i])
            params[f"b{i}"] = np.zeros(dims[i])
        last = len(hidden) + 1
        params[f"w{last}"] = rng.standard_normal((dims[-1], out_dim)) * 0.01
        params[f"b{last}"] = np.zeros(out_dim)
        return cls(params)

    def values(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Full K-vector of values for features x."""
        p = self.params
        n = self.n_layers
        h = x
        for i in range(1, n):
            h = np.maximum(0.0, h @ p[f"w{i}"] + p[f"b{i}"])
        return np.tanh(h @ p[f"w{n}"] + p[f"b{n}"])

    def value(self, x: NDArray[np.float64], k: int) -> float:
        """Value for hypothesis/world index k."""
        return float(self.values(x)[k])

    def save(self, path: str | Path) -> None:
        np.savez(Path(path), **self.params)

    @classmethod
    def load(cls, path: str | Path) -> VectorValueNet:
        with np.load(Path(path)) as data:
            return cls({k: np.asarray(data[k], dtype=np.float64) for k in data.files})
