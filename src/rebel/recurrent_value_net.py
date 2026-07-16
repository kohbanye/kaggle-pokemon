"""History-aware (LSTM) leaf value net for the ReBeL subgame solve -- numpy serving.

The scalar `ValueNet` sees only the CURRENT leaf state (no memory). This recurrent net
consumes the SEQUENCE of per-decision ``state_features`` (self+opponent board over the
game) through a 1-layer LSTM, then a linear+tanh head on the last hidden -> scalar value
in [-1, 1]. So the leaf value depends on how the game UNFOLDED, not just the snapshot --
closer to the proper ReBeL value(PBS) which conditions on public history.

Served affordably by CACHING the LSTM state at the search root (encode real history
game history ONCE) then threading only the few in-search steps per leaf via :meth:`step`
once) then threading only the in-search steps per leaf via :meth:`step` (h_t).
by a parity test -- same train-torch / serve-numpy split as the other nets.

LSTM math matches ``torch.nn.LSTM`` (gate order i,f,g,o) so the bridge is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.rebel.value_net import VALUE_IN_DIM

if TYPE_CHECKING:
    from numpy.typing import NDArray

_DEFAULT_HIDDEN = 128


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x))


class RecurrentValueNet:
    """1-layer LSTM over a feature sequence -> scalar value in [-1, 1] (numpy serving).

    Params (matching torch nn.LSTM, gate order i,f,g,o):
      w_ih (in, 4H), w_hh (H, 4H), b (4H,)   -- LSTM cell
      hw (H, 1), hb (1,)                      -- value head
    """

    def __init__(self, params: dict[str, NDArray[np.float64]]) -> None:
        self.params = params

    @property
    def in_dim(self) -> int:
        return int(self.params["w_ih"].shape[0])

    @property
    def hidden(self) -> int:
        return int(self.params["w_hh"].shape[0])

    @classmethod
    def random(
        cls, rng: np.random.Generator, in_dim: int = VALUE_IN_DIM,
        hidden: int = _DEFAULT_HIDDEN,
    ) -> RecurrentValueNet:
        k = 1.0 / np.sqrt(hidden)
        return cls({
            "w_ih": rng.uniform(-k, k, (in_dim, 4 * hidden)),
            "w_hh": rng.uniform(-k, k, (hidden, 4 * hidden)),
            "b": rng.uniform(-k, k, 4 * hidden),
            "hw": rng.standard_normal((hidden, 1)) * 0.01,
            "hb": np.zeros(1),
        })

    def zero_state(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        h = self.hidden
        return np.zeros(h), np.zeros(h)

    def step(
        self, state: tuple[NDArray[np.float64], NDArray[np.float64]],
        x: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """One LSTM cell: (h,c),x -> (h',c'). Threads memory through the tree."""
        p = self.params
        h, c = state
        z = x @ p["w_ih"] + h @ p["w_hh"] + p["b"]
        n = self.hidden
        i = _sigmoid(z[:n])
        f = _sigmoid(z[n:2 * n])
        g = np.tanh(z[2 * n:3 * n])
        o = _sigmoid(z[3 * n:])
        c2 = f * c + i * g
        h2 = o * np.tanh(c2)
        return h2, c2

    def value_from_h(self, h: NDArray[np.float64]) -> float:
        p = self.params
        return float(np.tanh(h @ p["hw"] + p["hb"]).reshape(-1)[0])

    def value(self, seq: NDArray[np.float64]) -> float:
        """Value from a full feature sequence ``seq`` of shape (T, in_dim)."""
        state = self.zero_state()
        for x in seq:
            state = self.step(state, x)
        return self.value_from_h(state[0])

    def save(self, path: str | Path) -> None:
        np.savez(Path(path), **self.params)

    @classmethod
    def load(cls, path: str | Path) -> RecurrentValueNet:
        with np.load(Path(path)) as data:
            return cls({k: np.asarray(data[k], dtype=np.float64) for k in data.files})
