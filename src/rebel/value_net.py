"""Leaf value network for the ReBeL subgame solve (design step E).

Replaces the crude prize heuristic (`engine_game.default_leaf_value`) at depth-limited
leaves with a learned estimate. First honest scope: a SCALAR value net V(features) ->
[-1, 1] for the acting player's win value at the leaf; the leaf callable returns it from
player 0's view (negate on player-1 leaves). Features are the same the play net reads --
`encode_state` (board, me-then-acting) ++ the belief-weighted opponent context -- so the
leaf sees the position AND who it is likely facing. The full ReBeL infostate value
VECTOR (a value per private state, from a proper range) is a later refinement; a scalar
per determinized leaf is the tractable start.

Train in torch, serve in numpy (same split as the play net): `TorchValueNet` mirrors
this forward with an exact bridge, pinned by a parity test. Pure numpy for serving,
engine-free (the caller supplies `CardFeatures` + the hypothesis set).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.net.encode import STATE_DIM, deck_context, encode_state
from src.net.features import CARD_FEAT_DIM
from src.net.nn import he_init
from src.search.opp_belief import consistency_scores, seen_opponent_ids

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from src.net.features import CardFeatures

VALUE_IN_DIM = STATE_DIM + CARD_FEAT_DIM   # state features + opponent-belief context
_DEFAULT_HIDDEN = (64,)
_OPP_SHARPNESS = 6.0


def _as_hidden(hidden: int | tuple[int, ...]) -> tuple[int, ...]:
    return (hidden,) if isinstance(hidden, int) else tuple(hidden)


class ValueNet:
    """N-layer numpy MLP: features -> scalar in [-1, 1] (ReLU hidden, tanh out).

    Params are ``w1,b1,...,wL,bL`` for L linear layers (L-1 ReLU + tanh). Backward-
    compatible with the original 2-layer nets (w1,b1,w2,b2)."""

    def __init__(self, params: dict[str, NDArray[np.float64]]) -> None:
        self.params = params

    @property
    def in_dim(self) -> int:
        return int(self.params["w1"].shape[0])

    @property
    def n_layers(self) -> int:
        return sum(1 for k in self.params if k.startswith("w"))

    @classmethod
    def random(
        cls, rng: np.random.Generator, in_dim: int = VALUE_IN_DIM,
        hidden: int | tuple[int, ...] = _DEFAULT_HIDDEN,
    ) -> ValueNet:
        hid = _as_hidden(hidden)
        params: dict[str, NDArray[np.float64]] = {}
        dims = [in_dim, *hid]
        for i in range(1, len(dims)):
            params[f"w{i}"] = he_init(rng, dims[i - 1], dims[i])
            params[f"b{i}"] = np.zeros(dims[i])
        last = len(hid) + 1
        params[f"w{last}"] = rng.standard_normal((dims[-1], 1)) * 0.01
        params[f"b{last}"] = np.zeros(1)
        return cls(params)

    def value(self, x: NDArray[np.float64]) -> float:
        p = self.params
        n = self.n_layers
        h = x
        for i in range(1, n):  # hidden layers: ReLU
            h = np.maximum(0.0, h @ p[f"w{i}"] + p[f"b{i}"])
        return float(np.tanh(h @ p[f"w{n}"] + p[f"b{n}"]).reshape(-1)[0])

    def save(self, path: str | Path) -> None:
        np.savez(Path(path), **self.params)

    @classmethod
    def load(cls, path: str | Path) -> ValueNet:
        with np.load(Path(path)) as data:
            return cls({k: np.asarray(data[k], dtype=np.float64) for k in data.files})


def _opp_context(current: dict, your_index: int, decks: list[list[int]],
                 ctxs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Belief-weighted expected opponent deck_context (mirror of net/opp_context, kept
    local so value_net has no play-net dependency)."""
    seen = seen_opponent_ids(current, your_index)
    if not seen or len(ctxs) == 0:
        return ctxs.mean(axis=0) if len(ctxs) else np.zeros(CARD_FEAT_DIM)
    frac = consistency_scores(decks, seen) / len(seen)
    w = np.exp(_OPP_SHARPNESS * (frac - frac.max()))
    w /= w.sum()
    return w @ ctxs


def state_features(
    current: dict, your_index: int, feats: CardFeatures,
    decks: list[list[int]], ctxs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Value-net input for a leaf: encoded state ++ belief-weighted opponent context."""
    return np.concatenate([
        encode_state(current, your_index, feats),
        _opp_context(current, your_index, decks, ctxs),
    ])


def build_hypothesis_ctxs(
    decks: list[list[int]], feats: CardFeatures,
) -> NDArray[np.float64]:
    """deck_context matrix for the belief hypotheses (built once per agent)."""
    return np.asarray([deck_context(d, feats) for d in decks])


def make_leaf_value(
    net: ValueNet, feats: CardFeatures, decks: list[list[int]],
    ctxs: NDArray[np.float64],
) -> Callable[[object], float]:
    """A `leaf_value` callable for `BeliefSubgame`: player-0 value from the net.

    Evaluates from the acting player's perspective and negates on player-1 leaves, so
    the return is always player 0's value (the game's utility convention)."""
    def leaf(state: object) -> float:
        cur = state.observation.current  # type: ignore[attr-defined]
        if cur is None:
            return 0.0
        acting = int(cur.yourIndex)
        x = state_features(_as_dict(cur), acting, feats, decks, ctxs)
        v = net.value(x)
        return v if acting == 0 else -v
    return leaf


def _as_dict(cur: object) -> dict:
    """`obs.current` is an engine dataclass at serve, already a dict in tests."""
    if isinstance(cur, dict):
        return cur
    from src.search.ismcts import _plain  # noqa: PLC0415
    return _plain(cur)  # type: ignore[return-value]
