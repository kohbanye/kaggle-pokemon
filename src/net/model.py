"""PolicyValueNet -- the pure-numpy policy / value / deck-construction skeleton.

A shared trunk feeds three heads, mirroring the paper's shared-embedding net:

- **value**   ``state -> scalar in [-1, 1]`` (tanh): win-ness of the position.
- **policy**  ``(state, presented options) -> one logit per option``. Scoring the
  *presented* options (instead of a fixed global action space) is what makes the
  net round-trip the Kaggle ``obs -> list[int]`` contract for free and stay legal
  -- the engine only ever offers legal options.
- **CB**      ``candidate card features -> one logit per card``, used at init to
  build a deck one legal card at a time. The init observation carries no
  information (``select`` and ``current`` are both None), so this head is
  context-free by design -- it just emits a card distribution.

Weights are a flat ``dict[str, ndarray]`` saved/loaded as a single ``.npz``, so
the submission runtime is numpy alone (plan SS D). Phase 3 only needs the
random-init forward; the matching backward lives in :mod:`src.net.nn` /
:mod:`src.net.train` for the learning-wiring sanity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.net.encode import OPTION_DIM, STATE_DIM
from src.net.features import CARD_FEAT_DIM
from src.net.nn import he_init

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class NetConfig:
    """Layer widths. Input dims default to the encoder's fixed feature sizes."""

    state_dim: int = STATE_DIM
    option_dim: int = OPTION_DIM
    card_dim: int = CARD_FEAT_DIM
    hidden: int = 64
    policy_hidden: int = 32
    cb_hidden: int = 32


# Small output-head init so random logits start near-uniform (stable softmax).
_HEAD_SCALE = 0.01


class PolicyValueNet:
    """Shared-trunk policy/value/CB net with a flat numpy parameter dict."""

    def __init__(
        self,
        config: NetConfig | None = None,
        params: dict[str, NDArray[np.float64]] | None = None,
    ) -> None:
        self.config = config or NetConfig()
        self.params: dict[str, NDArray[np.float64]] = params or {}

    # --- construction -------------------------------------------------------

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        config: NetConfig | None = None,
    ) -> PolicyValueNet:
        """A freshly random-initialised net (He weights, zero/biased small heads)."""
        cfg = config or NetConfig()
        p: dict[str, NDArray[np.float64]] = {}
        p["trunk_w1"] = he_init(rng, cfg.state_dim, cfg.hidden)
        p["trunk_b1"] = np.zeros(cfg.hidden)
        p["trunk_w2"] = he_init(rng, cfg.hidden, cfg.hidden)
        p["trunk_b2"] = np.zeros(cfg.hidden)
        p["value_w"] = rng.standard_normal((cfg.hidden, 1)) * _HEAD_SCALE
        p["value_b"] = np.zeros(1)
        p["policy_w1"] = he_init(rng, cfg.hidden + cfg.option_dim, cfg.policy_hidden)
        p["policy_b1"] = np.zeros(cfg.policy_hidden)
        p["policy_w2"] = rng.standard_normal((cfg.policy_hidden, 1)) * _HEAD_SCALE
        p["policy_b2"] = np.zeros(1)
        p["cb_w1"] = he_init(rng, cfg.card_dim, cfg.cb_hidden)
        p["cb_b1"] = np.zeros(cfg.cb_hidden)
        p["cb_w2"] = rng.standard_normal((cfg.cb_hidden, 1)) * _HEAD_SCALE
        p["cb_b2"] = np.zeros(1)
        return cls(cfg, p)

    # --- forward passes -----------------------------------------------------

    def trunk(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Shared body: ``state -> hidden`` (two ReLU layers). Accepts 1-D or 2-D."""
        x2 = np.atleast_2d(x)
        h = np.maximum(0.0, x2 @ self.params["trunk_w1"] + self.params["trunk_b1"])
        return np.maximum(0.0, h @ self.params["trunk_w2"] + self.params["trunk_b2"])

    def value(self, x: NDArray[np.float64]) -> float:
        """Scalar value in ``[-1, 1]`` for a single state vector."""
        h = self.trunk(x)
        v = np.tanh(h @ self.params["value_w"] + self.params["value_b"])
        return float(v.reshape(-1)[0])

    def policy_logits(
        self,
        x: NDArray[np.float64],
        option_feats: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """One logit per presented option (``option_feats`` is ``(K, option_dim)``)."""
        k = option_feats.shape[0]
        if k == 0:
            return np.zeros(0, dtype=np.float64)
        h = self.trunk(x)[0]  # (hidden,)
        joint = np.concatenate(
            [np.tile(h, (k, 1)), option_feats], axis=1,
        )  # (K, hidden+option_dim)
        z = np.maximum(0.0, joint @ self.params["policy_w1"] + self.params["policy_b1"])
        return (z @ self.params["policy_w2"] + self.params["policy_b2"]).reshape(-1)

    def card_logits(
        self,
        card_feats: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """One logit per candidate card (``card_feats`` is ``(N, card_dim)``)."""
        if card_feats.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        z = np.maximum(0.0, card_feats @ self.params["cb_w1"] + self.params["cb_b1"])
        return (z @ self.params["cb_w2"] + self.params["cb_b2"]).reshape(-1)

    # --- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the parameter dict to a single ``.npz`` archive."""
        np.savez(Path(path), **self.params)

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: NetConfig | None = None,
    ) -> PolicyValueNet:
        """Load a parameter dict saved by :meth:`save`."""
        with np.load(Path(path)) as data:
            params = {k: np.asarray(data[k], dtype=np.float64) for k in data.files}
        return cls(config or NetConfig(), params)

    def param_count(self) -> int:
        """Total number of scalar parameters (for logging / budget checks)."""
        return int(sum(v.size for v in self.params.values()))
