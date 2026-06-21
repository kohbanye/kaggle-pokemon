"""PolicyValueNet -- the pure-numpy policy / value / deck-construction skeleton.

A shared trunk feeds three heads, mirroring the paper's shared-embedding net:

- **value**   ``state -> scalar in [-1, 1]`` (tanh): win-ness of the position.
- **policy**  ``(state, presented options) -> one logit per option``. Scoring the
  *presented* options (instead of a fixed global action space) is what makes the
  net round-trip the Kaggle ``obs -> list[int]`` contract for free and stay legal
  -- the engine only ever offers legal options.
- **CB**      ``(LSTM hidden, candidate card) -> one logit per card``, run
  autoregressively at init to build a deck one legal card at a time. An LSTM over
  the picked-card sequence carries the running composition, so each pick is
  conditioned on the cards already chosen (Phase 5c) -- the head can balance the
  deck (e.g. add energy once it has enough attackers).

Weights are a flat ``dict[str, ndarray]`` saved/loaded as a single ``.npz``, so
the submission runtime is numpy alone (plan SS D). Training happens in the torch
mirror (:mod:`src.net.torch_model`) and the trained weights are exported back
into this dict; a parity test keeps the two forwards identical.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.net.encode import OPTION_DIM, STATE_DIM
from src.net.features import CARD_FEAT_DIM
from src.net.nn import he_init, sigmoid

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
    # Learned card embedding for the CB head (Phase 5b). The head's input width is
    # ``card_dim + embed_dim`` (fixed features concatenated with the embedding row);
    # ``n_cards`` is the build pool size, so the table is ``(n_cards + 1, embed_dim)``
    # (the trailing row is UNK). ``n_cards == 0`` => a 1-row UNK-only table (the
    # default config used by parity tests; real CB nets pass the pool size).
    embed_dim: int = 16
    n_cards: int = 0
    # LSTM over the deck-build sequence (Phase 5c): the picked-card embedding is the
    # input, the hidden state h_t feeds the CB head so each pick is conditioned on
    # the cards already chosen. CB head input width = lstm_hidden + card_dim +
    # embed_dim. Small on purpose (8 demo decks -> 256 would overfit).
    lstm_hidden: int = 32


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
        p["cb_w1"] = he_init(
            rng, cfg.lstm_hidden + cfg.card_dim + cfg.embed_dim, cfg.cb_hidden,
        )
        p["cb_b1"] = np.zeros(cfg.cb_hidden)
        p["cb_w2"] = rng.standard_normal((cfg.cb_hidden, 1)) * _HEAD_SCALE
        p["cb_b2"] = np.zeros(1)
        # Near-zero card embedding (last row = UNK): an untrained row contributes
        # ~0, so a card is ranked by its fixed features alone until BC/RL moves it.
        p["cb_embed"] = rng.standard_normal((cfg.n_cards + 1, cfg.embed_dim)) * 0.01
        # Deck-build LSTM cell (torch nn.LSTMCell layout: (4H, in)/(4H, H), gates
        # packed i,f,g,o). cb_start is the t=0 input token (empty deck).
        h_lstm = cfg.lstm_hidden
        scale = 1.0 / np.sqrt(h_lstm)
        p["lstm_w_ih"] = rng.standard_normal((4 * h_lstm, cfg.embed_dim)) * scale
        p["lstm_w_hh"] = rng.standard_normal((4 * h_lstm, h_lstm)) * scale
        p["lstm_b_ih"] = np.zeros(4 * h_lstm)
        p["lstm_b_hh"] = np.zeros(4 * h_lstm)
        p["cb_start"] = rng.standard_normal(cfg.embed_dim) * 0.01
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
        """CB MLP over a pre-assembled ``(N, cb_w1_in)`` matrix -> one logit per row.

        ``cb_w1_in == lstm_hidden + card_dim + embed_dim``; assemble the rows with
        :meth:`card_logits_with_state` (prepends the LSTM hidden state).
        """
        if card_feats.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        z = np.maximum(0.0, card_feats @ self.params["cb_w1"] + self.params["cb_b1"])
        return (z @ self.params["cb_w2"] + self.params["cb_b2"]).reshape(-1)

    def lstm_step(
        self,
        x: NDArray[np.float64],
        h: NDArray[np.float64],
        c: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """One ``nn.LSTMCell`` step (gates packed i,f,g,o in torch's layout)."""
        p = self.params
        z = (
            x @ p["lstm_w_ih"].T + p["lstm_b_ih"]
            + h @ p["lstm_w_hh"].T + p["lstm_b_hh"]
        )
        n = h.shape[0]
        i = sigmoid(z[:n])
        f = sigmoid(z[n : 2 * n])
        g = np.tanh(z[2 * n : 3 * n])
        o = sigmoid(z[3 * n : 4 * n])
        c2 = f * c + i * g
        return o * np.tanh(c2), c2

    def card_logits_with_state(
        self,
        h: NDArray[np.float64],
        card_matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Score candidates given the LSTM hidden state ``h``.

        ``card_matrix`` is ``(N, card_dim + embed_dim)`` (fixed features + card
        embedding); ``h`` ``(lstm_hidden,)`` is broadcast and prepended, then scored
        by the CB MLP.
        """
        n = card_matrix.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        joint = np.concatenate([np.tile(h, (n, 1)), card_matrix], axis=1)
        return self.card_logits(joint)

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
        """Load a parameter dict saved by :meth:`save`.

        When ``config`` is omitted, the embedding dims (``n_cards`` / ``embed_dim``)
        and ``lstm_hidden`` are recovered from the saved ``cb_embed`` / ``lstm_w_hh``
        shapes so the torch bridge (``from_numpy_net``) builds a matching-sized net.
        Other widths keep their defaults (they are the only ones we vary). Weights
        saved before the Phase-5c LSTM lack ``lstm_w_ih`` and are rejected -- they
        must be re-trained (the LSTM has no pre-image in the old weights).
        """
        with np.load(Path(path)) as data:
            params = {k: np.asarray(data[k], dtype=np.float64) for k in data.files}
        if config is None:
            if "lstm_w_ih" not in params:
                msg = "pre-LSTM weights; retrain with scripts/train_bc.py"
                raise ValueError(msg)
            emb = params["cb_embed"]
            config = replace(
                NetConfig(),
                n_cards=int(emb.shape[0]) - 1,
                embed_dim=int(emb.shape[1]),
                lstm_hidden=int(params["lstm_w_hh"].shape[1]),
            )
        return cls(config, params)

    def param_count(self) -> int:
        """Total number of scalar parameters (for logging / budget checks)."""
        return int(sum(v.size for v in self.params.values()))
