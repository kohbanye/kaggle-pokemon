"""Recurrent serving net -- the paper's obs-history LSTM on the play side.

The ByteDance Hearthstone paper (arXiv:2303.05197 SS4) aggregates the **observation
history** with an LSTM (hidden 256) so the net learns the hidden information
(opponent hand / deck) implicitly, with no determinisation or explicit belief
state. The Phase-5d play head was memoryless; this is the faithful replacement.

:class:`RecurrentPolicyValueNet` subclasses :class:`~src.net.model.PolicyValueNet`
and reuses its trunk, shared card embedding, CB (deck-build) head and deck LSTM
*unchanged*. The one architectural addition is a **play LSTM** inserted between the
trunk and the value/policy heads: at battle decision ``t`` the trunk turns the
current observation into a per-step embedding ``e_t``, the play LSTM carries it,
and the value + policy heads read the LSTM hidden ``h_t`` (which summarises every
prior decision this game) instead of the raw trunk output.

Serving is **stateful**: :class:`~src.agents.recurrent_agent.RecurrentNetAgent`
holds ``(h, c)`` across ``act`` calls and zeroes them at game start (``reset``).
Deck building at init is unchanged (the deck LSTM, not the play LSTM). This is
pure numpy; the torch mirror + sequence forward + parity test live in
:mod:`src.net.recurrent_torch`.

> Deliberate deviation from a *single* episode-long recurrence: the deck is fully
> observed at battle start, so the play LSTM is reset there rather than threaded
> out of the deck-build segment. The hidden-information benefit the paper's LSTM
> buys is in the battle, which this captures; see
> ``docs/research/paper-faithful-rewrite.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.net.encode import STATE_EMBED_SLOTS
from src.net.model import NetConfig, PolicyValueNet
from src.net.nn import he_init, sigmoid

if TYPE_CHECKING:
    from numpy.typing import NDArray

_HEAD_SCALE = 0.01


@dataclass(frozen=True)
class RecurrentNetConfig(NetConfig):
    """:class:`NetConfig` plus the play-LSTM width (paper default 256)."""

    play_lstm_hidden: int = 256


def lstm_cell(  # noqa: PLR0913 - an LSTM cell's weights are irreducibly four tensors
    x: NDArray[np.float64],
    h: NDArray[np.float64],
    c: NDArray[np.float64],
    w_ih: NDArray[np.float64],
    w_hh: NDArray[np.float64],
    b_ih: NDArray[np.float64],
    b_hh: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One ``nn.LSTMCell`` step (gates packed i,f,g,o, torch's layout)."""
    z = x @ w_ih.T + b_ih + h @ w_hh.T + b_hh
    n = h.shape[-1]
    i = sigmoid(z[..., :n])
    f = sigmoid(z[..., n : 2 * n])
    g = np.tanh(z[..., 2 * n : 3 * n])
    o = sigmoid(z[..., 3 * n : 4 * n])
    c2 = f * c + i * g
    return o * np.tanh(c2), c2


class RecurrentPolicyValueNet(PolicyValueNet):
    """Policy/value net with an obs-history play LSTM (stateful serving)."""

    config: RecurrentNetConfig

    @classmethod
    def random(  # type: ignore[override]
        cls,
        rng: np.random.Generator,
        config: RecurrentNetConfig | None = None,
    ) -> RecurrentPolicyValueNet:
        """A freshly random-initialised recurrent net.

        Reuses the base net's trunk / CB / deck-LSTM / embedding init, then sizes
        the value + policy heads off the play-LSTM width and adds the play-LSTM cell.
        """
        cfg = config or RecurrentNetConfig()
        base = PolicyValueNet.random(rng, cfg)
        p = dict(base.params)
        ph = cfg.play_lstm_hidden
        # Heads now read the play-LSTM hidden (h_t), not the trunk output.
        p["value_w"] = rng.standard_normal((ph, 1)) * _HEAD_SCALE
        p["policy_w1"] = he_init(
            rng, ph + cfg.option_dim + cfg.embed_dim, cfg.policy_hidden,
        )
        # Play LSTM cell: input = trunk output (hidden), state = play_lstm_hidden.
        scale = 1.0 / np.sqrt(ph)
        p["play_lstm_w_ih"] = rng.standard_normal((4 * ph, cfg.hidden)) * scale
        p["play_lstm_w_hh"] = rng.standard_normal((4 * ph, ph)) * scale
        p["play_lstm_b_ih"] = np.zeros(4 * ph)
        p["play_lstm_b_hh"] = np.zeros(4 * ph)
        return cls(cfg, p)

    # --- recurrent state ----------------------------------------------------

    def initial_state(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Zero ``(h, c)`` for the play LSTM (call at battle start)."""
        ph = self.config.play_lstm_hidden
        return np.zeros(ph), np.zeros(ph)

    def play_lstm_step(
        self,
        e: NDArray[np.float64],
        h: NDArray[np.float64],
        c: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Advance the play LSTM by one decision's trunk embedding ``e``."""
        p = self.params
        return lstm_cell(
            e, h, c,
            p["play_lstm_w_ih"], p["play_lstm_w_hh"],
            p["play_lstm_b_ih"], p["play_lstm_b_hh"],
        )

    def trunk_embed(
        self,
        x: NDArray[np.float64],
        rows: NDArray[np.intp],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Per-step observation embedding ``e_t`` (the play-LSTM input)."""
        return self.trunk(self.augment_state(x, rows, mask))[0]

    # --- heads off the recurrent hidden state -------------------------------

    def value_from_h(self, h: NDArray[np.float64]) -> float:
        """Scalar value in ``[-1, 1]`` from the play-LSTM hidden state."""
        v = np.tanh(h @ self.params["value_w"] + self.params["value_b"])
        return float(v.reshape(-1)[0])

    def policy_logits_from_h(
        self,
        h: NDArray[np.float64],
        option_feats: NDArray[np.float64],
        option_rows: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        """One logit per presented option, conditioned on the play-LSTM hidden ``h``."""
        k = option_feats.shape[0]
        if k == 0:
            return np.zeros(0, dtype=np.float64)
        opt_emb = self.params["cb_embed"][option_rows]
        joint = np.concatenate([np.tile(h, (k, 1)), option_feats, opt_emb], axis=1)
        z = np.maximum(0.0, joint @ self.params["policy_w1"] + self.params["policy_b1"])
        return (z @ self.params["policy_w2"] + self.params["policy_b2"]).reshape(-1)

    def step(  # noqa: PLR0913 - one decision threads state + obs + options
        self,
        x: NDArray[np.float64],
        rows: NDArray[np.intp],
        mask: NDArray[np.bool_],
        option_feats: NDArray[np.float64],
        option_rows: NDArray[np.intp],
        h: NDArray[np.float64],
        c: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64]]:
        """One battle decision: returns ``(logits, value, h', c')``.

        Advances the play LSTM with this observation, then scores the options and
        the position value off the new hidden state.
        """
        e = self.trunk_embed(x, rows, mask)
        h2, c2 = self.play_lstm_step(e, h, c)
        logits = self.policy_logits_from_h(h2, option_feats, option_rows)
        return logits, self.value_from_h(h2), h2, c2

    # --- persistence (recover the play-LSTM width too) ----------------------

    @classmethod
    def load(  # type: ignore[override]
        cls,
        path: str | Path,
        config: RecurrentNetConfig | None = None,
    ) -> RecurrentPolicyValueNet:
        """Load a recurrent net, recovering every width from the saved shapes."""
        with np.load(Path(path)) as data:
            params = {k: np.asarray(data[k], dtype=np.float64) for k in data.files}
        if "play_lstm_w_ih" not in params:
            msg = "not a recurrent net (no play_lstm_w_ih); use PolicyValueNet.load"
            raise ValueError(msg)
        if config is None:
            embed_dim = int(params["cb_embed"].shape[1])
            hidden = int(params["trunk_w1"].shape[1])
            ph = int(params["play_lstm_w_hh"].shape[1])
            config = RecurrentNetConfig(
                state_dim=(
                    int(params["trunk_w1"].shape[0]) - STATE_EMBED_SLOTS * embed_dim
                ),
                option_dim=int(params["policy_w1"].shape[0]) - ph - embed_dim,
                hidden=hidden,
                policy_hidden=int(params["policy_w1"].shape[1]),
                cb_hidden=int(params["cb_w1"].shape[1]),
                embed_dim=embed_dim,
                n_cards=int(params["cb_embed"].shape[0]) - 1,
                lstm_hidden=int(params["lstm_w_hh"].shape[1]),
                play_lstm_hidden=ph,
            )
        return cls(config, params)
