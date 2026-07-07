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

from src.net.deck_factored import N_CATEGORIES
from src.net.encode import STATE_EMBED_SLOTS
from src.net.features import CARD_FEAT_DIM
from src.net.model import NetConfig, PolicyValueNet
from src.net.nn import he_init, sigmoid

if TYPE_CHECKING:
    from numpy.typing import NDArray

_HEAD_SCALE = 0.01


@dataclass(frozen=True)
class RecurrentNetConfig(NetConfig):
    """:class:`NetConfig` plus the play-LSTM width (paper default 256).

    ``deck_ctx_dim > 0`` enables the **deck-conditioned** play LSTM: a fixed
    deck-context vector (:func:`~src.net.encode.deck_context`, ``deck_feat_dim``
    wide) is projected to ``deck_ctx_dim`` and concatenated to every LSTM input,
    so one net pilots different archetypes with different policies (the
    observation itself only carries ``deckCount``). 0 = exactly the old net.
    """

    play_lstm_hidden: int = 256
    deck_ctx_dim: int = 0
    deck_feat_dim: int = CARD_FEAT_DIM


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
        # Play LSTM cell: input = trunk output (hidden) [+ deck context], state =
        # play_lstm_hidden.
        scale = 1.0 / np.sqrt(ph)
        in_dim = cfg.hidden + cfg.deck_ctx_dim
        p["play_lstm_w_ih"] = rng.standard_normal((4 * ph, in_dim)) * scale
        p["play_lstm_w_hh"] = rng.standard_normal((4 * ph, ph)) * scale
        p["play_lstm_b_ih"] = np.zeros(4 * ph)
        p["play_lstm_b_hh"] = np.zeros(4 * ph)
        if cfg.deck_ctx_dim > 0:
            p["deck_ctx_w"] = he_init(rng, cfg.deck_feat_dim, cfg.deck_ctx_dim)
        # Deck category head (factored CB: {pokemon, trainer, energy}); reads the
        # deck-LSTM hidden. Near-zero init so picks start ~category-uniform.
        p["cat_w"] = rng.standard_normal((cfg.lstm_hidden, N_CATEGORIES)) * _HEAD_SCALE
        p["cat_b"] = np.zeros(N_CATEGORIES)
        return cls(cfg, p)

    def cat_logits(self, h: NDArray[np.float64]) -> NDArray[np.float64]:
        """Deck category logits ``(N_CATEGORIES,)`` from the deck-LSTM hidden ``h``."""
        return h @ self.params["cat_w"] + self.params["cat_b"]

    # --- recurrent state ----------------------------------------------------

    def initial_state(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Zero ``(h, c)`` for the play LSTM (call at battle start)."""
        ph = self.config.play_lstm_hidden
        return np.zeros(ph), np.zeros(ph)

    # --- deck conditioning ---------------------------------------------------

    def deck_ctx(
        self, deck_vec: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        """Project a deck-context vector for the play LSTM (None if disabled).

        ``deck_vec`` is :func:`~src.net.encode.deck_context` of the agent's own
        60 cards; compute the projection once per game and pass it to every
        :meth:`step`.
        """
        w = self.params.get("deck_ctx_w")
        if w is None:
            return None
        return np.tanh(deck_vec @ w)

    def enable_deck_ctx(
        self, rng: np.random.Generator, ctx_dim: int, feat_dim: int = CARD_FEAT_DIM,
    ) -> RecurrentPolicyValueNet:
        """A copy of this net with deck conditioning added, **behaviour-preserving**.

        The new LSTM input columns are ZERO so the context contributes nothing
        until training moves them -- loading a pre-conditioning checkpoint and
        migrating it changes no output. (A no-op copy if already enabled.)
        """
        if "deck_ctx_w" in self.params:
            return self
        p = dict(self.params)
        w_ih = p["play_lstm_w_ih"]
        p["play_lstm_w_ih"] = np.concatenate(
            [w_ih, np.zeros((w_ih.shape[0], ctx_dim))], axis=1,
        )
        p["deck_ctx_w"] = he_init(rng, feat_dim, ctx_dim)
        cfg = RecurrentNetConfig(
            **{**self.config.__dict__, "deck_ctx_dim": ctx_dim,
               "deck_feat_dim": feat_dim},
        )
        return type(self)(cfg, p)

    def widen_play_lstm(
        self, rng: np.random.Generator, new_hidden: int,
    ) -> RecurrentPolicyValueNet:
        """A copy with a WIDER play LSTM, **behaviour-preserving** (net2net).

        Old units keep their input rows and read the new units through ZERO
        recurrent weights (their dynamics are untouched); new units start with
        small random weights but the value/policy heads read them through ZERO
        rows -- so the widened net's outputs are identical until training moves
        the zeros. Lets a saturated 256-wide checkpoint grow without losing
        anything. (No-op copy if ``new_hidden <= current``.)
        """
        ph, h2 = self.config.play_lstm_hidden, new_hidden
        if h2 <= ph:
            return self
        p = dict(self.params)
        add = h2 - ph
        scale = 1.0 / np.sqrt(h2)
        in_dim = p["play_lstm_w_ih"].shape[1]
        p["play_lstm_w_ih"] = np.concatenate(
            [np.concatenate(
                [p["play_lstm_w_ih"][g * ph:(g + 1) * ph],
                 rng.standard_normal((add, in_dim)) * scale], axis=0)
             for g in range(4)], axis=0)
        whh = p["play_lstm_w_hh"]
        p["play_lstm_w_hh"] = np.concatenate(
            [np.concatenate(
                [np.concatenate([whh[g * ph:(g + 1) * ph],
                                 np.zeros((ph, add))], axis=1),
                 rng.standard_normal((add, h2)) * scale], axis=0)
             for g in range(4)], axis=0)
        for k in ("play_lstm_b_ih", "play_lstm_b_hh"):
            b = p[k]
            p[k] = np.concatenate(
                [np.concatenate([b[g * ph:(g + 1) * ph], np.zeros(add)])
                 for g in range(4)])
        p["value_w"] = np.concatenate([p["value_w"], np.zeros((add, 1))], axis=0)
        pw = p["policy_w1"]  # input rows are [play-hidden | option | embed]
        p["policy_w1"] = np.concatenate(
            [pw[:ph], np.zeros((add, pw.shape[1])), pw[ph:]], axis=0)
        cfg = RecurrentNetConfig(
            **{**self.config.__dict__, "play_lstm_hidden": h2})
        return type(self)(cfg, p)

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
        ctx: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64]]:
        """One battle decision: returns ``(logits, value, h', c')``.

        Advances the play LSTM with this observation (plus the per-game deck
        context ``ctx`` from :meth:`deck_ctx`, when conditioning is enabled),
        then scores the options and the position value off the new hidden state.
        """
        e = self.trunk_embed(x, rows, mask)
        if ctx is not None:
            e = np.concatenate([e, ctx])
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
            ctx = params.get("deck_ctx_w")  # deck conditioning, if trained with it
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
                deck_ctx_dim=int(ctx.shape[1]) if ctx is not None else 0,
                deck_feat_dim=(int(ctx.shape[0]) if ctx is not None
                               else CARD_FEAT_DIM),
            )
        return cls(config, params)
