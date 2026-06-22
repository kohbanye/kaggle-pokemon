"""RecurrentNetAgent -- the recurrent policy/value net as a Kaggle agent.

Stateful sibling of :class:`~src.agents.net_agent.NetAgent` for the paper-faithful
recurrent net (:class:`~src.net.recurrent_model.RecurrentPolicyValueNet`): it
carries the play LSTM's ``(h, c)`` across decisions and zeroes them at game start
(:meth:`reset`), so the value/policy heads see the whole observation history.

The play LSTM advances **only on single-select decisions** (the ones the V-Trace
trajectory records); multi-select sub-choices reuse the current hidden state and
are taken deterministically top-``maxCount`` -- so the serving recurrence matches
what the learner trains on. Each single-select decision exposes ``last_logp`` (the
log-probability of the option it returned), which the self-play collector logs as
the behaviour log-prob ``μ(a|s)``. Deck construction at init samples from the CB
head and stores the per-pick log-probs (:func:`~src.net.deck_sample`).

Like every agent here it never crashes: any error falls back to a legal selection.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np

from src.net.deck_sample import sample_deck_with_logp
from src.net.embedding import CardEmbeddingIndex
from src.net.encode import (
    encode_options,
    encode_state,
    option_embed_rows,
    state_embed_rows,
)
from src.net.features import CardFeatures
from src.net.nn import softmax

from .base import Agent, legal_fallback

if TYPE_CHECKING:
    from src.deck import CardPool
    from src.net.recurrent_model import RecurrentPolicyValueNet

_DEFAULT_SEED = 0
SINGLE_SELECT = 1


class RecurrentNetAgent(Agent):
    """Stateful recurrent policy/value agent; scores options, never crashes."""

    name = "recurrent"

    def __init__(  # noqa: PLR0913 - a policy ctor legitimately threads net sources
        self,
        deck: list[int],
        engine: dict | None,
        net: RecurrentPolicyValueNet,
        *,
        seed: int = _DEFAULT_SEED,
        cb_pool: CardPool | None = None,
        sample_deck: bool = False,
        build_deck_from_net: bool = True,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(deck)
        self.net = net
        self.feats = CardFeatures(engine)
        self.temperature = float(temperature)
        self._rng = np.random.default_rng(seed)
        self._index = CardEmbeddingIndex(cb_pool) if cb_pool is not None else None
        self._h, self._c = net.initial_state()
        self.last_logp = 0.0
        # Per-pick deck log-probs (the deck arm's behaviour log-probs), set when the
        # deck is sampled from the CB head; empty for a fixed (passed-in) deck.
        self.deck_logp: list[float] = []
        if cb_pool is not None and build_deck_from_net and sample_deck:
            with contextlib.suppress(Exception):
                self.deck, self.deck_logp = sample_deck_with_logp(
                    net, cb_pool, self.feats, self._rng,
                )

    def reset(self, seed: int) -> None:
        """Re-seed sampling and zero the play LSTM (call at game start)."""
        self._rng = np.random.default_rng(seed)
        self._h, self._c = self.net.initial_state()

    def act(self, obs: dict) -> list[int]:
        select = obs.get("select") or {}
        try:
            choice = self._decide(obs, select)
        except Exception:  # noqa: BLE001 - submission hygiene: never crash a match
            choice = None
        if choice is not None:
            return choice
        return legal_fallback(select)

    def _decide(self, obs: dict, select: dict) -> list[int] | None:
        options = select.get("option") or []
        max_count = int(select.get("maxCount", 0))
        if not options or max_count < 1:
            return None

        current = obs.get("current") or {}
        your_index = int(current.get("yourIndex", 0))
        state_vec = encode_state(current, your_index, self.feats)
        option_feats = encode_options(options, current, your_index, self.feats)
        rows, mask = state_embed_rows(current, your_index, self._index)
        option_rows = option_embed_rows(options, current, your_index, self._index)

        if max_count == SINGLE_SELECT:
            # Advance the play LSTM and score off the new hidden state.
            logits, _value, self._h, self._c = self.net.step(
                state_vec, rows, mask, option_feats, option_rows, self._h, self._c,
            )
            if logits.shape[0] != len(options):
                return None
            return self._single_select(logits)

        # Multi-select: reuse the current hidden state (no LSTM advance), take the
        # top-maxCount options deterministically (mirrors the trajectory's exclusion).
        logits = self.net.policy_logits_from_h(self._h, option_feats, option_rows)
        if logits.shape[0] != len(options):
            return None
        order = np.argsort(logits)[::-1]
        return [int(i) for i in order[:max_count]]

    def _single_select(self, logits: np.ndarray) -> list[int]:
        """Sample (``temperature > 0``) or argmax one option; record ``last_logp``."""
        if self.temperature > 0.0:
            probs = softmax(logits / self.temperature)
            idx = int(self._rng.choice(len(probs), p=probs))
        else:
            probs = softmax(logits)
            idx = int(np.argmax(logits))
        self.last_logp = float(np.log(probs[idx] + 1e-12))
        return [idx]
