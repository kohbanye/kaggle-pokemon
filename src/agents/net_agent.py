"""NetAgent -- the policy/value net wired into the Kaggle agent contract.

A pure ``dict -> list[int]`` policy (like every agent here) that wraps
:class:`~src.net.model.PolicyValueNet`: it encodes the observation, scores the
presented options with the policy head and returns the top option index/indices.
Because the head scores the *presented* options, the choice is always a legal,
in-range selection, so the contract round-trips for free. Engine card/attack
stats are injected (``engine=``) and turned into the net's fixed card features;
nothing here imports ``cg``.

Deck construction: when a card pool is supplied (``cb_pool=``) the CB head builds
the 60-card deck at construction (the learned init behaviour); otherwise the deck
passed in is used (clean agent-vs-agent comparison on a fixed deck). Every
decision is wrapped so any error falls back to a guaranteed-legal selection --
the agent must never crash a match (plan SS D).

Action selection is greedy (argmax) by default -- the strongest single move, used
for evaluation and submission. With ``temperature > 0`` a single-select decision
instead samples from the temperature-scaled policy softmax (``reset(seed)`` seeds
the per-game RNG): this is the stochastic behaviour the Phase-5 OSFP self-play
collector needs for exploration. Multi-select stays deterministic top-``maxCount``
(only single-select decisions are cloned / policy-gradient trained).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np

from src.net.cb import build_deck
from src.net.encode import encode_options, encode_state
from src.net.features import CardFeatures
from src.net.model import NetConfig, PolicyValueNet
from src.net.nn import softmax

from .base import Agent, legal_fallback

if TYPE_CHECKING:
    from pathlib import Path

    from src.deck import CardPool

_DEFAULT_SEED = 0


class NetAgent(Agent):
    """Policy/value-net agent; scores presented options, never crashes."""

    name = "net"

    def __init__(  # noqa: PLR0913 - a policy ctor legitimately threads net sources
        self,
        deck: list[int],
        engine: dict | None = None,
        net: PolicyValueNet | None = None,
        *,
        weights: str | Path | None = None,
        seed: int = _DEFAULT_SEED,
        cb_pool: CardPool | None = None,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(deck)
        self.feats = CardFeatures(engine)
        self.temperature = float(temperature)
        self._rng = np.random.default_rng(seed)
        if net is not None:
            self.net = net
        elif weights is not None:
            self.net = PolicyValueNet.load(weights)
        else:
            # A random net used for deck building needs its card embedding sized to
            # the pool, else the CB head can't score it (Phase 5b).
            cfg = NetConfig(n_cards=len(cb_pool.ids())) if cb_pool else NetConfig()
            self.net = PolicyValueNet.random(np.random.default_rng(seed), cfg)
        # CB head builds the deck at init when a pool is available; on any error
        # keep the deck passed in (a guaranteed-legal fallback).
        if cb_pool is not None:
            with contextlib.suppress(Exception):
                self.deck = build_deck(self.net, cb_pool, self.feats)

    def reset(self, seed: int) -> None:
        """Re-seed the per-game sampling RNG (used only when ``temperature > 0``)."""
        self._rng = np.random.default_rng(seed)

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
            return None  # nothing to choose -> legal fallback (possibly empty)

        current = obs.get("current") or {}
        your_index = int(current.get("yourIndex", 0))
        state_vec = encode_state(current, your_index, self.feats)
        option_feats = encode_options(options, current, your_index, self.feats)
        logits = self.net.policy_logits(state_vec, option_feats)
        if logits.shape[0] != len(options):
            return None

        # Stochastic single-select for self-play exploration: sample one option
        # from the temperature-scaled softmax. The taken index is what the OSFP
        # collector logs and the policy-gradient trainer learns from.
        if max_count == 1 and self.temperature > 0.0:
            probs = softmax(logits / self.temperature)
            return [int(self._rng.choice(len(probs), p=probs))]

        # Otherwise take the top-scoring options. Picking exactly maxCount distinct
        # indices mirrors the universal legal_fallback (range(maxCount)) -- always
        # legal -- and reduces to a single argmax when maxCount == 1.
        order = np.argsort(logits)[::-1]
        return [int(i) for i in order[:max_count]]
