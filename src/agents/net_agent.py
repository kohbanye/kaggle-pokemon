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
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np

from src.net.cb import build_deck
from src.net.encode import encode_options, encode_state
from src.net.features import CardFeatures
from src.net.model import PolicyValueNet

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
    ) -> None:
        super().__init__(deck)
        self.feats = CardFeatures(engine)
        if net is not None:
            self.net = net
        elif weights is not None:
            self.net = PolicyValueNet.load(weights)
        else:
            self.net = PolicyValueNet.random(np.random.default_rng(seed))
        # CB head builds the deck at init when a pool is available; on any error
        # keep the deck passed in (a guaranteed-legal fallback).
        if cb_pool is not None:
            with contextlib.suppress(Exception):
                self.deck = build_deck(self.net, cb_pool, self.feats)

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

        # Take the top-scoring options. Picking exactly maxCount distinct indices
        # mirrors the universal legal_fallback (range(maxCount)) -- always legal --
        # and reduces to a single argmax when maxCount == 1.
        order = np.argsort(logits)[::-1]
        return [int(i) for i in order[:max_count]]
