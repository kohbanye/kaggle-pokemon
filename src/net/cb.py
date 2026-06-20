"""Deck-construction (CB) head: build a legal 60-card deck card-by-card.

At init the engine asks for a 60-card deck and an illegal one is an instant loss,
so the CB head must emit *only* legal decks. It does this by scoring every card
once with the (context-free) CB head, then filling the 60 slots one at a time,
each step restricted to :func:`src.deck.legal_next_ids` -- the per-step legal mask
that enforces the copies-by-name cap, the single-ACE-SPEC rule and "at least one
Basic Pokemon". Greedy picks the highest-scoring legal card (deterministic);
sampling draws from the masked softmax (a mixed deck strategy, matching the
plan's "deck is a distribution, not fixed").

With random weights this yields a legal-but-untuned deck -- exactly the Phase-3
bar ("CB head always produces a deck the engine accepts"). Strength comes later
from BC warm-start (Phase 4) and OSFP (Phase 5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.deck import DECK_SIZE, legal_next_ids
from src.net.nn import softmax

if TYPE_CHECKING:
    from src.deck import CardPool
    from src.net.features import CardFeatures
    from src.net.model import PolicyValueNet


def card_scores(
    net: PolicyValueNet,
    pool: CardPool,
    feats: CardFeatures,
) -> dict[int, float]:
    """One CB logit per pool card (context-free: computed once for all slots)."""
    ids = pool.ids()
    if not ids:
        return {}
    matrix = np.stack([feats.vector(cid) for cid in ids])
    logits = net.card_logits(matrix)
    return {cid: float(logit) for cid, logit in zip(ids, logits, strict=True)}


def build_deck(
    net: PolicyValueNet,
    pool: CardPool,
    feats: CardFeatures,
    rng: np.random.Generator | None = None,
    *,
    greedy: bool = True,
) -> list[int]:
    """Build one legal 60-card deck from the CB scores under the legal mask.

    ``greedy`` (default) takes the highest-scoring legal card at each step --
    deterministic, for reproducible evaluation. Otherwise samples from the masked
    softmax using ``rng`` (which must be provided).
    """
    if not greedy and rng is None:
        msg = "sampling deck construction requires an rng"
        raise ValueError(msg)
    if rng is None:  # greedy path: rng is unused, but bind it for a clean type
        rng = np.random.default_rng()
    scores = card_scores(net, pool, feats)

    deck: list[int] = []
    while len(deck) < DECK_SIZE:
        legal = legal_next_ids(deck, pool)
        if not legal:  # the mask should always leave a legal completion
            break
        candidates = sorted(legal)
        if greedy:
            pick = max(candidates, key=lambda c: scores.get(c, 0.0))
        else:
            logits = np.asarray([scores.get(c, 0.0) for c in candidates])
            probs = softmax(logits)
            pick = candidates[int(rng.choice(len(candidates), p=probs))]
        deck.append(pick)
    return deck
