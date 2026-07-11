"""Observation-consistent opponent-deck belief for determinization.

PIMC's only real approximation is the opponent's hidden cards. With a *correct* deck
guess it decisively beats the raw net; with a blind one it does not. We can't see the
opponent's list, but we see the cards they reveal (active / bench / discard / logs), so
we keep a **belief over a hypothesis set of archetype decks** (the meta decklists + QD
archive elites) and weight each by how well it contains what we've seen. Early, when
little is revealed, the belief is near-uniform; as distinctive cards appear it sharpens
onto the matching archetype. Each determinization draws a deck from this belief, so the
PIMC average also integrates over *which deck* the opponent is on.

Pure ``dict`` / ``list[int]`` in, engine-free, unit-tested.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from src.search.determinize import seen_card_ids


def seen_opponent_ids(current: dict, your_index: int) -> list[int]:
    """Opponent CardData ids we can see (board + discard + face-up prize; not hand)."""
    opp = current["players"][1 - your_index]
    return seen_card_ids(opp, include_hand=False)


def consistency_scores(
    candidates: list[list[int]],
    seen_opp: list[int],
) -> np.ndarray:
    """Multiset overlap of each candidate deck with the seen opponent cards.

    Score = how many of the seen cards the candidate can account for (capped per card
    by the candidate's own copies). A deck that contains everything seen scores
    ``len(seen_opp)``; one missing a revealed card is penalised by exactly the cards
    it cannot explain -- what discriminates archetypes once distinctive cards show.
    """
    seen = Counter(seen_opp)
    out = np.empty(len(candidates), dtype=float)
    for i, deck in enumerate(candidates):
        dc = Counter(deck)
        out[i] = sum(min(n, dc.get(cid, 0)) for cid, n in seen.items())
    return out


def sample_consistent_deck(
    candidates: list[list[int]],
    seen_opp: list[int],
    rng: np.random.Generator,
    *,
    sharpness: float = 6.0,
) -> list[int]:
    """Draw one archetype deck weighted by consistency with the observation.

    With nothing seen the draw is uniform; otherwise it is ``softmax(sharpness * frac)``
    over the fraction of seen cards each candidate explains, so a single distinctive
    revealed card tilts the belief without ever zeroing an off-meta hypothesis.
    """
    if not candidates:
        return []
    if not seen_opp:
        return list(candidates[int(rng.integers(len(candidates)))])
    frac = consistency_scores(candidates, seen_opp) / len(seen_opp)
    weights = np.exp(sharpness * (frac - frac.max()))
    weights /= weights.sum()
    return list(candidates[int(rng.choice(len(candidates), p=weights))])
