"""OpponentBelief posterior: uniform before reveals, sharpens onto the archetype.

Engine-free -- a synthetic ``current`` (only the opponent's revealed board/discard
matters for the posterior). Validates the belief the ReBeL root chance node uses.
"""

from __future__ import annotations

import numpy as np

from src.rebel.belief import OpponentBelief

# Three archetypes sharing filler ids 1..5, each with a DISTINCTIVE id (10/20/30).
_HYPS = [
    [10, 10, *([1, 2, 3, 4, 5] * 11), 1, 2, 3],   # archetype A (distinctive 10)
    [20, 20, *([1, 2, 3, 4, 5] * 11), 1, 2, 3],   # archetype B (distinctive 20)
    [30, 30, *([1, 2, 3, 4, 5] * 11), 1, 2, 3],   # archetype C (distinctive 30)
]


def _current(opp_discard_ids: list[int]) -> dict:
    """Minimal State dict: only players[1] (opponent, from your_index=0) is read."""
    opp = {"active": [], "bench": [], "discard": [{"id": c} for c in opp_discard_ids],
           "prize": [None] * 6, "hand": None, "handCount": 5, "deckCount": 40}
    me = {"active": [], "bench": [], "discard": [], "prize": [None] * 6,
          "hand": [], "handCount": 5, "deckCount": 40}
    return {"players": [me, opp], "yourIndex": 0}


def test_posterior_uniform_when_nothing_seen() -> None:
    b = OpponentBelief(_HYPS)
    p = b.posterior(_current([]), your_index=0)
    assert np.allclose(p, 1.0 / 3)


def test_posterior_sharpens_onto_matching_archetype() -> None:
    b = OpponentBelief(_HYPS)
    # opponent has discarded their distinctive card 20 -> archetype B (index 1).
    p = b.posterior(_current([20]), your_index=0)
    assert p.argmax() == 1
    assert p[1] > 0.5
    # more copies revealed -> sharper.
    p2 = b.posterior(_current([20, 20]), your_index=0)
    assert p2[1] >= p[1]


def test_entropy_drops_after_a_distinctive_reveal() -> None:
    b = OpponentBelief(_HYPS)
    assert b.entropy(_current([20]), 0) < b.entropy(_current([]), 0)


def test_sample_deck_returns_a_hypothesis() -> None:
    b = OpponentBelief(_HYPS)
    rng = np.random.default_rng(0)
    deck = b.sample_deck(_current([30]), 0, rng)
    assert deck in _HYPS
    # heavily favours archetype C after seeing its distinctive card.
    picks = [b.sample_deck(_current([30]), 0, rng) is _HYPS[2] for _ in range(50)]
    assert sum(picks) > 40
