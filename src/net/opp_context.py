"""Observation-derived OPPONENT-deck belief context for the play net.

The opponent's decklist is hidden, but the cards they reveal (active / bench / discard /
face-up prizes) identify their archetype. We keep a FIXED hypothesis set of archetype
decks and, from what the opponent has shown, form a belief-weighted expected opponent
``deck_context`` (mean card features) -- the mirror of :func:`deck_context` for OUR
deck. Early (little shown) it is ~the average meta deck; as distinctive cards appear the
belief sharpens onto the matching archetype (softmax over multiset overlap, reusing
:mod:`src.search.opp_belief`).

Concatenated to the net input, this is the ONLY deck-adaptivity possible under hidden
information: a single policy that reads "who am I probably facing" from the board and
adjusts, rather than an impossible per-decklist specialist. The hypothesis set MUST be
identical at train and serve, so it is built deterministically from fixed decklist dirs.

Pure numpy + file reads; engine-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.net.encode import deck_context
from src.search.opp_belief import consistency_scores, seen_opponent_ids

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.net.features import CardFeatures

ROOT = Path(__file__).resolve().parent.parent.parent
# Archetype hypothesis set: meta aggro/QD decks + real-ladder anchors + QD candidates.
_HYP_DIRS = ("decklists", "decklists/anchors", "decklists/candidates")
_SHARPNESS = 6.0
# The hypothesis set is deterministic and constant per process (fixed decklists + one
# engine's card features); caching it is essential -- QMlpAgent builds one per
# construction (each with its own feats) and QD/collection make thousands.
_HYP_CACHE: list[tuple[list[list[int]], NDArray[np.float64]]] = []


def build_hypotheses(
    feats: CardFeatures,
) -> tuple[list[list[int]], NDArray[np.float64]]:
    """(hypothesis decks, their deck_context matrix) -- built once, shared train+serve.

    Deterministic: sorted files across the fixed dirs, de-duplicated by card multiset.
    Process-cached: the set is constant for a fixed engine + decklists, so every
    QMlpAgent (even with its own ``feats``) reuses the first build.
    """
    if _HYP_CACHE:
        return _HYP_CACHE[0]
    decks: list[list[int]] = []
    ctxs: list[NDArray[np.float64]] = []
    seen: set[tuple[int, ...]] = set()
    for d in _HYP_DIRS:
        for p in sorted((ROOT / d).glob("*.csv")):
            deck = [int(x) for x in p.read_text().split() if x.strip()]
            key = tuple(sorted(deck))
            if not deck or key in seen:
                continue
            seen.add(key)
            decks.append(deck)
            ctxs.append(deck_context(deck, feats))
    result = (decks, np.asarray(ctxs))
    _HYP_CACHE.append(result)
    return result


def opp_context(
    current: dict,
    your_index: int,
    decks: list[list[int]],
    ctxs: NDArray[np.float64],
    *,
    sharpness: float = _SHARPNESS,
) -> NDArray[np.float64]:
    """Belief-weighted expected opponent deck_context from what they've revealed.

    Nothing seen -> the mean hypothesis context (~average meta). Otherwise softmax over
    the fraction of seen cards each hypothesis explains, weighting their contexts.
    """
    seen = seen_opponent_ids(current, your_index)
    if not seen or len(ctxs) == 0:
        return ctxs.mean(axis=0)
    frac = consistency_scores(decks, seen) / len(seen)
    w = np.exp(sharpness * (frac - frac.max()))
    w /= w.sum()
    return w @ ctxs
