"""Deck operators for MAP-Elites: random legal deck, mutation, archetype descriptor.

Pure (no engine): every deck is a 60-card list that passes
:func:`~src.deck.legality_errors` by construction -- both the random generator and
the mutator build/repair through :func:`~src.deck.legal_next_ids`, so the QD search
only ever proposes **legal** decks (the engine's only hard rule). Playability beyond
legality is left to fitness, not hand-coded constraints.

The **behaviour descriptor** is the deck's archetype niche -- ``(primary colour,
energy-count bin)``: colour spreads coverage across the type wheel, the energy bin
across aggro (few energy) ↔ setup (many). MAP-Elites keeps the best deck per niche.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from src.deck import DECK_SIZE, card_kind, legal_next_ids

if TYPE_CHECKING:
    import numpy as np

    from src.deck import CardPool

# Energy-count niche edges (upper-inclusive): aggro/thin .. energy-heavy/setup.
ENERGY_BIN_EDGES = (8, 12, 16, 20)
NO_COLOUR = "C"  # colourless / typeless fallback descriptor


def random_legal_deck(pool: CardPool, rng: np.random.Generator) -> list[int]:
    """Build a uniformly-random **legal** 60-card deck (random legal pick per slot)."""
    deck: list[int] = []
    while len(deck) < DECK_SIZE:
        legal = sorted(legal_next_ids(deck, pool))
        if not legal:
            break
        deck.append(legal[int(rng.integers(len(legal)))])
    return deck


def mutate(
    deck: list[int],
    pool: CardPool,
    rng: np.random.Generator,
    n_swaps: int = 3,
) -> list[int]:
    """Swap ~``n_swaps`` cards and repair to a legal 60: a local archive variation.

    Removes ``n_swaps`` random cards (the remaining ≤60 multiset is still a legal
    prefix -- it extended to legal before) then re-fills to 60 through
    :func:`~src.deck.legal_next_ids`, so the result is always legal and differs from
    the parent by at most ~``n_swaps`` cards.
    """
    keep = list(deck)
    for _ in range(min(n_swaps, len(keep))):
        keep.pop(int(rng.integers(len(keep))))
    while len(keep) < DECK_SIZE:
        legal = sorted(legal_next_ids(keep, pool))
        if not legal:
            break
        keep.append(legal[int(rng.integers(len(legal)))])
    return keep


def primary_colour(deck: list[int], pool: CardPool) -> str:
    """The deck's dominant Pokemon energy colour (its archetype colour)."""
    colours: Counter[str] = Counter()
    for cid in deck:
        info = pool.cards.get(cid)
        if info is not None and info.supertype == "Pokemon" and info.card_type:
            colours[info.card_type] += 1
    colours.pop(NO_COLOUR, None)  # colourless is not an archetype identity
    return colours.most_common(1)[0][0] if colours else NO_COLOUR


def colour_count(deck: list[int], pool: CardPool) -> int:
    """Distinct *coloured* Pokemon types in the deck (its rainbow-ness).

    Colourless (``NO_COLOUR``) is excluded: colourless attackers run on any energy,
    so they do not demand an extra colour. Used by the QD soft colour penalty to bias
    the archive toward fewer-colour (more consistent) decks without forbidding any.
    """
    colours: set[str] = set()
    for cid in deck:
        info = pool.cards.get(cid)
        if info is not None and info.supertype == "Pokemon" and info.card_type:
            colours.add(info.card_type)
    colours.discard(NO_COLOUR)
    return len(colours)


def energy_count(deck: list[int], pool: CardPool) -> int:
    """Number of Energy cards in the deck (Basic + Special)."""
    return sum(card_kind(pool, cid) == "energy" for cid in deck)


def energy_bin(n: int) -> int:
    """Bin an energy count into a niche index ``0..len(ENERGY_BIN_EDGES)``."""
    return sum(n > edge for edge in ENERGY_BIN_EDGES)


def behaviour_descriptor(deck: list[int], pool: CardPool) -> tuple[str, int]:
    """Archetype niche of a deck: ``(primary colour, energy-count bin)``."""
    return primary_colour(deck, pool), energy_bin(energy_count(deck, pool))


def deck_stats(deck: list[int], pool: CardPool) -> dict[str, int]:
    """Coarse composition (for logging / inspection)."""
    kinds = Counter(card_kind(pool, cid) for cid in deck)
    return {
        "energy": kinds.get("energy", 0),
        "pokemon": kinds.get("pokemon", 0),
        "trainer": kinds.get("trainer", 0),
        "distinct": len(set(deck)),
    }
