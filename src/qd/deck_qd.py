"""Deck operators for MAP-Elites: random legal deck, mutation, archetype descriptor.

Pure (no engine): every deck is a 60-card list that passes
:func:`~src.deck.legality_errors` by construction -- both the random generator and
the mutator build/repair through :func:`~src.deck.legal_next_ids`, so the QD search
only ever proposes **legal** decks (the engine's only hard rule). Playability beyond
legality is left to fitness, not hand-coded constraints.

The **behaviour descriptor** is the deck's archetype niche -- ``(prize-liability bin,
setup-speed bin)``, the two genuine *trade-offs* that define the modern Mega-era meta:

- **prize liability** (attrition ↔ power): single-prize attackers give the opponent
  fewer prizes per KO and win the prize race, but ex (2 prizes) / Mega ex (3 prizes)
  hit harder and set up faster. Measured as "extra prize points" the deck can give up
  (ex = +1, Mega = +2 per copy); bin 0 is a pure single-prize deck.
- **setup speed** (aggro ↔ ramp): the cheapest attack the deck can field -- a
  1-energy attacker spams from turn 1, a 3-4 energy attacker needs a setup turn.

Both are derived from the *decklist alone* (no engine). The earlier ``(colour,
energy-count)`` descriptor failed to illuminate the space -- colour duplicated the
soft colour penalty (and pulled the archive toward the rainbow decks the penalty
discourages) while every strong deck piled into the top energy bin -- so the archive
degenerated to "best mono/duo deck per colour". Colour stays as the soft fitness
*penalty* (see :func:`colour_count`), not a niche axis. MAP-Elites keeps the best deck
per niche.
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

# Prize-liability niche edges over "extra prize points" (ex=+1, Mega=+2 per copy):
# bin 0 == pure single-prize deck, rising to all-Mega. 5 bins.
PRIZE_BIN_EDGES = (0, 4, 8, 12)
# Setup-speed niche edges over the deck's cheapest attack cost: <=1 (aggro) .. >=4
# (ramp). 4 bins; a deck with no attacker at all falls into the slowest bin.
SPEED_BIN_EDGES = (1, 2, 3)
_EX_PRIZES = 2
_MEGA_PRIZES = 3


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


def prize_value(info: object | None) -> int:
    """Prizes a single Pokemon gives up when KO'd: Mega ex 3, ex 2, other Pokemon 1.

    Non-Pokemon (energy / trainers) give up nothing toward the liability.
    """
    if info is None or getattr(info, "supertype", "") != "Pokemon":
        return 0
    if getattr(info, "is_mega", False):
        return _MEGA_PRIZES
    if getattr(info, "is_ex", False):
        return _EX_PRIZES
    return 1


def prize_points(deck: list[int], pool: CardPool) -> int:
    """Total *extra* prizes the deck can give up (ex = +1, Mega = +2 per copy).

    0 for a pure single-prize deck; grows with multi-prize density and severity --
    the attrition ↔ power axis of the prize race.
    """
    return sum(max(prize_value(pool.cards.get(cid)) - 1, 0) for cid in deck)


def prize_bin(points: int) -> int:
    """Bin prize points into ``0..len(PRIZE_BIN_EDGES)`` (0 == pure single-prize)."""
    return sum(points > edge for edge in PRIZE_BIN_EDGES)


def setup_cost(deck: list[int], pool: CardPool) -> int | None:
    """Cheapest attack the deck can field (min attack cost over its Pokemon).

    ``None`` if no card in the deck has an attack (treated as the slowest niche).
    """
    costs = [
        info.min_attack_cost
        for cid in deck
        if (info := pool.cards.get(cid)) is not None
        and info.min_attack_cost is not None
    ]
    return min(costs) if costs else None


def speed_bin(cost: int | None) -> int:
    """Bin the cheapest attack cost into ``0..len(SPEED_BIN_EDGES)`` (0 = aggro)."""
    if cost is None:
        return len(SPEED_BIN_EDGES)  # no attacker -> slowest niche
    return sum(cost > edge for edge in SPEED_BIN_EDGES)


def behaviour_descriptor(deck: list[int], pool: CardPool) -> tuple[int, int]:
    """Archetype niche of a deck: ``(prize-liability bin, setup-speed bin)``."""
    return prize_bin(prize_points(deck, pool)), speed_bin(setup_cost(deck, pool))


def deck_stats(deck: list[int], pool: CardPool) -> dict[str, int | None]:
    """Coarse composition + niche features (for logging / inspection)."""
    kinds = Counter(card_kind(pool, cid) for cid in deck)
    return {
        "energy": kinds.get("energy", 0),
        "pokemon": kinds.get("pokemon", 0),
        "trainer": kinds.get("trainer", 0),
        "distinct": len(set(deck)),
        "prize_points": prize_points(deck, pool),
        "min_attack_cost": setup_cost(deck, pool),
    }
