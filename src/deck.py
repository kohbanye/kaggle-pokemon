"""Deck-construction legality + legal-deck generation (PLAN.md Phase 1).

The deck is *learned*, not fixed (the OSFP CB head emits 60 card ids at init), so
we need (a) a validator for the Pokemon TCG deck-construction rules and (b) a
per-step legality mask so a card-by-card generator / CB head can *only* produce
legal 60-card decks -- an illegal deck means the engine refuses to start the
battle, i.e. an instant loss.

The pure rule functions here (:func:`legality_errors`, :func:`legal_next_ids`,
:func:`random_legal_deck`) need neither the ``cg`` engine nor pandas, so they
unit-test natively. Only :func:`build_pool` reads the card CSVs (via
``src.cards``). The *exact* rules the engine enforces are confirmed separately by
``scripts/probe_deck_legality.py`` (Docker); this module encodes the standard
Standard-format rules and is the single place to adjust once the probe lands.

Rules encoded:
- exactly 60 cards;
- at most 4 cards sharing the same *name* (Basic Energy is exempt);
- at least 1 Basic Pokemon;
- at most 1 ACE SPEC card in total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.cards import load_cards

if TYPE_CHECKING:
    import random
    from pathlib import Path

DECK_SIZE = 60
MAX_COPIES_BY_NAME = 4
MAX_ACE_SPEC = 1
MIN_BASIC_POKEMON = 1

# Raw values of the cards CSV "stage_or_type" column we key legality off.
_BASIC_POKEMON_STAGE = "Basic Pokémon"
_BASIC_ENERGY_STAGE = "Basic Energy"


@dataclass(frozen=True)
class CardInfo:
    """Minimal per-card facts needed for deck-construction legality."""

    card_id: int
    name: str
    supertype: str
    stage_or_type: str
    is_basic_pokemon: bool
    is_basic_energy: bool
    is_ace_spec: bool


@dataclass(frozen=True)
class CardPool:
    """Lookup of legal-to-build cards, keyed by card id."""

    cards: dict[int, CardInfo]

    def __contains__(self, card_id: int) -> bool:
        return card_id in self.cards

    def ids(self) -> list[int]:
        return list(self.cards)


def build_pool(lang: str = "EN", data_dir: Path | None = None) -> CardPool:
    """Build the card pool from the competition card CSVs (needs pandas/data)."""
    df = load_cards(lang=lang, data_dir=data_dir)
    cards: dict[int, CardInfo] = {}
    for row in df.to_dict(orient="records"):
        stage_val = row["stage_or_type"]
        stage = stage_val if isinstance(stage_val, str) else ""
        card_id = int(row["card_id"])
        cards[card_id] = CardInfo(
            card_id=card_id,
            name=str(row["name"]),
            supertype=str(row["supertype"]),
            stage_or_type=stage,
            is_basic_pokemon=stage == _BASIC_POKEMON_STAGE,
            is_basic_energy=stage == _BASIC_ENERGY_STAGE,
            is_ace_spec=bool(row["is_ace_spec"]),
        )
    return CardPool(cards)


def _nonenergy_name_counts(deck: list[int], pool: CardPool) -> dict[str, int]:
    """Count copies per name, skipping Basic Energy (exempt from the 4-copy cap)."""
    counts: dict[str, int] = {}
    for card_id in deck:
        info = pool.cards.get(card_id)
        if info is None or info.is_basic_energy:
            continue
        counts[info.name] = counts.get(info.name, 0) + 1
    return counts


def legality_errors(deck: list[int], pool: CardPool) -> list[str]:
    """Return a list of rule violations (empty list == legal deck)."""
    errors: list[str] = []

    if len(deck) != DECK_SIZE:
        errors.append(f"deck has {len(deck)} cards, must be {DECK_SIZE}")

    unknown = sorted({c for c in deck if c not in pool.cards})
    if unknown:
        errors.append(f"unknown card ids: {unknown}")

    known = [c for c in deck if c in pool.cards]

    over = sorted(
        name
        for name, n in _nonenergy_name_counts(known, pool).items()
        if n > MAX_COPIES_BY_NAME
    )
    if over:
        errors.append(f"more than {MAX_COPIES_BY_NAME} copies by name: {over}")

    n_ace = sum(pool.cards[c].is_ace_spec for c in known)
    if n_ace > MAX_ACE_SPEC:
        errors.append(f"{n_ace} ACE SPEC cards, max {MAX_ACE_SPEC}")

    n_basic = sum(pool.cards[c].is_basic_pokemon for c in known)
    if n_basic < MIN_BASIC_POKEMON:
        errors.append(f"{n_basic} Basic Pokemon, need >= {MIN_BASIC_POKEMON}")

    return errors


def is_legal(deck: list[int], pool: CardPool) -> bool:
    """True iff ``deck`` satisfies every encoded deck-construction rule."""
    return not legality_errors(deck, pool)


def legal_next_ids(partial: list[int], pool: CardPool) -> set[int]:
    """Card ids appendable to ``partial`` keeping a legal 60-card completion possible.

    Enforces the local caps (copies-by-name, single ACE SPEC, 60-card limit) and
    the one global constraint that can paint us into a corner: if only the final
    slot is left and no Basic Pokemon has been picked yet, restrict to Basics.
    """
    slots_left = DECK_SIZE - len(partial)
    if slots_left <= 0:
        return set()

    has_ace = any(pool.cards[c].is_ace_spec for c in partial if c in pool.cards)
    has_basic = any(
        pool.cards[c].is_basic_pokemon for c in partial if c in pool.cards
    )
    name_counts = _nonenergy_name_counts(partial, pool)
    must_be_basic = slots_left == 1 and not has_basic

    legal: set[int] = set()
    for card_id, info in pool.cards.items():
        if must_be_basic and not info.is_basic_pokemon:
            continue
        if info.is_ace_spec and has_ace:
            continue
        if (
            not info.is_basic_energy
            and name_counts.get(info.name, 0) >= MAX_COPIES_BY_NAME
        ):
            continue
        legal.add(card_id)
    return legal


def random_legal_deck(pool: CardPool, rng: random.Random) -> list[int]:
    """Sample one legal 60-card deck from the per-step legal mask.

    Proves the mask is sound and seeds the deck-eval opponent pool. The result is
    legal but *not* tuned for strength -- strength is what the learned CB head and
    the decklist demonstrations provide later.
    """
    basics = [c for c, info in pool.cards.items() if info.is_basic_pokemon]
    if not basics:
        msg = "pool has no Basic Pokemon; cannot build a legal deck"
        raise ValueError(msg)

    # Seed a Basic Pokemon up front so the must-be-basic corner never binds.
    deck: list[int] = [rng.choice(basics)]
    while len(deck) < DECK_SIZE:
        deck.append(rng.choice(sorted(legal_next_ids(deck, pool))))
    return deck


def load_deck_csv(path: Path) -> list[int]:
    """Read a whitespace/newline-separated deck.csv into a list of card ids."""
    return [int(x) for x in path.read_text().split() if x.strip()]
