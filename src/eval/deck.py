"""Deck loading and legality checks for the evaluation harness.

A deck is a list of 60 card IDs (the format the agent returns on the initial
selection). ``deck.csv`` in the sample submission is whitespace-separated IDs.
The card-aware rules (basic energy exempt from the 4-copy cap, at most one ACE
SPEC) need the card pool, so the relevant ID sets are passed in by the caller —
this keeps the checker pure and testable without the gitignored card data.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection

DECK_SIZE = 60
MAX_COPIES = 4
MAX_ACE_SPEC = 1


def load_deck(path: str | Path) -> list[int]:
    """Parse a whitespace-separated list of card IDs (e.g. ``deck.csv``)."""
    text = Path(path).read_text(encoding="utf-8")
    return [int(token) for token in text.split() if token.strip()]


def validate_deck(
    deck: Collection[int],
    *,
    basic_energy_ids: Collection[int] = (),
    ace_spec_ids: Collection[int] = (),
) -> list[str]:
    """Return a list of rule violations; an empty list means the deck is legal.

    Checks: exactly :data:`DECK_SIZE` cards; at most :data:`MAX_COPIES` of any
    card except basic energy; at most :data:`MAX_ACE_SPEC` ACE SPEC card. When
    ``basic_energy_ids`` / ``ace_spec_ids`` are omitted those exemptions/limits
    are simply not applied (structural size/copy checks still run).
    """
    cards = list(deck)
    basics = set(basic_energy_ids)
    aces = set(ace_spec_ids)
    problems: list[str] = []

    if len(cards) != DECK_SIZE:
        problems.append(f"deck has {len(cards)} cards, must be exactly {DECK_SIZE}")

    counts = Counter(cards)
    for card_id, count in sorted(counts.items()):
        if count > MAX_COPIES and card_id not in basics:
            problems.append(
                f"card {card_id} appears {count} times, max {MAX_COPIES} "
                f"(basic energy exempt)",
            )

    ace_total = sum(count for card_id, count in counts.items() if card_id in aces)
    if ace_total > MAX_ACE_SPEC:
        problems.append(f"{ace_total} ACE SPEC cards, max {MAX_ACE_SPEC}")

    return problems
