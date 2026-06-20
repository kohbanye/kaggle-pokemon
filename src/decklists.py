"""Human-readable decklist <-> card-id conversion (PLAN.md Phase 1).

Turns meta/community decklists (written as ``<count> <Card Name>`` lines) into
the 60 card ids the engine wants, resolved against the card pool, and writes
id-lists back to a deck.csv. The 4-copy rule is by *name*, so when a name spans
several printings (154 do) any id works -- we pick the lowest deterministically.

Native (data access only via src.cards); validate the result with src.deck.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.cards import load_cards

if TYPE_CHECKING:
    from pathlib import Path

_LINE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")


def name_to_ids(lang: str = "EN", data_dir: Path | None = None) -> dict[str, int]:
    """Map each card name to a representative (lowest) card id."""
    cards = load_cards(lang=lang, data_dir=data_dir)
    out: dict[str, int] = {}
    for row in cards.to_dict(orient="records"):
        name = str(row["name"])
        card_id = int(row["card_id"])
        if name not in out or card_id < out[name]:
            out[name] = card_id
    return out


def parse_decklist(text: str) -> list[tuple[int, str]]:
    """Parse ``<count> <name>`` lines; blank lines and ``#`` comments are ignored."""
    entries: list[tuple[int, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if match is None:
            msg = f"cannot parse decklist line: {raw!r}"
            raise ValueError(msg)
        entries.append((int(match.group(1)), match.group(2)))
    return entries


def decklist_to_ids(
    entries: list[tuple[int, str]], name_ids: dict[str, int],
) -> list[int]:
    """Expand (count, name) entries into a flat list of card ids."""
    ids: list[int] = []
    missing: list[str] = []
    for count, name in entries:
        card_id = name_ids.get(name)
        if card_id is None:
            missing.append(name)
            continue
        ids.extend([card_id] * count)
    if missing:
        msg = f"unknown card names: {missing}"
        raise ValueError(msg)
    return ids


def load_decklist(path: Path, name_ids: dict[str, int]) -> list[int]:
    """Read a ``<count> <name>`` decklist file into a list of card ids."""
    return decklist_to_ids(parse_decklist(path.read_text()), name_ids)


def save_deck_csv(ids: list[int], path: Path) -> None:
    """Write a card-id list as a one-id-per-line deck.csv."""
    path.write_text("\n".join(str(i) for i in ids) + "\n")
