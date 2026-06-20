"""Tests for human-readable decklist <-> card-id conversion (src.decklists)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.deck import build_pool, load_deck_csv
from src.decklists import (
    decklist_to_ids,
    name_to_ids,
    parse_decklist,
    save_deck_csv,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
needs_data = pytest.mark.skipif(
    not (DATA_DIR / "EN_Card_Data.csv").exists(),
    reason="competition data not downloaded",
)


def test_parse_decklist_skips_blanks_and_comments() -> None:
    text = "4 Pikachu ex\n# a comment\n\n2 Boss's Orders\n"
    assert parse_decklist(text) == [(4, "Pikachu ex"), (2, "Boss's Orders")]


def test_parse_decklist_rejects_bad_line() -> None:
    with pytest.raises(ValueError, match="cannot parse"):
        parse_decklist("not a deck line")


def test_decklist_to_ids_expands_counts() -> None:
    assert decklist_to_ids([(2, "A"), (1, "B")], {"A": 10, "B": 20}) == [10, 10, 20]


def test_decklist_to_ids_reports_missing() -> None:
    with pytest.raises(ValueError, match="unknown card names"):
        decklist_to_ids([(1, "Nope")], {"A": 10})


def test_save_and_load_deck_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "deck.csv"
    save_deck_csv([1, 2, 3], path)
    assert load_deck_csv(path) == [1, 2, 3]


@needs_data
def test_name_to_ids_resolves_known_card() -> None:
    # "Pikachu ex" has multiple printings; resolution picks one and it must
    # round-trip back to the same name.
    pid = name_to_ids()["Pikachu ex"]
    assert build_pool().cards[pid].name == "Pikachu ex"
