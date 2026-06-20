"""Tests for the demonstration-deck builder (src.deckbuild).

Structure tests use a synthetic pool; the all-legal check uses the real pool and
skips without the gitignored competition data.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.deck import DECK_SIZE, CardInfo, CardPool, build_pool, is_legal
from src.deckbuild import (
    COLORLESS_FILLER,
    COPIES,
    ENGINE,
    build_demo_decks,
    build_mono_aggro,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
needs_data = pytest.mark.skipif(
    not (DATA_DIR / "EN_Card_Data.csv").exists(),
    reason="competition data not downloaded",
)

ENGINE_TOTAL = sum(ENGINE.values())


def _ci(card_id: int) -> CardInfo:
    return CardInfo(card_id, f"c{card_id}", "", "", False, False, False)  # noqa: FBT003


def test_build_mono_aggro_structure() -> None:
    pool = CardPool({i: _ci(i) for i in [101, 102, 103, *ENGINE]})
    deck = build_mono_aggro([101, 102, 103], 200, pool)
    counts = Counter(deck)
    assert len(deck) == DECK_SIZE
    assert counts[101] == counts[102] == counts[103] == COPIES
    assert COLORLESS_FILLER not in counts  # 12 pokemon already >= MIN_POKEMON
    assert counts[200] == DECK_SIZE - 3 * COPIES - ENGINE_TOTAL


def test_build_mono_aggro_adds_filler_when_thin() -> None:
    pool = CardPool({i: _ci(i) for i in [101, COLORLESS_FILLER, *ENGINE]})
    deck = build_mono_aggro([101], 200, pool)
    counts = Counter(deck)
    assert counts[101] == COPIES
    assert counts[COLORLESS_FILLER] == COPIES  # only 4 pokemon otherwise


@needs_data
def test_build_demo_decks_all_legal() -> None:
    pool = build_pool()
    decks = build_demo_decks()
    assert len(decks) >= 6
    for deck in decks.values():
        assert len(deck) == DECK_SIZE
        assert is_legal(deck, pool)
