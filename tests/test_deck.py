"""Tests for deck-construction legality + generation (src.deck).

The rule tests run on a tiny synthetic pool (no data needed). A few sanity tests
exercise the real card pool and the sample deck, and skip when the gitignored
competition data has not been downloaded.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from src.deck import (
    DECK_SIZE,
    CardInfo,
    CardPool,
    build_pool,
    is_legal,
    legal_next_ids,
    legality_errors,
    load_deck_csv,
    random_legal_deck,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_DECK = DATA_DIR / "sample_submission" / "deck.csv"
needs_data = pytest.mark.skipif(
    not SAMPLE_DECK.exists(), reason="competition data not downloaded",
)

# Synthetic pool: a couple of Basics, a Stage 1, an item, an ACE SPEC, an energy.
PIKA, BULB, RAICHU, BALL, BELT, WENERGY = 1, 2, 3, 4, 5, 6


def _info(
    card_id: int,
    name: str,
    *,
    basic_pokemon: bool = False,
    basic_energy: bool = False,
    ace_spec: bool = False,
) -> CardInfo:
    return CardInfo(
        card_id=card_id,
        name=name,
        supertype="",
        stage_or_type="",
        is_basic_pokemon=basic_pokemon,
        is_basic_energy=basic_energy,
        is_ace_spec=ace_spec,
    )


def _pool() -> CardPool:
    return CardPool(
        {
            PIKA: _info(PIKA, "Pikachu", basic_pokemon=True),
            BULB: _info(BULB, "Bulbasaur", basic_pokemon=True),
            RAICHU: _info(RAICHU, "Raichu"),
            BALL: _info(BALL, "Poke Ball"),
            BELT: _info(BELT, "Maximum Belt", ace_spec=True),
            WENERGY: _info(WENERGY, "Basic Water Energy", basic_energy=True),
        },
    )


def _fill(cards: list[int], pad: int = WENERGY) -> list[int]:
    """Pad a partial deck up to DECK_SIZE with a (cap-exempt) basic energy."""
    return cards + [pad] * (DECK_SIZE - len(cards))


def test_legal_deck_is_legal() -> None:
    deck = _fill([PIKA, RAICHU, BALL])
    assert legality_errors(deck, _pool()) == []
    assert is_legal(deck, _pool())


def test_wrong_size_is_illegal() -> None:
    short = [PIKA, *([WENERGY] * 58)]  # 59 cards
    assert any("must be 60" in e for e in legality_errors(short, _pool()))


def test_more_than_four_by_name_is_illegal() -> None:
    deck = _fill([PIKA, *([BALL] * 5)])
    errs = legality_errors(deck, _pool())
    assert any("more than 4 copies" in e for e in errs)


def test_basic_energy_exempt_from_copy_cap() -> None:
    # 59 copies of a basic energy + 1 basic pokemon is legal.
    deck = _fill([PIKA])
    assert is_legal(deck, _pool())


def test_two_ace_spec_is_illegal() -> None:
    deck = _fill([PIKA, BELT, BELT])
    assert any("ACE SPEC" in e for e in legality_errors(deck, _pool()))


def test_no_basic_pokemon_is_illegal() -> None:
    deck = _fill([RAICHU, BALL])
    assert any("Basic Pokemon" in e for e in legality_errors(deck, _pool()))


def test_unknown_card_id_is_illegal() -> None:
    deck = _fill([PIKA, 99999])
    assert any("unknown card ids" in e for e in legality_errors(deck, _pool()))


def test_legal_next_ids_blocks_capped_name_keeps_energy() -> None:
    partial = [PIKA, *([BALL] * 4)]
    nxt = legal_next_ids(partial, _pool())
    assert BALL not in nxt  # name is at the 4-copy cap
    assert WENERGY in nxt  # basic energy is never capped


def test_legal_next_ids_blocks_second_ace_spec() -> None:
    assert BELT not in legal_next_ids([PIKA, BELT], _pool())
    assert BELT in legal_next_ids([PIKA], _pool())


def test_legal_next_ids_forces_basic_on_last_slot() -> None:
    partial = [RAICHU, *([WENERGY] * 58)]  # 59 cards, no Basic Pokemon yet
    nxt = legal_next_ids(partial, _pool())
    assert nxt == {PIKA, BULB}  # only Basics keep a legal completion possible


def test_legal_next_ids_empty_when_full() -> None:
    assert legal_next_ids(_fill([PIKA]), _pool()) == set()


def test_random_legal_deck_is_always_legal() -> None:
    pool = _pool()
    rng = random.Random(0)  # noqa: S311 - gameplay randomness, not crypto
    for _ in range(200):
        deck = random_legal_deck(pool, rng)
        assert len(deck) == DECK_SIZE
        assert is_legal(deck, pool)


@needs_data
def test_build_pool_classifies_known_cards() -> None:
    pool = build_pool()
    assert pool.cards[3].is_basic_energy  # Basic {W} Energy
    assert pool.cards[1158].is_ace_spec  # Maximum Belt
    assert pool.cards[721].is_basic_pokemon  # Kyogre


@needs_data
def test_sample_deck_is_legal() -> None:
    pool = build_pool()
    deck = load_deck_csv(SAMPLE_DECK)
    assert len(deck) == DECK_SIZE
    assert legality_errors(deck, pool) == []


@needs_data
def test_random_legal_deck_over_real_pool() -> None:
    pool = build_pool()
    rng = random.Random(1)  # noqa: S311 - gameplay randomness, not crypto
    for _ in range(20):
        assert is_legal(random_legal_deck(pool, rng), pool)
