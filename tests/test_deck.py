"""Tests for src.eval.deck (loading + legality checks)."""

from pathlib import Path

from src.eval.deck import load_deck, validate_deck


def test_load_deck_parses_whitespace_ids(tmp_path: Path) -> None:
    path = tmp_path / "deck.csv"
    path.write_text("1 2 3\n4\t5\n")
    assert load_deck(path) == [1, 2, 3, 4, 5]


def test_valid_deck_has_no_problems() -> None:
    deck = list(range(60))  # 60 unique cards
    assert validate_deck(deck) == []


def test_wrong_size_is_flagged() -> None:
    problems = validate_deck([1, 2, 3])
    assert any("must be exactly 60" in p for p in problems)


def test_five_copies_flagged_unless_basic_energy() -> None:
    deck = [7] * 5 + list(range(100, 155))  # 60 cards, five of card 7
    assert any("card 7 appears 5 times" in p for p in validate_deck(deck))
    # Exempt when card 7 is a basic energy.
    assert validate_deck(deck, basic_energy_ids={7}) == []


def test_two_ace_spec_flagged() -> None:
    deck = [100, 101, *range(58)]  # 60 cards
    problems = validate_deck(deck, ace_spec_ids={100, 101})
    assert any("ACE SPEC" in p for p in problems)
