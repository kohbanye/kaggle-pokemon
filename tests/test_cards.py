"""Tests for the pure parsing helpers in src.cards (no data files needed)."""

from src.cards import cost_total, parse_cost, parse_damage, supertype


def test_parse_cost_mixed_colors_and_colorless() -> None:
    assert parse_cost("{P}{D}{M}●") == {"P": 1, "D": 1, "M": 1, "C": 1}


def test_parse_cost_only_colorless() -> None:
    assert parse_cost("●●●") == {"C": 3}


def test_parse_cost_empty_or_missing() -> None:
    assert parse_cost(None) == {}
    assert parse_cost("n/a") == {}


def test_cost_total() -> None:
    assert cost_total("{R}{R}●") == 3
    assert cost_total(None) == 0


def test_parse_damage_strips_modifiers() -> None:
    assert parse_damage("30×") == 30
    assert parse_damage("120") == 120
    assert parse_damage(None) is None


def test_supertype_mapping() -> None:
    assert supertype("Basic Pokémon") == "Pokemon"
    assert supertype("Supporter") == "Trainer"
    assert supertype("Special Energy") == "Energy"
    assert supertype("???") == "Unknown"
