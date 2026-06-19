"""Tests for the pure selection-legality helpers (no engine needed)."""

from src.agents.base import is_legal, legal_fallback


def _select(n_options: int, min_count: int, max_count: int) -> dict:
    return {
        "option": [{"type": 14} for _ in range(n_options)],
        "minCount": min_count,
        "maxCount": max_count,
    }


def test_legal_fallback_picks_first_max_count() -> None:
    assert legal_fallback(_select(5, 1, 1)) == [0]
    assert legal_fallback(_select(5, 0, 3)) == [0, 1, 2]
    assert legal_fallback(_select(5, 0, 0)) == []


def test_legal_fallback_is_always_legal() -> None:
    sel = _select(4, 2, 3)
    assert is_legal(legal_fallback(sel), sel)


def test_is_legal_accepts_valid_choice() -> None:
    assert is_legal([0], _select(3, 1, 1))
    assert is_legal([0, 2], _select(3, 1, 2))


def test_is_legal_rejects_wrong_count() -> None:
    assert not is_legal([], _select(3, 1, 1))
    assert not is_legal([0, 1], _select(3, 1, 1))


def test_is_legal_rejects_out_of_range_and_dupes() -> None:
    assert not is_legal([3], _select(3, 1, 1))
    assert not is_legal([-1], _select(3, 1, 1))
    assert not is_legal([0, 0], _select(3, 2, 2))


def test_is_legal_rejects_non_list() -> None:
    assert not is_legal(None, _select(3, 1, 1))
    assert not is_legal("0", _select(3, 1, 1))
    assert not is_legal([0.0], _select(3, 1, 1))
