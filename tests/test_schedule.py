"""Tests for src.eval.schedule (paired, side-swapped round robin)."""

import pytest

from src.eval.schedule import round_robin


def test_round_robin_counts_and_pairs() -> None:
    matches = round_robin(["A", "B", "C"], games_per_pair=4)
    # 3 unordered pairs x 4 games each.
    assert len(matches) == 12


def test_round_robin_swaps_sides_within_a_pair() -> None:
    matches = round_robin(["A", "B"], games_per_pair=2)
    first, second = matches
    # Same seed, sides swapped.
    assert first.seed == second.seed
    assert first.pair_id == second.pair_id
    assert (first.player0, first.player1) == ("A", "B")
    assert (second.player0, second.player1) == ("B", "A")


def test_round_robin_seeds_distinct_across_pairs() -> None:
    matches = round_robin(["A", "B", "C"], games_per_pair=2)
    seeds = {m.seed for m in matches}
    # One shared seed per unordered pair -> 3 distinct seeds.
    assert len(seeds) == 3


def test_round_robin_balances_who_plays_first() -> None:
    matches = round_robin(["A", "B"], games_per_pair=6)
    as_player0 = sum(1 for m in matches if m.player0 == "A")
    assert as_player0 == 3


def test_round_robin_rejects_odd_games() -> None:
    with pytest.raises(ValueError, match="even"):
        round_robin(["A", "B"], games_per_pair=3)


def test_round_robin_rejects_zero_games() -> None:
    with pytest.raises(ValueError, match="even"):
        round_robin(["A", "B"], games_per_pair=0)
