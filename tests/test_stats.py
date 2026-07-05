"""Tests for src.eval.stats (Wilson interval + WinRate)."""

import math

import pytest

from src.eval.stats import WinRate, wilson_interval


def test_wilson_no_trials_is_maximally_uncertain() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_half_is_centered_on_half() -> None:
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high
    # Symmetric around 0.5 for a 50% observation.
    assert math.isclose((low + high) / 2, 0.5, abs_tol=1e-9)


def test_wilson_is_clamped_to_unit_interval() -> None:
    low, high = wilson_interval(10, 10)
    assert low >= 0.0
    assert high <= 1.0


def test_wilson_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="wins <= trials"):
        wilson_interval(11, 10)


def test_winrate_rate_ignores_ties() -> None:
    wr = WinRate(wins=6, losses=4, ties=10)
    assert wr.games == 20
    assert wr.decisive == 10
    assert wr.rate == 0.6


def test_winrate_rate_nan_without_decisive_games() -> None:
    assert math.isnan(WinRate(ties=5).rate)


def test_winrate_significance_far_from_half() -> None:
    # 90/100 wins: interval clearly above 0.5.
    assert WinRate(wins=90, losses=10).significant


def test_winrate_not_significant_near_half() -> None:
    # 11/20 wins: interval straddles 0.5.
    assert not WinRate(wins=11, losses=9).significant


def test_winrate_addition_sums_fields() -> None:
    total = WinRate(1, 2, 3) + WinRate(4, 5, 6)
    assert total == WinRate(5, 7, 9)
