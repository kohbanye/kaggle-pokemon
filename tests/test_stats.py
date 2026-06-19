"""Tests for the Wilson CI + aggregation harness (no engine needed)."""

import math

from src.harness.result import GameResult
from src.harness.stats import summarize, wilson_interval


def _game(*, a_won: bool = False, draw: bool = False) -> GameResult:
    winner = 2 if draw else (0 if a_won else 1)
    return GameResult(
        a_is_player0=True, winner=winner, turns=12, selections=60,
        agent_time_a=0.02, agent_time_b=0.02, moves_a=30, moves_b=30,
        max_move_a=0.004, max_move_b=0.003, wall_s=0.2,
    )


def _games(a_wins: int, b_wins: int = 0, draws: int = 0) -> list[GameResult]:
    return (
        [_game(a_won=True) for _ in range(a_wins)]
        + [_game(a_won=False) for _ in range(b_wins)]
        + [_game(draw=True) for _ in range(draws)]
    )


def test_wilson_half_split_known_values() -> None:
    point, low, high = wilson_interval(50, 100)
    assert point == 0.5
    assert math.isclose(low, 0.4038, abs_tol=1e-3)
    assert math.isclose(high, 0.5962, abs_tol=1e-3)


def test_wilson_empty() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0, 0.0)


def test_wilson_interval_brackets_point() -> None:
    point, low, high = wilson_interval(600, 1000)
    assert low < point < high
    assert math.isclose(point, 0.6)


def test_summarize_attribution_and_adopt() -> None:
    results = _games(600, 400)
    s = summarize(results, "greedy", "random")
    assert s["a_wins"] == 600
    assert s["b_wins"] == 400
    assert s["decisive"] == 1000
    assert s["draws"] == 0
    assert math.isclose(s["a_winrate"], 0.6)
    assert s["significant"] is True
    assert s["verdict"] == "adopt"


def test_summarize_draws_excluded_from_decisive() -> None:
    results = _games(8, draws=2)
    s = summarize(results)
    assert s["games"] == 10
    assert s["draws"] == 2
    assert s["decisive"] == 8
    assert s["a_wins"] == 8


def test_summarize_inconclusive_when_ci_straddles_half() -> None:
    results = _games(50, 50)
    s = summarize(results)
    assert s["verdict"] == "inconclusive"
    assert s["significant"] is False


def test_summarize_reject_when_clearly_below_half() -> None:
    results = _games(400, 600)
    s = summarize(results)
    assert s["verdict"] == "reject"
    assert s["significant"] is True
