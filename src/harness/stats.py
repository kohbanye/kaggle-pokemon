"""Win-rate aggregation with Wilson score confidence intervals.

The plan's keep/drop rule is "win rate + Wilson 95% CI; significant iff the CI
does not straddle 50%". This module computes that summary from a list of
:class:`~src.harness.result.GameResult`. Pure -- no engine, unit-tested.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .result import GameResult

Z95 = 1.959963984540054  # standard normal quantile for a 95% two-sided interval
EVEN = 0.50  # the break-even line a CI must clear to be significant
ADOPT_BAR = 0.55  # plan's "clearly adopt" threshold (section B)


def wilson_interval(wins: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """Return ``(point_estimate, low, high)`` for a binomial proportion.

    ``point_estimate`` is the raw ``wins / n``; ``low``/``high`` are the Wilson
    score interval bounds. For ``n == 0`` everything is ``0.0``.
    """
    if n <= 0:
        return (0.0, 0.0, 0.0)
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (p, centre - half, centre + half)


def _verdict(low: float, high: float) -> str:
    """Classify a CI against the 50% line (plan section B thresholds)."""
    if low > ADOPT_BAR:
        return "adopt"  # clearly above the 55% bar
    if low > EVEN:
        return "lean-adopt"  # significant edge but inside 50-55%
    if high < EVEN:
        return "reject"  # significantly worse
    return "inconclusive"  # CI straddles 50%


def summarize(
    results: Sequence[GameResult],
    name_a: str = "A",
    name_b: str = "B",
) -> dict:
    """Aggregate games into a JSON-ready summary (win rate, CI, timing)."""
    n = len(results)
    a_wins = sum(r.a_won for r in results)
    b_wins = sum(r.b_won for r in results)
    draws = sum(r.is_draw for r in results)
    aborted = sum(r.is_aborted for r in results)
    decisive = a_wins + b_wins

    point, low, high = wilson_interval(a_wins, decisive)

    total_moves_a = sum(r.moves_a for r in results) or 1
    total_moves_b = sum(r.moves_b for r in results) or 1
    total_wall = sum(r.wall_s for r in results)

    return {
        "agent_a": name_a,
        "agent_b": name_b,
        "games": n,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "aborted": aborted,
        "decisive": decisive,
        "a_winrate": point,
        "a_winrate_ci95": [low, high],
        "verdict": _verdict(low, high),
        "significant": low > EVEN or high < EVEN,
        "avg_turns": (sum(r.turns for r in results) / n) if n else 0.0,
        "avg_selections": (sum(r.selections for r in results) / n) if n else 0.0,
        "avg_move_ms_a": 1000 * sum(r.agent_time_a for r in results) / total_moves_a,
        "avg_move_ms_b": 1000 * sum(r.agent_time_b for r in results) / total_moves_b,
        "max_move_ms_a": 1000 * max((r.max_move_a for r in results), default=0.0),
        "max_move_ms_b": 1000 * max((r.max_move_b for r in results), default=0.0),
        "avg_game_wall_s": (total_wall / n) if n else 0.0,
        "games_per_sec": (n / total_wall) if total_wall > 0 else 0.0,
    }
