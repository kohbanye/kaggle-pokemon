"""Win-rate statistics for the evaluation harness.

Pure functions — no simulator or data files needed. The harness compares two
agents/decks by their head-to-head win rate and asks whether that rate differs
*significantly* from 50%, using a Wilson score interval (more honest than the
normal approximation at the sample sizes and extreme rates we hit). This is the
"is it real?" gate from PLAN.md §B: a difference counts only when the 95%
interval does not straddle 0.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Standard-normal quantile for a two-sided 95% interval (z_{0.975}).
Z_95 = 1.959963984540054

# The null "coin flip" win rate every comparison is measured against.
EVEN_ODDS = 0.5


def wilson_interval(wins: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``wins / trials``.

    Returns ``(low, high)`` clamped to ``[0, 1]``. With no trials the interval
    is maximally uncertain, ``(0.0, 1.0)``.
    """
    if trials < 0 or wins < 0 or wins > trials:
        msg = f"need 0 <= wins <= trials, got wins={wins}, trials={trials}"
        raise ValueError(msg)
    if trials == 0:
        return (0.0, 1.0)
    p = wins / trials
    n = trials
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True)
class WinRate:
    """Tally of one entrant's results in a head-to-head comparison."""

    wins: int = 0
    losses: int = 0
    ties: int = 0

    @property
    def decisive(self) -> int:
        """Games with a winner (ties carry no signal on who is stronger)."""
        return self.wins + self.losses

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def rate(self) -> float:
        """Win rate over decisive games; ``nan`` when there are none."""
        return self.wins / self.decisive if self.decisive else float("nan")

    @property
    def interval(self) -> tuple[float, float]:
        """Wilson 95% interval for :attr:`rate` (over decisive games)."""
        return wilson_interval(self.wins, self.decisive)

    @property
    def significant(self) -> bool:
        """True when the 95% interval does not contain 0.5 (a real edge)."""
        low, high = self.interval
        return low > EVEN_ODDS or high < EVEN_ODDS

    def __add__(self, other: WinRate) -> WinRate:
        return WinRate(
            self.wins + other.wins,
            self.losses + other.losses,
            self.ties + other.ties,
        )
