"""Round-robin scheduling with paired, side-swapped seeds.

Variance reduction (PLAN.md §B): compare entrants on the *same* determinization
seed with 先後 (who sits as engine player 0 / moves first) swapped, so a lucky
shuffle helps both sides equally. Games therefore come in pairs that share a
``seed``; the two halves of a pair differ only in which entrant is player 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class Match:
    """One scheduled game.

    ``player0`` sits as engine player index 0 (moves first under ``seed``).
    Games sharing a ``pair_id`` are the two side-swapped halves of a paired
    comparison and use the same ``seed``.
    """

    player0: str
    player1: str
    seed: int
    pair_id: int


def round_robin(
    entrants: Sequence[str],
    games_per_pair: int,
    base_seed: int = 0,
) -> list[Match]:
    """All-play-all schedule; every unordered pair plays ``games_per_pair`` games.

    ``games_per_pair`` must be a positive even number so the side swap is
    balanced (half the games with each entrant on the play). Seeds are shared
    within a side-swapped pair and are distinct across pairs.
    """
    if games_per_pair <= 0 or games_per_pair % 2 != 0:
        msg = f"games_per_pair must be a positive even number, got {games_per_pair}"
        raise ValueError(msg)

    matches: list[Match] = []
    seed = base_seed
    for first, second in combinations(entrants, 2):
        for _ in range(games_per_pair // 2):
            matches.append(Match(first, second, seed, pair_id=seed))
            matches.append(Match(second, first, seed, pair_id=seed))
            seed += 1
    return matches
