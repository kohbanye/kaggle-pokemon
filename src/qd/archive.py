"""MAP-Elites archive: best deck per archetype niche.

A pure container (no engine, no torch): a behaviour descriptor (the niche key)
maps to the highest-fitness deck found for that niche. Inserting a deck replaces
the niche's occupant only if it is fitter, so the archive monotonically improves
each cell while *covering* the descriptor space -- diversity is structural, it can
never collapse to one archetype the way a single gradient-trained head does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Hashable

    import numpy as np


@dataclass
class Elite:
    """One niche's incumbent: its deck, fitness, descriptor and free-form metadata."""

    deck: list[int]
    fitness: float
    descriptor: Hashable
    meta: dict = field(default_factory=dict)


class MapElitesArchive:
    """Niche -> best :class:`Elite`. Insert keeps the fitter occupant per niche."""

    def __init__(self) -> None:
        self.cells: dict[Hashable, Elite] = {}

    def insert(
        self,
        deck: list[int],
        fitness: float,
        descriptor: Hashable,
        meta: dict | None = None,
    ) -> bool:
        """Place ``deck`` in its niche if empty or fitter than the incumbent.

        Returns whether it was admitted (a new niche filled or an improvement).
        """
        cur = self.cells.get(descriptor)
        if cur is None or fitness > cur.fitness:
            self.cells[descriptor] = Elite(list(deck), float(fitness), descriptor,
                                           meta or {})
            return True
        return False

    def sample(self, rng: np.random.Generator) -> Elite:
        """Draw a uniformly-random incumbent (the parent for a variation step)."""
        elites = list(self.cells.values())
        return elites[int(rng.integers(len(elites)))]

    def elites(self) -> list[Elite]:
        """All incumbents, fittest first."""
        return sorted(self.cells.values(), key=lambda e: e.fitness, reverse=True)

    def best(self) -> Elite | None:
        """The single fittest incumbent (the submission/deck candidate), if any."""
        return max(self.cells.values(), key=lambda e: e.fitness, default=None)

    @property
    def coverage(self) -> int:
        """Number of filled niches."""
        return len(self.cells)

    def mean_fitness(self) -> float:
        """Mean fitness over filled niches (the QD-score proxy)."""
        if not self.cells:
            return 0.0
        return sum(e.fitness for e in self.cells.values()) / len(self.cells)
