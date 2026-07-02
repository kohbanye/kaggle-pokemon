"""Co-evolving opponent gauntlet for MAP-Elites (Step 4, GAME-style).

Pure (no engine): the pieces that turn the fixed meta gauntlet into an *evolving*
opponent pool. A fixed opponent set stops raising the bar once the archive beats it
(Miernik & Kowalski: fixed weak opponents degenerate); instead each round's gauntlet
mixes

- the current archive's **top-k elites** (one per niche, so the exploiters stay
  diverse -- the one-sided-archive hybrid from the redesign doc), and
- a sample from a **hall of fame** of past strong opponents (the meta decks it was
  seeded with plus each round's best), which keeps old exploiters in play and damps
  the rock-paper-scissors cycling coevolution is prone to.

The archive itself is warm-started across rounds by the caller (decks carry over,
fitness is re-scored against the new gauntlet -- fitness is gauntlet-relative, so
stale scores are not comparable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from src.qd.archive import MapElitesArchive


@dataclass
class HofEntry:
    """One hall-of-fame opponent: its deck and a human-readable provenance tag."""

    deck: list[int]
    tag: str


class HallOfFame:
    """FIFO-capped pool of past strong opponent decks (anti-cycling memory)."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.entries: list[HofEntry] = []

    def add(self, deck: list[int], tag: str) -> None:
        """Append an opponent; evict the oldest entry beyond the cap."""
        self.entries.append(HofEntry(list(deck), tag))
        if len(self.entries) > self.size:
            self.entries.pop(0)

    def sample(self, n: int, rng: np.random.Generator) -> list[HofEntry]:
        """Uniform sample without replacement, in random order (all if ``n`` >= len).

        Random *order* matters to the caller: :func:`build_gauntlet` requests more
        entries than it has slots and fills until the size cap, so the order decides
        which HoF opponents make the cut each round.
        """
        idx = rng.permutation(len(self.entries))[:n]
        return [self.entries[int(i)] for i in idx]


def build_gauntlet(
    arc: MapElitesArchive,
    hof: HallOfFame,
    size: int,
    top_k: int,
    rng: np.random.Generator,
) -> tuple[list[list[int]], list[str]]:
    """Next round's opponents: top-k archive elites + a hall-of-fame sample.

    Elites come first (one per niche, best ``key`` first) so the freshest, most
    diverse exploiters always face the candidates; the remaining slots are drawn
    from the hall of fame. Duplicate decklists are skipped (a round's best elite is
    usually also the newest HoF entry) so the gauntlet never wastes evaluation
    games on the same opponent twice.
    """
    decks: list[list[int]] = []
    tags: list[str] = []
    seen: set[tuple[int, ...]] = set()

    def push(deck: list[int], tag: str) -> None:
        fp = tuple(sorted(deck))
        if fp not in seen and len(decks) < size:
            seen.add(fp)
            decks.append(list(deck))
            tags.append(tag)

    for e in arc.elites()[:top_k]:
        push(e.deck, f"elite:{e.descriptor}")
    # oversample the HoF request: duplicates of the elites (or of each other) are
    # dropped by push(), so ask for everything and let the size cap cut it off
    for entry in hof.sample(len(hof.entries), rng):
        push(entry.deck, f"hof:{entry.tag}")
    return decks, tags
