"""Engine-agnostic match runner and result aggregation.

The runner never touches the compiled ``cg`` engine directly; it drives an
:class:`Engine` that yields either a :class:`Decision` (an agent must choose) or
an :class:`Outcome` (the game ended). This keeps all orchestration — routing
each decision to the correct agent, the step guard, tallying with side-swap —
pure and unit-testable with a fake engine. The real engine is wrapped by an
adapter that lives with the CLI (``scripts/run_tournament.py``), where the
observation-schema assumptions are isolated.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from src.eval.stats import WinRate

if TYPE_CHECKING:
    from src.eval.schedule import Match

# An observation is an opaque dict; an agent maps one to chosen option indices.
Obs = Mapping[str, object]
Agent = Callable[[Obs], list[int]]

# Safety valve so a non-terminating engine/agent can't hang a tournament.
MAX_SELECTIONS = 20_000


@dataclass(frozen=True)
class Decision:
    """The engine is asking ``player`` (0 or 1) to choose, given ``obs``."""

    player: int
    obs: Obs


@dataclass(frozen=True)
class Outcome:
    """The game ended. ``winner`` is 0, 1, or ``None`` for a tie."""

    winner: int | None


@dataclass(frozen=True)
class GameResult:
    """One finished game, recording which entrant sat on each side."""

    player0: str
    player1: str
    winner: int | None


class Engine(Protocol):
    """Minimal driver contract the runner needs; implemented by adapters/fakes."""

    def start(
        self,
        deck0: Sequence[int],
        deck1: Sequence[int],
        seed: int,
    ) -> Decision | Outcome: ...

    def choose(self, choice: list[int]) -> Decision | Outcome: ...

    def finish(self) -> None: ...


def play_match(
    engine: Engine,
    agents: tuple[Agent, Agent],
    decks: tuple[Sequence[int], Sequence[int]],
    seed: int,
) -> int | None:
    """Play one game to completion; return the winning player index (or None).

    ``agents[i]`` plays ``decks[i]`` as engine player ``i``. Raises if the game
    does not terminate within :data:`MAX_SELECTIONS` decisions.
    """
    step = engine.start(decks[0], decks[1], seed)
    for _ in range(MAX_SELECTIONS):
        if isinstance(step, Outcome):
            engine.finish()
            return step.winner
        choice = agents[step.player](step.obs)
        step = engine.choose(choice)
    engine.finish()
    msg = f"game did not terminate within {MAX_SELECTIONS} selections"
    raise RuntimeError(msg)


def run_tournament(
    engine: Engine,
    agent_factory: Callable[[Sequence[int]], Agent],
    decks: Mapping[str, Sequence[int]],
    matches: Sequence[Match],
) -> list[GameResult]:
    """Play every scheduled match with a fixed agent bound to each deck.

    ``agent_factory(deck)`` builds the (deck-fixed) agent used for that side —
    this is Phase 1's "agent fixed, deck varies" setup.
    """
    results: list[GameResult] = []
    for match in matches:
        deck0, deck1 = decks[match.player0], decks[match.player1]
        winner = play_match(
            engine,
            (agent_factory(deck0), agent_factory(deck1)),
            (deck0, deck1),
            match.seed,
        )
        results.append(GameResult(match.player0, match.player1, winner))
    return results


def aggregate(results: Iterable[GameResult]) -> dict[tuple[str, str], WinRate]:
    """Collapse games into a per-pair win rate from the first name's viewpoint.

    Keys are ``(a, b)`` with ``a < b`` lexicographically; the :class:`WinRate`
    is ``a``'s record against ``b``, summed over both side assignments.
    """
    table: dict[tuple[str, str], WinRate] = {}
    for game in results:
        low, high = sorted((game.player0, game.player1))
        key = (low, high)
        if game.winner is None:
            delta = WinRate(ties=1)
        else:
            winner_name = game.player0 if game.winner == 0 else game.player1
            delta = WinRate(wins=1) if winner_name == low else WinRate(losses=1)
        table[key] = table.get(key, WinRate()) + delta
    return table
