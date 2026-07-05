"""Tests for src.eval.runner (routing, termination, aggregation).

A FakeEngine stands in for the compiled cg engine so the orchestration logic is
tested without the simulator: it hands out a fixed number of Decisions,
alternating which player must act, then reports a scripted Outcome.
"""

import pytest

from src.eval.runner import (
    Decision,
    GameResult,
    Obs,
    Outcome,
    aggregate,
    play_match,
    run_tournament,
)
from src.eval.schedule import round_robin


class FakeEngine:
    def __init__(self, winner: int | None, decisions: int = 4) -> None:
        self.winner = winner
        self.decisions = decisions
        self._i = 0
        self.asked: list[int] = []

    def start(self, deck0: object, deck1: object, seed: object) -> Decision | Outcome:  # noqa: ARG002
        self._i = 0
        self.asked = []
        return self._next()

    def choose(self, choice: list[int]) -> Decision | Outcome:  # noqa: ARG002
        return self._next()

    def finish(self) -> None:
        return

    def _next(self) -> Decision | Outcome:
        if self._i >= self.decisions:
            return Outcome(self.winner)
        player = self._i % 2
        self._i += 1
        self.asked.append(player)
        return Decision(player, {"step": player})


def _echo_agent(obs: Obs) -> list[int]:
    return [int(obs["step"])]  # type: ignore[call-overload]


def test_play_match_returns_scripted_winner() -> None:
    engine = FakeEngine(winner=1, decisions=4)
    assert play_match(engine, (_echo_agent, _echo_agent), ([], []), seed=0) == 1


def test_play_match_routes_to_alternating_players() -> None:
    engine = FakeEngine(winner=0, decisions=4)
    play_match(engine, (_echo_agent, _echo_agent), ([], []), seed=0)
    assert engine.asked == [0, 1, 0, 1]


def test_play_match_raises_when_game_never_ends() -> None:
    class NeverEnds:
        def start(self, deck0: object, deck1: object, seed: object) -> Decision:  # noqa: ARG002
            return Decision(0, {"step": 0})

        def choose(self, choice: list[int]) -> Decision:  # noqa: ARG002
            return Decision(0, {"step": 0})

        def finish(self) -> None:
            return

    with pytest.raises(RuntimeError, match="did not terminate"):
        play_match(NeverEnds(), (_echo_agent, _echo_agent), ([], []), seed=0)


def test_aggregate_scores_from_low_name_and_swaps_sides() -> None:
    results = [
        GameResult("A", "B", winner=0),  # A wins as player0
        GameResult("B", "A", winner=0),  # B wins as player0 -> loss for A
        GameResult("A", "B", winner=None),  # tie
    ]
    table = aggregate(results)
    wr = table[("A", "B")]
    assert (wr.wins, wr.losses, wr.ties) == (1, 1, 1)


def test_run_tournament_plays_every_scheduled_match() -> None:
    engine = FakeEngine(winner=0, decisions=2)  # player0 always wins
    decks = {"A": [1], "B": [2]}
    matches = round_robin(["A", "B"], games_per_pair=2)
    results = run_tournament(engine, lambda deck: _echo_agent, decks, matches)  # noqa: ARG005
    assert len(results) == 2
    table = aggregate(results)
    # A wins when player0 (game 1); B wins when player0 (game 2) -> A: 1-1.
    assert (table[("A", "B")].wins, table[("A", "B")].losses) == (1, 1)
