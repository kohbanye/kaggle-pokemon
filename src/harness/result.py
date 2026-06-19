"""Per-game result record and winner attribution.

The engine reports a winner by *slot* (player 0 / player 1). Because the harness
swaps which slot the subject agent ("A") plays each game to cancel any first-
player advantage, attribution must map a slot result back to A vs B. That
mapping lives here, kept pure so it is unit-tested without the engine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Engine result codes (cg.api LogType.RESULT): slot index of the winner, 2 = draw.
DRAW = 2
ABORTED = -1


@dataclass
class GameResult:
    """One played game, recorded from the subject agent A's perspective."""

    a_is_player0: bool  # did agent A play slot 0 this game?
    winner: int  # raw engine result: 0, 1, 2 (draw), or -1 (aborted)
    turns: int  # final turn counter
    selections: int  # total selections made by both agents
    agent_time_a: float  # cumulative seconds A spent deciding
    agent_time_b: float
    moves_a: int  # number of selections A made
    moves_b: int
    max_move_a: float  # slowest single decision by A (seconds)
    max_move_b: float
    wall_s: float  # wall-clock for the whole game

    @property
    def a_won(self) -> bool:
        return self.winner == (0 if self.a_is_player0 else 1)

    @property
    def b_won(self) -> bool:
        return self.winner == (1 if self.a_is_player0 else 0)

    @property
    def is_draw(self) -> bool:
        return self.winner == DRAW

    @property
    def is_aborted(self) -> bool:
        return self.winner == ABORTED

    def winner_label(self) -> str:
        if self.a_won:
            return "A"
        if self.b_won:
            return "B"
        return "draw" if self.is_draw else "aborted"

    def as_row(self, game_id: int, seed: int, name_a: str, name_b: str) -> dict:
        """Flat dict for CSV logging."""
        return {
            "game_id": game_id,
            "seed": seed,
            "agent_a": name_a,
            "agent_b": name_b,
            "a_is_player0": self.a_is_player0,
            "winner": self.winner_label(),
            **asdict(self),
        }
