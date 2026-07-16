"""Minimal extensive-form game interface for the ReBeL estimator kill-gate (Gate B).

Before any engine/belief code, we must prove that our sampled counterfactual-regret
estimator is UNBIASED -- it converges to the SAME equilibrium as exact CFR on a game
a KNOWN solution (Codex review, docs/research/rebel-approximate-design.md Gate B). This
module defines the tiny game interface both `src.rebel.cfr` solvers consume, plus **Kuhn
poker** -- the canonical CFR benchmark: chance (the deal), asymmetric reach, an
information-revealing public action (bet/check), and an analytically-known Nash whose
exploitability is 0. If external-sampling MCCFR converges to Kuhn's Nash, the estimator
math (reach/importance weighting, per-player ranges) is sound; only then do we wire the
determinized engine onto this same interface.

A history ``h`` is an immutable tuple. Player 0 maximises ``utility``; player 1
minimises (two-player zero-sum). Chance is player ``CHANCE``.
"""

from __future__ import annotations

from typing import Protocol

CHANCE = -1


class Game(Protocol):
    """Two-player zero-sum extensive-form game with chance, keyed by history tuples."""

    def root(self) -> tuple: ...
    def is_terminal(self, h: tuple) -> bool: ...
    def utility(self, h: tuple) -> float:
        """Terminal payoff to player 0 (player 1 gets the negation)."""
        ...
    def current_player(self, h: tuple) -> int:
        """0, 1, or ``CHANCE``."""
        ...
    def chance_outcomes(self, h: tuple) -> list[tuple[object, float]]:
        """``[(action, prob), ...]`` at a chance node."""
        ...
    def legal_actions(self, h: tuple) -> list[object]: ...
    def infoset_key(self, h: tuple) -> str:
        """The acting player's information set (what THEY observe) — the CFR key."""
        ...
    def next(self, h: tuple, action: object) -> tuple: ...


# --- Kuhn poker ---------------------------------------------------------------

_KUHN_DEALS = [(a, b) for a in range(3) for b in range(3) if a != b]  # 6 equal deals
_PASS, _BET = "p", "b"


class KuhnPoker:
    """3-card Kuhn poker. History = (card0, card1, *actions). Cards J<Q<K = 0<1<2.

    Betting: each player antes 1. Action sequences and payoffs (to player 0):
    ``pp`` showdown ±1; ``bb``/``pbb`` showdown ±2; ``bp`` p0 wins 1; ``pbp`` p1 wins 1.
    Known Nash exploitability = 0 (p0's family parameterised by alpha in [0,1/3]).
    """

    def root(self) -> tuple:
        return ()

    def _dealt(self, h: tuple) -> bool:
        return len(h) >= 2  # noqa: PLR2004

    def current_player(self, h: tuple) -> int:
        if not self._dealt(h):
            return CHANCE
        return len(h[2:]) % 2

    def chance_outcomes(self, h: tuple) -> list[tuple[object, float]]:  # noqa: ARG002
        # One chance node deals both cards at once (a single joint outcome).
        return [(d, 1.0 / len(_KUHN_DEALS)) for d in _KUHN_DEALS]

    def next(self, h: tuple, action: object) -> tuple:
        if not self._dealt(h):
            c0, c1 = action  # type: ignore[misc]  # chance action is the (c0, c1) deal
            return (c0, c1)
        return (*h, action)

    def legal_actions(self, h: tuple) -> list[object]:  # noqa: ARG002
        return [_PASS, _BET]

    def _actions(self, h: tuple) -> str:
        return "".join(h[2:])

    def is_terminal(self, h: tuple) -> bool:
        a = self._actions(h)
        return a in {"pp", "bb", "bp", "pbp", "pbb"}

    def infoset_key(self, h: tuple) -> str:
        player = self.current_player(h)
        return f"{h[player]}:{self._actions(h)}"  # own card + public action history

    def utility(self, h: tuple) -> float:
        a = self._actions(h)
        c0, c1 = h[0], h[1]
        p0_high = c0 > c1
        if a == "bp":      # p0 bet, p1 folded -> p0 wins the ante
            return 1.0
        if a == "pbp":     # p1 bet, p0 folded -> p0 loses the ante
            return -1.0
        pot = 2.0 if a in {"bb", "pbb"} else 1.0  # showdown stake
        return pot if p0_high else -pot
