"""Swappable agent registry for the evaluation harness.

Register a policy here and it becomes selectable by name on the eval CLI
(``scripts/run_eval.py --a greedy --b random``). Every factory takes the deck
and an optional ``attack_damage`` map (``attackId -> damage``, supplied by the
runner from the engine) and returns a fresh :class:`~src.agents.base.Agent`.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import Agent
from .greedy_agent import GreedyAgent
from .random_agent import RandomAgent

AgentFactory = Callable[[list[int], dict[int, int] | None], Agent]

REGISTRY: dict[str, AgentFactory] = {
    "random": lambda deck, _damage: RandomAgent(deck),
    "greedy": lambda deck, damage: GreedyAgent(deck, damage),
}


def build_agent(
    name: str,
    deck: list[int],
    attack_damage: dict[int, int] | None = None,
) -> Agent:
    """Construct a registered agent by name."""
    try:
        factory = REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        msg = f"unknown agent {name!r}; registered: {known}"
        raise KeyError(msg) from None
    return factory(deck, attack_damage)


__all__ = ["REGISTRY", "Agent", "GreedyAgent", "RandomAgent", "build_agent"]
