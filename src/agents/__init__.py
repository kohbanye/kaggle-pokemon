"""Swappable agent registry for the evaluation harness.

Register a policy here and it becomes selectable by name on the eval CLI
(``scripts/run_eval.py --a heuristic --b greedy``). Every factory takes the deck
and an optional ``engine`` dict of engine-derived data supplied by the runner
under Docker:

    engine = {
        "attacks": {attackId: {"dmg": int, "cost": [EnergyType, ...]}},
        "cards":   {cardId:  {"hp", "retreat", "type", "weak", "ex", "mega",
                              "basic", "ctype", "attacks": [attackId, ...]}},
    }

so the agents stay pure ``dict -> list[int]`` policies that never import ``cg``.
``heuristic`` is registered alongside one ``heuristic_no_<flag>`` variant per
feature flag, so a full-vs-variant match isolates that feature's contribution
(Phase 2's feature-level ablation).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields

from .base import Agent
from .greedy_agent import GreedyAgent
from .heuristic_agent import HeuristicAgent, HeuristicConfig
from .net_agent import NetAgent
from .random_agent import RandomAgent

AgentFactory = Callable[[list[int], dict | None], Agent]


def _attack_damage(engine: dict | None) -> dict[int, int]:
    """``attackId -> damage`` view of the engine data, for greedy's ranking."""
    attacks = (engine or {}).get("attacks", {})
    return {aid: info["dmg"] for aid, info in attacks.items()}


REGISTRY: dict[str, AgentFactory] = {
    "random": lambda deck, _engine: RandomAgent(deck),
    "greedy": lambda deck, engine: GreedyAgent(deck, _attack_damage(engine)),
    "heuristic": lambda deck, engine: HeuristicAgent(deck, engine),
    # Phase 3 skeleton: random-init policy/value net (fixed seed = reproducible).
    "net": lambda deck, engine: NetAgent(deck, engine),
}

# One ablation variant per feature flag: `heuristic_no_<flag>` = that flag off.
for _flag in (f.name for f in fields(HeuristicConfig)):
    REGISTRY[f"heuristic_no_{_flag}"] = (
        lambda deck, engine, flag=_flag: HeuristicAgent(
            deck, engine, HeuristicConfig.with_disabled(flag),
        )
    )


def build_agent(
    name: str,
    deck: list[int],
    engine: dict | None = None,
) -> Agent:
    """Construct a registered agent by name."""
    try:
        factory = REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        msg = f"unknown agent {name!r}; registered: {known}"
        raise KeyError(msg) from None
    return factory(deck, engine)


__all__ = [
    "REGISTRY",
    "Agent",
    "GreedyAgent",
    "HeuristicAgent",
    "HeuristicConfig",
    "NetAgent",
    "RandomAgent",
    "build_agent",
]
