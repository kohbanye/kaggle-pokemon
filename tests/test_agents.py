"""Tests for the random / greedy baselines on synthetic observation dicts.

These never touch the engine: agents are pure ``dict -> list[int]`` policies.
"""

from src.agents import build_agent
from src.agents.base import (
    OPT_ABILITY,
    OPT_ATTACH,
    OPT_ATTACK,
    OPT_END,
    OPT_EVOLVE,
    OPT_PLAY,
)
from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent

DECK = list(range(1, 61))
DECK_REQUEST = {"select": None, "logs": [], "current": None}


def _main_obs(option_types: list[int], turn_action_count: int = 0,
              attack_ids: dict[int, int] | None = None) -> dict:
    """A MAIN selection (type 0, pick exactly one) over the given option types."""
    options = []
    for i, t in enumerate(option_types):
        opt = {"type": t}
        if attack_ids and i in attack_ids:
            opt["attackId"] = attack_ids[i]
        options.append(opt)
    return {
        "select": {
            "type": 0, "context": 0, "minCount": 1, "maxCount": 1, "option": options,
        },
        "current": {"turnActionCount": turn_action_count, "yourIndex": 0},
        "logs": [],
    }


# --- deck contract --------------------------------------------------------

def test_agents_return_deck_on_none_select() -> None:
    for name in ("random", "greedy"):
        agent = build_agent(name, DECK)
        assert agent(DECK_REQUEST) == DECK


# --- greedy ---------------------------------------------------------------

def test_greedy_develops_before_attacking() -> None:
    agent = GreedyAgent(DECK)
    # Energy can still be attached this turn: develop before swinging.
    obs = _main_obs([OPT_ATTACH, OPT_ATTACK, OPT_END])
    assert agent.act(obs) == [0]  # OPT_ATTACH, not the attack


def test_greedy_attacks_when_nothing_to_develop() -> None:
    agent = GreedyAgent(DECK)
    obs = _main_obs([OPT_ATTACK, OPT_END])
    assert agent.act(obs) == [0]  # the ATTACK option


def test_greedy_develops_before_ending() -> None:
    agent = GreedyAgent(DECK)
    # No attack available: attach energy beats play/ability, and beats ending.
    obs = _main_obs([OPT_PLAY, OPT_ATTACH, OPT_ABILITY, OPT_END])
    assert agent.act(obs) == [1]  # OPT_ATTACH


def test_greedy_prefers_evolve_over_ability() -> None:
    agent = GreedyAgent(DECK)
    obs = _main_obs([OPT_ABILITY, OPT_EVOLVE, OPT_END])
    assert agent.act(obs) == [1]  # OPT_EVOLVE


def test_greedy_ends_when_only_end() -> None:
    agent = GreedyAgent(DECK)
    obs = _main_obs([OPT_END])
    assert agent.act(obs) == [0]


def test_greedy_ranks_attacks_by_injected_damage() -> None:
    agent = GreedyAgent(DECK, attack_damage={101: 30, 202: 120})
    obs = _main_obs([OPT_ATTACK, OPT_ATTACK], attack_ids={0: 101, 1: 202})
    assert agent.act(obs) == [1]  # the 120-damage attack


def test_greedy_loop_guard_forces_end() -> None:
    agent = GreedyAgent(DECK)
    # Repeatable develop option, but the turn has dragged on: force progress.
    obs = _main_obs([OPT_ABILITY, OPT_END], turn_action_count=999)
    assert agent.act(obs) == [1]  # OPT_END, not the looping ability


def test_greedy_non_main_uses_legal_fallback() -> None:
    agent = GreedyAgent(DECK)
    obs = {
        "select": {"type": 1, "context": 8, "minCount": 1, "maxCount": 1,
                   "option": [{"type": 3}, {"type": 3}, {"type": 3}]},
        "current": {"yourIndex": 0},
        "logs": [],
    }
    assert agent.act(obs) == [0]


# --- random ---------------------------------------------------------------

def test_random_returns_legal_count_in_range() -> None:
    agent = RandomAgent(DECK, seed=0)
    obs = {
        "select": {"type": 1, "context": 8, "minCount": 1, "maxCount": 2,
                   "option": [{"type": 3} for _ in range(5)]},
        "current": {"yourIndex": 0},
        "logs": [],
    }
    choice = agent.act(obs)
    assert len(choice) == 2
    assert len(set(choice)) == 2
    assert all(0 <= i < 5 for i in choice)


def test_random_is_reproducible_after_reset() -> None:
    obs = _main_obs([OPT_ATTACH, OPT_ATTACK, OPT_END, OPT_PLAY])
    obs["select"]["maxCount"] = 1
    a, b = RandomAgent(DECK), RandomAgent(DECK)
    a.reset(7)
    b.reset(7)
    assert [a.act(obs) for _ in range(5)] == [b.act(obs) for _ in range(5)]


def test_random_handles_zero_max_count() -> None:
    agent = RandomAgent(DECK)
    obs = {
        "select": {
            "type": 9, "context": 43, "minCount": 0, "maxCount": 0, "option": [],
        },
        "current": {"yourIndex": 0},
        "logs": [],
    }
    assert agent.act(obs) == []
