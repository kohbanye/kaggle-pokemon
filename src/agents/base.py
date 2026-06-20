"""Agent base class and the engine-free option/selection constants.

Agents in this project are pure ``dict -> list[int]`` policies: they read the
raw observation dict (exactly the object the Kaggle harness passes to
``agent(obs)``) and return option indices. Crucially they do **not** import the
``cg`` engine, so they import and unit-test natively on any platform; engine-
derived data (e.g. attack damages) is *injected* by the runner under Docker.

The integer constants below mirror ``cg.api.OptionType`` / ``cg.api.SelectType``
so we don't have to import the (Linux-only) engine just to branch on them.
"""

from __future__ import annotations

# --- OptionType (mirror of cg.api.OptionType) -----------------------------
OPT_NUMBER = 0
OPT_YES = 1
OPT_NO = 2
OPT_CARD = 3
OPT_TOOL_CARD = 4
OPT_ENERGY_CARD = 5
OPT_ENERGY = 6
OPT_PLAY = 7
OPT_ATTACH = 8
OPT_EVOLVE = 9
OPT_ABILITY = 10
OPT_DISCARD = 11
OPT_RETREAT = 12
OPT_ATTACK = 13
OPT_END = 14
OPT_SKILL = 15
OPT_SPECIAL_CONDITION = 16

# --- SelectType (mirror of cg.api.SelectType) -----------------------------
SEL_MAIN = 0
SEL_CARD = 1

# --- AreaType (mirror of cg.api.AreaType) ---------------------------------
# Only the in-play areas the policies branch on. An ATTACH/EVOLVE option points
# at its on-field target via ``inPlayArea`` + ``inPlayIndex``; ``area`` (with
# ``index``) points at the source card (e.g. the hand slot of the energy).
AREA_HAND = 2
AREA_ACTIVE = 4
AREA_BENCH = 5

# --- CardType (mirror of cg.api.CardType) ---------------------------------
CARD_POKEMON = 0
CARD_ITEM = 1
CARD_TOOL = 2
CARD_SUPPORTER = 3
CARD_STADIUM = 4
CARD_BASIC_ENERGY = 5
CARD_SPECIAL_ENERGY = 6

# --- EnergyType (mirror of cg.api.EnergyType) -----------------------------
# Only the two we special-case: COLORLESS in an attack cost means "any energy",
# and RAINBOW attached energy pays for any single colored requirement.
ENERGY_COLORLESS = 0
ENERGY_RAINBOW = 10

# --- SelectContext (mirror of cg.api.SelectContext) -----------------------
# Promotion contexts: the engine is asking which Pokemon to put in the Active
# Spot (at set-up, after a Knock Out, or via a switch effect).
CTX_SETUP_ACTIVE = 1
CTX_SWITCH = 3
CTX_TO_ACTIVE = 4


def legal_fallback(select: dict) -> list[int]:
    """A guaranteed-legal selection for ``select``: the first ``maxCount`` indices.

    ``maxCount`` never exceeds ``len(option)`` and lies in ``[minCount, maxCount]``
    by construction, so ``range(maxCount)`` is always a legal, duplicate-free
    choice. This is the universal safety net (used on agent error / illegal
    output) that keeps a match from ever crashing.
    """
    max_count = int(select.get("maxCount", 0))
    return list(range(max_count))


def is_legal(choice: object, select: dict) -> bool:
    """True if ``choice`` is a legal selection for ``select``.

    Legal means: a list of distinct ints, each in ``[0, len(option))``, with
    length in ``[minCount, maxCount]``.
    """
    if not isinstance(choice, list):
        return False
    n_options = len(select.get("option", []))
    min_count = int(select.get("minCount", 0))
    max_count = int(select.get("maxCount", 0))
    if not min_count <= len(choice) <= max_count:
        return False
    if len(set(choice)) != len(choice):
        return False
    return all(isinstance(i, int) and 0 <= i < n_options for i in choice)


class Agent:
    """Base policy. Subclasses implement :meth:`act`.

    Calling the agent with the deck-selection observation (``select is None``,
    which the engine sends only at the very start) returns the 60-card deck;
    every other call is delegated to :meth:`act`.
    """

    name = "base"

    def __init__(self, deck: list[int]) -> None:
        self.deck = list(deck)

    def reset(self, seed: int) -> None:
        """Reseed per-game randomness. No-op for deterministic agents."""

    def act(self, obs: dict) -> list[int]:
        raise NotImplementedError

    def __call__(self, obs: dict) -> list[int]:
        if obs.get("select") is None:
            return list(self.deck)
        return self.act(obs)
