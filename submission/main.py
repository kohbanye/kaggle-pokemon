"""Self-contained Kaggle submission agent: greedy develop-then-attack.

Bundled as ``main.py`` + ``deck.csv`` + ``cg/`` for the ladder. The policy mirrors
the unit-tested ``src/agents/greedy_agent.py`` but is inlined here (the bundle has
no ``src``), and the attack-damage map is read from the bundled engine
(``cg.all_attack()``) at startup -- no baked data, no network. The agent never
crashes: any failure returns a guaranteed-legal selection.

This is the Phase-1 first-ladder-submission baseline (greedy + a strong deck) for
local<->ladder calibration; later phases replace it with the learned policy.
"""

import os

# OptionType / SelectType mirrors (see cg.api).
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY = 7, 8, 9, 10
OPT_ATTACK, OPT_END = 13, 14
SEL_MAIN = 0

# Develop best-first (attach/evolve advance the board most; ability last as some
# abilities are re-offered and could loop), then a loop guard, then attack/end.
_DEVELOP_PRIORITY = (OPT_ATTACH, OPT_EVOLVE, OPT_PLAY, OPT_ABILITY)
_MAX_DEVELOP_ACTIONS = 40


def _read_deck():
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path) as f:
        return [int(x) for x in f.read().split() if x.strip()]


def _load_attack_damage():
    try:
        from cg.api import all_attack

        return {a.attackId: int(a.damage) for a in all_attack()}
    except Exception:
        return {}


DECK = _read_deck()
ATTACK_DAMAGE = _load_attack_damage()


def _legal_fallback(select):
    return list(range(int(select.get("maxCount", 0))))


def _best_attack(options, idxs):
    best_idx, best_dmg = idxs[0], -1
    for i in idxs:
        dmg = ATTACK_DAMAGE.get(options[i].get("attackId"), -1)
        if dmg > best_dmg:
            best_idx, best_dmg = i, dmg
    # No damages known -> prefer the last attack (heuristically the stronger).
    return best_idx if best_dmg >= 0 else idxs[-1]


def _choose_main(options, turn_action_count):
    by_type = {}
    for i, opt in enumerate(options):
        by_type.setdefault(int(opt["type"]), []).append(i)

    looping = (
        turn_action_count is not None and turn_action_count > _MAX_DEVELOP_ACTIONS
    )
    if not looping:
        for opt_type in _DEVELOP_PRIORITY:
            if opt_type in by_type:
                return by_type[opt_type][0]
    if OPT_ATTACK in by_type:
        return _best_attack(options, by_type[OPT_ATTACK])
    if OPT_END in by_type:
        return by_type[OPT_END][0]
    return 0 if options else None


def agent(obs_dict):
    """Kaggle contract: obs dict -> list of option indices (deck at init)."""
    try:
        select = obs_dict.get("select")
        if select is None:
            return list(DECK)
        if int(select.get("type", -1)) == SEL_MAIN and int(select["maxCount"]) >= 1:
            current = obs_dict.get("current") or {}
            idx = _choose_main(select["option"], current.get("turnActionCount"))
            if idx is not None:
                return [idx]
        return _legal_fallback(select)
    except Exception:
        try:
            return _legal_fallback(obs_dict.get("select") or {})
        except Exception:
            return [0]
