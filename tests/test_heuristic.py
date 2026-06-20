"""Tests for the Phase-2 heuristic agent + state evaluator (no engine needed).

Engine card/attack stats are injected as plain dicts (exactly what the runner
builds from ``all_card_data()`` / ``all_attack()``), so these run natively. The
fixtures mirror the real sample deck: Kyogre (721), Snover (722), Mega Abomasnow
ex (723), a Tool (1158), a Supporter (1205), and Basic {W} Energy (3).
"""

import pytest

from src.agents import REGISTRY, build_agent
from src.agents.heuristic_agent import HeuristicAgent, HeuristicConfig, evaluate_state

# attackId -> {dmg, cost(list[EnergyType])}; WATER=3, COLORLESS=0.
ATTACKS = {
    1042: {"dmg": 0, "cost": [3]},          # Kyogre Riptide (effect-only)
    1043: {"dmg": 130, "cost": [3, 3, 0]},  # Kyogre Swirling Waves
    1044: {"dmg": 10, "cost": [3]},         # Snover Beat
    1045: {"dmg": 30, "cost": [3, 3]},      # Snover Icy Snow
    1046: {"dmg": 0, "cost": [3, 3]},       # Abomasnow Hammer-lanche (effect-only)
    1047: {"dmg": 200, "cost": [3, 3, 3]},  # Abomasnow Frost Barrier
}
# cardId -> compact stats. ctype: POKEMON=0, TOOL=2, SUPPORTER=3, BASIC_ENERGY=5.
CARDS = {
    3: {"hp": 0, "type": 3, "weak": None, "ex": False, "mega": False,
        "basic": False, "ctype": 5, "attacks": []},
    721: {"hp": 150, "type": 3, "weak": 4, "ex": False, "mega": False,
          "basic": True, "ctype": 0, "attacks": [1042, 1043]},
    722: {"hp": 90, "type": 3, "weak": 8, "ex": False, "mega": False,
          "basic": True, "ctype": 0, "attacks": [1044, 1045]},
    723: {"hp": 350, "type": 3, "weak": 8, "ex": False, "mega": True,
          "basic": False, "ctype": 0, "attacks": [1046, 1047]},
    1158: {"hp": 0, "type": 0, "weak": None, "ex": False, "mega": False,
           "basic": False, "ctype": 2, "attacks": []},
    1205: {"hp": 0, "type": 0, "weak": None, "ex": False, "mega": False,
           "basic": False, "ctype": 3, "attacks": []},
}
ENGINE = {"attacks": ATTACKS, "cards": CARDS}
DECK = list(range(1, 61))
DECK_REQUEST = {"select": None, "logs": [], "current": None}


def pkmn(cid: int, hp: int, energies: tuple[int, ...] = ()) -> dict:
    return {"id": cid, "hp": hp, "maxHp": CARDS[cid]["hp"], "energies": list(energies)}


def player(active: dict | None, bench: tuple[dict, ...] = (),
           prize: int = 6, hand: tuple[int, ...] = ()) -> dict:
    return {
        "active": [active] if active is not None else [],
        "bench": list(bench),
        "prize": [None] * prize,
        "hand": [{"id": c} for c in hand],
        "handCount": len(hand),
        "deckCount": 40,
    }


def main_obs(options: list[dict], me: dict, opp: dict,
             your_index: int = 0, turn_action_count: int = 0) -> dict:
    players = [me, opp] if your_index == 0 else [opp, me]
    return {
        "select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
                   "option": options},
        "current": {"yourIndex": your_index, "turnActionCount": turn_action_count,
                    "players": players},
        "logs": [],
    }


def heuristic(cfg: HeuristicConfig | None = None) -> HeuristicAgent:
    return HeuristicAgent(DECK, ENGINE, cfg)


# --- contract / registry --------------------------------------------------

def test_returns_deck_on_none_select() -> None:
    assert build_agent("heuristic", DECK, ENGINE)(DECK_REQUEST) == DECK


def test_registry_has_heuristic_and_ablation_variants() -> None:
    assert "heuristic" in REGISTRY
    for flag in ("attach_target", "attack_ko", "weakness", "bench_dev",
                 "retreat", "promote"):
        assert f"heuristic_no_{flag}" in REGISTRY
        # Each variant is constructible and answers the deck request.
        assert build_agent(f"heuristic_no_{flag}", DECK, ENGINE)(DECK_REQUEST) == DECK


def test_with_disabled_rejects_unknown_flag() -> None:
    with pytest.raises(AttributeError):
        HeuristicConfig.with_disabled("nope")


# --- attach_target: fuel the attacker that needs it -----------------------

def test_attach_targets_bench_attacker_over_full_active() -> None:
    # Active Mega Abomasnow already affords its 200 attack (3 energy); the
    # benched Snover gains a new attack from one more energy -> feed Snover.
    me = player(pkmn(723, 350, (3, 3, 3)), bench=(pkmn(722, 90, (3,)),), hand=(3,))
    opp = player(pkmn(721, 150, (3,)))
    options = [
        {"type": 8, "area": 2, "index": 0, "inPlayArea": 4, "inPlayIndex": 0},
        {"type": 8, "area": 2, "index": 0, "inPlayArea": 5, "inPlayIndex": 0},
        {"type": 14},
    ]
    assert heuristic().act(main_obs(options, me, opp)) == [1]  # bench target
    # With the feature off, attach reverts to greedy's first-offered target.
    assert heuristic(HeuristicConfig.with_disabled("attach_target")).act(
        main_obs(options, me, opp)) == [0]


# --- bench_dev: don't leave the bench empty -------------------------------

def test_bench_dev_plays_basic_when_bench_thin() -> None:
    me = player(pkmn(723, 350, (3, 3, 3)), bench=(), hand=(1205, 721))
    opp = player(pkmn(721, 150))
    options = [
        {"type": 7, "index": 0},  # play Supporter (Cyrano)
        {"type": 7, "index": 1},  # play Basic Pokemon (Kyogre)
        {"type": 14},
    ]
    assert heuristic().act(main_obs(options, me, opp)) == [1]  # bench the Basic
    assert heuristic(HeuristicConfig.with_disabled("bench_dev")).act(
        main_obs(options, me, opp)) == [0]


# --- retreat: pull a doomed active when a bench attacker can take over -----

def test_retreats_doomed_active_with_ready_bench() -> None:
    # Snover (90 HP) faces a Mega Abomasnow that hits for 200; Kyogre on the
    # bench is fuelled and can attack -> retreat rather than chip-and-die.
    me = player(pkmn(722, 90, (3, 3, 3)), bench=(pkmn(721, 150, (3, 3, 3)),))
    opp = player(pkmn(723, 350, (3, 3, 3)))
    options = [{"type": 12}, {"type": 13, "attackId": 1044}, {"type": 14}]
    assert heuristic().act(main_obs(options, me, opp)) == [0]  # retreat
    # Feature off -> never retreats; falls through to attacking.
    assert heuristic(HeuristicConfig.with_disabled("retreat")).act(
        main_obs(options, me, opp)) == [1]


def test_no_retreat_when_active_is_safe() -> None:
    # Opponent has no energy -> can't KO; keep attacking, don't waste retreat.
    me = player(pkmn(722, 90, (3, 3, 3)), bench=(pkmn(721, 150, (3, 3, 3)),))
    opp = player(pkmn(723, 350))
    options = [{"type": 12}, {"type": 13, "attackId": 1044}, {"type": 14}]
    assert heuristic().act(main_obs(options, me, opp)) == [1]  # attack, not retreat


# --- attack: pick the hardest-hitting (lethal-aware) attack ---------------

def test_best_attack_picks_higher_damage() -> None:
    me = player(pkmn(722, 90, (3, 3)))
    opp = player(pkmn(721, 150, (3,)))
    options = [
        {"type": 13, "attackId": 1044},  # 10
        {"type": 13, "attackId": 1045},  # 30
        {"type": 14},
    ]
    assert heuristic().act(main_obs(options, me, opp)) == [1]


# --- promote: put the strongest Pokemon in the Active Spot ----------------

def test_promote_picks_strongest_benched_pokemon() -> None:
    me = player(None, bench=(pkmn(722, 90), pkmn(723, 350, (3, 3, 3))))
    opp = player(pkmn(721, 150, (3,)))
    select = {
        "type": 1, "context": 4, "minCount": 1, "maxCount": 1,
        "option": [
            {"type": 3, "area": 5, "index": 0},  # Snover
            {"type": 3, "area": 5, "index": 1},  # fuelled Mega Abomasnow
        ],
    }
    obs = {"select": select,
           "current": {"yourIndex": 0, "players": [me, opp]}, "logs": []}
    assert heuristic().act(obs) == [1]  # the ready Mega Abomasnow
    assert heuristic(HeuristicConfig.with_disabled("promote")).act(obs) == [0]


# --- evaluate_state: feature-driven board score ---------------------------

def test_eval_symmetric_state_is_neutral() -> None:
    me = player(pkmn(722, 90))
    opp = player(pkmn(722, 90))
    state = {"players": [me, opp]}
    assert evaluate_state(state, 0, CARDS, ATTACKS) == 0.0


def test_eval_prize_lead_dominates() -> None:
    me = player(pkmn(722, 90), prize=2)
    opp = player(pkmn(722, 90), prize=6)
    state = {"players": [me, opp]}
    assert evaluate_state(state, 0, CARDS, ATTACKS) > 0


def test_eval_rewards_ko_opportunity_with_colorless_cost() -> None:
    # Kyogre's 130 attack costs {W}{W}+1 colorless; 3 energy affords it and KOs
    # a 90-HP Defender. Two energy can't pay the colorless -> no KO term.
    can_ko = {"players": [player(pkmn(721, 150, (3, 3, 3))), player(pkmn(722, 90))]}
    cannot = {"players": [player(pkmn(721, 150, (3, 3))), player(pkmn(722, 90))]}
    assert evaluate_state(can_ko, 0, CARDS, ATTACKS) > evaluate_state(
        cannot, 0, CARDS, ATTACKS)


def test_eval_punishes_missing_active() -> None:
    has_active = {"players": [player(pkmn(722, 90)), player(pkmn(722, 90))]}
    no_active = {"players": [player(None), player(pkmn(722, 90))]}
    assert evaluate_state(no_active, 0, CARDS, ATTACKS) < evaluate_state(
        has_active, 0, CARDS, ATTACKS)


# --- never crash ----------------------------------------------------------

def test_malformed_main_option_falls_back() -> None:
    # Option missing "type" raises inside scoring -> caught, legal fallback.
    obs = {"select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
                      "option": [{}]},
           "current": {"yourIndex": 0, "players": []}, "logs": []}
    assert heuristic().act(obs) == [0]


def test_unhandled_selection_uses_legal_fallback() -> None:
    # A YES/NO selection is not MAIN/promote -> legal fallback (first maxCount).
    obs = {"select": {"type": 9, "context": 43, "minCount": 1, "maxCount": 1,
                      "option": [{"type": 1}, {"type": 2}]},
           "current": {"yourIndex": 0, "players": []}, "logs": []}
    assert heuristic().act(obs) == [0]


def test_multi_select_uses_legal_fallback() -> None:
    obs = {"select": {"type": 1, "context": 2, "minCount": 0, "maxCount": 2,
                      "option": [{"type": 3}, {"type": 3}, {"type": 3}]},
           "current": {"yourIndex": 0, "players": []}, "logs": []}
    assert heuristic().act(obs) == [0, 1]
