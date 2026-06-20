"""Tests for the Phase-3 net skeleton (no ``cg`` engine needed).

Covers the wiring the plan's exit criteria call for: fixed feature encoding,
a forward pass that round-trips the option-index contract and stays legal, a CB
head that always builds a legal 60-card deck, a bounded value head, weight
save/load, and a learning-wiring sanity (numpy SGD drives the policy loss down).
Engine stats are injected as plain dicts, exactly as the runner builds them.
"""

import numpy as np
import pytest

from src.agents import REGISTRY, build_agent
from src.agents.base import is_legal as selection_is_legal
from src.agents.net_agent import NetAgent
from src.deck import DECK_SIZE, CardInfo, CardPool
from src.deck import is_legal as deck_is_legal
from src.net.cb import build_deck
from src.net.encode import (
    OPTION_DIM,
    STATE_DIM,
    encode_option,
    encode_options,
    encode_state,
)
from src.net.features import CARD_FEAT_DIM, CardFeatures
from src.net.model import NetConfig, PolicyValueNet
from src.net.train import policy_accuracy, train_policy

# --- shared fixtures (mirror the sample deck, like test_heuristic) ---------

ATTACKS = {
    1043: {"dmg": 130, "cost": [3, 3, 0]},
    1044: {"dmg": 10, "cost": [3]},
    1047: {"dmg": 200, "cost": [3, 3, 3]},
}
CARDS = {
    3: {"hp": 0, "type": 3, "weak": None, "ex": False, "mega": False,
        "basic": False, "ctype": 5, "retreat": 0, "attacks": []},
    721: {"hp": 150, "type": 3, "weak": 4, "ex": False, "mega": False,
          "basic": True, "ctype": 0, "retreat": 2, "attacks": [1043]},
    722: {"hp": 90, "type": 3, "weak": 8, "ex": False, "mega": False,
          "basic": True, "ctype": 0, "retreat": 1, "attacks": [1044]},
    723: {"hp": 350, "type": 3, "weak": 8, "ex": False, "mega": True,
          "basic": False, "ctype": 0, "retreat": 3, "attacks": [1047]},
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


def main_obs(options: list[dict], me: dict, opp: dict, your_index: int = 0) -> dict:
    players = [me, opp] if your_index == 0 else [opp, me]
    return {
        "select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
                   "option": options},
        "current": {"yourIndex": your_index, "turnActionCount": 0,
                    "players": players},
        "logs": [],
    }


def _card(cid: int, name: str, supertype: str, stage: str, *,  # noqa: PLR0913
          basic: bool = False, energy: bool = False, ace: bool = False) -> CardInfo:
    return CardInfo(cid, name, supertype, stage,
                    is_basic_pokemon=basic, is_basic_energy=energy, is_ace_spec=ace)


def make_pool() -> CardPool:
    """A small but rule-complete pool: distinct basics, trainers, ACE SPEC, energy."""
    rows = [
        _card(721, "Kyogre", "Pokemon", "Basic Pokémon", basic=True),
        _card(722, "Snover", "Pokemon", "Basic Pokémon", basic=True),
        _card(800, "Pikachu", "Pokemon", "Basic Pokémon", basic=True),
        _card(1205, "Cyrano", "Trainer", "Supporter"),
        _card(1158, "Tool", "Trainer", "Pokémon Tool"),
        _card(900, "Master Ball", "Trainer", "Item", ace=True),
        _card(3, "Water Energy", "Energy", "Basic Energy", energy=True),
    ]
    return CardPool({c.card_id: c for c in rows})


def net() -> PolicyValueNet:
    return PolicyValueNet.random(np.random.default_rng(0))


# --- card features --------------------------------------------------------

def test_card_feature_dim_and_unknown_is_zero() -> None:
    feats = CardFeatures(ENGINE)
    assert feats.vector(721).shape == (CARD_FEAT_DIM,)
    assert np.all(feats.vector(99999) == 0.0)  # unknown id -> zero vector
    assert np.all(feats.vector(None) == 0.0)


def test_card_features_distinguish_cards() -> None:
    feats = CardFeatures(ENGINE)
    assert not np.array_equal(feats.vector(721), feats.vector(3))  # Pokemon vs Energy
    assert feats.vector(721).sum() != 0.0  # a real card has non-zero features


# --- state / option encoding ----------------------------------------------

def test_encode_state_dim_and_determinism() -> None:
    feats = CardFeatures(ENGINE)
    obs = main_obs([{"type": 14}], player(pkmn(721, 150, (3,))), player(pkmn(722, 90)))
    vec = encode_state(obs["current"], 0, feats)
    assert vec.shape == (STATE_DIM,)
    assert np.array_equal(vec, encode_state(obs["current"], 0, feats))


def test_encode_state_handles_missing_current_and_active() -> None:
    feats = CardFeatures(ENGINE)
    assert encode_state(None, 0, feats).shape == (STATE_DIM,)
    assert np.all(encode_state(None, 0, feats) == 0.0)
    # An empty Active Spot must not raise.
    cur = main_obs([{"type": 14}], player(None), player(pkmn(722, 90)))["current"]
    assert encode_state(cur, 0, feats).shape == (STATE_DIM,)


def test_encode_state_perspective_swaps_with_your_index() -> None:
    feats = CardFeatures(ENGINE)
    cur = main_obs([{"type": 14}], player(pkmn(721, 150)), player(pkmn(722, 90)))[
        "current"]
    # Same board, opposite seat -> me/opp blocks swap, so encodings differ.
    assert not np.array_equal(
        encode_state(cur, 0, feats), encode_state(cur, 1, feats))


def test_encode_option_dim_and_type_onehot() -> None:
    feats = CardFeatures(ENGINE)
    cur = main_obs([{"type": 13}], player(pkmn(721, 150, (3, 3, 3))),
                   player(pkmn(722, 90)))["current"]
    vec = encode_option({"type": 13, "attackId": 1043}, cur, 0, feats)
    assert vec.shape == (OPTION_DIM,)
    assert vec[13] == 1.0  # OptionType 13 (ATTACK) one-hot is set


def test_encode_options_stacks() -> None:
    feats = CardFeatures(ENGINE)
    cur = main_obs([], player(pkmn(721, 150)), player(pkmn(722, 90)))["current"]
    opts = [{"type": 13, "attackId": 1043}, {"type": 14}]
    assert encode_options(opts, cur, 0, feats).shape == (2, OPTION_DIM)
    assert encode_options([], cur, 0, feats).shape == (0, OPTION_DIM)


# --- net forward ----------------------------------------------------------

def test_value_is_bounded_and_deterministic() -> None:
    n = net()
    x = np.random.default_rng(1).standard_normal(STATE_DIM)
    v = n.value(x)
    assert -1.0 <= v <= 1.0
    assert v == n.value(x)


def test_policy_logits_length_matches_options() -> None:
    n = net()
    x = np.zeros(STATE_DIM)
    opts = np.random.default_rng(2).standard_normal((5, OPTION_DIM))
    assert n.policy_logits(x, opts).shape == (5,)
    assert n.policy_logits(x, np.zeros((0, OPTION_DIM))).shape == (0,)


def test_card_logits_length_matches_cards() -> None:
    n = net()
    feats = np.random.default_rng(3).standard_normal((7, CARD_FEAT_DIM))
    assert n.card_logits(feats).shape == (7,)
    assert n.card_logits(np.zeros((0, CARD_FEAT_DIM))).shape == (0,)


def test_param_count_positive() -> None:
    assert net().param_count() > 0


def test_save_load_round_trip(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    n = net()
    path = tmp_path / "weights.npz"
    n.save(path)
    loaded = PolicyValueNet.load(path)
    x = np.random.default_rng(4).standard_normal(STATE_DIM)
    assert loaded.value(x) == n.value(x)
    for k, v in n.params.items():
        assert np.array_equal(loaded.params[k], v)


def test_custom_config_dims() -> None:
    cfg = NetConfig(hidden=8, policy_hidden=4, cb_hidden=4)
    n = PolicyValueNet.random(np.random.default_rng(0), cfg)
    assert n.params["trunk_w1"].shape == (STATE_DIM, 8)


# --- CB deck construction -------------------------------------------------

def test_cb_builds_legal_60_card_deck() -> None:
    pool = make_pool()
    deck = build_deck(net(), pool, CardFeatures(ENGINE))
    assert len(deck) == DECK_SIZE
    assert deck_is_legal(deck, pool)


def test_cb_greedy_is_deterministic() -> None:
    pool, feats, n = make_pool(), CardFeatures(ENGINE), net()
    assert build_deck(n, pool, feats) == build_deck(n, pool, feats)


def test_cb_sampling_is_legal() -> None:
    pool = make_pool()
    deck = build_deck(net(), pool, CardFeatures(ENGINE),
                      np.random.default_rng(0), greedy=False)
    assert deck_is_legal(deck, pool)


def test_cb_sampling_requires_rng() -> None:
    with pytest.raises(ValueError, match="rng"):
        build_deck(net(), make_pool(), CardFeatures(ENGINE), greedy=False)


# --- NetAgent contract: legal, crash-free ---------------------------------

def test_net_in_registry_and_returns_deck_on_init() -> None:
    assert "net" in REGISTRY
    agent = build_agent("net", DECK, ENGINE)
    assert agent(DECK_REQUEST) == DECK  # no pool -> the given deck is returned


def test_cb_pool_overrides_deck_at_construction() -> None:
    pool = make_pool()
    agent = NetAgent(DECK, ENGINE, cb_pool=pool)
    out = agent(DECK_REQUEST)
    assert len(out) == DECK_SIZE
    assert deck_is_legal(out, pool)


def test_act_returns_legal_single_select() -> None:
    agent = NetAgent(DECK, ENGINE)
    options = [{"type": 13, "attackId": 1043}, {"type": 13, "attackId": 1047},
               {"type": 14}]
    obs = main_obs(options, player(pkmn(721, 150, (3, 3, 3))), player(pkmn(722, 90)))
    choice = agent.act(obs)
    assert selection_is_legal(choice, obs["select"])
    assert len(choice) == 1


def test_act_returns_legal_multi_select() -> None:
    agent = NetAgent(DECK, ENGINE)
    select = {"type": 1, "context": 2, "minCount": 0, "maxCount": 2,
              "option": [{"type": 3, "area": 5, "index": 0},
                         {"type": 3, "area": 5, "index": 1},
                         {"type": 3, "area": 5, "index": 2}]}
    obs = {"select": select, "logs": [],
           "current": {"yourIndex": 0,
                       "players": [player(None, bench=(pkmn(721, 150), pkmn(722, 90))),
                                   player(pkmn(722, 90))]}}
    choice = agent.act(obs)
    assert selection_is_legal(choice, select)
    assert len(choice) == 2


def test_act_malformed_option_falls_back() -> None:
    agent = NetAgent(DECK, ENGINE)
    # maxCount valid but options malformed; must not raise.
    select = {"type": 0, "context": 0, "minCount": 1, "maxCount": 1, "option": [{}]}
    obs = {"select": select, "current": {"yourIndex": 0, "players": []}, "logs": []}
    assert selection_is_legal(agent.act(obs), select)


def test_act_empty_options_returns_empty_legal() -> None:
    agent = NetAgent(DECK, ENGINE)
    obs = {"select": {"type": 0, "context": 0, "minCount": 0, "maxCount": 0,
                      "option": []},
           "current": {"yourIndex": 0, "players": []}, "logs": []}
    assert agent.act(obs) == []


# --- learning wiring: SGD drives the policy loss down ---------------------

def test_policy_training_reduces_loss_and_beats_chance() -> None:
    rng = np.random.default_rng(0)
    batch, k = 96, 4
    states = rng.standard_normal((batch, STATE_DIM))
    options = rng.standard_normal((batch, k, OPTION_DIM))
    # Target = the option whose first feature is largest: a signal the policy
    # head sees directly, so a working forward+backward+SGD must learn it.
    targets = options[:, :, 0].argmax(axis=1)

    n = net()
    losses = train_policy(n, states, options, targets, lr=0.1, steps=300)
    assert losses[-1] < losses[0] * 0.7  # loss fell substantially
    assert policy_accuracy(n, states, options, targets) > 0.5  # vs 0.25 chance
