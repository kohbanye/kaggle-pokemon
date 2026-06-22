"""Tests for the Phase-4 behaviour-cloning pipeline (torch side, no ``cg``).

Covers the wiring the plan's exit criteria depend on: the log->batch encoding
(collate shapes / mask / padding), the value-target sign and discount, the CB
supervision legality invariant, the CB loss training and its isolation to the CB
head, and an end-to-end train -> export -> ``NetAgent`` legal-selection smoke.
"""

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.agents.base import is_legal as selection_is_legal
from src.agents.net_agent import NetAgent
from src.deck import CardInfo, CardPool
from src.deck import is_legal as deck_is_legal
from src.net.bc_data import (
    CBSequenceDataset,
    PolicyDataset,
    PolicySample,
    build_policy_samples,
    cb_rl_samples,
    cb_sequences,
    cb_supervision,
    collate_cb,
    collate_cb_seq,
    collate_policy,
)
from src.net.cb import build_deck
from src.net.encode import OPTION_DIM, SLOT_MAX, STATE_DIM, STATE_EMBED_SLOTS
from src.net.features import CARD_FEAT_DIM, CardFeatures
from src.net.lit import LitCBSeq, LitPolicyValue
from src.net.model import NetConfig
from src.net.torch_model import TorchPolicyValueNet

# A small hand-written engine dump, exactly as the runner injects it.
ENGINE = {
    "cards": {
        10: {"hp": 60, "type": 0, "weak": None, "ex": False, "mega": False,
             "basic": True, "ctype": 0, "retreat": 1, "attacks": [100]},
        11: {"hp": 120, "type": 1, "weak": 0, "ex": True, "mega": False,
             "basic": True, "ctype": 0, "retreat": 2, "attacks": [101]},
        2: {"hp": 0, "type": 0, "weak": None, "ex": False, "mega": False,
            "basic": False, "ctype": 5, "retreat": 0, "attacks": []},
    },
    "attacks": {100: {"dmg": 20, "cost": [0]}, 101: {"dmg": 90, "cost": [1, 1]}},
}
FEATS = CardFeatures(ENGINE)


def _sample(
    state: np.ndarray, options: np.ndarray, target: int, value: float,
) -> PolicySample:
    """A PolicySample with empty (UNK) shared-embedding rows for the play head."""
    rows = np.zeros((STATE_EMBED_SLOTS, SLOT_MAX), dtype=np.intp)
    mask = np.zeros((STATE_EMBED_SLOTS, SLOT_MAX), dtype=np.bool_)
    opt_rows = np.zeros(options.shape[0], dtype=np.intp)
    return PolicySample(state, rows, mask, options, opt_rows, target, value)


def _decision(slot: int, agent: str, target: int, n_options: int = 2) -> dict:
    """A minimal single-select decision record (END-type options, no card)."""
    options = [{"type": 14} for _ in range(n_options)]
    obs = {
        "select": {"maxCount": 1, "minCount": 1, "option": options},
        "current": {"yourIndex": slot, "players": [{}, {}]},
    }
    return {"slot": slot, "agent": agent, "obs": obs, "choice": [target]}


def _info(
    card_id: int,
    name: str,
    *,
    basic_pokemon: bool = False,
    basic_energy: bool = False,
    ace_spec: bool = False,
) -> CardInfo:
    return CardInfo(
        card_id=card_id, name=name, supertype="", stage_or_type="",
        is_basic_pokemon=basic_pokemon, is_basic_energy=basic_energy,
        is_ace_spec=ace_spec,
    )


def _pool() -> CardPool:
    """A tiny legal-to-build pool: two basics, an item, an ACE SPEC, an energy."""
    return CardPool({
        10: _info(10, "Alpha", basic_pokemon=True),
        11: _info(11, "Beta", basic_pokemon=True),
        20: _info(20, "Item"),
        30: _info(30, "Ace", ace_spec=True),
        2: _info(2, "Fire Energy", basic_energy=True),
    })


# --- policy batch encoding -------------------------------------------------

def test_collate_policy_shapes_and_mask() -> None:
    rng = np.random.default_rng(0)
    samples = [
        _sample(rng.standard_normal(STATE_DIM), rng.standard_normal((k, OPTION_DIM)),
                0, 0.5)
        for k in (2, 3, 1)
    ]
    states, _state_rows, _state_mask, options, mask, _opt_rows, targets, values = (
        collate_policy(samples)
    )
    assert states.shape == (3, STATE_DIM)
    assert options.shape == (3, 3, OPTION_DIM)  # padded to batch-max K = 3
    assert mask.dtype == torch.bool
    assert targets.dtype == torch.long
    assert values.dtype == torch.float32
    assert mask.tolist() == [
        [True, True, False], [True, True, True], [True, False, False],
    ]
    # Padded option rows are zeroed (so a masked logit can never leak signal).
    assert torch.count_nonzero(options[0, 2]) == 0
    assert torch.count_nonzero(options[2, 1:]) == 0


def test_value_targets_final_and_discounted() -> None:
    # slot 0 decides twice and wins (winner == 0); slot 1 decides once.
    games = [{"winner": 0, "decisions": [
        _decision(0, "heuristic", 0),
        _decision(1, "greedy", 1),
        _decision(0, "heuristic", 1),
    ]}]
    final = build_policy_samples(games, FEATS, teachers={"heuristic", "greedy"})
    assert [round(s.value, 3) for s in final] == [1.0, -1.0, 1.0]

    # Only the heuristic (slot 0) decisions; first has one own decision left (d=1).
    disc = build_policy_samples(games, FEATS, teachers={"heuristic"}, discount=0.5)
    assert [round(s.value, 3) for s in disc] == [0.5, 1.0]


# --- CB supervision --------------------------------------------------------

def test_cb_supervision_targets_are_legal() -> None:
    pool = _pool()
    deck = [10, 10, 11, 20, 2, 2]  # short legal prefix; per-step caps never bind
    card_feats, samples = cb_supervision(
        [deck], pool, FEATS, np.random.default_rng(0), shuffles=3,
    )
    n_pool = len(pool.ids())
    assert card_feats.shape == (n_pool, CARD_FEAT_DIM)
    assert len(samples) == 3 * len(deck)
    for sample in samples:
        assert sample.legal_mask.shape == (n_pool,)
        assert bool(sample.legal_mask[sample.target_idx])  # target always legal

    # Inverse-copy weight: energy (id 2, 2 copies) -> 0.5; the 4-of style is 1/n.
    by_target = {s.target_idx: s.weight for s in samples}
    energy_idx = sorted(pool.ids()).index(2)
    alpha_idx = sorted(pool.ids()).index(10)  # 2 copies in the deck -> 0.5
    assert by_target[energy_idx] == 0.5
    assert by_target[alpha_idx] == 0.5

    masks, targets, weights = collate_cb(samples[:4])
    assert masks.shape == (4, n_pool)
    assert masks.dtype == torch.bool
    assert targets.dtype == torch.long
    assert weights.dtype == torch.float32


# --- CB sequence: the LSTM deck head trains and stays isolated --------------

def _seq_trainer() -> L.Trainer:
    return L.Trainer(
        max_epochs=40, accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False,
    )


def test_lstm_seq_bc_learns_demo_cards() -> None:
    # Phase 5c: the autoregressive LSTM deck head, BC-cloned on a demo deck, makes
    # greedy decode pick the demo's distinct cards (legal). Only the CB head + LSTM
    # train; trunk/policy/value are frozen.
    torch.manual_seed(0)
    pool = _pool()
    net = TorchPolicyValueNet(NetConfig(n_cards=len(pool.ids())))
    before = {name: p.detach().clone() for name, p in net.named_parameters()}
    deck = [10] * 4 + [11] * 4 + [20] * 4 + [30] + [2] * 47  # legal 60
    card_feats, seqs = cb_sequences(
        [deck], pool, FEATS, np.random.default_rng(0), shuffles=6,
    )
    litcb = LitCBSeq(net, card_feats, lr=0.05)
    loader = DataLoader(
        CBSequenceDataset(seqs), batch_size=8, collate_fn=collate_cb_seq,
    )
    _seq_trainer().fit(litcb, loader)

    # Frozen / trained checks BEFORE exporting (to_numpy_net's .double() mutates net).
    for name, param in net.named_parameters():
        if name.startswith(("trunk", "policy", "value")):
            assert torch.allclose(param, before[name])  # frozen
    assert not torch.allclose(net.cb1.weight, before["cb1.weight"])
    assert not torch.allclose(net.cb_lstm.weight_ih, before["cb_lstm.weight_ih"])

    out = build_deck(net.double().to_numpy_net(), pool, FEATS)
    assert deck_is_legal(out, pool)
    assert {10, 11, 20} <= set(out)  # the LSTM head learned the demo's distinct cards


def test_cb_rl_samples_advantage_and_legal() -> None:
    pool = _pool()
    good = [10, 11, 20, 30, 2, 2]  # a deck that "won" -> positive advantage
    bad = [2, 2, 2, 10, 11, 20]  # a deck that "lost" -> negative advantage
    card_feats, samples = cb_rl_samples(
        [(good, 1.0), (bad, -1.0)], pool, FEATS, normalize=True,
    )
    assert card_feats.shape[0] == len(pool.ids())
    # returns [+1, -1] -> baseline 0, std 1 -> advantages [+1, -1] (shared per deck).
    weights = [s.weight for s in samples]
    assert any(w > 0 for w in weights)  # good-deck steps
    assert any(w < 0 for w in weights)  # bad-deck steps
    # every logged pick is legal at its step (masks recomputed from the prefix).
    for s in samples:
        assert s.legal_mask[s.target_idx]


# --- end-to-end: train -> export numpy -> NetAgent legal selection ---------

def test_train_export_and_netagent_legal(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    samples = [
        _sample(rng.standard_normal(STATE_DIM), rng.standard_normal((3, OPTION_DIM)),
                int(rng.integers(3)), float(rng.choice([-1.0, 1.0])))
        for _ in range(64)
    ]
    loader = DataLoader(
        PolicyDataset(samples), batch_size=16, collate_fn=collate_policy,
    )
    lit = LitPolicyValue(lr=0.05)
    L.Trainer(
        max_epochs=2, accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False,
    ).fit(lit, loader)

    path = tmp_path / "bc.npz"
    lit.net.double().to_numpy_net().save(path)

    agent = NetAgent([10, 11] * 30, engine=ENGINE, weights=path)
    obs = {
        "select": {
            "maxCount": 1, "minCount": 1,
            "option": [{"type": 13, "attackId": 101}, {"type": 14}],
        },
        "current": {"yourIndex": 0, "players": [{}, {}]},
    }
    assert selection_is_legal(agent(obs), obs["select"])
