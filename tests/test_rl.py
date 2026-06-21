"""Tests for the Phase-5a OSFP RL pieces (torch side, no ``cg``).

Covers the policy-gradient trainer (advantaged actions get more probable, the
value head regresses to returns, the masked entropy is NaN-safe under padding, the
CB head stays frozen), the RL reinterpretation of the BC sample builder
(learner-tagged decisions only, target = sampled action, value = return), the
stochastic ``NetAgent`` (legal + reproducible, argmax default unchanged), and an
end-to-end loop smoke (``run_osfp`` with a synthetic generator, no Docker).
"""

import sys
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.agents.base import is_legal as selection_is_legal
from src.agents.net_agent import NetAgent
from src.deck import CardInfo, CardPool
from src.net.bc_data import (
    PolicyDataset,
    PolicySample,
    build_policy_samples,
    collate_policy,
)
from src.net.encode import OPTION_DIM, STATE_DIM
from src.net.features import CardFeatures
from src.net.lit import LitPolicyGradient
from src.net.model import NetConfig, PolicyValueNet
from src.net.torch_model import TorchPolicyValueNet

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from train_deck_osfp import DeckOsfpConfig, run_deck_osfp  # noqa: E402
from train_osfp import OsfpConfig, run_osfp  # noqa: E402

# A small hand-written engine dump, exactly as the runner injects it (mirrors
# tests/test_bc.py so the synthetic decisions encode without the real engine).
ENGINE = {
    "cards": {
        10: {"hp": 60, "type": 0, "weak": None, "ex": False, "mega": False,
             "basic": True, "ctype": 0, "retreat": 1, "attacks": [100]},
        11: {"hp": 120, "type": 1, "weak": 0, "ex": True, "mega": False,
             "basic": True, "ctype": 0, "retreat": 2, "attacks": [101]},
    },
    "attacks": {100: {"dmg": 20, "cost": [0]}, 101: {"dmg": 90, "cost": [1, 1]}},
}
FEATS = CardFeatures(ENGINE)


def _decision(slot: int, agent: str, target: int, n_options: int = 2) -> dict:
    """A minimal single-select decision record (END-type options, no card)."""
    options = [{"type": 14} for _ in range(n_options)]
    obs = {
        "select": {"maxCount": 1, "minCount": 1, "option": options},
        "current": {"yourIndex": slot, "players": [{}, {}]},
    }
    return {"slot": slot, "agent": agent, "obs": obs, "choice": [target]}


def _trainer(epochs: int) -> L.Trainer:
    return L.Trainer(
        max_epochs=epochs, accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False,
    )


def _mean_logp_taken(
    net: TorchPolicyValueNet,
    batch: tuple,
) -> float:
    states, options, mask, targets, _ = batch
    with torch.no_grad():
        logits = net.policy_logits(states, options).masked_fill(~mask, float("-inf"))
        logp = F.log_softmax(logits, dim=1)
        return logp.gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()


def _select_obs(n_options: int = 3) -> dict:
    return {
        "select": {
            "maxCount": 1, "minCount": 1,
            "option": [{"type": 14} for _ in range(n_options)],
        },
        "current": {"yourIndex": 0, "players": [{}, {}]},
    }


# --- policy-gradient trainer -----------------------------------------------

def _win_samples(rng: np.random.Generator, n: int = 16) -> list[PolicySample]:
    """``n`` winning (return +1) decisions that all took option 0 (K = 3)."""
    return [
        PolicySample(
            rng.standard_normal(STATE_DIM),
            rng.standard_normal((3, OPTION_DIM)),
            0,
            1.0,
        )
        for _ in range(n)
    ]


def test_pg_advantaged_action_more_probable() -> None:
    torch.manual_seed(0)
    net = TorchPolicyValueNet()
    samples = _win_samples(np.random.default_rng(0))
    batch = collate_policy(samples)
    first = _mean_logp_taken(net, batch)

    # Isolate the policy gradient (no value/entropy coupling on the shared trunk):
    # every sample is a win that took action 0, so action 0 must get more probable.
    lit = LitPolicyGradient(net, lr=0.05, value_coef=0.0, entropy_coef=0.0)
    loader = DataLoader(PolicyDataset(samples), batch_size=8, collate_fn=collate_policy)
    _trainer(40).fit(lit, loader)

    assert _mean_logp_taken(net, batch) > first


def test_pg_value_regresses_to_returns() -> None:
    torch.manual_seed(0)
    net = TorchPolicyValueNet()
    samples = _win_samples(np.random.default_rng(1))
    states = collate_policy(samples)[0]
    before = net.value(states).mean().item()

    lit = LitPolicyGradient(net, lr=0.02, value_coef=1.0, entropy_coef=0.0)
    loader = DataLoader(PolicyDataset(samples), batch_size=8, collate_fn=collate_policy)
    _trainer(40).fit(lit, loader)

    assert net.value(states).mean().item() > before  # value head learns +1 returns


def test_pg_entropy_no_nan_under_padding() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = TorchPolicyValueNet()
    # Variable option counts -> padded options -> masked -inf logits, the case the
    # entropy term must not turn into nan (0 * -inf).
    samples = [
        PolicySample(
            rng.standard_normal(STATE_DIM),
            rng.standard_normal((k, OPTION_DIM)),
            0,
            1.0,
        )
        for k in (1, 2, 3, 1, 2)
    ]
    loader = DataLoader(PolicyDataset(samples), batch_size=5, collate_fn=collate_policy)
    lit = LitPolicyGradient(net, lr=0.05, entropy_coef=0.1)
    _trainer(2).fit(lit, loader)
    for param in net.parameters():
        assert torch.isfinite(param).all()


def test_pg_freezes_cb_head() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = TorchPolicyValueNet()
    before = {name: p.detach().clone() for name, p in net.named_parameters()}
    samples = [
        PolicySample(
            rng.standard_normal(STATE_DIM), rng.standard_normal((3, OPTION_DIM)),
            int(rng.integers(3)), float(rng.choice([-1.0, 1.0])),
        )
        for _ in range(16)
    ]
    loader = DataLoader(PolicyDataset(samples), batch_size=8, collate_fn=collate_policy)
    _trainer(3).fit(LitPolicyGradient(net, lr=0.05), loader)

    assert torch.allclose(net.cb1.weight, before["cb1.weight"])  # CB head frozen
    assert torch.allclose(net.cb2.weight, before["cb2.weight"])
    assert torch.allclose(net.cb_embed, before["cb_embed"])  # card embedding frozen too
    assert not torch.allclose(net.value_head.weight, before["value_head.weight"])
    assert not torch.allclose(net.policy2.weight, before["policy2.weight"])


# --- RL reinterpretation of the BC sample builder --------------------------

def test_build_samples_selects_learner_only() -> None:
    games = [{"winner": 0, "decisions": [
        _decision(0, "learner", 1),
        _decision(1, "opp", 0),
    ]}]
    samples = build_policy_samples(games, FEATS, teachers={"learner"})
    assert len(samples) == 1  # the opponent's decision is environment, not trained on
    assert samples[0].target == 1  # the action actually sampled
    assert samples[0].value == 1.0  # slot 0 won -> return +1 (gamma = 1)


# --- stochastic NetAgent ---------------------------------------------------

def test_netagent_argmax_default_unchanged() -> None:
    net = PolicyValueNet.random(np.random.default_rng(0))
    agent = NetAgent([10, 11] * 30, ENGINE, net=net)  # temperature 0 = argmax
    obs = _select_obs()
    first = agent(obs)
    assert agent(obs) == first  # deterministic
    assert selection_is_legal(first, obs["select"])


def test_netagent_temperature_legal_and_explores() -> None:
    net = PolicyValueNet.random(np.random.default_rng(0))
    agent = NetAgent([10, 11] * 30, ENGINE, net=net, temperature=1.0, seed=0)
    obs = _select_obs()
    picks = [tuple(agent(obs)) for _ in range(100)]
    assert all(selection_is_legal(list(p), obs["select"]) for p in picks)
    assert len(set(picks)) >= 2  # sampling explores >1 option


def test_netagent_reset_reproducible() -> None:
    net = PolicyValueNet.random(np.random.default_rng(0))
    agent = NetAgent([10, 11] * 30, ENGINE, net=net, temperature=1.0)
    obs = _select_obs()
    agent.reset(123)
    seq1 = [tuple(agent(obs)) for _ in range(20)]
    agent.reset(123)
    seq2 = [tuple(agent(obs)) for _ in range(20)]
    assert seq1 == seq2


# --- end-to-end: train -> export -> NetAgent, and the OSFP loop ------------

def test_train_export_and_netagent_legal(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    games = [{"winner": w, "decisions": [
        _decision(0, "learner", i % 2) for i in range(8)
    ] + [_decision(1, "opp", 0)]} for w in (0, 1)]
    samples = build_policy_samples(games, FEATS, teachers={"learner"})
    loader = DataLoader(PolicyDataset(samples), batch_size=4, collate_fn=collate_policy)

    net = TorchPolicyValueNet()
    _trainer(2).fit(LitPolicyGradient(net, lr=0.05), loader)
    path = tmp_path / "osfp.npz"
    net.double().to_numpy_net().save(path)

    agent = NetAgent([10, 11] * 30, engine=ENGINE, weights=path)
    obs = {
        "select": {
            "maxCount": 1, "minCount": 1,
            "option": [{"type": 13, "attackId": 101}, {"type": 14}],
        },
        "current": {"yourIndex": 0, "players": [{}, {}]},
    }
    assert selection_is_legal(agent(obs), obs["select"])


def test_run_osfp_loop_with_fake_generator(tmp_path) -> None:  # noqa: ANN001 - fixture
    bc_path = tmp_path / "bc.npz"
    PolicyValueNet.random(np.random.default_rng(0)).save(bc_path)

    def fake_generate(_spec) -> list[dict]:  # noqa: ANN001 - injected stub
        return [
            {"winner": w, "decisions": [
                _decision(0, "learner", 0),
                _decision(0, "learner", 1),
                _decision(1, "opp", 0),
            ]}
            for w in (0, 1)
        ]

    cfg = OsfpConfig(
        deck=tmp_path / "unused.csv", bc_weights=bc_path, iter_dir=tmp_path / "osfp",
        baselines=["greedy"], iterations=3, games_per_iter=4, batch_size=4,
        eval_every=100, self_play_prob=0.0, patience=1,
    )
    result = run_osfp(cfg, FEATS, generate=fake_generate, evaluate=None)

    assert len(result.iterations) == 3
    assert result.final_weights.exists()
    assert result.pool.num_checkpoints == 3  # patience=1 admits every iteration
    assert all(s.n_samples == 4 for s in result.iterations)  # 2 learner moves x 2 games

    agent = NetAgent([10, 11] * 30, engine=ENGINE, weights=result.final_weights)
    obs = _select_obs()
    assert selection_is_legal(agent(obs), obs["select"])


# --- deck self-play OSFP loop ---------------------------------------------

def _cb_pool() -> CardPool:
    def info(cid: int, name: str, *, bp: bool = False, be: bool = False,
             ace: bool = False) -> CardInfo:
        return CardInfo(cid, name, "", "", bp, be, ace)
    return CardPool({
        10: info(10, "A", bp=True), 11: info(11, "B", bp=True),
        20: info(20, "Item"), 30: info(30, "Ace", ace=True), 2: info(2, "E", be=True),
    })


def test_run_deck_osfp_loop_with_fake_generator(tmp_path) -> None:  # noqa: ANN001
    pool = _cb_pool()
    bc = tmp_path / "bc.npz"
    PolicyValueNet.random(
        np.random.default_rng(0), NetConfig(n_cards=len(pool.ids())),
    ).save(bc)

    # Decks of varied strength -> the deck self-play advantage is non-zero (the fix
    # for the vs-fixed design where every deck lost). The fake generator stands in
    # for the Docker deck-self-play collector (opponent decks drawn from the net).
    def fake_gen(_spec) -> list[dict]:  # noqa: ANN001 - injected stub
        return [
            {"deck": [10, 11, 20, 30, 2, 2], "wins": w, "losses": lo, "draws": 0}
            for w, lo in [(8, 2), (2, 8), (5, 5), (7, 3)]
        ]

    cfg = DeckOsfpConfig(
        init_weights=bc, iter_dir=tmp_path / "deck", gate_deck=tmp_path / "x.csv",
        iterations=2, decks_per_iter=4, games_per_deck=10, batch_size=8,
        eval_every=100, seed=0,
    )
    result = run_deck_osfp(cfg, pool, FEATS, generate=fake_gen, evaluate=None)
    assert len(result.iterations) == 2
    assert result.final_weights.exists()
    assert all(s.n_samples > 0 for s in result.iterations)
