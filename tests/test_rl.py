"""Tests for the Phase-5d joint OSFP RL pieces (torch side, no ``cg``).

Covers the joint policy-gradient trainer (advantaged play actions get more
probable, the value head regresses to returns, masked entropy is NaN-safe under
padding, and -- the whole point of Phase 5d -- a joint play+deck update trains
*every* head AND the shared card embedding), the RL reinterpretation of the BC
sample builder (learner-tagged decisions only, target = sampled action, value =
return), the shared-embedding row encoders, the stochastic ``NetAgent``, and an
end-to-end loop smoke (``run_joint_osfp`` with a synthetic generator, no Docker).
"""

import sys
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.agents.base import is_legal as selection_is_legal
from src.agents.net_agent import NetAgent
from src.deck import CardInfo, CardPool
from src.net.bc_data import (
    CBSequenceDataset,
    PolicyDataset,
    PolicySample,
    build_policy_samples,
    cb_rl_sequences,
    collate_cb_seq,
    collate_policy,
)
from src.net.embedding import CardEmbeddingIndex
from src.net.encode import (
    OPTION_DIM,
    SLOT_MAX,
    STATE_DIM,
    STATE_EMBED_SLOTS,
    option_embed_rows,
    state_embed_rows,
)
from src.net.features import CardFeatures
from src.net.lit import LitJointPolicyGradient
from src.net.model import NetConfig, PolicyValueNet
from src.net.torch_model import TorchPolicyValueNet

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from train_joint_osfp import JointOsfpConfig, _split, run_joint_osfp  # noqa: E402

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


def _play_sample(
    rng: np.random.Generator,
    *,
    k: int = 3,
    target: int = 0,
    value: float = 1.0,
    n_cards: int = 0,
) -> PolicySample:
    """A synthetic play sample; embedding rows reference real rows when n_cards>0."""
    rows = np.zeros((STATE_EMBED_SLOTS, SLOT_MAX), dtype=np.intp)
    mask = np.zeros((STATE_EMBED_SLOTS, SLOT_MAX), dtype=np.bool_)
    opt_rows = np.zeros(k, dtype=np.intp)
    if n_cards > 0:
        rows[0, 0] = int(rng.integers(n_cards))
        mask[0, 0] = True  # one real board card -> state embedding is exercised
        opt_rows = rng.integers(0, n_cards, size=k).astype(np.intp)
    return PolicySample(
        rng.standard_normal(STATE_DIM), rows, mask,
        rng.standard_normal((k, OPTION_DIM)), opt_rows, target, value,
    )


def _fit_joint(  # noqa: PLR0913 - a test helper threading the joint trainer's knobs
    net: TorchPolicyValueNet,
    play_samples: list,
    card_feats: np.ndarray,
    deck_seqs: list,
    *,
    epochs: int,
    lr: float,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> None:
    loaders: dict[str, DataLoader] = {}
    if play_samples:
        loaders["play"] = DataLoader(
            PolicyDataset(play_samples), batch_size=8, collate_fn=collate_policy,
        )
    if deck_seqs:
        loaders["deck"] = DataLoader(
            CBSequenceDataset(deck_seqs), batch_size=8, collate_fn=collate_cb_seq,
        )
    lit = LitJointPolicyGradient(
        net, card_feats, lr=lr, value_coef=value_coef, entropy_coef=entropy_coef,
    )
    _trainer(epochs).fit(lit, CombinedLoader(loaders, mode="max_size_cycle"))


def _mean_logp_taken(net: TorchPolicyValueNet, batch: tuple) -> float:
    states, state_rows, state_mask, options, mask, option_rows, targets, _ = batch
    with torch.no_grad():
        logits = net.policy_logits(
            states, state_rows, state_mask, options, option_rows,
        ).masked_fill(~mask, float("-inf"))
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


_DUMMY_CARD_FEATS = np.zeros((1, FEATS.vector(None).shape[0]), dtype=np.float64)


# --- joint policy-gradient trainer (play arm) ------------------------------

def _win_samples(rng: np.random.Generator, n: int = 16) -> list[PolicySample]:
    """``n`` winning (return +1) decisions that all took option 0 (K = 3)."""
    return [_play_sample(rng, target=0, value=1.0) for _ in range(n)]


def test_joint_play_arm_advantages_action() -> None:
    torch.manual_seed(0)
    net = TorchPolicyValueNet()
    samples = _win_samples(np.random.default_rng(0))
    batch = collate_policy(samples)
    first = _mean_logp_taken(net, batch)

    # Isolate the policy gradient (no value/entropy coupling on the shared trunk):
    # every sample is a win that took action 0, so action 0 must get more probable.
    _fit_joint(
        net, samples, _DUMMY_CARD_FEATS, [], epochs=40, lr=0.05,
        value_coef=0.0, entropy_coef=0.0,
    )
    assert _mean_logp_taken(net, batch) > first


def test_joint_value_regresses_to_returns() -> None:
    torch.manual_seed(0)
    net = TorchPolicyValueNet()
    samples = _win_samples(np.random.default_rng(1))
    states, state_rows, state_mask = collate_policy(samples)[:3]
    before = net.value(states, state_rows, state_mask).mean().item()

    _fit_joint(
        net, samples, _DUMMY_CARD_FEATS, [], epochs=40, lr=0.02,
        value_coef=1.0, entropy_coef=0.0,
    )
    after = net.value(states, state_rows, state_mask).mean().item()
    assert after > before  # value head learns the +1 returns


def test_joint_entropy_no_nan_under_padding() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = TorchPolicyValueNet()
    # Variable option counts -> padded options -> masked -inf logits, the case the
    # entropy term must not turn into nan (0 * -inf).
    samples = [_play_sample(rng, k=k) for k in (1, 2, 3, 1, 2)]
    _fit_joint(net, samples, _DUMMY_CARD_FEATS, [], epochs=2, lr=0.05, entropy_coef=0.1)
    for param in net.parameters():
        assert torch.isfinite(param).all()


def _cb_pool() -> CardPool:
    def info(cid: int, name: str, *, bp: bool = False, be: bool = False,
             ace: bool = False) -> CardInfo:
        return CardInfo(cid, name, "", "", bp, be, ace)
    return CardPool({
        10: info(10, "A", bp=True), 11: info(11, "B", bp=True),
        20: info(20, "Item"), 30: info(30, "Ace", ace=True), 2: info(2, "E", be=True),
    })


def test_joint_update_trains_all_heads_and_shared_embedding() -> None:
    # The Phase-5d contract: one play+deck update moves the play head, the value
    # head, the deck (CB) head AND the shared card embedding -- nothing is frozen,
    # and the embedding gets gradient from *both* arms.
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    pool = _cb_pool()
    n_cards = len(pool.ids())
    net = TorchPolicyValueNet(NetConfig(n_cards=n_cards))
    before = {name: p.detach().clone() for name, p in net.named_parameters()}

    play = [
        _play_sample(rng, target=int(rng.integers(3)),
                     value=float(rng.choice([-1.0, 1.0])), n_cards=n_cards)
        for _ in range(16)
    ]
    scored = [([10, 11, 20, 30, 2, 2], r) for r in (0.8, -0.6, 0.2, -0.4)]
    card_feats, deck_seqs = cb_rl_sequences(scored, pool, FEATS, normalize=True)
    _fit_joint(net, play, card_feats, deck_seqs, epochs=4, lr=0.05)

    changed = {
        name for name, p in net.named_parameters()
        if not torch.allclose(p, before[name])
    }
    # Play head, value head, CB head, LSTM and the shared embedding all updated.
    for name in ("policy2.weight", "value_head.weight", "cb1.weight", "cb_embed"):
        assert name in changed, f"{name} did not update in the joint step"


# --- RL reinterpretation of the BC sample builder --------------------------

def test_build_samples_selects_learner_only() -> None:
    index = CardEmbeddingIndex(_cb_pool())
    games = [{"winner": 0, "decisions": [
        _decision(0, "learner", 1),
        _decision(1, "opp", 0),
    ]}]
    samples = build_policy_samples(games, FEATS, index, teachers={"learner"})
    assert len(samples) == 1  # the opponent's decision is environment, not trained on
    assert samples[0].target == 1  # the action actually sampled
    assert samples[0].value == 1.0  # slot 0 won -> return +1 (gamma = 1)


# --- shared-embedding row encoders -----------------------------------------

def test_state_embed_rows_active_and_bench() -> None:
    index = CardEmbeddingIndex(_cb_pool())
    current = {
        "yourIndex": 0,
        "players": [
            {"active": [{"id": 10}], "bench": [{"id": 11}, {"id": 20}]},
            {"active": [{"id": 30}], "bench": []},
        ],
    }
    rows, mask = state_embed_rows(current, 0, index)
    assert rows.shape == (STATE_EMBED_SLOTS, SLOT_MAX)
    assert mask[0, 0]
    assert rows[0, 0] == index.row(10)  # my active
    assert mask[1, 0]
    assert rows[1, 0] == index.row(30)  # opp active
    assert mask[2, :2].all()  # my bench: first two slots filled
    assert not mask[2, 2:].any()  # ...and exactly two cards
    assert not mask[3].any()  # opp bench empty


def test_embed_rows_none_index_is_unk() -> None:
    _rows, mask = state_embed_rows({"players": [{}, {}]}, 0, None)
    assert not mask.any()  # no index -> every slot empty (UNK / zero contribution)
    opt = option_embed_rows([{"type": 14}, {"type": 14}], None, 0, None)
    assert opt.tolist() == [0, 0]


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


# --- end-to-end: the joint OSFP loop ---------------------------------------

def test_run_joint_osfp_loop_with_fake_generator(tmp_path) -> None:  # noqa: ANN001
    pool = _cb_pool()
    bc = tmp_path / "bc.npz"
    PolicyValueNet.random(
        np.random.default_rng(0), NetConfig(n_cards=len(pool.ids())),
    ).save(bc)

    # The fake generator stands in for the Docker/native joint collector: it returns
    # mixed "game" lines (play arm) and "deck" lines of varied strength (deck arm).
    def fake_gen(_spec) -> list[dict]:  # noqa: ANN001 - injected stub
        games = [
            {"type": "game", "winner": w, "decisions": [
                _decision(0, "learner", 0),
                _decision(0, "learner", 1),
                _decision(1, "opp", 0),
            ]}
            for w in (0, 1)
        ]
        decks = [
            {"type": "deck", "deck": [10, 11, 20, 30, 2, 2],
             "wins": w, "losses": lo, "draws": 0}
            for w, lo in [(8, 2), (2, 8), (5, 5), (7, 3)]
        ]
        return games + decks

    cfg = JointOsfpConfig(
        init_weights=bc, iter_dir=tmp_path / "joint", gate_deck=tmp_path / "x.csv",
        iterations=2, decks_per_iter=4, games_per_deck=10, batch_size=8,
        eval_every=100, self_play_prob=0.0, patience=1, seed=0,
    )
    result = run_joint_osfp(cfg, pool, FEATS, generate=fake_gen, evaluate=None)
    assert len(result.iterations) == 2
    assert result.final_weights.exists()
    assert all(s.n_play_samples > 0 for s in result.iterations)
    assert all(s.n_deck_samples > 0 for s in result.iterations)

    agent = NetAgent([10, 11] * 30, engine=ENGINE, weights=result.final_weights)
    obs = _select_obs()
    assert selection_is_legal(agent(obs), obs["select"])


def test_split_distributes_decks_across_workers() -> None:
    # Parallel collector fans decks_per_iter across workers; chunks near-even,
    # nothing dropped, idle workers when decks < workers.
    assert _split(16, 6) == [3, 3, 3, 3, 2, 2]
    assert sum(_split(17, 5)) == 17
    assert _split(10, 1) == [10]
    assert _split(3, 8).count(0) == 5  # more workers than decks -> some idle
