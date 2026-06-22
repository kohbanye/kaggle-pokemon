"""Integration: synthetic episodes -> collate -> one V-Trace/PPO learner step.

No engine needed -- hand-built episodes with correct shapes drive
:class:`~src.net.lit_vtrace.LitVtracePPO` so the whole stage-3/4 wiring (trajectory
collation, recurrent forward, V-Trace targets, PPO surrogate over both arms, shared
embedding) is exercised and shown to produce a finite loss with gradients reaching
both heads + the shared card embedding.
"""

from __future__ import annotations

import numpy as np
import torch

from src.net.encode import OPTION_DIM, SLOT_MAX, STATE_DIM, STATE_EMBED_SLOTS
from src.net.features import CARD_FEAT_DIM
from src.net.lit_vtrace import LitVtracePPO
from src.net.recurrent_model import RecurrentNetConfig
from src.net.recurrent_torch import TorchRecurrentNet
from src.net.trajectory_data import BattleStep, Episode, collate_episodes

_N_POOL = 10
_CFG = RecurrentNetConfig(n_cards=_N_POOL - 1, play_lstm_hidden=16, hidden=12)


def _battle_step(rng: np.random.Generator, k: int) -> BattleStep:
    return BattleStep(
        state=rng.standard_normal(STATE_DIM),
        state_rows=rng.integers(0, _N_POOL, (STATE_EMBED_SLOTS, SLOT_MAX)).astype(
            np.intp,
        ),
        state_mask=rng.random((STATE_EMBED_SLOTS, SLOT_MAX)) < 0.5,
        options=rng.standard_normal((k, OPTION_DIM)),
        option_rows=rng.integers(0, _N_POOL, k).astype(np.intp),
        action=int(rng.integers(0, k)),
        behaviour_logp=float(np.log(1.0 / k)),  # uniform behaviour policy
    )


def _episode(rng: np.random.Generator) -> Episode:
    t_battle = int(rng.integers(2, 6))
    t_deck = int(rng.integers(2, 5))
    rows = rng.integers(0, _N_POOL, t_deck).astype(np.int64)
    legal = rng.random((t_deck, _N_POOL)) < 0.7
    for t in range(t_deck):
        legal[t, rows[t]] = True  # the picked card is always legal
    return Episode(
        battle=[_battle_step(rng, int(rng.integers(2, 5))) for _ in range(t_battle)],
        deck_rows=rows,
        deck_legal=legal,
        deck_logp=np.log(rng.uniform(0.1, 0.9, t_deck)),
        ret=float(rng.choice([-1.0, 1.0])),
    )


def test_learner_step_finite_and_grads_flow() -> None:
    """One training step yields a finite loss; grads reach both arms + embedding."""
    rng = np.random.default_rng(0)
    episodes = [_episode(rng) for _ in range(6)]
    battle, deck = collate_episodes(episodes)

    net = TorchRecurrentNet(_CFG)
    card_feats = np.random.default_rng(1).standard_normal((_N_POOL, CARD_FEAT_DIM))
    lit = LitVtracePPO(net, card_feats, lr=1e-2)

    loss = lit.training_step((battle, deck), 0)
    assert torch.isfinite(loss)

    loss.backward()
    # Battle arm reaches the play LSTM + value head; deck arm reaches the deck LSTM;
    # both reach the shared card embedding.
    for name in ("play_lstm.weight_ih", "value_head.weight", "cb_lstm.weight_ih"):
        grad = dict(net.named_parameters())[name].grad
        assert grad is not None
        assert torch.isfinite(grad).all()
    embed_grad = net.cb_embed.grad
    assert embed_grad is not None
    assert embed_grad.abs().sum() > 0


def test_learner_optimises_on_fixed_batch() -> None:
    """A few Adam steps on a fixed batch keep the loss finite and move the params."""
    rng = np.random.default_rng(2)
    episodes = [_episode(rng) for _ in range(8)]
    batch = collate_episodes(episodes)

    net = TorchRecurrentNet(_CFG)
    card_feats = np.random.default_rng(3).standard_normal((_N_POOL, CARD_FEAT_DIM))
    lit = LitVtracePPO(net, card_feats, lr=5e-3)
    opt = lit.configure_optimizers()

    before = net.cb_embed.detach().clone()
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = lit.training_step(batch, 0)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert all(np.isfinite(losses))
    assert not torch.allclose(before, net.cb_embed)  # the embedding actually trained
