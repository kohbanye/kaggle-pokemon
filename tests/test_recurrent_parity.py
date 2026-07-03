"""Parity: the torch recurrent sequence forward == the numpy stateful forward.

Mirrors ``tests/test_net_torch.py`` for the recurrent net. If these diverge,
"train in torch, serve in numpy" is unsafe -- so we pin the play-LSTM value /
policy logits step-by-step, plus the save/load round-trip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from src.net.encode import OPTION_DIM, SLOT_MAX, STATE_DIM, STATE_EMBED_SLOTS
from src.net.recurrent_model import RecurrentNetConfig, RecurrentPolicyValueNet
from src.net.recurrent_torch import TorchRecurrentNet

if TYPE_CHECKING:
    from pathlib import Path

# Small widths so the test is fast but exercises every dim (incl. a real pool).
_CFG = RecurrentNetConfig(n_cards=12, play_lstm_hidden=24, hidden=16)
_N_POOL = _CFG.n_cards + 1  # UNK row included


def _random_step_inputs(
    rng: np.random.Generator,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One decision's encoded inputs (state, slot rows/mask, options, option rows)."""
    state = rng.standard_normal(STATE_DIM)
    rows = rng.integers(0, _N_POOL, size=(STATE_EMBED_SLOTS, SLOT_MAX)).astype(np.intp)
    mask = rng.random((STATE_EMBED_SLOTS, SLOT_MAX)) < 0.6
    options = rng.standard_normal((k, OPTION_DIM))
    option_rows = rng.integers(0, _N_POOL, size=k).astype(np.intp)
    return state, rows, mask, options, option_rows


def test_recurrent_play_parity() -> None:
    """numpy stateful step == torch play_sequence, for every step of a trajectory."""
    rng = np.random.default_rng(0)
    net = RecurrentPolicyValueNet.random(rng, _CFG)
    # float64 *before* loading so the bridge is exact (matches test_net_torch).
    torch_net = TorchRecurrentNet(_CFG).double().eval()
    torch_net.load_numpy_params(net.params)

    t_len, k = 6, 5
    steps = [_random_step_inputs(rng, k) for _ in range(t_len)]

    # numpy: carry (h, c) across steps.
    h, c = net.initial_state()
    np_logits, np_values = [], []
    for state, rows, mask, options, option_rows in steps:
        logits, value, h, c = net.step(state, rows, mask, options, option_rows, h, c)
        np_logits.append(logits)
        np_values.append(value)

    # torch: one (B=1, T) sequence forward.
    states = torch.tensor(np.stack([s[0] for s in steps]))[None]
    state_rows = torch.tensor(np.stack([s[1] for s in steps]))[None]
    state_mask = torch.tensor(np.stack([s[2] for s in steps]))[None]
    options = torch.tensor(np.stack([s[3] for s in steps]))[None]
    option_rows = torch.tensor(np.stack([s[4] for s in steps]))[None]
    with torch.no_grad():
        t_logits, t_values = torch_net.play_sequence(
            states, state_rows, state_mask, options, option_rows,
        )

    for t in range(t_len):
        np.testing.assert_allclose(np_logits[t], t_logits[0, t].numpy(), atol=1e-9)
        np.testing.assert_allclose(np_values[t], float(t_values[0, t]), atol=1e-9)


def test_cat_head_parity() -> None:
    """The factored deck category head bridges + forwards identically in numpy/torch."""
    rng = np.random.default_rng(7)
    net = RecurrentPolicyValueNet.random(rng, _CFG)
    torch_net = TorchRecurrentNet(_CFG).double().eval()
    torch_net.load_numpy_params(net.params)
    h = rng.standard_normal(_CFG.lstm_hidden)
    with torch.no_grad():
        t_cat = torch_net.cat_head(torch.tensor(h)).numpy()
    np.testing.assert_allclose(net.cat_logits(h), t_cat, atol=1e-9)
    # Weight bridge round-trips the category head.
    back = torch_net.to_numpy_net()
    np.testing.assert_allclose(back.params["cat_w"], net.params["cat_w"], atol=1e-9)
    np.testing.assert_allclose(back.params["cat_b"], net.params["cat_b"], atol=1e-9)


def test_recurrent_save_load_roundtrip(tmp_path: object) -> None:
    """A saved recurrent net reloads to identical weights + config."""
    rng = np.random.default_rng(1)
    net = RecurrentPolicyValueNet.random(rng, _CFG)
    path = f"{tmp_path}/recur.npz"
    net.save(path)
    loaded = RecurrentPolicyValueNet.load(path)
    assert loaded.config == _CFG
    for key, value in net.params.items():
        np.testing.assert_array_equal(value, loaded.params[key])


# --- deck conditioning (deck_ctx_dim > 0) ------------------------------------

_CTX_CFG = RecurrentNetConfig(
    n_cards=12, play_lstm_hidden=24, hidden=16, deck_ctx_dim=6, deck_feat_dim=10,
)


def test_deck_ctx_parity() -> None:
    """Deck-conditioned numpy step == torch play_sequence with deck_ctx."""
    rng = np.random.default_rng(1)
    net = RecurrentPolicyValueNet.random(rng, _CTX_CFG)
    torch_net = TorchRecurrentNet(_CTX_CFG).double().eval()
    torch_net.load_numpy_params(net.params)

    deck_vec = rng.standard_normal(_CTX_CFG.deck_feat_dim)
    ctx = net.deck_ctx(deck_vec)
    assert ctx is not None
    assert ctx.shape == (_CTX_CFG.deck_ctx_dim,)

    t_len, k = 5, 4
    steps = [_random_step_inputs(rng, k) for _ in range(t_len)]
    h, c = net.initial_state()
    np_logits, np_values = [], []
    for state, rows, mask, options, option_rows in steps:
        logits, value, h, c = net.step(
            state, rows, mask, options, option_rows, h, c, ctx=ctx,
        )
        np_logits.append(logits)
        np_values.append(value)

    states = torch.tensor(np.stack([s[0] for s in steps]))[None]
    state_rows = torch.tensor(np.stack([s[1] for s in steps]))[None]
    state_mask = torch.tensor(np.stack([s[2] for s in steps]))[None]
    options = torch.tensor(np.stack([s[3] for s in steps]))[None]
    option_rows = torch.tensor(np.stack([s[4] for s in steps]))[None]
    with torch.no_grad():
        t_ctx = torch_net.deck_ctx(torch.tensor(deck_vec)[None])
        np.testing.assert_allclose(ctx, t_ctx[0].numpy(), atol=1e-12)
        t_logits, t_values = torch_net.play_sequence(
            states, state_rows, state_mask, options, option_rows, deck_ctx=t_ctx,
        )
    for t in range(t_len):
        np.testing.assert_allclose(np_logits[t], t_logits[0, t].numpy(), atol=1e-9)
        np.testing.assert_allclose(np_values[t], float(t_values[0, t]), atol=1e-9)


def test_enable_deck_ctx_preserves_behaviour_and_roundtrips(
    tmp_path: Path,
) -> None:
    """Migrating an unconditioned net is output-identical; save/load round-trips."""
    rng = np.random.default_rng(2)
    net = RecurrentPolicyValueNet.random(rng, _CFG)
    mig = net.enable_deck_ctx(rng, ctx_dim=6, feat_dim=10)
    assert mig.config.deck_ctx_dim == 6

    deck_vec = rng.standard_normal(10)
    ctx = mig.deck_ctx(deck_vec)
    state, rows, mask, options, option_rows = _random_step_inputs(rng, 4)
    h0, c0 = net.initial_state()
    base = net.step(state, rows, mask, options, option_rows, h0, c0)
    cond = mig.step(state, rows, mask, options, option_rows, h0, c0, ctx=ctx)
    np.testing.assert_allclose(base[0], cond[0], atol=1e-12)  # logits identical
    np.testing.assert_allclose(base[1], cond[1], atol=1e-12)  # value identical

    path = tmp_path / "ctx.npz"
    mig.save(path)
    loaded = RecurrentPolicyValueNet.load(path)
    assert loaded.config.deck_ctx_dim == 6
    assert loaded.config.deck_feat_dim == 10
    re = loaded.step(state, rows, mask, options, option_rows, h0, c0,
                     ctx=loaded.deck_ctx(deck_vec))
    np.testing.assert_allclose(base[0], re[0], atol=1e-12)
