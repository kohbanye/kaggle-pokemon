"""Parity: the torch recurrent sequence forward == the numpy stateful forward.

Mirrors ``tests/test_net_torch.py`` for the recurrent net. If these diverge,
"train in torch, serve in numpy" is unsafe -- so we pin the play-LSTM value /
policy logits step-by-step, plus the save/load round-trip.
"""

from __future__ import annotations

import numpy as np
import torch

from src.net.encode import OPTION_DIM, SLOT_MAX, STATE_DIM, STATE_EMBED_SLOTS
from src.net.recurrent_model import RecurrentNetConfig, RecurrentPolicyValueNet
from src.net.recurrent_torch import TorchRecurrentNet

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
