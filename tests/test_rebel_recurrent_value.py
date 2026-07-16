"""Parity: numpy `RecurrentValueNet` == torch `TorchRecurrentValueNet` (bridge exact).

Same guarantee as the scalar value net's parity test: train in torch, serve the numpy
LSTM forward, pinned so the served leaf value is exactly the trained one.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rebel.recurrent_value_net import RecurrentValueNet

torch = pytest.importorskip("torch")
from src.rebel.recurrent_value_net_torch import (  # noqa: E402
    from_numpy_recurrent,
)

_IN = 12
_HID = 8


def test_numpy_torch_recurrent_parity() -> None:
    rng = np.random.default_rng(0)
    net = RecurrentValueNet.random(rng, in_dim=_IN, hidden=_HID)
    torch_net = from_numpy_recurrent(net)
    for t_len in (1, 3, 7):
        seq = rng.standard_normal((t_len, _IN))
        np_v = net.value(seq)
        with torch.no_grad():
            t_v = float(torch_net(torch.tensor(seq)[None, ...]).item())
        assert abs(np_v - t_v) < 1e-9, f"len {t_len}: numpy {np_v} vs torch {t_v}"


def test_recurrent_value_in_range() -> None:
    rng = np.random.default_rng(1)
    net = RecurrentValueNet.random(rng, in_dim=_IN, hidden=_HID)
    for _ in range(20):
        seq = rng.standard_normal((rng.integers(1, 10), _IN))
        assert -1.0 <= net.value(seq) <= 1.0


def test_recurrent_bridge_roundtrips() -> None:
    rng = np.random.default_rng(2)
    net = RecurrentValueNet.random(rng, in_dim=_IN, hidden=_HID)
    back = from_numpy_recurrent(net).to_numpy_net()
    for k in net.params:
        assert np.allclose(net.params[k], back.params[k], atol=1e-9), k


def test_step_matches_value() -> None:
    """Threading via step() equals the full-sequence value() (search h_t path)."""
    rng = np.random.default_rng(3)
    net = RecurrentValueNet.random(rng, in_dim=_IN, hidden=_HID)
    seq = rng.standard_normal((5, _IN))
    state = net.zero_state()
    for x in seq:
        state = net.step(state, x)
    assert abs(net.value_from_h(state[0]) - net.value(seq)) < 1e-12
