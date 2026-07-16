"""Parity + round-trip for the ReBeL leaf value net (torch train <-> numpy serve).

If the two forwards diverge, "train in torch, serve in numpy" is unsafe -- so we pin the
numpy `ValueNet.value` to torch `TorchValueNet.forward` and the bridge round-trip.
"""

from __future__ import annotations

import numpy as np
import torch

from src.rebel.value_net import VALUE_IN_DIM, ValueNet
from src.rebel.value_net_torch import from_numpy_value_net

_HIDDEN = 16
_IN = 24


def test_numpy_torch_value_parity() -> None:
    rng = np.random.default_rng(0)
    net = ValueNet.random(rng, in_dim=_IN, hidden=_HIDDEN)
    torch_net = from_numpy_value_net(net)
    for _ in range(5):
        x = rng.standard_normal(_IN)
        with torch.no_grad():
            t = float(torch_net(torch.tensor(x)[None]))
        np.testing.assert_allclose(net.value(x), t, atol=1e-9)


def test_value_in_output_range() -> None:
    rng = np.random.default_rng(1)
    net = ValueNet.random(rng, in_dim=_IN, hidden=_HIDDEN)
    for _ in range(20):
        v = net.value(rng.standard_normal(_IN) * 10)
        assert -1.0 <= v <= 1.0


def test_bridge_roundtrips(tmp_path: object) -> None:
    rng = np.random.default_rng(2)
    net = ValueNet.random(rng, in_dim=_IN, hidden=_HIDDEN)
    back = from_numpy_value_net(net).to_numpy_net()
    x = rng.standard_normal(_IN)
    np.testing.assert_allclose(net.value(x), back.value(x), atol=1e-9)
    path = f"{tmp_path}/vn.npz"
    net.save(path)
    assert abs(ValueNet.load(path).value(x) - net.value(x)) < 1e-12


def test_default_in_dim() -> None:
    assert VALUE_IN_DIM == 191 + 40  # STATE_DIM + CARD_FEAT_DIM
