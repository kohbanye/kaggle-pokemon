"""Parity: numpy VectorValueNet == torch TorchVectorValueNet (infostate-vector)."""

from __future__ import annotations

import numpy as np
import pytest

from src.rebel.vector_value_net import VectorValueNet

torch = pytest.importorskip("torch")
from src.rebel.vector_value_net_torch import from_numpy_vector  # noqa: E402

_IN = 12
_K = 5
_HID = (8, 8)


def test_vector_numpy_torch_parity() -> None:
    rng = np.random.default_rng(0)
    net = VectorValueNet.random(rng, out_dim=_K, in_dim=_IN, hidden=_HID)
    torch_net = from_numpy_vector(net)
    for _ in range(10):
        x = rng.standard_normal(_IN)
        np_v = net.values(x)
        with torch.no_grad():
            t_v = torch_net(torch.tensor(x)[None, :]).numpy()[0]
        assert np.allclose(np_v, t_v, atol=1e-9)


def test_vector_out_dim_and_range() -> None:
    rng = np.random.default_rng(1)
    net = VectorValueNet.random(rng, out_dim=_K, in_dim=_IN, hidden=_HID)
    assert net.out_dim == _K
    x = rng.standard_normal(_IN)
    v = net.values(x)
    assert v.shape == (_K,)
    assert np.all((v >= -1.0) & (v <= 1.0))
    assert net.value(x, 2) == float(v[2])  # value(x, k) reads the k-th output


def test_vector_bridge_roundtrips() -> None:
    rng = np.random.default_rng(2)
    net = VectorValueNet.random(rng, out_dim=_K, in_dim=_IN, hidden=_HID)
    back = from_numpy_vector(net).to_numpy_net()
    for k in net.params:
        assert np.allclose(net.params[k], back.params[k], atol=1e-9), k
