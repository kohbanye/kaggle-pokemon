"""Torch mirror of :class:`src.rebel.value_net.ValueNet` (train in torch, serve numpy).

Same 2-layer MLP + tanh, with an exact weight bridge to/from the numpy serving net,
pinned by ``tests/test_rebel_value_net.py``. Trained by the ReBeL self-play loop (F)
on subgame-solve value targets. ty-excluded (nn.Module dynamic attrs), like the other
torch training modules.
"""

from __future__ import annotations

import torch
from torch import nn

from src.rebel.value_net import ValueNet


class TorchValueNet(nn.Module):
    """Leaf value net in torch: features -> scalar in [-1, 1] (N-layer MLP)."""

    def __init__(self, in_dim: int, hidden: int | tuple[int, ...] = (64,)) -> None:
        super().__init__()
        self.hidden = (hidden,) if isinstance(hidden, int) else tuple(hidden)
        dims = [in_dim, *self.hidden, 1]
        self.fcs = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for fc in self.fcs[:-1]:
            h = torch.relu(fc(h))
        return torch.tanh(self.fcs[-1](h)).squeeze(-1)

    def to_numpy_net(self) -> ValueNet:
        params = {}
        for i, fc in enumerate(self.fcs, start=1):
            params[f"w{i}"] = fc.weight.detach().cpu().numpy().T.astype("float64")
            params[f"b{i}"] = fc.bias.detach().cpu().numpy().astype("float64")
        return ValueNet(params)

    def load_numpy(self, net: ValueNet) -> None:
        with torch.no_grad():
            for i, fc in enumerate(self.fcs, start=1):
                fc.weight.copy_(torch.tensor(net.params[f"w{i}"].T))
                fc.bias.copy_(torch.tensor(net.params[f"b{i}"]))


def from_numpy_value_net(
    net: ValueNet, hidden: tuple[int, ...] | None = None,
) -> TorchValueNet:
    if hidden is None:  # infer hidden sizes from the numpy net's weight shapes
        n = net.n_layers
        hidden = tuple(int(net.params[f"w{i}"].shape[1]) for i in range(1, n))
    torch_net = TorchValueNet(net.in_dim, hidden).double()
    torch_net.load_numpy(net)
    return torch_net
