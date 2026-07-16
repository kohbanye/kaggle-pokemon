"""Torch mirror of :class:`VectorValueNet` (train torch, serve numpy), exact bridge."""

from __future__ import annotations

import torch
from torch import nn

from src.rebel.vector_value_net import VectorValueNet


class TorchVectorValueNet(nn.Module):
    """N-layer MLP -> K values in [-1, 1] (tanh)."""

    def __init__(self, in_dim: int, out_dim: int,
                 hidden: tuple[int, ...] = (64,)) -> None:
        super().__init__()
        self.hidden = tuple(hidden)
        dims = [in_dim, *self.hidden, out_dim]
        self.fcs = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for fc in self.fcs[:-1]:
            h = torch.relu(fc(h))
        return torch.tanh(self.fcs[-1](h))

    def to_numpy_net(self) -> VectorValueNet:
        params = {}
        for i, fc in enumerate(self.fcs, start=1):
            params[f"w{i}"] = fc.weight.detach().cpu().numpy().T.astype("float64")
            params[f"b{i}"] = fc.bias.detach().cpu().numpy().astype("float64")
        return VectorValueNet(params)

    def load_numpy(self, net: VectorValueNet) -> None:
        with torch.no_grad():
            for i, fc in enumerate(self.fcs, start=1):
                fc.weight.copy_(torch.tensor(net.params[f"w{i}"].T))
                fc.bias.copy_(torch.tensor(net.params[f"b{i}"]))


def from_numpy_vector(net: VectorValueNet) -> TorchVectorValueNet:
    n = net.n_layers
    hidden = tuple(int(net.params[f"w{i}"].shape[1]) for i in range(1, n))
    torch_net = TorchVectorValueNet(net.in_dim, net.out_dim, hidden).double()
    torch_net.load_numpy(net)
    return torch_net
