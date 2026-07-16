"""Torch mirror of :class:`RecurrentValueNet` (train torch, serve numpy), exact bridge.

nn.LSTM(1 layer) + linear head; the bridge maps torch's stacked (i,f,g,o) gate weights
to the numpy net's ``w_ih/w_hh/b`` and the head to ``hw/hb``, pinned by
``tests/test_rebel_recurrent_value.py``. ty-excluded like the other torch modules.
"""

from __future__ import annotations

import torch
from torch import nn

from src.rebel.recurrent_value_net import RecurrentValueNet


class TorchRecurrentValueNet(nn.Module):
    """LSTM value net in torch: feature sequence -> scalar in [-1, 1]."""

    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, in). Returns (B,) value from the LAST (or length-th) hidden."""
        out, _ = self.lstm(x)  # (B, T, H)
        if lengths is None:
            last = out[:, -1, :]
        else:
            idx = (lengths - 1).clamp(min=0)
            last = out[torch.arange(out.shape[0]), idx]
        return torch.tanh(self.head(last)).squeeze(-1)

    def to_numpy_net(self) -> RecurrentValueNet:
        li = self.lstm
        return RecurrentValueNet({
            "w_ih": li.weight_ih_l0.detach().cpu().numpy().T.astype("float64"),
            "w_hh": li.weight_hh_l0.detach().cpu().numpy().T.astype("float64"),
            "b": (li.bias_ih_l0 + li.bias_hh_l0).detach().cpu()
                 .numpy().astype("float64"),
            "hw": self.head.weight.detach().cpu().numpy().T.astype("float64"),
            "hb": self.head.bias.detach().cpu().numpy().astype("float64"),
        })

    def load_numpy(self, net: RecurrentValueNet) -> None:
        li = self.lstm
        with torch.no_grad():
            li.weight_ih_l0.copy_(torch.tensor(net.params["w_ih"].T))
            li.weight_hh_l0.copy_(torch.tensor(net.params["w_hh"].T))
            li.bias_ih_l0.copy_(torch.tensor(net.params["b"]))
            li.bias_hh_l0.zero_()  # numpy stores the summed bias in b; put it all in ih
            self.head.weight.copy_(torch.tensor(net.params["hw"].T))
            self.head.bias.copy_(torch.tensor(net.params["hb"]))


def from_numpy_recurrent(net: RecurrentValueNet) -> TorchRecurrentValueNet:
    torch_net = TorchRecurrentValueNet(net.in_dim, net.hidden).double()
    torch_net.load_numpy(net)
    return torch_net
