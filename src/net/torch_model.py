"""Torch mirror of :class:`~src.net.model.PolicyValueNet` (training side).

Training (Phase 4 BC, Phase 5 OSFP) runs in torch + Lightning -- autograd,
optimisers, checkpoints, and multi-GPU later -- while the *submission* keeps the
pure-numpy forward so the agent stays a light, numpy-only bundle (plan SS D).
This module is the bridge between the two: the identical architecture in torch,
plus exact weight conversion to/from the numpy parameter dict. A parity test
(``tests/test_net_torch.py``) asserts the two forwards agree, so they can never
silently diverge -- train here, export with :meth:`to_numpy_net`, serve there.

The forwards are batch-first (``(B, ...)``) to feed the Lightning trainer; the
numpy net's single-sample forwards are the ``B == 1`` case, which is what the
parity test checks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from src.net.model import NetConfig, PolicyValueNet

if TYPE_CHECKING:
    from numpy.typing import NDArray

# numpy stores a dense layer as ``x @ w + b`` with ``w`` shaped ``(in, out)``;
# torch's ``nn.Linear`` stores ``w`` as ``(out, in)`` and computes ``x @ w.T``,
# so the two weight matrices are transposes of each other.


class TorchPolicyValueNet(nn.Module):
    """Same net as :class:`PolicyValueNet`, in torch, with a numpy weight bridge."""

    def __init__(self, config: NetConfig | None = None) -> None:
        super().__init__()
        cfg = config or NetConfig()
        self.config = cfg
        self.trunk1 = nn.Linear(cfg.state_dim, cfg.hidden)
        self.trunk2 = nn.Linear(cfg.hidden, cfg.hidden)
        self.value_head = nn.Linear(cfg.hidden, 1)
        self.policy1 = nn.Linear(cfg.hidden + cfg.option_dim, cfg.policy_hidden)
        self.policy2 = nn.Linear(cfg.policy_hidden, 1)
        self.cb1 = nn.Linear(cfg.card_dim, cfg.cb_hidden)
        self.cb2 = nn.Linear(cfg.cb_hidden, 1)

    # --- forward passes (batch-first) ---------------------------------------

    def trunk(self, states: torch.Tensor) -> torch.Tensor:
        """Shared body: ``(B, state_dim) -> (B, hidden)`` (two ReLU layers)."""
        h = torch.relu(self.trunk1(states))
        return torch.relu(self.trunk2(h))

    def value(self, states: torch.Tensor) -> torch.Tensor:
        """Value in ``[-1, 1]`` per state: ``(B, state_dim) -> (B,)``."""
        return torch.tanh(self.value_head(self.trunk(states))).squeeze(-1)

    def policy_logits(
        self,
        states: torch.Tensor,
        options: torch.Tensor,
    ) -> torch.Tensor:
        """One logit per option: ``(B, state), (B, K, option) -> (B, K)``."""
        h = self.trunk(states)
        k = options.shape[1]
        h_rep = h.unsqueeze(1).expand(-1, k, -1)
        joint = torch.cat([h_rep, options], dim=-1)
        return self.policy2(torch.relu(self.policy1(joint))).squeeze(-1)

    def card_logits(self, card_feats: torch.Tensor) -> torch.Tensor:
        """One logit per candidate card (CB head): ``(N, card_dim) -> (N,)``."""
        return self.cb2(torch.relu(self.cb1(card_feats))).squeeze(-1)

    # --- numpy weight bridge ------------------------------------------------

    @torch.no_grad()
    def load_numpy_params(self, params: dict[str, NDArray[np.float64]]) -> None:
        """Copy weights from a :class:`PolicyValueNet` parameter dict into self."""
        for lin, w, b in self._layer_keys():
            lin.weight.copy_(torch.as_tensor(params[w].T, dtype=lin.weight.dtype))
            lin.bias.copy_(torch.as_tensor(params[b], dtype=lin.bias.dtype))

    @torch.no_grad()
    def to_numpy_params(self) -> dict[str, NDArray[np.float64]]:
        """Export weights as a numpy parameter dict (transposed back, float64)."""
        out: dict[str, NDArray[np.float64]] = {}
        for lin, w, b in self._layer_keys():
            out[w] = lin.weight.detach().cpu().numpy().T.astype(np.float64)
            out[b] = lin.bias.detach().cpu().numpy().astype(np.float64)
        return out

    def to_numpy_net(self) -> PolicyValueNet:
        """A numpy :class:`PolicyValueNet` with this net's weights (for serving)."""
        return PolicyValueNet(self.config, self.to_numpy_params())

    def _layer_keys(self) -> list[tuple[nn.Linear, str, str]]:
        """(layer, weight-key, bias-key) for each dense layer, numpy-dict naming."""
        return [
            (self.trunk1, "trunk_w1", "trunk_b1"),
            (self.trunk2, "trunk_w2", "trunk_b2"),
            (self.value_head, "value_w", "value_b"),
            (self.policy1, "policy_w1", "policy_b1"),
            (self.policy2, "policy_w2", "policy_b2"),
            (self.cb1, "cb_w1", "cb_b1"),
            (self.cb2, "cb_w2", "cb_b2"),
        ]


def from_numpy_net(net: PolicyValueNet) -> TorchPolicyValueNet:
    """Build a torch net initialised from a numpy :class:`PolicyValueNet`."""
    torch_net = TorchPolicyValueNet(net.config)
    torch_net.load_numpy_params(net.params)
    return torch_net
