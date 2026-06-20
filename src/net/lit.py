"""Lightning module for supervised warm-start (the Phase-4 behaviour-cloning base).

Trains the torch net to (a) imitate a teacher policy -- masked cross-entropy over
the presented options -- and (b) regress the game outcome with the value head.
Variable option counts are handled by padding every sample to ``K`` options and
carrying a 0/1 ``option_mask`` (padded slots are masked to ``-inf`` before the
softmax, so they never receive probability).

A batch is the 5-tuple ``(states, options, option_mask, targets, values)``:

- ``states``      ``(B, state_dim)``      encoded observations
- ``options``     ``(B, K, option_dim)``  encoded presented options (padded)
- ``option_mask`` ``(B, K)`` bool         True for real options, False for padding
- ``targets``     ``(B,)`` long           index of the teacher's chosen option
- ``values``      ``(B,)`` float          outcome target in ``[-1, 1]``

After training, export to the numpy serving net with
``module.net.to_numpy_net().save(path)`` and load it into ``NetAgent`` -- the
submission never imports torch. The CB (deck) head is trained the same way in
Phase 4 (masked CE over candidate cards) and is left out here on purpose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import torch
from torch.nn import functional as F

from src.net.model import NetConfig
from src.net.torch_model import TorchPolicyValueNet

if TYPE_CHECKING:
    from collections.abc import Sequence

_NEG_INF = float("-inf")


class LitPolicyValue(L.LightningModule):
    """Behaviour-cloning + value-regression trainer for the policy/value net."""

    def __init__(
        self,
        config: NetConfig | None = None,
        lr: float = 1e-3,
        value_coef: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.net = TorchPolicyValueNet(config or NetConfig())
        self.lr = lr
        self.value_coef = value_coef

    def policy_loss(
        self,
        states: torch.Tensor,
        options: torch.Tensor,
        option_mask: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Masked cross-entropy of the policy head against the teacher's choice."""
        logits = self.net.policy_logits(states, options)
        logits = logits.masked_fill(~option_mask, _NEG_INF)
        return F.cross_entropy(logits, targets)

    def value_loss(
        self,
        states: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """MSE of the value head against the (discounted) outcome target."""
        return F.mse_loss(self.net.value(states), values)

    def training_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,  # noqa: ARG002 - required by the Lightning step signature
    ) -> torch.Tensor:
        states, options, option_mask, targets, values = batch
        p_loss = self.policy_loss(states, options, option_mask, targets)
        v_loss = self.value_loss(states, values)
        loss = p_loss + self.value_coef * v_loss
        self.log_dict(
            {"loss": loss, "policy_loss": p_loss, "value_loss": v_loss},
            prog_bar=False,
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
