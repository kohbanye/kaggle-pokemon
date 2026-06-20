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
submission never imports torch. The CB (deck) head is trained the same way --
masked CE over candidate cards -- by :class:`LitCB` (below), which wraps the same
net and optimises only the CB layers so the warm-started heads merge into one
export.

:class:`LitPolicyGradient` (below) is the Phase-5 OSFP self-play trainer: it
consumes the *same* 5-tuple batch but reinterprets it -- ``targets`` is the action
actually sampled in self-play and ``values`` is that decision's game return -- and
optimises a REINFORCE policy-gradient with the value head as baseline (plus an
entropy bonus). It freezes the CB head (deck held fixed in the play/value-only
arm), so the BC-warm-started deck weights pass through untouched.
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

    from numpy.typing import NDArray

_NEG_INF = float("-inf")


def cb_loss(
    net: TorchPolicyValueNet,
    card_feats: torch.Tensor,
    legal_mask: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked cross-entropy of the (context-free) CB head against the demo card.

    ``card_feats`` ``(N_pool, card_dim)`` is the whole pool's fixed features (the
    CB head scores it once); ``legal_mask`` ``(B, N_pool)`` restricts each sample
    to its legal next cards (illegal -> -inf, mirroring the deck builder's mask);
    ``target`` ``(B,)`` is the pool index of the demo deck's card at that step.
    ``weights`` ``(B,)`` optionally re-weights each step (the inverse-copy weight
    that stops Basic Energy from dominating -- see :class:`bc_data.CBSample`).
    """
    logits = net.card_logits(card_feats)  # (N_pool,)
    logits = logits.unsqueeze(0).expand(legal_mask.shape[0], -1)  # (B, N_pool)
    logits = logits.masked_fill(~legal_mask, _NEG_INF)
    if weights is None:
        return F.cross_entropy(logits, target)
    losses = F.cross_entropy(logits, target, reduction="none")
    return (losses * weights).sum() / weights.sum()


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

    def cb_loss(
        self,
        card_feats: torch.Tensor,
        legal_mask: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Masked CE of the CB head (delegates to the module-level :func:`cb_loss`)."""
        return cb_loss(self.net, card_feats, legal_mask, target, weights)

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


class LitCB(L.LightningModule):
    """Behaviour-clones the CB (deck) head on the demo decklists.

    Wraps the *same* :class:`TorchPolicyValueNet` as :class:`LitPolicyValue` so
    the two warm-started halves merge into one export, and optimises **only** the
    CB layers (``cb1`` / ``cb2``). The CB head is independent of the trunk
    (``card_logits`` never touches it), so this leaves the policy/value weights
    exactly as trained. A batch is ``(legal_mask (B, N_pool), target (B,))``; the
    fixed pool feature matrix is held as a buffer.
    """

    def __init__(
        self,
        net: TorchPolicyValueNet,
        card_feats: NDArray,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.net = net
        self.register_buffer(
            "card_feats", torch.as_tensor(card_feats, dtype=torch.float32),
        )
        self.lr = lr

    def training_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,  # noqa: ARG002 - required by the Lightning step signature
    ) -> torch.Tensor:
        legal_mask, target, weights = batch
        loss = cb_loss(self.net, self.card_feats, legal_mask, target, weights)
        self.log("cb_loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = [*self.net.cb1.parameters(), *self.net.cb2.parameters()]
        return torch.optim.Adam(params, lr=self.lr)


class LitPolicyGradient(L.LightningModule):
    """OSFP self-play trainer: REINFORCE + value baseline + entropy (Phase 5a).

    Wraps an existing :class:`TorchPolicyValueNet` (warm-started from BC) and
    improves the **play (policy) and value** heads from self-play returns. The
    batch is the same 5-tuple the BC collate produces, reinterpreted for RL:

    - ``targets`` = the option index the behaviour policy **actually sampled**
      (not a teacher's choice);
    - ``values``  = that decision's **return** -- the deciding slot's episodic
      game outcome in ``[-1, 1]`` (``gamma = 1`` -> the raw final result).

    The update is REINFORCE with the value head as a baseline::

        advantage   = (return - V(s)).detach()
        policy_loss = -(advantage * log pi(a|s)).mean()
        value_loss  =  mse(V(s), return)
        loss        =  policy_loss + value_coef * value_loss - entropy_coef * H

    The **CB head is frozen** (only trunk/policy/value are optimised): the deck is
    held fixed in this arm, so the BC-cloned deck weights are exported unchanged.
    PPO clipping / V-Trace are deliberately omitted -- one training pass over
    freshly generated, near-on-policy data makes the importance ratio ~1, so they
    are later ablations rather than MVP machinery.
    """

    def __init__(
        self,
        net: TorchPolicyValueNet,
        *,
        lr: float = 1e-3,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ) -> None:
        super().__init__()
        self.net = net
        self.lr = lr
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def training_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,  # noqa: ARG002 - required by the Lightning step signature
    ) -> torch.Tensor:
        states, options, option_mask, targets, returns = batch
        logits = self.net.policy_logits(states, options)
        logits = logits.masked_fill(~option_mask, _NEG_INF)
        logp = F.log_softmax(logits, dim=1)
        logp_taken = logp.gather(1, targets.unsqueeze(1)).squeeze(1)

        values = self.net.value(states)
        advantage = (returns - values).detach()
        policy_loss = -(advantage * logp_taken).mean()
        value_loss = F.mse_loss(values, returns)

        # Masked entropy. Padded options carry logp = -inf; multiplying that by its
        # probability (0) is 0 * -inf = nan, and -- crucially -- masking the product
        # *after* the multiply still leaves a nan in the backward pass (the mul's
        # grad multiplies the saved -inf input). So zero the -inf out of logp
        # *before* the product: masked probs are already 0, so 0 * 0 = 0 cleanly.
        safe_logp = logp.masked_fill(~option_mask, 0.0)
        entropy = -(logp.exp() * safe_logp).sum(dim=1).mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        self.log_dict(
            {
                "loss": loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
            },
            prog_bar=False,
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Freeze the CB head: train trunk/policy/value only (deck held fixed).
        params = [
            p for name, p in self.net.named_parameters() if not name.startswith("cb")
        ]
        return torch.optim.Adam(params, lr=self.lr)
