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


# --- Phase 5c: autoregressive LSTM deck head (sequence training) ------------


def cb_sequence_logits(
    net: TorchPolicyValueNet,
    card_feats: torch.Tensor,
    target_rows: torch.Tensor,
) -> torch.Tensor:
    """CB logits for every step of a batch of deck-build sequences.

    Runs the deck LSTM over the pick order: at step ``t`` the input is the picked
    card of step ``t-1`` (``cb_start`` at ``t=0``), the hidden state ``h_t`` is
    prepended to every candidate's (fixed ⊕ embedding) features, and the CB MLP
    scores the pool. ``card_feats`` is ``(N_pool, card_dim)``; ``target_rows`` is
    ``(B, T)`` the pick-order pool rows. Returns ``(B, T, N_pool)``. The per-step
    loop keeps memory at ``(B, N_pool, ...)`` (never materialises ``(B,T,N,feat)``).
    """
    bsz, t_len = target_rows.shape
    n_pool = card_feats.shape[0]
    card_matrix = torch.cat([card_feats, net.cb_embed[:n_pool]], dim=-1)
    hid = net.cb_lstm.hidden_size
    h = card_feats.new_zeros(bsz, hid)
    c = card_feats.new_zeros(bsz, hid)
    out: list[torch.Tensor] = []
    for t in range(t_len):
        x = (
            net.cb_start.unsqueeze(0).expand(bsz, -1)
            if t == 0
            else net.cb_embed[target_rows[:, t - 1]]
        )
        h, c = net.cb_lstm(x, (h, c))
        joint = torch.cat(
            [
                h.unsqueeze(1).expand(-1, n_pool, -1),
                card_matrix.unsqueeze(0).expand(bsz, -1, -1),
            ],
            dim=-1,
        )
        out.append(net.cb2(torch.relu(net.cb1(joint))).squeeze(-1))
    return torch.stack(out, dim=1)


def _cb_seq_params(net: TorchPolicyValueNet) -> list:
    """CB-head + LSTM + embedding params (trunk/policy/value stay frozen)."""
    return [
        *net.cb_lstm.parameters(),
        net.cb_start,
        *net.cb1.parameters(),
        *net.cb2.parameters(),
        net.cb_embed,
    ]


class LitCBSeq(L.LightningModule):
    """BC trainer for the autoregressive LSTM deck head (Phase 5c).

    A batch is ``(targets (B,T), legal_masks (B,T,N_pool), weights (B,T), valid
    (B,T))`` from :func:`~src.net.bc_data.cb_sequences`. Masked cross-entropy at
    each build step (weighted by the inverse-copy weight), averaged over valid
    steps. Optimises the CB head + LSTM + card embedding; trunk/policy/value frozen.
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
        targets, masks, weights, valid = batch
        logits = cb_sequence_logits(self.net, self.card_feats, targets)
        logits = logits.masked_fill(~masks, _NEG_INF)
        logp = F.log_softmax(logits, dim=-1)
        chosen = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B,T)
        w = weights * valid
        loss = -(chosen * w).sum() / w.sum().clamp(min=1e-8)
        self.log("cb_seq_loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(_cb_seq_params(self.net), lr=self.lr)


class LitCBSeqPolicyGradient(L.LightningModule):
    """REINFORCE trainer for the LSTM deck head on sampled decks (Phase 5b-ii redux).

    Same sequence forward as :class:`LitCBSeq`, but ``weights`` is the deck's
    (signed, normalised) advantage shared across its steps, and the loss is
    ``-(advantage * log pi(pick)).sum() / valid.sum()`` (REINFORCE; divided by the
    valid-step count, NOT by the weight sum -- signed advantages would break that).
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
        targets, masks, weights, valid = batch
        logits = cb_sequence_logits(self.net, self.card_feats, targets)
        logits = logits.masked_fill(~masks, _NEG_INF)
        logp = F.log_softmax(logits, dim=-1)
        chosen = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        loss = -(chosen * weights * valid).sum() / valid.sum().clamp(min=1)
        self.log("cb_seq_pg_loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(_cb_seq_params(self.net), lr=self.lr)
