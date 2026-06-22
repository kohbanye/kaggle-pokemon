"""Lightning module for supervised warm-start (the Phase-4 behaviour-cloning base).

Trains the torch net to (a) imitate a teacher policy -- masked cross-entropy over
the presented options -- and (b) regress the game outcome with the value head.
Variable option counts are handled by padding every sample to ``K`` options and
carrying a 0/1 ``option_mask`` (padded slots are masked to ``-inf`` before the
softmax, so they never receive probability).

A batch is the 8-tuple ``(states, state_rows, state_mask, options, option_mask,
option_rows, targets, values)`` from :func:`~src.net.bc_data.collate_policy`:
``states``/``options`` are the fixed encoded features, ``state_rows``/``state_mask``
and ``option_rows`` index the **shared card embedding** for the board's Pokemon and
each option's target card, ``targets`` is the teacher's chosen option, and
``values`` the outcome in ``[-1, 1]``.

After training, export to the numpy serving net with
``module.net.to_numpy_net().save(path)`` and load it into ``NetAgent`` -- the
submission never imports torch. The CB (deck) head is trained the same way --
masked CE over candidate cards -- by :class:`LitCBSeq` (below), which wraps the same
net and optimises only the CB layers so the warm-started heads merge into one
export.

:class:`LitJointPolicyGradient` (below) is the Phase-5d joint OSFP self-play
trainer: it consumes a combined ``{"play", "deck"}`` batch and improves the play,
value and deck heads -- and the **shared card embedding** both heads read -- in one
update (REINFORCE with a value baseline + entropy on the play arm, REINFORCE over
build sequences on the deck arm). Nothing is frozen, so the embedding is trained
jointly by both objectives.
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

    def policy_loss(  # noqa: PLR0913 - the play batch's fields, threaded explicitly
        self,
        states: torch.Tensor,
        state_rows: torch.Tensor,
        state_mask: torch.Tensor,
        options: torch.Tensor,
        option_mask: torch.Tensor,
        option_rows: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Masked cross-entropy of the policy head against the teacher's choice."""
        logits = self.net.policy_logits(
            states, state_rows, state_mask, options, option_rows,
        )
        logits = logits.masked_fill(~option_mask, _NEG_INF)
        return F.cross_entropy(logits, targets)

    def value_loss(
        self,
        states: torch.Tensor,
        state_rows: torch.Tensor,
        state_mask: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """MSE of the value head against the (discounted) outcome target."""
        return F.mse_loss(self.net.value(states, state_rows, state_mask), values)

    def training_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,  # noqa: ARG002 - required by the Lightning step signature
    ) -> torch.Tensor:
        (
            states, state_rows, state_mask, options, option_mask, option_rows,
            targets, values,
        ) = batch
        p_loss = self.policy_loss(
            states, state_rows, state_mask, options, option_mask, option_rows, targets,
        )
        v_loss = self.value_loss(states, state_rows, state_mask, values)
        loss = p_loss + self.value_coef * v_loss
        self.log_dict(
            {"loss": loss, "policy_loss": p_loss, "value_loss": v_loss},
            prog_bar=False,
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Train trunk/policy/value only -- freeze the deck head AND the shared card
        # embedding during play BC. The play head now *reads* the embedding, so
        # optimising it here would pull the table toward the play task and disturb
        # the deck-head BC that runs next (measured: an all-energy-free deck). The
        # embedding is set by the CB BC (LitCBSeq) and then co-adapted in joint OSFP.
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


# --- Phase 5d: joint OSFP (play + deck + shared embedding in one update) -----


class LitJointPolicyGradient(L.LightningModule):
    """Joint self-play trainer: πBT and πCB optimised together (Phase 5d).

    Mirrors the ByteDance Hearthstone paper's joint OSFP: one update improves the
    **play (policy) and value heads AND the deck (CB+LSTM) head**, and -- crucially
    -- the **shared card embedding** ``cb_embed``, which both heads now read (the
    play head embeds each option's target card + the board's Pokemon; the deck head
    embeds candidate cards). No head is frozen, so the embedding receives gradient
    from *both* objectives in the same backward pass -- that is what makes it a
    genuinely shared representation rather than two separate tables.

    ``training_step`` consumes a :class:`~lightning.pytorch.utilities.CombinedLoader`
    batch ``{"play": play_8tuple, "deck": deck_4tuple}`` (built by the loop, in
    ``max_size_cycle`` mode):

    - play arm (REINFORCE + value baseline + entropy): the 8-tuple from
      :func:`~src.net.bc_data.collate_policy` with ``targets`` = sampled option and
      ``values`` = the deciding slot's game return.
    - deck arm (REINFORCE): the 4-tuple from
      :func:`~src.net.bc_data.collate_cb_seq` with ``weights`` = each deck's
      (normalised) advantage shared across its build steps.

    A missing key (an iteration that produced only one kind of sample) simply drops
    that arm's loss. PPO / V-Trace are omitted: one pass over fresh near-on-policy
    data keeps the importance ratio ~1.
    """

    def __init__(
        self,
        net: TorchPolicyValueNet,
        card_feats: NDArray,
        *,
        lr: float = 1e-3,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ) -> None:
        super().__init__()
        self.net = net
        self.register_buffer(
            "card_feats", torch.as_tensor(card_feats, dtype=torch.float32),
        )
        self.lr = lr
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def _play_loss(self, batch: Sequence[torch.Tensor]) -> torch.Tensor:
        (
            states, state_rows, state_mask, options, option_mask, option_rows,
            targets, returns,
        ) = batch
        logits = self.net.policy_logits(
            states, state_rows, state_mask, options, option_rows,
        )
        logits = logits.masked_fill(~option_mask, _NEG_INF)
        logp = F.log_softmax(logits, dim=1)
        logp_taken = logp.gather(1, targets.unsqueeze(1)).squeeze(1)

        values = self.net.value(states, state_rows, state_mask)
        advantage = (returns - values).detach()
        policy_loss = -(advantage * logp_taken).mean()
        value_loss = F.mse_loss(values, returns)

        # Masked entropy: zero the -inf out of logp *before* the product so a padded
        # option's 0 * -inf never produces a nan in the backward pass.
        safe_logp = logp.masked_fill(~option_mask, 0.0)
        entropy = -(logp.exp() * safe_logp).sum(dim=1).mean()
        return policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

    def _deck_loss(self, batch: Sequence[torch.Tensor]) -> torch.Tensor:
        targets, masks, weights, valid = batch
        logits = cb_sequence_logits(self.net, self.card_feats, targets)
        logits = logits.masked_fill(~masks, _NEG_INF)
        logp = F.log_softmax(logits, dim=-1)
        chosen = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return -(chosen * weights * valid).sum() / valid.sum().clamp(min=1)

    def training_step(
        self,
        batch: dict,
        batch_idx: int,  # noqa: ARG002 - required by the Lightning step signature
    ) -> torch.Tensor:
        play = self._play_loss(batch["play"]) if "play" in batch else None
        deck = self._deck_loss(batch["deck"]) if "deck" in batch else None
        terms = [t for t in (play, deck) if t is not None]
        loss = sum(terms)
        self.log_dict(
            {
                "loss": loss,
                **({"play_loss": play} if play is not None else {}),
                **({"deck_loss": deck} if deck is not None else {}),
            },
            prog_bar=False,
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # No freezing: trunk/policy/value AND cb/lstm AND the shared embedding.
        return torch.optim.Adam(self.parameters(), lr=self.lr)
