"""V-Trace + PPO learner (the paper's RL update, Phase-rewrite stage 4).

Replaces the Phase-5d plain-REINFORCE :class:`~src.net.lit.LitJointPolicyGradient`
with the paper's actual objective (arXiv:2303.05197 SS5): **V-Trace** off-policy
correction (:mod:`src.net.vtrace`) + a **PPO** clipped surrogate, over whole
trajectories produced by the recurrent net (:mod:`src.net.recurrent_torch`).

One ``training_step`` consumes an aligned ``(battle, deck)`` batch from
:func:`~src.net.trajectory_data.collate_episodes` and updates the policy, value and
deck heads + the shared trunk / play-LSTM / card embedding in a single backward:

- **battle arm** -- recurrent forward -> per-step value + option logits; V-Trace
  turns (behaviour log-probs, target log-probs, values, terminal reward) into value
  targets ``vs`` and PG advantages; PPO surrogate + value MSE + entropy.
- **deck arm** -- the deck LSTM's pick logits; advantage = ``return - V(battle
  start)`` (the shared value as a learned baseline, no batch-mean / KL hacks);
  PPO surrogate + entropy.

The V-Trace targets are computed off-graph (numpy, detached) -- they are regression
targets, never differentiated through -- which is exactly correct.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import torch
from torch.nn import functional as F

from src.net.lit import cb_sequence_logits
from src.net.vtrace import vtrace

if TYPE_CHECKING:
    from src.net.recurrent_torch import TorchRecurrentNet

_NEG_INF = float("-inf")


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over the True entries of ``valid`` (0 if none)."""
    denom = valid.sum().clamp(min=1)
    return (values * valid).sum() / denom


def _entropy(
    logp: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Mean per-step policy entropy over valid steps (masked candidates excluded)."""
    safe_logp = logp.masked_fill(~mask, 0.0)
    per_step = -(logp.exp() * safe_logp).sum(dim=-1)  # (B, T)
    return _masked_mean(per_step, valid)


def _ppo_surrogate(
    logp_taken: torch.Tensor,
    behaviour_logp: torch.Tensor,
    advantage: torch.Tensor,
    valid: torch.Tensor,
    clip_eps: float,
) -> torch.Tensor:
    """PPO clipped-surrogate **loss** (to minimise), averaged over valid steps."""
    ratio = torch.exp(logp_taken - behaviour_logp)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    return -_masked_mean(torch.minimum(unclipped, clipped), valid)


class LitVtracePPO(L.LightningModule):
    """V-Trace + PPO trainer over recurrent self-play trajectories."""

    def __init__(  # noqa: PLR0913 - keyword-only training hyperparameters
        self,
        net: TorchRecurrentNet,
        card_feats: object,
        *,
        lr: float = 1e-3,
        gamma: float = 1.0,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        deck_entropy_coef: float = 0.01,
        clip_eps: float = 0.2,
        clip_rho: float = 1.0,
        clip_c: float = 1.0,
        rho_min: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = net
        self.register_buffer(
            "card_feats", torch.as_tensor(card_feats, dtype=torch.float32),
        )
        self.lr = lr
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.deck_entropy_coef = deck_entropy_coef
        self.clip_eps = clip_eps
        self.clip_rho = clip_rho
        self.clip_c = clip_c
        self.rho_min = rho_min

    # --- battle arm (V-Trace + PPO) -----------------------------------------

    def _battle_loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loss, battle_start_value)``; the value feeds the deck baseline."""
        valid = batch["valid"]
        logits, values = self.net.play_sequence(
            batch["states"], batch["state_rows"], batch["state_mask"],
            batch["options"], batch["option_rows"],
        )
        logits = logits.masked_fill(~batch["option_mask"], _NEG_INF)
        logp = F.log_softmax(logits, dim=-1)
        logp_taken = logp.gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1)

        # V-Trace targets off-graph (detached numpy) -- they are regression targets.
        vt = vtrace(
            behaviour_logp=logp_taken.detach().cpu().numpy(),
            target_logp=logp_taken.detach().cpu().numpy(),
            values=values.detach().cpu().numpy(),
            rewards=batch["rewards"].cpu().numpy(),
            valid=valid.cpu().numpy(),
            bootstrap_value=batch["bootstrap"].cpu().numpy(),
            gamma=self.gamma, clip_rho=self.clip_rho, clip_c=self.clip_c,
            rho_min=self.rho_min,
        )
        vs = torch.as_tensor(vt.vs, dtype=values.dtype, device=values.device)
        adv = torch.as_tensor(
            vt.pg_advantages, dtype=values.dtype, device=values.device,
        )

        policy_loss = _ppo_surrogate(
            logp_taken, batch["behaviour_logp"], adv, valid, self.clip_eps,
        )
        value_loss = _masked_mean((values - vs) ** 2, valid)
        entropy = _entropy(logp, batch["option_mask"], valid)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        self.log_dict(
            {"battle_policy": policy_loss, "battle_value": value_loss,
             "battle_entropy": entropy}, prog_bar=False,
        )
        # Battle-start value (first step) is the learned baseline for the deck arm.
        return loss, values[:, 0].detach()

    # --- deck arm (REINFORCE w/ shared-value baseline, PPO-clipped) ----------

    def _deck_loss(
        self,
        batch: dict[str, torch.Tensor],
        battle_start_value: torch.Tensor,
    ) -> torch.Tensor:
        valid = batch["valid"]
        targets = batch["targets"]
        logits = cb_sequence_logits(self.net, self.card_feats, targets)
        logits = logits.masked_fill(~batch["legal"], _NEG_INF)
        logp = F.log_softmax(logits, dim=-1)
        logp_taken = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        advantage = (batch["returns"] - battle_start_value).unsqueeze(1)  # (B,1)
        policy_loss = _ppo_surrogate(
            logp_taken, batch["behaviour_logp"], advantage, valid, self.clip_eps,
        )
        entropy = _entropy(logp, batch["legal"], valid)
        loss = policy_loss - self.deck_entropy_coef * entropy
        self.log_dict(
            {"deck_policy": policy_loss, "deck_entropy": entropy}, prog_bar=False,
        )
        return loss

    def training_step(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        batch_idx: int,  # noqa: ARG002 - required by the Lightning step signature
    ) -> torch.Tensor:
        battle, deck = batch
        battle_loss, battle_start_value = self._battle_loss(battle)
        deck_loss = self._deck_loss(deck, battle_start_value)
        loss = battle_loss + deck_loss
        self.log("loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Optimise the whole recurrent net -- trunk, play LSTM, value/policy heads,
        # deck LSTM + CB head, and the shared card embedding. Nothing is frozen.
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)
