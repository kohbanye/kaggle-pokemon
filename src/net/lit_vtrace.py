"""V-Trace + PPO learner (the paper's RL update, Phase-rewrite stage 4).

Implements the paper's training objective (arXiv:2303.05197 SS5): **V-Trace**
off-policy correction (:mod:`src.net.vtrace`) + a **PPO** clipped surrogate, over
whole trajectories produced by the recurrent net (:mod:`src.net.recurrent_torch`).

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

from src.net.recurrent_torch import deck_sequence_factored
from src.net.vtrace import vtrace

if TYPE_CHECKING:
    from src.net.recurrent_torch import TorchRecurrentNet

_NEG_INF = float("-inf")


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over the True entries of ``valid`` (0 if none)."""
    denom = valid.sum().clamp(min=1)
    return (values * valid).sum() / denom


def _entropy_per_step(logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-step policy entropy ``(B, T)`` (masked candidates excluded)."""
    safe_logp = logp.masked_fill(~mask, 0.0)
    return -(logp.exp() * safe_logp).sum(dim=-1)


def _entropy(
    logp: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Mean per-step policy entropy over valid steps (masked candidates excluded)."""
    return _masked_mean(_entropy_per_step(logp, mask), valid)


def _normalize_adv(adv: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Whiten advantages over the valid steps (mean 0, std 1); padded steps -> 0.

    Standard PPO advantage normalisation. It de-biases the systematically-positive
    advantages that arise when the agent wins most games (every sampled action then
    gets a positive raw advantage ``z - V`` and is reinforced regardless of quality),
    turning them into a *relative* signal, and fixes the per-batch gradient scale (an
    all-wins batch no longer pushes every action up by a large amount).
    """
    mean = _masked_mean(adv, valid)
    var = _masked_mean((adv - mean) ** 2, valid)
    return ((adv - mean) / (var.sqrt() + 1e-8)) * valid


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
        train_deck: bool = True,
        normalize_adv: bool = True,
        entropy_floor: float = 0.0,
        entropy_floor_coef: float = 0.0,
    ) -> None:
        super().__init__()
        # train_deck=False -> battle-only: the deck (CB) head is NOT trained, used
        # when an external deck source (the QD MAP-Elites archive) owns the decks and
        # we only learn to PLAY them (the "QD decks + RL play" split).
        self.train_deck = train_deck
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
        # Battle-arm advantage whitening (standard PPO; on by default).
        self.normalize_adv = normalize_adv
        # Entropy floor: hinge-penalise steps whose per-step entropy drops below
        # ``entropy_floor`` (nats), weighted by ``entropy_floor_coef``. Unlike the mean
        # entropy bonus (diluted 1/T over the trajectory, so a single collapsed
        # high-leverage step -- e.g. the go-first/second opening -- gets ~no gradient),
        # the floor targets exactly the collapsed steps and leaves diverse ones alone.
        self.entropy_floor = entropy_floor
        self.entropy_floor_coef = entropy_floor_coef

    # --- battle arm (V-Trace + PPO) -----------------------------------------

    def _battle_loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loss, battle_start_value)``; the value feeds the deck baseline."""
        valid = batch["valid"]
        ctx = None
        if self.net.config.deck_ctx_dim > 0 and "deck_vec" in batch:
            ctx = self.net.deck_ctx(batch["deck_vec"])
        logits, values = self.net.play_sequence(
            batch["states"], batch["state_rows"], batch["state_mask"],
            batch["options"], batch["option_rows"], deck_ctx=ctx,
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
        # Whiten the PG advantage (the value targets ``vs`` stay raw -- they are a
        # regression target, not a gradient signal).
        if self.normalize_adv:
            adv = _normalize_adv(adv, valid)

        policy_loss = _ppo_surrogate(
            logp_taken, batch["behaviour_logp"], adv, valid, self.clip_eps,
        )
        value_loss = _masked_mean((values - vs) ** 2, valid)
        per_step_ent = _entropy_per_step(logp, batch["option_mask"])
        entropy = _masked_mean(per_step_ent, valid)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        floor_pen = torch.zeros((), device=loss.device, dtype=loss.dtype)
        if self.entropy_floor > 0.0 and self.entropy_floor_coef > 0.0:
            floor_pen = _masked_mean(
                torch.relu(self.entropy_floor - per_step_ent), valid,
            )
            loss = loss + self.entropy_floor_coef * floor_pen
        self.log_dict(
            {"battle_policy": policy_loss, "battle_value": value_loss,
             "battle_entropy": entropy, "battle_entropy_floor": floor_pen},
            prog_bar=False,
        )
        # Battle-start value (first step) is the learned baseline for the deck arm.
        return loss, values[:, 0].detach()

    # --- deck arm (REINFORCE w/ shared-value baseline, PPO-clipped) ----------

    def _deck_loss(
        self,
        batch: dict[str, torch.Tensor],
        battle_start_value: torch.Tensor,
    ) -> torch.Tensor:
        """Factored (category->card) deck PPO loss + targeted category entropy.

        The pick log-prob is ``log P(category) + log P(card | category)``; the entropy
        bonus is on the 3-way **category** distribution (where it actually keeps the
        energy budget alive, unlike entropy over the whole pool).
        """
        valid = batch["valid"]
        targets = batch["targets"]
        cat_logits, card_logits = deck_sequence_factored(
            self.net, self.card_feats, targets,
        )
        cat_logits = cat_logits.masked_fill(~batch["cat_legal"], _NEG_INF)
        card_logits = card_logits.masked_fill(~batch["card_legal"], _NEG_INF)
        cat_logp = F.log_softmax(cat_logits, dim=-1)
        card_logp = F.log_softmax(card_logits, dim=-1)
        cat_taken = cat_logp.gather(-1, batch["target_cat"].unsqueeze(-1)).squeeze(-1)
        card_taken = card_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        logp_taken = cat_taken + card_taken

        advantage = (batch["returns"] - battle_start_value).unsqueeze(1)  # (B,1)
        policy_loss = _ppo_surrogate(
            logp_taken, batch["behaviour_logp"], advantage, valid, self.clip_eps,
        )
        entropy = _entropy(cat_logp, batch["cat_legal"], valid)  # category entropy
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
        loss = battle_loss
        if self.train_deck:
            loss = loss + self._deck_loss(deck, battle_start_value)
        self.log("loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Optimise the whole recurrent net -- trunk, play LSTM, value/policy heads,
        # deck LSTM + CB head, and the shared card embedding. Nothing is frozen.
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)
