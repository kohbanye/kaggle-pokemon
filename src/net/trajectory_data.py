"""Trajectory data for V-Trace/PPO self-play (the paper's actor-learner format).

Unlike the Phase-4 BC pipeline (independent ``(state, action)`` samples), V-Trace
needs **whole ordered trajectories** with the actor's behaviour log-probs, so the
importance ratio ``π/μ`` can be formed. One :class:`Episode` is one player's game:
the deck picks (CB head) then the single-select battle decisions (BT head), with the
terminal ±1 return.

The recurrence steps on the **single-select** battle decisions only (multi-select
sub-choices are mostly forced and the serving agent derives them deterministically
without advancing the play LSTM -- train and serve agree). Battle trajectories are
kept whole; cost is controlled by sampling whole games, not thinning decisions.

This module is pure (numpy + torch, no ``cg``): the engine-bound collector writes
raw game/deck JSONL, and :func:`build_episodes` encodes it here (so the encoder can
be ablated without re-simulating). :func:`collate_episodes` pads a batch into the
aligned ``(battle, deck)`` tensor dicts that
:class:`~src.net.lit_vtrace.LitVtracePPO` consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.deck import legal_next_ids
from src.net.encode import (
    OPTION_DIM,
    encode_options,
    encode_state,
    option_embed_rows,
    state_embed_rows,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from src.deck import CardPool
    from src.net.embedding import CardEmbeddingIndex
    from src.net.features import CardFeatures

SINGLE_SELECT = 1


@dataclass
class BattleStep:
    """One single-select battle decision: encoded inputs + the actor's pick/log-prob."""

    state: NDArray[np.float64]  # (STATE_DIM,)
    state_rows: NDArray[np.intp]  # (S, SLOT_MAX)
    state_mask: NDArray[np.bool_]  # (S, SLOT_MAX)
    options: NDArray[np.float64]  # (K, OPTION_DIM)
    option_rows: NDArray[np.intp]  # (K,)
    action: int  # index of the sampled option
    behaviour_logp: float  # log μ(action | state) at collection time


@dataclass
class Episode:
    """One player's game: deck-build picks ⊕ battle steps, terminal return.

    ``deck_rows`` is the deck in CB pick order (pool rows); ``deck_legal`` the
    per-step legal mask; ``deck_logp`` the actor's per-pick log-prob. ``ret`` is the
    terminal outcome in ``[-1, 1]`` from this player's view.
    """

    battle: list[BattleStep]
    deck_rows: NDArray[np.int64]  # (Td,)
    deck_legal: NDArray[np.bool_]  # (Td, N_pool)
    deck_logp: NDArray[np.float64]  # (Td,)
    ret: float


# --- building episodes from the raw collector logs --------------------------


def _outcome(winner: int, slot: int) -> float:
    """Terminal return in ``[-1, 1]`` from ``slot``'s view."""
    if winner == slot:
        return 1.0
    if winner == 1 - slot:
        return -1.0
    return 0.0


def _battle_steps(
    decisions: list[dict],
    slot: int,
    feats: CardFeatures,
    index: CardEmbeddingIndex | None,
) -> list[BattleStep]:
    """Encode one slot's ordered single-select decisions into battle steps."""
    steps: list[BattleStep] = []
    for decision in decisions:
        if int(decision.get("slot", -1)) != slot:
            continue
        obs = decision.get("obs") or {}
        select = obs.get("select") or {}
        if int(select.get("maxCount", 0)) != SINGLE_SELECT:
            continue
        choice = decision.get("choice") or []
        options = select.get("option") or []
        if len(choice) != SINGLE_SELECT or not options:
            continue
        action = int(choice[0])
        if not 0 <= action < len(options):
            continue
        current = obs.get("current") or {}
        steps.append(BattleStep(
            state=encode_state(current, slot, feats),
            state_rows=state_embed_rows(current, slot, index)[0],
            state_mask=state_embed_rows(current, slot, index)[1],
            options=encode_options(options, current, slot, feats),
            option_rows=option_embed_rows(options, current, slot, index),
            action=action,
            behaviour_logp=float(decision.get("logp", 0.0)),
        ))
    return steps


def _deck_arrays(
    deck: list[int],
    deck_logp: list[float],
    pool: CardPool,
    index: CardEmbeddingIndex,
) -> tuple[NDArray[np.int64], NDArray[np.bool_], NDArray[np.float64]] | None:
    """Recompute (pool rows, per-step legal masks, log-probs) from a pick order."""
    rows: list[int] = []
    masks: list[NDArray[np.bool_]] = []
    logps: list[float] = []
    for t, card in enumerate(deck):
        row = index.row(card)
        if row >= index.n_pool:
            continue
        legal = legal_next_ids(deck[:t], pool)
        if card not in legal:
            continue
        mask = np.zeros(index.n_pool, dtype=np.bool_)
        for cid in legal:
            r = index.row(cid)
            if r < index.n_pool:
                mask[r] = True
        rows.append(row)
        masks.append(mask)
        logps.append(deck_logp[t] if t < len(deck_logp) else 0.0)
    if not rows:
        return None
    return (
        np.array(rows, dtype=np.int64),
        np.stack(masks),
        np.array(logps, dtype=np.float64),
    )


def build_episodes(
    games: Iterable[dict],
    feats: CardFeatures,
    index: CardEmbeddingIndex,
    pool: CardPool,
    *,
    teachers: set[str] | None = None,
) -> list[Episode]:
    """Encode raw ``"game"`` records into :class:`Episode` objects.

    Each game line carries ``winner``, the learner ``deck`` (pick order) +
    ``deck_logp``, and ``decisions`` (each with ``slot``/``agent``/``obs``/
    ``choice``/``logp``). ``teachers`` restricts which slots become episodes (the
    learner's; self-play tags both). A game yields one episode per qualifying slot.
    """
    episodes: list[Episode] = []
    for game in games:
        winner = int(game.get("winner", -1))
        decisions = game.get("decisions") or []
        deck = [int(c) for c in (game.get("deck") or [])]
        deck_logp = [float(x) for x in (game.get("deck_logp") or [])]
        deck_arrays = _deck_arrays(deck, deck_logp, pool, index)
        if deck_arrays is None:
            continue
        slots = {int(d.get("slot", 0)) for d in decisions
                 if teachers is None or d.get("agent") in teachers}
        for slot in sorted(slots):
            steps = _battle_steps(decisions, slot, feats, index)
            if not steps:
                continue
            rows, masks, logps = deck_arrays
            episodes.append(Episode(steps, rows, masks, logps, _outcome(winner, slot)))
    return episodes


# --- collation into padded batch tensors ------------------------------------


def _collate_battle(episodes: list[Episode]) -> dict[str, torch.Tensor]:
    """Pad battle trajectories to ``(B, T, ...)`` with a ``valid`` mask.

    The terminal reward (the episode return) is placed on each row's last valid
    step; every earlier step's reward is 0 (gamma=1, terminal-only). ``bootstrap``
    is 0 (episodes are complete).
    """
    bsz = len(episodes)
    max_t = max(len(ep.battle) for ep in episodes)
    max_k = max((s.options.shape[0] for ep in episodes for s in ep.battle), default=1)
    s0 = episodes[0].battle[0]
    n_slots, slot_max = s0.state_rows.shape

    states = torch.zeros(bsz, max_t, s0.state.shape[0])
    state_rows = torch.zeros(bsz, max_t, n_slots, slot_max, dtype=torch.long)
    state_mask = torch.zeros(bsz, max_t, n_slots, slot_max, dtype=torch.bool)
    options = torch.zeros(bsz, max_t, max_k, OPTION_DIM)
    option_mask = torch.zeros(bsz, max_t, max_k, dtype=torch.bool)
    option_rows = torch.zeros(bsz, max_t, max_k, dtype=torch.long)
    actions = torch.zeros(bsz, max_t, dtype=torch.long)
    behaviour_logp = torch.zeros(bsz, max_t)
    rewards = torch.zeros(bsz, max_t)
    valid = torch.zeros(bsz, max_t, dtype=torch.bool)

    for i, ep in enumerate(episodes):
        for t, step in enumerate(ep.battle):
            states[i, t] = torch.from_numpy(step.state)
            state_rows[i, t] = torch.from_numpy(step.state_rows)
            state_mask[i, t] = torch.from_numpy(step.state_mask)
            k = step.options.shape[0]
            options[i, t, :k] = torch.from_numpy(step.options).float()
            option_mask[i, t, :k] = True
            option_rows[i, t, :k] = torch.from_numpy(step.option_rows)
            actions[i, t] = step.action
            behaviour_logp[i, t] = step.behaviour_logp
            valid[i, t] = True
        # Padded steps get a single dummy-legal option so their masked softmax is
        # finite (an all-(-inf) row -> nan, and nan*0 would poison the masked means).
        # ``valid`` already excludes them from every loss term.
        option_mask[i, len(ep.battle) :, 0] = True
        rewards[i, len(ep.battle) - 1] = ep.ret  # terminal reward on the last step
    return {
        "states": states, "state_rows": state_rows, "state_mask": state_mask,
        "options": options, "option_mask": option_mask, "option_rows": option_rows,
        "actions": actions, "behaviour_logp": behaviour_logp, "rewards": rewards,
        "valid": valid, "bootstrap": torch.zeros(bsz),
    }


def _collate_deck(episodes: list[Episode]) -> dict[str, torch.Tensor]:
    """Pad deck-build sequences to ``(B, Td, ...)`` with a ``valid`` mask."""
    bsz = len(episodes)
    max_t = max(ep.deck_rows.shape[0] for ep in episodes)
    n_pool = episodes[0].deck_legal.shape[1]

    targets = torch.zeros(bsz, max_t, dtype=torch.long)
    legal = torch.zeros(bsz, max_t, n_pool, dtype=torch.bool)
    behaviour_logp = torch.zeros(bsz, max_t)
    valid = torch.zeros(bsz, max_t, dtype=torch.bool)
    returns = torch.zeros(bsz)
    for i, ep in enumerate(episodes):
        t = ep.deck_rows.shape[0]
        targets[i, :t] = torch.from_numpy(ep.deck_rows)
        legal[i, :t] = torch.from_numpy(ep.deck_legal)
        behaviour_logp[i, :t] = torch.from_numpy(ep.deck_logp).float()
        valid[i, :t] = True
        legal[i, t:, 0] = True  # dummy-legal for padded steps (avoid all-(-inf))
        returns[i] = ep.ret
    return {
        "targets": targets, "legal": legal, "behaviour_logp": behaviour_logp,
        "valid": valid, "returns": returns,
    }


def collate_episodes(
    episodes: list[Episode],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Collate a batch of episodes into aligned ``(battle, deck)`` tensor dicts.

    Row ``i`` of the battle dict and row ``i`` of the deck dict are the **same
    episode**, so the learner can use the battle-start value as the deck baseline.
    """
    return _collate_battle(episodes), _collate_deck(episodes)


class EpisodeDataset(torch.utils.data.Dataset):
    """A list of :class:`Episode`; pair with :func:`collate_episodes`."""

    def __init__(self, episodes: list[Episode]) -> None:
        self.episodes = episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Episode:
        return self.episodes[idx]
