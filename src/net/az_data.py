"""AlphaZero trajectory data for the RECURRENT net: build padded battle-sequence
batches carrying the search policy target ``pi`` and outcome ``z``.

Mirrors :mod:`src.net.trajectory_data`'s battle encoding + collation (so the same
``net.play_sequence`` LSTM forward drives training), but play-only (no deck arm) and
carrying the AlphaZero targets: per single-select step the ISMCTS visit-count ``pi``
(``None`` where no search ran -> policy loss masked out) and, on every step, the game
outcome ``z`` as the value target. Reconstructing ``h_t`` over the FULL ordered
single-select trajectory is why every decision is recorded, not just searched ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.net.encode import (
    OPTION_DIM,
    deck_context,
    encode_options,
    encode_state,
    option_embed_rows,
    state_embed_rows,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.net.embedding import CardEmbeddingIndex
    from src.net.features import CardFeatures

SINGLE_SELECT = 1


@dataclass
class AzStep:
    """One single-select decision: encoded state/options + search policy target."""

    state: NDArray[np.float64]
    state_rows: NDArray[np.intp]
    state_mask: NDArray[np.bool_]
    options: NDArray[np.float64]
    option_rows: NDArray[np.intp]
    pi: NDArray[np.float64] | None   # (K,) visit-count target, or None (unsearched)


@dataclass
class AzEpisode:
    steps: list[AzStep]
    z: float                          # game outcome from this slot's perspective
    deck_vec: NDArray[np.float64]


def build_az_episodes(
    games: list[dict], feats: CardFeatures, index: CardEmbeddingIndex,
) -> list[AzEpisode]:
    """Encode collected games into play-only AZ episodes.

    Each game: ``{winner, decisions: [{obs, choice, pi|None}], deck}`` where ``obs`` is
    the agent's observation at one of ITS single-select decisions (in order).
    """
    episodes: list[AzEpisode] = []
    for game in games:
        z = float(game.get("z", 0.0))
        deck = [int(c) for c in (game.get("deck") or [])]
        steps: list[AzStep] = []
        for mv in game.get("moves") or []:
            cur = mv.get("current") or {}
            sel = mv.get("select") or {}
            options = sel.get("option") or []
            if int(sel.get("maxCount", 0)) != SINGLE_SELECT or not options or not cur:
                continue
            slot = int(cur.get("yourIndex", 0))
            rows, mask = state_embed_rows(cur, slot, index)
            pi = mv.get("pi")
            pi_arr = (np.asarray(pi, dtype=np.float64)
                      if pi and len(pi) == len(options) else None)
            steps.append(AzStep(
                state=encode_state(cur, slot, feats),
                state_rows=rows, state_mask=mask,
                options=encode_options(options, cur, slot, feats),
                option_rows=option_embed_rows(options, cur, slot, index),
                pi=pi_arr,
            ))
        if steps:
            episodes.append(AzEpisode(steps, z, deck_context(deck, feats)))
    return episodes


def collate_az(episodes: list[AzEpisode]) -> dict[str, torch.Tensor]:
    """Pad AZ episodes to ``(B, T, ...)`` for ``net.play_sequence`` + AZ losses.

    Returns states/state_rows/state_mask/options/option_rows/option_mask (the
    play_sequence inputs), ``deck_ctx`` (B, ctx), ``pi`` (B,T,K) policy target,
    ``pi_valid`` (B,T) mask (searched steps), ``value_target`` (B,T) = z, and ``valid``.
    """
    bsz = len(episodes)
    max_t = max(len(ep.steps) for ep in episodes)
    max_k = max((s.options.shape[0] for ep in episodes for s in ep.steps), default=1)
    s0 = episodes[0].steps[0]
    n_slots, slot_max = s0.state_rows.shape

    states = torch.zeros(bsz, max_t, s0.state.shape[0])
    state_rows = torch.zeros(bsz, max_t, n_slots, slot_max, dtype=torch.long)
    state_mask = torch.zeros(bsz, max_t, n_slots, slot_max, dtype=torch.bool)
    options = torch.zeros(bsz, max_t, max_k, OPTION_DIM)
    option_mask = torch.zeros(bsz, max_t, max_k, dtype=torch.bool)
    option_rows = torch.zeros(bsz, max_t, max_k, dtype=torch.long)
    pi = torch.zeros(bsz, max_t, max_k)
    pi_valid = torch.zeros(bsz, max_t, dtype=torch.bool)
    value_target = torch.zeros(bsz, max_t)
    valid = torch.zeros(bsz, max_t, dtype=torch.bool)
    deck_vec = torch.zeros(bsz, episodes[0].deck_vec.shape[0])

    for i, ep in enumerate(episodes):
        deck_vec[i] = torch.from_numpy(ep.deck_vec).float()
        for t, step in enumerate(ep.steps):
            states[i, t] = torch.from_numpy(step.state)
            state_rows[i, t] = torch.from_numpy(step.state_rows)
            state_mask[i, t] = torch.from_numpy(step.state_mask)
            k = step.options.shape[0]
            options[i, t, :k] = torch.from_numpy(step.options).float()
            option_mask[i, t, :k] = True
            option_rows[i, t, :k] = torch.from_numpy(step.option_rows)
            value_target[i, t] = ep.z
            valid[i, t] = True
            if step.pi is not None:
                pi[i, t, :k] = torch.from_numpy(step.pi / max(step.pi.sum(), 1e-9))
                pi_valid[i, t] = True
        option_mask[i, len(ep.steps):, 0] = True  # dummy-legal for padded softmax
    return {
        "states": states, "state_rows": state_rows, "state_mask": state_mask,
        "options": options, "option_mask": option_mask, "option_rows": option_rows,
        "deck_vec": deck_vec, "pi": pi, "pi_valid": pi_valid,
        "value_target": value_target, "valid": valid,
    }
