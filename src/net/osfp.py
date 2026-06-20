"""OSFP opponent pool -- the self-play meta-loop's history ``H`` (Phase 5).

A scaled-down version of the paper's Algorithm 1 (arXiv:2303.05197 SS3): the
learner best-responds to a *recency-weighted mixture of past checkpoints* plus the
scripted baselines, and (with probability ``self_play_prob``) to a copy of itself.
Weighting the mixture toward recent checkpoints is the "optimistic" part that
yields last-iterate convergence, so the final checkpoint can be submitted directly
(no separate average policy to maintain).

This module is pure (numpy only -- no ``cg``, no torch), so the whole meta-loop is
unit-testable on the host. The engine-bound self-play that turns a sampled
opponent into game data lives in ``scripts/collect_selfplay.py`` (Docker); the
orchestration that ties them together is ``scripts/train_osfp.py`` (native).

Weighting: an opponent's sample weight is

- ``baseline_floor`` for each scripted baseline (a fixed floor so the learner never
  forgets how to beat ``random`` / ``greedy`` / ``heuristic`` -- the cycling that
  OSFP's smoothing guards against), and
- ``decay ** (current_iter - admit_iter)`` for each checkpoint (recent = heavier).

``decay`` in ``(0, 1]`` tunes recency: ``decay < 1`` favours recent checkpoints
(recency ON); ``decay == 1`` weights every checkpoint equally (recency OFF -- the
ablation arm); ``decay -> 0`` collapses onto the most recent only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray


@dataclass(frozen=True)
class PoolEntry:
    """One opponent in the pool: a scripted baseline or a saved checkpoint.

    ``ref`` is the registered agent name for a baseline (``"random"`` /
    ``"greedy"`` / ``"heuristic"``, optionally a ``net`` weights path) or the
    ``.npz`` checkpoint path for a learned checkpoint. ``iteration`` is the
    checkpoint's admission iteration (used for recency weighting); baselines use
    ``-1`` (they never decay).
    """

    kind: Literal["baseline", "checkpoint"]
    ref: str
    iteration: int


class OpponentPool:
    """Recency-weighted opponent history ``H`` with checkpoint admission."""

    def __init__(  # noqa: PLR0913 - a config ctor legitimately threads its knobs
        self,
        baselines: Sequence[str],
        *,
        decay: float = 0.5,
        self_play_prob: float = 0.3,
        baseline_floor: float = 0.1,
        threshold: float = 0.55,
        patience: int = 3,
    ) -> None:
        self._baselines = [PoolEntry("baseline", name, -1) for name in baselines]
        self._checkpoints: list[PoolEntry] = []
        self.decay = float(decay)
        self.self_play_prob = float(self_play_prob)
        self.baseline_floor = float(baseline_floor)
        self.threshold = float(threshold)
        self.patience = int(patience)
        self._last_admit_iter = 0

    # --- inspection ---------------------------------------------------------

    @property
    def num_checkpoints(self) -> int:
        return len(self._checkpoints)

    def opponents(self) -> list[PoolEntry]:
        """All sampleable opponents: scripted baselines then admitted checkpoints."""
        return [*self._baselines, *self._checkpoints]

    def _raw_weights(self, current_iter: int) -> NDArray[np.float64]:
        weights = []
        for entry in self.opponents():
            if entry.kind == "baseline":
                weights.append(self.baseline_floor)
            else:
                distance = max(0, current_iter - entry.iteration)
                weights.append(self.decay**distance)
        return np.asarray(weights, dtype=np.float64)

    def weights(self, current_iter: int) -> dict[PoolEntry, float]:
        """Normalised sample probability per opponent (for tests / logging)."""
        raw = self._raw_weights(current_iter)
        total = float(raw.sum())
        probs = raw / total if total > 0 else raw
        return dict(zip(self.opponents(), (float(p) for p in probs), strict=True))

    # --- sampling / admission ----------------------------------------------

    def sample(
        self,
        current_iter: int,
        rng: np.random.Generator,
    ) -> PoolEntry | None:
        """Pick an opponent for one game, or ``None`` to self-play the learner.

        With probability ``self_play_prob`` returns ``None`` (the orchestrator
        plays the current learner against a copy of itself); otherwise draws an
        opponent from the recency-weighted mixture.
        """
        if rng.random() < self.self_play_prob:
            return None
        entries = self.opponents()
        if not entries:
            return None
        raw = self._raw_weights(current_iter)
        total = float(raw.sum())
        if total <= 0:
            return None
        probs = raw / total
        return entries[int(rng.choice(len(entries), p=probs))]

    def admit(
        self,
        checkpoint_path: str,
        iteration: int,
        winrates: Mapping[str, float],
    ) -> bool:
        """Add the current checkpoint to ``H`` if it earned a place.

        Admit when the checkpoint beats every evaluated opponent by more than
        ``threshold`` (a strong, generalising checkpoint) **or** ``patience``
        iterations have passed since the last admission (so a plateauing run still
        refreshes the pool). Returns whether it was admitted.
        """
        beat_all = bool(winrates) and all(
            wr > self.threshold for wr in winrates.values()
        )
        patience_hit = (iteration - self._last_admit_iter) >= self.patience
        if beat_all or patience_hit:
            self._checkpoints.append(
                PoolEntry("checkpoint", checkpoint_path, iteration),
            )
            self._last_admit_iter = iteration
            return True
        return False
