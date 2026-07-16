"""Opponent-archetype belief for the ReBeL PBS solve (design step D).

The subgame's root chance node deals a hidden WORLD from the belief
(`BeliefSubgame`). This module produces that belief: a posterior over a fixed set of
archetype decklists, sharpened by the opponent's revealed cards, and a sampler that
turns it into determinized worlds (particles) for the solver.

The posterior reuses :mod:`src.search.opp_belief` (multiset overlap of revealed cards
with each archetype = a consistency score), the signal `opp_context` conditions the play
net on. It is recomputed from the observation each decision, so the belief sharpens over
the game with no explicit filter state; a solve then FIXES the sampled worlds, giving
CFR the fixed range it needs (Codex review).

Known approximation (honest): the score is revealed-card *overlap*, not a
policy/history-conditioned generative likelihood, and within an archetype the
determinizer draws hidden cards heuristically (`determinize.sample_determinization`).
Both are ISMCTS-grade belief, adequate to start; a policy-aware likelihood comes later.

Pure numpy + file reads; engine-free, unit-tested.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.search.determinize import sample_determinization
from src.search.opp_belief import consistency_scores, seen_opponent_ids

if TYPE_CHECKING:
    from src.search.determinize import Determinization

ROOT = Path(__file__).resolve().parent.parent.parent
_HYP_DIRS = ("decklists", "decklists/anchors", "decklists/candidates")
_DEFAULT_SHARPNESS = 6.0


def load_hypotheses(dirs: tuple[str, ...] = _HYP_DIRS) -> list[list[int]]:
    """Archetype decklists from the fixed hypothesis dirs, de-duplicated by multiset.

    Deterministic and engine-free (must match at train and serve, like
    :func:`src.net.opp_context.build_hypotheses`)."""
    decks: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for d in dirs:
        for p in sorted((ROOT / d).glob("*.csv")):
            deck = [int(x) for x in p.read_text().split() if x.strip()]
            key = tuple(sorted(deck))
            if deck and key not in seen:
                seen.add(key)
                decks.append(deck)
    return decks


class OpponentBelief:
    """Posterior over archetype decklists from the opponent's revealed cards."""

    def __init__(
        self,
        hypotheses: list[list[int]],
        *,
        sharpness: float = _DEFAULT_SHARPNESS,
    ) -> None:
        if not hypotheses:
            msg = "OpponentBelief needs >=1 hypothesis deck"
            raise ValueError(msg)
        self.hypotheses = hypotheses
        self.sharpness = float(sharpness)

    @classmethod
    def from_dirs(cls, dirs: tuple[str, ...] = _HYP_DIRS,
                  *, sharpness: float = _DEFAULT_SHARPNESS) -> OpponentBelief:
        return cls(load_hypotheses(dirs), sharpness=sharpness)

    def posterior(self, current: dict, your_index: int) -> np.ndarray:
        """Belief weight per hypothesis. Uniform when nothing is seen; otherwise a
        softmax over the fraction of the seen cards each archetype explains."""
        seen = seen_opponent_ids(current, your_index)
        n = len(self.hypotheses)
        if not seen:
            return np.full(n, 1.0 / n)
        frac = consistency_scores(self.hypotheses, seen) / len(seen)
        w = np.exp(self.sharpness * (frac - frac.max()))
        return w / w.sum()

    def entropy(self, current: dict, your_index: int) -> float:
        """Shannon entropy (nats) of the posterior -- a belief-sharpness diagnostic."""
        p = self.posterior(current, your_index)
        return float(-(p * np.log(p + 1e-12)).sum())

    def sample_deck(
        self, current: dict, your_index: int, rng: np.random.Generator,
    ) -> list[int]:
        """Sample one archetype decklist proportional to the posterior."""
        return self.hypotheses[self.sample_index(current, your_index, rng)]

    def sample_index(
        self, current: dict, your_index: int, rng: np.random.Generator,
    ) -> int:
        """Sample one archetype INDEX (into ``hypotheses``) proportional to the
        posterior -- the infostate label a proper-ReBeL value vector is indexed by."""
        p = self.posterior(current, your_index)
        return int(rng.choice(len(self.hypotheses), p=p))


def build_worlds(  # noqa: PLR0913 - a determinized world legitimately has many parts
    current: dict,
    your_index: int,
    our_deck: list[int],
    belief: OpponentBelief,
    n: int,
    rng: np.random.Generator,
    *,
    opp_basics: list[int] | None = None,
) -> list[tuple[Determinization, float]]:
    """``n`` belief-sampled determinized worlds (particles) for a `BeliefSubgame`.

    Each particle draws an archetype ~ posterior, then a consistent determinization of
    the hidden cards from it. Equal weights (the archetype sampling already encodes the
    posterior), so the subgame's root chance node reproduces the belief in expectation.
    """
    worlds: list[tuple[Determinization, float]] = []
    for _ in range(n):
        opp_prior = belief.sample_deck(current, your_index, rng)
        det = sample_determinization(
            current, your_index, our_deck, opp_prior, rng, opp_basics=opp_basics)
        worlds.append((det, 1.0 / n))
    return worlds


def build_worlds_hyp(  # noqa: PLR0913 - a determinized world legitimately has many parts
    current: dict,
    your_index: int,
    our_deck: list[int],
    belief: OpponentBelief,
    n: int,
    rng: np.random.Generator,
    *,
    opp_basics: list[int] | None = None,
) -> tuple[list[tuple[Determinization, float]], list[int]]:
    """Like :func:`build_worlds` but ALSO returns the archetype hypothesis index each
    world was sampled from (the infostate label). Proper ReBeL needs this: a world drawn
    from hypothesis ``h_k`` reads the value net's ``output[h_k]`` at its leaf, and its
    CFR root value trains that same slot -- so the net learns a value per infostate."""
    worlds: list[tuple[Determinization, float]] = []
    hyps: list[int] = []
    for _ in range(n):
        idx = belief.sample_index(current, your_index, rng)
        det = sample_determinization(
            current, your_index, our_deck, belief.hypotheses[idx], rng,
            opp_basics=opp_basics)
        worlds.append((det, 1.0 / n))
        hyps.append(idx)
    return worlds, hyps
