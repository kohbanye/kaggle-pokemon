"""Online winrate surrogate for the QD deck search (Step 2, DSA-ME-lite).

Each real fitness evaluation costs hundreds of engine games, so the QD loop can
afford only ~24 children per generation. The surrogate predicts a deck's mean
winrate from its decklist alone, letting each generation *oversample* children
(``batch * oversample``), pre-screen them, and spend the same real-battle budget
on the most promising ones -- the DSA-ME idea (Zhang/Fontaine, GECCO 2022) sized
to our data: a run produces only a few hundred (deck, winrate) pairs, so the
model is a **ridge regression** on compact deck features (pure numpy, closed
form, refit every generation), not a deep net. If ridge shows signal, a deeper
model is a later ablation.

Anti-bias guards (the doc's Step-2 risk mitigations):
- the surrogate trains *online* on every real evaluation (co-improving), and
- :func:`select_children` always reserves an ``explore_frac`` of the real-eval
  budget for randomly-chosen children, so calibration data keeps covering the
  region the surrogate would otherwise starve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.qd.deck_qd import colour_count, deck_stats

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from src.deck import CardPool

# Fit only once we have at least a generation's worth of real evaluations.
MIN_FIT = 24
# Composition block size appended after the mean per-card feature vector.
_N_COMP = 7
# Sentinel "slowest" attack cost for a deck with no attacker (matches speed_bin's
# treatment of None) before normalisation.
_NO_ATTACK_COST = 4


class DeckFeaturizer:
    """Decklist -> fixed vector: mean per-card features + composition stats.

    ``card_vec`` maps a card id to its fixed feature vector (e.g.
    :meth:`src.net.features.CardFeatures.vector`); the deck vector is the mean of
    those over the 60 cards, concatenated with coarse composition stats (counts,
    prize points, cheapest attack, colour count) scaled to O(1). Compact (~56
    dims) on purpose: a QD run yields only hundreds of samples.
    """

    def __init__(self, pool: CardPool, card_vec: Callable[[int], NDArray[np.float64]],
                 ) -> None:
        self.pool = pool
        self.card_vec = card_vec
        self.dim = int(card_vec(next(iter(pool.cards))).shape[0]) + _N_COMP

    def vector(self, deck: list[int]) -> NDArray[np.float64]:
        cards = np.stack([self.card_vec(cid) for cid in deck])
        stats = deck_stats(deck, self.pool)
        cost = stats["min_attack_cost"]
        comp = np.array([
            (stats["energy"] or 0) / 60.0,
            (stats["pokemon"] or 0) / 60.0,
            (stats["trainer"] or 0) / 60.0,
            (stats["distinct"] or 0) / 60.0,
            (stats["prize_points"] or 0) / 30.0,
            (_NO_ATTACK_COST if cost is None else cost) / 4.0,
            colour_count(deck, self.pool) / 5.0,
        ])
        return np.concatenate([cards.mean(axis=0), comp])


class RidgeSurrogate:
    """Online ridge regression deck -> mean winrate (closed form, standardized).

    ``add`` accumulates every real evaluation; ``fit`` re-solves (cheap at our
    sizes); ``predict`` returns 0.5 (an uninformative prior) until ``ready``.
    """

    def __init__(self, featurizer: DeckFeaturizer, l2: float = 1.0) -> None:
        self.featurizer = featurizer
        self.l2 = l2
        self._x: list[NDArray[np.float64]] = []
        self._y: list[float] = []
        self._w: NDArray[np.float64] | None = None
        self._mu: NDArray[np.float64] | None = None
        self._sigma: NDArray[np.float64] | None = None
        self._y_mean = 0.0

    @property
    def n_samples(self) -> int:
        return len(self._y)

    @property
    def ready(self) -> bool:
        """Enough data to trust predictions for pre-screening."""
        return self._w is not None and self.n_samples >= MIN_FIT

    def add(self, deck: list[int], winrate: float) -> None:
        self._x.append(self.featurizer.vector(deck))
        self._y.append(float(winrate))

    def fit(self) -> None:
        if len(self._y) < 2:  # noqa: PLR2004 - need >=2 points for a slope
            return
        x = np.stack(self._x)
        y = np.array(self._y)
        self._mu = x.mean(axis=0)
        sigma = x.std(axis=0)
        self._sigma = np.where(sigma > 0, sigma, 1.0)
        xs = (x - self._mu) / self._sigma
        self._y_mean = float(y.mean())
        yc = y - self._y_mean
        d = xs.shape[1]
        self._w = np.linalg.solve(xs.T @ xs + self.l2 * np.eye(d), xs.T @ yc)

    def predict(self, decks: list[list[int]]) -> NDArray[np.float64]:
        if self._w is None or self._mu is None or self._sigma is None:
            return np.full(len(decks), 0.5)
        x = np.stack([self.featurizer.vector(d) for d in decks])
        return (x - self._mu) / self._sigma @ self._w + self._y_mean


def select_children(
    scores: NDArray[np.float64],
    n: int,
    explore_frac: float,
    rng: np.random.Generator,
) -> list[int]:
    """Indices of ``n`` children to really evaluate: top-scored + random explores.

    ``ceil(n * explore_frac)`` slots are drawn uniformly from the non-top rest
    (surrogate-blind), keeping calibration data flowing from the region the
    surrogate would otherwise starve; the remainder are the highest-scored. With
    fewer candidates than ``n``, everything is selected.
    """
    m = len(scores)
    if m <= n:
        return list(range(m))
    n_explore = min(int(np.ceil(n * explore_frac)), n)
    n_top = n - n_explore
    order = np.argsort(scores)[::-1]
    top = order[:n_top].tolist()
    rest = order[n_top:]
    explore = rng.choice(rest, size=n_explore, replace=False).tolist()
    return [int(i) for i in top + explore]
