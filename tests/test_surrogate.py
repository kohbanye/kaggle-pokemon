"""Unit tests for the QD winrate surrogate (Step 2) -- pure numpy, no engine."""

from __future__ import annotations

import numpy as np

from src.deck import CardInfo, CardPool
from src.qd import DeckFeaturizer, RidgeSurrogate, select_children
from src.qd.surrogate import MIN_FIT


def _pool() -> CardPool:
    """Minimal deck-buildable pool: two attackers, a trainer, basic energy."""
    infos = [
        CardInfo(1, "AtkR", "Pokemon", "Basic Pokémon", True, False, False, "R",
                 min_attack_cost=1),
        CardInfo(2, "AtkW", "Pokemon", "Basic Pokémon", True, False, False, "W",
                 is_ex=True, min_attack_cost=2),
        CardInfo(10, "TrItem", "Trainer", "Item", False, False, False, ""),
        CardInfo(20, "Fire Energy", "Energy", "Basic Energy", False, True, False, "R"),
    ]
    return CardPool({info.card_id: info for info in infos})


def _card_vec(cid: int) -> np.ndarray:
    """Tiny stand-in for CardFeatures.vector: a 3-dim one-hot-ish embedding."""
    return np.array([float(cid == 1), float(cid == 2), float(cid >= 10)])


def _deck(n1: int, n2: int) -> list[int]:
    """A 60-card deck with n1/n2 copies of the attackers, rest energy."""
    return [1] * n1 + [2] * n2 + [20] * (60 - n1 - n2)


def test_featurizer_shape_and_determinism() -> None:
    fz = DeckFeaturizer(_pool(), _card_vec)
    v = fz.vector(_deck(4, 4))
    assert v.shape == (fz.dim,)
    assert fz.dim == 3 + 7  # card_vec dim + composition block
    assert np.allclose(v, fz.vector(_deck(4, 4)))  # deterministic
    # order-invariant (multiset feature)
    assert np.allclose(v, fz.vector(list(reversed(_deck(4, 4)))))


def test_ridge_predicts_planted_signal() -> None:
    """Winrate depends linearly on the deck's AtkW density: ridge must rank it."""
    fz = DeckFeaturizer(_pool(), _card_vec)
    sur = RidgeSurrogate(fz)
    rng = np.random.default_rng(0)
    for _ in range(40):
        n2 = int(rng.integers(0, 5)) + int(rng.integers(0, 5))
        deck = _deck(int(rng.integers(1, 5)), n2)
        sur.add(deck, 0.3 + 0.05 * n2)  # planted: more AtkW -> higher winrate
    sur.fit()
    assert sur.ready
    preds = sur.predict([_deck(4, 0), _deck(4, 4), _deck(4, 8)])
    assert preds[0] < preds[1] < preds[2]  # recovers the monotone signal


def test_ridge_unfitted_returns_prior_and_not_ready() -> None:
    sur = RidgeSurrogate(DeckFeaturizer(_pool(), _card_vec))
    assert not sur.ready
    assert np.allclose(sur.predict([_deck(4, 4)]), 0.5)  # uninformative prior
    for i in range(MIN_FIT - 1):  # below the warm-up threshold
        sur.add(_deck(1 + i % 4, 0), 0.5)
    sur.fit()
    assert not sur.ready
    sur.add(_deck(4, 4), 0.6)
    sur.fit()
    assert sur.ready


def test_select_children_top_plus_explore() -> None:
    rng = np.random.default_rng(1)
    scores = np.arange(20, dtype=float)  # best = index 19, 18, ...
    idx = select_children(scores, n=8, explore_frac=0.25, rng=rng)
    assert len(idx) == len(set(idx)) == 8
    n_explore = 2  # ceil(8 * 0.25)
    top = set(range(20 - (8 - n_explore), 20))  # the 6 top-scored indices
    assert top <= set(idx)  # exploit slots are exactly the top-scored
    assert len(set(idx) - top) == n_explore  # rest drawn from the non-top pool
    # explore_frac=0 -> pure top-n
    assert set(select_children(scores, 8, 0.0, rng)) == set(range(12, 20))
    # fewer candidates than budget -> take everything
    assert select_children(np.array([1.0, 2.0]), 8, 0.25, rng) == [0, 1]
