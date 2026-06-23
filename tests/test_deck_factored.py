"""Unit tests for the factored (category->card) deck action space."""

from __future__ import annotations

import numpy as np

from src.net.deck_factored import (
    N_CATEGORIES,
    factored_logp,
    factored_pick,
)


def test_joint_is_a_proper_distribution() -> None:
    """exp(joint log-probs) sums to 1 over the candidates."""
    rng = np.random.default_rng(0)
    cat_logits = rng.standard_normal(N_CATEGORIES)
    cand_cats = np.array([0, 0, 1, 2, 2, 2], dtype=np.intp)
    card_logits = rng.standard_normal(cand_cats.shape[0])
    joint = factored_logp(cat_logits, card_logits, cand_cats)
    np.testing.assert_allclose(np.exp(joint).sum(), 1.0, atol=1e-12)


def test_joint_equals_category_times_card() -> None:
    """joint[i] == P(cat_i) * P(card_i | cat_i) computed independently."""
    cat_logits = np.array([1.0, 0.0, -1.0])
    cand_cats = np.array([2, 2, 0], dtype=np.intp)  # only cats 0 and 2 present
    card_logits = np.array([0.5, -0.5, 2.0])
    joint = np.exp(factored_logp(cat_logits, card_logits, cand_cats))

    # P(cat) over the present categories {0, 2}.
    e = np.exp([cat_logits[0], cat_logits[2]])
    p_cat = {0: e[0] / e.sum(), 2: e[1] / e.sum()}
    # P(card | cat=2) over the two cat-2 candidates.
    e2 = np.exp([card_logits[0], card_logits[1]])
    expected = np.array([
        p_cat[2] * e2[0] / e2.sum(),
        p_cat[2] * e2[1] / e2.sum(),
        p_cat[0] * 1.0,  # lone candidate in cat 0
    ])
    np.testing.assert_allclose(joint, expected, atol=1e-12)


def test_energy_category_can_dominate_despite_many_other_cards() -> None:
    """A high energy-category logit lifts energy even with 1 energy vs many others.

    This is the whole point: a single energy candidate competing against many
    pokemon cards still gets ~P(cat=energy) mass, which a flat softmax could not give
    it.
    """
    cat_logits = np.array([-2.0, -2.0, 3.0])  # energy category strongly preferred
    cand_cats = np.array([0] * 20 + [2], dtype=np.intp)  # 20 pokemon, 1 energy
    card_logits = np.zeros(21)  # all cards equal within their category
    joint = np.exp(factored_logp(cat_logits, card_logits, cand_cats))
    # The lone energy card should carry P(cat=energy), far above any single pokemon.
    assert joint[-1] > 0.8
    assert joint[-1] > joint[0] * 50


def test_greedy_pick_is_joint_argmax() -> None:
    cat_logits = np.array([0.0, 0.0, 5.0])
    cand_cats = np.array([0, 2], dtype=np.intp)
    card_logits = np.array([10.0, 0.0])  # card 0 high, but its category is weak
    pick, logp = factored_pick(cat_logits, card_logits, cand_cats, None, greedy=True)
    assert pick == 1  # energy category wins despite card 0's high card-logit
    assert logp < 0.0


def test_sampling_is_seeded_and_in_range() -> None:
    rng = np.random.default_rng(3)
    cand_cats = np.array([0, 1, 2], dtype=np.intp)
    picks = {
        factored_pick(
            np.zeros(N_CATEGORIES), np.zeros(3), cand_cats, rng, greedy=False,
        )[0]
        for _ in range(50)
    }
    assert picks <= {0, 1, 2}
    assert len(picks) > 1  # actually stochastic
