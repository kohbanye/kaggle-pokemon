"""Factored deck-build action space: category -> card (the paper's type->target).

The flat per-card CB softmax starves Basic Energy: each energy card competes 1-of-
~1267, so "spend ~15 picks on energy" can't accumulate probability and the deck
collapses to zero-energy (measured). The paper factors the build action as
``(category, card)``; here the category is the coarse card type
:func:`~src.deck.card_kind` returns -- ``{pokemon, trainer, energy}``. Energy
becomes one learnable budget category (its probability can grow), targeted entropy
on the tiny 3-way category distribution actually keeps it alive, and the
within-energy card softmax still spans every colour, so colour / archetype
exploration stays open.

Key identity (keeps serving simple): sampling category ``c ~ P(cat)`` then card
``a ~ P(card | c)`` is identical to sampling a candidate from a flat softmax whose
logit is ``log P(cat_a) + log P(a | cat_a)``. So the serving forward only adjusts
the candidate logits by the category log-prob; the learner uses the two factors
separately (so it can regularise the category distribution).

Pure numpy -- the serving sampler (:mod:`src.net.deck_sample`,
:func:`src.net.cb.build_deck`) calls :func:`factored_pick`; the torch learner mirrors
the math on full-pool logits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.deck import card_kind
from src.net.embedding import CardEmbeddingIndex

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.deck import CardPool

CAT_POKEMON = 0
CAT_TRAINER = 1
CAT_ENERGY = 2
N_CATEGORIES = 3

_KIND_TO_CAT = {"pokemon": CAT_POKEMON, "trainer": CAT_TRAINER, "energy": CAT_ENERGY}


def category_of_rows(pool: CardPool) -> NDArray[np.intp]:
    """Map each shared-embedding pool row -> its category index (``CardEmbeddingIndex``
    order, i.e. ``sorted(pool.ids())``)."""
    index = CardEmbeddingIndex(pool)
    return np.array(
        [_KIND_TO_CAT[card_kind(pool, cid)] for cid in index.ids], dtype=np.intp,
    )


def _log_softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable log-softmax over a 1-D vector (finite entries only)."""
    m = logits.max()
    shifted = logits - m
    return shifted - np.log(np.exp(shifted).sum())


def factored_logp(
    cat_logits: NDArray[np.float64],
    card_logits: NDArray[np.float64],
    cand_cats: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Joint per-candidate log-prob ``log P(cat) + log P(card | cat)``.

    ``cat_logits`` is ``(N_CATEGORIES,)``; ``card_logits`` / ``cand_cats`` are
    ``(K,)`` over the legal candidates. Categories with no legal candidate are masked
    out of the category softmax. The returned ``(K,)`` log-probs sum (in exp space) to
    1 over the candidates, so they are a proper flat distribution to sample from.
    """
    present = np.unique(cand_cats)
    cat_mask = np.full(cat_logits.shape[0], -np.inf)
    cat_mask[present] = cat_logits[present]
    cat_logp = _log_softmax(cat_mask)  # (N_CATEGORIES,)

    card_logp_within = np.full(card_logits.shape[0], -np.inf)
    for c in present:
        idx = np.where(cand_cats == c)[0]
        card_logp_within[idx] = _log_softmax(card_logits[idx])
    return cat_logp[cand_cats] + card_logp_within


def factored_pick(
    cat_logits: NDArray[np.float64],
    card_logits: NDArray[np.float64],
    cand_cats: NDArray[np.intp],
    rng: np.random.Generator | None,
    *,
    greedy: bool,
) -> tuple[int, float]:
    """Pick a candidate under the factored policy; return ``(index, log-prob)``.

    ``greedy`` takes the joint argmax; otherwise samples from the joint (equivalent
    to sampling category then card). ``rng`` is required when not greedy.
    """
    joint = factored_logp(cat_logits, card_logits, cand_cats)
    if greedy:
        pick = int(np.argmax(joint))
    else:
        if rng is None:
            msg = "sampling factored_pick requires an rng"
            raise ValueError(msg)
        pick = int(rng.choice(joint.shape[0], p=np.exp(joint)))
    return pick, float(joint[pick])
