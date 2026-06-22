"""Deck-construction (CB) head: build a legal 60-card deck card-by-card.

At init the engine asks for a 60-card deck and an illegal one is an instant loss,
so the CB head must emit *only* legal decks. It builds the deck **autoregressively
with an LSTM** (Phase 5c): the LSTM consumes the embedding of the card just picked
and its hidden state ``h_t`` carries the running composition, so each slot is
scored *conditioned on the cards already chosen* (it knows how much energy /
how many attackers it has). Each step is restricted to
:func:`src.deck.legal_next_ids` (copies-by-name cap, single ACE SPEC, keep a
Basic). Greedy takes the highest-scoring legal card (deterministic); sampling
draws from the masked softmax (a mixed deck strategy).

Conditioning on the partial deck is what lets greedy decode produce a *balanced,
functional* deck -- the context-free predecessor collapsed to e.g. 0 energy
(Phase 5b finding). Strength comes from BC warm-start and OSFP self-play.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from src.deck import DECK_SIZE, card_kind, legal_next_ids
from src.net.embedding import CardEmbeddingIndex
from src.net.nn import softmax

if TYPE_CHECKING:
    from src.deck import CardPool
    from src.net.features import CardFeatures
    from src.net.model import PolicyValueNet

# Greedy decode caps each card type at the net's own *sampled* mean count, over
# this many sample decks. Greedy argmax otherwise compounds the single highest-
# probability card (the lone Basic Energy) into a degenerate ~all-energy deck even
# when the learned distribution is balanced; the cap reins that in deterministically
# without constraining the sampled decks that self-play actually trains on.
_CAP_SAMPLES = 8


def build_deck(
    net: PolicyValueNet,
    pool: CardPool,
    feats: CardFeatures,
    rng: np.random.Generator | None = None,
    *,
    greedy: bool = True,
) -> list[int]:
    """Build one legal 60-card deck autoregressively (LSTM over the pick sequence).

    ``greedy`` (default) takes the highest-scoring legal card at each step,
    **capped per card type at the net's own sampled-mean composition** so argmax
    can't compound the lone Basic Energy into a degenerate all-energy deck --
    deterministic, for reproducible evaluation and submission. Otherwise samples
    (uncapped) from the masked softmax using ``rng`` (which must be provided); the
    sampled decks are what self-play trains on, so they stay free to explore any
    composition. Returns the deck in pick order.
    """
    if not greedy and rng is None:
        msg = "sampling deck construction requires an rng"
        raise ValueError(msg)
    if rng is None:  # greedy path: rng is unused for picks, but bind it for typing
        rng = np.random.default_rng()
    caps = _type_caps(net, pool, feats) if greedy else None
    return _decode_deck(net, pool, feats, rng, greedy=greedy, caps=caps)


def _decode_deck(  # noqa: PLR0913 - the decode threads net + pool + rng + caps
    net: PolicyValueNet,
    pool: CardPool,
    feats: CardFeatures,
    rng: np.random.Generator,
    *,
    greedy: bool,
    caps: dict[str, int] | None,
) -> list[int]:
    """Autoregressive LSTM deck decode; ``caps`` (greedy only) bounds each type."""
    index = CardEmbeddingIndex(pool)
    if not index.ids:
        return []
    # Candidate features (fixed ⊕ embedding), scored once per step against h_t.
    cand_matrix = index.matrix(feats, net.params["cb_embed"])
    cb_embed = net.params["cb_embed"]
    n_hidden = net.config.lstm_hidden
    h = np.zeros(n_hidden, dtype=np.float64)
    c = np.zeros(n_hidden, dtype=np.float64)
    x = net.params["cb_start"]  # t=0 input token (empty deck)

    deck: list[int] = []
    kinds: Counter[str] = Counter()
    while len(deck) < DECK_SIZE:
        h, c = net.lstm_step(x, h, c)
        legal = legal_next_ids(deck, pool)
        if not legal:  # the mask should always leave a legal completion
            break
        if caps is not None:  # drop types already at their cap (keep >=1 legal card)
            under = [c for c in legal if kinds[card_kind(pool, c)] < caps.get(
                card_kind(pool, c), DECK_SIZE)]
            if under:
                legal = under
        candidates = sorted(legal)
        rows = [index.row(cid) for cid in candidates]
        logits = net.card_logits_with_state(h, cand_matrix[rows])
        if greedy:
            pick = candidates[int(np.argmax(logits))]
        else:
            pick = candidates[int(rng.choice(len(candidates), p=softmax(logits)))]
        deck.append(pick)
        kinds[card_kind(pool, pick)] += 1
        x = cb_embed[index.row(pick)]  # feed the picked card's embedding
    return deck


def _type_caps(
    net: PolicyValueNet, pool: CardPool, feats: CardFeatures,
) -> dict[str, int]:
    """Per-type greedy cap = the net's mean type counts over sampled decks.

    Data-driven (no hardcoded composition) and deterministic (fixed seed), so the
    greedy deck reproduces the model's central tendency instead of the argmax mode.
    """
    rng = np.random.default_rng(0)
    totals: Counter[str] = Counter()
    for _ in range(_CAP_SAMPLES):
        for cid in _decode_deck(net, pool, feats, rng, greedy=False, caps=None):
            totals[card_kind(pool, cid)] += 1
    return {kind: round(n / _CAP_SAMPLES) for kind, n in totals.items()}
