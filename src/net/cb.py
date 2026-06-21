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

from typing import TYPE_CHECKING

import numpy as np

from src.deck import DECK_SIZE, legal_next_ids
from src.net.embedding import CardEmbeddingIndex
from src.net.nn import softmax

if TYPE_CHECKING:
    from src.deck import CardPool
    from src.net.features import CardFeatures
    from src.net.model import PolicyValueNet


def build_deck(
    net: PolicyValueNet,
    pool: CardPool,
    feats: CardFeatures,
    rng: np.random.Generator | None = None,
    *,
    greedy: bool = True,
) -> list[int]:
    """Build one legal 60-card deck autoregressively (LSTM over the pick sequence).

    ``greedy`` (default) takes the highest-scoring legal card at each step --
    deterministic, for reproducible evaluation. Otherwise samples from the masked
    softmax using ``rng`` (which must be provided). Returns the deck in pick order.
    """
    if not greedy and rng is None:
        msg = "sampling deck construction requires an rng"
        raise ValueError(msg)
    if rng is None:  # greedy path: rng is unused, but bind it for a clean type
        rng = np.random.default_rng()

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
    while len(deck) < DECK_SIZE:
        h, c = net.lstm_step(x, h, c)
        legal = legal_next_ids(deck, pool)
        if not legal:  # the mask should always leave a legal completion
            break
        candidates = sorted(legal)
        rows = [index.row(cid) for cid in candidates]
        logits = net.card_logits_with_state(h, cand_matrix[rows])
        if greedy:
            pick = candidates[int(np.argmax(logits))]
        else:
            pick = candidates[int(rng.choice(len(candidates), p=softmax(logits)))]
        deck.append(pick)
        x = cb_embed[index.row(pick)]  # feed the picked card's embedding
    return deck
