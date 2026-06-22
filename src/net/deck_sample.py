"""Sample a deck from the CB head **and record per-pick behaviour log-probs**.

The V-Trace/PPO deck arm needs the actor's ``log μ(pick_t | prefix)`` at each of
the 60 build steps to form the importance ratio. :func:`~src.net.cb.build_deck`
returns only the deck, so this is the sampling decode that also returns the
log-probs -- the masked-softmax log-probability of each sampled pick over the legal
candidates (exactly what the learner's masked ``log_softmax`` reproduces).

Mirrors the (uncapped) sampling path of ``src.net.cb._decode_deck``; greedy / the
type caps are eval-only and don't need log-probs.
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


def sample_deck_with_logp(
    net: PolicyValueNet,
    pool: CardPool,
    feats: CardFeatures,
    rng: np.random.Generator,
) -> tuple[list[int], list[float]]:
    """Sample one legal 60-card deck; return ``(deck, per-pick log-probs)``.

    The deck is in CB pick order; ``logp[t]`` is the log-probability the CB head
    assigned the card picked at step ``t`` (masked softmax over the legal set).
    """
    index = CardEmbeddingIndex(pool)
    if not index.ids:
        return [], []
    cand_matrix = index.matrix(feats, net.params["cb_embed"])
    cb_embed = net.params["cb_embed"]
    n_hidden = net.config.lstm_hidden
    h = np.zeros(n_hidden, dtype=np.float64)
    c = np.zeros(n_hidden, dtype=np.float64)
    x = net.params["cb_start"]

    deck: list[int] = []
    logps: list[float] = []
    while len(deck) < DECK_SIZE:
        h, c = net.lstm_step(x, h, c)
        legal = legal_next_ids(deck, pool)
        if not legal:
            break
        candidates = sorted(legal)
        rows = [index.row(cid) for cid in candidates]
        probs = softmax(net.card_logits_with_state(h, cand_matrix[rows]))
        idx = int(rng.choice(len(candidates), p=probs))
        deck.append(candidates[idx])
        logps.append(float(np.log(probs[idx] + 1e-12)))
        x = cb_embed[index.row(candidates[idx])]
    return deck, logps
