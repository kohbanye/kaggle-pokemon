"""Learned card-embedding index for the deck-construction (CB) head (Phase 5b).

The CB head originally scored each card by its **fixed** features
(:mod:`src.net.features`), so it ranked feature *profiles*, not card *identities*
-- greedy deck decode collapsed to two distinct cards (Phase 4 finding). Phase 5b
adds a **learnable per-card embedding** so the head can prefer specific cards.

Because only ~37 distinct cards appear across the demo decklists but the build
pool has ~1000+ cards, a *pure* embedding would leave >97% of its rows untrained
(random init) and greedy decode would pick that junk. So the embedding is
**concatenated with** the fixed features, not a replacement: every card keeps a
sane fixed-feature default and the embedding only adds identity for cards seen in
training (and is initialised near-zero). The CB head input width therefore becomes
``CARD_FEAT_DIM + embed_dim``.

This module owns the single source of truth for the ``card_id -> embedding row``
mapping, so training (:mod:`src.net.lit` / :mod:`src.net.bc_data`) and serving
(:mod:`src.net.cb`) can never drift: the rows are ``sorted(pool.ids())`` -- the
exact order :func:`src.net.bc_data.cb_supervision` already uses for its CB targets.
A trailing **UNK row** (index ``n_pool``) backs None / unknown ids. Pure numpy, so
it stays on the inference/submission path with no torch dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.net.features import CARD_FEAT_DIM

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.deck import CardPool
    from src.net.features import CardFeatures


class CardEmbeddingIndex:
    """Maps a fixed card pool to embedding-table rows (``sorted(pool.ids())``).

    Row ``i`` is ``ids[i]`` (``ids = sorted(pool.ids())``); the final row
    ``n_pool`` is the UNK row for None / unknown ids. The embedding table the net
    carries is shaped ``(n_pool + 1, embed_dim)`` to match.
    """

    def __init__(self, pool: CardPool) -> None:
        self.pool = pool
        self.ids: list[int] = sorted(pool.ids())
        self.n_pool = len(self.ids)
        self._id_to_row = {cid: i for i, cid in enumerate(self.ids)}
        self._fixed: NDArray[np.float64] | None = None

    def row(self, card_id: int | None) -> int:
        """Embedding-table row for ``card_id`` (the UNK row for None / unknown)."""
        if card_id is None:
            return self.n_pool
        return self._id_to_row.get(card_id, self.n_pool)

    def fixed_matrix(self, feats: CardFeatures) -> NDArray[np.float64]:
        """The pool's fixed feature matrix ``(n_pool, CARD_FEAT_DIM)`` (cached)."""
        if self._fixed is None:
            if not self.ids:
                self._fixed = np.zeros((0, CARD_FEAT_DIM), dtype=np.float64)
            else:
                self._fixed = np.stack([feats.vector(cid) for cid in self.ids])
        return self._fixed

    def matrix(
        self,
        feats: CardFeatures,
        cb_embed: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Concatenated CB input ``(n_pool, CARD_FEAT_DIM + embed_dim)``.

        ``cb_embed`` is the net's embedding table ``(n_pool + 1, embed_dim)``; the
        pool rows ``[:n_pool]`` are concatenated to the fixed features (the UNK row
        is excluded -- pool candidates are always known ids).
        """
        fixed = self.fixed_matrix(feats)
        emb = np.asarray(cb_embed, dtype=np.float64)[: self.n_pool]
        return np.concatenate([fixed, emb], axis=1)
