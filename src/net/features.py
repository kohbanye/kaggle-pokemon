"""Fixed per-card numeric features (the net's card representation).

The paper uses a learned card embedding. We start one level simpler and more
sample-efficient: a *fixed* numeric feature vector per card, derived from the
engine card/attack stats the runner injects, and let the net learn an MLP
projection of it (that projection is our "card embedding"). Fixed features keep
the Phase-3 backward pass a plain MLP -- no shared-embedding multi-path
gradients -- and let an unseen card id fall back to a zero vector instead of a
missing table row. A learned lookup embedding can be ablated in later phases.

The feature vector concatenates (length :data:`CARD_FEAT_DIM`):

- card-type one-hot (Pokemon / Item / Tool / Supporter / Stadium / Basic- /
  Special-Energy);
- the three prize-relevant flags (basic, ex, mega);
- normalised HP and retreat cost;
- the Pokemon/energy type one-hot and the weakness one-hot;
- the card's best attack: normalised max damage, max cost, attack count, and a
  "has any attack" flag.

:class:`CardFeatures` is built from the injected engine dict (``cards`` +
``attacks``) and is a pure object -- it needs neither ``cg`` nor pandas, so it
unit-tests natively with a hand-written engine dict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Mirrors of cg.api enums (sizes only -- we never import the Linux-only engine).
NUM_CARD_TYPES = 7  # CardType 0..6
NUM_ENERGY_TYPES = 12  # EnergyType 0..11 (COLORLESS..TEAM_ROCKET)

# Normalisers chosen so typical values land in roughly [0, 1.5]; exact scale is
# irrelevant (the trunk's first linear layer rescales), they just keep inputs
# numerically comparable.
_HP_NORM = 100.0
_RETREAT_NORM = 4.0
_DAMAGE_NORM = 200.0
_COST_NORM = 5.0
_ATTACK_COUNT_NORM = 3.0

# Feature-block widths, in concatenation order.
_FLAG_WIDTH = 3  # basic, ex, mega
_SCALAR_HP_RETREAT_WIDTH = 2
_ATTACK_WIDTH = 4  # max_damage, max_cost, n_attacks, has_attack

CARD_FEAT_DIM = (
    NUM_CARD_TYPES
    + _FLAG_WIDTH
    + _SCALAR_HP_RETREAT_WIDTH
    + NUM_ENERGY_TYPES
    + NUM_ENERGY_TYPES
    + _ATTACK_WIDTH
)


def _onehot(value: int | None, size: int) -> list[float]:
    """One-hot of length ``size``; all-zeros for None / out-of-range."""
    vec = [0.0] * size
    if value is not None and 0 <= value < size:
        vec[value] = 1.0
    return vec


def _attack_summary(
    attack_ids: list[int],
    attacks: dict[int, dict],
) -> tuple[float, float, int]:
    """Max damage, max cost length and count across a card's known attacks."""
    max_dmg = 0
    max_cost = 0
    n = 0
    for aid in attack_ids:
        info = attacks.get(aid)
        if info is None:
            continue
        n += 1
        max_dmg = max(max_dmg, int(info.get("dmg", 0)))
        max_cost = max(max_cost, len(info.get("cost", [])))
    return float(max_dmg), float(max_cost), n


class CardFeatures:
    """Cache of fixed per-card feature vectors, plus the raw engine stats.

    Holds the injected ``cards`` / ``attacks`` dicts (so the encoder can read raw
    fields like a target's attack damage) and a lazily-filled ``card_id ->
    vector`` cache. :meth:`vector` returns a zero vector for an unknown id, so the
    encoder is robust to ids the engine dump did not cover.
    """

    def __init__(self, engine: dict | None = None) -> None:
        engine = engine or {}
        self.cards: dict[int, dict] = engine.get("cards", {})
        self.attacks: dict[int, dict] = engine.get("attacks", {})
        self._zero: NDArray[np.float64] = np.zeros(CARD_FEAT_DIM, dtype=np.float64)
        self._cache: dict[int, NDArray[np.float64]] = {}

    def _build(self, card_id: int) -> NDArray[np.float64]:
        card = self.cards.get(card_id)
        if card is None:
            return self._zero
        max_dmg, max_cost, n_attacks = _attack_summary(
            card.get("attacks", []), self.attacks,
        )
        feats: list[float] = []
        feats += _onehot(card.get("ctype"), NUM_CARD_TYPES)
        feats += [
            float(bool(card.get("basic"))),
            float(bool(card.get("ex"))),
            float(bool(card.get("mega"))),
        ]
        feats += [
            float(card.get("hp", 0)) / _HP_NORM,
            float(card.get("retreat", 0)) / _RETREAT_NORM,
        ]
        feats += _onehot(card.get("type"), NUM_ENERGY_TYPES)
        feats += _onehot(card.get("weak"), NUM_ENERGY_TYPES)
        feats += [
            max_dmg / _DAMAGE_NORM,
            max_cost / _COST_NORM,
            n_attacks / _ATTACK_COUNT_NORM,
            1.0 if n_attacks else 0.0,
        ]
        return np.asarray(feats, dtype=np.float64)

    def vector(self, card_id: int | None) -> NDArray[np.float64]:
        """Fixed feature vector for ``card_id`` (zeros if None / unknown)."""
        if card_id is None:
            return self._zero
        cached = self._cache.get(card_id)
        if cached is None:
            cached = self._build(card_id)
            self._cache[card_id] = cached
        return cached
