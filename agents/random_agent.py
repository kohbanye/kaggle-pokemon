"""A uniform-random legal-move agent — the Phase 0 floor baseline.

Mirrors the verified logic in ``scripts/sim_smoke.py``: on the initial selection
(``obs.select is None``) it returns its 60-card deck; otherwise it samples
``maxCount`` distinct option indices uniformly at random. The observation is
decoded with the engine's ``to_observation_class`` (imported lazily so importing
this module never drags in the gitignored engine, and so tests can inject a fake
decoder to exercise the logic without the simulator).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence

Decoder = Callable[[Mapping[str, object]], object]


def _default_decoder(obs: Mapping[str, object]) -> object:
    from cg.api import to_observation_class  # noqa: PLC0415

    return to_observation_class(obs)


class RandomAgent:
    """Deck-fixed random policy. Reproducible when constructed with a ``seed``."""

    def __init__(
        self,
        deck: Sequence[int],
        *,
        seed: int | None = None,
        decoder: Decoder | None = None,
    ) -> None:
        self._deck = list(deck)
        self._rng = random.Random(seed)  # noqa: S311  (game RNG, not cryptographic)
        self._decode = decoder or _default_decoder

    def __call__(self, obs: Mapping[str, object]) -> list[int]:
        decoded = self._decode(obs)
        select = decoded.select  # type: ignore[attr-defined]
        if select is None:
            return list(self._deck)
        count: int = select.maxCount
        n_options: int = len(select.option)
        return self._rng.sample(range(n_options), count)
