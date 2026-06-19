"""Random baseline: pick a legal selection uniformly at random.

Mirrors the reference ``main.py`` (always returns ``maxCount`` distinct option
indices), but with an *instance-local* RNG so the harness can reseed it per game
for reproducible agent behaviour (engine randomness is separate and not seedable
-- see ``scripts/run_eval.py``).
"""

from __future__ import annotations

import random

from .base import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, deck: list[int], seed: int = 0) -> None:
        super().__init__(deck)
        # Game-play randomness, not security; a stdlib PRNG is exactly right.
        self.rng = random.Random(seed)  # noqa: S311

    def reset(self, seed: int) -> None:
        self.rng.seed(seed)

    def act(self, obs: dict) -> list[int]:
        select = obs["select"]
        n_options = len(select["option"])
        max_count = int(select["maxCount"])
        if max_count <= 0:
            return []
        return self.rng.sample(range(n_options), max_count)
