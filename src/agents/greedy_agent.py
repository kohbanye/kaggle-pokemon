"""Greedy baseline: a deterministic one-ply policy that never wastes a turn.

The random agent routinely ends its turn without attacking and, fatally, stops
benching Pokemon -- so its lone Active gets Knocked Out with an empty bench and
it loses on "no Pokemon in the Active Spot". Greedy fixes the **MAIN** decision
each turn by playing the natural Pokemon TCG turn order:

  1. Develop the board with every available non-turn-ending action
     (attach energy > evolve > play/bench > ability). Benching basics and
     attaching energy each turn is what keeps greedy alive and attacking.
  2. When nothing left to develop, attack -- the highest-damage one when
     damages are known; this ends the turn.
  3. Otherwise end the turn.

A ``turnActionCount`` guard stops greedy from looping forever on a repeatable
develop option (it skips to attack/end). Every *non*-MAIN sub-selection
(choosing cards, targets, energies, yes/no, counts) falls back to a fixed, legal
default, so the whole policy is deterministic -- no RNG, reproducible.

Attack damages are *injected* (``attack_damage``: ``attackId -> damage``) by the
runner from ``cg.all_attack()``; with no map, greedy still prefers attacking but
breaks ties by the last (usually stronger) attack.
"""

from __future__ import annotations

from .base import (
    OPT_ABILITY,
    OPT_ATTACH,
    OPT_ATTACK,
    OPT_END,
    OPT_EVOLVE,
    OPT_PLAY,
    SEL_MAIN,
    Agent,
    legal_fallback,
)

# Develop actions, best-first. Attach energy and evolve advance the board the
# most; ability is last because some abilities are re-offered and could loop.
_DEVELOP_PRIORITY = (OPT_ATTACH, OPT_EVOLVE, OPT_PLAY, OPT_ABILITY)

# After this many actions in a single turn, stop developing and force the turn
# to progress (attack or end). Guards against a repeatable develop option.
_MAX_DEVELOP_ACTIONS = 40


class GreedyAgent(Agent):
    name = "greedy"

    def __init__(
        self,
        deck: list[int],
        attack_damage: dict[int, int] | None = None,
    ) -> None:
        super().__init__(deck)
        self.attack_damage = dict(attack_damage or {})

    def act(self, obs: dict) -> list[int]:
        select = obs["select"]
        if int(select.get("type", -1)) == SEL_MAIN and int(select["maxCount"]) >= 1:
            current = obs.get("current") or {}
            idx = self._choose_main(select["option"], current.get("turnActionCount"))
            if idx is not None:
                return [idx]
        # Non-MAIN selection (or empty MAIN): deterministic legal default.
        return legal_fallback(select)

    def _choose_main(
        self,
        options: list[dict],
        turn_action_count: int | None,
    ) -> int | None:
        by_type: dict[int, list[int]] = {}
        for i, opt in enumerate(options):
            by_type.setdefault(int(opt["type"]), []).append(i)

        looping = (
            turn_action_count is not None
            and turn_action_count > _MAX_DEVELOP_ACTIONS
        )
        # 1. Develop the board first (non-turn-ending actions), unless the turn
        #    has dragged on -- then skip straight to attack/end (loop guard).
        if not looping:
            for opt_type in _DEVELOP_PRIORITY:
                if opt_type in by_type:
                    return by_type[opt_type][0]

        # 2. Attack -- the highest-damage one; this ends the turn.
        if OPT_ATTACK in by_type:
            return self._best_attack(options, by_type[OPT_ATTACK])

        # 3. Nothing left to do: end the turn.
        if OPT_END in by_type:
            return by_type[OPT_END][0]
        # 4. Forced selection with no END offered: take the first option.
        return 0 if options else None

    def _best_attack(self, options: list[dict], idxs: list[int]) -> int:
        best_idx, best_dmg = idxs[0], -1
        for i in idxs:
            dmg = self.attack_damage.get(options[i].get("attackId"), -1)
            if dmg > best_dmg:
                best_idx, best_dmg = i, dmg
        # No damages known -> prefer the last attack (heuristically the stronger).
        return best_idx if best_dmg >= 0 else idxs[-1]
