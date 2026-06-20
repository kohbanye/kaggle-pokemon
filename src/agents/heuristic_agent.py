"""Heuristic policy (Phase 2): a state-aware, search-free one-ply agent.

Where ``greedy`` is *blind* -- it picks an option by its **type** alone (attach
energy, then evolve, then play, then attack the hardest) and never reads the
board -- ``heuristic`` reads the observation and chooses the best option *within*
each type:

  * **attach_target** -- attach energy to the Pokemon that most advances an
    attack (one that crosses an attack's cost threshold, the active attacker, a
    Pokemon still short of its biggest attack), instead of greedy's "first
    offered" target that dumps energy onto an already-fuelled Active.
  * **attack_ko** -- prefer an attack that Knocks Out the Defending Pokemon
    (weighted by the prizes it yields: ex = 2, Mega ex = 3), not merely the
    biggest raw number.
  * **weakness** -- double damage when the Defender is Weak to the attacker's
    type, so KO detection and attack ranking are accurate.
  * **bench_dev** -- when the Bench is thin, play a Basic to it (don't get
    Knocked Out with an empty Bench); otherwise prefer draw/search Supporters.
  * **retreat** -- retreat an Active that the opponent can Knock Out next turn
    when a benched attacker can take over (greedy never retreats).
  * **promote** -- when promoting a Pokemon to the Active Spot (set-up, after a
    KO, or a switch), pick the strongest available one, not option ``0``.

Each feature is a flag on :class:`HeuristicConfig` so the runner can ablate them
one at a time (``heuristic`` = all on; ``heuristic_no_<flag>`` = that one off).
With every flag off the policy reduces to greedy's develop-then-attack order.

The module also exposes :func:`evaluate_state`, a single scalar board-quality
score built from the same features (prize race, board HP, energy, Bench, KO
threat). It is the Phase-2 "state evaluation function" and is designed to be the
**leaf evaluator** for the Phase-3 search; the policy and the evaluator share the
same KO/affordability primitives.

Like every agent here this is a pure ``dict -> list[int]`` policy: engine card
and attack stats are *injected* (``engine={"cards": ..., "attacks": ...}``) by
the runner, never imported, so it unit-tests natively with no ``cg`` engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import (
    AREA_ACTIVE,
    AREA_BENCH,
    AREA_HAND,
    CARD_ITEM,
    CARD_POKEMON,
    CARD_SUPPORTER,
    CARD_TOOL,
    CTX_SETUP_ACTIVE,
    CTX_SWITCH,
    CTX_TO_ACTIVE,
    ENERGY_COLORLESS,
    ENERGY_RAINBOW,
    OPT_ABILITY,
    OPT_ATTACH,
    OPT_ATTACK,
    OPT_CARD,
    OPT_END,
    OPT_EVOLVE,
    OPT_PLAY,
    OPT_RETREAT,
    SEL_CARD,
    SEL_MAIN,
    Agent,
    legal_fallback,
)

_NEG_INF = float("-inf")

# After this many actions in a turn, stop developing and progress (loop guard,
# mirrors greedy: some develop options are re-offered and could loop forever).
_MAX_DEVELOP_ACTIONS = 40
# Below this many benched Pokemon, prioritise playing a Basic to the Bench.
_BENCH_MIN = 2

# An attack that secures a Knock Out dominates any chip damage; the prize value
# of the target breaks ties among lethal attacks (take down the ex first).
_LETHAL_BASE = 100_000.0
_LETHAL_PRIZE = 1_000.0

# evaluate_state weights -- initial and *untuned*; this scalar's job in Phase 2
# is sane ordering (a KO/extra prize beats chip, a lost Active is near-terminal),
# and in Phase 3 it becomes the search leaf evaluator that gets tuned then.
_W_PRIZE = 30.0
_W_HP = 5.0
_W_ENERGY = 1.0
_W_BENCH = 2.0
_W_KO = 10.0
_W_NO_ACTIVE = 100.0
_BENCH_CAP = 3
_PLAYER_COUNT = 2


@dataclass
class HeuristicConfig:
    """Feature flags for the heuristic policy (all on = full ``heuristic``).

    Turning a flag off reverts that one decision to greedy's behaviour, so a
    full-vs-``no_<flag>`` match measures exactly that feature's contribution.
    """

    attach_target: bool = True
    attack_ko: bool = True
    weakness: bool = True
    bench_dev: bool = True
    retreat: bool = True
    promote: bool = True

    @classmethod
    def with_disabled(cls, flag: str) -> HeuristicConfig:
        """Full config with a single ``flag`` turned off (for ablation)."""
        cfg = cls()
        if not hasattr(cfg, flag):
            msg = f"unknown heuristic feature {flag!r}"
            raise AttributeError(msg)
        setattr(cfg, flag, False)
        return cfg


# --- pure board / engine helpers (shared by the policy and the evaluator) ---


def _active(player: dict) -> dict | None:
    """The Active Pokemon dict, or None (empty spot or face-down)."""
    spot = player.get("active") or []
    return spot[0] if spot else None


def _all_pokemon(player: dict) -> list[dict]:
    """Every in-play Pokemon (Active + Bench), skipping a face-down Active."""
    pokemon = list(player.get("bench") or [])
    active = _active(player)
    if active is not None:
        pokemon.append(active)
    return pokemon


def _pokemon_at(player: dict, area: int, index: int) -> dict | None:
    """The card an option points at, by ``(area, index)`` in ``player``."""
    if index < 0:
        return None
    if area == AREA_ACTIVE:
        spot = player.get("active") or []
    elif area == AREA_BENCH:
        spot = player.get("bench") or []
    elif area == AREA_HAND:
        spot = player.get("hand") or []
    else:
        return None
    return spot[index] if 0 <= index < len(spot) else None


def _can_afford(cost: list[int], energies: list[int]) -> bool:
    """True if ``energies`` can pay ``cost`` (a list of required EnergyTypes).

    Colored requirements are paid by the same colour or a RAINBOW; COLORLESS
    requirements by any leftover energy. Conservative (TEAM_ROCKET and other
    multi-colour specials are not treated as wildcards), which is fine: we use
    this for "would one more energy enable an attack?" and KO-threat estimates.
    """
    pool: dict[int, int] = {}
    for e in energies:
        pool[e] = pool.get(e, 0) + 1
    colorless = 0
    for c in cost:
        if c == ENERGY_COLORLESS:
            colorless += 1
        elif pool.get(c, 0) > 0:
            pool[c] -= 1
        elif pool.get(ENERGY_RAINBOW, 0) > 0:
            pool[ENERGY_RAINBOW] -= 1
        else:
            return False
    return sum(pool.values()) >= colorless


def _eff_damage(
    attacker_type: int,
    target_card: dict | None,
    base_dmg: int,
    *,
    weakness: bool,
) -> int:
    """Base attack damage, doubled when the target is Weak to the attacker."""
    if (
        weakness
        and base_dmg > 0
        and target_card is not None
        and target_card.get("weak") == attacker_type
    ):
        return base_dmg * 2
    return base_dmg


def _prize_value(card: dict | None) -> int:
    """Prizes the opponent takes for Knocking this out: Mega ex 3, ex 2, else 1."""
    if card is None:
        return 1
    if card.get("mega"):
        return 3
    if card.get("ex"):
        return 2
    return 1


def _best_affordable_damage(
    pokemon: dict | None,
    target_card: dict | None,
    cards: dict,
    attacks: dict,
    *,
    weakness: bool,
) -> int:
    """Most effective damage ``pokemon`` could deal *now* with its current energy.

    Used for KO detection / KO-risk (effect-only ``dmg=0`` attacks contribute 0,
    same blind spot greedy has). Returns 0 if the Pokemon or its data is unknown.
    """
    if pokemon is None:
        return 0
    card = cards.get(pokemon.get("id"))
    if card is None:
        return 0
    energies = pokemon.get("energies") or []
    attacker_type = card.get("type", ENERGY_COLORLESS)
    best = 0
    for aid in card.get("attacks", []):
        info = attacks.get(aid)
        if info is None or not _can_afford(info["cost"], energies):
            continue
        eff = _eff_damage(attacker_type, target_card, info["dmg"], weakness=weakness)
        best = max(best, eff)
    return best


def _hp_fraction(player: dict) -> float:
    """Sum of current/max HP across the player's in-play Pokemon (board health)."""
    total = 0.0
    for pk in _all_pokemon(player):
        mx = pk.get("maxHp") or 0
        if mx > 0:
            total += pk.get("hp", 0) / mx
    return total


def _energy_total(player: dict) -> int:
    """Total energy attached across the player's in-play Pokemon."""
    return sum(len(pk.get("energies") or []) for pk in _all_pokemon(player))


def evaluate_state(
    state: dict,
    my_index: int,
    cards: dict,
    attacks: dict,
    *,
    weakness: bool = True,
) -> float:
    """Scalar board quality from ``my_index``'s perspective (higher = better).

    The Phase-2 state evaluation function and the Phase-3 search leaf evaluator.
    Combines the prize race (dominant term -- fewest prizes left wins), board
    HP, attached energy, Bench development, and immediate KO threat both ways.
    """
    players = state.get("players") or []
    if not 0 <= my_index < len(players) or len(players) < _PLAYER_COUNT:
        return 0.0
    me = players[my_index]
    opp = players[1 - my_index]

    score = _W_PRIZE * (len(opp.get("prize") or []) - len(me.get("prize") or []))
    score += _W_HP * (_hp_fraction(me) - _hp_fraction(opp))
    score += _W_ENERGY * (_energy_total(me) - _energy_total(opp))
    score += _W_BENCH * (
        min(len(me.get("bench") or []), _BENCH_CAP)
        - min(len(opp.get("bench") or []), _BENCH_CAP)
    )

    my_active = _active(me)
    opp_active = _active(opp)
    if my_active is None:
        score -= _W_NO_ACTIVE
    if opp_active is None:
        score += _W_NO_ACTIVE

    my_card = cards.get(my_active.get("id")) if my_active else None
    opp_card = cards.get(opp_active.get("id")) if opp_active else None
    if opp_active is not None and _best_affordable_damage(
        my_active, opp_card, cards, attacks, weakness=weakness,
    ) >= opp_active.get("hp", 1):
        score += _W_KO * _prize_value(opp_card)
    if my_active is not None and _best_affordable_damage(
        opp_active, my_card, cards, attacks, weakness=weakness,
    ) >= my_active.get("hp", 1):
        score -= _W_KO * _prize_value(my_card)
    return score


class HeuristicAgent(Agent):
    """State-aware one-ply policy; see the module docstring for the features."""

    name = "heuristic"

    def __init__(
        self,
        deck: list[int],
        engine: dict | None = None,
        config: HeuristicConfig | None = None,
    ) -> None:
        super().__init__(deck)
        engine = engine or {}
        self.cards: dict = engine.get("cards", {})
        self.attacks: dict = engine.get("attacks", {})
        self.cfg = config or HeuristicConfig()

    def act(self, obs: dict) -> list[int]:
        select = obs.get("select") or {}
        try:
            idx = self._decide(obs, select)
        except Exception:  # noqa: BLE001 - submission hygiene: never crash a match
            idx = None
        if idx is not None:
            return [idx]
        return legal_fallback(select)

    def _decide(self, obs: dict, select: dict) -> int | None:
        """Return a single chosen option index, or None to use the fallback."""
        max_count = int(select.get("maxCount", 0))
        min_count = int(select.get("minCount", 0))
        if max_count < 1 or min_count > 1:
            return None  # multi-select: legal fallback handles it
        state = obs.get("current") or {}
        stype = int(select.get("type", -1))
        options = select.get("option", [])
        if stype == SEL_MAIN:
            return self._choose_main(options, state)
        if self.cfg.promote and stype == SEL_CARD:
            return self._choose_promote(select, state)
        return None

    # --- MAIN: develop the board (smart targets), then attack ---------------

    def _choose_main(self, options: list[dict], state: dict) -> int | None:
        yidx = int(state.get("yourIndex", 0))
        players = state.get("players") or []
        me = players[yidx] if yidx < len(players) else {}
        opp = players[1 - yidx] if len(players) > 1 else {}

        by_type: dict[int, list[int]] = {}
        for i, opt in enumerate(options):
            by_type.setdefault(int(opt["type"]), []).append(i)

        looping = int(state.get("turnActionCount", 0)) > _MAX_DEVELOP_ACTIONS
        if not looping:
            develop = self._choose_develop(by_type, options, me, opp)
            if develop is not None:
                return develop
        # Attack -- the best one (lethal-aware); this ends the turn.
        if OPT_ATTACK in by_type:
            return self._best_attack(options, by_type[OPT_ATTACK], me, opp)
        if OPT_END in by_type:
            return by_type[OPT_END][0]
        return 0 if options else None

    def _choose_develop(
        self, by_type: dict[int, list[int]], options: list[dict],
        me: dict, opp: dict,
    ) -> int | None:
        """Best non-turn-ending action: greedy's cross-type order, smart within.

        Returns None when there is nothing worth developing (fall through to
        attack/end).
        """
        if OPT_ATTACH in by_type:
            return self._best_attach(options, by_type[OPT_ATTACH], me)
        if OPT_EVOLVE in by_type:
            return by_type[OPT_EVOLVE][0]
        if OPT_PLAY in by_type:
            return self._best_play(options, by_type[OPT_PLAY], me)
        if OPT_ABILITY in by_type:
            return by_type[OPT_ABILITY][0]
        if OPT_RETREAT in by_type and self._should_retreat(me, opp):
            return by_type[OPT_RETREAT][0]
        return None

    def _best_attach(self, options: list[dict], idxs: list[int], me: dict) -> int:
        if not self.cfg.attach_target:
            return idxs[0]
        best_idx, best_score = idxs[0], _NEG_INF
        for i in idxs:
            opt = options[i]
            target = _pokemon_at(
                me, int(opt.get("inPlayArea", -1)), int(opt.get("inPlayIndex", -1)),
            )
            score = self._attach_score(target, me, opt)
            if score > best_score:
                best_idx, best_score = i, score
        return best_idx

    def _attach_score(self, target: dict | None, me: dict, opt: dict) -> float:
        if target is None:
            return -10.0
        is_active = int(opt.get("inPlayArea", -1)) == AREA_ACTIVE
        base = 1.0 if is_active else 0.5
        attached = _pokemon_at(me, AREA_HAND, int(opt.get("index", -1)))
        card = self.cards.get(target.get("id"))
        if card is None or card.get("ctype") != CARD_POKEMON:
            return base
        # A Tool (e.g. damage booster) belongs on the active attacker.
        if attached is not None and self.cards.get(
            attached.get("id"), {},
        ).get("ctype") == CARD_TOOL:
            return base + (2.0 if is_active else 0.0)
        # Energy: favour the target that most advances an attack.
        energies = list(target.get("energies") or [])
        costs = [
            len(self.attacks[aid]["cost"])
            for aid in card.get("attacks", [])
            if aid in self.attacks
        ]
        if not costs:
            return base
        color = ENERGY_RAINBOW
        if attached is not None:
            color = self.cards.get(attached.get("id"), {}).get("type", ENERGY_RAINBOW)
        score = base
        if self._enables_attack(card, energies, color):
            score += 5.0  # crossing a cost threshold is the most valuable attach
        cur = len(energies)
        max_cost = max(costs)
        score += 2.0 if cur < max_cost else -2.0  # productive vs over-fuelling
        score += 0.1 * min(cur, max_cost)  # tip toward the more-developed attacker
        return score

    def _enables_attack(self, card: dict, energies: list[int], color: int) -> bool:
        """True if one more ``color`` energy newly affords some attack of ``card``."""
        extended = [*energies, color]
        for aid in card.get("attacks", []):
            info = self.attacks.get(aid)
            if info is None:
                continue
            if not _can_afford(info["cost"], energies) and _can_afford(
                info["cost"], extended,
            ):
                return True
        return False

    def _best_play(self, options: list[dict], idxs: list[int], me: dict) -> int:
        if not self.cfg.bench_dev:
            return idxs[0]
        need_bench = len(me.get("bench") or []) < _BENCH_MIN
        best_idx, best_score = idxs[0], _NEG_INF
        for i in idxs:
            played = _pokemon_at(me, AREA_HAND, int(options[i].get("index", -1)))
            card = self.cards.get(played.get("id")) if played else None
            score = 0.0
            ctype = card.get("ctype") if card else None
            if ctype == CARD_POKEMON and card and card.get("basic"):
                score = 3.0 if need_bench else 1.0
            elif ctype == CARD_SUPPORTER:
                score = 2.0  # draw / search engine
            elif ctype == CARD_ITEM:
                score = 1.5
            if score > best_score:
                best_idx, best_score = i, score
        return best_idx

    def _best_attack(
        self, options: list[dict], idxs: list[int], me: dict, opp: dict,
    ) -> int:
        my_active = _active(me)
        my_card = self.cards.get(my_active.get("id")) if my_active else None
        my_type = my_card.get("type", ENERGY_COLORLESS) if my_card else ENERGY_COLORLESS
        opp_active = _active(opp)
        opp_card = self.cards.get(opp_active.get("id")) if opp_active else None
        opp_hp = opp_active.get("hp") if opp_active else None

        best_idx, best_score = idxs[0], _NEG_INF
        for i in idxs:
            info = self.attacks.get(options[i].get("attackId"), {})
            eff = _eff_damage(
                my_type, opp_card, info.get("dmg", 0), weakness=self.cfg.weakness,
            )
            if self.cfg.attack_ko and opp_hp is not None and eff >= opp_hp and eff > 0:
                score = _LETHAL_BASE + _LETHAL_PRIZE * _prize_value(opp_card)
            else:
                score = float(eff)
            if score > best_score:
                best_idx, best_score = i, score
        return best_idx

    def _should_retreat(self, me: dict, opp: dict) -> bool:
        if not self.cfg.retreat:
            return False
        my_active = _active(me)
        if my_active is None:
            return False
        opp_active = _active(opp)
        my_card = self.cards.get(my_active.get("id"))
        threat = _best_affordable_damage(
            opp_active, my_card, self.cards, self.attacks, weakness=self.cfg.weakness,
        )
        if threat < my_active.get("hp", 1):
            return False  # not doomed -- keep attacking
        # Only retreat if a benched Pokemon can actually take over the attack.
        opp_card = self.cards.get(opp_active.get("id")) if opp_active else None
        return any(
            _best_affordable_damage(
                pk, opp_card, self.cards, self.attacks, weakness=self.cfg.weakness,
            )
            > 0
            for pk in (me.get("bench") or [])
        )

    # --- CARD: choose the strongest Pokemon to promote to the Active Spot ---

    def _choose_promote(self, select: dict, state: dict) -> int | None:
        if int(select.get("context", -1)) not in (
            CTX_SETUP_ACTIVE,
            CTX_SWITCH,
            CTX_TO_ACTIVE,
        ):
            return None
        yidx = int(state.get("yourIndex", 0))
        players = state.get("players") or []
        me = players[yidx] if yidx < len(players) else {}
        opp = players[1 - yidx] if len(players) > 1 else {}
        opp_active = _active(opp)
        opp_card = self.cards.get(opp_active.get("id")) if opp_active else None

        best_idx, best_score = None, _NEG_INF
        for i, opt in enumerate(select.get("option", [])):
            if int(opt.get("type", -1)) != OPT_CARD:
                continue
            pk = _pokemon_at(me, int(opt.get("area", -1)), int(opt.get("index", -1)))
            score = self._promote_score(pk, opp_card)
            if score > best_score:
                best_idx, best_score = i, score
        return best_idx

    def _promote_score(self, pk: dict | None, opp_card: dict | None) -> float:
        if pk is None:
            return _NEG_INF
        dmg = _best_affordable_damage(
            pk, opp_card, self.cards, self.attacks, weakness=self.cfg.weakness,
        )
        hp = pk.get("hp")
        if hp is None:  # a hand card (set-up): use its printed HP
            hp = self.cards.get(pk.get("id"), {}).get("hp", 0)
        return dmg * 2.0 + hp * 0.1  # can-attack-now first, then bulk
