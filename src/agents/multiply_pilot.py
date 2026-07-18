"""MultiPly Beam Search agent (ported from the public Kaggle notebook
``multiply-agent-best-940-lb``).

The original is a beam search over the engine's ``search_begin``/``search_step``
lookahead API, wrapped around a *deck-specific* heuristic (``AdvancedPolicy``)
that is hardcoded for its own Fighting / Mega-Lucario ex deck. We port it here so
it can (a) pilot our Archaludon deck and (b) run its own deck against our agents.

Two important porting notes:

* The notebook targets an OLDER ``cg`` search API where ``search_begin`` took only
  the ``search_begin_input`` string (the engine determinized hidden info itself)
  and returned an ``ApiResult`` with ``.state``/``.error``. OUR ``cg.api`` requires
  a full explicit determinization -- ``your_deck``/``your_prize``/``opponent_deck``/
  ``opponent_prize``/``opponent_hand``/``opponent_active`` -- and returns a
  ``SearchState`` directly (raising on error). We supply a determinization and
  rewrite the loop accordingly (see ``_determinize`` / ``SEARCH_ALGO``).
* Because the beam only expands nodes on OUR own turn (it freezes at the opponent
  turn boundary), the opponent-deck belief has little effect on the within-turn
  lookahead -- so a coarse opponent determinization is acceptable here.

Like the other bespoke pilots this module imports the Linux-only ``cg`` engine at
load time, so it is NOT imported by ``src/agents/__init__``; ``scripts/run_eval.py``
wires it in lazily under the name ``multiply``. It is crash-proof (legal fallback
on any error) and respects a per-move time budget.
"""

from __future__ import annotations

import os
import random
import sys
import time
from collections import defaultdict

try:
    _ROOT = __file__
except NameError:
    _ROOT = None
_CG_PATH = "/kaggle_simulations/agent"
for _p in ([os.path.dirname(os.path.abspath(_ROOT))] if _ROOT else []) + [_CG_PATH]:
    if _p and _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

from cg.api import (  # noqa: E402
    AreaType, Card, CardType, EnergyType, Observation, OptionType,
    Pokemon, SelectContext, all_card_data, to_observation_class,
)

_SEARCH_OK = False
try:
    from cg.api import search_begin, search_end, search_step  # noqa: E402
    _SEARCH_OK = True
except Exception:
    pass

# ---- Tunable search budget (kept well under the ~1s per-move Kaggle limit) ----
USE_SEARCH = True
SEARCH_TIME_BUDGET = 0.8    # seconds of beam expansion per MAIN decision (rarely binds; engine search is native/fast)
SEARCH_MAX_CANDIDATES = 6   # root branching over the heuristic's top-k order (original value)
BEAM_WIDTH = 3
BEAM_DEPTH = 4              # max within-turn expansion steps

# ---- MultiPly's own deck (its shipped deck.csv) ------------------------------
MULTIPLY_DECK = [
    673, 673, 674, 674, 675, 675, 676, 676,
    676, 677, 677, 677, 678, 678, 678, 678,
    1102, 1102, 1102, 1102, 1123, 1123, 1141, 1141,
    1141, 1141, 1142, 1142, 1142, 1142, 1152, 1152,
    6, 1159, 1182, 1182, 1192, 1192, 1192, 1192,
    1227, 1227, 1227, 1227, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 1182, 677, 1252,
]


class C:
    KYOGRE, SNOVER, MEGA_ABOMASNOW_EX = 721, 722, 723
    MAKUHITA, HARIYAMA = 673, 674
    LUNATONE, SOLROCK = 675, 676
    RIOLU, MEGA_LUCARIO_EX = 677, 678
    BASIC_FIGHTING_ENERGY = 6
    DUSK_BALL, SWITCH, PREMIUM_POWER_PRO, FIGHTING_GONG = 1102, 1123, 1141, 1142
    POKE_PAD, HERO_CAPE, BOSS_ORDERS = 1152, 1159, 1182
    CARMINE, LILLIE_DETERMINATION, GRAVITY_MOUNTAIN = 1192, 1227, 1252
    LUMIOSE_CITY, LILLIES_PEARL, LEGACY_ENERGY = 1267, 1172, 12


MEGA_BRAVE = 983
LOW_DECK_COUNT = 10

all_card = all_card_data()
card_table = {card.cardId: card for card in all_card}

# Basic ids used to pad determinization lists (any legal basics work).
_BASIC_POKE_ID = next(
    (c.cardId for c in all_card if c.cardType == CardType.POKEMON and c.basic), 673,
)
_BASIC_ENERGY_ID = next(
    (c.cardId for c in all_card if c.cardType == CardType.BASIC_ENERGY), 6,
)

# ---- module-level per-turn memory (mirrors the notebook globals) -------------
plan = None
pre_turn = -1
ability_used = False


def _get_deck() -> list[int]:
    for path in ("deck.csv", f"{_CG_PATH}/deck.csv"):
        try:
            with open(path, encoding="utf-8") as f:
                d = [int(x) for x in f.read().splitlines() if x.strip()]
            if d:
                return d
        except Exception:
            continue
    return list(MULTIPLY_DECK)


my_deck = _get_deck()


def get_card(obs, area, index, player_index):
    player = obs.current.players[player_index]
    if area == AreaType.DECK:
        return obs.select.deck[index]
    if area == AreaType.HAND:
        return player.hand[index]
    if area == AreaType.DISCARD:
        return player.discard[index]
    if area == AreaType.ACTIVE:
        return player.active[index]
    if area == AreaType.BENCH:
        return player.bench[index]
    if area == AreaType.PRIZE:
        return player.prize[index]
    if area == AreaType.STADIUM:
        return obs.current.stadium[index]
    if area == AreaType.LOOKING:
        return obs.current.looking[index]
    return None


def prize_count(pokemon):
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == C.LEGACY_ENERGY:
            count -= 1
    for card in pokemon.tools:
        if card.id == C.LILLIES_PEARL and "Lillie" in data.name:
            count -= 1
    return max(0, count)


def target_score(pokemon):
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 2000 + len(pokemon.energies) * 300 + len(pokemon.tools) * 200
    if data.stage2:
        score += 500
    elif data.stage1:
        score += 250
    if pokemon.id in {144, 322, 323, 337}:
        score -= 200
    if pokemon.id == C.SNOVER:
        score += 950
    elif pokemon.id == C.MEGA_ABOMASNOW_EX:
        score += 250
    if pokemon.id == C.RIOLU:
        score += 800
    elif pokemon.id == C.MEGA_LUCARIO_EX:
        score += 100
    return score + pokemon.hp


class AttackPlan:
    def __init__(self, attacker=-1, target=-1, attack_index=-1, remain_hp=-1, needs_energy=False):
        self.attacker, self.target = attacker, target
        self.attack_index, self.remain_hp = attack_index, remain_hp
        self.needs_energy = needs_energy


plan = AttackPlan()


class AdvancedPolicy:
    def __init__(self, obs):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]

        self.field_counts = defaultdict(int)
        self.hand_counts = defaultdict(int)
        self.discard_counts = defaultdict(int)
        self.has_ready_lucario_line = False
        self.has_ready_hariyama_line = False
        self.can_switch = self.can_gust = self.can_attack = self.can_use_mega_brave = False
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0

        self._count_cards()
        self._scan_main_options()

    def choose(self):
        if not self.select.option or self.select.maxCount == 0:
            return []
        if self.context == SelectContext.MAIN:
            self._plan_attack()
        scores = [self._score_option(option) for option in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        self._remember_lunatone_ability(ranked)
        return ranked[: self.select.maxCount]

    def _count_cards(self):
        for pokemon in self.me.active + self.me.bench:
            if pokemon is None:
                continue
            self.field_counts[pokemon.id] += 1
            if pokemon.id in {C.MAKUHITA, C.HARIYAMA} and len(pokemon.energies) >= 3:
                self.has_ready_hariyama_line = True
            if pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX} and len(pokemon.energies) >= 2:
                self.has_ready_lucario_line = True
        for card in self.me.hand:
            self.hand_counts[card.id] += 1
        for card in self.me.discard:
            self.discard_counts[card.id] += 1

    def _scan_main_options(self):
        if self.context != SelectContext.MAIN:
            return
        for option in self.select.option:
            if option.type == OptionType.PLAY:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card.id == C.SWITCH:
                    self.can_switch = True
                elif card.id == C.BOSS_ORDERS:
                    self.can_gust = True
            elif option.type == OptionType.EVOLVE:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card.id == C.HARIYAMA:
                    self.can_gust = True
            elif option.type == OptionType.RETREAT:
                self.can_switch = True
            elif option.type == OptionType.ATTACK:
                self.can_attack = True
                if option.attackId == MEGA_BRAVE:
                    self.can_use_mega_brave = True

    def _my_board(self):
        return self.me.active + self.me.bench

    def _opponent_board(self):
        return self.opponent.active + self.opponent.bench

    def _opponent_has(self, ids):
        return any(p is not None and p.id in ids for p in self._opponent_board())

    def _opponent_is_water_deck(self):
        return self._opponent_has({C.KYOGRE, C.SNOVER, C.MEGA_ABOMASNOW_EX})

    def _opponent_is_crustle_wall(self):
        return self._opponent_has({344, 345})

    def _can_evolve_board_index(self, board_index):
        for option in self.select.option:
            if option.type != OptionType.EVOLVE:
                continue
            target_index = option.inPlayIndex + (1 if option.inPlayArea == AreaType.BENCH else 0)
            if target_index == board_index:
                return True
        return False

    def _base_attack(self, pokemon, attack_index):
        energy_required, base_damage, base_score = 0, 0, 0
        if pokemon.id == C.MEGA_LUCARIO_EX:
            if attack_index == 0:
                energy_required, base_damage = 1, 130
                base_score += 60 * min(3, self.discard_counts[C.BASIC_FIGHTING_ENERGY])
            else:
                energy_required, base_damage = 2, 270
            if self._opponent_is_water_deck() and len(self.opponent.prize) <= 3:
                base_score -= 500
        elif attack_index == 1:
            return None
        elif pokemon.id == C.HARIYAMA:
            energy_required, base_damage = 3, 210
        elif pokemon.id == C.MAKUHITA:
            return None
        elif pokemon.id == C.SOLROCK and self.field_counts[C.LUNATONE] >= 1:
            energy_required, base_damage = 1, 70
        if base_damage <= 0:
            return None
        return energy_required, base_damage, base_score

    def _base_attack_after_evolution(self, pokemon, board_index, attack_index):
        if pokemon.id == C.MAKUHITA and attack_index == 0 and self._can_evolve_board_index(board_index):
            return 3, 210, -100
        return self._base_attack(pokemon, attack_index)

    def _plan_attack(self):
        global plan
        best_score = -1
        plan = AttackPlan()
        if self.state.turn < 2:
            return

        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None:
                continue
            if attacker_index != 0 and not self.can_switch:
                break

            for attack_index in range(2):
                attack = self._base_attack_after_evolution(my_pokemon, attacker_index, attack_index)
                if attack is None:
                    continue
                energy_required, base_damage, base_score = attack
                energy_count = len(my_pokemon.energies)
                if attack_index == 1 and attacker_index == 0 and energy_count >= 2 and not self.can_use_mega_brave:
                    break
                needs_energy = False
                if energy_count < energy_required:
                    if self.hand_counts[C.BASIC_FIGHTING_ENERGY] >= 1 and not self.state.energyAttached:
                        energy_count += 1
                        needs_energy = energy_count >= energy_required
                    if not needs_energy:
                        continue

                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None:
                        continue
                    if target_index != 0 and not self.can_gust:
                        break
                    if self._opponent_is_crustle_wall() and my_pokemon.id == C.MEGA_LUCARIO_EX and op_pokemon.id == 345:
                        continue

                    damage = base_damage
                    op_data = card_table[op_pokemon.id]
                    if op_data.weakness == EnergyType.FIGHTING:
                        damage *= 2
                    elif op_data.resistance == EnergyType.FIGHTING:
                        damage -= 30

                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0:
                        score *= damage / op_pokemon.hp
                    if len(self.opponent.prize) <= prize:
                        score = 500000

                    score += base_score + (220 if attacker_index == 0 else 0) + (300 if target_index == 0 else 0) + energy_count
                    if score > best_score:
                        best_score = score
                        plan = AttackPlan(attacker_index, target_index, attack_index, op_pokemon.hp - damage, needs_energy)

    def _energy_target_score(self, pokemon, active):
        energy_count = len(pokemon.energies)
        score = 8000 + (10 if active else 0)
        if pokemon.id in {C.MAKUHITA, C.HARIYAMA}:
            if pokemon.id == C.HARIYAMA:
                score += 1
            if self._opponent_is_crustle_wall():
                score += 260 if energy_count < 3 else 30
            else:
                score += 100 if energy_count < 3 else 0
                score -= 50 if self.has_ready_hariyama_line else 0
        elif pokemon.id == C.LUNATONE:
            score -= 100
        elif pokemon.id == C.SOLROCK:
            score += 20 if energy_count < 1 else -100
        elif pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX}:
            if pokemon.id == C.MEGA_LUCARIO_EX:
                score += 1
            score += 100 if energy_count < 2 else 0
            score -= 50 if self.has_ready_lucario_line else 0
        return score

    def _score_option(self, option):
        if option.type == OptionType.NUMBER:
            return option.number
        if option.type == OptionType.YES:
            return 100 if self.context == SelectContext.IS_FIRST else 1
        if option.type == OptionType.NO:
            return 0
        if option.type == OptionType.CARD:
            return self._score_card_choice(option)
        if option.type == OptionType.PLAY:
            return self._score_play(option)
        if option.type == OptionType.ATTACH:
            return self._score_attach(option)
        if option.type == OptionType.EVOLVE:
            return self._score_evolve(option)
        if option.type == OptionType.ABILITY:
            return self._score_ability(option)
        if option.type == OptionType.RETREAT:
            return 2000 if plan.attacker >= 1 else -1
        if option.type == OptionType.ATTACK:
            if self._opponent_is_crustle_wall() and self.me.active and self.opponent.active and self.me.active[0].id == C.MEGA_LUCARIO_EX and self.opponent.active[0].id == 345 and plan.target < 0:
                return -1
            return 1100 if (option.attackId == MEGA_BRAVE) == (plan.attack_index == 1) else 1000
        return 0

    def _score_card_choice(self, option):
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0
        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return 2 if card.id == C.SOLROCK and self.state.firstPlayer == self.my_index else 4 if card.id == C.SOLROCK else 3 if card.id == C.RIOLU else 1 if card.id == C.MAKUHITA else 0
        if self.context == SelectContext.TO_HAND:
            score = 200 - self.hand_counts[card.id] * 100
            if card.id == C.MAKUHITA:
                score += (80 if self.field_counts[card.id] < 2 else -20) if self._opponent_is_crustle_wall() else (-10 if self.field_counts[card.id] >= 1 else 10)
            elif card.id == C.HARIYAMA:
                score += (120 if self.field_counts[C.MAKUHITA] >= 1 else -5) if self._opponent_is_crustle_wall() else (20 if self.field_counts[C.MAKUHITA] >= 1 else -20)
            elif card.id == C.LUNATONE:
                score += -250 if self.field_counts[card.id] >= 1 else 60
            elif card.id == C.SOLROCK:
                score += -250 if self.field_counts[card.id] >= 1 else 50
            elif card.id == C.RIOLU:
                score += -150 if (self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 2) else -3 if (self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 1) else 40
            elif card.id == C.MEGA_LUCARIO_EX:
                score += 40 if self.field_counts[C.RIOLU] >= 1 else -15
            elif card.id == C.BASIC_FIGHTING_ENERGY:
                score += 30 if not ability_used or not self.state.energyAttached else -1
            return score
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            return self._energy_target_score(card, option.area == AreaType.ACTIVE)
        return 0

    def _score_active_choice(self, option, card):
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex != self.my_index:
            return 100 if option.index == plan.target - 1 else 0
        score = len(card.energies) * 2
        if option.index == plan.attacker - 1:
            score += 100
        if card.id == C.MEGA_LUCARIO_EX:
            score += 8 if self._opponent_is_water_deck() and len(self.opponent.prize) <= 3 else 20
        elif card.id == C.HARIYAMA and len(card.energies) >= 2:
            score += 45 if self._opponent_is_crustle_wall() else 15
        elif card.id == C.MAKUHITA and len(card.energies) >= 2:
            score += 35 if self._opponent_is_crustle_wall() else 10
        elif card.id == C.SOLROCK:
            score += 5
        elif card.id == C.RIOLU:
            score += 4
        return score

    def _score_play(self, option):
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        data = card_table[card.id]
        if data.cardType == CardType.POKEMON:
            if card.id in {C.LUNATONE, C.SOLROCK} and self.field_counts[card.id] >= 1:
                return -1
            if card.id == C.RIOLU and self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 2:
                return -1
            return 20000
        if card.id == C.SWITCH:
            return 6000 if plan.attacker > 0 else -1
        if card.id == C.PREMIUM_POWER_PRO:
            if self.state.supporterPlayed and plan.remain_hp <= 0:
                return -1
            if not self.can_attack:
                return 3050 if (not self.state.supporterPlayed and self.hand_counts[C.CARMINE] > 0 and self.hand_counts[C.LILLIE_DETERMINATION] == 0 and not self.me.deckCount <= LOW_DECK_COUNT) else -1
            return 5000
        if card.id == C.BOSS_ORDERS:
            return 3200 if plan.target >= 1 else -1
        if card.id == C.CARMINE:
            return -1 if self.me.deckCount <= LOW_DECK_COUNT else 3000
        if card.id == C.LILLIE_DETERMINATION:
            return -1 if self.me.deckCount <= LOW_DECK_COUNT else 3100
        if card.id == C.GRAVITY_MOUNTAIN:
            return 3500 if any(p is not None and card_table[p.id].stage2 for p in self._opponent_board()) else (1200 if self.stadium_id else -1)
        return 10000

    def _score_attach(self, option):
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0
        if card.id == C.HERO_CAPE:
            score = 7000
            if self._opponent_is_water_deck():
                return 12200 if pokemon.id == C.RIOLU else 12800 if pokemon.id == C.MEGA_LUCARIO_EX else score
            if pokemon.id == C.RIOLU:
                score += 100
            elif pokemon.id == C.MEGA_LUCARIO_EX:
                score += 200
            return score
        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
        board_index = option.inPlayIndex if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
        if board_index == plan.attacker and plan.needs_energy:
            score += 200
        return score

    def _score_evolve(self, option):
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0
        if pokemon.id == C.MAKUHITA and plan.target == 0 and not self._opponent_is_crustle_wall():
            return -1
        return 9000 + len(pokemon.energies)

    def _score_ability(self, option):
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card.id == C.LUMIOSE_CITY:
            return 1
        if card.id == C.LUNATONE and self.me.deckCount <= LOW_DECK_COUNT:
            return -1
        return 30000

    def _remember_lunatone_ability(self, ranked):
        global ability_used
        if self.context != SelectContext.MAIN or not ranked:
            return
        option = self.select.option[ranked[0]]
        if option.type != OptionType.ABILITY:
            return
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card is not None and card.id == C.LUNATONE:
            ability_used = True


def evaluate_state(obs):
    st = obs.current
    if st is None:
        return 0.0
    me, op = st.players[st.yourIndex], st.players[1 - st.yourIndex]

    prize_diff = len(op.prize) - len(me.prize)
    if len(me.prize) == 0:
        return 9999999.0
    if len(op.prize) == 0:
        return -9999999.0
    val = prize_diff * 10000.0

    for p in [me.active[0] if me.active else None] + list(me.bench):
        if p is None:
            continue
        val += len(p.energies) * 200.0
        if p.id == C.MEGA_LUCARIO_EX:
            val += 500.0
        elif p.id == C.HARIYAMA:
            val += 300.0
        elif p.id in {C.RIOLU, C.MAKUHITA}:
            val += 100.0

    if me.active and me.active[0] is not None:
        val += me.active[0].hp * 2.0
        if len(me.active[0].energies) >= 2:
            val += 500.0
    if op.active and op.active[0] is not None:
        val -= op.active[0].hp * 2.5

    val += getattr(me, "handCount", len(me.hand or [])) * 10.0
    if getattr(me, "deckCount", 60) < 5:
        val -= 5000.0
    return val


def _pad(ids, n, filler):
    """Return a list of length >= n: the given ids plus filler to reach n."""
    ids = list(ids)
    if len(ids) < n:
        ids += [filler] * (n - len(ids))
    return ids


def _determinize(obs, own_deck):
    """Build the (your/opponent) card-id lists ``search_begin`` requires.

    OUR ``cg.api.search_begin`` needs an explicit determinization of all hidden
    info. We reconstruct our own remaining deck+prizes from the known 60-card
    list minus visible cards, and give the opponent a coarse legal filler (a
    couple of Basic Pokemon + Basic Energy). The beam only expands OUR-turn
    nodes, so the opponent belief barely affects the lookahead.
    """
    st = obs.current
    me = st.players[st.yourIndex]
    op = st.players[1 - st.yourIndex]

    # Own unseen pool = full deck minus everything we can see of ours.
    remaining = defaultdict(int)
    for cid in own_deck:
        remaining[cid] += 1

    def _remove_pokemon(p):
        if p is None:
            return
        remaining[p.id] -= 1
        for e in p.energyCards:
            remaining[e.id] -= 1
        for t in p.tools:
            remaining[t.id] -= 1
        for pre in p.preEvolution:
            remaining[pre.id] -= 1

    for p in (me.active or []) + list(me.bench):
        _remove_pokemon(p)
    for c in me.discard:
        remaining[c.id] -= 1
    if me.hand:
        for c in me.hand:
            remaining[c.id] -= 1
    for pz in me.prize:
        if pz is not None:
            remaining[pz.id] -= 1

    pool = []
    for cid, n in remaining.items():
        if n > 0:
            pool.extend([cid] * n)

    n_prize = len(me.prize)
    n_deck = me.deckCount
    your_prize = _pad(pool[:n_prize], n_prize, _BASIC_ENERGY_ID)
    your_deck = _pad(pool[n_prize:], n_deck, _BASIC_ENERGY_ID)

    # Opponent: coarse legal filler with at least one Basic Pokemon.
    opp_deck = _pad([_BASIC_POKE_ID, _BASIC_POKE_ID], op.deckCount + 2, _BASIC_ENERGY_ID)
    opp_prize = _pad([], len(op.prize), _BASIC_ENERGY_ID)
    opp_hand = _pad([], op.handCount, _BASIC_ENERGY_ID)

    opp_active = []
    if op.active and (len(op.active) == 0 or op.active[0] is None):
        opp_active = [_BASIC_POKE_ID]

    return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active


def SEARCH_ALGO(obs, base_order):
    """Beam search over ``search_begin``/``search_step``; returns a reordered
    option list (best first) or ``None`` to fall back to the heuristic order.
    """
    if not (_SEARCH_OK and USE_SEARCH):
        return None
    select = obs.select
    if select is None or select.context != SelectContext.MAIN:
        return None
    if obs.search_begin_input is None:
        return None

    t0 = time.time()
    try:
        det = _determinize(obs, my_deck)
        root = search_begin(obs, *det)
    except Exception:
        return None

    try:
        beam = []
        for first in base_order[:SEARCH_MAX_CANDIDATES]:
            try:
                s = search_step(root.searchId, [first])
            except Exception:
                continue
            beam.append((evaluate_state(s.observation), s.searchId, first, s.observation))

        depth = 0
        while depth < BEAM_DEPTH:
            if time.time() - t0 > SEARCH_TIME_BUDGET:
                break
            next_beam = []
            expanded_any = False
            for val, sid, first_action, cur_obs in beam:
                cur = cur_obs.current
                if (cur.result is not None and cur.result != -1) or cur.yourIndex != obs.current.yourIndex:
                    next_beam.append((val, sid, first_action, cur_obs))
                    continue
                if cur_obs.select.context != SelectContext.MAIN:
                    sub = AdvancedPolicy(cur_obs).choose()
                    sel = sub[: max(1, cur_obs.select.minCount)]
                    try:
                        s = search_step(sid, sel)
                    except Exception:
                        next_beam.append((val, sid, first_action, cur_obs))
                        continue
                    next_beam.append((evaluate_state(s.observation), s.searchId, first_action, s.observation))
                    expanded_any = True
                else:
                    opts = AdvancedPolicy(cur_obs).choose()[:BEAM_WIDTH]
                    for opt in opts:
                        try:
                            s = search_step(sid, [opt])
                        except Exception:
                            continue
                        next_beam.append((evaluate_state(s.observation), s.searchId, first_action, s.observation))
                        expanded_any = True
            if not expanded_any:
                break
            beam = sorted(next_beam, key=lambda x: x[0], reverse=True)[:BEAM_WIDTH]
            depth += 1

        if not beam:
            return None
        best = beam[0][2]
        return [best] + [i for i in base_order if i != best]
    except Exception:
        return None
    finally:
        try:
            search_end()
        except Exception:
            pass


def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        return my_deck if obs_dict.get("select") is None else [0]
    if obs.select is None:
        return my_deck

    global pre_turn, ability_used, plan
    if obs.current is not None and pre_turn != obs.current.turn:
        pre_turn = obs.current.turn
        ability_used = False
        plan = AttackPlan()

    try:
        base_order = AdvancedPolicy(obs).choose()
        ordered = SEARCH_ALGO(obs, base_order)
        if ordered is None:
            ordered = base_order
        n = len(obs.select.option)
        ordered = [i for i in ordered if 0 <= i < n]
        if not ordered:
            return list(range(min(max(1, obs.select.minCount), n)))
        k = max(min(obs.select.maxCount, n), min(max(1, obs.select.minCount), n))
        return ordered[:k]
    except Exception:
        n = len(obs.select.option)
        return list(range(min(max(1, obs.select.minCount), n)))


class MultiPlyAgent:
    """run_eval Agent adapter around the module-level ``agent`` fn.

    Returns the injected deck on the DECK_REQUEST frame (ignoring deck.csv) and
    delegates every other frame to ``agent``; keeps per-turn state in module
    globals, reset per game.
    """

    def __init__(self, deck, engine=None):
        global my_deck
        self._deck = list(deck)
        my_deck = list(deck)

    def reset(self, seed=0):
        global pre_turn, ability_used, plan, my_deck
        pre_turn = -1
        ability_used = False
        plan = AttackPlan()
        my_deck = list(self._deck)
        random.seed(seed)

    def __call__(self, obs_dict):
        if obs_dict.get("select") is None:
            self.reset(0)
            return list(self._deck)
        return agent(obs_dict)
