"""Marnie's Grimmsnarl ex + Munkidori (Dark) bespoke rule-based pilot.

This is the highest-share meta archetype (Marnie's Grimmsnarl ex Dark). A deck is
worthless without a pilot tuned to its mechanics -- generic ``greedy_plus`` scores
~0.08 with this deck because it never assembles the Stage-2 line nor uses the
combo. This module is a self-contained option-scoring agent (same shape as
``archaludon_pilot`` / ``alakazam_pilot``): it imports the Linux-only ``cg`` engine
at load time, so it is NOT imported by ``src.agents.__init__`` -- it is wired into
``scripts/run_eval.py`` lazily (name ``grimmsnarl``) and excluded from ruff/ty.

Deck concept:
  Marnie's Impidimp (646, Basic) -> Marnie's Morgrem (647) -> Marnie's Grimmsnarl
  ex (648, Stage 2, 320 HP). Rare Candy (1079) skips Morgrem.

  * Punk Up (Grimmsnarl ex ability): on evolving from hand, search the deck for up
    to 5 Basic {D} Energy and attach them to your Marnie's Pokemon -- the energy
    engine. Manifested by the engine as ACTIVATE -> ATTACH_TO (pick energies) ->
    ATTACH_FROM (pick recipient per energy).
  * Shadow Bullet ({D}{D} = 180): main attack; also snipes ~30 to 1 opponent
    Benched Pokemon (DAMAGE context after the attack).
  * Munkidori (112, Basic): Adrena-Brain -- once per turn, if it has {D} Energy
    attached, move up to 3 damage counters from 1 of YOUR Pokemon to 1 of your
    OPPONENT's Pokemon (heals us + snipes/finishes). Manifested as ABILITY ->
    REMOVE_DAMAGE_COUNTER (our mon) -> DAMAGE_COUNTER (opp mon).
  * Froslass (104, from Snorunt 860): Freezing Shroud -- at each Checkup put 1
    damage counter on every Pokemon that HAS an Ability (both players) except
    Froslass. Double-edged (chips our own Munkidori/Grimmsnarl), so evolved only
    when the opponent runs ability Pokemon.
  * Spikemuth Gym (1259, stadium): once per turn each player may search their deck
    for a Marnie's Pokemon -- the consistency engine (ABILITY on the stadium).

Consistency: Buddy-Buddy Poffin (1086, bench 2 basics <=70 HP: Impidimp/Snorunt),
Poke Pad (1152, search a non-rule-box Pokemon), Team Rocket's Petrel (1219, search
a Trainer), Dawn (1231, search basic+stage1+stage2), Lillie's Determination (1227,
draw 6/8), Night Stretcher (1097, recover), Unfair Stamp (1080, comeback draw),
Boss's Orders (1182, gust), Handheld Fan (1161, tool). 10x Basic {D} Energy (7).

Score system (mirrors the sister pilots):
  Setup/play/evolve/ability/attach: 1000..30000 (high = do first).
  Attack: base damage (always last -- attacking ends the turn).
  Negative = skip if we are already at ``minCount``.
"""

import os
import random
import sys

try:
    ROOT = __file__
except NameError:
    ROOT = None
CG_PATH = "/kaggle_simulations/agent"
for p in ([os.path.dirname(os.path.abspath(ROOT))] if ROOT else []) + [CG_PATH]:
    if p and p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)

from cg.api import (
    AreaType,
    OptionType,
    SelectContext,
    all_card_data,
    to_observation_class,
)

try:
    from cg.api import all_attack
    ALL_ATTACKS = {a.attackId: a for a in all_attack()}
except Exception:
    ALL_ATTACKS = {}

CARD_DB = {c.cardId: c for c in all_card_data()}

# ── Card IDs ──
IMPIDIMP = 646
MORGREM = 647
GRIMM = 648            # Marnie's Grimmsnarl ex (Stage 2)
MUNKIDORI = 112
SNORUNT = 860
FROSLASS = 104
DARK = 7               # Basic {D} Energy

SPIKEMUTH = 1259       # stadium
BOSS = 1182            # supporter (gust)
LILLIE = 1227          # supporter (draw 6/8)
PETREL = 1219          # supporter (search a Trainer)
DAWN = 1231            # supporter (search basic+stage1+stage2)
POKEPAD = 1152         # item (search a non-rule-box Pokemon)
POFFIN = 1086          # item (bench up to 2 basics <=70 HP)
NIGHT_STRETCHER = 1097  # item (recover Pokemon / basic energy)
UNFAIR_STAMP = 1080    # item (comeback draw 5, usable only after a KO)
HANDHELD_FAN = 1161    # tool
RARE_CANDY = 1079      # item (skip Stage 1)

# Attack IDs
SHADOW_BULLET = 937    # Grimmsnarl ex: {D}{D} 180 + bench snipe
MORGREM_PUNCH = 936    # Morgrem: {D}{D} 60
IMPIDIMP_PUNCH = 935   # Impidimp: {D} 10
MUNK_MINDBEND = 141    # Munkidori: 60
FROSLASS_SMASH = 131   # Froslass: 60
SNORUNT_CHILLY = 1239  # Snorunt: 10

LINE = {IMPIDIMP, MORGREM, GRIMM}
MARNIES = {IMPIDIMP, MORGREM, GRIMM}
BASIC_POKEMON = {IMPIDIMP, MUNKIDORI, SNORUNT}
SUPPORTERS = {BOSS, LILLIE, PETREL, DAWN}

BENCH_SNIPE_DMG = 30  # Shadow Bullet's benched-Pokemon rider


# ── Deck loader (fail-safe: absent when used as a run_eval opponent) ──
def read_deck_csv():
    fp = "deck.csv"
    if not os.path.exists(fp):
        fp = "/kaggle_simulations/agent/deck.csv"
    out = []
    try:
        with open(fp) as f:
            for line in f.read().strip().split("\n"):
                if line.strip():
                    out.append(int(line))
    except (FileNotFoundError, ValueError, IndexError, OSError):
        out = []
    return out


# ── Observation helpers ──
def my_state(obs):
    return obs.current.players[obs.current.yourIndex]


def opp_state(obs):
    return obs.current.players[1 - obs.current.yourIndex]


def active_of(ps):
    return ps.active[0] if ps.active else None


def bench_of(ps):
    return [p for p in (ps.bench or []) if p]


def pokes_of(ps):
    return [p for p in ((ps.active or []) + (ps.bench or [])) if p]


def hand_ids(obs):
    hand = my_state(obs).hand
    return [c.id for c in hand if c] if hand else []


def discard_ids(obs):
    return [c.id for c in (my_state(obs).discard or []) if c]


def energy_count(pokemon):
    if pokemon is None:
        return 0
    if getattr(pokemon, "energyCards", None) is not None:
        return len(pokemon.energyCards)
    return len(getattr(pokemon, "energies", []) or [])


def dark_count(pokemon):
    if pokemon is None:
        return 0
    ecs = getattr(pokemon, "energyCards", None)
    if ecs is None:
        return energy_count(pokemon)  # assume all dark in this mono-{D} deck
    return sum(1 for e in ecs if getattr(e, "id", None) == DARK)


def damage_on(pokemon):
    if pokemon is None:
        return 0
    return max(0, getattr(pokemon, "maxHp", pokemon.hp) - pokemon.hp)


def has_tool(pokemon):
    return bool(getattr(pokemon, "tools", []) or [])


def retreat_cost(pokemon):
    data = CARD_DB.get(pokemon.id) if pokemon else None
    return int(getattr(data, "retreatCost", 0) or 0) if data else 0


def attacker_rank(pokemon):
    """How much we want this Pokemon in the Active Spot as our attacker.

    Grimmsnarl ex with 2 {D} (Shadow Bullet ready) is the top attacker; a weak
    body like Snorunt is near-worthless up front. Used to decide retreats/promotes.
    """
    if pokemon is None:
        return -1
    e = energy_count(pokemon)
    cid = pokemon.id
    if cid == GRIMM:
        return 100 + (30 if e >= 2 else e * 5)
    if cid == MORGREM:
        return 55 + (15 if e >= 2 else 0)
    if cid == MUNKIDORI:
        return 45 + (15 if e >= 1 else 0)
    if cid == FROSLASS:
        return 40 + (10 if e >= 2 else 0)
    if cid == IMPIDIMP:
        return 20
    if cid == SNORUNT:
        return 10
    return 15


def card_of(obs, area, index, player_index):
    if area is None or index is None:
        return None
    ps = obs.current.players[player_index]
    try:
        if area == AreaType.DECK and obs.select and obs.select.deck is not None:
            return obs.select.deck[index]
        if area == AreaType.HAND and ps.hand is not None:
            return ps.hand[index]
        if area == AreaType.DISCARD:
            return ps.discard[index]
        if area == AreaType.ACTIVE:
            return ps.active[index]
        if area == AreaType.BENCH:
            return ps.bench[index]
        if area == AreaType.PRIZE:
            return ps.prize[index]
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
        if area == AreaType.LOOKING and obs.current.looking is not None:
            return obs.current.looking[index]
    except (IndexError, TypeError):
        return None
    return None


def option_card(obs, opt):
    yi = obs.current.yourIndex
    pi = opt.playerIndex if opt.playerIndex is not None else yi
    if opt.type == OptionType.PLAY:
        return card_of(obs, AreaType.HAND, opt.index, pi)
    return card_of(obs, opt.area, opt.index, pi)


def option_target(obs, opt):
    if opt.inPlayArea is None or opt.inPlayIndex is None:
        return None
    return card_of(obs, opt.inPlayArea, opt.inPlayIndex, obs.current.yourIndex)


def count_in_play(obs, cid):
    return sum(1 for p in pokes_of(my_state(obs)) if p.id == cid)


def has_in_play(obs, cid):
    return any(p.id == cid for p in pokes_of(my_state(obs)))


def prize_value(pokemon):
    data = CARD_DB.get(pokemon.id) if pokemon else None
    if data and getattr(data, "megaEx", False):
        return 3
    if data and getattr(data, "ex", False):
        return 2
    return 1


def weakness_of(cid):
    data = CARD_DB.get(cid)
    w = getattr(data, "weakness", None) if data else None
    if w is None:
        return None
    return getattr(w, "value", w)


def opp_ability_pokemon(obs):
    """Opponent Pokemon that have an Ability (relevant to Froslass tech)."""
    out = 0
    for p in pokes_of(opp_state(obs)):
        data = CARD_DB.get(p.id)
        if data and getattr(data, "skills", None):
            out += 1
    return out


# ── Attack helpers ──
def attack_base_dmg(attack_id):
    a = ALL_ATTACKS.get(attack_id)
    return int(a.damage) if a else 0


def eff_dmg(base, target):
    """Effective damage vs target, doubling on Dark weakness (our attacker is {D})."""
    if base <= 0 or target is None:
        return base
    return base * 2 if weakness_of(target.id) == DARK else base


def best_attack_of(pokemon):
    """(attack_id, base_dmg) of the highest-damage attack the active can use, or None."""
    if pokemon is None:
        return None
    data = CARD_DB.get(pokemon.id)
    if not data or not getattr(data, "attacks", None):
        return None
    e = energy_count(pokemon)
    best = None
    for aid in data.attacks:
        a = ALL_ATTACKS.get(aid)
        if a is None:
            continue
        if len(a.energies) > e:  # not enough energy attached
            continue
        if best is None or a.damage > best[1]:
            best = (aid, int(a.damage))
    return best


def our_attack_damage(obs):
    """Best base damage our current Active can throw right now (0 if none ready)."""
    active = active_of(my_state(obs))
    ba = best_attack_of(active)
    if ba is None:
        return 0
    dmg = ba[1]
    # Shadow Bullet snipe already covered separately; Munkidori mind-bend etc.
    return dmg


# ── Setup scoring ──
_SETUP_ACTIVE = {IMPIDIMP: 3000, MUNKIDORI: 2000, SNORUNT: 800}


def score_setup(obs, opt):
    ctx = obs.select.context
    card = option_card(obs, opt)
    cid = card.id if card else None

    if ctx == SelectContext.MULLIGAN:
        return (10000, "no mulligan") if opt.type == OptionType.NO else (0, "mulligan")
    if ctx == SelectContext.IS_FIRST:
        # Tank/control deck: go FIRST to land Spikemuth + start the evolution line
        # undisrupted (measured markedly better than going second). OPT_YES = first.
        return (10000, "go first") if opt.type == OptionType.YES else (0, "go second")
    if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
        return _SETUP_ACTIVE.get(cid, (500, "active")), "setup active"
    if ctx == SelectContext.SETUP_BENCH_POKEMON:
        # Bench everything useful during set-up (build a wide board of basics).
        if cid == IMPIDIMP:
            return 3000, "bench Impidimp"
        if cid == MUNKIDORI:
            return 2800, "bench Munkidori"
        if cid == SNORUNT:
            return 1500, "bench Snorunt"
        return 500, "bench basic"
    return 0, "non-setup"


# ── PLAY scoring ──
def score_play(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else None
    ids = hand_ids(obs)
    cur = obs.current
    ps = my_state(obs)
    bench_n = len(bench_of(ps))
    bench_free = (getattr(ps, "benchMax", 5) or 5) - bench_n
    grimm_in_play = has_in_play(obs, GRIMM)
    impid_in_play = count_in_play(obs, IMPIDIMP)
    morg_in_play = count_in_play(obs, MORGREM)

    # ── Basic Pokemon: develop the board ──
    if cid == IMPIDIMP:
        if bench_free <= 0:
            return -200, "bench full"
        line_bodies = impid_in_play + morg_in_play + count_in_play(obs, GRIMM)
        if line_bodies >= 3:
            return 2000, "extra Impidimp"
        return (17000 if line_bodies < 2 else 8000), "play Impidimp"
    if cid == MUNKIDORI:
        if bench_free <= 0:
            return -200, "bench full"
        # Two Munkidori is enough for the Adrena-Brain combo; more are just
        # 1-prize liabilities that clog the bench and feed the opponent.
        n_munk = count_in_play(obs, MUNKIDORI)
        if n_munk >= 2:
            return -300, "enough Munkidori"
        return (14000 if n_munk == 0 else 9000), "play Munkidori"
    if cid == SNORUNT:
        if bench_free <= 0:
            return -200, "bench full"
        return (7000 if opp_ability_pokemon(obs) >= 1 and not has_in_play(obs, SNORUNT)
                else 1500), "play Snorunt"

    # ── Stadium ──
    if cid == SPIKEMUTH:
        stad = cur.stadium
        # Replace only if no stadium, or a foreign stadium is out.
        if stad and any(c and c.id == SPIKEMUTH for c in stad):
            return -300, "Spikemuth already out"
        return 15000, "play Spikemuth Gym"

    # ── Rare Candy: skip to Grimmsnarl ex (Punk Up) ──
    if cid == RARE_CANDY:
        if GRIMM in ids and impid_in_play >= 1:
            return 26000, "Rare Candy -> Grimmsnarl ex"
        return -500, "save Rare Candy"

    # ── Item search/draw ──
    if cid == POFFIN:
        # Fetches Impidimp/Snorunt (<=70 HP basics) to the bench.
        if bench_free <= 0:
            return -200, "bench full"
        line_bodies = impid_in_play + morg_in_play + count_in_play(obs, GRIMM)
        return (24000 if line_bodies < 2 else 12000), "Buddy-Buddy Poffin"
    if cid == POKEPAD:
        # Search a non-rule-box Pokemon (Impidimp/Morgrem/Munkidori/Snorunt/Froslass).
        return 18000, "Poke Pad"
    if cid == NIGHT_STRETCHER:
        disc = discard_ids(obs)
        urgent = (
            (GRIMM not in ids and GRIMM in disc and grimm_in_play is False)
            or (IMPIDIMP in disc and impid_in_play == 0 and IMPIDIMP not in ids)
            or (MORGREM in disc and MORGREM not in ids)
            or (DARK in disc and DARK not in ids and not any(
                dark_count(p) for p in pokes_of(ps)))
        )
        return (15000 if urgent else -400), "Night Stretcher"
    if cid == UNFAIR_STAMP:
        # Only offered when legal (a KO happened last turn); a free refuel to 5.
        return 17000, "Unfair Stamp"
    if cid == HANDHELD_FAN:
        return -400, "tool: handled via ATTACH"

    # ── Supporters (one per turn) ──
    if cid in SUPPORTERS:
        if cur.supporterPlayed:
            return -1000, "supporter already used"
        if cid == BOSS:
            return _score_boss(obs)
        if cid == PETREL:
            return 8000, "Petrel: search Trainer"
        if cid == DAWN:
            # Search the whole line at once; great when we're missing pieces.
            need = (impid_in_play + morg_in_play + count_in_play(obs, GRIMM)) < 2 \
                or GRIMM not in ids
            return (9000 if need else 3500), "Dawn: search line"
        if cid == LILLIE:
            # Draw supporter; save it if a Boss line is set up and we can act.
            if BOSS in ids and our_attack_damage(obs) > 0:
                return 2500, "Lillie (Boss primed)"
            hand_n = ps.handCount if ps.handCount is not None else len(ids)
            return (7000 if hand_n <= 4 else 4000), "Lillie: draw"

    return 1000, "generic play"


def _score_boss(obs):
    """Boss's Orders: gust a benched opponent Pokemon to the Active Spot."""
    opp = opp_state(obs)
    opp_act = active_of(opp)
    bench = bench_of(opp)
    if not bench:
        return -500, "Boss: no bench target"
    dmg = our_attack_damage(obs)
    if dmg <= 0:
        return -500, "Boss: no attacker ready"
    remaining = len(my_state(obs).prize)

    # Lethal: KO a benched target that wins the game outright.
    for t in bench:
        if eff_dmg(dmg, t) >= t.hp and prize_value(t) >= remaining:
            return 20000, "LETHAL Boss"

    active_koable = opp_act and eff_dmg(dmg, opp_act) >= opp_act.hp
    best = -500
    for t in bench:
        if eff_dmg(dmg, t) >= t.hp:
            # Prefer gusting a KO-able bench target when the active is a fat tank
            # we can't KO, or when the bench target is worth more prizes.
            val = 6000 + prize_value(t) * 800
            if not active_koable:
                val += 4000
            elif prize_value(t) > (prize_value(opp_act) if opp_act else 1):
                val += 2000
            else:
                val -= 3000  # active KO is simpler; don't waste Boss
            best = max(best, val)
    if best > 0:
        return best, "Boss: gust KO target"
    # Drag a fragile/undeveloped benched Pokemon to stall/deny setup.
    if not active_koable:
        weakest = min(bench, key=lambda p: (p.hp, energy_count(p)))
        return 3000, "Boss: drag weak bench" if weakest else (-500, "save Boss")
    return -500, "save Boss"


# ── EVOLVE scoring ──
def score_evolve(obs, opt):
    card = option_card(obs, opt)
    target = option_target(obs, opt)
    cid = card.id if card else None
    tid = target.id if target else None
    active_is = opt.inPlayArea == AreaType.ACTIVE

    if cid == GRIMM and tid in (MORGREM, IMPIDIMP):
        # Grimmsnarl ex (also the Rare Candy-context evolve): triggers Punk Up
        # (deck-wide {D} acceleration). But each fielded ex is a 2-prize liability,
        # so field ONE attacker + at most one backup; hold further copies in hand.
        n_grimm = count_in_play(obs, GRIMM)
        if n_grimm == 0:
            base = 30000 + energy_count(target) * 500 + (1500 if active_is else 0)
            return base, "evolve -> Grimmsnarl ex (Punk Up)"
        if n_grimm == 1:
            active = active_of(my_state(obs))
            need_backup = active is None or active.id != GRIMM \
                or damage_on(active) >= 160
            return (16000 if need_backup else 5000), "evolve -> backup Grimmsnarl ex"
        return -500, "hold 3rd Grimmsnarl ex"
    if cid == MORGREM and tid == IMPIDIMP:
        # Only if we can't Rare Candy straight to Grimmsnarl on this body.
        base = 20000 + (1000 if active_is else 0)
        return base, "evolve -> Morgrem"
    if cid == FROSLASS and tid == SNORUNT:
        # Double-edged: Freezing Shroud chips our own ability Pokemon too.
        if opp_ability_pokemon(obs) >= 1:
            return 6000, "evolve -> Froslass (opp abilities)"
        return -500, "skip Froslass (self-chip)"
    return 9000, "generic evolve"


# ── ATTACH scoring ──
def _attach_target_score(obs, target, area):
    if target is None:
        return -1000
    cid = target.id
    e = energy_count(target)
    is_active = area == AreaType.ACTIVE

    if cid == GRIMM:
        if e >= 2:
            return -800 + 200  # already attack-ready; low value
        score = 9000 + (2 - e) * 1500
        return score + (1500 if is_active else 0)
    if cid == MORGREM:
        if e >= 2:
            return -600
        return 6000 + (2 - e) * 1000 + (1000 if is_active else 0)
    if cid == MUNKIDORI:
        # 1 {D} unlocks Adrena-Brain; more than 1 is low value.
        if e >= 1:
            return -700
        return 7000
    if cid == IMPIDIMP:
        return 2500 - e * 800  # pre-load a future attacker a little
    if cid == FROSLASS:
        return 1500 - e * 500
    return 500 - e * 500


def score_attach(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else None
    target = option_target(obs, opt)

    if cid == HANDHELD_FAN:
        # Defensive tool: on the Active tank it strips an Energy off the attacker
        # each time it's hit (disrupts the opponent's attack engine, e.g. metal).
        if target and not has_tool(target) and target.id in (GRIMM, MORGREM, MUNKIDORI):
            active = active_of(my_state(obs))
            is_active = active is not None and target.id == active.id \
                and opt.inPlayArea == AreaType.ACTIVE
            return (5000 if is_active else 2000), "Handheld Fan"
        return -1000, "save Handheld Fan"
    if cid != DARK:
        return -500, "skip non-Dark attach"
    if obs.current.energyAttached:
        return -1000, "already attached this turn"

    # Enable a retreat: if a fuelled attacker is benched but the weak active can't
    # pay its retreat cost, put the energy on the active to unlock promotion.
    ps = my_state(obs)
    active = active_of(ps)
    if (target is not None and opt.inPlayArea == AreaType.ACTIVE
            and active is not None and active.id not in (GRIMM, MORGREM)
            and max((attacker_rank(p) for p in bench_of(ps)), default=-1) >= 100
            and energy_count(active) < retreat_cost(active)):
        return 10000, "attach to enable retreat"

    return _attach_target_score(obs, target, opt.inPlayArea), "attach Dark"


# ── RETREAT scoring ──
def score_retreat(obs):
    ps = my_state(obs)
    active = active_of(ps)
    if active is None:
        return -100, "no active"
    # Never retreat our tank away from the front (unless it literally can't act,
    # handled implicitly -- Grimmsnarl can always attack once fuelled).
    if active.id == GRIMM:
        return -5000, "keep Grimmsnarl active"

    cur_rank = attacker_rank(active)
    best_bench = max((attacker_rank(p) for p in bench_of(ps)), default=-1)
    if best_bench <= cur_rank:
        return -300, "active is our best attacker; hold"

    # A clearly better attacker (esp. a fuelled Grimmsnarl ex) is waiting on the
    # bench and the active is a weak body -- retreat to promote it. The engine only
    # offers RETREAT when the retreat cost is payable, so we can score it high.
    gap = best_bench - cur_rank
    if best_bench >= 100:  # a Grimmsnarl ex is on the bench
        return 22000, "retreat -> promote Grimmsnarl ex"
    if gap >= 20:
        return 9000, "retreat -> promote better attacker"
    return -300, "hold"


# ── ABILITY scoring (Munkidori Adrena-Brain, Spikemuth Gym) ──
def score_ability(obs, opt):
    src = card_of(obs, opt.area, opt.index, obs.current.yourIndex)
    if opt.area == AreaType.STADIUM:
        # Spikemuth Gym: search a Marnie's Pokemon (free consistency).
        return 26000, "Spikemuth search"
    sid = src.id if src else None
    if sid == MUNKIDORI:
        # Adrena-Brain: move damage off us onto the opponent -- pure value.
        return 24000, "Munkidori Adrena-Brain"
    return 1, "generic ability"


# ── Card-selection contexts ──
def _search_priority(obs, cid, *, in_hand_ok=True):
    """Generic 'how much do we want this card in hand' for search/draw effects."""
    ids = hand_ids(obs)
    impid = count_in_play(obs, IMPIDIMP)
    morg = count_in_play(obs, MORGREM)
    grimm = count_in_play(obs, GRIMM)
    line_bodies = impid + morg + grimm

    if cid == GRIMM:
        if GRIMM in ids:
            return 40  # already have one
        if morg >= 1 or (impid >= 1 and RARE_CANDY in ids):
            return 320
        return 180
    if cid == MORGREM:
        if impid >= 1 and MORGREM not in ids and (RARE_CANDY not in ids or GRIMM not in ids):
            return 260
        return 90
    if cid == IMPIDIMP:
        if line_bodies < 2:
            return 240
        return 70
    if cid == MUNKIDORI:
        return 210 if not has_in_play(obs, MUNKIDORI) else 60
    if cid == RARE_CANDY:
        return 250 if (GRIMM in ids and impid >= 1) else 110
    if cid == POFFIN:
        return 150
    if cid == POKEPAD:
        return 130
    if cid == DARK:
        attached = any(dark_count(p) for p in pokes_of(my_state(obs)))
        return 120 if not attached else 40
    if cid == BOSS:
        return 100
    if cid == SPIKEMUTH:
        stad = obs.current.stadium
        return 140 if not (stad and any(c and c.id == SPIKEMUTH for c in stad)) else 30
    if cid == LILLIE:
        return 90
    if cid in (PETREL, DAWN):
        return 85
    if cid == SNORUNT:
        return 50 if opp_ability_pokemon(obs) >= 1 else 20
    if cid == FROSLASS:
        return 40 if opp_ability_pokemon(obs) >= 1 else 10
    if cid == NIGHT_STRETCHER:
        return 60
    return 50


def score_to_hand(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    if cid is None:
        return 10, "unknown take"
    return _search_priority(obs, cid), "take"


def score_to_bench(obs, opt):
    # Buddy-Buddy Poffin puts basics onto the bench.
    card = option_card(obs, opt)
    cid = card.id if card else None
    if cid == IMPIDIMP:
        return 240, "bench Impidimp"
    if cid == SNORUNT:
        return (120 if opp_ability_pokemon(obs) >= 1 else 40), "bench Snorunt"
    return 60, "bench basic"


def score_discard(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    ids = hand_ids(obs)
    # Keep the win condition; pitch surplus.
    if cid == GRIMM:
        return (-4000 if ids.count(GRIMM) <= 1 else 500), "keep Grimmsnarl"
    if cid == MORGREM:
        return (-2000 if ids.count(MORGREM) <= 1 else 800), "keep Morgrem"
    if cid == IMPIDIMP:
        return (-1500 if count_in_play(obs, IMPIDIMP) == 0 and ids.count(IMPIDIMP) <= 1
                else 1000), "Impidimp"
    if cid == MUNKIDORI:
        return (-1200 if not has_in_play(obs, MUNKIDORI) else 900), "Munkidori"
    if cid == RARE_CANDY:
        return -800, "keep Rare Candy"
    if cid == DARK:
        have = sum(dark_count(p) for p in pokes_of(my_state(obs)))
        return (900 if (have >= 2 or ids.count(DARK) > 1) else -600), "Dark energy"
    return 700, "generic discard"


def score_damage_snipe(obs, opt):
    """Shadow Bullet: put ~30 on 1 opponent Benched Pokemon."""
    card = option_card(obs, opt)
    if card is None:
        return 10, "snipe"
    # KO now if possible (highest prize value); else chip the lowest-HP body
    # (sets up a Munkidori / next-turn finish), tie-break high prize value.
    if card.hp <= BENCH_SNIPE_DMG:
        return 10000 + prize_value(card) * 500 - card.hp, "snipe KO"
    return 3000 - card.hp + prize_value(card) * 200, "snipe chip"


def score_remove_counter(obs, opt):
    """Munkidori source: heal one of OUR Pokemon (prefer the valuable, damaged one)."""
    card = option_card(obs, opt)
    if card is None:
        return 10, "heal src"
    val = damage_on(card)
    if card.id == GRIMM:
        val += 2000
    elif card.id in (MORGREM, MUNKIDORI):
        val += 500
    return val, "Munkidori heal source"


def score_damage_counter(obs, opt):
    """Munkidori destination: put counters on an OPPONENT Pokemon."""
    card = option_card(obs, opt)
    if card is None:
        return 10, "put counter"
    # Prefer finishing a low-HP body; weight active + prize value.
    is_active = opt.area == AreaType.ACTIVE
    score = 5000 - card.hp + prize_value(card) * 400
    if is_active:
        score += 800
    return score, "Munkidori put counter"


def score_target(obs, opt):
    """SWITCH / TO_ACTIVE (Boss gust target, or our own promotion after a KO)."""
    ctx = obs.select.context
    yi = obs.current.yourIndex
    pi = opt.playerIndex if opt.playerIndex is not None else yi
    card = option_card(obs, opt)
    if card is None:
        return 10, "target"

    if pi != yi:
        # Choosing which opponent Pokemon to drag up (Boss's Orders).
        dmg = our_attack_damage(obs)
        koable = eff_dmg(dmg, card) >= card.hp if dmg else False
        if koable:
            return 20000 + prize_value(card) * 2000 - card.hp, "Boss: KO target"
        return 5000 + prize_value(card) * 500 + energy_count(card) * 100, "Boss: drag"

    # Promote one of OUR Pokemon to Active (after a KO / retreat).
    ba = best_attack_of(card)
    score = 1000
    if card.id == GRIMM:
        score = 15000
    elif card.id == MORGREM:
        score = 9000
    elif card.id == MUNKIDORI:
        score = 6000
    elif card.id == IMPIDIMP:
        score = 3000
    if ba is not None:
        score += 2000  # can attack immediately
    score += energy_count(card) * 100
    return score, "promote"


def score_attach_punkup(obs, opt):
    """Punk Up recipient (ATTACH_FROM) / energy pick (ATTACH_TO)."""
    ctx = obs.select.context
    if ctx == SelectContext.ATTACH_TO:
        # These are the energies to pull from the deck -- take as many as offered.
        return 5000, "Punk Up: take energy"
    # ATTACH_FROM: choose the recipient Marnie's Pokemon.
    card = option_card(obs, opt)
    if card is None:
        return 10, "recipient"
    e = energy_count(card)
    if card.id == GRIMM:
        # Fill the attacker to 2 first, then keep loading it (tank fuel).
        return (9000 if e < 2 else 4000), "Punk Up -> Grimmsnarl"
    if card.id == MUNKIDORI:
        return (7000 if e < 1 else 2000), "Punk Up -> Munkidori"
    if card.id == MORGREM:
        return 5000 - e * 500, "Punk Up -> Morgrem"
    if card.id == IMPIDIMP:
        return 3000 - e * 500, "Punk Up -> Impidimp"
    return 1000, "Punk Up recipient"


# ── Dispatch ──
_MAIN = {
    OptionType.PLAY: score_play,
    OptionType.EVOLVE: score_evolve,
    OptionType.ATTACH: score_attach,
    OptionType.ABILITY: score_ability,
}


def score_option(obs, opt):
    ctx = obs.select.context

    if ctx in (SelectContext.IS_FIRST, SelectContext.MULLIGAN,
               SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON):
        return score_setup(obs, opt)

    if opt.type in (OptionType.YES, OptionType.NO):
        if ctx == SelectContext.ACTIVATE:
            # Punk Up / other beneficial "may" prompts: always accept.
            return (30000, "activate yes") if opt.type == OptionType.YES \
                else (-30000, "activate no")
        return (1, "yes") if opt.type == OptionType.YES else (0, "no")

    if opt.type == OptionType.NUMBER:
        return (opt.number or 0), "number (max)"

    if ctx == SelectContext.MAIN:
        fn = _MAIN.get(opt.type)
        if fn:
            return fn(obs, opt)
        if opt.type == OptionType.RETREAT:
            return score_retreat(obs)
        if opt.type == OptionType.ATTACK:
            return attack_base_dmg(opt.attackId), "attack"
        if opt.type == OptionType.END:
            return 0, "end turn"
        return 500, "generic MAIN"

    if ctx == SelectContext.ATTACK:
        return attack_base_dmg(opt.attackId), "attack"
    if ctx == SelectContext.TO_HAND:
        return score_to_hand(obs, opt)
    if ctx == SelectContext.TO_BENCH:
        return score_to_bench(obs, opt)
    if ctx == SelectContext.TO_FIELD:
        return score_to_bench(obs, opt)
    if ctx in (SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
               SelectContext.DISCARD_ENERGY, SelectContext.TO_DECK,
               SelectContext.TO_DECK_BOTTOM):
        return score_discard(obs, opt)
    if ctx == SelectContext.DAMAGE:
        return score_damage_snipe(obs, opt)
    if ctx in (SelectContext.REMOVE_DAMAGE_COUNTER,):
        return score_remove_counter(obs, opt)
    if ctx in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
        return score_damage_counter(obs, opt)
    if ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        return score_target(obs, opt)
    if ctx in (SelectContext.ATTACH_TO, SelectContext.ATTACH_FROM):
        return score_attach_punkup(obs, opt)
    if ctx == SelectContext.HEAL:
        card = option_card(obs, opt)
        return (damage_on(card), "heal") if card else (10, "heal")
    if opt.type == OptionType.CARD:
        return score_to_hand(obs, opt)
    if opt.type == OptionType.ENERGY:
        return 1000, "energy"
    if opt.type == OptionType.END:
        return 0, "end"
    return 100, "fallback"


# ── Choose & Agent ──
def choose_options(obs):
    scored = []
    for i, opt in enumerate(obs.select.option):
        try:
            res = score_option(obs, opt)
            score = res[0] if isinstance(res, tuple) else res
        except Exception as e:  # noqa: BLE001 - never crash on a scoring bug
            score = -999999
        scored.append((score, i))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    selected = []
    for score, i in scored:
        if len(selected) >= obs.select.maxCount:
            break
        if score < 0 and len(selected) >= obs.select.minCount:
            continue
        selected.append(i)

    if len(selected) < obs.select.minCount:
        selected = [i for _, i in scored[:obs.select.minCount]]
    return selected


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    if not obs.select.option:
        return []
    try:
        return choose_options(obs)
    except Exception:  # noqa: BLE001 - guaranteed-legal fallback
        n = len(obs.select.option)
        mc = obs.select.maxCount
        return list(range(min(mc, n)))


# ── Harness adapter (run_eval Agent protocol) ──
class GrimmsnarlAgent:
    """Wrap the module-level ``agent`` fn behind the run_eval Agent protocol."""

    def __init__(self, deck, engine=None):
        self._deck = list(deck)

    def reset(self, seed=0):
        random.seed(seed)

    def __call__(self, obs_dict):
        if obs_dict.get("select") is None:
            self.reset(0)
            return list(self._deck)
        return agent(obs_dict)
