"""Mega Starmie ex + Dusknoir (Water) bespoke rule-based pilot.

Deck concept (``decklists/starmie_ladder.csv``, a real ladder list that reached
Elo 1037 with its owner's pilot):

  * Mega Starmie ex (1031) is a Stage-1 from Staryu (1030) -- a single evolution
    step to a 330-HP attacker. Jetting Blow ({W}=120) also snipes 50 onto a
    benched Pokemon; it costs only ONE Water so it is sustainable every turn.
    Nebula Beam ({C}{C}{C}=210) ignores Weakness/Resistance and any effects on
    the opponent's Active -- the big single-target nuke.
  * Ignition Energy (17) is a special energy: on an Evolution it provides
    {C}{C}{C}, so a single Ignition on Mega Starmie ex instantly powers Nebula
    Beam (it is discarded at end of turn -- a one-turn burst).
  * The Duskull(131)->Dusclops(132)->Dusknoir(133) line is a "bomb" engine: it
    has NO usable attack here (no Psychic energy in the deck) and exists purely
    for the Cursed Blast Ability -- put 13 damage counters (130) [Dusknoir] or 5
    (50) [Dusclops] on ANY opponent Pokemon, then this Pokemon is Knocked Out.
    Sacrifice it to set up / finish KOs (a 1-prize trade for a big counter dump).
  * Mega Starmie ex is worth 3 prizes when KO'd -- protect it, trade carefully.

Trainers: Buddy-Buddy Poffin (2 basics <=70HP = Staryu+Duskull), Poke Pad
(non-ex Pokemon), Ultra Ball (any Pokemon, discard 2), Pokegear 3.0 (a Supporter
off top 7), Hilda (Evolution + Energy), Lillie's Determination (draw 6/8), Wally's
Compassion (fully heal a Mega ex + return its energy), Carmine (discard hand,
draw 5), Judge (both draw 4 = disruption), Deluxe Bomb (ACE SPEC tool: 120 to the
attacker when the holder is damaged in the Active spot).

Not imported by ``src/agents/__init__`` (it imports the Linux-only ``cg`` engine
at load time); wired into ``scripts/run_eval.py`` lazily (name ``starmie``) and
excluded from ruff/ty like the other bespoke ports. Score system mirrors the
Archaludon/Alakazam pilots: setup/play/evolve/attach score in the thousands,
attacks score by damage (last, because attacking ends the turn), a lethal line
scores astronomically, and negative = skip if above ``minCount``.
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
    CardType,
    OptionType,
    SelectContext,
    all_card_data,
    to_observation_class,
)

DEBUG = bool(os.environ.get("STARMIE_DEBUG"))

# ── Card IDs ──
STARYU = 1030
MEGA_STARMIE = 1031
DUSKULL = 131
DUSCLOPS = 132
DUSKNOIR = 133

WATER_ENERGY = 3
IGNITION_ENERGY = 17

POFFIN = 1086
POKE_PAD = 1152
ULTRA_BALL = 1121
POKEGEAR = 1122
HILDA = 1225
LILLIE = 1227
WALLY = 1229
CARMINE = 1192
JUDGE = 1213
DELUXE_BOMB = 1167

# Attack IDs
WATER_GUN = 1486     # Staryu 20
JETTING_BLOW = 1487  # Mega Starmie ex 120 + 50 bench snipe, cost {W}
NEBULA_BEAM = 1488   # Mega Starmie ex 210, ignores weak/effects, cost {C}{C}{C}

STARMIE_LINE = {STARYU, MEGA_STARMIE}
BOMB_LINE = {DUSKULL, DUSCLOPS, DUSKNOIR}
SUPPORTERS = {HILDA, LILLIE, WALLY, CARMINE, JUDGE}
SEARCH_ITEMS = {POFFIN, POKE_PAD, ULTRA_BALL, POKEGEAR}

WATER_TYPE = 3
LIGHTNING_TYPE = 4

CARD_DB = {c.cardId: c for c in all_card_data()}


# ── Board helpers ──

def read_deck_csv():
    fp = "deck.csv"
    if not os.path.exists(fp):
        fp = "/kaggle_simulations/agent/deck.csv"
    try:
        with open(fp) as f:
            return [int(line) for line in f.read().strip().split("\n")][:60]
    except (FileNotFoundError, ValueError, IndexError):
        return []


def get_card(obs, area, index, player_index):
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
    if opt.type in (OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE):
        return get_card(obs, AreaType.HAND, opt.index, yi)
    return get_card(obs, opt.area, opt.index, pi)


def option_target(obs, opt):
    if opt.inPlayArea is None or opt.inPlayIndex is None:
        return None
    yi = obs.current.yourIndex
    pi = opt.playerIndex if opt.playerIndex is not None else yi
    return get_card(obs, opt.inPlayArea, opt.inPlayIndex, pi)


def my_state(obs):
    return obs.current.players[obs.current.yourIndex]


def opp_state(obs):
    return obs.current.players[1 - obs.current.yourIndex]


def active_pokemon(obs):
    ps = my_state(obs)
    return ps.active[0] if ps.active else None


def opp_active(obs):
    ps = opp_state(obs)
    return ps.active[0] if ps.active else None


def opp_bench(obs):
    return [p for p in opp_state(obs).bench if p]


def all_my_pokemon(obs):
    ps = my_state(obs)
    return [p for p in (ps.active + ps.bench) if p]


def hand_ids(obs):
    hand = my_state(obs).hand
    return [c.id for c in hand if c] if hand else []


def discard_ids(obs):
    return [c.id for c in (my_state(obs).discard or []) if c]


def energy_count(pokemon):
    if pokemon is None:
        return 0
    return len(getattr(pokemon, "energies", []) or [])


def water_energy_on(pokemon):
    """Count real Water energy attached (Ignition provides only colorless)."""
    if pokemon is None:
        return 0
    return sum(1 for c in (getattr(pokemon, "energyCards", None) or [])
               if c and c.id == WATER_ENERGY)


def retreat_cost(pokemon):
    data = CARD_DB.get(pokemon.id) if pokemon else None
    return getattr(data, "retreatCost", 0) if data else 0


def damage_on(pokemon):
    if pokemon is None:
        return 0
    return max(0, getattr(pokemon, "maxHp", pokemon.hp) - pokemon.hp)


def count_in_play(obs, card_id):
    return sum(1 for p in all_my_pokemon(obs) if p.id == card_id)


def has_in_play(obs, card_id):
    return any(p.id == card_id for p in all_my_pokemon(obs))


def prize_value(pokemon):
    data = CARD_DB.get(pokemon.id) if pokemon else None
    if data and getattr(data, "megaEx", False):
        return 3
    if data and getattr(data, "ex", False):
        return 2
    return 1


def is_water_weak(pokemon):
    data = CARD_DB.get(pokemon.id) if pokemon else None
    w = getattr(data, "weakness", None) if data else None
    return w is not None and int(getattr(w, "value", w)) == WATER_TYPE


def jetting_damage(target):
    return 120 * 2 if is_water_weak(target) else 120


def want_nebula(obs):
    """True when we should burst Nebula Beam (210, ignores Weakness/effects) this
    turn instead of Jetting Blow: only the Active Mega Starmie attacks, and Jetting
    Blow would not outright KO the opponent's Active (so the extra raw damage and
    effect-ignoring of Nebula matter -- e.g. vs high-HP metal walls behind Full
    Metal Lab, which reduces Jetting Blow but not Nebula)."""
    active = active_pokemon(obs)
    if active is None or active.id != MEGA_STARMIE:
        return False
    opp = opp_active(obs)
    if opp is None:
        return False
    return jetting_damage(opp) < opp.hp


# ── Attack planning ──

def best_attacker(obs):
    """The Mega Starmie ex we intend to attack with (active preferred)."""
    active = active_pokemon(obs)
    if active and active.id == MEGA_STARMIE:
        return active
    for p in my_state(obs).bench:
        if p and p.id == MEGA_STARMIE:
            return p
    return None


def can_ko_active_now(obs):
    """Return (can_ko, best_damage) against opp active with a ready attack option."""
    opp = opp_active(obs)
    if opp is None:
        return False, 0
    best = 0
    for opt in obs.select.option:
        if opt.type == OptionType.ATTACK:
            if opt.attackId == JETTING_BLOW:
                best = max(best, jetting_damage(opp))
            elif opt.attackId == NEBULA_BEAM:
                best = max(best, 210)
            elif opt.attackId == WATER_GUN:
                best = max(best, 20 * (2 if is_water_weak(opp) else 1))
    return best >= opp.hp, best


# ── Scoring: SETUP ──

def score_setup(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else None
    ctx = obs.select.context

    if ctx == SelectContext.MULLIGAN:
        return (10000, "no mulligan") if opt.type == OptionType.NO else (0, "mulligan")
    if ctx == SelectContext.IS_FIRST:
        # Go FIRST: you cannot evolve on your first turn, so going first means our
        # first turn is turn 1 and we evolve Staryu -> Mega Starmie ex (+attack) on
        # turn 3, a full turn sooner than going second (evolve turn 4). Tempo wins.
        return (10000, "go first") if opt.type == OptionType.YES else (0, "go second")
    if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
        if cid == STARYU:
            return 10000, "Active: Staryu (main line)"
        if cid == DUSKULL:
            return 3000, "Active: Duskull fallback"
        return 100, "Active: other basic"
    if ctx == SelectContext.SETUP_BENCH_POKEMON:
        if cid == STARYU:
            return 8000, "bench Staryu"
        if cid == DUSKULL:
            return 7000, "bench Duskull (start bomb line)"
        return 1000, "bench basic"
    return 0, "non-setup"


# ── Scoring: PLAY ──

def _need_bomb_piece(obs):
    """How far the Dusknoir line is set up (want at least one line going)."""
    return count_in_play(obs, DUSKULL) + count_in_play(obs, DUSCLOPS) + count_in_play(obs, DUSKNOIR)


def _want_bomb(obs):
    """Only invest search/draw resources in the slow Dusknoir bomb line once the
    Starmie engine is secured (a Mega Starmie ex is online, or we already have
    two of the Staryu line down). Before that, race with Starmie -- committing to
    a 2-2-2 Stage-2 line early loses tempo, esp. vs aggro."""
    starmie_secure = has_in_play(obs, MEGA_STARMIE) or \
        (count_in_play(obs, STARYU) + count_in_play(obs, MEGA_STARMIE)) >= 2
    return starmie_secure


def score_play_pokemon(obs, cid):
    bench_free = my_state(obs).benchMax - len([p for p in my_state(obs).bench if p])
    if bench_free <= 0:
        return -1000, "bench full"
    if cid == STARYU:
        n = count_in_play(obs, STARYU) + count_in_play(obs, MEGA_STARMIE)
        if n == 0:
            return 21000, "bench Staryu (need attacker)"
        if n < 2:
            return 18500, "bench Staryu (backup)"
        return 3000, "bench extra Staryu"
    if cid == DUSKULL:
        if _need_bomb_piece(obs) == 0 and _want_bomb(obs):
            return 18000, "bench Duskull (start bomb)"
        if _need_bomb_piece(obs) == 0:
            return 9000, "bench Duskull (free board)"
        return 5000, "bench extra Duskull"
    return 15000, "bench Pokemon"


def _safe_discard_available(obs, need):
    """Count cards in hand we are willing to discard for Ultra Ball (need 2)."""
    ids = hand_ids(obs)
    safe = 0
    counts = {}
    for c in ids:
        counts[c] = counts.get(c, 0) + 1
    # duplicate energies, duplicate items, spare supporters
    water = counts.get(WATER_ENERGY, 0)
    if water > 1:
        safe += water - 1
    for cid, cnt in counts.items():
        if cid in (POKEGEAR, POKE_PAD, JUDGE, POFFIN) and cnt > 0:
            safe += min(cnt, 2)
    return safe >= need


def score_play(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else None
    data = CARD_DB.get(cid)
    if data is None:
        return 500, "unknown play"
    ids = hand_ids(obs)
    st = obs.current
    dc = my_state(obs).deckCount

    if data.cardType == CardType.POKEMON:
        return score_play_pokemon(obs, cid)

    # ── Items (no supporter guard) ──
    if cid == POFFIN:
        # Search 2 basics <=70HP: Staryu + Duskull. Great when developing board.
        need = (count_in_play(obs, STARYU) + count_in_play(obs, MEGA_STARMIE) < 2) or \
            (_want_bomb(obs) and _need_bomb_piece(obs) == 0)
        bench_free = my_state(obs).benchMax - len([p for p in my_state(obs).bench if p])
        if bench_free <= 0:
            return -500, "Poffin: bench full"
        return (20000 if need else 12000), "Poffin: search basics"

    if cid == POKE_PAD:
        # Search a non-ex Pokemon into hand (Staryu / Duskull / Dusclops / Dusknoir).
        if dc <= 1:
            return -500, "Poke Pad: deck too thin"
        return 15000, "Poke Pad: search Pokemon"

    if cid == ULTRA_BALL:
        # Discard 2, search any Pokemon (incl. Mega Starmie ex).
        if not _safe_discard_available(obs, 2):
            return -1000, "Ultra Ball: no safe discards"
        need_mega = has_in_play(obs, STARYU) and not has_in_play(obs, MEGA_STARMIE) \
            and MEGA_STARMIE not in ids
        need_line = (count_in_play(obs, STARYU) + count_in_play(obs, MEGA_STARMIE) < 2)
        if need_mega:
            return 20000, "Ultra Ball: get Mega Starmie ex"
        if need_line:
            return 14000, "Ultra Ball: search Starmie line"
        if _want_bomb(obs) and _need_bomb_piece(obs) < 2:
            return 12000, "Ultra Ball: search bomb line"
        return -800, "Ultra Ball: nothing to search"

    if cid == POKEGEAR:
        if st.supporterPlayed:
            return 9000, "Pokegear: dig for supporter"
        # Only worth it if we don't already have a supporter in hand.
        has_sup = any(c in SUPPORTERS for c in ids)
        return (7000 if has_sup else 13000), "Pokegear: find supporter"

    if cid == DELUXE_BOMB:
        # ACE SPEC tool: retaliation. Attach to the Active attacker.
        active = active_pokemon(obs)
        if active and active.id == MEGA_STARMIE and not (getattr(active, "tools", None) or []):
            return 4000, "Deluxe Bomb on Active"
        return -500, "save Deluxe Bomb"

    # ── Supporters (once per turn) ──
    if cid in SUPPORTERS:
        if st.supporterPlayed:
            return -1000, "supporter already used"

    if cid == HILDA:
        # Search Evolution + Energy. Excellent to grab Mega Starmie ex (+Water) or
        # a bomb-line evolution when setting up.
        need_mega = has_in_play(obs, STARYU) and not has_in_play(obs, MEGA_STARMIE) \
            and MEGA_STARMIE not in ids
        need_evo = need_mega or (_want_bomb(obs) and (
            (has_in_play(obs, DUSKULL) and DUSCLOPS not in ids)
            or (has_in_play(obs, DUSCLOPS) and DUSKNOIR not in ids)))
        if dc <= 1:
            return -500, "Hilda: deck too thin"
        return (17000 if need_evo else 9000), "Hilda: search evo+energy"

    if cid == CARMINE:
        # Discard hand, draw 5. Strong early raw draw / hand reset.
        if dc <= 3:
            return -600, "Carmine: deck too thin"
        hc = my_state(obs).handCount
        if st.turn <= 3 or hc <= 3:
            return 8000, "Carmine: draw 5"
        return 3000, "Carmine: refuel"

    if cid == LILLIE:
        if dc <= 4:
            hc = my_state(obs).handCount
            if hc >= dc + 4:
                return 16000, "Lillie: refill deck"
            return -600, "Lillie: deck too thin"
        hc = my_state(obs).handCount
        if hc <= 2:
            return 8500, "Lillie: draw 6 (empty hand)"
        if st.turn <= 3:
            return 6000, "Lillie: draw 6"
        return 3500, "Lillie: refuel"

    if cid == JUDGE:
        if dc <= 4:
            return -600, "Judge: deck too thin"
        if opp_state(obs).handCount >= 6:
            return 14000, "Judge: disrupt big opp hand"
        if my_state(obs).handCount <= 2:
            return 5000, "Judge: reset our small hand"
        return -400, "save Judge"

    if cid == WALLY:
        # Heal a Mega ex fully + return its energy to hand. Defensive; only when a
        # Mega Starmie ex is meaningfully damaged.
        best = None
        for p in all_my_pokemon(obs):
            if p.id == MEGA_STARMIE and damage_on(p) >= 120:
                if best is None or damage_on(p) > damage_on(best):
                    best = p
        if best is not None:
            return 12000, "Wally: heal damaged Mega Starmie"
        return -800, "save Wally"

    return 1500, "generic play"


# ── Scoring: EVOLVE ──

def score_evolve(obs, opt):
    card = option_card(obs, opt)
    target = option_target(obs, opt)
    cid = card.id if card else None
    tid = target.id if target else None
    active_is_target = opt.inPlayArea == AreaType.ACTIVE

    if cid == MEGA_STARMIE and tid == STARYU:
        # One-step to a 330 HP attacker. Evolve the Active first (get attacking).
        if active_is_target:
            return 28000, "evolve Active -> Mega Starmie ex"
        # Bench Mega Starmie only if we already have an attacker up.
        if has_in_play(obs, MEGA_STARMIE):
            return 12000, "evolve bench -> Mega Starmie ex (backup)"
        return 8000, "evolve bench -> Mega Starmie ex"

    if cid == DUSKNOIR and tid == DUSCLOPS:
        return 22000, "evolve -> Dusknoir (bomb ready)"
    if cid == DUSCLOPS and tid == DUSKULL:
        return 20000, "evolve -> Dusclops"

    return 10000, "generic evolve"


# ── Scoring: ATTACH ──

def score_attach(obs, opt):
    card = option_card(obs, opt)
    target = option_target(obs, opt)
    cid = card.id if card else None
    tid = target.id if target else None
    if obs.current.energyAttached:
        return -1000, "already attached this turn"
    if target is None:
        return -500, "no attach target"

    area = opt.inPlayArea
    is_active = area == AreaType.ACTIVE

    # Only ever fuel the Starmie line (bomb line uses abilities, needs no energy).
    if tid not in STARMIE_LINE:
        return -800, "skip: not a Starmie attacker"

    attacker_ready = tid == MEGA_STARMIE
    e = energy_count(target)
    w = water_energy_on(target)

    if cid == WATER_ENERGY:
        # Water enables Jetting Blow (needs exactly 1 real Water) and counts toward
        # Nebula Beam. Highest value: first Water onto the Active Mega Starmie.
        if tid == MEGA_STARMIE:
            base = 14000 if is_active else 9000
            if w == 0:
                base += 2000  # unlock Jetting Blow
            if e >= 3:
                base -= 8000  # already fully fueled
            return base, "Water -> Mega Starmie ex"
        if tid == STARYU:
            # Pre-fuel a Staryu that will evolve; modest.
            return 5000, "Water -> Staryu (pre-fuel)"
        return 2000, "Water attach"

    if cid == IGNITION_ENERGY:
        # Ignition = 3 colorless on an Evolution (one-turn burst -> Nebula Beam 210),
        # discarded end of turn. Prime use: burst Nebula Beam vs a target Jetting
        # Blow can't KO (high-HP / effect walls). Attaching it to the Active Mega
        # Starmie ex (with <3 energy) instantly powers Nebula this turn.
        if tid == MEGA_STARMIE and is_active and e < 3:
            if want_nebula(obs):
                return 17000, "Ignition -> Mega Starmie ex (Nebula burst)"
            if e == 0:
                # No energy at all: Ignition lets us attack (Nebula) now.
                return 7000, "Ignition -> Mega Starmie ex (enable attack)"
        return -600, "save Ignition"

    return 1000, "generic attach"


# ── Scoring: ABILITY (Cursed Blast bomb) ──

def _cursed_blast_value(obs, dmg):
    """Value of firing a Cursed Blast that places ``dmg`` on the best target.

    Self-KOs the user (gives opp 1 prize), so only fire it when it KOs a target
    (prize-positive vs a >=2 prize target, or reaches lethal), or finishes a big
    threat. Returns (score, reason).
    """
    my_prizes = len(my_state(obs).prize)
    targets = ([opp_active(obs)] if opp_active(obs) else []) + opp_bench(obs)
    best = -500
    best_reason = "save Cursed Blast"
    for t in targets:
        if t is None:
            continue
        pv = prize_value(t)
        kos = dmg >= t.hp
        if kos:
            # Lethal: KO gives >= our remaining prizes (minus 1 we hand back).
            if pv >= my_prizes:
                return 90000, "LETHAL Cursed Blast KO"
            # Prize-positive trade only if target is worth >=2 (we give back 1).
            if pv >= 2:
                s = 15000 + pv * 1000 + t.hp
                if s > best:
                    best, best_reason = s, "Cursed Blast KO (ex target)"
            else:
                s = 3000 + t.hp // 10
                if s > best:
                    best, best_reason = s, "Cursed Blast KO (1-prize, even trade)"
        else:
            # Not a KO alone: only good if it sets up a Jetting Blow / Nebula KO.
            opp_a = opp_active(obs)
            if t is opp_a:
                jb = jetting_damage(t)
                if dmg + jb >= t.hp or dmg + 210 >= t.hp:
                    s = 8000 + pv * 500
                    if s > best:
                        best, best_reason = s, "Cursed Blast sets up KO"
    return best, best_reason


def score_ability(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else None
    if cid == DUSKNOIR:
        return _cursed_blast_value(obs, 130)
    if cid == DUSCLOPS:
        # Only 50, and sacrifices a piece that could become Dusknoir. Conservative.
        s, r = _cursed_blast_value(obs, 50)
        # Prefer keeping Dusclops to evolve unless it KOs something valuable.
        if not has_in_play(obs, DUSKNOIR) and DUSKNOIR in hand_ids(obs):
            return min(s, 2000), r + " (hold for Dusknoir)"
        return s, r
    return 1, "ability"


# ── Scoring: RETREAT ──

def score_retreat(obs):
    active = active_pokemon(obs)
    if active is None:
        return -100, "no active"
    # Retreat only to bring an attacker up when a bomb-line piece is stuck active.
    if active.id in BOMB_LINE:
        has_attacker = any(p and p.id in STARMIE_LINE for p in my_state(obs).bench)
        if has_attacker:
            return 12000, "retreat: bring up Starmie"
    if active.id == STARYU and has_in_play(obs, MEGA_STARMIE):
        # Prefer the Mega Starmie active if a benched one is ready.
        for p in my_state(obs).bench:
            if p and p.id == MEGA_STARMIE:
                return 9000, "retreat: promote Mega Starmie ex"
    return -200, "avoid retreat"


# ── Scoring: ATTACK ──

def score_attack(obs, opt):
    opp = opp_active(obs)
    aid = opt.attackId
    if aid == JETTING_BLOW:
        dmg = jetting_damage(opp) if opp else 120
    elif aid == NEBULA_BEAM:
        dmg = 210
    elif aid == WATER_GUN:
        dmg = (20 * 2 if (opp and is_water_weak(opp)) else 20)
    else:
        dmg = int(getattr(opt, "number", 0) or 50)

    if opp is not None:
        my_prizes = len(my_state(obs).prize)
        if dmg >= opp.hp:
            # KO. If it wins the game, top priority. Among KO attacks prefer the
            # cheaper/sustainable Jetting Blow (keeps the Ignition burst for later).
            if prize_value(opp) >= my_prizes:
                return 100000, "LETHAL attack"
            bonus = 300 if aid == JETTING_BLOW else 0
            return 5000 + dmg + bonus, "KO attack"
    # No KO: score by raw damage, so Nebula Beam (210, ignores effects) is chosen
    # over Jetting Blow (120) when both are available (we already burned Ignition to
    # enable it). Small bonus for Jetting's 50 bench snipe as a tiebreak.
    bonus = 30 if aid == JETTING_BLOW else 0
    return dmg + bonus, "attack"


# ── Scoring: card-selection contexts ──

def score_to_hand(obs, opt):
    """Choosing cards to put into hand (Poke Pad / Hilda / Pokegear / Ultra Ball
    search results, or generic draw-selection)."""
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    ids = hand_ids(obs)
    effect = getattr(obs.select, "effect", None)
    eff_id = effect.id if effect else None

    have_mega = has_in_play(obs, MEGA_STARMIE) or MEGA_STARMIE in ids
    have_staryu = has_in_play(obs, STARYU) or STARYU in ids
    line_count = count_in_play(obs, STARYU) + count_in_play(obs, MEGA_STARMIE)

    # Hilda: pick one Evolution + one Energy. Rank evolutions and energies high.
    if eff_id == HILDA:
        if cid == MEGA_STARMIE and has_in_play(obs, STARYU) and not have_mega:
            return 30000, "Hilda: Mega Starmie ex"
        if cid == DUSKNOIR and has_in_play(obs, DUSCLOPS):
            return 25000, "Hilda: Dusknoir"
        if cid == DUSCLOPS and has_in_play(obs, DUSKULL):
            return 24000, "Hilda: Dusclops"
        if cid == MEGA_STARMIE:
            return 20000, "Hilda: Mega Starmie ex (future)"
        if cid == WATER_ENERGY:
            return 15000, "Hilda: Water energy"
        if cid == IGNITION_ENERGY:
            return 12000, "Hilda: Ignition energy"

    # Ultra Ball: get the best Pokemon.
    if eff_id == ULTRA_BALL:
        if cid == MEGA_STARMIE and have_staryu and not have_mega:
            return 30000, "UB: Mega Starmie ex"
        if cid == STARYU and line_count == 0:
            return 28000, "UB: Staryu"
        if cid == DUSKNOIR and has_in_play(obs, DUSCLOPS):
            return 24000, "UB: Dusknoir"
        if cid == DUSCLOPS and has_in_play(obs, DUSKULL):
            return 22000, "UB: Dusclops"
        if cid == DUSKULL and _need_bomb_piece(obs) == 0 and _want_bomb(obs):
            return 20000, "UB: Duskull"
        if cid == STARYU:
            return 15000, "UB: Staryu backup"

    # Poke Pad: non-ex Pokemon only.
    if eff_id == POKE_PAD:
        if cid == STARYU and line_count < 2:
            return 25000, "Pad: Staryu"
        if cid == DUSKNOIR and has_in_play(obs, DUSCLOPS):
            return 24000, "Pad: Dusknoir"
        if cid == DUSCLOPS and has_in_play(obs, DUSKULL):
            return 22000, "Pad: Dusclops"
        if cid == DUSKULL and _need_bomb_piece(obs) == 0 and _want_bomb(obs):
            return 20000, "Pad: Duskull"

    # Pokegear: pick a supporter (prefer Hilda early, else draw).
    if eff_id == POKEGEAR:
        if cid == HILDA:
            return 20000, "Pokegear: Hilda"
        if cid == CARMINE:
            return 16000, "Pokegear: Carmine"
        if cid == LILLIE:
            return 15000, "Pokegear: Lillie"
        if cid == JUDGE:
            return 8000, "Pokegear: Judge"
        if cid in SUPPORTERS:
            return 10000, "Pokegear: supporter"

    # Generic ranking (e.g. Poffin puts basics to bench; other TO_HAND).
    if cid == MEGA_STARMIE and not have_mega:
        return 22000, "take Mega Starmie ex"
    if cid == STARYU and line_count < 2:
        return 20000, "take Staryu"
    if cid == DUSKNOIR and has_in_play(obs, DUSCLOPS):
        return 19000, "take Dusknoir"
    if cid == DUSCLOPS and has_in_play(obs, DUSKULL):
        return 18000, "take Dusclops"
    if cid == DUSKULL and _need_bomb_piece(obs) == 0 and _want_bomb(obs):
        return 17000, "take Duskull"
    if cid == WATER_ENERGY:
        return 10000, "take Water"
    if cid == IGNITION_ENERGY:
        return 8000, "take Ignition"
    if cid in SUPPORTERS:
        return 6000, "take supporter"
    return 3000, "generic take"


# Cards we never want to discard (line pieces, our few energies) unless forced.
def score_discard(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    ids = hand_ids(obs)
    counts = {c: ids.count(c) for c in set(ids)}

    if cid == MEGA_STARMIE:
        return (-5000 if counts.get(cid, 0) <= 1 else 2000), "keep Mega Starmie ex"
    if cid in (STARYU, DUSKULL, DUSCLOPS, DUSKNOIR):
        return (-4000 if counts.get(cid, 0) <= 1 else 3000), "keep line piece"
    if cid == WATER_ENERGY:
        return (12000 if counts.get(cid, 0) > 1 else -2000), "Water discard"
    if cid == IGNITION_ENERGY:
        return 9000, "discard Ignition"
    if cid == DELUXE_BOMB:
        return 6000, "discard Deluxe Bomb"
    if cid in (POKEGEAR, POKE_PAD, POFFIN) and counts.get(cid, 0) > 1:
        return 11000, "discard duplicate item"
    if cid == JUDGE and counts.get(cid, 0) > 1:
        return 10000, "discard duplicate Judge"
    if cid in SUPPORTERS and counts.get(cid, 0) > 1:
        return 8500, "discard duplicate supporter"
    if cid in (POKEGEAR, POKE_PAD, POFFIN):
        return 7000, "discard spare item"
    if cid in SUPPORTERS:
        return 4000, "discard supporter"
    return 1000, "generic discard"


def score_target(obs, opt):
    """Targeting contexts: promotions, Boss-like gusts, Jetting Blow snipe,
    Cursed Blast damage-counter placement, healing."""
    ctx = obs.select.context
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    yi = obs.current.yourIndex
    pi = getattr(opt, "playerIndex", yi)
    targets_opp = pi is not None and pi != yi

    # Placing damage counters (Cursed Blast) / dealing effect damage on opponents.
    if ctx in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY,
               SelectContext.DAMAGE, SelectContext.EFFECT_TARGET):
        if card is None:
            return 500, "target"
        if targets_opp or ctx in (SelectContext.DAMAGE_COUNTER,
                                  SelectContext.DAMAGE_COUNTER_ANY,
                                  SelectContext.DAMAGE):
            pv = prize_value(card)
            # Prefer a target we can KO; among those, highest prize; else lowest HP.
            remain = getattr(obs.select, "remainDamageCounter", 0) or 0
            dmg = remain * 10 if remain else 130
            kos = dmg >= card.hp
            base = 5000 + pv * 1500
            if kos:
                base += 4000
            base += max(0, 400 - card.hp // 10)  # nudge toward finishing low HP
            return base, "place damage on opp"
        return 1000, "target"

    if ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        if targets_opp and card is not None:
            # Boss-like gust: drag the most valuable / killable opponent up.
            pv = prize_value(card)
            _, best_dmg = can_ko_active_now(obs)
            killable = best_dmg >= card.hp
            base = 5000 + pv * 2000 + energy_count(card) * 100
            if killable:
                base += 8000
            return base, "gust opp target"
        # Promote our own: Mega Starmie ex > Staryu > bomb line.
        if cid == MEGA_STARMIE:
            return 15000, "promote Mega Starmie ex"
        if cid == STARYU:
            return 9000, "promote Staryu"
        return 1000, "promote"

    if ctx in (SelectContext.TO_FIELD, SelectContext.TO_BENCH):
        if cid == STARYU:
            return 16000, "field Staryu"
        if cid == DUSKULL:
            return 15000, "field Duskull"
        if cid == MEGA_STARMIE:
            return 14000, "field Mega Starmie ex"
        return 3000, "field Pokemon"

    if ctx == SelectContext.HEAL:
        return 20000 + damage_on(card), "heal most damaged"

    if ctx in (SelectContext.ATTACH_TO, SelectContext.ATTACH_FROM):
        if cid in STARMIE_LINE:
            return 6000 - energy_count(card) * 500, "attach to Starmie"
        return 1000, "attach target"

    return 1000, "generic target"


# ── Dispatch ──

def score_option(obs, opt):
    ctx = obs.select.context

    if ctx in (SelectContext.IS_FIRST, SelectContext.MULLIGAN,
               SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON):
        return score_setup(obs, opt)

    t = opt.type
    if t in (OptionType.YES, OptionType.NO):
        if ctx == SelectContext.ACTIVATE:
            return (50000, "activate yes") if t == OptionType.YES else (-50000, "no")
        return (1, "yes") if t == OptionType.YES else (0, "no")
    if t == OptionType.NUMBER:
        # Damage-counter count / draw count: place/take the max (finish a KO).
        return (opt.number or 0) * 100, "number(max)"

    if ctx == SelectContext.MAIN:
        if t == OptionType.PLAY:
            return score_play(obs, opt)
        if t == OptionType.EVOLVE:
            return score_evolve(obs, opt)
        if t == OptionType.ATTACH:
            return score_attach(obs, opt)
        if t == OptionType.ABILITY:
            return score_ability(obs, opt)
        if t == OptionType.RETREAT:
            return score_retreat(obs)
        if t == OptionType.ATTACK:
            return score_attack(obs, opt)
        if t == OptionType.END:
            return 0, "end turn"
        return 500, "generic MAIN"

    if ctx == SelectContext.ATTACK:
        return score_attack(obs, opt)
    if ctx == SelectContext.TO_HAND:
        return score_to_hand(obs, opt)
    if ctx in (SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
               SelectContext.DISCARD_ENERGY_CARD, SelectContext.DISCARD_ENERGY):
        return score_discard(obs, opt)
    if t == OptionType.CARD or ctx in (
            SelectContext.SWITCH, SelectContext.TO_ACTIVE, SelectContext.TO_FIELD,
            SelectContext.TO_BENCH, SelectContext.HEAL, SelectContext.DAMAGE,
            SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY,
            SelectContext.EFFECT_TARGET, SelectContext.ATTACH_TO,
            SelectContext.ATTACH_FROM):
        return score_target(obs, opt)
    if t == OptionType.ATTACK:
        return score_attack(obs, opt)
    if t == OptionType.END:
        return 0, "end"
    return 100, "fallback"


def choose_options(obs):
    scored = []
    for i, opt in enumerate(obs.select.option):
        try:
            score, reason = score_option(obs, opt)
        except Exception as e:  # noqa: BLE001
            score, reason = -999999, f"error {type(e).__name__}: {e}"
        scored.append((score, i, reason))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    if DEBUG:
        top = scored[0]
        opt = obs.select.option[top[1]]
        # obs enum fields arrive as raw ints; format defensively.
        sys.stderr.write(
            f"[starmie] ctx={int(obs.select.context)} maxC={obs.select.maxCount} "
            f"pick={int(opt.type)} score={top[0]} :: {top[2]}\n")

    selected = []
    for score, i, _ in scored:
        if len(selected) >= obs.select.maxCount:
            break
        if score < 0 and len(selected) >= obs.select.minCount:
            continue
        selected.append(i)

    if len(selected) < obs.select.minCount:
        selected = [i for _, i, _ in scored[:obs.select.minCount]]

    return selected


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    if not obs.select.option:
        return []
    try:
        return choose_options(obs)
    except Exception:  # noqa: BLE001
        n = len(obs.select.option)
        k = max(obs.select.minCount, 1)
        return random.sample(range(n), min(k, n))


class StarmieAgent:
    """run_eval Agent-protocol adapter over the module-level ``agent`` fn."""

    def __init__(self, deck, engine=None):
        self._deck = list(deck)

    def reset(self, seed=0):
        random.seed(seed)

    def __call__(self, obs_dict):
        if obs_dict.get("select") is None:
            self.reset(0)
            return list(self._deck)
        return agent(obs_dict)
