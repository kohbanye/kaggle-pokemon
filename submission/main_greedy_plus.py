"""Self-contained Kaggle submission: greedy_plus (tempo-tuned heuristic pilot).

Mirrors ``src/agents/heuristic_agent.py`` with the ``greedy_plus`` config
(attach_target=False, retreat=False): keeps the tempo-POSITIVE features -- KO/prize-
aware attacks (ex=2, Mega ex=3 prizes), weakness doubling, Bench development, strongest-
promote -- and drops the two tempo-NEGATIVE ones (smart energy-spreading, retreat). It
beats plain greedy at equal deck on every deck tested (+3.5..+20pp local). Inlined
(the bundle has no ``src``); card/attack stats come from the bundled engine
(``cg.all_card_data``/``all_attack``) at startup. Never crashes: any failure returns a
guaranteed-legal selection.
"""

import os

# OptionType / SelectType / AreaType / CardType / EnergyType / SelectContext mirrors.
OPT_ABILITY, OPT_ATTACH, OPT_ATTACK, OPT_END = 10, 8, 13, 14
OPT_EVOLVE, OPT_PLAY, OPT_CARD = 9, 7, 3
SEL_MAIN, SEL_CARD = 0, 1
AREA_HAND, AREA_ACTIVE, AREA_BENCH = 2, 4, 5
CARD_POKEMON, CARD_ITEM, CARD_TOOL, CARD_SUPPORTER = 0, 1, 2, 3
ENERGY_COLORLESS, ENERGY_RAINBOW = 0, 10
CTX_SETUP_ACTIVE, CTX_SWITCH, CTX_TO_ACTIVE = 1, 3, 4

_MAX_DEVELOP_ACTIONS = 40
_BENCH_MIN = 2
_LETHAL_BASE, _LETHAL_PRIZE = 100_000.0, 1_000.0
_NEG_INF = float("-inf")


def _read_deck():
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path) as f:
        return [int(x) for x in f.read().split() if x.strip()]


def _load_engine():
    """cardId -> {hp, weak, ex, mega, basic, ctype, type, attacks}; attackId -> {dmg, cost}."""
    cards, attacks = {}, {}
    try:
        from cg.api import all_attack, all_card_data

        for a in all_attack():
            attacks[a.attackId] = {
                "dmg": int(a.damage),
                "cost": [int(getattr(e, "value", e)) for e in (a.energies or [])],
            }
        for c in all_card_data():
            weak = c.weakness
            cards[c.cardId] = {
                "hp": int(c.hp or 0),
                "weak": int(getattr(weak, "value", weak)) if weak is not None else None,
                "ex": bool(c.ex),
                "mega": bool(getattr(c, "megaEx", False)),
                "basic": bool(c.basic),
                "ctype": int(getattr(c.cardType, "value", c.cardType)),
                "type": int(getattr(c.energyType, "value", c.energyType)),
                "attacks": list(c.attacks or []),
            }
    except Exception:
        pass
    return cards, attacks


DECK = _read_deck()
CARDS, ATTACKS = _load_engine()


def _legal_fallback(select):
    return list(range(int(select.get("maxCount", 0))))


def _active(player):
    spot = player.get("active") or []
    return spot[0] if spot else None


def _pokemon_at(player, area, index):
    if index < 0:
        return None
    spot = {AREA_ACTIVE: player.get("active"), AREA_BENCH: player.get("bench"),
            AREA_HAND: player.get("hand")}.get(area) or []
    return spot[index] if 0 <= index < len(spot) else None


def _can_afford(cost, energies):
    pool = {}
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


def _eff_damage(attacker_type, target_card, base_dmg):
    if base_dmg > 0 and target_card is not None and target_card.get("weak") == attacker_type:
        return base_dmg * 2
    return base_dmg


def _prize_value(card):
    if card is None:
        return 1
    if card.get("mega"):
        return 3
    if card.get("ex"):
        return 2
    return 1


def _best_affordable_damage(pokemon, target_card):
    if pokemon is None:
        return 0
    card = CARDS.get(pokemon.get("id"))
    if card is None:
        return 0
    energies = pokemon.get("energies") or []
    atype = card.get("type", ENERGY_COLORLESS)
    best = 0
    for aid in card.get("attacks", []):
        info = ATTACKS.get(aid)
        if info is None or not _can_afford(info["cost"], energies):
            continue
        best = max(best, _eff_damage(atype, target_card, info["dmg"]))
    return best


def _best_play(options, idxs, me):
    need_bench = len(me.get("bench") or []) < _BENCH_MIN
    best_idx, best_score = idxs[0], _NEG_INF
    for i in idxs:
        played = _pokemon_at(me, AREA_HAND, int(options[i].get("index", -1)))
        card = CARDS.get(played.get("id")) if played else None
        ctype = card.get("ctype") if card else None
        score = 0.0
        if ctype == CARD_POKEMON and card and card.get("basic"):
            score = 3.0 if need_bench else 1.0
        elif ctype == CARD_SUPPORTER:
            score = 2.0
        elif ctype == CARD_ITEM:
            score = 1.5
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


def _best_attack(options, idxs, me, opp):
    my_active = _active(me)
    my_card = CARDS.get(my_active.get("id")) if my_active else None
    my_type = my_card.get("type", ENERGY_COLORLESS) if my_card else ENERGY_COLORLESS
    opp_active = _active(opp)
    opp_card = CARDS.get(opp_active.get("id")) if opp_active else None
    opp_hp = opp_active.get("hp") if opp_active else None
    best_idx, best_score = idxs[0], _NEG_INF
    for i in idxs:
        info = ATTACKS.get(options[i].get("attackId"), {})
        eff = _eff_damage(my_type, opp_card, info.get("dmg", 0))
        if opp_hp is not None and eff >= opp_hp and eff > 0:
            score = _LETHAL_BASE + _LETHAL_PRIZE * _prize_value(opp_card)
        else:
            score = float(eff)
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


def _choose_main(options, state):
    yidx = int(state.get("yourIndex", 0))
    players = state.get("players") or []
    me = players[yidx] if yidx < len(players) else {}
    opp = players[1 - yidx] if len(players) > 1 else {}
    by_type = {}
    for i, opt in enumerate(options):
        by_type.setdefault(int(opt["type"]), []).append(i)

    looping = int(state.get("turnActionCount", 0)) > _MAX_DEVELOP_ACTIONS
    if not looping:
        # greedy_plus develop order: attach (first offered), evolve, play (smart), ability.
        if OPT_ATTACH in by_type:
            return by_type[OPT_ATTACH][0]
        if OPT_EVOLVE in by_type:
            return by_type[OPT_EVOLVE][0]
        if OPT_PLAY in by_type:
            return _best_play(options, by_type[OPT_PLAY], me)
        if OPT_ABILITY in by_type:
            return by_type[OPT_ABILITY][0]
    if OPT_ATTACK in by_type:
        return _best_attack(options, by_type[OPT_ATTACK], me, opp)
    if OPT_END in by_type:
        return by_type[OPT_END][0]
    return 0 if options else None


def _choose_promote(select, state):
    if int(select.get("context", -1)) not in (CTX_SETUP_ACTIVE, CTX_SWITCH, CTX_TO_ACTIVE):
        return None
    yidx = int(state.get("yourIndex", 0))
    players = state.get("players") or []
    me = players[yidx] if yidx < len(players) else {}
    opp = players[1 - yidx] if len(players) > 1 else {}
    opp_active = _active(opp)
    opp_card = CARDS.get(opp_active.get("id")) if opp_active else None
    best_idx, best_score = None, _NEG_INF
    for i, opt in enumerate(select.get("option", [])):
        if int(opt.get("type", -1)) != OPT_CARD:
            continue
        pk = _pokemon_at(me, int(opt.get("area", -1)), int(opt.get("index", -1)))
        if pk is None:
            continue
        hp = pk.get("hp")
        if hp is None:
            hp = CARDS.get(pk.get("id"), {}).get("hp", 0)
        score = _best_affordable_damage(pk, opp_card) * 2.0 + hp * 0.1
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


def agent(obs_dict):
    """Kaggle contract: obs dict -> list of option indices (deck at init)."""
    try:
        select = obs_dict.get("select")
        if select is None:
            return list(DECK)
        max_count, min_count = int(select.get("maxCount", 0)), int(select.get("minCount", 0))
        state = obs_dict.get("current") or {}
        stype = int(select.get("type", -1))
        if max_count >= 1 and min_count <= 1:
            if stype == SEL_MAIN:
                idx = _choose_main(select["option"], state)
                if idx is not None:
                    return [idx]
            elif stype == SEL_CARD:
                idx = _choose_promote(select, state)
                if idx is not None:
                    return [idx]
        return _legal_fallback(select)
    except Exception:
        try:
            return _legal_fallback(obs_dict.get("select") or {})
        except Exception:
            return [0]
