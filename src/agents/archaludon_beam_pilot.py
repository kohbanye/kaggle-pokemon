"""Within-turn beam search on top of the ``archaludon_judge`` rule pilot.

The ``archaludon_judge`` pilot (``archaludon_judge_pilot.py``) is a strong
rule-based *additive-option-scorer*: at every decision it scores each legal
option and greedily picks the highest. Greedy scoring can, however, get the
*ordering* of a turn's actions wrong (e.g. attach-then-evolve vs evolve-then-
attach, or which of several plays to fire before attacking). This module keeps
the rule pilot's priors but *searches* over our own turn's action SEQUENCE to
end-of-turn and picks the sequence that reaches the best end-of-turn STATE.

Why this shape (and not the shelved cross-turn ISMCTS): we NEVER search into the
opponent's hidden turn -- the beam freezes at the turn boundary -- so opponent
hidden-info determinization (which sank ISMCTS on the ladder) barely matters. We
also never replace the rule pilot's move priors with a naive value pick: search
candidates are the rule pilot's own top-K options, so search only REFINES the
already-good greedy order.

Design (measured on the native x86-64 engine):
  * candidate generation at each MAIN node = TOP-K options by the rule pilot's
    own ``score_option`` (K = ``CAND_K``);
  * expansion via ``search_begin``/``search_step`` within OUR turn only, beam
    width ``BEAM_WIDTH``, up to ``BEAM_DEPTH`` steps (end-of-turn);
  * sub-selections that a play spawns (discard for Ultra Ball, search for
    Explorer, ...) are resolved deterministically by the rule pilot's own
    ``choose_options`` -- no branching, keeps the good priors;
  * leaf eval at end-of-turn = ``_eval_state`` (prize differential dominant,
    plus our board / attacker fuel / active HP, minus opponent active HP and
    deck-out risk);
  * EXHAUSTIVE LETHAL: if any sequence reaches a winning board (we take our last
    prize / the engine reports our win) it is taken immediately.

Like the other bespoke pilots it imports the Linux-only ``cg`` engine at load
time, so it is NOT imported by ``src/agents/__init__``; ``scripts/run_eval.py``
wires it in lazily under the name ``archaludon_beam``. It is crash-proof: any
error at any level falls back to the plain rule pilot's choice.
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
    AreaType,
    CardType,
    OptionType,
    SelectContext,
    all_card_data,
    to_observation_class,
)

# Reuse the rule pilot wholesale: its score_option / choose_options are the move
# priors we refine, and its constants/board helpers describe the Archaludon plan.
from src.agents import archaludon_judge_pilot as ajp  # noqa: E402

_SEARCH_OK = False
try:
    from cg.api import search_begin, search_end, search_step  # noqa: E402
    _SEARCH_OK = True
except Exception:
    pass

# ---- Search mode --------------------------------------------------------------
# "lethal_only" (default): the beam is used ONLY to find a provable game-winning
#   sequence this turn (exhaustive lethal); on any non-winning turn we defer to
#   the rule pilot's exact choice. This can only add found kills, never override
#   the expert priors -- safe.
# "full": also re-order the whole turn to maximise the end-of-turn leaf value.
#   MEASURED NET-NEGATIVE (see module docstring / report): the crude leaf eval
#   overrides the rule pilot's tuned priors ~70% of the time and plays far worse
#   (beam vs judge same-deck 0.03). Kept for reproducibility, NOT for use.
BEAM_MODE = "lethal_only"

# ---- Tunable search budget (huge headroom under the 600s/game sandbox) --------
USE_SEARCH = True
SEARCH_TIME_BUDGET = 2.5   # seconds of beam expansion per MAIN decision (rarely binds; native engine search is fast)
ROOT_CANDIDATES = 4        # root branching over the rule pilot's top-k order
CAND_K = 3                 # branching at each interior MAIN node
BEAM_WIDTH = 3
BEAM_DEPTH = 16            # max within-turn expansion steps (to end-of-turn)

_WIN = 1e9

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

_BASIC_POKE_ID = next(
    (c.cardId for c in all_card if c.cardType == CardType.POKEMON and c.basic), 673,
)
_BASIC_ENERGY_ID = next(
    (c.cardId for c in all_card if c.cardType == CardType.BASIC_ENERGY), 6,
)

# Attacker card ids (fuel/board bonuses in the leaf eval).
_ARCH = ajp.ARCHALUDON_EX
_DURA = ajp.DURALUDON

# ---- module-level per-turn memory --------------------------------------------
_my_deck: list[int] = []
_pre_turn = -1


def _get_deck() -> list[int]:
    for path in ("deck.csv", f"{_CG_PATH}/deck.csv"):
        try:
            with open(path, encoding="utf-8") as f:
                d = [int(x) for x in f.read().splitlines() if x.strip()]
            if d:
                return d
        except Exception:
            continue
    return list(_my_deck)


# ---- leaf evaluation ---------------------------------------------------------


def _eval_state(obs) -> float:
    """Heuristic value of an end-of-(our)-turn state, from OUR perspective.

    Prize differential dominates; then our attacker board (Archaludon ex online
    and fueled, energy on metal attackers, active HP), minus the opponent's
    active HP (reward damage dealt this turn) and deck-out risk.
    """
    st = obs.current
    if st is None:
        return 0.0
    me = st.players[st.yourIndex]
    op = st.players[1 - st.yourIndex]

    # Terminal: prize list empty => that player has taken all their prizes.
    if len(me.prize) == 0:
        return _WIN
    if len(op.prize) == 0:
        return -_WIN

    val = (len(op.prize) - len(me.prize)) * 100000.0

    # Our board: attacker fuel + Archaludon-ex online bonus.
    for p in ([me.active[0]] if me.active else []) + list(me.bench):
        if p is None:
            continue
        e = len(p.energies)
        if p.id in (_ARCH, _DURA):
            val += min(e, 3) * 800.0
            if p.id == _ARCH:
                val += 600.0
                if e >= 3:
                    val += 1200.0  # fueled ex = ready to Metal Defender (220)

    # Active HP (survivability) and prize liability of an exposed ex.
    if me.active and me.active[0] is not None:
        a = me.active[0]
        val += a.hp * 3.0
        # A damaged active ex is a prize liability; nudge away from ending the
        # turn with a badly hurt ex in the Active slot when a healthier line
        # exists (only a small term -- prize diff dominates).
        max_hp = getattr(a, "maxHp", a.hp) or a.hp
        if a.id == _ARCH and max_hp > 0 and a.hp < max_hp * 0.34:
            val -= 400.0

    # Damage dealt: reward a low-HP opponent active (KO progress).
    if op.active and op.active[0] is not None:
        val -= op.active[0].hp * 4.0

    val += getattr(me, "handCount", 0) * 15.0
    if getattr(me, "deckCount", 60) < 4:
        val -= 4000.0
    return val


def _is_our_win(obs, my_index: int) -> bool:
    st = obs.current
    if st is None:
        return False
    if st.result is not None and st.result == my_index:
        return True
    me = st.players[my_index]
    return len(me.prize) == 0


# ---- candidate generation (rule-pilot priors) --------------------------------


def _ranked_main_options(obs) -> list[int]:
    """Option indices at a MAIN node, sorted by the rule pilot's score (desc)."""
    scored = []
    for i, opt in enumerate(obs.select.option):
        try:
            score, _ = ajp.score_option(obs, opt)
        except Exception:
            score = -1e9
        scored.append((score, -i, i))
    scored.sort(reverse=True)
    return [i for _, _, i in scored]


def _resolve_subcontext(obs) -> list[int]:
    """Resolve a non-MAIN sub-selection deterministically with the rule pilot."""
    try:
        sel = ajp.choose_options(obs)
    except Exception:
        sel = []
    n = len(obs.select.option)
    sel = [i for i in sel if 0 <= i < n]
    mn = max(1, obs.select.minCount)
    if len(sel) < mn:
        sel = (sel + [i for i in range(n) if i not in sel])[:mn]
    return sel[: max(mn, obs.select.maxCount)]


# ---- determinization (own deck exact, opponent coarse) -----------------------


def _pad(ids, n, filler):
    ids = list(ids)
    if len(ids) < n:
        ids += [filler] * (n - len(ids))
    return ids


def _determinize(obs, own_deck):
    st = obs.current
    me = st.players[st.yourIndex]
    op = st.players[1 - st.yourIndex]

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

    opp_deck = _pad([_BASIC_POKE_ID, _BASIC_POKE_ID], op.deckCount + 2, _BASIC_ENERGY_ID)
    opp_prize = _pad([], len(op.prize), _BASIC_ENERGY_ID)
    opp_hand = _pad([], op.handCount, _BASIC_ENERGY_ID)

    opp_active = []
    if op.active and (len(op.active) == 0 or op.active[0] is None):
        opp_active = [_BASIC_POKE_ID]

    return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active


# ---- beam search over our own turn -------------------------------------------


def _beam_choose(obs, base_order):
    """Beam search within our turn. Returns reordered option indices (best
    first) or ``None`` to fall back to the rule pilot order.
    """
    if not (_SEARCH_OK and USE_SEARCH):
        return None
    select = obs.select
    if select is None or select.context != SelectContext.MAIN:
        return None
    if obs.search_begin_input is None:
        return None
    if select.maxCount != 1 or select.minCount != 1:
        return None  # MAIN is single-select; anything else -> rule pilot

    my_index = obs.current.yourIndex
    t0 = time.time()
    try:
        det = _determinize(obs, _my_deck)
        root = search_begin(obs, *det)
    except Exception:
        return None

    try:
        # Root beam: expand the rule pilot's top candidates.
        beam = []  # (value, searchId, first_action, observation)
        for first in base_order[:ROOT_CANDIDATES]:
            try:
                s = search_step(root.searchId, [first])
            except Exception:
                continue
            if _is_our_win(s.observation, my_index):
                return [first] + [i for i in base_order if i != first]
            beam.append((_eval_state(s.observation), s.searchId, first, s.observation))
        if not beam:
            return None

        depth = 0
        while depth < BEAM_DEPTH:
            if time.time() - t0 > SEARCH_TIME_BUDGET:
                break
            next_beam = []
            expanded_any = False
            for val, sid, first_action, cur_obs in beam:
                cur = cur_obs.current
                # Leaf: game decided, or the turn has passed to the opponent.
                if cur is None or (cur.result is not None and cur.result != -1) \
                        or cur.yourIndex != my_index:
                    next_beam.append((val, sid, first_action, cur_obs))
                    continue
                if cur_obs.select is None or not cur_obs.select.option:
                    next_beam.append((val, sid, first_action, cur_obs))
                    continue
                if cur_obs.select.context != SelectContext.MAIN:
                    sel = _resolve_subcontext(cur_obs)
                    try:
                        s = search_step(sid, sel)
                    except Exception:
                        next_beam.append((val, sid, first_action, cur_obs))
                        continue
                    if _is_our_win(s.observation, my_index):
                        return [first_action] + [i for i in base_order if i != first_action]
                    next_beam.append(
                        (_eval_state(s.observation), s.searchId, first_action, s.observation))
                    expanded_any = True
                else:
                    for opt in _ranked_main_options(cur_obs)[:CAND_K]:
                        try:
                            s = search_step(sid, [opt])
                        except Exception:
                            continue
                        if _is_our_win(s.observation, my_index):
                            return [first_action] + [i for i in base_order if i != first_action]
                        next_beam.append(
                            (_eval_state(s.observation), s.searchId, first_action, s.observation))
                        expanded_any = True
            if not expanded_any:
                break
            beam = sorted(next_beam, key=lambda x: x[0], reverse=True)[:BEAM_WIDTH]
            depth += 1

        # No lethal found. In lethal_only mode defer to the rule pilot exactly;
        # in full mode re-order the turn by the leaf value (measured negative).
        if BEAM_MODE == "lethal_only":
            return None
        if not beam:
            return None
        best = max(beam, key=lambda x: x[0])
        best_first = best[2]
        return [best_first] + [i for i in base_order if i != best_first]
    except Exception:
        return None
    finally:
        try:
            search_end()
        except Exception:
            pass


# ---- agent -------------------------------------------------------------------


def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        return _my_deck if obs_dict.get("select") is None else [0]
    if obs.select is None:
        _reset_state()
        return list(_my_deck)

    global _pre_turn
    if obs.current is not None and _pre_turn != obs.current.turn:
        _pre_turn = obs.current.turn

    # Keep the rule pilot's opponent-attack tracking in sync (once per frame).
    try:
        ajp._update_opp_attack_tracking(obs)
    except Exception:
        pass

    if not obs.select.option:
        return []

    try:
        base_order = _ranked_main_options(obs) if obs.select.context == SelectContext.MAIN \
            else None
        if obs.select.context == SelectContext.MAIN:
            ordered = _beam_choose(obs, base_order)
            if ordered is not None:
                n = len(obs.select.option)
                ordered = [i for i in ordered if 0 <= i < n]
                k = max(min(obs.select.maxCount, n), min(max(1, obs.select.minCount), n))
                if ordered:
                    return ordered[:k]
        # Non-MAIN, or search declined/failed: plain rule pilot choice.
        return ajp.choose_options(obs)
    except Exception:
        n = len(obs.select.option)
        try:
            return random.sample(range(n), min(max(1, obs.select.minCount), n))
        except Exception:
            return list(range(min(max(1, obs.select.minCount), n)))


def _reset_state():
    global _pre_turn
    _pre_turn = -1
    ajp._opp_last_attack_id = None
    ajp._cur_turn_logs = []


class BeamAgent:
    """run_eval Agent adapter: within-turn beam search over the Judge rule pilot.

    Returns the injected deck on the DECK_REQUEST frame (ignoring deck.csv) and
    delegates every other frame to :func:`agent`; per-turn/search state lives in
    module globals, reset per game.
    """

    def __init__(self, deck, engine=None):
        global _my_deck
        self._deck = list(deck)
        _my_deck = list(deck)

    def reset(self, seed=0):
        global _my_deck
        _my_deck = list(self._deck)
        _reset_state()
        random.seed(seed)

    def __call__(self, obs_dict):
        if obs_dict.get("select") is None:
            self.reset(0)
            return list(self._deck)
        return agent(obs_dict)
