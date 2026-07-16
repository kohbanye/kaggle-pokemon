"""Self-contained Kaggle submission agent: ReBeL (belief + depth-limited MCCFR).

Unlike the greedy bundle, this ships the real `src/` packages (agents/net/rebel/search
+ the belief decklists + a value net) alongside `main.py` + `deck.csv` + `cg/`, and
builds a `RebelAgent`. It never crashes: any failure falls back to the bundled greedy_plus
pilot (also RebelAgent's own internal fallback), and a cumulative-game budget switches to
greedy_plus late in a game so the total time stays within the sandbox limit.

Assembled by `scripts/build_rebel_submission.py`.

Kaggle loading constraints this file MUST respect (learned the hard way from ladder
errors):
  * Kaggle exec()s this file WITHOUT defining ``__file__`` -> never touch __file__ at
    top level (guard it).
  * Kaggle uses the LAST callable defined in the module as the agent -> ``agent`` MUST be
    the last function defined here; do NOT define any function/class after it.
  * The agent runs in a NON-main thread -> no ``signal``/SIGALRM (raises ValueError).
"""

import os
import sys
import time

# --- bundle path setup: make `import src...` / `import cg` work from the archive root.
_KAGGLE = "/kaggle_simulations/agent"
_ROOTS = [_KAGGLE, os.getcwd()]
_f = globals().get("__file__")  # guarded: undefined under Kaggle's exec()
if _f:
    _ROOTS.append(os.path.dirname(os.path.abspath(_f)))
for _p in _ROOTS:
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# --- solver config (conservative: depth-3 + top_k beam => ~0.2-1.2s/solve locally) -----
_EXPLOIT = False          # eq for Dragapult midrange
_MAX_DEPTH = 3
_TOP_K = 3
_N_PARTICLES = 4
_SWEEPS = 16
_TPW = 4
_GAME_BUDGET_S = 480.0    # once a game has spent this much wall-clock, play greedy_plus

_STATE: dict = {}


def _deck_path() -> str:
    p = "deck.csv"
    if not os.path.exists(p):
        p = os.path.join(_KAGGLE, "deck.csv")
    return p


def _read_deck() -> list:
    with open(_deck_path()) as f:
        return [int(x) for x in f.read().split() if x.strip()]


def _npz_path():
    for root in (*_ROOTS, "."):
        p = os.path.join(root, "value_net.npz")
        if os.path.exists(p):
            return p
    return None


def _legal_fallback(select: dict) -> list:
    opts = (select or {}).get("option") or []
    lo = int((select or {}).get("minCount", 1) or 1)
    n = max(1, min(lo, len(opts)))
    return list(range(n)) if opts else [0]


def _init() -> dict:
    """Lazy one-time build of the engine data, value net, and RebelAgent."""
    from cg.api import all_attack, all_card_data  # noqa: PLC0415

    from src.agents import build_agent  # noqa: PLC0415
    from src.rebel.agent import RebelAgent  # noqa: PLC0415

    attacks = {a.attackId: {"dmg": int(a.damage), "cost": [int(e) for e in a.energies]}
               for a in all_attack()}
    cards = {c.cardId: {"hp": int(c.hp), "retreat": int(c.retreatCost),
                        "type": int(c.energyType),
                        "weak": None if c.weakness is None else int(c.weakness),
                        "ex": bool(c.ex), "mega": bool(c.megaEx), "basic": bool(c.basic),
                        "ctype": int(c.cardType), "attacks": list(c.attacks)}
             for c in all_card_data()}
    engine = {"attacks": attacks, "cards": cards}
    deck = _read_deck()

    value_net = None
    npz = _npz_path()
    if npz is not None:
        from src.rebel.value_net import ValueNet  # noqa: PLC0415
        value_net = ValueNet.load(npz)

    reb = RebelAgent(
        deck, engine, n_particles=_N_PARTICLES, max_depth=_MAX_DEPTH, sweeps=_SWEEPS,
        traversals_per_world=_TPW, top_k=_TOP_K, cfr_plus=True, exploit=_EXPLOIT,
        value_net=value_net, seed=0)
    reb.reset(0)
    fallback = build_agent("greedy_plus", deck, engine)
    fallback.reset(0)
    return {"agent": reb, "fallback": fallback, "deck": deck,
            "game_t": 0.0, "ready": True}


# NOTE: `agent` MUST be the LAST callable defined in this module (Kaggle picks the last
# callable as the agent). Do not add any def/class below it.
def agent(obs_dict: dict) -> list:
    # deck-selection request: obs.select is None -> return the 60 card ids. This call is
    # not on the per-move clock, so do the heavy one-time _init() HERE (not on move 1).
    if obs_dict.get("select") is None:
        try:
            if not _STATE.get("ready"):
                _STATE.update(_init())
        except Exception:  # noqa: BLE001 - init failure => plain greedy_plus later
            pass
        try:
            return _STATE.get("deck") or _read_deck()
        except Exception:  # noqa: BLE001
            return _read_deck()

    select = obs_dict.get("select") or {}
    try:
        if not _STATE.get("ready"):
            _STATE.update(_init())
        st = _STATE
        # new game? (a fresh battle resets to a low turn) -- reset the game clock + agent
        cur = obs_dict.get("current") or {}
        turn = int(cur.get("turn", 0)) if cur else 0
        if turn <= 1 and st.get("game_t", 0.0) > 0.0:
            st["game_t"] = 0.0
            st["agent"].reset(0)
            st["fallback"].reset(0)

        # budget guard: late in the game -> cheap greedy_plus (no search)
        if st["game_t"] > _GAME_BUDGET_S:
            return st["fallback"](obs_dict)

        t0 = time.perf_counter()
        move = st["agent"](obs_dict)  # RebelAgent: solves, falls back to gp on any error
        st["game_t"] += time.perf_counter() - t0
        if move is not None and len(move) > 0:
            return move
        return st["fallback"](obs_dict)
    except Exception:  # noqa: BLE001 - never crash a match
        try:
            return _STATE["fallback"](obs_dict)
        except Exception:  # noqa: BLE001
            return _legal_fallback(select)
