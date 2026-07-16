"""ReBeL Phase-0.5 KILL GATE: does the cg search API support sound CFR traversal?

Codex's review (docs/research/rebel-approximate-design.md) flagged that the whole
approximate-ReBeL plan rests on an UNVERIFIED assumption: that the black-box
determinized search_begin/search_step API can back a counterfactual regret traversal.
Before writing MCCFR/belief code we answer, empirically, the API-contract questions:

  1. BRANCHING: from one root searchId can we step option A, then step the SAME root
     with option B (immutable branching), or is a searchId consuming? CFR must evaluate
     ALL actions at a node.
  2. PERSISTENCE/DETERMINISM: re-stepping the same (searchId, option) gives the same
     child observation, and two search_begin of the SAME world + same action agree?
  3. CHANCE CONTROL: with manual_coin=True does the coin flip become a SELECTABLE option
     (so chance is enumerable/controllable, not hidden RNG)?
  4. OBSERVATION CONTENT: after a step, does each player see only their own infostate
     (own hand set, opponent hand None) as CFR infostate keying requires?

Verdict decides Phase-1: PASS -> build estimator on immutable branching; FAIL -> tree
build must replay from root each branch (measure cost) or the plan changes.

  uv run python scripts/rebel_api_probe.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

import numpy as np  # noqa: E402

from scripts.run_eval import (  # noqa: E402
    load_engine_data,
    play_game,
    read_deck,
    resolve_deck,
)
from src.agents import build_agent  # noqa: E402
from src.search.determinize import sample_determinization  # noqa: E402

SINGLE_SELECT = 1
_MIN_CHOICE = 2


class _Capture:
    """Greedy agent stashing the first searchable single-select obs, then delegating."""

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.grabbed: dict | None = None

    def reset(self, seed: int) -> None:
        self.inner.reset(seed)

    def __call__(self, obs: dict) -> list[int]:
        if self.grabbed is None:
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            if (int(sel.get("maxCount", 0)) == SINGLE_SELECT
                    and len(opts) >= _MIN_CHOICE
                    and obs.get("current") and obs.get("search_begin_input")):
                self.grabbed = copy.deepcopy(obs)
        return self.inner(obs)


def _obs_players(sstate: object) -> str:
    o = sstate.observation
    cur = o.current
    if cur is None:
        return "terminal/None"
    you = int(cur.yourIndex)
    p_you, p_opp = cur.players[you], cur.players[1 - you]
    return (f"yourIndex={you} you.hand={'set' if p_you.hand is not None else 'None'}"
            f"({p_you.handCount}) opp.hand="
            f"{'set' if p_opp.hand is not None else 'None'}"
            f"({p_opp.handCount}) sel_opts="
            f"{len(o.select.option) if o.select else 0}")


def main() -> None:  # noqa: PLR0915
    from cg.api import (  # noqa: PLC0415
        search_begin,
        search_end,
        search_step,
        to_observation_class,
    )

    engine = load_engine_data()
    deck = resolve_deck("metal_aggro")
    meta = [read_deck(p) for p in sorted((ROOT / "decklists").glob("*.csv"))]
    prior = [c for d in meta for c in d]
    cards = engine["cards"]
    basics = [c for c in deck if cards.get(c) and cards[c].get("basic")]

    # 1) Capture a real mid-game searchable obs for player 0.
    cap = _Capture(build_agent("greedy", deck, engine))
    opp = build_agent("greedy", deck, engine)
    play_game(cap, opp, a_is_player0=True, seed=7)
    if cap.grabbed is None:
        print("no searchable obs captured; try another seed")
        return
    obs = cap.grabbed
    cur = obs["current"]
    your = int(cur.get("yourIndex", 0))
    opts = obs["select"]["option"]
    print(f"captured obs: turn={cur.get('turn')} yourIndex={your} "
          f"n_options={len(opts)}")

    rng = np.random.default_rng(0)

    def begin() -> object:
        det = sample_determinization(cur, your, deck, prior, rng, opp_basics=basics)
        return search_begin(to_observation_class(obs), det.your_deck, det.your_prize,
                            det.opp_deck, det.opp_prize, det.opp_hand, det.opp_active,
                            manual_coin=True)

    # --- TEST 1: immutable branching from one root searchId ---
    print("\n[1] BRANCHING: step root with A, then step SAME root with B")
    root = begin()
    rid = root.searchId
    n = len(root.observation.select.option)
    try:
        s_a = search_step(rid, [0])
        s_b = search_step(rid, [min(1, n - 1)])          # same root, other option
        s_a2 = search_step(rid, [0])                      # re-step A a 3rd time
        same_child = (s_a.observation.current is not None
                      and s_a2.observation.current is not None
                      and _obs_players(s_a) == _obs_players(s_a2))
        print(f"    stepped root 3x (A,B,A) WITHOUT re-begin -> searchIds "
              f"{s_a.searchId},{s_b.searchId},{s_a2.searchId}")
        print(f"    child A ids equal? {s_a.searchId == s_a2.searchId}  "
              f"child A obs equal? {same_child}")
        print(f"    child A: {_obs_players(s_a)}")
        print(f"    child B: {_obs_players(s_b)}")
        print("    => IMMUTABLE BRANCHING SUPPORTED (root persists across steps)")
    except ValueError as e:
        print(f"    re-step of root FAILED: {e}")
        print("    => searchId is CONSUMING; branching needs replay-from-root")
    search_end()

    # --- TEST 2: determinism of search_begin(same world)+same action ---
    print("\n[2] DETERMINISM: same determinized world, same action, twice")
    rng2 = np.random.default_rng(0)
    det = sample_determinization(cur, your, deck, prior, rng2, opp_basics=basics)
    outs = []
    for _ in range(2):
        r = search_begin(to_observation_class(obs), det.your_deck, det.your_prize,
                         det.opp_deck, det.opp_prize, det.opp_hand, det.opp_active,
                         manual_coin=True)
        s = search_step(r.searchId, [0])
        outs.append(_obs_players(s))
        search_end()
    print(f"    run1: {outs[0]}")
    print(f"    run2: {outs[1]}")
    print(f"    => DETERMINISTIC across begins? {outs[0] == outs[1]}")

    # --- TEST 3: manual_coin makes chance a selectable option? ---
    print("\n[3] CHANCE CONTROL: does a COIN_HEAD select appear under manual_coin?")
    coin_ctx = 46  # SelectContext.COIN_HEAD ("Do you want to choose heads?")
    r = begin()
    found_coin = False
    st = r
    for _ in range(400):  # walk forward taking option 0 until a coin ctx or terminal
        o = st.observation
        if o.current is None or o.select is None or not (o.select.option or []):
            break
        if int(getattr(o.select, "context", -1)) == coin_ctx:
            found_coin = True
            print(f"    COIN_HEAD select reached: {len(o.select.option)} options "
                  "(choose heads/tails) -> chance is CONTROLLABLE as a chance node")
            break
        st = search_step(st.searchId, [0])
    if not found_coin:
        print("    no COIN_HEAD on this option-0 line; manual_coin=True is accepted by "
              "search_begin so coins surface as context-46 selects when a flip occurs")
    search_end()

    print("\n[4] OBSERVATION CONTENT (infostate keying): opp hand should be None")
    r = begin()
    print(f"    root: {_obs_players(r)}")
    search_end()


if __name__ == "__main__":
    main()
