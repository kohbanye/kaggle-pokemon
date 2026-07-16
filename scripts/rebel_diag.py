"""ReBeL oracle-decomposition diagnostic (Koh's research plan #1/#6/#11).

depth3 ~= depth4 means: with the CURRENT value net / CFR / action set / PBS repr, going
one ply deeper does NOT change the search teacher. This script finds WHICH axis is the
ceiling -- WITHOUT noisy win-rate. On a FIXED set of representative PBS it measures how
the ROOT strategy (policy, value, top action) moves when we independently swap:

  * SOLVER strength -- sweeps in {lo, mid, hi} (is CFR even converged?)
  * LEAF value -- net vs prize-heuristic vs greedy_plus rollout (does the leaf matter?)

Internal metrics (noise-free): KL(pi || pi_ref), top-action swap rate, |dV_root|.
Reading:
  * policy keeps moving as sweeps rise -> solver NOT converged (CFR is the ceiling)
  * net vs heuristic ~ identical -> leaf value doesn't matter at this depth
    (=> depth/actions/PBS are the ceiling, not the net)
  * rollout/oracle leaf moves policy a lot -> a STRONGER value net WOULD help
    (the value target/data is the ceiling)

  uv run python scripts/rebel_diag.py --n-pbs 20 --depth 3
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.run_eval import load_engine_data, play_game, resolve_deck  # noqa: E402
from src.agents import build_agent  # noqa: E402
from src.agents.base import Agent  # noqa: E402
from src.rebel.belief import OpponentBelief, build_worlds  # noqa: E402
from src.rebel.cfr import solve_mccfr_belief  # noqa: E402
from src.rebel.engine_game import BeliefSubgame, default_leaf_value  # noqa: E402

SINGLE_SELECT = 1
_MIN_CHOICE = 2


class _Recorder(Agent):
    """Wrap a scripted agent; snapshot obs at genuine searchable single-select nodes."""

    name = "recorder"

    def __init__(self, inner: Agent, store: list[dict], cap: int) -> None:
        super().__init__(inner.deck)
        self.inner = inner
        self.store = store
        self.cap = cap

    def reset(self, seed: int) -> None:
        self.inner.reset(seed)

    def act(self, obs: dict) -> list[int]:
        select = obs.get("select") or {}
        cur = obs.get("current")
        options = select.get("option") or []
        single = int(select.get("maxCount", 0)) == SINGLE_SELECT and cur is not None
        searchable = (single and len(options) >= _MIN_CHOICE
                      and bool(obs.get("search_begin_input")))
        if searchable and len(self.store) < self.cap:
            self.store.append(copy.deepcopy(obs))
        return self.inner.act(obs)


def collect_pbs(deck: list[int], engine: dict, n_target: int, seed: int) -> list[dict]:
    """Play greedy_plus vs greedy, snapshotting representative decision PBS."""
    store: list[dict] = []
    g = 0
    while len(store) < n_target and g < 200:
        rec = _Recorder(build_agent("greedy_plus", deck, engine), store, n_target)
        opp = build_agent("greedy", deck, engine)
        sf = g % 2 == 0
        p0, p1 = (rec, opp) if sf else (opp, rec)
        play_game(p0, p1, a_is_player0=sf, seed=4000 + seed + g)
        g += 1
    return store[:n_target]


def make_rollout_leaf(
    deck: list[int], engine: dict, cap: int = 40,
) -> Callable[[object], float]:
    """A leaf value = greedy_plus self-rollout from the leaf to terminal (p0 view).

    Safe: the engine branches immutably (search_step returns a child, parent untouched),
    verified in rebel_api_probe -- so rolling a leaf forward never corrupts cached tree
    nodes."""
    from cg.api import search_step  # noqa: PLC0415

    from src.search.ismcts import obs_to_dict  # noqa: PLC0415
    gp = build_agent("greedy_plus", deck, engine)

    def leaf(state: object) -> float:
        s = state
        for _ in range(cap):
            o = s.observation  # type: ignore[attr-defined]
            cur = o.current
            if cur is None or int(getattr(cur, "result", -1)) != -1:
                res = int(getattr(cur, "result", -1)) if cur is not None else -1
                return 1.0 if res == 0 else (-1.0 if res != -1 else 0.0)
            sel = o.select
            opts = (sel.option or []) if sel is not None else []
            if not opts:
                return 0.0
            try:
                move = gp.act(obs_to_dict(o))
            except Exception:  # noqa: BLE001
                move = [0]
            s = search_step(s.searchId, move)  # type: ignore[attr-defined]
        return default_leaf_value(s)  # cap hit -> prize heuristic

    return leaf


def solve_root(  # noqa: PLR0913 - the swap axes are genuinely separate inputs
        obs: dict, deck: list[int], belief: OpponentBelief, basics: list[int],
        leaf_value: Callable[[object], float], n_particles: int, depth: int,
        sweeps: int, tpw: int, seed: int, *,
        cfr_plus: bool = False,
        top_k: int | None = None) -> tuple[dict[int, float], float]:
    """Solve one PBS subgame; return (root policy over option idx, root value)."""
    cur = obs["current"]
    your = int(cur.get("yourIndex", 0))
    rng = np.random.default_rng(seed)
    worlds = build_worlds(cur, your, deck, belief, n_particles, rng, opp_basics=basics)
    game = BeliefSubgame(obs, worlds, max_depth=depth, leaf_value=leaf_value,
                         top_k=top_k)
    try:
        avg, root_value = solve_mccfr_belief(
            game, sweeps=sweeps, traversals_per_world=tpw,
            seed=int(rng.integers(1 << 30)), cfr_plus=cfr_plus)
        strat = avg.get(game.infoset_key((0,))) or {}
    finally:
        game.close()
    return {int(a): float(p) for a, p in strat.items()}, float(root_value)


def _kl(p: dict[int, float], q: dict[int, float]) -> float:
    """KL(p || q) over the union support (q floored to avoid /0)."""
    keys = set(p) | set(q)
    eps = 1e-9
    out = 0.0
    for k in keys:
        pk = p.get(k, 0.0)
        if pk <= 0:
            continue
        qk = max(q.get(k, 0.0), eps)
        out += pk * np.log(pk / qk)
    return float(out)


def _top(p: dict[int, float]) -> int:
    return max(p, key=p.get) if p else -1


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="metal_aggro")
    ap.add_argument("--net", default="data/rebel/vnd3_metal_r4.npz")
    ap.add_argument("--n-pbs", type=int, default=20)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--particles", type=int, default=4)
    ap.add_argument("--tpw", type=int, default=4)
    ap.add_argument("--sweeps-lo", type=int, default=4)
    ap.add_argument("--sweeps-mid", type=int, default=16)
    ap.add_argument("--sweeps-hi", type=int, default=64)
    ap.add_argument("--out", default="results/rebel_diag.json")
    args = ap.parse_args()

    engine = load_engine_data()
    deck = resolve_deck(args.deck)
    cards = engine.get("cards", {})
    basics = [c for c in deck if cards.get(c) and cards[c].get("basic")]
    belief = OpponentBelief.from_dirs()

    # value-net leaf
    from src.net.features import CardFeatures  # noqa: PLC0415
    from src.rebel.value_net import (  # noqa: PLC0415
        ValueNet,
        build_hypothesis_ctxs,
        make_leaf_value,
    )
    feats = CardFeatures(engine)
    decks = belief.hypotheses
    ctxs = build_hypothesis_ctxs(decks, feats)
    net = ValueNet.load(str(ROOT / args.net))
    leaf_net = make_leaf_value(net, feats, decks, ctxs)
    leaf_heur = default_leaf_value
    leaf_roll = make_rollout_leaf(deck, engine)

    print(f"collecting {args.n_pbs} representative PBS ...", flush=True)
    pbs = collect_pbs(deck, engine, args.n_pbs, seed=0)
    print(f"  got {len(pbs)} PBS", flush=True)

    # convergence to a CFR+ @ hi-sweeps reference: is CFR+ closer at LOW sweeps?
    cfr_lo, cfrp_lo, cfr_mid, cfrp_mid = [], [], [], []
    leaf_heur_kl, leaf_roll_kl = [], []
    leaf_heur_swap, leaf_roll_swap = 0, 0
    dv_heur, dv_roll = [], []
    t0 = time.perf_counter()

    def run(obs: dict, leaf: Callable[[object], float], sweeps: int, sd: int,
            *, cfr_plus: bool = False) -> tuple[dict[int, float], float]:
        return solve_root(obs, deck, belief, basics, leaf, args.particles,
                          args.depth, sweeps, args.tpw, sd, cfr_plus=cfr_plus)

    for i, obs in enumerate(pbs):
        sd = 100 + i
        # reference = the best-converged solve (CFR+ at hi sweeps)
        p_ref, v_ref = run(obs, leaf_net, args.sweeps_hi, sd, cfr_plus=True)
        # convergence: plain CFR vs CFR+ at lo and mid sweeps, distance to ref
        pc_lo, _ = run(obs, leaf_net, args.sweeps_lo, sd, cfr_plus=False)
        pp_lo, _ = run(obs, leaf_net, args.sweeps_lo, sd, cfr_plus=True)
        pc_mid, _ = run(obs, leaf_net, args.sweeps_mid, sd, cfr_plus=False)
        pp_mid, _ = run(obs, leaf_net, args.sweeps_mid, sd, cfr_plus=True)
        # leaf axis at the converged reference (CFR+ hi): heuristic + rollout vs net
        p_hr, v_hr = run(obs, leaf_heur, args.sweeps_hi, sd, cfr_plus=True)
        p_ro, v_ro = run(obs, leaf_roll, args.sweeps_hi, sd, cfr_plus=True)
        klc_lo, klp_lo = _kl(pc_lo, p_ref), _kl(pp_lo, p_ref)
        klc_mid, klp_mid = _kl(pc_mid, p_ref), _kl(pp_mid, p_ref)
        klh, klr = _kl(p_hr, p_ref), _kl(p_ro, p_ref)
        cfr_lo.append(klc_lo)
        cfrp_lo.append(klp_lo)
        cfr_mid.append(klc_mid)
        cfrp_mid.append(klp_mid)
        leaf_heur_kl.append(klh)
        leaf_roll_kl.append(klr)
        leaf_heur_swap += int(_top(p_hr) != _top(p_ref))
        leaf_roll_swap += int(_top(p_ro) != _top(p_ref))
        dv_heur.append(abs(v_hr - v_ref))
        dv_roll.append(abs(v_ro - v_ref))
        print(f"  pbs {i}: conv@lo CFR/CFR+={klc_lo:.3f}/{klp_lo:.3f} "
              f"@mid={klc_mid:.3f}/{klp_mid:.3f} "
              f"leafKL(heur/roll)={klh:.3f}/{klr:.3f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    n = len(pbs)
    def mean(x: list[float]) -> float:
        return float(np.mean(x)) if x else 0.0
    summary = {
        "n_pbs": n, "depth": args.depth,
        "sweeps": [args.sweeps_lo, args.sweeps_mid, args.sweeps_hi],
        "convergence_vs_cfrplus_hi_ref": {
            "KL_CFR_lo": mean(cfr_lo), "KL_CFRplus_lo": mean(cfrp_lo),
            "KL_CFR_mid": mean(cfr_mid), "KL_CFRplus_mid": mean(cfrp_mid),
        },
        "leaf_sensitivity": {
            "KL_heuristic_vs_net": mean(leaf_heur_kl),
            "KL_rollout_vs_net": mean(leaf_roll_kl),
            "topswap_heuristic": leaf_heur_swap / n,
            "topswap_rollout": leaf_roll_swap / n,
            "dV_heuristic": mean(dv_heur), "dV_rollout": mean(dv_roll),
        },
    }
    Path(ROOT / args.out).write_text(json.dumps(summary, indent=2))
    print("\n=== ReBeL diagnostic summary ===")
    print(json.dumps(summary, indent=2))
    print("\nREAD: KL_CFRplus_lo << KL_CFR_lo => CFR+ converges at LOW sweeps -> serve "
          "CFR+. leaf KL large => value net drives strategy (better targets help); "
          "net-vs-rollout gap => net is mis-calibrated.")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
