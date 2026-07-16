"""What does ReBeL LOSE to? -- per-opponent-archetype breakdown vs the ladder-like pool.

Before spending a submission slot, characterise ReBeL's failure modes locally where we
control the opponents. Plays three subjects on the SAME deck+leaf -- greedy_plus (the
baseline pilot), equilibrium ReBeL (Nash) and best-response ReBeL (opp fixed to
greedy_plus) -- vs the heldout2 pool (real-ladder-style decks) piloted by
{greedy, heuristic, greedyFF}, slot-swapped. Reports per-OPPONENT-DECK win-rate for each
subject, so we see which archetypes beat ReBeL and whether gp handles them (=ReBeL-
specific losses). Picks the stronger variant (eq vs br) for the submission.

  uv run python scripts/rebel_loss_analysis.py --deck metal_aggro \
      --net data/rebel/vnd3_metal_r4.npz --depth 3 --top-k 3 --games 8
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.heldout_eval import _ForcedFirst  # noqa: E402
from scripts.run_eval import (  # noqa: E402
    load_engine_data,
    play_game,
    read_deck,
    resolve_deck,
)
from src.agents import build_agent  # noqa: E402
from src.harness.stats import wilson_interval  # noqa: E402

OPP_PILOTS = ("greedy", "heuristic", "greedyFF")
_G: dict = {}


def _init(deck_name: str, opp_names: list[str], net_path: str | None,
          params: dict) -> None:
    _G["engine"] = load_engine_data()
    _G["deck"] = resolve_deck(deck_name)
    _G["params"] = params
    _G["net"] = None
    if net_path:
        from src.rebel.value_net import ValueNet  # noqa: PLC0415
        _G["net"] = ValueNet.load(net_path)
    _G["opp"] = {nm: read_deck(ROOT / "decklists" / "heldout2" / f"{nm}.csv")
                 for nm in opp_names}


def _subject(kind: str) -> object:
    if kind == "greedy_plus":
        return build_agent("greedy_plus", _G["deck"], _G["engine"])
    from src.rebel.agent import RebelAgent  # noqa: PLC0415
    p = dict(_G["params"])
    p["exploit"] = kind == "br"
    return RebelAgent(_G["deck"], _G["engine"], value_net=_G["net"], **p)


def _opponent(opp_deck: str, pilot: str) -> object:
    deck = _G["opp"][opp_deck]
    if pilot == "greedyFF":
        return _ForcedFirst(build_agent("greedy", deck, _G["engine"]))
    return build_agent(pilot, deck, _G["engine"])


def _play(task: dict) -> dict:
    subj = _subject(task["kind"])
    opp = _opponent(task["opp"], task["opp_pilot"])
    sf = task["sf"]
    p0, p1 = (subj, opp) if sf else (opp, subj)
    res = play_game(p0, p1, a_is_player0=sf, seed=task["seed"])
    return {"kind": task["kind"], "opp": task["opp"], "won": int(res.a_won),
            "dec": int(res.a_won or res.b_won)}


def _wr(rows: list[dict], kind: str, opp: str | None = None) -> dict:
    rs = [r for r in rows if r["kind"] == kind and (opp is None or r["opp"] == opp)]
    w, d = sum(x["won"] for x in rs), sum(x["dec"] for x in rs)
    p, lo, hi = wilson_interval(w, d)
    return {"winrate": round(p, 3), "ci": [round(lo, 3), round(hi, 3)], "n": d}


def main() -> None:
    ap = argparse.ArgumentParser(description="ReBeL loss-mode analysis vs heldout pool")
    ap.add_argument("--deck", default="metal_aggro")
    ap.add_argument("--net", default="data/rebel/vnd3_metal_r4.npz")
    ap.add_argument("--opp-decks", default="")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--particles", type=int, default=4)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--sweeps", type=int, default=16)
    ap.add_argument("--tpw", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/rebel_loss_analysis.json")
    args = ap.parse_args()

    net_path = args.net if args.net and Path(ROOT / args.net).exists() else None
    opp_names = ([d for d in args.opp_decks.split(",") if d] or
                 sorted(p.stem
                        for p in (ROOT / "decklists" / "heldout2").glob("*.csv")))
    params = {"n_particles": args.particles, "max_depth": args.depth,
              "sweeps": args.sweeps, "traversals_per_world": args.tpw,
              "top_k": args.top_k, "cfr_plus": True, "exploit_opp": "greedy_plus"}
    kinds = ["greedy_plus", "eq", "br"]
    tasks = [
        {"kind": kind, "opp": opp, "opp_pilot": op, "sf": k % 2 == 0,
         "seed": (oi * 131 + pi) * 1000 + k}
        for kind in kinds
        for pi, op in enumerate(OPP_PILOTS)
        for oi, opp in enumerate(opp_names)
        for k in range(args.games)
    ]
    print(f"deck={args.deck} leaf={'net' if net_path else 'heuristic'} "
          f"opp_decks={len(opp_names)} kinds={kinds} total={len(tasks)}", flush=True)

    leaf = str(ROOT / net_path) if net_path else None
    with Pool(args.workers, initializer=_init,
              initargs=(args.deck, opp_names, leaf, params)) as pp:
        rows = pp.map(_play, tasks)

    agg = {k: _wr(rows, k) for k in kinds}
    per_deck = {opp: {k: _wr(rows, k, opp)["winrate"] for k in kinds}
                for opp in opp_names}
    out = {"deck": args.deck, "depth": args.depth, "top_k": args.top_k,
           "aggregate": agg, "per_deck": per_deck}
    args.out.write_text(json.dumps(out, indent=2))

    print(f"\n=== ReBeL loss analysis on {args.deck} (leaf=net, "
          f"depth={args.depth}, top_k={args.top_k}) ===")
    print(f"  aggregate:  gp={agg['greedy_plus']['winrate']:.3f}  "
          f"eq={agg['eq']['winrate']:.3f}  br={agg['br']['winrate']:.3f}")
    print("  per-opp-deck winrate (gp/eq/br)  [* = ReBeL loses where gp wins]:")
    for opp in opp_names:
        d = per_deck[opp]
        flag = " *" if (d["greedy_plus"] >= 0.5 > min(d["eq"], d["br"])) else ""
        print(f"    {opp[:22]:<22} gp={d['greedy_plus']:.2f}  eq={d['eq']:.2f}  "
              f"br={d['br']:.2f}{flag}")
    best = "br" if agg["br"]["winrate"] >= agg["eq"]["winrate"] else "eq"
    print(f"\nstronger variant vs this pool: {best.upper()} "
          f"(eq={agg['eq']['winrate']:.3f} vs br={agg['br']['winrate']:.3f})")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
