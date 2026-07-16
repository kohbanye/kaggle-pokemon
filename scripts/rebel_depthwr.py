"""Does DEEPER search (with the top_k width bound) beat greedy_plus? -- win-rate sweep.

`rebel_depthdiag` showed that under CFR+ with a ``top_k`` beam the converged root policy
diverges a LOT as depth grows (KL 2.8->5.9 d4..d6), and it stays servable (d4 0.4s, d5
1.0s/solve). Policy CHANGE is necessary but not sufficient -- this script is the judge:
it plays ReBeL at each depth (SAME leaf, same beam) AND greedy_plus vs the SAME held-out
opponents with slot-swaps, reporting ``winrate(rebel@depth) - winrate(gp)`` per depth.
The gp baseline is played ONCE and reused. A depth whose delta turns positive (CI above
0) is the deeper-search lever the plan was after; flat/negative across depths says the
depth-3 tie is the true ceiling and DECK(QD) stays the only Elo lever.

  uv run python scripts/rebel_depthwr.py --deck metal_aggro --depths 3,4,5 --top-k 3 \
      --net data/rebel/vnd3_metal_r4.npz --games 6
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


def _init(deck_name: str, opp_names: list[str],
          net_path: str | None, params: dict) -> None:
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
    depth = int(kind.split("d")[1])  # kind = "rebel_d4"
    p = dict(_G["params"])
    p["max_depth"] = depth
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
    ap = argparse.ArgumentParser(description="ReBeL depth win-rate sweep")
    ap.add_argument("--deck", default="metal_aggro")
    ap.add_argument("--depths", default="3,4,5")
    ap.add_argument("--net", default="data/rebel/vnd3_metal_r4.npz",
                    help="leaf value net (fixed across depths); empty => heuristic")
    ap.add_argument("--opp-decks", default="")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--particles", type=int, default=4)
    ap.add_argument("--sweeps", type=int, default=16)
    ap.add_argument("--tpw", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "results/rebel_depthwr.json")
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",") if d]
    net_path = args.net if args.net and Path(ROOT / args.net).exists() else None
    opp_names = ([d for d in args.opp_decks.split(",") if d] or
                 sorted(p.stem
                        for p in (ROOT / "decklists" / "heldout2").glob("*.csv")))
    params = {"n_particles": args.particles, "sweeps": args.sweeps,
              "traversals_per_world": args.tpw, "top_k": args.top_k, "cfr_plus": True}
    kinds = ["greedy_plus", *(f"rebel_d{d}" for d in depths)]
    tasks = [
        {"kind": kind, "opp": opp, "opp_pilot": op, "sf": k % 2 == 0,
         "seed": (oi * 131 + pi) * 1000 + k}
        for kind in kinds
        for pi, op in enumerate(OPP_PILOTS)
        for oi, opp in enumerate(opp_names)
        for k in range(args.games)
    ]
    print(f"deck={args.deck} depths={depths} leaf={'net' if net_path else 'heuristic'} "
          f"opp_decks={len(opp_names)} total={len(tasks)}", flush=True)

    leaf_arg = str(ROOT / net_path) if net_path else None
    with Pool(args.workers, initializer=_init,
              initargs=(args.deck, opp_names, leaf_arg, params)) as pp:
        rows = pp.map(_play, tasks)

    gp = _wr(rows, "greedy_plus")
    curve = []
    for d in depths:
        reb = _wr(rows, f"rebel_d{d}")
        # per-deck delta vs gp -- to tell a CONSISTENT deck-class effect from noise
        per_deck = {opp: round(_wr(rows, f"rebel_d{d}", opp)["winrate"]
                               - _wr(rows, "greedy_plus", opp)["winrate"], 3)
                    for opp in opp_names}
        curve.append({"depth": d, "rebel_wr": reb["winrate"], "ci": reb["ci"],
                      "play_delta": round(reb["winrate"] - gp["winrate"], 3),
                      "per_deck_delta": per_deck})
    out = {"deck": args.deck, "top_k": args.top_k, "greedy_plus": gp, "curve": curve}
    args.out.write_text(json.dumps(out, indent=2))

    print(f"\n=== ReBeL depth win-rate on {args.deck} "
          f"(greedy_plus={gp['winrate']:.3f}, top_k={args.top_k}) ===")
    for c in curve:
        print(f"  depth {c['depth']}  rebel={c['rebel_wr']:.3f} CI{c['ci']} "
              f"Δ={c['play_delta']:+.3f}")
        pd = c["per_deck_delta"]
        signs = "".join("+" if v > 0.02 else "-" if v < -0.02 else "0"
                        for v in pd.values())
        print(f"      per-deck Δ vs gp: {signs}  "
              f"({', '.join(f'{k[:8]}:{v:+.2f}' for k, v in pd.items())})")
    deltas = [c["play_delta"] for c in curve]
    best = max(deltas)
    verdict = "DEEPER HELPS" if best > 0.02 else "flat/negative (depth-3 ceiling)"
    print(f"best Δ={best:+.3f} at depth {depths[deltas.index(best)]}; {verdict}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
