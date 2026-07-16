"""ReBeL LEARNING CURVE: does play improve as the value net trains on more games?

The goal (Koh): reach a regime where more training games -> better ReBeL play. This
evals a SEQUENCE of value-net checkpoints (the per-round nets from rebel_selfplay) as
the ReBeL leaf, each vs the SAME held-out pool on the SAME deck, and reports
``play_delta = winrate(rebel_r) - winrate(greedy_plus)`` per checkpoint. greedy_plus
(the baseline) is played ONCE and reused. A rising delta is the learning curve we want;
a flat/negative one says the training signal is not improving play (fix targets /
opponents / capacity, not just add games).

  uv run python scripts/rebel_curve.py --deck metal_aggro \
      --nets data/rebel/vn2_metal_r0.npz,...,_r5.npz --games 8
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


def _init(deck_name: str, opp_names: list[str], nets: list[str], params: dict) -> None:
    _G["recurrent"] = params.pop("recurrent", False)
    _G["vector"] = params.pop("vector", False)
    if _G["vector"]:
        from src.rebel.vector_value_net import VectorValueNet  # noqa: PLC0415
        _G["nets"] = {p: VectorValueNet.load(p) for p in nets}
    elif _G["recurrent"]:
        from src.rebel.recurrent_value_net import RecurrentValueNet  # noqa: PLC0415
        _G["nets"] = {p: RecurrentValueNet.load(p) for p in nets}
    else:
        from src.rebel.value_net import ValueNet  # noqa: PLC0415
        _G["nets"] = {p: ValueNet.load(p) for p in nets}
    _G["engine"] = load_engine_data()
    _G["deck"] = resolve_deck(deck_name)
    _G["params"] = params
    _G["opp"] = {nm: read_deck(ROOT / "decklists" / "heldout2" / f"{nm}.csv")
                 for nm in opp_names}


def _subject(kind: str) -> object:
    if kind == "greedy_plus":
        return build_agent("greedy_plus", _G["deck"], _G["engine"])
    from src.rebel.agent import RebelAgent  # noqa: PLC0415
    if _G.get("vector"):
        net_kw = {"vector_net": _G["nets"][kind]}
    elif _G.get("recurrent"):
        net_kw = {"recurrent_net": _G["nets"][kind]}
    else:
        net_kw = {"value_net": _G["nets"][kind]}
    return RebelAgent(_G["deck"], _G["engine"], **net_kw, **_G["params"])


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
    return {"kind": task["kind"], "won": int(res.a_won),
            "dec": int(res.a_won or res.b_won)}


def _wr(rows: list[dict], kind: str) -> dict:
    rs = [r for r in rows if r["kind"] == kind]
    w, d = sum(x["won"] for x in rs), sum(x["dec"] for x in rs)
    p, lo, hi = wilson_interval(w, d)
    return {"winrate": round(p, 3), "ci": [round(lo, 3), round(hi, 3)], "n": d}


def main() -> None:
    ap = argparse.ArgumentParser(description="ReBeL value-net learning curve")
    ap.add_argument("--deck", default="metal_aggro")
    ap.add_argument("--nets", required=True,
                    help="comma list of ValueNet .npz checkpoints")
    ap.add_argument("--opp-decks", default="")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--particles", type=int, default=8)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--sweeps", type=int, default=4)
    ap.add_argument("--tpw", type=int, default=4)
    ap.add_argument("--exploit", action="store_true",
                    help="best-respond to greedy_plus in-search (BR-to-gp)")
    ap.add_argument("--horizon-turns", type=int, default=None,
                    help="turn-boundary horizon (public-event depth)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="growing-tree width bound: expand only top_k options per node")
    ap.add_argument("--recurrent", action="store_true",
                    help="serve a history-aware LSTM value net (RecurrentValueNet)")
    ap.add_argument("--vector", action="store_true",
                    help="serve a proper-ReBeL infostate VECTOR net (VectorValueNet)")
    ap.add_argument("--out", type=Path, default=ROOT / "results/rebel_curve.json")
    args = ap.parse_args()

    nets = [p for p in args.nets.split(",") if p]
    opp_names = ([d for d in args.opp_decks.split(",") if d] or
                 sorted(p.stem
                        for p in (ROOT / "decklists" / "heldout2").glob("*.csv")))
    params = {"n_particles": args.particles, "max_depth": args.depth,
              "sweeps": args.sweeps, "traversals_per_world": args.tpw,
              "exploit": args.exploit, "horizon_turns": args.horizon_turns,
              "top_k": args.top_k, "recurrent": args.recurrent,
              "vector": args.vector}
    kinds = ["greedy_plus", *nets]
    tasks = [
        {"kind": kind, "opp": opp, "opp_pilot": op, "sf": k % 2 == 0,
         "seed": (oi * 131 + pi) * 1000 + k}
        for kind in kinds
        for pi, op in enumerate(OPP_PILOTS)
        for oi, opp in enumerate(opp_names)
        for k in range(args.games)
    ]
    print(f"deck={args.deck} checkpoints={len(nets)} opp_decks={len(opp_names)} "
          f"total={len(tasks)}", flush=True)

    with Pool(args.workers, initializer=_init,
              initargs=(args.deck, opp_names, nets, params)) as pp:
        rows = pp.map(_play, tasks)

    gp = _wr(rows, "greedy_plus")
    curve = []
    for p in nets:
        reb = _wr(rows, p)
        curve.append({"net": Path(p).stem, "rebel_wr": reb["winrate"],
                      "ci": reb["ci"],
                      "play_delta": round(reb["winrate"] - gp["winrate"], 3)})
    out = {"deck": args.deck, "greedy_plus": gp, "curve": curve}
    args.out.write_text(json.dumps(out, indent=2))

    print(f"\n=== ReBeL learning curve on {args.deck} "
          f"(greedy_plus={gp['winrate']:.3f}) ===")
    for c in curve:
        print(f"  {c['net']:<18} rebel={c['rebel_wr']:.3f} CI{c['ci']} "
              f"Δ={c['play_delta']:+.3f}")
    deltas = [c["play_delta"] for c in curve]
    rising = len(deltas) >= 2 and deltas[-1] > deltas[0]
    trend = "RISING (more games -> better)" if rising else "flat/negative"
    print(f"trend: {trend}; best Δ={max(deltas):+.3f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
