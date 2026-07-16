"""Does ReBeL beat greedy_plus on the SAME deck? -> results/rebel_eval.json.

The decisive test of the ReBeL pivot: on a deck (default metal_aggro, the aggro deck
where the AZ play net was most tempo-biased and LOST), score `RebelAgent` (with a
trained ValueNet leaf) AND `greedy_plus` vs the SAME held-out opponent pool, and
compare. A positive,
CI-separated `play_delta = winrate(rebel) - winrate(greedy_plus)` means ReBeL's play
broke the tempo-bias fixed point that AZ never could. Mirrors az_eval's decomposition
(same harness / Wilson CI), with the `rebel` pilot instead of the recurrent net.

RebelAgent is slow (~seconds/solve), so keep the pool + games modest.

  uv run python scripts/rebel_eval.py --net data/rebel/vn_metal_r3.npz \
      --deck metal_aggro --opp-decks heldout2 --games 8
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


def _init(deck_name: str, opp_names: list[str], net_path: str, params: dict) -> None:
    from src.rebel.value_net import ValueNet  # noqa: PLC0415

    _G["engine"] = load_engine_data()
    _G["deck"] = resolve_deck(deck_name)
    _G["net"] = ValueNet.load(net_path)
    _G["params"] = params
    _G["opp"] = {nm: read_deck(ROOT / "decklists" / "heldout2" / f"{nm}.csv")
                 for nm in opp_names}


def _subject(kind: str) -> object:
    if kind == "rebel":
        from src.rebel.agent import RebelAgent  # noqa: PLC0415
        return RebelAgent(_G["deck"], _G["engine"], value_net=_G["net"], **_G["params"])
    return build_agent("greedy_plus", _G["deck"], _G["engine"])


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


def main() -> None:
    ap = argparse.ArgumentParser(description="ReBeL vs greedy_plus, same deck")
    ap.add_argument("--net", type=Path, required=True, help="trained ValueNet .npz")
    ap.add_argument("--deck", default="metal_aggro")
    ap.add_argument("--opp-decks", default="",
                    help="comma heldout2 stems; empty = all of decklists/heldout2")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--particles", type=int, default=4)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--sweeps", type=int, default=4)
    ap.add_argument("--tpw", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "results/rebel_eval.json")
    args = ap.parse_args()

    opp_names = ([d for d in args.opp_decks.split(",") if d] or
                 sorted(p.stem
                        for p in (ROOT / "decklists" / "heldout2").glob("*.csv")))
    params = {"n_particles": args.particles, "max_depth": args.depth,
              "sweeps": args.sweeps, "traversals_per_world": args.tpw}
    tasks = [
        {"kind": kind, "opp": opp, "opp_pilot": op, "sf": k % 2 == 0,
         "seed": (oi * 131 + pi) * 1000 + k}
        for kind in ("rebel", "greedy_plus")
        for pi, op in enumerate(OPP_PILOTS)
        for oi, opp in enumerate(opp_names)
        for k in range(args.games)
    ]
    print(f"deck={args.deck} opp_decks={len(opp_names)} pilots={len(OPP_PILOTS)} "
          f"total={len(tasks)} (rebel is slow)", flush=True)

    with Pool(args.workers, initializer=_init,
              initargs=(args.deck, opp_names, str(args.net), params)) as pp:
        rows = pp.map(_play, tasks)

    def wr(kind: str) -> dict:
        rs = [r for r in rows if r["kind"] == kind]
        w, d = sum(x["won"] for x in rs), sum(x["dec"] for x in rs)
        p, lo, hi = wilson_interval(w, d)
        return {"winrate": round(p, 3), "ci": [round(lo, 3), round(hi, 3)], "n": d}

    reb, gp = wr("rebel"), wr("greedy_plus")
    delta = round(reb["winrate"] - gp["winrate"], 3)
    sig = ("+" if reb["ci"][0] > gp["ci"][1]
           else "-" if reb["ci"][1] < gp["ci"][0] else "~")
    out = {"deck": args.deck, "net": str(args.net), "n_opp_decks": len(opp_names),
           "rebel": reb, "greedy_plus": gp, "play_delta": delta, "sig": sig}
    args.out.write_text(json.dumps(out, indent=2))

    print(f"\n=== ReBeL vs greedy_plus on {args.deck} (heldout2) ===")
    print(f"  rebel        {reb['winrate']:.3f} CI{reb['ci']} n={reb['n']}")
    print(f"  greedy_plus  {gp['winrate']:.3f} CI{gp['ci']} n={gp['n']}")
    verdict = ("ReBeL BEATS greedy_plus" if sig == "+" else
               "greedy_plus beats ReBeL" if sig == "-" else "no CI-separated diff")
    print(f"  play_delta = {delta:+.3f} ({sig})  ->  {verdict}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
