"""Is the ISMCTS TEACHER stronger than greedy_plus? -> results/teacher_strength.json.

The AZ loop distils the ISMCTS search policy (teacher) into the net (student). If the
student never beats greedy_plus, either (a) the TEACHER is already <= greedy_plus (a
target-strength problem -- no amount of distillation helps), or (b) the teacher beats
greedy_plus but the student regresses to it (a distillation / capacity / self-play
problem). This script discriminates them by scoring THREE subjects on the SAME decks vs
the held-out real-ladder pool:

    net       -- the raw recurrent net (the student)
    ismcts    -- the same net + SO-ISMCTS search at serve (the teacher)
    greedy_plus

Read per deck: if ismcts ~ greedy_plus on aggro decks (where net loses) -> search FIXES
the tempo bias -> the failure is DISTILLATION (the LSTM can't capture the teacher). If
ismcts ~ net < greedy_plus -> the SEARCH ITSELF is tempo-biased -> the failure is the
TARGET. Mirrors heldout_eval's harness/CI.

  uv run python scripts/teacher_strength.py --net data/coevo_az_v2/gen9/net.npz \
      --decks metal_aggro,grass_aggro,qd7_r5 --games 10
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
    deck_path,
    load_engine_data,
    play_game,
    read_deck,
)
from src.agents import build_agent  # noqa: E402
from src.harness.stats import wilson_interval  # noqa: E402

OPP_PILOTS = ("greedy", "heuristic", "greedyFF")
SUBJECTS = ("net", "ismcts", "greedy_plus")
_G: dict = {}


def _init(decks: list[str], heldout: list[str], net_path: str, iters: int) -> None:
    from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: PLC0415

    _G["engine"] = load_engine_data()
    _G["net"] = RecurrentPolicyValueNet.load(net_path)
    _G["iters"] = iters
    _G["decks"] = {nm: read_deck(deck_path(nm)) for nm in decks}
    _G["decks"].update({nm: read_deck(ROOT / "decklists" / "heldout2" / f"{nm}.csv")
                        for nm in heldout})
    cards = _G["engine"]["cards"]
    _G["basics"] = {nm: [c for c in d if cards.get(c) and cards[c].get("basic")]
                    for nm, d in _G["decks"].items()}
    meta = [read_deck(p) for p in sorted((ROOT / "decklists").glob("*.csv"))]
    _G["meta_prior"] = [c for dd in meta for c in dd]
    _G["hyp"] = meta + [read_deck(p)
                        for p in sorted((ROOT / "decklists" / "anchors").glob("*.csv"))]


def _subject(kind: str, deck_name: str) -> object:
    deck = _G["decks"][deck_name]
    if kind == "net":
        from src.agents.recurrent_agent import RecurrentNetAgent  # noqa: PLC0415
        return RecurrentNetAgent(deck, _G["engine"], net=_G["net"], cb_pool=None,
                                 build_deck_from_net=False, temperature=0.0)
    if kind == "ismcts":  # the TEACHER: net + SO-ISMCTS at serve (gate on, not self-play)  # noqa: E501
        from src.search.ismcts import RecurrentIsmctsAgent  # noqa: PLC0415
        return RecurrentIsmctsAgent(
            deck, _G["engine"], _G["net"], opp_prior=_G["meta_prior"],
            opp_basics=_G["basics"][deck_name], opp_decks=_G["hyp"],
            iterations=_G["iters"], rollout_cap=0, move_budget_s=999.0,
            gate_margin=0.10, self_play=False, record=False,
            cb_pool=None, build_deck_from_net=False, temperature=0.0)
    return build_agent("greedy_plus", deck, _G["engine"])


def _opponent(deck_name: str, pilot: str) -> object:
    deck = _G["decks"][deck_name]
    if pilot == "greedyFF":
        return _ForcedFirst(build_agent("greedy", deck, _G["engine"]))
    return build_agent(pilot, deck, _G["engine"])


def _play(task: dict) -> dict:
    subj = _subject(task["kind"], task["deck"])
    opp = _opponent(task["opp"], task["opp_pilot"])
    sf = task["subj_first"]
    p0, p1 = (subj, opp) if sf else (opp, subj)
    res = play_game(p0, p1, a_is_player0=sf, seed=task["seed"])
    return {"kind": task["kind"], "deck": task["deck"],
            "won": int(res.a_won), "dec": int(res.a_won or res.b_won)}


def main() -> None:
    ap = argparse.ArgumentParser(description="ISMCTS teacher vs greedy_plus strength")
    ap.add_argument("--net", type=Path, default=ROOT / "data/coevo_az_v2/gen9/net.npz")
    ap.add_argument("--decks", default="metal_aggro,grass_aggro,qd7_r5")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--iters", type=int, default=96)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=Path, default=ROOT / "results/teacher_strength.json")
    args = ap.parse_args()

    decks = [d for d in args.decks.split(",") if d]
    heldout = sorted(p.stem for p in (ROOT / "decklists" / "heldout2").glob("*.csv"))
    tasks = [
        {"kind": kind, "deck": deck, "opp": opp, "opp_pilot": op,
         "subj_first": k % 2 == 0, "seed": (si * 911 + oi) * 1000 + pi * 137 + k}
        for si, deck in enumerate(decks)
        for kind in SUBJECTS
        for pi, op in enumerate(OPP_PILOTS)
        for oi, opp in enumerate(heldout)
        for k in range(args.games)
    ]
    print(f"decks={decks} subjects={SUBJECTS} heldout={len(heldout)} "
          f"total={len(tasks)} (ismcts iters={args.iters})", flush=True)

    with Pool(args.workers, initializer=_init,
              initargs=(decks, heldout, str(args.net), args.iters)) as pp:
        rows = pp.map(_play, tasks)

    agg: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        agg.setdefault((r["deck"], r["kind"]), []).append(r)

    def wr(rs: list[dict]) -> dict:
        w, d = sum(x["won"] for x in rs), sum(x["dec"] for x in rs)
        p, lo, hi = wilson_interval(w, d)
        return {"winrate": round(p, 3), "ci": [round(lo, 3), round(hi, 3)], "n": d}

    out: dict = {"net": str(args.net), "decks": decks, "per_deck": {}}
    for deck in decks:
        out["per_deck"][deck] = {k: wr(agg[(deck, k)]) for k in SUBJECTS}
    args.out.write_text(json.dumps(out, indent=2))

    print("\n=== Teacher (ismcts) vs student (net) vs greedy_plus, heldout2 ===")
    print(f"{'deck':<14}{'net':>10}{'ismcts':>10}{'greedy_plus':>13}   read")
    for deck in decks:
        d = out["per_deck"][deck]
        n, i = d["net"]["winrate"], d["ismcts"]["winrate"]
        g = d["greedy_plus"]["winrate"]
        read = ("search FIXES (distill fails)" if i >= g - 0.02 and n < g - 0.02 else
                "search too weak (TARGET biased)" if i < g - 0.02 else "n/a")
        print(f"{deck:<14}{n:>10.3f}{i:>10.3f}{g:>13.3f}   {read}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
