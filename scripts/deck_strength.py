"""Deck power under a FIXED robust pilot -> results/deck_strength.json.

play_diag fixes the deck and varies the pilot; this fixes the pilot (greedy, the
develop-then-attack reference -- and also the run7 net) and varies the DECK, scoring
each candidate decklist against the diverse type-deck panel. Answers "is the submitted
deck simply weaker than the metal deck greedy used on the ladder?", isolating deck
power from play.

Native/Docker (imports cg). Run:
  uv run python scripts/deck_strength.py --games 30
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

from scripts.run_eval import load_engine_data, play_game, read_deck  # noqa: E402
from src.agents import build_agent  # noqa: E402
from src.agents.recurrent_agent import RecurrentNetAgent  # noqa: E402
from src.deck import build_pool  # noqa: E402
from src.harness.stats import wilson_interval  # noqa: E402

# candidate decks to score (ladder context in the label)
CANDIDATES = [
    "metal_aggro", "grass_aggro", "run7_best", "run4_best", "single_prize_psychic",
]
# diverse opponent panel (greedy pilots each)
PANEL = [
    "metal_aggro", "grass_aggro", "fire_aggro", "water_aggro",
    "lightning_aggro", "psychic_aggro", "fighting_aggro", "darkness_aggro",
]
NET = "data/qdcoevo/run7/round_6/rl/paper_final.npz"

_G: dict = {}


def _init() -> None:
    from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: PLC0415

    _G["engine"] = load_engine_data()
    _G["pool"] = build_pool()
    _G["net"] = RecurrentPolicyValueNet.load(str(ROOT / NET))
    names = set(CANDIDATES) | set(PANEL)
    _G["decks"] = {nm: read_deck(ROOT / "decklists" / f"{nm}.csv") for nm in names}


def _pilot(kind: str, deck: list[int]) -> object:
    if kind == "net":
        return RecurrentNetAgent(
            deck, _G["engine"], net=_G["net"], cb_pool=_G["pool"],
            build_deck_from_net=False, temperature=0.0)
    return build_agent("greedy", deck, _G["engine"])


def _play(task: dict) -> dict:
    cand = _pilot(task["pilot"], _G["decks"][task["cand"]])
    opp = build_agent("greedy", _G["decks"][task["opp"]], _G["engine"])
    cand_first = task["cand_first"]
    p0, p1 = (cand, opp) if cand_first else (opp, cand)
    res = play_game(p0, p1, a_is_player0=cand_first, seed=task["seed"])
    return {"pilot": task["pilot"], "cand": task["cand"], "opp": task["opp"],
            "won": int(res.a_won), "dec": int(res.a_won or res.b_won)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Deck power under a fixed pilot")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=Path, default=ROOT / "results/deck_strength.json")
    args = ap.parse_args()

    tasks = [
        {"pilot": pilot, "cand": cand, "opp": opp,
         "cand_first": k % 2 == 0, "seed": (ci * 911 + oi) * 1000 + k}
        for pilot in ("greedy", "net")
        for ci, cand in enumerate(CANDIDATES)
        for oi, opp in enumerate(PANEL)
        for k in range(args.games)
    ]
    print(f"candidates={len(CANDIDATES)} panel={len(PANEL)} total={len(tasks)}")

    with Pool(args.workers, initializer=_init) as pp:
        rows = pp.map(_play, tasks)

    by: dict[tuple[str, str], list[dict]] = {}      # (pilot, cand) -> rows
    by_full: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        by.setdefault((r["pilot"], r["cand"]), []).append(r)
        by_full.setdefault((r["pilot"], r["cand"], r["opp"]), []).append(r)

    def wr(rs: list[dict]) -> dict:
        w, d = sum(x["won"] for x in rs), sum(x["dec"] for x in rs)
        p, lo, hi = wilson_interval(w, d)
        return {"winrate": round(p, 3), "ci": [round(lo, 3), round(hi, 3)], "n": d}

    out = {"panel": PANEL, "games_per_pair": args.games, "overall": {}, "per_opp": {}}
    for (pilot, cand), rs in by.items():
        out["overall"][f"{pilot}|{cand}"] = wr(rs)
    for (pilot, cand, opp), rs in by_full.items():
        out["per_opp"].setdefault(f"{pilot}|{cand}", {})[opp] = wr(rs)["winrate"]

    args.out.write_text(json.dumps(out, indent=2))
    print(f"-> {args.out}")
    for pilot in ("greedy", "net"):
        print(f"-- pilot={pilot} --")
        for cand in CANDIDATES:
            o = out["overall"][f"{pilot}|{cand}"]
            print(f"  {cand:<22} field-winrate={o['winrate']}  CI{o['ci']}  n={o['n']}")


if __name__ == "__main__":
    main()
