"""Win rate of one deck (piloted by a given net) vs the meta gauntlet.

Used for the specialist test: compare a hand-built single-prize deck piloted by the
generalist net vs the same deck piloted by a net trained specifically on it. Both the
candidate and each meta opponent are piloted by the SAME net (so it measures the deck
matchup under that pilot), matching the QD fitness convention.

Native/Docker (imports cg). Run:
  uv run python scripts/deck_vs_meta.py --net <net.npz> \
      --deck decklists/single_prize_psychic.csv --games 30
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

from scripts.run_eval import load_engine_data, play_game  # noqa: E402
from src.agents.recurrent_agent import RecurrentNetAgent  # noqa: E402
from src.deck import build_pool  # noqa: E402
from src.harness.stats import wilson_interval  # noqa: E402

_G: dict = {}


def _read(p: Path) -> list[int]:
    return [int(x) for x in p.read_text().split() if x.strip()]


def _init(net_path: str, deck: list[int], gauntlet: list[list[int]]) -> None:
    from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: PLC0415

    _G["engine"] = load_engine_data()
    _G["pool"] = build_pool()
    _G["net"] = RecurrentPolicyValueNet.load(net_path)
    _G["deck"] = deck
    _G["gauntlet"] = gauntlet


def _agent(deck: list[int]) -> RecurrentNetAgent:
    return RecurrentNetAgent(deck, _G["engine"], net=_G["net"], cb_pool=_G["pool"],
                             build_deck_from_net=False, temperature=0.0)


def _play(task: dict) -> dict:
    cand, opp = _agent(_G["deck"]), _agent(_G["gauntlet"][task["gi"]])
    cand_first = task["cand_first"]
    p0, p1 = (cand, opp) if cand_first else (opp, cand)
    res = play_game(p0, p1, a_is_player0=cand_first, seed=task["seed"])
    return {"gi": task["gi"], "won": int(res.a_won), "dec": int(res.a_won or res.b_won)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Deck vs meta gauntlet (net pilots both)")
    ap.add_argument("--net", type=Path, required=True)
    ap.add_argument("--deck", type=Path,
                    default=ROOT / "decklists/single_prize_psychic.csv")
    ap.add_argument("--games", type=int, default=30, help="games vs each meta deck")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=Path, default=ROOT / "results/deck_vs_meta.json")
    args = ap.parse_args()

    deck = _read(args.deck)
    gauntlet, names = [], []
    for p in sorted((ROOT / "decklists").glob("*.csv")):
        if p.resolve() == args.deck.resolve():
            continue  # don't play the candidate against itself
        gauntlet.append(_read(p))
        names.append(p.stem)

    tasks = [
        {"gi": gi, "cand_first": k % 2 == 0, "seed": gi * 1000 + k}
        for gi in range(len(gauntlet)) for k in range(args.games)
    ]
    with Pool(args.workers, initializer=_init,
              initargs=(str(args.net), deck, gauntlet)) as pp:
        rows = pp.map(_play, tasks)

    per: dict[str, list[dict]] = {}
    for r in rows:
        per.setdefault(names[r["gi"]], []).append(r)
    wins = sum(r["won"] for r in rows)
    dec = sum(r["dec"] for r in rows)
    p, lo, hi = wilson_interval(wins, dec)

    per_deck = {}
    for nm, rs in per.items():
        w, d = sum(r["won"] for r in rs), sum(r["dec"] for r in rs)
        per_deck[nm] = round(w / d, 3) if d else None
    out = {"net": str(args.net), "deck": str(args.deck),
           "overall_winrate": round(p, 3), "ci": [round(lo, 3), round(hi, 3)],
           "decisive": dec, "per_meta_deck": per_deck}
    args.out.write_text(json.dumps(out, indent=2))
    print(f"net={args.net.name}  deck={args.deck.stem}")
    print(f"  vs-meta overall = {p:.3f}  CI[{lo:.3f},{hi:.3f}]  (n={dec})")
    print(f"  per meta deck: {per_deck}")


if __name__ == "__main__":
    main()
