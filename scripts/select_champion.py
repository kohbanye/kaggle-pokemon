"""Select a run's champion deck by HELD-OUT screening, not archive fitness.

The qd7_r5 lesson: archive fitness is gauntlet-relative, so the last round's
top-fitness cell can be a mediocre external deck while the true champion sits
in an earlier round (run3: round 5's best scored +18pp held-out over round 7's).
This script makes the correct selection procedure a one-liner: gather the
top-k cells of EVERY round's archive (deduped), screen them all against a
held-out pool (greedy pilot, slot-swapped), and rank by external win rate.

  uv run python scripts/select_champion.py --run-dir data/qdrl_run3 \
      --pool heldout2 --games 8 --write-best decklists/candidates/foo.csv
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
from src.harness.stats import wilson_interval  # noqa: E402

_G: dict = {}


def _init(decks: dict[str, list[int]], opponents: dict[str, list[int]]) -> None:
    _G["engine"] = load_engine_data()
    _G["decks"] = decks
    _G["opps"] = opponents


def _play(task: dict) -> tuple[str, int, int]:
    subj = build_agent("greedy", _G["decks"][task["cand"]], _G["engine"])
    opp = build_agent("greedy", _G["opps"][task["opp"]], _G["engine"])
    first = task["k"] % 2 == 0
    p0, p1 = (subj, opp) if first else (opp, subj)
    res = play_game(p0, p1, a_is_player0=first, seed=task["seed"])
    return task["cand"], int(res.a_won), int(res.a_won or res.b_won)


def gather_candidates(run_dir: Path, top_k: int) -> dict[str, list[int]]:
    """Top-k cells of every round archive under ``run_dir``, deduped."""
    cands: dict[str, list[int]] = {}
    seen: set[tuple[int, ...]] = set()
    for arc_path in sorted(run_dir.glob("round_*/qd_archive.json")):
        rnd = arc_path.parent.name
        arc = json.loads(arc_path.read_text())
        cells = sorted(arc["cells"], key=lambda c: -c["fitness"])[:top_k]
        for c in cells:
            fp = tuple(sorted(c["deck"]))
            if fp in seen:
                continue
            seen.add(fp)
            cands[f"{rnd}:{tuple(c['descriptor'])}:f{c['fitness']:.2f}"] = c["deck"]
    return cands


def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out champion selection")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="qdrl run directory containing round_*/qd_archive.json")
    ap.add_argument("--top-k", type=int, default=3, help="cells per round")
    ap.add_argument("--pool", type=str, default="heldout2")
    ap.add_argument("--games", type=int, default=8, help="games per opponent")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--write-best", type=Path, default=None,
                    help="write the winning decklist to this CSV")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cands = gather_candidates(args.run_dir, args.top_k)
    opps = {p.stem: read_deck(p)
            for p in sorted((ROOT / "decklists" / args.pool).glob("*.csv"))}
    print(f"screening {len(cands)} candidate decks x {len(opps)} opponents "
          f"x {args.games} games")

    tasks = [{"cand": nm, "opp": op, "k": k, "seed": (ci * 991 + oi) * 100 + k}
             for ci, nm in enumerate(cands)
             for oi, op in enumerate(opps)
             for k in range(args.games)]
    with Pool(args.workers, initializer=_init, initargs=(cands, opps)) as pp:
        rows = pp.map(_play, tasks)

    agg: dict[str, list[int]] = {nm: [0, 0] for nm in cands}
    for nm, won, dec in rows:
        agg[nm][0] += won
        agg[nm][1] += dec
    ranked = []
    for nm in cands:
        w, n = agg[nm]
        p, lo, hi = wilson_interval(w, n)
        ranked.append({"cand": nm, "winrate": round(p, 3),
                       "ci": [round(lo, 3), round(hi, 3)], "n": n})
    ranked.sort(key=lambda r: -r["winrate"])
    for r in ranked:
        print(f"{r['winrate']:.3f} {r['ci']} {r['cand']}")
    if args.out:
        args.out.write_text(json.dumps(
            {"pool": args.pool, "ranked": ranked}, indent=1))
    if args.write_best and ranked:
        best = cands[ranked[0]["cand"]]
        args.write_best.write_text("\n".join(map(str, best)) + "\n")
        print(f"best -> {args.write_best}")


if __name__ == "__main__":
    main()
