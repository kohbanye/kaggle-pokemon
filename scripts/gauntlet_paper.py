"""Rank paper-OSFP checkpoints by win rate vs greedy (native, Docker-free here).

Loads the engine + CB pool ONCE, then for each checkpoint builds a NetAgent
(``--cb`` deck from its own CB head) and plays N slot-swapped games vs greedy.
Prints a sorted table and writes JSON so we can pick the strongest ckpt to submit.

  .venv/bin/python scripts/gauntlet_paper.py --dir data/paperosfp/main \
      --games 120 --seed 0 --top 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.run_eval import (  # noqa: E402
    DEFAULT_DECK,
    load_engine_data,
    play_game,
    read_deck,
)
from src.agents import build_agent  # noqa: E402
from src.agents.net_agent import NetAgent  # noqa: E402
from src.deck import build_pool  # noqa: E402
from src.harness.stats import summarize  # noqa: E402

_ITER = re.compile(r"paperiter_(\d+)\.npz$")


def _iter_no(p: Path) -> int:
    m = _ITER.search(p.name)
    return int(m.group(1)) if m else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "data/paperosfp/main")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opp", default="greedy")
    ap.add_argument("--iters", default=None, help="comma list of iters; default all")
    ap.add_argument("--out", type=Path, default=ROOT / "results/gauntlet_paper.json")
    args = ap.parse_args()

    ckpts = sorted(args.dir.glob("paperiter_*.npz"), key=_iter_no)
    if args.iters:
        want = {int(x) for x in args.iters.split(",")}
        ckpts = [c for c in ckpts if _iter_no(c) in want]
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.dir}")

    engine = load_engine_data()
    pool = build_pool()
    deck = read_deck(DEFAULT_DECK)
    opp = build_agent(args.opp, deck, engine)

    print(f"== gauntlet: {len(ckpts)} ckpts vs {args.opp}, "
          f"{args.games} games each, --cb ==")
    rows = []
    t0 = time.perf_counter()
    for c in ckpts:
        net = NetAgent(deck, engine, weights=c, cb_pool=pool)
        results = []
        for g in range(args.games):
            a_is_p0 = g % 2 == 0
            if a_is_p0:
                results.append(
                    play_game(net, opp, a_is_player0=True, seed=args.seed + g))
            else:
                results.append(
                    play_game(opp, net, a_is_player0=False, seed=args.seed + g))
        s = summarize(results, f"iter{_iter_no(c)}", args.opp)
        lo, hi = s["a_winrate_ci95"]
        rows.append({
            "iter": _iter_no(c), "file": str(c), "winrate": s["a_winrate"],
            "ci_lo": lo, "ci_hi": hi, "decisive": s["decisive"],
        })
        print(f"  iter {_iter_no(c):>4}: wr={s['a_winrate']:.3f} "
              f"[{lo:.3f},{hi:.3f}] (dec={s['decisive']})", flush=True)

    rows.sort(key=lambda r: r["winrate"], reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\n== top by winrate vs {args.opp} ==")
    for r in rows[:10]:
        print(f"  iter {r['iter']:>4}: {r['winrate']:.3f} "
              f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]")
    print(f"\nwrote {args.out}  (total {time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
