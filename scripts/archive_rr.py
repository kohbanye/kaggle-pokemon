"""Intransitivity check: net-piloted round-robin among a run's archive decks.

The QD fitness scores each deck vs the FIXED strong-meta gauntlet, so diverse niches
look weak by construction (they lose the prize race to fast aggro). This asks the real
question for archive-relative (co-evolutionary) fitness: do the slow / single-prize /
ramp decks beat any *peers* in the archive? If they have favourable matchups (rock-
paper-scissors), co-evo fitness would keep them competitive; if they lose to everyone,
it's a genuine power floor.

Every deck is piloted by the SAME net (the run's final net), so the win matrix isolates
the *deck* matchup, not piloting. Native/Docker (imports cg). Run:
  uv run python scripts/archive_rr.py \
      --archive data/qdcoevo/run7/round_6/qd_archive.json \
      --net data/qdcoevo/run7/round_6/rl/paper_final.npz --games 30
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
from src.qd.deck_qd import behaviour_descriptor  # noqa: E402

_G: dict = {}


def _init(net_path: str, decks: list[list[int]]) -> None:
    from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: PLC0415

    _G["engine"] = load_engine_data()
    _G["pool"] = build_pool()
    _G["net"] = RecurrentPolicyValueNet.load(net_path)
    _G["decks"] = decks


def _agent(deck: list[int]) -> RecurrentNetAgent:
    return RecurrentNetAgent(deck, _G["engine"], net=_G["net"], cb_pool=_G["pool"],
                             build_deck_from_net=False, temperature=0.0)


def _play(task: dict) -> dict:
    a, b = _agent(_G["decks"][task["i"]]), _agent(_G["decks"][task["j"]])
    a_first = task["a_first"]
    p0, p1 = (a, b) if a_first else (b, a)
    res = play_game(p0, p1, a_is_player0=a_first, seed=task["seed"])
    return {"i": task["i"], "j": task["j"], "a_won": int(res.a_won),
            "dec": int(res.a_won or res.b_won)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Archive deck round-robin (net-piloted)")
    ap.add_argument("--archive", type=Path,
                    default=ROOT / "data/qdcoevo/run7/round_6/qd_archive.json")
    ap.add_argument("--net", type=Path,
                    default=ROOT / "data/qdcoevo/run7/round_6/rl/paper_final.npz")
    ap.add_argument("--games", type=int, default=30, help="games per ordered pair")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=Path, default=ROOT / "results/run7_archive_rr.json")
    args = ap.parse_args()

    pool = build_pool()
    cells = json.loads(args.archive.read_text())["cells"]
    decks = [c["deck"] for c in cells]
    labels = [str(behaviour_descriptor(c["deck"], pool)) for c in cells]
    metawr = [c.get("stats", {}).get("winrate") for c in cells]  # vs-meta (for ref)
    n = len(decks)

    tasks = [
        {"i": i, "j": j, "a_first": k % 2 == 0, "seed": (i * 97 + j) * 1000 + k}
        for i in range(n) for j in range(n) if i != j
        for k in range(args.games)
    ]
    with Pool(args.workers, initializer=_init,
              initargs=(str(args.net), decks)) as pp:
        rows = pp.map(_play, tasks)

    # win[i][j] = i's winrate piloting deck i vs deck j
    wins = [[0] * n for _ in range(n)]
    dec = [[0] * n for _ in range(n)]
    for r in rows:
        wins[r["i"]][r["j"]] += r["a_won"]
        dec[r["i"]][r["j"]] += r["dec"]
    mat = [[(wins[i][j] / dec[i][j] if dec[i][j] else None) for j in range(n)]
           for i in range(n)]

    def field_mean(i: int) -> float:
        vals = [mat[i][j] for j in range(n) if j != i and mat[i][j] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    order = sorted(range(n), key=field_mean, reverse=True)
    print(f"net={args.net.name}  decks={n}  games/pair={args.games}")
    print("\nlabel(prize,speed)  vs-meta  field-mean  beats(>0.55): peers")
    for i in order:
        beats = [labels[j] for j in range(n)
                 if j != i and mat[i][j] is not None and mat[i][j] > 0.55]
        mw = f"{metawr[i]:.2f}" if metawr[i] is not None else "  - "
        print(f"  {labels[i]:<14} {mw:>6}   {field_mean(i):.3f}     {beats}")

    args.out.write_text(json.dumps(
        {"labels": labels, "meta_winrate": metawr,
         "matrix": mat, "field_mean": [field_mean(i) for i in range(n)]}, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
