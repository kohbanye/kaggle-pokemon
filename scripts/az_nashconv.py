"""Multi-metric robustness eval for the CURRENT population: NashConv + exploitability.

Elo (the ladder) and a single mean win-rate are gameable and hide worst-case fragility
(Balduzzi "Re-evaluating Evaluation"; [[rl-eval-methodology]]). This computes, over a
small CURATED current population of (pilot, deck) strategies, the full slot-swapped win
matrix ``W[i][j] = P(i beats j)`` and reports the clone-invariant robustness metrics:

    exploitability(i) = 0.5 - min_j W[i][j]      (worst-case matchup deficit)
    NashConv(i)       = 2 * exploitability(i)     (lower = harder to exploit = robust)

plus the symmetric-population Nash mixture (SUPPORT = the strategies that survive a
rational meta-game -- if the AZ net is absent it is dominated). Complements
the metric that actually tracks the ladder, complementing az_eval's play-decomposition.

  uv run python scripts/az_nashconv.py --net data/coevo_az_v2/gen9/net.npz --games 30
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

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

# Curated CURRENT population: (label, pilot, deck, force_first). "net" uses --net.
POP: list[tuple[str, str, str, bool]] = [
    ("gp|g9_e0", "greedy_plus", "g9_e0", False),      # submission candidate (best deck)
    ("gp|qd7_r5", "greedy_plus", "qd7_r5", False),    # prior best deck
    ("gp|metal", "greedy_plus", "metal_aggro", False),
    ("net|g9_e0", "net", "g9_e0", False),             # AZ student on its co-evo deck
    ("net|metal", "net", "metal_aggro", False),       # AZ student on aggro
    ("greedy|metal", "greedy", "metal_aggro", False),
    ("heur|g9_e0", "heuristic", "g9_e0", False),
    ("greedyFF|metal", "greedy", "metal_aggro", True),  # go-first exploiter
]
_G: dict = {}


def _init(net_path: str) -> None:
    from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: PLC0415

    _G["engine"] = load_engine_data()
    _G["net"] = RecurrentPolicyValueNet.load(net_path)
    _G["decks"] = {d: read_deck(deck_path(d)) for _, _, d, _ in POP}


def _agent(pilot: str, deck_name: str, force_first: bool) -> object:  # noqa: FBT001
    deck = _G["decks"][deck_name]
    if pilot == "net":
        from src.agents.recurrent_agent import RecurrentNetAgent  # noqa: PLC0415
        base = RecurrentNetAgent(deck, _G["engine"], net=_G["net"], cb_pool=None,
                                 build_deck_from_net=False, temperature=0.0)
    else:
        base = build_agent(pilot, deck, _G["engine"])
    return _ForcedFirst(base) if force_first else base


def _play(task: dict) -> dict:
    i, j = task["i"], task["j"]
    _, pi_, di, fi = POP[i]
    _, pj, dj, fj = POP[j]
    a = _agent(pi_, di, fi)
    b = _agent(pj, dj, fj)
    sf = task["subj_first"]
    p0, p1 = (a, b) if sf else (b, a)
    res = play_game(p0, p1, a_is_player0=sf, seed=task["seed"])
    return {"i": i, "j": j, "won": int(res.a_won), "dec": int(res.a_won or res.b_won)}


def _nash_symmetric(w: np.ndarray, iters: int = 20000) -> np.ndarray:
    """Symmetric Nash mixture of the population game via replicator dynamics on payoff
    ``A = W - 0.5`` (zero-sum, antisymmetric). Returns the mixture over strategies."""
    n = w.shape[0]
    a = w - 0.5
    x = np.full(n, 1.0 / n)
    for _ in range(iters):
        fit = a @ x
        x = x * np.exp(0.02 * (fit - x @ fit))
        x = np.clip(x, 1e-12, None)
        x /= x.sum()
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description="NashConv / exploitability, current pop")
    ap.add_argument("--net", type=Path, default=ROOT / "data/coevo_az_v2/gen9/net.npz")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=Path, default=ROOT / "results/az_nashconv.json")
    args = ap.parse_args()

    n = len(POP)
    tasks = [
        {"i": i, "j": j, "subj_first": k % 2 == 0,
         "seed": (i * 97 + j) * 1000 + k}
        for i, j in itertools.product(range(n), range(n)) if i != j
        for k in range(args.games)
    ]
    print(f"population={n} pairs={n * (n - 1)} games/pair={args.games} "
          f"total={len(tasks)}", flush=True)

    with Pool(args.workers, initializer=_init, initargs=(str(args.net),)) as pp:
        rows = pp.map(_play, tasks)

    win = np.full((n, n), np.nan)
    acc: dict[tuple[int, int], list[int]] = {}
    for r in rows:
        acc.setdefault((r["i"], r["j"]), []).append(r["won"])
    for (i, j), ws in acc.items():
        win[i, j] = float(np.mean(ws))
    for i in range(n):
        win[i, i] = 0.5

    expl = np.array([0.5 - np.nanmin(win[i]) for i in range(n)])
    nashconv = 2.0 * expl
    nash_mix = _nash_symmetric(np.nan_to_num(win, nan=0.5))

    labels = [p[0] for p in POP]
    order = np.argsort(nashconv)
    out = {
        "net": str(args.net), "labels": labels,
        "win_matrix": [[round(float(win[i, j]), 3)
                       for j in range(n)] for i in range(n)],
        "per_strategy": {
            labels[i]: {
                "exploitability": round(float(expl[i]), 3),
                "nashconv": round(float(nashconv[i]), 3),
                "worst_vs": labels[int(np.nanargmin(win[i]))],
                "worst_wr": round(float(np.nanmin(win[i])), 3),
                "nash_weight": round(float(nash_mix[i]), 3),
            } for i in range(n)
        },
    }
    args.out.write_text(json.dumps(out, indent=2))

    print("\n=== NashConv / exploitability (lower = more robust), current pop ===")
    print(f"{'strategy':<16}{'NashConv':>9}{'exploit':>9}{'nash_wt':>9}  worst matchup")
    for i in order:
        s = out["per_strategy"][labels[i]]
        print(f"{labels[i]:<16}{s['nashconv']:>9.3f}{s['exploitability']:>9.3f}"
              f"{s['nash_weight']:>9.3f}  vs {s['worst_vs']} ({s['worst_wr']:.2f})")
    supp = [labels[i] for i in range(n) if nash_mix[i] > 0.01]
    print(f"\nNash support (survives the meta-game): {supp}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
