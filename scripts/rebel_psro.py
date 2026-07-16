"""Outer PSRO / double-oracle over (deck, pilot) strategies (design step H).

The ReBeL solver optimises PLAY for a fixed deck matchup; QD optimises DECKS.
Composing them naively invites non-transitive deck-play cycles, so the design's outer
layer is a PSRO/double-oracle meta-game over COMPLETE ``(deck, pilot)`` strategies
(Codex: evaluate the pair, not a coordinate best response).

This runs empirical (restricted) PSRO: build the full slot-swapped cross-play win
matrix over a candidate pool (QD decks x pilots -- ``greedy_plus`` / ``heuristic`` /
go-first exploiter / optional ``rebel``), then a DOUBLE-ORACLE loop on it -- start from
one strategy, each round meta-solve the symmetric Nash of the current subset and add
the pool candidate that best-responds to it, until none beats the mixture (converged).
Reports the meta-Nash support + exploitability over the pool (restricted lower bound).

Full oracle GENERATION (a fresh QD deck BR, a ReBeL net trained vs the meta) is the
the expensive extension; here the "oracles" are drawn from the fixed candidate pool.

  uv run python scripts/rebel_psro.py --decks g9_e0,qd7_r5,metal_aggro --games 24
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

_G: dict = {}


def _pool(decks: list[str], pilots: list[str]) -> list[tuple[str, str, str, bool]]:
    """(label, pilot, deck, force_first) candidate strategies."""
    out: list[tuple[str, str, str, bool]] = []
    for d in decks:
        for p in pilots:
            if p == "greedyFF":
                out.append((f"greedyFF|{d}", "greedy", d, True))
            else:
                out.append((f"{p}|{d}", p, d, False))
    return out


def _init(pool: list[tuple[str, str, str, bool]], net_path: str | None) -> None:
    _G["engine"] = load_engine_data()
    _G["pool"] = pool
    _G["decks"] = {d: read_deck(deck_path(d)) for _, _, d, _ in pool}
    _G["net_path"] = net_path


def _agent(pilot: str, deck_name: str, force_first: bool) -> object:  # noqa: FBT001
    deck = _G["decks"][deck_name]
    if pilot == "rebel":
        from src.rebel.agent import RebelAgent  # noqa: PLC0415
        from src.rebel.value_net import ValueNet  # noqa: PLC0415
        vn = ValueNet.load(_G["net_path"]) if _G.get("net_path") else None
        base: object = RebelAgent(deck, _G["engine"], value_net=vn)
    else:
        base = build_agent(pilot, deck, _G["engine"])
    return _ForcedFirst(base) if force_first else base


def _play(task: dict) -> dict:
    i, j = task["i"], task["j"]
    _, pi_, di, fi = _G["pool"][i]
    _, pj, dj, fj = _G["pool"][j]
    a, b = _agent(pi_, di, fi), _agent(pj, dj, fj)
    sf = task["subj_first"]
    p0, p1 = (a, b) if sf else (b, a)
    res = play_game(p0, p1, a_is_player0=sf, seed=task["seed"])
    return {"i": i, "j": j, "won": int(res.a_won)}


def _nash(w: np.ndarray, iters: int = 20000) -> np.ndarray:
    """Symmetric Nash of the population game via replicator on ``A = W - 0.5``."""
    n = w.shape[0]
    a = w - 0.5
    x = np.full(n, 1.0 / n)
    for _ in range(iters):
        fit = a @ x
        x = np.clip(x * np.exp(0.02 * (fit - x @ fit)), 1e-12, None)
        x /= x.sum()
    return x


def _double_oracle(w: np.ndarray, labels: list[str]) -> tuple[list[int], np.ndarray]:
    """Restricted double-oracle: grow the support by adding the best response to the
    response to the current meta-Nash until none beats the mixture."""
    support = [int(np.argmax((w - 0.5).sum(axis=1)))]  # start: best average strategy
    for _ in range(len(labels)):
        sub = w[np.ix_(support, support)]
        mix_sub = _nash(sub)
        mix = np.zeros(len(labels))
        for s, m in zip(support, mix_sub, strict=True):
            mix[s] = m
        payoff_vs_mix = (w - 0.5) @ mix  # each candidate's edge vs the meta-Nash
        best = int(np.argmax(payoff_vs_mix))
        if best in support or payoff_vs_mix[best] <= 1e-3:
            break
        support.append(best)
    sub = w[np.ix_(support, support)]
    full_mix = np.zeros(len(labels))
    for s, m in zip(support, _nash(sub), strict=True):
        full_mix[s] = m
    return support, full_mix


def main() -> None:
    ap = argparse.ArgumentParser(description="PSRO / double-oracle over (deck, pilot)")
    ap.add_argument("--decks", default="g9_e0,qd7_r5,metal_aggro,grass_aggro")
    ap.add_argument("--pilots", default="greedy_plus,heuristic,greedyFF")
    ap.add_argument("--rebel-net", type=Path, default=None,
                    help="add a 'rebel' pilot using this ValueNet (slow)")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=Path, default=ROOT / "results/rebel_psro.json")
    args = ap.parse_args()

    decks = [d for d in args.decks.split(",") if d]
    pilots = [p for p in args.pilots.split(",") if p]
    if args.rebel_net:
        pilots = [*pilots, "rebel"]
    pool = _pool(decks, pilots)
    labels = [p[0] for p in pool]
    n = len(pool)
    tasks = [{"i": i, "j": j, "subj_first": k % 2 == 0, "seed": (i * 97 + j) * 1000 + k}
             for i, j in itertools.product(range(n), range(n)) if i != j
             for k in range(args.games)]
    print(f"pool={n} strategies, {n * (n - 1)} pairs x {args.games} games", flush=True)

    with Pool(args.workers, initializer=_init,
              initargs=(pool, str(args.rebel_net) if args.rebel_net else None)) as pp:
        rows = pp.map(_play, tasks)

    win = np.full((n, n), 0.5)
    acc: dict[tuple[int, int], list[int]] = {}
    for r in rows:
        acc.setdefault((r["i"], r["j"]), []).append(r["won"])
    for (i, j), ws in acc.items():
        win[i, j] = float(np.mean(ws))

    support, mix = _double_oracle(win, labels)
    expl = float(np.max((win - 0.5) @ mix))  # best response edge vs the meta-Nash
    out = {
        "labels": labels,
        "double_oracle_support": [labels[i] for i in support],
        "meta_nash": {labels[i]: round(float(mix[i]), 3)
                      for i in range(n) if mix[i] > 0.01},
        "exploitability_lower_bound": round(expl, 3),
        "win_matrix": [[round(float(win[i, j]), 3) for j in range(n)]
                       for i in range(n)],
    }
    args.out.write_text(json.dumps(out, indent=2))

    print("\n=== PSRO double-oracle over (deck, pilot) ===")
    print(f"support added (in order): {out['double_oracle_support']}")
    print("meta-Nash mixture:")
    for lbl, p in sorted(out["meta_nash"].items(), key=lambda kv: -kv[1]):
        print(f"  {lbl:<22} {p:.3f}")
    print(f"exploitability (lower bound) = {out['exploitability_lower_bound']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
