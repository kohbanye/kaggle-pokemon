"""Strength over training iterations via a checkpoint round-robin -> JSON.

A greedy-independent strength measure: a set of checkpoints play each other (full
agents, own greedy decks, slot-swapped), giving a win matrix from which a
Bradley-Terry / Elo rating per checkpoint is fit. The rating-vs-iteration curve is
the real "did it get stronger" signal (the in-loop gate vs greedy is confounded);
the matrix itself exposes any intransitivity (rock-paper-scissors between
checkpoints) that a single rating hides.

Also samples K decks per checkpoint (CB head, no engine) and stores them, so the
notebook can embed the explored deck space (t-SNE) coloured by iteration.

  uv run python scripts/strength_over_iters.py            # full (parallel)
  uv run python scripts/strength_over_iters.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.run_eval import load_engine_data, play_game  # noqa: E402
from src.agents.recurrent_agent import RecurrentNetAgent  # noqa: E402
from src.deck import build_pool  # noqa: E402
from src.net.cb import build_deck  # noqa: E402
from src.net.deck_sample import sample_deck_with_logp  # noqa: E402
from src.net.features import CardFeatures, load_engine_json  # noqa: E402
from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: E402

ENGINE_JSON = ROOT / "data/bc/engine.json"
ITERS = [50, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]

_G: dict = {}


def _ckpt(it: int) -> Path:
    return ROOT / f"data/paperosfp/main/paperiter_{it}.npz"


def _init(decks: dict) -> None:
    _G["engine"] = load_engine_data()
    _G["pool"] = build_pool()
    _G["decks"] = decks
    _G["nets"] = {}


def _agent(it: int) -> RecurrentNetAgent:
    if it not in _G["nets"]:
        _G["nets"][it] = RecurrentPolicyValueNet.load(str(_ckpt(it)))
    return RecurrentNetAgent(
        _G["decks"][it], _G["engine"], net=_G["nets"][it], cb_pool=_G["pool"],
        build_deck_from_net=False, temperature=0.0,
    )


def _play(task: dict) -> dict:
    i, j = task["i"], task["j"]
    a = _agent(i)
    b = _agent(j)
    i_first = task["i_first"]
    p0, p1 = (a, b) if i_first else (b, a)
    res = play_game(p0, p1, a_is_player0=i_first, seed=task["seed"])
    return {"i": i, "j": j, "i_won": int(res.a_won), "dec": int(res.a_won or res.b_won)}


def bradley_terry(wins: np.ndarray, games: np.ndarray, n_iter: int = 500) -> np.ndarray:
    """Bradley-Terry strengths (smoothed) from a win/games matrix -> Elo per row."""
    n = wins.shape[0]
    w = wins + 0.5  # smoothing so an undefeated checkpoint doesn't diverge
    g = games + 1.0
    p = np.ones(n)
    for _ in range(n_iter):
        wtot = w.sum(axis=1)
        denom = np.array([
            sum(g[i, j] / (p[i] + p[j]) for j in range(n) if j != i)
            for i in range(n)
        ])
        p = wtot / np.where(denom > 0, denom, 1.0)
        p /= np.exp(np.mean(np.log(p)))  # geometric-mean normalise
    return 400.0 * np.log10(p) + 1500.0  # Elo-scaled, centred at 1500


def main() -> None:
    ap = argparse.ArgumentParser(description="Checkpoint round-robin strength")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--games", type=int, default=40, help="games per pair (swapped)")
    ap.add_argument("--decks", type=int, default=40, help="decks sampled per ckpt")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results/strength.json")
    args = ap.parse_args()
    iters = ITERS[::3] if args.quick else ITERS
    n_games = 12 if args.quick else args.games
    n_decks = 10 if args.quick else args.decks

    pool = build_pool()
    feats = CardFeatures(load_engine_json(ENGINE_JSON))
    # Each checkpoint's greedy deck (for the matches) + sampled decks (for the embed).
    rng = np.random.default_rng(0)
    greedy_decks, sampled = {}, {}
    for it in iters:
        net = RecurrentPolicyValueNet.load(_ckpt(it))
        greedy_decks[it] = build_deck(net, pool, feats)
        sampled[it] = [sample_deck_with_logp(net, pool, feats, rng)[0]
                       for _ in range(n_decks)]

    tasks = [
        {"i": i, "j": j, "i_first": k % 2 == 0, "seed": idx * 10000 + k}
        for idx, (i, j) in enumerate(combinations(iters, 2))
        for k in range(n_games)
    ]
    with Pool(args.workers, initializer=_init, initargs=(greedy_decks,)) as pp:
        rows = pp.map(_play, tasks)

    n = len(iters)
    pos = {it: k for k, it in enumerate(iters)}
    wins = np.zeros((n, n))
    games = np.zeros((n, n))
    for r in rows:
        if not r["dec"]:
            continue
        a, b = pos[r["i"]], pos[r["j"]]
        games[a, b] += 1
        games[b, a] += 1
        if r["i_won"]:
            wins[a, b] += 1
        else:
            wins[b, a] += 1

    elo = bradley_terry(wins, games)
    winrate = np.divide(wins, games, out=np.full_like(wins, np.nan), where=games > 0)
    results = {
        "iters": iters,
        "elo": [round(float(e), 1) for e in elo],
        "win_matrix": [[round(float(x), 3) if np.isfinite(x) else None for x in row]
                       for row in winrate],
        "sampled_decks": {str(it): sampled[it] for it in iters},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results))
    print("iter   Elo")
    for it, e in zip(iters, elo, strict=True):
        print(f"{it:>5}  {e:7.1f}")


if __name__ == "__main__":
    main()
