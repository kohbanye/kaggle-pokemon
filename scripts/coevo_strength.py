"""Co-evolution play-strength trajectory: round-robin Elo over the rounds -> JSON.

The 2x2 compares only the *endpoints* (init vs round-6); this fills in the curve so
we can see the **slope** -- still climbing (more rounds would help) or plateaued
(stop, switch levers). The play nets ``init`` + ``round_1..R`` play a round-robin, all
piloting the **same fixed meta deck** (so the deck is held constant and the rating is
*pure play*), slots swapped. A Bradley-Terry / Elo rating per round is fit from the win
matrix; the matrix itself exposes any intransitivity a single rating hides.

  uv run python scripts/coevo_strength.py            # full (parallel)
  uv run python scripts/coevo_strength.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from scripts.run_eval import load_engine_data, play_game, read_deck  # noqa: E402
from scripts.strength_over_iters import bradley_terry  # noqa: E402
from src.agents.recurrent_agent import RecurrentNetAgent  # noqa: E402
from src.deck import build_pool  # noqa: E402
from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: E402

INIT = ROOT / "data/paperosfp/main/paper_final.npz"
RUN = ROOT / "data/qdcoevo/run1"
METAL = ROOT / "decklists/metal_aggro.csv"

_G: dict = {}


def _round_net(run: Path, name: str) -> Path:
    return INIT if name == "init" else run / f"round_{name[1:]}/rl/paper_final.npz"


def _init(deck: list[int], names: list[str], run: Path) -> None:
    _G["engine"] = load_engine_data()
    _G["pool"] = build_pool()
    _G["deck"] = deck
    _G["nets"] = {n: RecurrentPolicyValueNet.load(_round_net(run, n)) for n in names}


def _agent(name: str) -> RecurrentNetAgent:
    return RecurrentNetAgent(
        _G["deck"], _G["engine"], net=_G["nets"][name], cb_pool=_G["pool"],
        build_deck_from_net=False, temperature=0.0,
    )


def _play(task: dict) -> dict:
    a, b = _agent(task["i"]), _agent(task["j"])
    i_first = task["i_first"]
    p0, p1 = (a, b) if i_first else (b, a)
    res = play_game(p0, p1, a_is_player0=i_first, seed=task["seed"])
    return {"i": task["i"], "j": task["j"],
            "i_won": int(res.a_won), "dec": int(res.a_won or res.b_won)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Co-evolution round-robin play strength")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--run", type=Path, default=RUN,
                    help="co-evolution run dir holding round_*/rl/paper_final.npz")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--games", type=int, default=80, help="games per pair (swapped)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results/coevo_strength.json")
    args = ap.parse_args()
    n_games = 16 if args.quick else args.games

    names = ["init", *[f"r{r}" for r in range(1, args.rounds + 1)]]
    deck = read_deck(METAL)  # shared fixed deck -> rating is pure play

    tasks = [
        {"i": i, "j": j, "i_first": k % 2 == 0, "seed": idx * 10000 + k}
        for idx, (i, j) in enumerate(combinations(names, 2))
        for k in range(n_games)
    ]
    with Pool(args.workers, initializer=_init,
              initargs=(deck, names, args.run)) as pp:
        rows = pp.map(_play, tasks)

    n = len(names)
    pos = {nm: k for k, nm in enumerate(names)}
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
        "names": names,
        "deck": "metal_aggro",
        "elo": [round(float(e), 1) for e in elo],
        "win_matrix": [[round(float(x), 3) if np.isfinite(x) else None for x in row]
                       for row in winrate],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results))
    print("round   Elo")
    for nm, e in zip(names, elo, strict=True):
        print(f"{nm:>5}  {e:7.1f}")


if __name__ == "__main__":
    main()
