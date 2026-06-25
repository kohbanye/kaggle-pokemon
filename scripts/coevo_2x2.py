"""Co-evolution deck x play 2x2: did RL play *generalise* across the QD archive?

Crosses two **play nets** (the pre-co-evolution init net vs the round-6 final net)
against two **decks** (the final net's greedy deck / a neutral meta deck). Each cell
is ``pilot vs greedy`` with **both sides on the same deck**, so the deck is controlled
*within* a cell and the win rate isolates play quality:

                       play = init        play = round6
    deck = net deck      cell (1)            cell (2)
    deck = meta deck     cell (3)            cell (4)

Reading it: compare **down a column is meaningless; compare across a row** -- (2)>(1)
and (4)>(3) mean the round-6 play is stronger *with the deck held fixed* (pure play
delta). (4) in particular -- the net piloting a deck it did **not** design -- is the
direct test of "deck-specialised play" being cured.

Two **deck-controlled duels** (``final`` play vs ``init`` play on the *same* deck,
slots swapped) are also recorded -- the most sensitive play delta. ``duel_metadeck``
cross-checks the head-to-head in ``eval_paper_vs`` (which, with ``sample_deck`` off,
also pits the two nets on the shared fallback meta deck).

Parallel (one process per game, both nets cached per worker).

  uv run python scripts/coevo_2x2.py --quick
  uv run python scripts/coevo_2x2.py \\
      --final data/qdcoevo/run1/round_6/rl/paper_final.npz
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
from src.net.cb import build_deck  # noqa: E402
from src.net.features import CardFeatures, load_engine_json  # noqa: E402
from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: E402

INIT = ROOT / "data/paperosfp/main/paper_final.npz"
FINAL = ROOT / "data/qdcoevo/run1/round_6/rl/paper_final.npz"
METAL = ROOT / "decklists/metal_aggro.csv"
ENGINE_JSON = ROOT / "data/bc/engine.json"

_G: dict = {}


def _init(init_path: str, final_path: str) -> None:
    _G["engine"] = load_engine_data()
    _G["pool"] = build_pool()
    _G["init"] = RecurrentPolicyValueNet.load(init_path)
    _G["final"] = RecurrentPolicyValueNet.load(final_path)


def _agent(play: str, deck: list[int]) -> object:
    """``init`` / ``final`` -> that net (temperature 0); anything else -> baseline."""
    if play in ("init", "final"):
        return RecurrentNetAgent(
            deck, _G["engine"], net=_G[play], cb_pool=_G["pool"],
            build_deck_from_net=False, temperature=0.0,
        )
    return build_agent(play, deck, _G["engine"])


def _play(task: dict) -> dict:
    a = _agent(task["pa"], task["da"])
    b = _agent(task["pb"], task["db"])
    a_first = task["a_first"]
    p0, p1 = (a, b) if a_first else (b, a)
    res = play_game(p0, p1, a_is_player0=a_first, seed=task["seed"])
    return {"m": task["m"], "a_won": int(res.a_won),
            "dec": int(res.a_won or res.b_won)}


def _wr(rows: list[dict]) -> dict:
    wins = sum(r["a_won"] for r in rows)
    dec = sum(r["dec"] for r in rows)
    p, lo, hi = wilson_interval(wins, dec)
    return {"winrate": round(p, 3), "ci": [round(lo, 3), round(hi, 3)],
            "decisive": dec, "games": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Co-evolution deck x play 2x2")
    ap.add_argument("--init", type=Path, default=INIT, help="pre-co-evo play net")
    ap.add_argument("--final", type=Path, default=FINAL, help="round-6 play net")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results/coevo_2x2.json")
    args = ap.parse_args()
    n = 24 if args.quick else 160

    pool = build_pool()
    feats = CardFeatures(load_engine_json(ENGINE_JSON))
    final_net = RecurrentPolicyValueNet.load(args.final)
    net_deck = build_deck(final_net, pool, feats)  # the deck the final net designs
    metal = read_deck(METAL)                       # a deck neither net designed

    # 2x2: each cell = pilot vs greedy, BOTH on the same deck (deck controlled).
    cells = [
        {"m": "init_play_netdeck", "pa": "init", "da": net_deck,
         "pb": "greedy", "db": net_deck},
        {"m": "final_play_netdeck", "pa": "final", "da": net_deck,
         "pb": "greedy", "db": net_deck},
        {"m": "init_play_metadeck", "pa": "init", "da": metal,
         "pb": "greedy", "db": metal},
        {"m": "final_play_metadeck", "pa": "final", "da": metal,
         "pb": "greedy", "db": metal},
        # deck-controlled duels: final play vs init play on the SAME deck (the most
        # sensitive play delta; ``duel_metadeck`` cross-checks eval_paper_vs).
        {"m": "duel_netdeck", "pa": "final", "da": net_deck,
         "pb": "init", "db": net_deck},
        {"m": "duel_metadeck", "pa": "final", "da": metal,
         "pb": "init", "db": metal},
    ]

    tasks = [
        {**s, "a_first": k % 2 == 0, "seed": si * 100000 + k}
        for si, s in enumerate(cells) for k in range(n)
    ]
    with Pool(args.workers, initializer=_init,
              initargs=(str(args.init), str(args.final))) as pp:
        rows = pp.map(_play, tasks)

    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["m"], []).append(r)
    matchups = {m: _wr(rs) for m, rs in by.items()}

    results = {
        "init": str(args.init), "final": str(args.final),
        "games_per_cell": n, "matchups": matchups,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
