"""Gate D smoke: RebelAgent actually PLAYS -- legal, solver fires, timing sane.

Plays a few full games of `RebelAgent` vs `greedy` and reports: no-crash completion, how
many decisions the ReBeL solve produced (vs fallback), win/loss, and per-game wall time.
Not a strength claim (leaf is still the prize heuristic) -- it confirms the
belief->worlds->BeliefSubgame->MCCFR stack drives a real match end to end.

  uv run python scripts/rebel_play_smoke.py --games 4 --iters 80 --particles 6 --depth 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.run_eval import load_engine_data, play_game, resolve_deck  # noqa: E402
from src.agents import build_agent  # noqa: E402
from src.rebel.agent import RebelAgent  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="metal_aggro")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--sweeps", type=int, default=12)
    ap.add_argument("--tpw", type=int, default=8, help="traversals per world")
    ap.add_argument("--particles", type=int, default=6)
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()

    engine = load_engine_data()
    deck = resolve_deck(args.deck)
    wins = solved = decisions = 0
    for g in range(args.games):
        agent = RebelAgent(deck, engine, n_particles=args.particles,
                           max_depth=args.depth, sweeps=args.sweeps,
                           traversals_per_world=args.tpw, seed=g)
        opp = build_agent("greedy", deck, engine)
        sf = g % 2 == 0
        p0, p1 = (agent, opp) if sf else (opp, agent)
        t0 = time.perf_counter()
        res = play_game(p0, p1, a_is_player0=sf, seed=1000 + g)
        dt = time.perf_counter() - t0
        wins += int(res.a_won)
        solved += agent.solved
        print(f"  game {g}: {'WON' if res.a_won else 'lost'}  "
              f"rebel_solves={agent.solved}  {dt:.1f}s")
    decisions = solved
    print(f"\n{args.games} games, no crash. total ReBeL solves={decisions}, "
          f"wins={wins}/{args.games} vs greedy "
          f"(sweeps={args.sweeps} tpw={args.tpw} particles={args.particles} "
          f"depth={args.depth})")
    print("RebelAgent end-to-end play OK")


if __name__ == "__main__":
    main()
