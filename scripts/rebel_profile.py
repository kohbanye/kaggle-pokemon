"""Profile where a depth-4 ReBeL solve spends time -- to find the perf lever that would
make deep search servable (depth is the only proven play lever; depth>=4 is currently
serving-infeasible at ~56s/move). Runs ONE short RebelAgent-vs-greedy game under cProfile
and prints the top cumulative-time functions, then per-solve wall-time stats.

  uv run python scripts/rebel_profile.py --depth 4 --particles 4 --sweeps 4 --moves 8
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.run_eval import load_engine_data, play_game, resolve_deck  # noqa: E402
from src.agents import build_agent  # noqa: E402
from src.rebel.agent import RebelAgent  # noqa: E402


class _Timed(RebelAgent):
    """RebelAgent that records per-solve wall time."""

    solve_times: list[float]

    def _solve(self, obs: dict):  # type: ignore[override]
        t0 = time.perf_counter()
        try:
            return super()._solve(obs)
        finally:
            self.solve_times.append(time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="metal_aggro")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--particles", type=int, default=4)
    ap.add_argument("--sweeps", type=int, default=4)
    ap.add_argument("--tpw", type=int, default=4)
    ap.add_argument("--net", default="data/rebel/vnd3_metal_r4.npz")
    ap.add_argument("--topn", type=int, default=30)
    args = ap.parse_args()

    engine = load_engine_data()
    deck = resolve_deck(args.deck)
    net = None
    npz = ROOT / args.net
    if npz.exists():
        from src.rebel.value_net import ValueNet  # noqa: PLC0415
        net = ValueNet.load(str(npz))

    agent = _Timed(deck, engine, n_particles=args.particles, max_depth=args.depth,
                   sweeps=args.sweeps, traversals_per_world=args.tpw, seed=0,
                   value_net=net)
    agent.solve_times = []
    opp = build_agent("greedy", deck, engine)

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    play_game(agent, opp, a_is_player0=True, seed=1234)
    pr.disable()
    dt = time.perf_counter() - t0

    st = sorted(agent.solve_times)
    n = len(st)
    print(f"\n=== depth={args.depth} particles={args.particles} sweeps={args.sweeps} "
          f"tpw={args.tpw} ===")
    print(f"game wall={dt:.1f}s  ReBeL solves={n}  "
          f"solve mean={sum(st)/n if n else 0:.2f}s  "
          f"median={st[n//2] if n else 0:.2f}s  max={st[-1] if n else 0:.2f}s")
    print(f"total-solve={sum(st):.1f}s ({100*sum(st)/dt:.0f}% of game)\n")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(args.topn)
    # keep only the readable table lines
    for line in s.getvalue().splitlines():
        if "src/rebel" in line or "cg/" in line or "search_" in line or \
           "value_net" in line or "encode" in line or "opp_context" in line or \
           "ncalls" in line or "function calls" in line:
            print(line)


if __name__ == "__main__":
    main()
