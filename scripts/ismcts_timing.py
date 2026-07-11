"""Time net-guided SO-ISMCTS self-play under different leaf modes / net sizes.

Measures wall-time per game and per searched decision for:
  - leaf = full net rollout to terminal (rollout_cap large)  -- the original cost;
  - leaf = value head (rollout_cap 0)  -- speedup lever (1), AlphaZero-style;
  - a short-rollout warmup (rollout_cap 8);
across the trained 256-wide net and a bigger (untrained) net -- so we see the value-head
speedup and the net-size cost before committing to the AlphaZero build.

  uv run python scripts/ismcts_timing.py --games 4 --iterations 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.run_eval import load_engine_data, play_game, read_deck  # noqa: E402
from src.agents import build_agent  # noqa: E402
from src.net.recurrent_model import (  # noqa: E402
    RecurrentNetConfig,
    RecurrentPolicyValueNet,
)
from src.search.ismcts import RecurrentIsmctsAgent  # noqa: E402

NET = ROOT / "data/qdcoevo/run7/round_6/rl/paper_final.npz"


def _bigger_net(width: int) -> RecurrentPolicyValueNet:
    """Fresh (untrained) net at a larger play-LSTM width -- for forward-cost timing."""
    import numpy as np  # noqa: PLC0415

    cfg = RecurrentNetConfig(play_lstm_hidden=width)
    return RecurrentPolicyValueNet.random(np.random.default_rng(0), cfg)


def _time_config(label: str, net: RecurrentPolicyValueNet, deck: list[int],  # noqa: PLR0913
                 engine: dict, opp_prior: list[int], basics: list[int],
                 games: int, iterations: int, rollout_cap: int) -> None:
    opp = build_agent("greedy_plus", deck, engine)
    t0 = time.perf_counter()
    n_searches = 0
    for k in range(games):
        agent = RecurrentIsmctsAgent(
            deck, engine, net, opp_prior=opp_prior, opp_basics=basics,
            iterations=iterations, rollout_cap=rollout_cap, move_budget_s=999.0,
            cb_pool=None, build_deck_from_net=False, temperature=0.0)
        sf = k % 2 == 0
        p0, p1 = (agent, opp) if sf else (opp, agent)
        play_game(p0, p1, a_is_player0=sf, seed=k)
        n_searches += agent.n_searches
    dt = time.perf_counter() - t0
    per_game = dt / max(games, 1)
    per_search = dt / max(n_searches, 1)
    print(f"  {label:28} {dt:6.1f}s total | {per_game:6.2f}s/game | "
          f"{per_search*1000:6.1f}ms/search (searches={n_searches})",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="qdgp_best")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=64)
    ap.add_argument("--big-width", type=int, default=384)
    args = ap.parse_args()

    engine = load_engine_data()
    dp = ROOT / "decklists" / f"{args.deck}.csv"
    if not dp.exists():
        dp = ROOT / "decklists" / "candidates" / f"{args.deck}.csv"
    deck = read_deck(dp)
    cards = engine["cards"]
    basics = [c for c in deck if cards.get(c) and cards[c].get("basic")]
    meta = [read_deck(p) for p in sorted((ROOT / "decklists").glob("*.csv"))]
    opp_prior = [c for d in meta for c in d]

    net256 = RecurrentPolicyValueNet.load(str(NET))
    net_big = _bigger_net(args.big_width)
    print(f"deck={args.deck} games={args.games} iterations={args.iterations}  "
          f"(256=trained, {args.big_width}=fresh)", flush=True)
    print("== 256-wide net ==", flush=True)
    _time_config("value-leaf (cap0) [lever 1]", net256, deck, engine, opp_prior,
                 basics, args.games, args.iterations, 0)
    _time_config("short-rollout (cap8)", net256, deck, engine, opp_prior, basics,
                 args.games, args.iterations, 8)
    _time_config("full-rollout (cap200) [orig]", net256, deck, engine, opp_prior,
                 basics, args.games, args.iterations, 200)
    print(f"== {args.big_width}-wide net ==", flush=True)
    _time_config("value-leaf (cap0) [lever 1]", net_big, deck, engine, opp_prior,
                 basics, args.games, args.iterations, 0)


if __name__ == "__main__":
    main()
