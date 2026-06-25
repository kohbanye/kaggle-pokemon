"""Head-to-head evaluation of two nets (cross-architecture), the real strength test.

Unlike the in-loop gate (both sides the *same* net -> confounded), this pits two
**independent** nets against each other: each builds its own greedy deck and plays
it with deterministic (temperature 0) play, slots swapped each game, Wilson 95% CI.

Agent specs: ``recurrent:<npz>`` (the paper net) or ``base:<npz>`` (the Phase-5d /
flat net). Example:

  python scripts/eval_paper_vs.py \
      --a recurrent:data/paperosfp/main/paper_final.npz \
      --b base:/abs/path/jointiter_649.npz --games 300
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CG_PARENT = ROOT / "data" / "sample_submission"
sys.path.insert(0, str(CG_PARENT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_eval import load_engine_data, play_game, read_deck  # noqa: E402

from src.agents.net_agent import NetAgent  # noqa: E402
from src.agents.recurrent_agent import RecurrentNetAgent  # noqa: E402
from src.deck import build_pool  # noqa: E402
from src.harness.stats import wilson_interval  # noqa: E402
from src.net.model import PolicyValueNet  # noqa: E402
from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: E402


def _make_agent(spec: str, fallback: list[int], engine: dict, pool: object) -> object:
    """Build an agent from a ``kind:path`` spec (greedy deck, temperature-0 play)."""
    kind, _, path = spec.partition(":")
    if kind == "recurrent":
        net = RecurrentPolicyValueNet.load(path)
        return RecurrentNetAgent(fallback, engine, net=net, cb_pool=pool)
    if kind == "base":
        net = PolicyValueNet.load(path)
        return NetAgent(fallback, engine, net=net, cb_pool=pool)
    msg = f"unknown agent kind {kind!r} (use recurrent: or base:)"
    raise SystemExit(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-to-head net evaluation")
    parser.add_argument("--a", required=True, help="recurrent:<npz> or base:<npz>")
    parser.add_argument("--b", required=True, help="recurrent:<npz> or base:<npz>")
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fallback-deck", type=Path, default=ROOT / "decklists" / "metal_aggro.csv",
    )
    args = parser.parse_args()

    engine = load_engine_data()
    pool = build_pool()
    fallback = read_deck(args.fallback_deck)
    agent_a = _make_agent(args.a, fallback, engine, pool)
    agent_b = _make_agent(args.b, fallback, engine, pool)

    wins = decisive = 0
    for k in range(args.games):
        a_first = k % 2 == 0
        p0, p1 = (agent_a, agent_b) if a_first else (agent_b, agent_a)
        res = play_game(p0, p1, a_is_player0=a_first, seed=args.seed + k)
        if res.a_won or res.b_won:
            decisive += 1
            wins += res.a_won
    p, low, high = wilson_interval(wins, decisive)
    print(
        f"A={args.a}\nB={args.b}\n"
        f"A wins {wins}/{decisive} decisive ({args.games} played)  "
        f"winrate={p:.3f}  95% CI=[{low:.3f}, {high:.3f}]  "
        f"{'A stronger' if low > 0.5 else 'B stronger' if high < 0.5 else 'inconclusive'}",  # noqa: E501
    )


if __name__ == "__main__":
    main()
