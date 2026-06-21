"""Score CB-head decks for Phase 5b-ii deck RL (Docker only).

Samples decks from the net's CB head and scores each by playing it K times against
a fixed reference deck, with **both** players using the same (frozen) play head --
so the only thing that varies is the deck. Writes one JSON line per sampled deck:
the deck in CB pick order plus its win/loss/draw tally. ``scripts/train_cb.py``
turns that into a REINFORCE advantage (``cb_rl_samples``) and updates the CB head.

With ``--greedy`` it scores a single greedily-decoded deck over many games -- the
**gate**: does the learned deck beat the fixed reference (metal_aggro)?

  docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim \
      python scripts/collect_cb.py --weights /work/data/bc/bc_net_emb.npz \
        --opp-deck /work/decklists/metal_aggro.csv --decks 16 --games-per-deck 16 \
        --out /work/data/cb/iter0
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CG_PARENT = ROOT / "data" / "sample_submission"
sys.path.insert(0, str(CG_PARENT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from run_eval import load_engine_data, play_game, read_deck  # noqa: E402

from src.agents.net_agent import NetAgent  # noqa: E402
from src.deck import build_pool  # noqa: E402
from src.net.cb import build_deck  # noqa: E402
from src.net.features import CardFeatures  # noqa: E402
from src.net.model import PolicyValueNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score CB-head decks (Phase 5b-ii)")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--opp-deck", type=Path, required=True, help="reference deck")
    parser.add_argument("--decks", type=int, default=16, help="decks to sample")
    parser.add_argument("--games-per-deck", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--greedy", action="store_true", help="score one greedy deck (gate mode)",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def score_deck(  # noqa: PLR0913 - a sim helper legitimately threads its inputs
    deck: list[int],
    opp_deck: list[int],
    engine: dict,
    net: PolicyValueNet,
    games: int,
    base_seed: int,
) -> tuple[int, int, int]:
    """Play ``deck`` vs ``opp_deck`` (same play head) ``games`` times: w/l/draw."""
    wins = losses = draws = 0
    for k in range(games):
        learner = NetAgent(deck, engine, net=net)
        opp = NetAgent(opp_deck, engine, net=net)
        learner_first = k % 2 == 0
        p0, p1 = (learner, opp) if learner_first else (opp, learner)
        res = play_game(p0, p1, a_is_player0=learner_first, seed=base_seed + k)
        if res.a_won:  # a_is_player0 tracks the learner, so a_won == learner won
            wins += 1
        elif res.b_won:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws


def main() -> None:
    args = parse_args()
    engine = load_engine_data()
    pool = build_pool()
    feats = CardFeatures(engine)
    opp_deck = read_deck(args.opp_deck)
    net = PolicyValueNet.load(args.weights)
    rng = np.random.default_rng(args.seed)

    games_dir = args.out / "decks"
    games_dir.mkdir(parents=True, exist_ok=True)
    (args.out / "engine.json").write_text(json.dumps(engine))
    shard = games_dir / f"cb_s{args.seed}.jsonl"

    mode = "greedy(gate)" if args.greedy else "sampled"
    print(
        f"== collect_cb [{mode}]: weights={args.weights.name} opp={args.opp_deck.stem} "
        f"decks={args.decks} games/deck={args.games_per_deck} ==",
    )
    with shard.open("w") as handle:
        for i in range(args.decks):
            deck = build_deck(net, pool, feats, rng, greedy=args.greedy)
            wins, losses, draws = score_deck(
                deck, opp_deck, engine, net, args.games_per_deck,
                args.seed + i * args.games_per_deck,
            )
            handle.write(json.dumps({
                "deck": deck, "wins": wins, "losses": losses, "draws": draws,
            }) + "\n")
            handle.flush()
            dec = wins + losses
            wr = wins / dec if dec else 0.0
            print(
                f"  deck {i}: distinct={len(set(deck))} "
                f"w/l/d={wins}/{losses}/{draws} wr={wr:.3f}",
            )
    print(f"wrote {shard} ({args.decks} decks)")


if __name__ == "__main__":
    main()
