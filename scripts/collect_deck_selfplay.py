"""Collect deck self-play games for OSFP deck learning (Docker only).

The learner samples a deck from its CB head and scores it by playing K games
against **opponent decks ALSO sampled from a CB head** -- either the learner's own
(``--self-play``) or a past checkpoint's (``--opp-weights``). Both sides use the
**same frozen play head**, so the only thing that varies is the deck -- the winner
is decided by deck strength. This is the fix for the Phase-5b-ii flaw: scoring a
deck against a *fixed* reference (metal_aggro) made every learned deck lose (no
advantage signal); here the opponent is drawn from the net's own deck distribution,
so ~50% win by symmetry and the REINFORCE advantage is always informative.

Writes one JSON line per learner deck: the deck in CB pick order + its win/loss/
draw tally -- the format ``src.net.bc_data.cb_rl_sequences`` consumes (via
``scripts/train_deck_osfp.py``).

``--gate-deck DECK.csv`` is an **eval yardstick** (not training): it scores a
single *greedy* learner deck against a fixed deck to track whether self-play is
making the deck stronger over time. Scoring against a fixed deck is fine for a
read-only yardstick; it is only as the *training signal* that it fails.

  docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim \
      python scripts/collect_deck_selfplay.py \
        --weights /work/data/bc/bc_net_lstm.npz --self-play \
        --decks 16 --games-per-deck 16 --out /work/data/deckosfp/iter0
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
from src.deck import CardPool, build_pool  # noqa: E402
from src.net.cb import build_deck  # noqa: E402
from src.net.features import CardFeatures  # noqa: E402
from src.net.model import PolicyValueNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect deck self-play games")
    parser.add_argument("--weights", type=Path, required=True, help="learner net")
    parser.add_argument(
        "--opp-weights", type=Path, default=None,
        help="opponent CB checkpoint; omit (with --self-play) to use the learner",
    )
    parser.add_argument(
        "--self-play", action="store_true", help="opponent decks come from the learner",
    )
    parser.add_argument("--decks", type=int, default=16, help="learner decks to sample")
    parser.add_argument("--games-per-deck", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gate-deck", type=Path, default=None,
        help="eval yardstick: score one greedy deck vs this fixed deck (not training)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.self_play and args.opp_weights is None and args.gate_deck is None:
        parser.error("need --self-play, --opp-weights, or --gate-deck")
    return args


def _score_vs_opponent(  # noqa: PLR0913 - a sim helper legitimately threads inputs
    learner_deck: list[int],
    engine: dict,
    pool: CardPool,
    learner_net: PolicyValueNet,
    opp_net: PolicyValueNet,
    games: int,
    base_seed: int,
) -> tuple[int, int, int]:
    """Play ``learner_deck`` (fixed) vs freshly CB-sampled opponent decks: w/l/d.

    Both agents play with their net's (frozen, identical) play head, so the deck
    decides. The opponent draws a new sampled deck every game (``sample_deck``).
    """
    wins = losses = draws = 0
    for k in range(games):
        learner = NetAgent(learner_deck, engine, net=learner_net)
        opp = NetAgent(
            learner_deck, engine, net=opp_net, cb_pool=pool, sample_deck=True,
            seed=base_seed + k,
        )
        learner_first = k % 2 == 0
        p0, p1 = (learner, opp) if learner_first else (opp, learner)
        res = play_game(p0, p1, a_is_player0=learner_first, seed=base_seed + k)
        if res.a_won:  # a tracks the learner
            wins += 1
        elif res.b_won:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws


def _score_vs_fixed(  # noqa: PLR0913 - a sim helper legitimately threads inputs
    learner_deck: list[int],
    fixed_deck: list[int],
    engine: dict,
    net: PolicyValueNet,
    games: int,
    base_seed: int,
) -> tuple[int, int, int]:
    """Yardstick: ``learner_deck`` vs a fixed deck, same play head: w/l/d."""
    wins = losses = draws = 0
    for k in range(games):
        learner = NetAgent(learner_deck, engine, net=net)
        opp = NetAgent(fixed_deck, engine, net=net)
        learner_first = k % 2 == 0
        p0, p1 = (learner, opp) if learner_first else (opp, learner)
        res = play_game(p0, p1, a_is_player0=learner_first, seed=base_seed + k)
        if res.a_won:
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
    learner_net = PolicyValueNet.load(args.weights)
    opp_net = (
        PolicyValueNet.load(args.opp_weights)
        if args.opp_weights is not None else learner_net
    )
    rng = np.random.default_rng(args.seed)

    decks_dir = args.out / "decks"
    decks_dir.mkdir(parents=True, exist_ok=True)
    (args.out / "engine.json").write_text(json.dumps(engine))
    shard = decks_dir / f"deck_s{args.seed}.jsonl"

    if args.gate_deck is not None:  # eval yardstick: one greedy deck vs a fixed deck
        fixed = read_deck(args.gate_deck)
        deck = build_deck(learner_net, pool, feats)  # greedy
        wins, losses, draws = _score_vs_fixed(
            deck, fixed, engine, learner_net, args.games_per_deck, args.seed,
        )
        with shard.open("w") as handle:
            handle.write(json.dumps({
                "deck": deck, "wins": wins, "losses": losses, "draws": draws,
            }) + "\n")
        dec = wins + losses
        print(
            f"== gate vs {args.gate_deck.stem}: distinct={len(set(deck))} "
            f"w/l/d={wins}/{losses}/{draws} wr={wins / dec if dec else 0.0:.3f} ==",
        )
        return

    opp_label = "self" if args.opp_weights is None else args.opp_weights.name
    print(
        f"== deck self-play: learner={args.weights.name} opp={opp_label} "
        f"decks={args.decks} games/deck={args.games_per_deck} ==",
    )
    with shard.open("w") as handle:
        for i in range(args.decks):
            deck = build_deck(learner_net, pool, feats, rng, greedy=False)  # sampled
            wins, losses, draws = _score_vs_opponent(
                deck, engine, pool, learner_net, opp_net, args.games_per_deck,
                args.seed + i * args.games_per_deck,
            )
            handle.write(json.dumps({
                "deck": deck, "wins": wins, "losses": losses, "draws": draws,
            }) + "\n")
            handle.flush()
            dec = wins + losses
            print(
                f"  deck {i}: distinct={len(set(deck))} "
                f"w/l/d={wins}/{losses}/{draws} wr={wins / dec if dec else 0.0:.3f}",
            )
    print(f"wrote {shard} ({args.decks} decks)")


if __name__ == "__main__":
    main()
