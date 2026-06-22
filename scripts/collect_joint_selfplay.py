"""Collect JOINT self-play games for Phase-5d OSFP (Docker / native x86).

One batch of games yields BOTH kinds of training data the joint loop needs, from
the *same* games (the ByteDance Hearthstone setup -- πBT and πCB learned together):

- **play** transitions: every learner decision (obs + sampled option), for the
  policy/value REINFORCE arm;
- **deck** scores: each sampled learner deck's win/loss tally, for the CB REINFORCE
  arm.

Per deck unit ``i``: sample one learner deck ``D_i`` from the CB head, then play
``games-per-deck`` games of ``D_i`` against opponent decks **freshly sampled from a
CB head** -- the learner's own (``--self-play``) or a past checkpoint's
(``--opp-weights``). Both sides act with the (shared) play head at
``--temperature`` > 0 for exploration. Each game writes a ``"game"`` line
(``{winner-slot, decisions}``, the format ``build_policy_samples`` consumes); each
deck unit writes a ``"deck"`` line (``{deck, wins, losses, draws}``, the format
``cb_rl_sequences`` consumes via ``_deck_return``).

Self-play tags BOTH slots ``"learner"`` (both trajectories are on-policy under the
same net, so train on both); vs an opponent only the learner slot is ``"learner"``.

``--gate-deck DECK.csv`` is the read-only yardstick (not training): a single greedy
learner deck vs a fixed deck, written as one ``"deck"`` line.

  python scripts/collect_joint_selfplay.py --weights data/bc/bc_net_joint.npz \
      --self-play --decks 16 --games-per-deck 16 --out data/jointosfp/iter0
"""

import argparse
import copy
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

LEARNER = "learner"
OPPONENT = "opp"


class _DecisionRecorder:
    """Captures play-arm decisions, **subsampled before the deep-copy**.

    ``deepcopy(obs)`` is the single biggest collection cost (the engine reuses one
    obs dict, so each kept decision must be snapshotted). A self-play iteration
    produces ~200 correlated decisions/game, far more than the policy-gradient step
    needs, so we keep each decision with probability ``keep_prob`` and only copy the
    survivors -- cutting both the copies and the JSONL the loop reads by ~1/keep_prob
    (the loop subsamples again as a safety cap).
    """

    def __init__(self, keep_prob: float, rng: np.random.Generator) -> None:
        self._names: tuple[str, str] = ("", "")
        self._keep = keep_prob
        self._rng = rng
        self.decisions: list[dict] = []

    def begin(self, names: tuple[str, str]) -> None:
        self._names = names
        self.decisions = []

    def on_decision(self, slot: int, obs: dict, choice: list[int]) -> None:
        if self._keep < 1.0 and self._rng.random() >= self._keep:
            return  # dropped before the expensive deepcopy
        self.decisions.append({
            "slot": int(slot),
            "agent": self._names[slot],
            "obs": copy.deepcopy(obs),  # battle_select reuses the dict -> must copy
            "choice": [int(c) for c in choice],
        })

    def on_end(self, winner: int) -> None:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect joint self-play games")
    parser.add_argument("--weights", type=Path, required=True, help="learner net")
    parser.add_argument(
        "--opp-weights", type=Path, default=None,
        help="opponent checkpoint; omit (with --self-play) to use the learner",
    )
    parser.add_argument(
        "--self-play", action="store_true",
        help="opponent decks/plays come from the learner net",
    )
    parser.add_argument("--decks", type=int, default=16, help="learner decks to sample")
    parser.add_argument("--games-per-deck", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--play-keep-prob", type=float, default=1.0,
        help="keep each play decision with this prob (subsample before deepcopy)",
    )
    parser.add_argument(
        "--gate-deck", type=Path, default=None,
        help="eval yardstick: score one greedy deck vs this fixed deck (not training)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.self_play and args.opp_weights is None and args.gate_deck is None:
        parser.error("need --self-play, --opp-weights, or --gate-deck")
    return args


def _play_deck_unit(  # noqa: PLR0913 - a sim helper legitimately threads inputs
    learner_deck: list[int],
    engine: dict,
    pool: CardPool,
    nets: tuple[PolicyValueNet, PolicyValueNet],
    self_play: bool,  # noqa: FBT001 - an internal sim helper, not a public flag
    games: int,
    base_seed: int,
    temperature: float,
    handle,  # noqa: ANN001 - an open text file
    rec: _DecisionRecorder,
) -> tuple[int, int, int]:
    """Play ``learner_deck`` (fixed) vs CB-sampled opponent decks; write game lines.

    Returns the learner's (wins, losses, draws). Both sides act with the play head
    at ``temperature`` for exploration; the opponent samples a fresh deck per game.
    """
    learner_net, opp_net = nets
    wins = losses = draws = 0
    for k in range(games):
        seed = base_seed + k
        # Learner keeps the unit's fixed deck but still gets the embedding index.
        learner = NetAgent(
            learner_deck, engine, net=learner_net, cb_pool=pool,
            build_deck_from_net=False, temperature=temperature, seed=seed,
        )
        opp = NetAgent(
            learner_deck, engine, net=opp_net, cb_pool=pool, sample_deck=True,
            temperature=temperature, seed=seed,
        )
        learner_first = k % 2 == 0
        if self_play:  # both on-policy under one net -> train on both trajectories
            names = (LEARNER, LEARNER)
        else:
            names = (LEARNER, OPPONENT) if learner_first else (OPPONENT, LEARNER)
        p0, p1 = (learner, opp) if learner_first else (opp, learner)
        rec.begin(names)
        res = play_game(p0, p1, a_is_player0=learner_first, seed=seed, recorder=rec)
        if res.a_won:  # a tracks the learner regardless of slot
            wins += 1
        elif res.b_won:
            losses += 1
        else:
            draws += 1
        handle.write(json.dumps({
            "type": "game", "winner": int(res.winner), "decisions": rec.decisions,
        }) + "\n")
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

    games_dir = args.out / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    (args.out / "engine.json").write_text(json.dumps(engine))
    shard = games_dir / f"joint_s{args.seed}.jsonl"

    if args.gate_deck is not None:  # eval yardstick: one greedy deck vs a fixed deck
        fixed = read_deck(args.gate_deck)
        deck = build_deck(learner_net, pool, feats)  # greedy
        wins = losses = draws = 0
        for k in range(args.games_per_deck):
            learner = NetAgent(
                deck, engine, net=learner_net, cb_pool=pool, build_deck_from_net=False,
            )
            opp = NetAgent(
                fixed, engine, net=learner_net, cb_pool=pool, build_deck_from_net=False,
            )
            learner_first = k % 2 == 0
            p0, p1 = (learner, opp) if learner_first else (opp, learner)
            res = play_game(p0, p1, a_is_player0=learner_first, seed=args.seed + k)
            wins += res.a_won
            losses += res.b_won
            draws += not (res.a_won or res.b_won)
        with shard.open("w") as handle:
            handle.write(json.dumps({
                "type": "deck", "deck": deck,
                "wins": int(wins), "losses": int(losses), "draws": int(draws),
            }) + "\n")
        dec = wins + losses
        print(
            f"== gate vs {args.gate_deck.stem}: distinct={len(set(deck))} "
            f"w/l/d={wins}/{losses}/{draws} wr={wins / dec if dec else 0.0:.3f} ==",
        )
        return

    opp_label = "self" if args.opp_weights is None else args.opp_weights.name
    print(
        f"== joint self-play: learner={args.weights.name} opp={opp_label} "
        f"decks={args.decks} games/deck={args.games_per_deck} t={args.temperature} ==",
    )
    rec = _DecisionRecorder(args.play_keep_prob, np.random.default_rng(args.seed))
    with shard.open("w") as handle:
        for i in range(args.decks):
            deck = build_deck(learner_net, pool, feats, rng, greedy=False)  # sampled
            wins, losses, draws = _play_deck_unit(
                deck, engine, pool, (learner_net, opp_net), args.self_play,
                args.games_per_deck, args.seed + i * args.games_per_deck,
                args.temperature, handle, rec,
            )
            handle.write(json.dumps({
                "type": "deck", "deck": deck,
                "wins": wins, "losses": losses, "draws": draws,
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
