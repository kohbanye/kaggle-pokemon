"""Collect recurrent self-play trajectories for the paper-faithful V-Trace loop.

This logs **whole trajectories with behaviour log-probs**, which V-Trace needs
(an earlier collector logged independent, subsampled decisions instead):

- each game writes a ``"game"`` line ``{winner, decks, decisions}`` where
  ``decks`` is ``{slot: {deck, deck_logp}}`` for each learner slot (both in
  self-play, one vs an opponent) and ``decisions`` is every applied decision with
  its ``slot`` / ``obs`` / ``choice`` / ``logp`` (the actor's ``log μ(a|s)``).

A fresh deck is sampled from the CB head **per game** (one episode = one game), so
the deck arm trains on on-policy picks. Both sides act with the recurrent play head
at ``--temperature`` > 0 for exploration. ``build_episodes`` turns these lines into
training episodes on the host.

  python scripts/collect_paper_selfplay.py --weights data/paper/init.npz \
      --self-play --games 64 --out data/paperosfp/iter0
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

from src.agents.recurrent_agent import RecurrentNetAgent  # noqa: E402
from src.deck import build_pool  # noqa: E402
from src.net.cb import build_deck  # noqa: E402
from src.net.features import CardFeatures  # noqa: E402
from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: E402

LEARNER = "learner"
OPPONENT = "opp"


class _TrajectoryRecorder:
    """Logs every decision with the acting agent's behaviour log-prob.

    Holds the live agents so it can read ``agents[slot].last_logp`` (set by the
    recurrent agent for each single-select decision) at decision time. ``obs`` is
    deep-copied because ``battle_select`` reuses the dict.
    """

    def __init__(self) -> None:
        self.agents: tuple[RecurrentNetAgent, RecurrentNetAgent] | None = None
        self.names: tuple[str, str] = ("", "")
        self.decisions: list[dict] = []

    def begin(
        self,
        agents: tuple[RecurrentNetAgent, RecurrentNetAgent],
        names: tuple[str, str],
    ) -> None:
        self.agents = agents
        self.names = names
        self.decisions = []

    def on_decision(self, slot: int, obs: dict, choice: list[int]) -> None:
        logp = self.agents[slot].last_logp if self.agents is not None else 0.0
        self.decisions.append({
            "slot": int(slot),
            "agent": self.names[slot],
            "obs": copy.deepcopy(obs),
            "choice": [int(c) for c in choice],
            "logp": float(logp),
        })

    def on_end(self, winner: int) -> None:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect recurrent self-play")
    parser.add_argument("--weights", type=Path, required=True, help="learner net")
    parser.add_argument("--opp-weights", type=Path, default=None)
    parser.add_argument(
        "--opp-deck", type=Path, default=None,
        help="opponent plays this fixed meta deck piloted by the learner net "
             "(the OSFP meta-deck baseline: external pressure on deck quality)",
    )
    parser.add_argument("--self-play", action="store_true")
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fallback-deck", type=Path, default=ROOT / "decklists" / "metal_aggro.csv",
        help="legal deck used if a CB sample fails (submission hygiene)",
    )
    parser.add_argument(
        "--gate-deck", type=Path, default=None,
        help="yardstick: greedy learner deck vs this fixed deck (writes a gate line)",
    )
    parser.add_argument(
        "--deck-pool", type=Path, default=None,
        help="JSON list of decks; both sides draw a deck from it and play with the "
             "learner net (QD-archive self-play -- decks fixed, only play is trained)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.self_play
        and args.opp_weights is None
        and args.opp_deck is None
        and args.gate_deck is None
        and args.deck_pool is None
    ):
        parser.error("need --self-play / --opp-weights / --opp-deck / --gate-deck / "
                     "--deck-pool")
    return args


def run_gate(args: argparse.Namespace, engine: dict, pool: object, shard: Path) -> None:
    """Greedy learner deck vs a fixed deck (read-only yardstick, not training)."""
    feats = CardFeatures(engine)
    net = RecurrentPolicyValueNet.load(args.weights)
    deck = build_deck(net, pool, feats)  # greedy, capped composition
    fixed = read_deck(args.gate_deck)
    wins = losses = draws = 0
    for k in range(args.games):
        learner = RecurrentNetAgent(
            deck, engine, net=net, cb_pool=pool, build_deck_from_net=False,
        )
        opp = RecurrentNetAgent(
            fixed, engine, net=net, cb_pool=pool, build_deck_from_net=False,
        )
        learner_first = k % 2 == 0
        p0, p1 = (learner, opp) if learner_first else (opp, learner)
        res = play_game(p0, p1, a_is_player0=learner_first, seed=args.seed + k)
        wins += res.a_won
        losses += res.b_won
        draws += not (res.a_won or res.b_won)
    with shard.open("w") as handle:
        handle.write(json.dumps({
            "type": "gate", "wins": int(wins), "losses": int(losses),
            "draws": int(draws), "distinct": len(set(deck)),
        }) + "\n")
    dec = wins + losses
    print(f"== gate vs {args.gate_deck.stem}: w/l/d={wins}/{losses}/{draws} "
          f"wr={wins / dec if dec else 0.0:.3f} ==")


def main() -> None:
    args = parse_args()
    engine = load_engine_data()
    pool = build_pool()
    games_dir = args.out / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    (args.out / "engine.json").write_text(json.dumps(engine))
    shard = games_dir / f"paper_s{args.seed}.jsonl"

    if args.gate_deck is not None:
        run_gate(args, engine, pool, shard)
        return

    fallback = read_deck(args.fallback_deck)
    learner_net = RecurrentPolicyValueNet.load(args.weights)
    opp_net = (
        RecurrentPolicyValueNet.load(args.opp_weights)
        if args.opp_weights is not None else learner_net
    )
    # --opp-deck: opponent plays a fixed meta deck piloted by the learner net (an
    # external strong deck that punishes degenerate learner decks).
    opp_deck = read_deck(args.opp_deck) if args.opp_deck is not None else None
    # --deck-pool: both sides draw a deck from the QD archive (decks fixed; only the
    # play head is trained -- the "QD decks + RL play" split). Both slots are LEARNER.
    # Accepts a plain ``[[deck], ...]`` list or a MAP-Elites archive ``{"cells": ...}``.
    deck_pool = None
    if args.deck_pool is not None:
        raw = json.loads(args.deck_pool.read_text())
        deck_pool = (
            [c["deck"] for c in raw["cells"]] if isinstance(raw, dict) else raw
        )

    rec = _TrajectoryRecorder()
    wins = losses = draws = 0
    with shard.open("w") as handle:
        for k in range(args.games):
            seed = args.seed + k
            if deck_pool is not None:  # archive self-play: both decks from the pool
                drng = np.random.default_rng(seed)
                ldeck = list(deck_pool[int(drng.integers(len(deck_pool)))])
                odeck = list(deck_pool[int(drng.integers(len(deck_pool)))])
                learner = RecurrentNetAgent(
                    ldeck, engine, net=learner_net, cb_pool=pool,
                    build_deck_from_net=False, temperature=args.temperature, seed=seed,
                )
                opp = RecurrentNetAgent(
                    odeck, engine, net=learner_net, cb_pool=pool,
                    build_deck_from_net=False, temperature=args.temperature,
                    seed=seed + 7919,
                )
            else:
                learner = RecurrentNetAgent(
                    fallback, engine, net=learner_net, cb_pool=pool, sample_deck=True,
                    temperature=args.temperature, seed=seed,
                )
                if opp_deck is not None:  # fixed meta deck, piloted by the learner net
                    opp = RecurrentNetAgent(
                        opp_deck, engine, net=learner_net, cb_pool=pool,
                        build_deck_from_net=False, temperature=args.temperature,
                        seed=seed + 7919,
                    )
                else:  # opponent samples its own deck (self-play or a checkpoint)
                    opp = RecurrentNetAgent(
                        fallback, engine, net=opp_net, cb_pool=pool, sample_deck=True,
                        temperature=args.temperature, seed=seed + 7919,
                    )
            learner_first = k % 2 == 0
            names = (
                (LEARNER, LEARNER) if args.self_play or deck_pool is not None
                else (LEARNER, OPPONENT) if learner_first
                else (OPPONENT, LEARNER)
            )
            p0, p1 = (learner, opp) if learner_first else (opp, learner)
            rec.begin((p0, p1), names)
            res = play_game(p0, p1, a_is_player0=learner_first, seed=seed, recorder=rec)
            wins += res.a_won
            losses += res.b_won
            draws += not (res.a_won or res.b_won)

            decks = {
                str(slot): {"deck": agent.deck, "deck_logp": agent.deck_logp}
                for slot, agent in enumerate((p0, p1))
                if names[slot] == LEARNER
            }
            handle.write(json.dumps({
                "type": "game", "winner": int(res.winner),
                "decks": decks, "decisions": rec.decisions,
            }) + "\n")
            handle.flush()
    dec = wins + losses
    print(
        f"== paper self-play: games={args.games} w/l/d={wins}/{losses}/{draws} "
        f"learner_wr={wins / dec if dec else 0.0:.3f} -> {shard} ==",
    )


if __name__ == "__main__":
    main()
