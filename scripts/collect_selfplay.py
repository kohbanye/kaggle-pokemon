"""Collect OSFP self-play trajectories for Phase-5 RL (Docker only).

Plays the current learner net (stochastic, ``temperature > 0``) against either a
copy of itself (``--self-play``) or a sampled opponent (another checkpoint via
``--opp-weights`` or a scripted baseline via ``--opp-agent``) on a *fixed* deck,
and records every decision in the same JSONL format ``scripts/collect_bc.py``
emits, so ``src.net.bc_data.build_policy_samples`` consumes it unchanged. The
learner's game outcome becomes the policy-gradient return (``train_osfp.py``).

Tagging convention (this is what ``build_policy_samples(teachers={"learner"})``
filters on):

- **self-play**: both slots are tagged ``"learner"`` -- both trajectories are
  on-policy and the winner's moves get +1 / the loser's -1 (zero-sum balanced),
  so we train on both;
- **vs an opponent**: only the learner's slot is tagged ``"learner"`` (the
  opponent's moves are environment, never trained on), tagged ``"opp"``.

Output: ``<out>/games/*.jsonl`` (one game per line) + ``<out>/engine.json``.

  docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim \
      python scripts/collect_selfplay.py \
        --learner-weights /work/data/bc/bc_net.npz --opp-agent heuristic \
        --deck /work/decklists/metal_aggro.csv --games 64 --temperature 1.0 \
        --out /work/data/osfp/iter0
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # make `src` importable
CG_PARENT = ROOT / "data" / "sample_submission"
sys.path.insert(0, str(CG_PARENT))  # make `cg` importable
sys.path.insert(0, str(ROOT / "scripts"))  # reuse run_eval / collect_bc helpers

from collect_bc import BCRecorder  # noqa: E402
from run_eval import load_engine_data, play_game, read_deck  # noqa: E402

from src.agents import Agent, build_agent  # noqa: E402
from src.agents.net_agent import NetAgent  # noqa: E402
from src.net.model import PolicyValueNet  # noqa: E402

LEARNER = "learner"
OPPONENT = "opp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Phase-5 self-play logs")
    parser.add_argument("--learner-weights", type=Path, required=True)
    parser.add_argument("--opp-weights", type=Path, default=None, help="checkpoint")
    parser.add_argument("--opp-agent", default=None, help="scripted baseline name")
    parser.add_argument(
        "--self-play", action="store_true", help="opponent is the learner itself",
    )
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--deck", type=Path, required=True, help="fixed deck for both")
    parser.add_argument("--seed", type=int, default=0, help="base seed; g uses seed+g")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.self_play and not (args.opp_weights or args.opp_agent):
        parser.error("need --self-play, --opp-weights, or --opp-agent")
    return args


def main() -> None:
    args = parse_args()
    engine = load_engine_data()
    deck = read_deck(args.deck)
    temp = args.temperature

    learner_net = PolicyValueNet.load(args.learner_weights)
    opp_net = (
        PolicyValueNet.load(args.opp_weights)
        if not args.self_play and args.opp_weights
        else None
    )

    def make_learner() -> NetAgent:
        return NetAgent(deck, engine, net=learner_net, temperature=temp)

    def make_opp() -> Agent:
        if args.self_play:
            return NetAgent(deck, engine, net=learner_net, temperature=temp)
        if opp_net is not None:
            return NetAgent(deck, engine, net=opp_net, temperature=temp)
        return build_agent(args.opp_agent, deck, engine)

    opp_label = "self" if args.self_play else (args.opp_agent or str(args.opp_weights))
    games_dir = args.out / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    (args.out / "engine.json").write_text(json.dumps(engine))
    shard = games_dir / f"selfplay_s{args.seed}.jsonl"

    print(
        f"== self-play: learner={args.learner_weights} opp={opp_label} "
        f"deck={args.deck.stem} games={args.games} temp={temp} ==",
    )
    wins = {LEARNER: 0, OPPONENT: 0, "other": 0}
    with shard.open("w") as handle:
        recorder = BCRecorder(handle)
        for g in range(args.games):
            learner, opp = make_learner(), make_opp()
            learner_first = g % 2 == 0
            if args.self_play:
                names = (LEARNER, LEARNER)
            else:
                names = (LEARNER, OPPONENT) if learner_first else (OPPONENT, LEARNER)
            p0, p1 = (learner, opp) if learner_first else (opp, learner)
            recorder.begin_game(names)
            res = play_game(
                p0, p1, a_is_player0=learner_first, seed=args.seed + g,
                recorder=recorder,
            )
            winner_name = names[res.winner] if res.winner in (0, 1) else "other"
            wins[winner_name if winner_name in wins else "other"] += 1
            if g % 25 == 0:
                print(f"  game {g}: opp={opp_label} winner={res.winner}")
    print(
        f"wrote {shard} ({args.games} games) -- "
        f"learner_wins={wins[LEARNER]} opp_wins={wins[OPPONENT]} other={wins['other']}",
    )


if __name__ == "__main__":
    main()
