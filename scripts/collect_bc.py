"""Collect teacher battle logs for Phase-4 behaviour cloning (Docker only).

Drives teacher-vs-opponent games across the demo decks and records, for every
in-game decision, the deep-copied observation, the chosen action, the deciding
slot and the agent name, plus each game's winner. Output: JSONL shards under
``<out>/games/`` (one game per line) and the engine card/attack dump
``<out>/engine.json``. ``scripts/train_bc.py`` consumes both on the host -- the
engine is Linux-only, so collection runs here and training runs natively.

Both players' decisions are logged (tagged by agent name) so the same logs can
warm-start either teacher; ``train_bc.py`` filters with ``--teachers``.

  docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim \
      python scripts/collect_bc.py --teacher heuristic --games 400 --out data/bc
"""

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # make `src` importable
CG_PARENT = ROOT / "data" / "sample_submission"
sys.path.insert(0, str(CG_PARENT))  # make `cg` importable
sys.path.insert(0, str(ROOT / "scripts"))  # reuse run_eval helpers

from run_eval import load_engine_data, play_game, read_deck  # noqa: E402

from src.agents import build_agent  # noqa: E402


class BCRecorder:
    """Accumulates one game's decisions and flushes a JSON line per game."""

    def __init__(self, out) -> None:  # noqa: ANN001 - a writable text handle
        self._out = out
        self._names: tuple[str, str] = ("", "")
        self._decisions: list[dict] = []

    def begin_game(self, names: tuple[str, str]) -> None:
        """Set the slot->agent-name map for the game about to be played."""
        self._names = names
        self._decisions = []

    def on_decision(self, slot: int, obs: dict, choice: list[int]) -> None:
        """Record one applied decision (obs is deep-copied; it is reused upstream)."""
        self._decisions.append({
            "slot": slot,
            "agent": self._names[slot],
            "obs": copy.deepcopy(obs),
            "choice": [int(c) for c in choice],
        })

    def on_end(self, winner: int) -> None:
        """Flush the finished game as one JSON line."""
        record = {"winner": int(winner), "decisions": self._decisions}
        self._out.write(json.dumps(record) + "\n")
        self._out.flush()
        self._decisions = []


def build_matchups(
    deck_names: list[str], opponents: list[str],
) -> list[tuple[str, str, str]]:
    """All (teacher-deck, opponent-deck, opponent-agent) combinations for diversity."""
    return [
        (teacher_deck, opp_deck, opp)
        for opp in opponents
        for teacher_deck in deck_names
        for opp_deck in deck_names
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Phase-4 BC teacher logs")
    parser.add_argument("--teacher", default="heuristic", help="teacher agent name")
    parser.add_argument(
        "--opponents", default="heuristic,greedy", help="comma-separated opponents",
    )
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0, help="base seed; g uses seed+g")
    parser.add_argument("--decks", type=Path, default=ROOT / "decklists")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "bc")
    args = parser.parse_args()

    engine = load_engine_data()
    decks = {p.stem: read_deck(p) for p in sorted(args.decks.glob("*.csv"))}
    if not decks:
        raise SystemExit(f"no decks found in {args.decks}")
    opponents = [o for o in args.opponents.split(",") if o]
    matchups = build_matchups(sorted(decks), opponents)

    games_dir = args.out / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    (args.out / "engine.json").write_text(json.dumps(engine))
    shard = games_dir / f"{args.teacher}_s{args.seed}.jsonl"

    print(
        f"== collect: teacher={args.teacher} opponents={opponents} "
        f"decks={len(decks)} matchups={len(matchups)} games={args.games} ==",
    )
    with shard.open("w") as handle:
        recorder = BCRecorder(handle)
        for g in range(args.games):
            teacher_deck, opp_deck, opp = matchups[g % len(matchups)]
            teacher_agent = build_agent(args.teacher, decks[teacher_deck], engine)
            opp_agent = build_agent(opp, decks[opp_deck], engine)
            teacher_first = g % 2 == 0
            if teacher_first:
                names = (args.teacher, opp)
                p0, p1 = teacher_agent, opp_agent
            else:
                names = (opp, args.teacher)
                p0, p1 = opp_agent, teacher_agent
            recorder.begin_game(names)
            res = play_game(
                p0, p1, a_is_player0=teacher_first, seed=args.seed + g,
                recorder=recorder,
            )
            if g % 25 == 0:
                print(
                    f"  game {g}: {names} vs decks=({teacher_deck},{opp_deck}) "
                    f"winner={res.winner}",
                )
    print(f"wrote {shard} ({args.games} games) and {args.out / 'engine.json'}")


if __name__ == "__main__":
    main()
