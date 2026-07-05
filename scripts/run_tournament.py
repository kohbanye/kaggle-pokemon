"""Phase 1 deck round-robin: fix one agent, vary the deck, measure win rates.

Runs every candidate deck against every other with a fixed policy (the random
baseline by default), using paired side-swapped seeds, and prints a matchup
matrix with Wilson 95% intervals. This is the "agent fixed, deck varies" setup
from PLAN.md Phase 1 — the biggest lever on Elo.

Requires the gitignored competition download (`bash scripts/download_data.sh`)
so the `cg` engine and card data are present. Candidate decks are read from the
`decks/` directory (one whitespace-separated list of 60 card IDs per file); the
sample deck at data/sample_submission/deck.csv is included automatically.

    uv run python scripts/run_tournament.py --games-per-pair 40

NOTE: the CgEngine adapter below is the single place that depends on the raw
observation schema (how the engine reports whose turn it is and who won). Those
assumptions are marked; confirm them against the real engine once data/ is
downloaded, and change only this adapter if they differ.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "data" / "sample_submission"))  # cg package lives here

from agents.random_agent import RandomAgent  # noqa: E402
from src.cards import load_cards  # noqa: E402
from src.eval.deck import load_deck, validate_deck  # noqa: E402
from src.eval.runner import Decision, Outcome, aggregate, run_tournament  # noqa: E402
from src.eval.schedule import round_robin  # noqa: E402
from src.eval.stats import WinRate  # noqa: E402

DATA = REPO / "data"
DECKS_DIR = REPO / "decks"
SAMPLE_DECK = DATA / "sample_submission" / "deck.csv"


class CgEngine:
    """Adapter wrapping the compiled cg engine to the runner's Engine protocol.

    The engine holds a single global game, so one CgEngine drives one game at a
    time (which is all the runner needs).
    """

    def __init__(self) -> None:
        from cg.game import battle_finish, battle_select, battle_start

        self._start = battle_start
        self._select = battle_select
        self._finish = battle_finish

    def start(
        self, deck0: list[int], deck1: list[int], seed: int,  # noqa: ARG002
    ) -> Decision | Outcome:
        # NOTE: seed threading depends on the engine's start signature; the
        # sample uses battle_start(deck0, deck1). If the engine accepts a seed,
        # pass it here for reproducible determinization.
        obs, start = self._start(list(deck0), list(deck1))
        if obs is None:
            msg = f"battle failed to start: errorType={start.errorType}"
            raise RuntimeError(msg)
        return self._step(obs)

    def choose(self, choice: list[int]) -> Decision | Outcome:
        return self._step(self._select(choice))

    def finish(self) -> None:
        self._finish()

    def _step(self, obs: dict) -> Decision | Outcome:
        current = obs.get("current")
        if current is not None and current.get("result", -1) != -1:
            result = current["result"]  # engine player index of the winner
            return Outcome(winner=None if result < 0 else int(result))
        # ASSUMPTION: the observation names the player to act. Confirm the key
        # against the real engine; sim_smoke.py only ever drives a single agent.
        player = int(obs.get("turnPlayer", obs.get("player", 0)))
        return Decision(player=player, obs=obs)


def discover_decks() -> dict[str, list[int]]:
    decks: dict[str, list[int]] = {}
    if SAMPLE_DECK.exists():
        decks["sample"] = load_deck(SAMPLE_DECK)
    if DECKS_DIR.exists():
        for path in sorted(DECKS_DIR.glob("*.txt")) + sorted(DECKS_DIR.glob("*.csv")):
            decks[path.stem] = load_deck(path)
    return decks


def card_legality_sets() -> tuple[set[int], set[int]]:
    cards = load_cards()
    basics = set(cards.loc[cards["stage_or_type"] == "Basic Energy", "card_id"])
    aces = set(cards.loc[cards["is_ace_spec"], "card_id"])
    return basics, aces


def fmt(wr: WinRate) -> str:
    low, high = wr.interval
    star = "*" if wr.significant else " "
    record = f"{wr.wins}-{wr.losses}-{wr.ties}"
    return f"{wr.rate:5.1%} [{low:4.1%},{high:4.1%}]{star} ({record})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 deck round-robin")
    parser.add_argument("--games-per-pair", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    decks = discover_decks()
    if len(decks) < 2:
        raise SystemExit(
            f"need >=2 candidate decks, found {len(decks)}. "
            f"Download data and/or add deck files under {DECKS_DIR}.",
        )

    basics, aces = card_legality_sets()
    for name, deck in decks.items():
        problems = validate_deck(deck, basic_energy_ids=basics, ace_spec_ids=aces)
        if problems:
            raise SystemExit(f"deck {name!r} is illegal: {problems}")

    matches = round_robin(
        list(decks), games_per_pair=args.games_per_pair, base_seed=args.seed,
    )
    print(
        f"{len(decks)} decks, {len(matches)} games "
        f"({args.games_per_pair}/pair, seed {args.seed})",
    )

    engine = CgEngine()
    results = run_tournament(engine, lambda deck: RandomAgent(deck), decks, matches)
    table = aggregate(results)

    for (low_name, high_name), wr in sorted(table.items()):
        print(f"{low_name:>16} vs {high_name:<16}  {fmt(wr)}")


if __name__ == "__main__":
    main()
