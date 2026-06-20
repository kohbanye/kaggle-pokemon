"""Battle runner / evaluation harness (Linux x86-64, run under Docker).

Plays two registered agents head-to-head over N games with first/second slot
swapping, fixed per-game agent seeds, and per-move timing, then prints and logs
a win-rate + Wilson 95% CI summary. This is Phase 0's "ruler": the same harness
every later ablation is measured on.

  docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim \
      python scripts/run_eval.py --a greedy --b random --games 500 --seed 0

Reproducibility note: the engine's internal RNG (deck shuffles, coin flips) is
*not* exposed by the public API, so individual games cannot be replayed bit-for-
bit. We seed every agent's randomness per game and log all outcomes; relative
comparisons are reproducible statistically (large N + slot swap + Wilson CI),
which is what the keep/drop decisions rely on.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # make `src` importable when run as a script
CG_PARENT = ROOT / "data" / "sample_submission"
sys.path.insert(0, str(CG_PARENT))

from cg.api import all_attack, all_card_data  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from src.agents import Agent, build_agent  # noqa: E402
from src.agents.base import is_legal, legal_fallback  # noqa: E402
from src.agents.net_agent import NetAgent  # noqa: E402
from src.deck import CardPool, build_pool  # noqa: E402
from src.harness.result import ABORTED, GameResult  # noqa: E402
from src.harness.stats import summarize  # noqa: E402

DECK_REQUEST = {"select": None, "logs": [], "current": None}
DEFAULT_DECK = CG_PARENT / "deck.csv"
MAX_SELECTIONS = 5000  # hard stop; a healthy game finishes in well under this


class Recorder(Protocol):
    """Optional per-decision hook for ``play_game`` (Phase-4 teacher-log collection)."""

    def on_decision(self, slot: int, obs: dict, choice: list[int]) -> None: ...
    def on_end(self, winner: int) -> None: ...


def _make_agent(
    name: str,
    deck: list[int],
    engine: dict,
    weights: Path | None,
    cb_pool: CardPool | None,
) -> Agent:
    """Build an agent; a ``net`` with a weights path loads the trained BC net."""
    if name == "net" and weights is not None:
        return NetAgent(deck, engine, weights=weights, cb_pool=cb_pool)
    return build_agent(name, deck, engine)


def read_deck(path: Path) -> list[int]:
    deck = [int(x) for x in path.read_text().split() if x.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck must be 60 cards, got {len(deck)} from {path}")
    return deck


def load_engine_data() -> dict:
    """Engine-derived card/attack stats injected into agents (see src.agents).

    Heuristic agents read these; greedy uses only the per-attack damage. Kept as
    plain ints/dicts so the agents never need to import the (Linux-only) engine.
    """
    attacks = {
        a.attackId: {"dmg": int(a.damage), "cost": [int(e) for e in a.energies]}
        for a in all_attack()
    }
    cards = {
        c.cardId: {
            "hp": int(c.hp),
            "retreat": int(c.retreatCost),
            "type": int(c.energyType),
            "weak": None if c.weakness is None else int(c.weakness),
            "ex": bool(c.ex),
            "mega": bool(c.megaEx),
            "basic": bool(c.basic),
            "ctype": int(c.cardType),
            "attacks": list(c.attacks),
        }
        for c in all_card_data()
    }
    return {"attacks": attacks, "cards": cards}


def play_game(
    agent_p0: Agent,
    agent_p1: Agent,
    *,
    a_is_player0: bool,
    seed: int,
    recorder: Recorder | None = None,
) -> GameResult:
    """Play one game; agents are addressed by slot via ``current.yourIndex``.

    When a ``recorder`` is given, each applied decision is reported via
    ``on_decision(slot, obs, choice)`` (the recorder must deep-copy ``obs`` --
    ``battle_select`` reuses the dict) and the winner via ``on_end`` at the end.
    """
    agent_p0.reset(seed)
    agent_p1.reset(seed)
    deck0 = agent_p0(DECK_REQUEST)
    deck1 = agent_p1(DECK_REQUEST)

    wall0 = time.perf_counter()
    obs, start = battle_start(deck0, deck1)
    if obs is None:
        raise RuntimeError(f"battle failed to start: errorType={start.errorType}")

    agents = (agent_p0, agent_p1)
    times = [0.0, 0.0]
    moves = [0, 0]
    max_move = [0.0, 0.0]
    selections = 0
    winner = ABORTED

    while True:
        cur = obs["current"]
        if cur is not None and cur.get("result", -1) != -1:
            winner = cur["result"]
            break
        yidx = 0 if cur is None else int(cur.get("yourIndex", 0))
        select = obs["select"]

        t0 = time.perf_counter()
        try:
            choice = agents[yidx](obs)
        except Exception:  # noqa: BLE001 - submission hygiene: never crash a match
            choice = None
        dt = time.perf_counter() - t0

        if not is_legal(choice, select):
            choice = legal_fallback(select)

        times[yidx] += dt
        moves[yidx] += 1
        max_move[yidx] = max(max_move[yidx], dt)

        if recorder is not None:
            recorder.on_decision(yidx, obs, choice)

        obs = battle_select(choice)
        selections += 1
        if selections >= MAX_SELECTIONS:
            winner = ABORTED
            break

    wall = time.perf_counter() - wall0
    turns = obs["current"]["turn"] if obs["current"] is not None else -1
    battle_finish()
    if recorder is not None:
        recorder.on_end(winner)

    # Re-attribute slot-indexed stats to A / B.
    order = (0, 1) if a_is_player0 else (1, 0)
    a, b = order
    return GameResult(
        a_is_player0=a_is_player0,
        winner=winner,
        turns=turns,
        selections=selections,
        agent_time_a=times[a],
        agent_time_b=times[b],
        moves_a=moves[a],
        moves_b=moves[b],
        max_move_a=max_move[a],
        max_move_b=max_move[b],
        wall_s=wall,
    )


def run_match(  # noqa: PLR0913 - a CLI runner legitimately threads its config
    name_a: str,
    name_b: str,
    deck_a: list[int],
    deck_b: list[int],
    engine: dict,
    *,
    games: int,
    base_seed: int,
    swap: bool,
    progress_every: int,
    a_weights: Path | None = None,
    b_weights: Path | None = None,
    cb_pool: CardPool | None = None,
) -> list[GameResult]:
    agent_a = _make_agent(name_a, deck_a, engine, a_weights, cb_pool)
    agent_b = _make_agent(name_b, deck_b, engine, b_weights, cb_pool)

    results: list[GameResult] = []
    for g in range(games):
        seed = base_seed + g
        a_is_player0 = not (swap and g % 2 == 1)
        if a_is_player0:
            res = play_game(agent_a, agent_b, a_is_player0=True, seed=seed)
        else:
            res = play_game(agent_b, agent_a, a_is_player0=False, seed=seed)
        results.append(res)
        if progress_every and (g + 1) % progress_every == 0:
            wins = sum(r.a_won for r in results)
            dec = sum(r.a_won or r.b_won for r in results)
            rate = wins / dec if dec else 0.0
            print(f"  [{g + 1}/{games}] {name_a} winrate so far: {rate:.3f}")
    return results


def write_logs(
    results: list[GameResult], summary: dict, out_dir: Path, tag: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{tag}.csv"
    json_path = out_dir / f"{tag}.json"

    rows = [
        r.as_row(i, summary["base_seed"] + i, summary["agent_a"], summary["agent_b"])
        for i, r in enumerate(results)
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {csv_path} and {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pokemon TCG battle runner / eval")
    parser.add_argument("--a", required=True, help="subject agent name (registered)")
    parser.add_argument("--b", required=True, help="opponent agent name (registered)")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0, help="base seed; g uses seed+g")
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK, help="deck for both")
    parser.add_argument("--deck-a", type=Path, default=None, help="override deck for A")
    parser.add_argument("--deck-b", type=Path, default=None, help="override deck for B")
    parser.add_argument("--no-swap", action="store_true", help="disable slot swap")
    parser.add_argument("--out", type=Path, default=ROOT / "results", help="log dir")
    parser.add_argument("--tag", default=None, help="output filename stem")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--a-weights", type=Path, default=None, help="net A weights")
    parser.add_argument("--b-weights", type=Path, default=None, help="net B weights")
    parser.add_argument(
        "--cb", action="store_true", help="net builds its deck from the CB head",
    )
    args = parser.parse_args()

    deck_a = read_deck(args.deck_a or args.deck)
    deck_b = read_deck(args.deck_b or args.deck)
    engine = load_engine_data()
    cb_pool = build_pool() if args.cb else None

    tag = args.tag or f"{args.a}_vs_{args.b}_n{args.games}_s{args.seed}"
    print(f"== {args.a} (A) vs {args.b} (B): {args.games} games, "
          f"swap={'off' if args.no_swap else 'on'}, base_seed={args.seed} ==")

    t0 = time.perf_counter()
    results = run_match(
        args.a, args.b, deck_a, deck_b, engine,
        games=args.games, base_seed=args.seed, swap=not args.no_swap,
        progress_every=args.progress_every,
        a_weights=args.a_weights, b_weights=args.b_weights, cb_pool=cb_pool,
    )
    elapsed = time.perf_counter() - t0

    summary = summarize(results, args.a, args.b)
    summary["base_seed"] = args.seed
    summary["swap"] = not args.no_swap
    summary["total_wall_s"] = elapsed

    print("\n== summary ==")
    print(json.dumps(summary, indent=2))
    lo, hi = summary["a_winrate_ci95"]
    print(f"\n{args.a} win rate: {summary['a_winrate']:.3f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]  -> {summary['verdict']}  "
          f"({summary['decisive']} decisive, {summary['draws']} draws, "
          f"{summary['aborted']} aborted)")

    write_logs(results, summary, args.out, tag)


if __name__ == "__main__":
    main()
