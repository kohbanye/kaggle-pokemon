"""Deck-evaluation harness: round-robin over decks (Linux x86-64 / Docker).

Fixes the agent policy and varies the *deck*: plays every pair of decks
head-to-head (slot-swapped) and reports a win-rate matrix + a marginal ranking
(each deck's mean win rate vs the field). This is the Phase 1 measurement tool --
it (a) quantifies whether deck choice moves win rate more than agent choice,
(b) provides the diverse opponent pool for OSFP self-play, and (c) anchors the
local<->ladder calibration.

Reuses the Phase-0 match driver in ``run_eval`` (deck eval == same agent on both
sides, different decks), so the engine-touching code lives in one place.

  docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim \
      python scripts/run_deck_eval.py --agent greedy \
          --deck data/sample_submission/deck.csv --random 3 --games 100
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # import sibling run_eval
CG_PARENT = ROOT / "data" / "sample_submission"
sys.path.insert(0, str(CG_PARENT))

import run_eval  # noqa: E402 - engine-touching match driver (Docker only)

from src.deck import build_pool, random_legal_deck  # noqa: E402
from src.harness.stats import summarize  # noqa: E402

DEFAULT_DECK = CG_PARENT / "deck.csv"


def load_decks(
    deck_paths: list[Path], deck_dir: Path | None, n_random: int, seed: int,
) -> tuple[list[str], list[list[int]]]:
    """Resolve the deck set: explicit files + a directory's *.csv + N random."""
    paths = list(deck_paths)
    if deck_dir is not None:
        paths.extend(sorted(deck_dir.glob("*.csv")))
    names: list[str] = []
    decks: list[list[int]] = []
    for path in paths:
        names.append(path.stem)
        decks.append(run_eval.read_deck(path))
    if n_random:
        pool = build_pool()
        rng = random.Random(seed)  # noqa: S311 - gameplay randomness, not crypto
        for k in range(n_random):
            names.append(f"rand{k}")
            decks.append(random_legal_deck(pool, rng))
    return names, decks


def round_robin(  # noqa: PLR0913 - a runner legitimately threads its config
    agent: str,
    names: list[str],
    decks: list[list[int]],
    engine: dict,
    *,
    games: int,
    seed: int,
    swap: bool,
) -> tuple[list[list[float | None]], list[dict]]:
    """Play every deck pair; return a win-rate matrix and per-pair summaries."""
    n = len(decks)
    matrix: list[list[float | None]] = [[None] * n for _ in range(n)]
    pairs: list[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            results = run_eval.run_match(
                agent, agent, decks[i], decks[j], engine,
                games=games, base_seed=seed, swap=swap, progress_every=0,
            )
            s = summarize(results, names[i], names[j])
            wr = s["a_winrate"]
            matrix[i][j] = wr
            matrix[j][i] = 1.0 - wr  # decisive-only complement
            pairs.append({
                "a": names[i], "b": names[j], "a_winrate": wr,
                "ci95": s["a_winrate_ci95"], "decisive": s["decisive"],
                "draws": s["draws"],
            })
            print(f"  {names[i]} vs {names[j]}: {wr:.3f} "
                  f"({s['decisive']} dec, {s['draws']} draw)")
    return matrix, pairs


def marginals(
    names: list[str], matrix: list[list[float | None]],
) -> list[tuple[str, float]]:
    """Each deck's mean win rate vs the rest of the field, sorted best first."""
    ranking: list[tuple[str, float]] = []
    for i, name in enumerate(names):
        vals = [v for j, v in enumerate(matrix[i]) if j != i and v is not None]
        ranking.append((name, sum(vals) / len(vals) if vals else 0.0))
    ranking.sort(key=lambda t: t[1], reverse=True)
    return ranking


def print_matrix(names: list[str], matrix: list[list[float | None]]) -> None:
    width = max(8, *(len(n) for n in names))
    header = " " * width + " | " + " ".join(f"{n[:6]:>6}" for n in names)
    print("\n" + header)
    for i, name in enumerate(names):
        cells = " ".join(
            "   -  " if matrix[i][j] is None else f"{matrix[i][j]:>6.3f}"
            for j in range(len(names))
        )
        print(f"{name[:width]:<{width}} | {cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin deck evaluation")
    parser.add_argument("--agent", default="greedy", help="fixed policy (registered)")
    parser.add_argument("--deck", type=Path, action="append", default=[],
                        help="deck file (repeatable)")
    parser.add_argument("--deck-dir", type=Path, default=None,
                        help="also load every *.csv in this directory")
    parser.add_argument("--random", type=int, default=0,
                        help="also add N random legal decks")
    parser.add_argument("--games", type=int, default=100, help="games per pair")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-swap", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    deck_paths = list(args.deck)
    n_random = args.random
    if not deck_paths and args.deck_dir is None and not n_random:
        deck_paths = [DEFAULT_DECK]  # default smoke: sample + 3 random
        n_random = 3

    names, decks = load_decks(deck_paths, args.deck_dir, n_random, args.seed)
    if len(decks) < 2:
        raise SystemExit("need >= 2 decks to run a round-robin")
    engine = run_eval.load_engine_data()

    print(f"== deck round-robin: agent={args.agent}, {len(decks)} decks, "
          f"{args.games} games/pair, swap={'off' if args.no_swap else 'on'} ==")
    print(f"   decks: {', '.join(names)}")

    t0 = time.perf_counter()
    matrix, pairs = round_robin(
        args.agent, names, decks, engine,
        games=args.games, seed=args.seed, swap=not args.no_swap,
    )
    elapsed = time.perf_counter() - t0

    print_matrix(names, matrix)
    ranking = marginals(names, matrix)
    print("\n== marginal win rate vs field (best first) ==")
    for name, wr in ranking:
        print(f"  {name:<16} {wr:.3f}")
    spread = ranking[0][1] - ranking[-1][1]
    print(f"\ndeck spread (best - worst marginal): {spread:.3f}  "
          f"(cf. agent差 greedy-vs-heuristic ~0.50 mirror) in {elapsed:.1f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"deckeval_{args.agent}_n{args.games}_s{args.seed}"
    summary = {
        "agent": args.agent, "games_per_pair": args.games, "base_seed": args.seed,
        "swap": not args.no_swap, "decks": names, "matrix": matrix,
        "pairs": pairs, "ranking": ranking, "spread": spread,
        "total_wall_s": elapsed,
    }
    (args.out / f"{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out / f'{tag}.json'}")


if __name__ == "__main__":
    main()
