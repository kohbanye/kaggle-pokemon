"""MAP-Elites deck search piloted by the recurrent net -> archive JSON.

Stage 2+3 of "QD decks + RL play": illuminate the deck space with MAP-Elites. A
candidate deck's **fitness** is its win rate over a fixed gauntlet (the meta decks),
both sides piloted by the same recurrent play net -- so fitness measures the *deck*
(given a fixed pilot), and unplayable decks lose and are discarded without any
hand-coded constraint. The archive keeps the best deck per ``(colour, energy-bin)``
niche, so coverage is diverse by construction.

  uv run python scripts/qd_deck_search.py --generations 40    # full (parallel)
  uv run python scripts/qd_deck_search.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.run_eval import load_engine_data, play_game, read_deck  # noqa: E402
from src.agents.recurrent_agent import RecurrentNetAgent  # noqa: E402
from src.deck import build_pool  # noqa: E402
from src.net.deck_sample import sample_deck_with_logp  # noqa: E402
from src.net.features import CardFeatures, load_engine_json  # noqa: E402
from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: E402
from src.qd import (  # noqa: E402
    MapElitesArchive,
    behaviour_descriptor,
    colour_count,
    deck_stats,
    mutate,
    random_legal_deck,
)

PILOT = ROOT / "data/paperosfp/main/paper_final.npz"
ENGINE_JSON = ROOT / "data/bc/engine.json"
_G: dict = {}


def _init(pilot: str, gauntlet: list[list[int]]) -> None:
    _G["engine"] = load_engine_data()
    _G["pool"] = build_pool()
    _G["net"] = RecurrentPolicyValueNet.load(pilot)
    _G["gauntlet"] = gauntlet


def _agent(deck: list[int]) -> RecurrentNetAgent:
    return RecurrentNetAgent(
        deck, _G["engine"], net=_G["net"], cb_pool=_G["pool"],
        build_deck_from_net=False, temperature=0.0,
    )


def _fitness(task: dict) -> dict:
    """Win rate of one candidate deck over the gauntlet (net pilots both sides)."""
    cand = task["deck"]
    wins = dec = 0
    for gi, opp in enumerate(_G["gauntlet"]):
        for k in range(task["n_games"]):
            cand_first = k % 2 == 0
            a, b = _agent(cand), _agent(opp)
            p0, p1 = (a, b) if cand_first else (b, a)
            res = play_game(p0, p1, a_is_player0=cand_first,
                            seed=task["seed"] + gi * 1000 + k)
            dec += res.a_won or res.b_won
            wins += res.a_won
    return {"idx": task["idx"], "fitness": wins / dec if dec else 0.0}


def _evaluate(pp: Pool, decks: list[list[int]], n_games: int, seed: int) -> list[float]:
    tasks = [{"idx": i, "deck": d, "n_games": n_games, "seed": seed + i * 100}
             for i, d in enumerate(decks)]
    out = [0.0] * len(decks)
    for r in pp.map(_fitness, tasks):
        out[r["idx"]] = r["fitness"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="MAP-Elites deck search")
    ap.add_argument("--pilot", type=Path, default=PILOT)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument(
        "--init", type=int, default=64,
        help="extra seed decks beyond the metas (half CB-sampled, half random)",
    )
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--batch", type=int, default=24, help="children per generation")
    ap.add_argument(
        "--n-games", type=int, default=6, help="games vs each gauntlet deck",
    )
    ap.add_argument("--n-swaps", type=int, default=4)
    ap.add_argument(
        "--colour-penalty", "--color-penalty", type=float, default=0.03,
        dest="colour_penalty",
        help="soft penalty subtracted per distinct coloured Pokemon type: "
             "fitness = winrate - penalty * n_colours (0 disables; biases the "
             "archive toward fewer-colour decks without forbidding any)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results/qd_archive.json")
    args = ap.parse_args()
    if args.quick:
        args.init, args.generations, args.batch, args.n_games = 16, 5, 8, 3

    pool = build_pool()
    feats = CardFeatures(load_engine_json(ENGINE_JSON))
    pilot_net = RecurrentPolicyValueNet.load(args.pilot)
    rng = np.random.default_rng(args.seed)
    gauntlet = [read_deck(p) for p in sorted((ROOT / "decklists").glob("*.csv"))]
    arc = MapElitesArchive()

    # Seed the archive with *functional* starting points -- the meta decks and the
    # net's own CB-head samples (which carry energy) -- plus random decks. This is
    # not a hard constraint: it just puts decks across the energy axis in the archive
    # so mutation can explore energy-rich niches (random legal decks are energy-poor,
    # so from-random alone never reaches them). Fitness still decides what survives.
    seeds = [*gauntlet]
    seeds += [sample_deck_with_logp(pilot_net, pool, feats, rng)[0]
              for _ in range(args.init // 2)]
    seeds += [random_legal_deck(pool, rng) for _ in range(args.init // 2)]

    def admit(decks: list[list[int]], fits: list[float]) -> int:
        n = 0
        for d, f in zip(decks, fits, strict=True):
            nc = colour_count(d, pool)
            f_pen = f - args.colour_penalty * nc  # soft colour penalty
            bd = behaviour_descriptor(d, pool)
            meta = {**deck_stats(d, pool), "winrate": round(f, 3), "colours": nc}
            if arc.insert(d, f_pen, bd, meta=meta):
                n += 1
        return n

    history = []
    with Pool(args.workers, initializer=_init,
              initargs=(str(args.pilot), gauntlet)) as pp:
        seed_gen = iter(range(10_000, 10_000_000, 1000))
        admit(seeds, _evaluate(pp, seeds, args.n_games, next(seed_gen)))
        print(f"init: coverage={arc.coverage} best={arc.best().fitness:.3f}")
        for gen in range(1, args.generations + 1):
            children = [mutate(arc.sample(rng).deck, pool, rng, args.n_swaps)
                        for _ in range(args.batch)]
            fits = _evaluate(pp, children, args.n_games, next(seed_gen))
            n_adm = admit(children, fits)
            best = arc.best()
            history.append({"gen": gen, "coverage": arc.coverage,
                            "best": round(best.fitness, 3),
                            "mean": round(arc.mean_fitness(), 3), "admitted": n_adm})
            print(f"gen {gen:>3}: coverage={arc.coverage:>2} admitted={n_adm:>2} "
                  f"best={best.fitness:.3f} mean={arc.mean_fitness():.3f}")

    elites = arc.elites()
    results = {
        "pilot": str(args.pilot),
        "history": history,
        "cells": [{"descriptor": list(e.descriptor), "fitness": round(e.fitness, 3),
                   "deck": e.deck, "stats": e.meta} for e in elites],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results))
    best = arc.best()
    print(f"== done: coverage={arc.coverage} best={best.fitness:.3f} "
          f"deck={deck_stats(best.deck, pool)} -> {args.out} ==")


if __name__ == "__main__":
    main()
