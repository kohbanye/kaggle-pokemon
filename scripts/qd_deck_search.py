"""MAP-Elites deck search piloted by the recurrent net -> archive JSON.

Stage 2+3 of "QD decks + RL play": illuminate the deck space with MAP-Elites. A
candidate deck's **fitness** is its win rate over a fixed gauntlet (the meta decks),
both sides piloted by the same recurrent play net -- so fitness measures the *deck*
(given a fixed pilot), and unplayable decks lose and are discarded without any
hand-coded constraint. The archive keeps the best deck per ``(colour, energy-bin)``
niche, so coverage is diverse by construction.

With ``--rounds > 1`` (Step 4) the opponent gauntlet **co-evolves**: each round it is
rebuilt from the archive's top elites plus a hall-of-fame sample (meta decks + past
round bests, the anti-cycling memory), and the archive warm-starts across rounds
(decks carry over, re-scored against the new gauntlet).

  uv run python scripts/qd_deck_search.py --generations 40    # full (parallel)
  uv run python scripts/qd_deck_search.py --rounds 5 --generations 20   # Step 4 coevo
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
from src.agents import build_agent  # noqa: E402
from src.agents.recurrent_agent import RecurrentNetAgent  # noqa: E402
from src.deck import build_pool  # noqa: E402
from src.net.deck_sample import sample_deck_with_logp  # noqa: E402
from src.net.features import CardFeatures, load_engine_json  # noqa: E402
from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: E402
from src.qd import (  # noqa: E402
    DeckFeaturizer,
    HallOfFame,
    MapElitesArchive,
    RidgeSurrogate,
    behaviour_descriptor,
    build_gauntlet,
    colour_count,
    deck_stats,
    mutate,
    ramp_ids,
    random_legal_deck,
    random_legal_deck_biased,
    select_children,
    single_prize_ids,
)

PILOT = ROOT / "data/paperosfp/main/paper_final.npz"
ENGINE_JSON = ROOT / "data/bc/engine.json"
_G: dict = {}


def _init(pilot: str, gauntlet: list[list[int]], pilots: list[str]) -> None:
    _G["engine"] = load_engine_data()
    _G["pool"] = build_pool()
    _G["pilots"] = pilots
    if "net" in pilots:
        _G["net"] = RecurrentPolicyValueNet.load(pilot)
    _G["gauntlet"] = gauntlet


def _make_pilot(kind: str, deck: list[int]) -> object:
    """One pilot agent for ``deck``: the recurrent net, or a greedy/heuristic agent."""
    if kind == "net":
        return RecurrentNetAgent(
            deck, _G["engine"], net=_G["net"], cb_pool=_G["pool"],
            build_deck_from_net=False, temperature=0.0,
        )
    return build_agent(kind, deck, _G["engine"])


def _fitness(task: dict) -> dict:
    """Multi-pilot deck fitness: per (pilot, opponent) winrate cell -> mean + variance.

    Each pilot drives BOTH sides of its cell (fitness measures the *deck* given a fixed
    pilot). ``mean`` is the lexicographic primary; low ``var`` across cells -- strong no
    matter who pilots it / who it faces -- is the secondary tie-break.
    """
    cand = task["deck"]
    cells: list[float] = []
    per_pilot: dict[str, float] = {}
    for pi, pilot in enumerate(_G["pilots"]):
        rates: list[float] = []
        for gi, opp in enumerate(_G["gauntlet"]):
            wins = dec = 0
            for k in range(task["n_games"]):
                cand_first = k % 2 == 0
                a, b = _make_pilot(pilot, cand), _make_pilot(pilot, opp)
                p0, p1 = (a, b) if cand_first else (b, a)
                res = play_game(p0, p1, a_is_player0=cand_first,
                                seed=task["seed"] + pi * 100_000 + gi * 1000 + k)
                dec += res.a_won or res.b_won
                wins += res.a_won
            rates.append(wins / dec if dec else 0.0)
        per_pilot[pilot] = round(sum(rates) / len(rates), 3)
        cells.extend(rates)
    mean = sum(cells) / len(cells)
    var = sum((c - mean) ** 2 for c in cells) / len(cells)  # population variance
    return {"idx": task["idx"], "mean": mean, "var": var, "per_pilot": per_pilot}


def _evaluate(pp: Pool, decks: list[list[int]], n_games: int, seed: int) -> list[dict]:
    tasks = [{"idx": i, "deck": d, "n_games": n_games, "seed": seed + i * 100}
             for i, d in enumerate(decks)]
    out: list[dict] = [{} for _ in decks]
    for r in pp.map(_fitness, tasks):
        out[r["idx"]] = r
    return out


def _dump(  # noqa: PLR0913 - checkpoint writer bundling every run artefact
    out: Path, pilot: str, history: list[dict], arc: MapElitesArchive,
    config: dict | None = None, hof: HallOfFame | None = None,
    rounds: list[dict] | None = None,
) -> None:
    """Atomically write the archive JSON (temp file + rename) so a timeout-kill mid-run
    still leaves the latest complete checkpoint -- this is called every generation.

    ``config`` records the run's invocation (stringified argv) so a checkpoint is
    self-describing -- earlier runs (qd_step1*.json) didn't, and their flags are now
    unrecoverable.
    """
    results = {
        "pilot": pilot,
        "config": config or {},
        "history": history,
        "cells": [{"descriptor": list(e.descriptor), "fitness": round(e.fitness, 3),
                   "deck": e.deck, "stats": e.meta} for e in arc.elites()],
    }
    if hof is not None:
        results["hof"] = [{"tag": h.tag, "deck": h.deck} for h in hof.entries]
    if rounds is not None:
        results["rounds"] = rounds
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(results))
    tmp.replace(out)


def _build_seeds(  # noqa: PLR0913 - distinct seed inputs, not a bundle
    pool: object, pilot_net: object, feats: object, gauntlet: list[list[int]],
    n_init: int, rng: np.random.Generator,
) -> list[list[int]]:
    """Initial archive seeds: meta decks + CB-head samples + a 3-way random split.

    The random half is uniform / single-prize-biased / ramp-biased, because uniform
    random can't reach the descriptor's "exclusion" niches (a pure single-prize deck,
    or a deck whose cheapest attacker is expensive) -- the empty cells the
    (prize, speed) descriptor exposes. Re-seeded every co-evo round (fresh archive), so
    RL keeps seeing the diverse archetypes. Fitness still decides what survives.
    """
    seeds = [*gauntlet]
    seeds += [sample_deck_with_logp(pilot_net, pool, feats, rng)[0]
              for _ in range(n_init // 2)]
    n_rand = n_init - n_init // 2
    n_bias = n_rand // 3  # each of single-prize / ramp gets a third of the random half
    sp_ids, rmp_ids = single_prize_ids(pool), ramp_ids(pool)
    seeds += [random_legal_deck(pool, rng) for _ in range(n_rand - 2 * n_bias)]
    seeds += [random_legal_deck_biased(pool, rng, sp_ids) for _ in range(n_bias)]
    seeds += [random_legal_deck_biased(pool, rng, rmp_ids) for _ in range(n_bias)]
    return seeds


def main() -> None:  # noqa: PLR0915, C901 - CLI driver accumulating ablation arms
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
    ap.add_argument(
        "--pilots", type=str, default="greedy,net,heuristic",
        help="comma list of pilot kinds for each candidate (greedy,net,heuristic); "
             "fitness = mean winrate over pilots x opponents, tie-broken by low var",
    )
    ap.add_argument(
        "--eps", type=float, default=0.03,
        help="winrate band width for the lexicographic primary (round(winrate/eps)); "
             "ties within a band are broken by lower cross-pilot/opponent variance",
    )
    ap.add_argument(
        "--mutation", choices=("heuristic", "random"), default="heuristic",
        help="mutation operator: 'heuristic' (Step 3 role/package/energy-aware) or "
             "'random' (Step 1 uniform-swap baseline, for the A/B)",
    )
    ap.add_argument(
        "--surrogate", action="store_true",
        help="Step 2: pre-screen oversampled children with an online ridge "
             "winrate surrogate; same real-battle budget, spent on the "
             "most promising children (off = Step 3 baseline, for the A/B)",
    )
    ap.add_argument(
        "--oversample", type=int, default=4,
        help="children generated per real-eval slot when --surrogate is on",
    )
    ap.add_argument(
        "--explore-frac", type=float, default=0.25,
        help="fraction of each real-eval batch reserved for surrogate-blind "
             "random children (calibration / anti-bias)",
    )
    ap.add_argument(
        "--rounds", type=int, default=1,
        help="Step 4 co-evolution rounds: 1 = fixed meta gauntlet (Step 3 "
             "behaviour); >1 rebuilds the opponent gauntlet each round from the "
             "archive's top elites + a hall-of-fame sample, warm-starting the "
             "archive (decks carry over, re-scored vs the new gauntlet)",
    )
    ap.add_argument(
        "--coevo-top-k", type=int, default=4,
        help="archive elites (one per niche, best first) promoted into each "
             "co-evolution round's gauntlet",
    )
    ap.add_argument(
        "--hof-size", type=int, default=16,
        help="hall-of-fame capacity (seeded with the meta decks; each round's "
             "best deck is added, oldest evicted) -- the anti-cycling memory",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results/qd_archive.json")
    args = ap.parse_args()
    if args.quick:
        args.init, args.generations, args.batch, args.n_games = 16, 5, 8, 3
    pilots = [p.strip() for p in args.pilots.split(",") if p.strip()]

    pool = build_pool()
    feats = CardFeatures(load_engine_json(ENGINE_JSON))
    pilot_net = RecurrentPolicyValueNet.load(args.pilot)
    rng = np.random.default_rng(args.seed)
    meta_paths = sorted((ROOT / "decklists").glob("*.csv"))
    gauntlet = [read_deck(p) for p in meta_paths]
    arc = MapElitesArchive()

    seeds = _build_seeds(pool, pilot_net, feats, gauntlet, args.init, rng)

    def admit(decks: list[list[int]], results: list[dict]) -> tuple[int, int]:
        """Insert candidates; return (admitted, sensible). "sensible" = legal AND
        non-negative colour-penalised fitness (the sensible-deck-rate numerator).
        """
        n = n_sensible = 0
        for d, r in zip(decks, results, strict=True):
            nc = colour_count(d, pool)
            mean, var = r["mean"], r["var"]
            f_pen = mean - args.colour_penalty * nc  # colour penalty on the primary
            n_sensible += int(f_pen >= 0)
            key = (round(f_pen / args.eps), -var)  # lexicographic: band, then low var
            bd = behaviour_descriptor(d, pool)
            meta = {**deck_stats(d, pool), "winrate": round(mean, 3),
                    "variance": round(var, 4), "per_pilot": r["per_pilot"],
                    "colours": nc}
            if arc.insert(d, f_pen, bd, meta=meta, key=key):
                n += 1
        return n, n_sensible

    featurizer = DeckFeaturizer(pool, feats.vector)
    sur = RidgeSurrogate(featurizer) if args.surrogate else None

    def screen(cands: list[list[int]]) -> tuple[list[list[int]], list[float] | None]:
        """Surrogate pre-screen: pick the real-eval batch from the oversampled pool.

        Selection score matches admit's primary (predicted winrate minus the exact,
        deck-derived colour penalty). Returns the chosen children and their raw
        winrate predictions (for calibration logging), or all/None pre-warm-up.
        """
        if sur is None or not sur.ready:
            return cands[: args.batch], None
        raw = sur.predict(cands)
        adj = raw - args.colour_penalty * np.array(
            [colour_count(d, pool) for d in cands])
        idx = select_children(adj, args.batch, args.explore_frac, rng)
        return [cands[i] for i in idx], [float(raw[i]) for i in idx]

    def learn(decks: list[list[int]], results: list[dict],
              preds: list[float] | None) -> dict:
        """Feed real evals back into the surrogate; return calibration stats."""
        if sur is None:
            return {}
        for d, r in zip(decks, results, strict=True):
            sur.add(d, r["mean"])
        sur.fit()
        if preds is None:
            return {}
        p, m = np.array(preds), np.array([r["mean"] for r in results])
        cal = {"cal_mae": round(float(np.abs(p - m).mean()), 3)}
        if p.std() > 0 and m.std() > 0:
            cal["cal_r"] = round(float(np.corrcoef(p, m)[0, 1]), 3)
        return cal

    config = {k: str(v) for k, v in vars(args).items()}
    history: list[dict] = []
    hof = HallOfFame(args.hof_size)
    for p, d in zip(meta_paths, gauntlet, strict=True):
        hof.add(d, p.stem)
    seed_gen = iter(range(10_000, 10_000_000, 1000))
    gen_no = 0
    rounds_log: list[dict] = []
    for rnd in range(1, args.rounds + 1):
        if rnd == 1:
            opp, opp_tags = gauntlet, [p.stem for p in meta_paths]
        else:
            opp, opp_tags = build_gauntlet(arc, hof, len(gauntlet),
                                           args.coevo_top_k, rng)
        rounds_log.append({"round": rnd, "gauntlet": opp_tags})
        if args.rounds > 1:
            print(f"round {rnd}: gauntlet = {', '.join(opp_tags)}", flush=True)
        with Pool(args.workers, initializer=_init,
                  initargs=(str(args.pilot), opp, pilots)) as pp:
            if rnd == 1:
                seed_fits = _evaluate(pp, seeds, args.n_games, next(seed_gen))
                admit(seeds, seed_fits)
                learn(seeds, seed_fits, None)  # seed evals warm the surrogate up
                print(f"init: coverage={arc.coverage} best={arc.best().fitness:.3f}")
            else:
                # Warm-start: fitness is gauntlet-relative, so carry the DECKS
                # over and re-score them against the new opponents from scratch.
                carried = [e.deck for e in arc.elites()]
                arc = MapElitesArchive()
                refits = _evaluate(pp, carried, args.n_games, next(seed_gen))
                admit(carried, refits)
                if sur is not None:  # surrogate labels are gauntlet-relative too
                    sur = RidgeSurrogate(featurizer)
                learn(carried, refits, None)
                print(f"round {rnd} re-score: coverage={arc.coverage} "
                      f"best={arc.best().fitness:.3f}", flush=True)
            for _ in range(args.generations):
                gen_no += 1
                n_cand = args.batch * (args.oversample if sur is not None else 1)
                cands = [mutate(arc.sample(rng).deck, pool, rng, args.n_swaps,
                                strategy=args.mutation)
                         for _ in range(n_cand)]
                children, preds = screen(cands)
                fits = _evaluate(pp, children, args.n_games, next(seed_gen))
                n_adm, n_sensible = admit(children, fits)
                cal = learn(children, fits, preds)
                best = arc.best()
                history.append({"gen": gen_no, "round": rnd,
                                "coverage": arc.coverage,
                                "best": round(best.fitness, 3),
                                "mean": round(arc.mean_fitness(), 3),
                                "admitted": n_adm,
                                "sensible": n_sensible, **cal})
                cal_s = f" cal_r={cal['cal_r']}" if "cal_r" in cal else ""
                print(f"gen {gen_no:>3}: coverage={arc.coverage:>2} "
                      f"admitted={n_adm:>2} sensible={n_sensible:>2}/{args.batch} "
                      f"best={best.fitness:.3f} mean={arc.mean_fitness():.3f}{cal_s}",
                      flush=True)
                _dump(args.out, str(args.pilot), history, arc, config, hof,
                      rounds_log)
        hof.add(arc.best().deck, f"r{rnd}_best")

    _dump(args.out, str(args.pilot), history, arc, config, hof, rounds_log)
    best = arc.best()
    print(f"== done: coverage={arc.coverage} best={best.fitness:.3f} "
          f"deck={deck_stats(best.deck, pool)} -> {args.out} ==")


if __name__ == "__main__":
    main()
