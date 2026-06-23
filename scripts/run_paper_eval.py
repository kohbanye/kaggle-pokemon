"""Multi-faceted evaluation of the canonical recurrent paper net -> results JSON.

Runs the heavy in-engine matchups in parallel (one process per game, nets/engine
cached per worker) and writes ``results/paper_eval.json``, which
``scripts/build_eval_notebook.py`` turns into tables/plots. Dimensions:

- head-to-head full agent (own deck + play) vs Phase-5d;
- play-skill on a shared fixed deck vs greedy / random (isolates the play head);
- checkpoint progression vs a fixed reference (the learning curve, by head-to-head);
- gauntlet: the net's deck vs each meta archetype (greedy-piloted);
- deck composition + sampled-deck diversity; per-move inference cost.

  uv run python scripts/run_paper_eval.py            # full (~1900 games, parallel)
  uv run python scripts/run_paper_eval.py --quick    # small N (smoke)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.run_eval import load_engine_data, play_game, read_deck  # noqa: E402
from src.agents import build_agent  # noqa: E402
from src.agents.net_agent import NetAgent  # noqa: E402
from src.agents.recurrent_agent import RecurrentNetAgent  # noqa: E402
from src.deck import build_pool, card_kind  # noqa: E402
from src.harness.stats import wilson_interval  # noqa: E402
from src.net.cb import build_deck  # noqa: E402
from src.net.deck_sample import sample_deck_with_logp  # noqa: E402
from src.net.features import CardFeatures, load_engine_json  # noqa: E402
from src.net.model import PolicyValueNet  # noqa: E402
from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: E402

FINAL = ROOT / "data/paperosfp/main/paper_final.npz"
P5D = ROOT / "data/jointosfp/run2/jointiter_649.npz"
METAL = ROOT / "decklists/metal_aggro.csv"
ENGINE_JSON = ROOT / "data/bc/engine.json"
CKPT_ITERS = [50, 250, 500, 1000, 2000, 3500, 5000]

_G: dict = {}


def _init() -> None:
    _G["engine"] = load_engine_data()
    _G["pool"] = build_pool()
    _G["nets"] = {}


def _net(kind: str, path: str) -> object:
    cache = _G["nets"]
    if path not in cache:
        loader = RecurrentPolicyValueNet if kind == "recurrent" else PolicyValueNet
        cache[path] = loader.load(path)
    return cache[path]


def _agent(spec: dict, deck: list[int]) -> object:
    kind = spec["kind"]
    if kind in ("recurrent", "base"):
        cls = RecurrentNetAgent if kind == "recurrent" else NetAgent
        return cls(
            deck, _G["engine"], net=_net(kind, spec["w"]), cb_pool=_G["pool"],
            build_deck_from_net=False, temperature=0.0,
        )
    return build_agent(kind, deck, _G["engine"])


def _play(task: dict) -> dict:
    a = _agent(task["a"], task["da"])
    b = _agent(task["b"], task["db"])
    a_first = task["a_first"]
    p0, p1 = (a, b) if a_first else (b, a)
    res = play_game(p0, p1, a_is_player0=a_first, seed=task["seed"])
    return {
        "m": task["m"], "a_won": int(res.a_won),
        "dec": int(res.a_won or res.b_won), "turns": res.turns,
        "move_ms": (res.agent_time_a / max(res.moves_a, 1)) * 1000,
        "max_move_ms": res.max_move_a * 1000,
    }


def _rec(w: Path) -> dict:
    return {"kind": "recurrent", "w": str(w)}


def _base(w: Path) -> dict:
    return {"kind": "base", "w": str(w)}


def _scripted(name: str) -> dict:
    return {"kind": name, "w": ""}


def _tasks(matches: list[dict], n: int) -> list[dict]:
    """Expand each (label, A, deckA, B, deckB) match into ``n`` slot-swapped games."""
    return [
        {"m": mt["m"], "a": mt["a"], "da": mt["da"], "b": mt["b"], "db": mt["db"],
         "a_first": k % 2 == 0, "seed": mi * 100000 + k}
        for mi, mt in enumerate(matches)
        for k in range(n)
    ]


def _aggregate(rows: list[dict]) -> dict[str, dict]:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["m"], []).append(r)
    out = {}
    for m, rs in by.items():
        wins = sum(r["a_won"] for r in rs)
        dec = sum(r["dec"] for r in rs)
        p, lo, hi = wilson_interval(wins, dec)
        moves = [r["move_ms"] for r in rs]
        out[m] = {
            "winrate": p, "ci_lo": lo, "ci_hi": hi, "wins": wins,
            "decisive": dec, "games": len(rs),
            "avg_turns": sum(r["turns"] for r in rs) / len(rs),
            "avg_move_ms": sum(moves) / len(moves),
            "max_move_ms": max(r["max_move_ms"] for r in rs),
        }
    return out


def _deck_report(net: RecurrentPolicyValueNet, pool: object, feats: object) -> dict:
    """Greedy deck composition + sampled-deck diversity (no games)."""
    greedy = build_deck(net, pool, feats)
    comp = Counter(card_kind(pool, c) for c in greedy)
    rng = np.random.default_rng(0)
    samples = [sample_deck_with_logp(net, pool, feats, rng)[0] for _ in range(30)]
    return {
        "greedy_comp": dict(comp),
        "greedy_distinct": len(set(greedy)),
        "greedy_top": Counter(greedy).most_common(8),
        "sampled_distinct": [len(set(d)) for d in samples],
        "sampled_energy": [sum(card_kind(pool, c) == "energy" for c in d)
                           for d in samples],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-faceted recurrent net eval")
    ap.add_argument("--final", type=Path, default=FINAL)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--quick", action="store_true", help="small N smoke")
    ap.add_argument("--out", type=Path, default=ROOT / "results/paper_eval.json")
    args = ap.parse_args()

    n_h2h, n_skill, n_ckpt, n_gaunt = (20, 20, 12, 12) if args.quick else (
        300, 200, 80, 80)

    pool = build_pool()
    feats = CardFeatures(load_engine_json(ENGINE_JSON))
    final_net = RecurrentPolicyValueNet.load(args.final)

    # Precompute decks (each net's own greedy deck; the meta archetypes).
    net_deck = build_deck(final_net, pool, feats)
    p5d_deck = build_deck(PolicyValueNet.load(P5D), pool, feats)
    metal = read_deck(METAL)
    metas = {p.stem: read_deck(p) for p in sorted((ROOT / "decklists").glob("*.csv"))}
    ckpt_dir = ROOT / "data/paperosfp/main"
    ckpts = [(it, ckpt_dir / f"paperiter_{it}.npz") for it in CKPT_ITERS]
    ckpts = [(it, p) for it, p in ckpts if p.exists()]
    ckpt_decks = {it: build_deck(RecurrentPolicyValueNet.load(p), pool, feats)
                  for it, p in ckpts}

    rows: list[dict] = []
    with Pool(args.workers, initializer=_init) as pool_proc:
        # 1) full agent vs Phase-5d, 2) play-skill on a shared deck vs greedy/random
        head = [
            {"m": "vs_phase5d", "a": _rec(args.final), "da": net_deck,
             "b": _base(P5D), "db": p5d_deck},
            {"m": "vs_greedy_samedeck", "a": _rec(args.final), "da": metal,
             "b": _scripted("greedy"), "db": metal},
            {"m": "vs_random_samedeck", "a": _rec(args.final), "da": metal,
             "b": _scripted("random"), "db": metal},
        ]
        rows += pool_proc.map(_play, _tasks(head[:1], n_h2h))
        rows += pool_proc.map(_play, _tasks(head[1:], n_skill))
        # 3) checkpoint progression vs the fixed Phase-5d reference
        prog = [{"m": f"ckpt_{it}", "a": _rec(p), "da": ckpt_decks[it],
                 "b": _base(P5D), "db": p5d_deck} for it, p in ckpts]
        rows += pool_proc.map(_play, _tasks(prog, n_ckpt))
        # 4) gauntlet: the net's deck vs each meta archetype (greedy-piloted)
        gaunt = [{"m": f"gauntlet_{name}", "a": _rec(args.final), "da": net_deck,
                  "b": _scripted("greedy"), "db": deck}
                 for name, deck in metas.items()]
        rows += pool_proc.map(_play, _tasks(gaunt, n_gaunt))

    results = {
        "final": str(args.final),
        "phase5d": str(P5D),
        "matches": _aggregate(rows),
        "checkpoint_iters": [it for it, _ in ckpts],
        "deck": _deck_report(final_net, pool, feats),
        "engine": "native",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out} ({len(rows)} games)")
    for m, s in sorted(results["matches"].items()):
        print(f"  {m:28s} wr={s['winrate']:.3f} [{s['ci_lo']:.3f},{s['ci_hi']:.3f}] "
              f"n={s['decisive']}")


if __name__ == "__main__":
    main()
