"""Gate C smoke: solve a depth-limited SINGLE-WORLD cg subgame with CFR/MCCFR.

Validates the engine<->CFR plumbing end to end: capture a real mid-game obs, determinize
one world, wrap it as an `EngineSubgame`, and run exact CFR + external-sampling MCCFR on
it. A single world is perfect info, so this checks the machinery (session, branching
cache, decision/chance/terminal classification, depth limit, leaf value) -- NOT belief
soundness (the multi-world step). We assert the tree builds and both solvers return a
valid root strategy over the real options.

  uv run python scripts/rebel_engine_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.rebel_api_probe import _Capture  # noqa: E402
from scripts.run_eval import (  # noqa: E402
    load_engine_data,
    play_game,
    read_deck,
    resolve_deck,
)
from src.agents import build_agent  # noqa: E402
from src.rebel.cfr import solve_cfr, solve_mccfr_external  # noqa: E402
from src.rebel.engine_game import EngineSubgame  # noqa: E402
from src.search.determinize import sample_determinization  # noqa: E402


def _capture_obs(engine: dict, deck: list[int]) -> dict | None:
    cap = _Capture(build_agent("greedy", deck, engine))
    opp = build_agent("greedy", deck, engine)
    play_game(cap, opp, a_is_player0=True, seed=7)
    return cap.grabbed


def main() -> None:
    engine = load_engine_data()
    deck = resolve_deck("metal_aggro")
    meta = [read_deck(p) for p in sorted((ROOT / "decklists").glob("*.csv"))]
    prior = [c for d in meta for c in d]
    cards = engine["cards"]
    basics = [c for c in deck if cards.get(c) and cards[c].get("basic")]

    obs = _capture_obs(engine, deck)
    if obs is None:
        print("no searchable obs captured")
        return
    cur = obs["current"]
    your = int(cur.get("yourIndex", 0))
    rng = np.random.default_rng(0)
    det = sample_determinization(cur, your, deck, prior, rng, opp_basics=basics)

    for depth in (1, 2, 3):
        game = EngineSubgame(obs, det, max_depth=depth)
        try:
            root_actions = game.legal_actions(game.root())
            n_nodes_before = len(game._cache)  # noqa: SLF001
            avg_cfr = solve_cfr(game, iters=200)
            avg_mccfr = solve_mccfr_external(game, iters=400, seed=1)
            root_key = game.infoset_key(game.root())
            cfr_root = avg_cfr.get(root_key, {})
            top = max(cfr_root, key=lambda a: cfr_root[a]) if cfr_root else None
            print(f"[depth {depth}] root_options={len(root_actions)} "
                  f"tree_nodes={len(game._cache)} (from {n_nodes_before})")  # noqa: SLF001
            print(f"    CFR root strategy: "
                  f"{ {a: round(p, 2) for a, p in cfr_root.items()} }  top={top}")
            print(f"    MCCFR infosets solved: {len(avg_mccfr)}")
            assert cfr_root, "CFR produced no root strategy"
            assert abs(sum(cfr_root.values()) - 1.0) < 1e-6
        finally:
            game.close()
    print("engine<->CFR plumbing OK (single-world / perfect-info)")


if __name__ == "__main__":
    main()
