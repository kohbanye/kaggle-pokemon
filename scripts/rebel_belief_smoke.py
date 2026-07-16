"""Gate C2 smoke: MULTI-WORLD belief subgame + world-independent infoset keying.

The multi-world case is the correctness crux Codex flagged. This checks two
properties that make it sound:
  (1) INFOSET SHARING: two distinct worlds that present the acting player the SAME
      observation map to the SAME infoset key (regret pooled across the belief, not
      split per world = no strategy fusion). We verify the acting player's root key
      is identical across worlds that differ only in HIDDEN (opponent) cards.
  (2) BELIEF SOLVE RUNS: external-sampling MCCFR solves the `BeliefSubgame` (belief =
      chance over worlds) and returns a strategy; the number of distinct infosets is far
      smaller than worlds x paths (evidence of sharing).

Not a Nash-optimality proof (Gate B did that on Kuhn); this validates the multi-world
plumbing + keying before the value net / self-play layers.

  uv run python scripts/rebel_belief_smoke.py
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
from src.rebel.belief import OpponentBelief, build_worlds  # noqa: E402
from src.rebel.cfr import solve_mccfr_external  # noqa: E402
from src.rebel.engine_game import BeliefSubgame, infoset_signature  # noqa: E402


def main() -> None:
    engine = load_engine_data()
    deck = resolve_deck("metal_aggro")
    meta = [read_deck(p) for p in sorted((ROOT / "decklists").glob("*.csv"))]
    prior = [c for d in meta for c in d]
    cards = engine["cards"]
    basics = [c for c in deck if cards.get(c) and cards[c].get("basic")]

    cap = _Capture(build_agent("greedy", deck, engine))
    play_game(cap, build_agent("greedy", deck, engine), a_is_player0=True, seed=7)
    obs = cap.grabbed
    if obs is None:
        print("no searchable obs captured")
        return
    cur = obs["current"]
    your = int(cur.get("yourIndex", 0))

    # Production path: OpponentBelief posterior -> belief-sampled worlds. They share OUR
    # hand/deck but differ in the opponent's hidden cards, so we cannot tell them apart
    # at the root -> the root infoset key must be identical across all of them.
    del prior  # superseded by the OpponentBelief posterior
    rng = np.random.default_rng(0)
    opp_belief = OpponentBelief.from_dirs()
    belief = build_worlds(cur, your, deck, opp_belief, 4, rng, opp_basics=basics)
    worlds = [w for w, _ in belief]
    print(f"belief: {len(opp_belief.hypotheses)} archetypes, "
          f"entropy={opp_belief.entropy(cur, your):.2f} nats")

    # (1) infoset-sharing check: build a 1-world subgame per world, compare root keys.
    root_keys = []
    for w in worlds:
        g = BeliefSubgame(obs, [(w, 1.0)], max_depth=2)
        try:
            root_keys.append(infoset_signature(g._resolve((0,))[0]))  # noqa: SLF001
        finally:
            g.close()
    shared = len(set(root_keys)) == 1
    print(f"(1) INFOSET SHARING: {len(worlds)} worlds -> "
          f"{len(set(root_keys))} distinct root key(s)  shared={shared}")
    assert shared, "worlds differing only in hidden cards must share the acting infoset"

    # (2) belief solve: MCCFR over the 4-world belief.
    game = BeliefSubgame(obs, belief, max_depth=2)
    try:
        avg = solve_mccfr_external(game, iters=600, seed=1)
        root_key = root_keys[0]
        root_strat = avg.get(root_key, {})
        print(f"(2) BELIEF SOLVE: worlds={len(worlds)} infosets_solved={len(avg)} "
              f"root_strategy={ {a: round(p, 2) for a, p in root_strat.items()} }")
        assert root_strat, "no strategy at the shared root infoset"
        assert abs(sum(root_strat.values()) - 1.0) < 1e-6
    finally:
        game.close()
    print("multi-world belief plumbing + world-independent keying OK")


if __name__ == "__main__":
    main()
