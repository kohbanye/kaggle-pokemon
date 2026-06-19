"""Probe the cg engine's observation shape and determinism (Linux/Docker only).

Throwaway investigation used to design the battle runner:
  * What does the first observation from ``battle_start`` look like
    (is ``current`` None? is ``select`` present? what is ``yourIndex``)?
  * How does turn/selection routing work across the opening selections?
  * Is the engine RNG deterministic across runs (no seed is exposed)?

Run inside Docker (see README): ``python scripts/probe_obs.py``.
"""

import random
import sys
from pathlib import Path

CG_PARENT = Path(__file__).resolve().parent.parent / "data" / "sample_submission"
sys.path.insert(0, str(CG_PARENT))

from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402


def read_deck() -> list[int]:
    text = (CG_PARENT / "deck.csv").read_text()
    return [int(x) for x in text.split() if x.strip()]


def random_select(obs_dict: dict, deck: list[int]) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return deck
    return random.sample(range(len(obs.select.option)), obs.select.maxCount)


def play_once(deck: list[int], log_opening: int = 0) -> tuple[int, int]:
    """Play one random game; return (winner, n_selections)."""
    obs, start = battle_start(deck, list(deck))
    if obs is None:
        raise SystemExit(f"battle failed to start: errorType={start.errorType}")

    if log_opening:
        print(f"  first obs: current is None? {obs['current'] is None}; "
              f"select is None? {obs['select'] is None}; "
              f"top-level keys={sorted(obs.keys())}")

    steps = 0
    while obs["current"] is None or obs["current"].get("result", -1) == -1:
        if log_opening and steps < log_opening:
            cur = obs["current"]
            sel = obs["select"]
            yidx = None if cur is None else cur.get("yourIndex")
            stype = None if sel is None else sel.get("type")
            sctx = None if sel is None else sel.get("context")
            nopt = None if sel is None else len(sel.get("option", []))
            mn = None if sel is None else sel.get("minCount")
            mx = None if sel is None else sel.get("maxCount")
            print(f"  step {steps:>3}: yourIndex={yidx} selType={stype} "
                  f"ctx={sctx} nOptions={nopt} min={mn} max={mx}")
        obs = battle_select(random_select(obs, deck))
        steps += 1
        if steps > 5000:
            raise SystemExit("did not terminate in 5000 steps")

    winner = obs["current"]["result"]
    battle_finish()
    return winner, steps


def main() -> None:
    deck = read_deck()
    assert len(deck) == 60

    print("== opening structure (one game) ==")
    play_once(deck, log_opening=12)

    print("\n== determinism check: same fixed agent seed, two runs ==")
    results = []
    for run in range(2):
        random.seed(1234)
        winner, steps = play_once(deck)
        results.append((winner, steps))
        print(f"  run {run}: winner={winner} steps={steps}")
    print(f"  identical across runs? {results[0] == results[1]}")

    print("\n== variability check: different seeds ==")
    for seed in (1, 2, 3):
        random.seed(seed)
        winner, steps = play_once(deck)
        print(f"  seed {seed}: winner={winner} steps={steps}")


if __name__ == "__main__":
    main()
