"""Probe the Phase-3 net agent against the real engine (Linux x86-64 / Docker).

Validates the Phase-3 exit criteria that need the engine (the rest are covered
natively by ``tests/test_net.py``):

  1. the CB head's decks are accepted by ``battle_start`` (always-legal 60 cards);
  2. ``NetAgent`` never raises and never returns an illegal selection across a
     full game (crash-free option-index round-trip);
  3. per-move CPU inference time leaves ample margin under the turn budget.

  docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim \
      python scripts/probe_net.py
"""

import argparse
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # make `src` importable
CG_PARENT = ROOT / "data" / "sample_submission"
sys.path.insert(0, str(CG_PARENT))  # make `cg` importable
sys.path.insert(0, str(ROOT / "scripts"))  # reuse run_eval helpers

import numpy as np  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from run_eval import (  # noqa: E402
    DECK_REQUEST,
    DEFAULT_DECK,
    load_engine_data,
    read_deck,
)

from src.agents import build_agent  # noqa: E402
from src.agents.base import Agent, is_legal, legal_fallback  # noqa: E402
from src.agents.net_agent import NetAgent  # noqa: E402
from src.deck import CardPool, build_pool, load_deck_csv  # noqa: E402
from src.deck import is_legal as deck_is_legal  # noqa: E402
from src.net.cb import build_deck  # noqa: E402
from src.net.features import CardFeatures  # noqa: E402
from src.net.model import PolicyValueNet  # noqa: E402

MAX_SELECTIONS = 5000


def report_cb_overlap(net: PolicyValueNet, pool: CardPool, feats: CardFeatures) -> None:
    """Print the CB greedy deck's multiset overlap with each demo decklist."""
    print("\n== CB head: demo-deck overlap (Phase-4 CB BC sanity) ==")
    deck = build_deck(net, pool, feats)
    deck_counts = Counter(deck)
    for path in sorted((ROOT / "decklists").glob("*.csv")):
        demo = load_deck_csv(path)
        inter = sum((deck_counts & Counter(demo)).values())
        print(f"  {path.stem:>16}: overlap {inter}/{len(demo)} = {inter / 60:.2f}")


def validate_cb(
    net: PolicyValueNet, pool: CardPool, feats: CardFeatures, opponent: list[int],
) -> bool:
    """Build CB decks and confirm the engine accepts each (errorType 0)."""
    print("\n== CB head: deck legality (local validator + engine battle_start) ==")
    rng = np.random.default_rng(0)
    decks = {
        "greedy": build_deck(net, pool, feats),
        "sampled-1": build_deck(net, pool, feats, rng, greedy=False),
        "sampled-2": build_deck(net, pool, feats, rng, greedy=False),
    }
    ok = True
    for label, deck in decks.items():
        local = deck_is_legal(deck, pool)
        obs, start = battle_start(deck, opponent)
        accepted = obs is not None
        if accepted:
            battle_finish()
        ok = ok and local and accepted
        print(
            f"  [{label}] len={len(deck)} distinct_names={len(set(deck))} "
            f"local_legal={local} engine_accepted={accepted} "
            f"errorType={start.errorType}",
        )
    return ok


def play_timed(agent_p0: Agent, agent_p1: Agent) -> dict | None:
    """Play one game; count agent crashes / illegal returns and time each move."""
    obs, _ = battle_start(agent_p0(DECK_REQUEST), agent_p1(DECK_REQUEST))
    if obs is None:
        return None
    agents = (agent_p0, agent_p1)
    crashes = illegal = moves = selections = 0
    total_t = max_t = 0.0
    winner = -1
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
        except Exception:  # noqa: BLE001 - the probe must finish the game
            choice, crashes = None, crashes + 1
        dt = time.perf_counter() - t0
        total_t, max_t, moves = total_t + dt, max(max_t, dt), moves + 1
        if not is_legal(choice, select):
            illegal += 1
            choice = legal_fallback(select)
        obs = battle_select(choice)
        selections += 1
        if selections >= MAX_SELECTIONS:
            break
    battle_finish()
    return {
        "selections": selections, "crashes": crashes, "illegal": illegal,
        "avg_ms": 1000.0 * total_t / max(moves, 1), "max_ms": 1000.0 * max_t,
        "winner": winner,
    }


def run_games(
    label: str,
    make0: Callable[[], Agent],
    make1: Callable[[], Agent],
    games: int,
) -> dict:
    """Play ``games`` matches, aggregating crash / illegal / timing stats."""
    agg = {"crashes": 0, "illegal": 0, "avg_ms": 0.0, "max_ms": 0.0}
    print(f"\n== {label}: {games} games ==")
    for g in range(games):
        res = play_timed(make0(), make1())
        if res is None:
            print(f"  game {g}: battle failed to start")
            continue
        agg["crashes"] += res["crashes"]
        agg["illegal"] += res["illegal"]
        agg["avg_ms"] = max(agg["avg_ms"], res["avg_ms"])
        agg["max_ms"] = max(agg["max_ms"], res["max_ms"])
        print(
            f"  game {g}: selections={res['selections']} winner={res['winner']} "
            f"crashes={res['crashes']} illegal={res['illegal']} "
            f"avg={res['avg_ms']:.3f}ms max={res['max_ms']:.3f}ms",
        )
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the net agent in-engine")
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="trained .npz to load (default: a random-init net)",
    )
    args = parser.parse_args()

    engine = load_engine_data()
    feats = CardFeatures(engine)
    pool = build_pool()
    net = (
        PolicyValueNet.load(args.weights)
        if args.weights is not None
        else PolicyValueNet.random(np.random.default_rng(0))
    )
    src = args.weights if args.weights is not None else "random-init"
    print(f"net params: {net.param_count()}  pool: {len(pool.cards)} cards  ({src})")

    sample = read_deck(DEFAULT_DECK)
    cb_ok = validate_cb(net, pool, feats, sample)
    report_cb_overlap(net, pool, feats)

    deck = read_deck(ROOT / "decklists" / "metal_aggro.csv")

    def make_net() -> NetAgent:
        return NetAgent(deck, engine, net=net)

    vs_greedy = run_games(
        "net vs greedy", make_net, lambda: build_agent("greedy", deck, engine), 4,
    )
    vs_net = run_games("net vs net", make_net, make_net, 2)

    crashes = vs_greedy["crashes"] + vs_net["crashes"]
    illegal = vs_greedy["illegal"] + vs_net["illegal"]
    worst_max = max(vs_greedy["max_ms"], vs_net["max_ms"])
    print("\n== summary ==")
    print(f"  CB decks accepted: {cb_ok}")
    print(f"  total crashes: {crashes}  total illegal returns: {illegal}")
    print(f"  worst per-move time: {worst_max:.3f} ms")
    verdict = "PASS" if cb_ok and crashes == 0 and illegal == 0 else "FAIL"
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
