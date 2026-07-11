"""Collect AlphaZero targets from net-guided SO-ISMCTS self-play.

The recurrent net guides ISMCTS (value-head leaves, lever (1)) piloting ``--deck`` vs a
diverse opponent pool (tempo pilots x decks). Every searched single-select logs the root
visit-count policy ``pi``; the game OUTCOME (+/-1 from the agent's perspective) is the
value target ``z`` for all of that game's decisions -- the AlphaZero (s, pi, z) tuples.

  uv run python scripts/collect_ismcts.py --deck qdgp_best --games 400 \
      --iterations 64 --opp-pilots greedy,greedy_plus,heuristic --out data/az/r0.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.run_eval import (  # noqa: E402
    load_engine_data,
    play_game,
    read_deck,
    resolve_deck,
)
from src.agents import build_agent  # noqa: E402

NET = ROOT / "data/qdcoevo/run7/round_6/rl/paper_final.npz"
_G: dict = {}


def _resolve(name: str) -> list[int]:
    return resolve_deck(name)


def _init(deck_name: str, net_path: str, opp_decks: list[str] | None) -> None:
    from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: PLC0415

    _G["engine"] = load_engine_data()
    _G["deck"] = _resolve(deck_name)
    _G["net"] = RecurrentPolicyValueNet.load(net_path)
    cards = _G["engine"]["cards"]
    _G["basics"] = [c for c in _G["deck"] if cards.get(c) and cards[c].get("basic")]
    meta = [read_deck(p) for p in sorted((ROOT / "decklists").glob("*.csv"))]
    _G["meta_prior"] = [c for d in meta for c in d]
    # belief hypothesis decks (meta + anchors) for opponent-deck determinization.
    _G["hyp"] = meta + [read_deck(p)
                        for p in sorted((ROOT / "decklists" / "anchors").glob("*.csv"))]
    _G["opp_pool"] = ([(nm, _resolve(nm)) for nm in opp_decks]
                      if opp_decks else [(deck_name, _G["deck"])])


def _play(task: dict) -> dict:
    from src.search.ismcts import RecurrentIsmctsAgent  # noqa: PLC0415

    deck, engine = _G["deck"], _G["engine"]
    agent = RecurrentIsmctsAgent(
        deck, engine, _G["net"], opp_prior=_G["meta_prior"], opp_basics=_G["basics"],
        opp_decks=_G["hyp"], iterations=task["iterations"], rollout_cap=0,
        move_budget_s=task["budget"], gate_margin=task["gate"], record=True,
        cb_pool=None, build_deck_from_net=False, temperature=0.0)
    game_opp = task["opp_pilots"][task["seed"] % len(task["opp_pilots"])]
    opp_name, opp_deck = _G["opp_pool"][task["seed"] % len(_G["opp_pool"])]
    opp = build_agent(game_opp, opp_deck, engine)
    sf = task["subj_first"]
    p0, p1 = (agent, opp) if sf else (opp, agent)
    res = play_game(p0, p1, a_is_player0=sf, seed=task["seed"])
    z = 1.0 if res.a_won else (-1.0 if res.b_won else 0.0)   # AZ value target
    return {"z": z, "deck": deck, "searches": agent.n_searches,
            "overrode": agent.searched, "opp": game_opp, "opp_deck": opp_name,
            "moves": agent.move_log}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="qdgp_best")
    ap.add_argument("--net", default=str(NET))
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--iterations", type=int, default=64)
    ap.add_argument("--budget", type=float, default=999.0)
    ap.add_argument("--gate", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--opp-pilots", default="greedy,greedy_plus,heuristic")
    ap.add_argument("--opp-decks", default="")
    ap.add_argument("--out", type=Path, default=ROOT / "data/az/r0.jsonl")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    opp_pilots = [p for p in args.opp_pilots.split(",") if p]
    opp_decks = [d for d in args.opp_decks.split(",") if d]
    tasks = [{"subj_first": k % 2 == 0, "seed": 5000 + k, "iterations": args.iterations,
              "budget": args.budget, "gate": args.gate, "opp_pilots": opp_pilots}
             for k in range(args.games)]
    print(f"ISMCTS collecting {args.games} games on {args.deck} "
          f"(iters={args.iterations}) -> {args.out}", flush=True)
    n_g = n_pi = n_ovr = 0
    with args.out.open("w") as fh, Pool(
        args.workers, initializer=_init,
        initargs=(args.deck, args.net, opp_decks or None),
    ) as pp:
        for i, rec in enumerate(pp.imap_unordered(_play, tasks), 1):
            fh.write(json.dumps(rec) + "\n")
            n_g += 1
            n_pi += len(rec["moves"])
            n_ovr += rec["overrode"]
            if i % 25 == 0:
                print(f"  {i}/{args.games}  {n_pi} pi-targets  "
                      f"{n_ovr / n_g:.1f} overrides/game", flush=True)
    print(f"== done: {n_g} games, {n_pi} pi-targets -> {args.out} ==")


if __name__ == "__main__":
    main()
