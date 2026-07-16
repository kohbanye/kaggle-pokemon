"""Does ReBeL actually PLAY differently from greedy_plus? (diagnose the win-rate pin.)

Two different value nets gave an IDENTICAL win-rate under CFR+ serving -> hypothesis:
the served (temperature-0) ARGMAX of the converged ReBeL solve ~= greedy_plus's move, so
ReBeL is greedy_plus by construction and no solver/value change can move win-rate. This
measures, over real games, the AGREEMENT rate: of the decisions where ReBeL genuinely
solved, how often its move == greedy_plus's move on the same obs. High agreement => the
lever is action candidates greedy_plus never considers (#9) or a gp-EXPLOITER (#11), not
the solver/value.

  uv run python scripts/rebel_agree.py --games 6 --sweeps 16 --net NET.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.run_eval import load_engine_data, play_game, resolve_deck  # noqa: E402
from src.agents import build_agent  # noqa: E402
from src.rebel.agent import RebelAgent  # noqa: E402

SINGLE_SELECT = 1
_MIN_CHOICE = 2


class _Agree(RebelAgent):
    """RebelAgent logging, per searched decision, agreement with greedy_plus."""

    def __init__(self, *a: object, **k: object) -> None:
        super().__init__(*a, **k)
        self.n_total = 0
        self.n_searchable = 0
        self.n_rebel = 0      # decisions where ReBeL actually produced a move
        self.n_agree = 0      # of those, move == greedy_plus's move
        self.n_multi = 0      # multi-select (auto-abstracted, ReBeL never searches)

    def act(self, obs: dict) -> list[int]:
        self.n_total += 1
        select = obs.get("select") or {}
        cur = obs.get("current")
        options = select.get("option") or []
        maxc = int(select.get("maxCount", 0))
        if maxc > SINGLE_SELECT:
            self.n_multi += 1
        single = maxc == SINGLE_SELECT and cur is not None
        searchable = (single and len(options) >= _MIN_CHOICE
                      and bool(obs.get("search_begin_input")))
        before = self.solved
        move = super().act(obs)
        if searchable:
            self.n_searchable += 1
            if self.solved > before:  # ReBeL genuinely produced this move
                self.n_rebel += 1
                try:
                    gp = self._fallback.act(obs)
                except Exception:  # noqa: BLE001
                    gp = None
                if gp is not None and list(move) == list(gp):
                    self.n_agree += 1
        return move


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="metal_aggro")
    ap.add_argument("--net", default="data/rebel/vncfrp_metal_r2.npz")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--sweeps", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--particles", type=int, default=4)
    ap.add_argument("--tpw", type=int, default=4)
    args = ap.parse_args()

    engine = load_engine_data()
    deck = resolve_deck(args.deck)
    from src.rebel.value_net import ValueNet  # noqa: PLC0415
    net = ValueNet.load(str(ROOT / args.net))

    tot = {"total": 0, "searchable": 0, "rebel": 0, "agree": 0, "multi": 0}
    for g in range(args.games):
        agent = _Agree(deck, engine, n_particles=args.particles, max_depth=args.depth,
                       sweeps=args.sweeps, traversals_per_world=args.tpw, seed=g,
                       value_net=net)
        opp = build_agent("greedy", deck, engine)
        sf = g % 2 == 0
        p0, p1 = (agent, opp) if sf else (opp, agent)
        play_game(p0, p1, a_is_player0=sf, seed=5000 + g)
        tot["total"] += agent.n_total
        tot["searchable"] += agent.n_searchable
        tot["rebel"] += agent.n_rebel
        tot["agree"] += agent.n_agree
        tot["multi"] += agent.n_multi
        ag = agent.n_agree / agent.n_rebel if agent.n_rebel else 0.0
        print(f"  game {g}: decisions={agent.n_total} searchable={agent.n_searchable} "
              f"rebel-solved={agent.n_rebel} multi={agent.n_multi} agree={ag:.2f}",
              flush=True)

    agree_rate = tot["agree"] / tot["rebel"] if tot["rebel"] else 0.0
    print("\n=== ReBeL vs greedy_plus MOVE AGREEMENT ===")
    print(f"  total decisions       : {tot['total']}")
    print(f"  multi-select (abstr)  : {tot['multi']} "
          f"({tot['multi']/max(tot['total'],1):.0%} of decisions ReBeL never searches)")
    print(f"  searchable            : {tot['searchable']}")
    print(f"  ReBeL genuinely solved: {tot['rebel']} "
          f"({tot['rebel']/max(tot['total'],1):.0%} of all decisions)")
    print(f"  move == greedy_plus   : {tot['agree']}/{tot['rebel']} = {agree_rate:.1%}")
    print("\nREAD: if agreement is high AND ReBeL-solved is a small % of decisions, "
          "ReBeL ~= greedy_plus -> win-rate cannot move via solver/value; lever "
          "is action candidates gp never plays (#9) or a gp-exploiter objective (#11).")


if __name__ == "__main__":
    main()
