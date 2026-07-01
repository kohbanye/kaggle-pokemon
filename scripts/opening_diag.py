"""Opening-play adherence vs Pokémon-TCG conventions -> results/opening_diag.json.

The ladder doesn't correlate with self-play / vs-greedy win rate, so this scores play
against *normative* opening principles (non-self-referential), focusing on the early
turns where correct play is near-scripted:

  - go-first rate           : engine asks IS_FIRST; convention = go first by default
                              (extra attach/draw/setup), second only for fast decks.
  - bench after turn 1 / 2  : develop the board with Basics early.
  - attach-every-turn rate  : attach your one energy each early turn (tempo).
  - attach-last rate        : energy attach is irreversible -> do it AFTER the turn's
                              draw/develop actions (attach should be the last develop).
  - wasted-turn rate        : fraction of own turns that did nothing but End.

Deck fixed to metal_aggro; subjects = greedy (reference) + the training trajectory.
Native/Docker (imports cg). Run:  uv run python scripts/opening_diag.py --games 40
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

from scripts.play_diag import _G, _init, _subject_agent  # noqa: E402
from scripts.play_diag import SUBJECTS as SUBJECTS  # noqa: E402, PLC0414
from scripts.run_eval import play_game  # noqa: E402
from src.agents import build_agent  # noqa: E402
from src.agents.base import (  # noqa: E402
    OPT_ATTACH,
    OPT_END,
    OPT_EVOLVE,
    OPT_PLAY,
    OPT_YES,
    SEL_MAIN,
)

OPT_ABILITY = 10
CTX_IS_FIRST = 41   # SelectContext.IS_FIRST (YesNo: would you like to go first?)
DEVELOP = {OPT_PLAY, OPT_EVOLVE, OPT_ABILITY}
SUBJECT_DECK = "metal_aggro"
OPP_DECKS = ["metal_aggro", "fire_aggro", "psychic_aggro"]


class _Rec:
    """Per-turn opening recorder for ONE subject slot."""

    def __init__(self, slot: int) -> None:
        self.slot = slot
        self.go_first: str | None = None
        self.turn_acts: dict[int, list[int]] = {}  # turn -> action types, in order
        self.bench_end: dict[int, int] = {}   # turn -> last-seen bench size

    def on_decision(self, slot: int, obs: dict, choice: list[int]) -> None:
        if slot != self.slot or not choice:
            return
        cur = obs.get("current") or {}
        sel = obs.get("select") or {}
        turn = int(cur.get("turn", 0))
        opts = sel.get("option") or []
        idx = choice[0]
        if idx >= len(opts):
            return
        otype = int(opts[idx].get("type", -1))
        ctx = int(sel.get("context", -1))  # SelectContext
        if ctx == CTX_IS_FIRST:
            self.go_first = "first" if otype == OPT_YES else "second"
        players = cur.get("players") or []
        if len(players) == 2:
            self.bench_end[turn] = len(players[slot].get("bench") or [])
        if int(sel.get("type", -1)) == SEL_MAIN:
            self.turn_acts.setdefault(turn, []).append(otype)

    def on_end(self, winner: int) -> None:
        _ = winner

    def summary(self) -> dict:
        my_turns = sorted(self.turn_acts)
        first2 = my_turns[:2]
        bench1 = self.bench_end.get(first2[0]) if first2 else None
        bench2 = self.bench_end.get(first2[1]) if len(first2) > 1 else None
        # consider the first 3 own turns for tempo/sequencing
        early = my_turns[:3]
        attach_turns = sum(OPT_ATTACH in self.turn_acts[t] for t in early)
        # attach-last: in attach+develop turns, did attach come after the last develop?
        seq_ok = seq_tot = 0
        for t in early:
            acts = self.turn_acts[t]
            if OPT_ATTACH in acts and any(a in DEVELOP for a in acts):
                seq_tot += 1
                last_attach = max(i for i, a in enumerate(acts) if a == OPT_ATTACH)
                last_dev = max(i for i, a in enumerate(acts) if a in DEVELOP)
                seq_ok += last_attach > last_dev
        wasted = sum(self.turn_acts[t] == [OPT_END] for t in early)
        return {
            "go_first": self.go_first, "bench_t1": bench1, "bench_t2": bench2,
            "attach_turns_first3": attach_turns, "early_turns": len(early),
            "seq_ok": seq_ok, "seq_tot": seq_tot, "wasted_first3": wasted,
        }


def _play(task: dict) -> dict:
    deck = _G["decks"][SUBJECT_DECK]
    subj = _subject_agent(task["subject"], deck)
    opp = build_agent("greedy", _G["decks"][task["opp"]], _G["engine"])
    subj_first = task["subj_first"]
    rec = _Rec(0 if subj_first else 1)
    p0, p1 = (subj, opp) if subj_first else (opp, subj)
    play_game(p0, p1, a_is_player0=subj_first, seed=task["seed"], recorder=rec)
    return {"subject": task["subject"], **rec.summary()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Opening-play adherence diagnostics")
    ap.add_argument("--games", type=int, default=40, help="games per subject/opponent")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=Path, default=ROOT / "results/opening_diag.json")
    args = ap.parse_args()

    net_paths = {lbl: str(ROOT / p) for lbl, (k, p) in SUBJECTS.items()
                 if k == "net" and p is not None}
    tasks = [
        {"subject": s, "opp": opp, "subj_first": k % 2 == 0,
         "seed": (si * 137 + oi) * 1000 + k}
        for si, s in enumerate(SUBJECTS)
        for oi, opp in enumerate(OPP_DECKS)
        for k in range(args.games)
    ]
    print(f"subjects={len(SUBJECTS)} opps={len(OPP_DECKS)} total={len(tasks)}")

    with Pool(args.workers, initializer=_init, initargs=(net_paths,)) as pp:
        rows = pp.map(_play, tasks)

    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["subject"], []).append(r)

    out = {}
    for s, rs in by.items():
        n = len(rs)
        gf = [r["go_first"] for r in rs if r["go_first"] is not None]
        b1 = [r["bench_t1"] for r in rs if r["bench_t1"] is not None]
        b2 = [r["bench_t2"] for r in rs if r["bench_t2"] is not None]
        att = sum(r["attach_turns_first3"] for r in rs)
        ear = sum(r["early_turns"] for r in rs)
        sok = sum(r["seq_ok"] for r in rs)
        stot = sum(r["seq_tot"] for r in rs)
        wst = sum(r["wasted_first3"] for r in rs)
        out[s] = {
            "go_first_rate":
                round(sum(g == "first" for g in gf) / len(gf), 3) if gf else None,
            "bench_after_t1": round(sum(b1) / len(b1), 2) if b1 else None,
            "bench_after_t2": round(sum(b2) / len(b2), 2) if b2 else None,
            "attach_per_early_turn": round(att / ear, 3) if ear else None,
            "attach_last_rate": round(sok / stot, 3) if stot else None,
            "wasted_turn_rate": round(wst / ear, 3) if ear else None,
            "games": n,
        }

    args.out.write_text(json.dumps(out, indent=2))
    print(f"-> {args.out}")
    keys = ["go_first_rate", "bench_after_t1", "bench_after_t2",
            "attach_per_early_turn", "attach_last_rate", "wasted_turn_rate"]
    print(f"{'subject':<22}" + "".join(f"{k[:11]:>13}" for k in keys))
    for s in SUBJECTS:
        print(f"{s:<22}" + "".join(f"{out[s][k]!s:>13}" for k in keys))


if __name__ == "__main__":
    main()
