"""Play-quality diagnostics across co-evo checkpoints -> results/play_diag.json.

Isolates PLAY from DECK: every subject pilots the SAME fixed deck (the ladder-best
metal_aggro), so win-rate and behaviour differences are pure play, not deck choice.
For each checkpoint we record:

  (a) win rate vs a *greedy* opponent on a panel of DIVERSE opponent decks. Greedy-on-
      metal (mirror) approximates the in-distribution / training-anchor signal; greedy
      on the other type decks is the closest local proxy for the unknown-deck ladder.
      A widening gap (mirror high, off-deck low) as training proceeds = overfitting.

  (b) per-decision behavioural metrics via a recorder (action mix, first-attack turn,
      bench development, pass rate, prizes), so "how the net plays" can be compared
      to greedy (the develop-then-attack reference) and across training rounds.

Native/Docker (imports cg). Run:
  uv run python scripts/play_diag.py --games 24
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

from scripts.run_eval import load_engine_data, play_game, read_deck  # noqa: E402
from src.agents import build_agent  # noqa: E402
from src.agents.base import OPT_ATTACH, OPT_ATTACK, OPT_END, SEL_MAIN  # noqa: E402
from src.agents.recurrent_agent import RecurrentNetAgent  # noqa: E402
from src.deck import build_pool  # noqa: E402
from src.harness.stats import wilson_interval  # noqa: E402

# label -> (kind, checkpoint-path|None).  "net" loads the recurrent net; greedy/random
# are the heuristic references.  The checkpoint list is the training trajectory.
SUBJECTS: dict[str, tuple[str, str | None]] = {
    "greedy(ref)": ("greedy", None),
    "init(pre-coevo)": ("net", "data/paperosfp/main/paper_final.npz"),
    "run3_r6(ladder451)": ("net", "data/qdcoevo/run3/round_6/rl/paper_final.npz"),
    "run7_r1": ("net", "data/qdcoevo/run7/round_1/rl/paper_final.npz"),
    "run7_r2": ("net", "data/qdcoevo/run7/round_2/rl/paper_final.npz"),
    "run7_r3": ("net", "data/qdcoevo/run7/round_3/rl/paper_final.npz"),
    "run7_r4": ("net", "data/qdcoevo/run7/round_4/rl/paper_final.npz"),
    "run7_r5": ("net", "data/qdcoevo/run7/round_5/rl/paper_final.npz"),
    "run7_r6(submitted)": ("net", "data/qdcoevo/run7/round_6/rl/paper_final.npz"),
}

# Opponent decks (greedy pilots each).  metal == subject's deck (mirror, in-dist); the
# rest are the diverse out-of-distribution panel.  random-on-metal is a sanity floor.
OPP_DECKS = [
    "metal_aggro", "grass_aggro", "fire_aggro", "water_aggro",
    "lightning_aggro", "psychic_aggro", "fighting_aggro", "darkness_aggro",
]
SUBJECT_DECK = "metal_aggro"

# action-type code -> short label (for the MAIN-context behaviour histogram)
ACT_LABEL = {7: "play", 8: "attach", 9: "evolve", 10: "ability",
             11: "discard", 12: "retreat", 13: "attack", 14: "end"}

_G: dict = {}


def _ckpt(rel: str) -> Path:
    return ROOT / rel


def _init(net_paths: dict[str, str]) -> None:
    from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: PLC0415

    _G["engine"] = load_engine_data()
    _G["pool"] = build_pool()
    _G["nets"] = {k: RecurrentPolicyValueNet.load(v) for k, v in net_paths.items()}
    _G["decks"] = {nm: read_deck(ROOT / "decklists" / f"{nm}.csv") for nm in OPP_DECKS}


def _subject_agent(label: str, deck: list[int]) -> object:
    kind, _ = SUBJECTS[label]
    if kind == "net":
        return RecurrentNetAgent(deck, _G["engine"], net=_G["nets"][label],
                                 cb_pool=_G["pool"], build_deck_from_net=False,
                                 temperature=0.0)
    return build_agent(kind, deck, _G["engine"])


class _Rec:
    """Per-game behaviour recorder for ONE subject slot."""

    def __init__(self, slot: int) -> None:
        self.slot = slot
        self.acts: dict[int, int] = {}
        self.n_main = 0
        self.n_pass = 0           # chose END while other actions were legal
        self.attach = 0
        self.first_attack_turn: int | None = None
        self.max_turn = 0
        self.max_bench = 0
        self.bench_t3: int | None = None       # bench size around the 3rd own turn
        self.min_own_prize = 6
        self.min_opp_prize = 6

    def on_decision(self, slot: int, obs: dict, choice: list[int]) -> None:
        if slot != self.slot:
            return
        cur = obs.get("current") or {}
        sel = obs.get("select") or {}
        turn = int(cur.get("turn", 0))
        self.max_turn = max(self.max_turn, turn)
        players = cur.get("players") or []
        if len(players) == 2 and choice:
            me, opp = players[slot], players[1 - slot]
            bench = len(me.get("bench") or [])
            self.max_bench = max(self.max_bench, bench)
            if turn >= 5 and self.bench_t3 is None:  # ~3rd turn for this player
                self.bench_t3 = bench
            self.min_own_prize = min(self.min_own_prize, len(me.get("prize") or []))
            self.min_opp_prize = min(self.min_opp_prize, len(opp.get("prize") or []))
        if int(sel.get("type", -1)) != SEL_MAIN or not choice:
            return
        opts = sel.get("option") or []
        idx = choice[0]
        if idx >= len(opts):
            return
        t = int(opts[idx].get("type", -1))
        self.acts[t] = self.acts.get(t, 0) + 1
        self.n_main += 1
        if t == OPT_END and len(opts) > 1:
            self.n_pass += 1
        if t == OPT_ATTACH:
            self.attach += 1
        if t == OPT_ATTACK and self.first_attack_turn is None:
            self.first_attack_turn = turn

    def on_end(self, winner: int) -> None:
        pass


def _play(task: dict) -> dict:
    deck = _G["decks"][SUBJECT_DECK]
    subj = _subject_agent(task["subject"], deck)
    opp_kind = task["opp_kind"]
    opp_deck = _G["decks"][task["opp_deck"]]
    opp = build_agent(opp_kind, opp_deck, _G["engine"])
    subj_first = task["subj_first"]
    subj_slot = 0 if subj_first else 1
    rec = _Rec(subj_slot)
    p0, p1 = (subj, opp) if subj_first else (opp, subj)
    res = play_game(p0, p1, a_is_player0=subj_first, seed=task["seed"], recorder=rec)
    return {
        "subject": task["subject"], "opp": task["opp_label"],
        "won": int(res.a_won), "dec": int(res.a_won or res.b_won),
        "turns": rec.max_turn, "first_attack_turn": rec.first_attack_turn,
        "n_main": rec.n_main, "n_pass": rec.n_pass, "attach": rec.attach,
        "max_bench": rec.max_bench, "bench_t3": rec.bench_t3,
        "prizes_taken": 6 - rec.min_own_prize, "prizes_lost": 6 - rec.min_opp_prize,
        "acts": rec.acts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Play diagnostics across checkpoints")
    ap.add_argument("--games", type=int, default=24, help="games per subject/opponent")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=Path, default=ROOT / "results/play_diag.json")
    args = ap.parse_args()

    net_paths = {lbl: str(_ckpt(p)) for lbl, (k, p) in SUBJECTS.items()
                 if k == "net" and p is not None}

    # opponents: greedy on each deck + a random-on-metal floor
    opps = [{"opp_kind": "greedy", "opp_deck": d,
             "opp_label": f"greedy/{d.split('_')[0]}"} for d in OPP_DECKS]
    opps.append({"opp_kind": "random", "opp_deck": SUBJECT_DECK,
                 "opp_label": "random/metal"})

    tasks = [
        {**o, "subject": s, "subj_first": k % 2 == 0,
         "seed": (si * 131 + oi) * 1000 + k}
        for si, s in enumerate(SUBJECTS)
        for oi, o in enumerate(opps)
        for k in range(args.games)
    ]
    print(f"subjects={len(SUBJECTS)} opponents={len(opps)} "
          f"games/pair={args.games} total={len(tasks)}")

    with Pool(args.workers, initializer=_init, initargs=(net_paths,)) as pp:
        rows = pp.map(_play, tasks)

    # aggregate per (subject, opponent) and per subject
    by_pair: dict[tuple[str, str], list[dict]] = {}
    by_subj: dict[str, list[dict]] = {}
    for r in rows:
        by_pair.setdefault((r["subject"], r["opp"]), []).append(r)
        by_subj.setdefault(r["subject"], []).append(r)

    def wr(rs: list[dict]) -> dict:
        w, d = sum(x["won"] for x in rs), sum(x["dec"] for x in rs)
        p, lo, hi = wilson_interval(w, d)
        return {"winrate": round(p, 3), "ci": [round(lo, 3), round(hi, 3)], "n": d}

    per_opp = {s: {} for s in SUBJECTS}
    for (s, o), rs in by_pair.items():
        per_opp[s][o] = wr(rs)

    # behaviour: aggregate over ALL games for the subject (deck fixed, so stable)
    behaviour = {}
    for s, rs in by_subj.items():
        acts: dict[int, int] = {}
        for r in rs:
            for t, c in r["acts"].items():
                acts[int(t)] = acts.get(int(t), 0) + c
        tot = sum(acts.values()) or 1
        fa = [r["first_attack_turn"] for r in rs if r["first_attack_turn"] is not None]
        bt3 = [r["bench_t3"] for r in rs if r["bench_t3"] is not None]
        behaviour[s] = {
            "act_mix": {ACT_LABEL.get(t, str(t)): round(c / tot, 3)
                        for t, c in sorted(acts.items())},
            "avg_turns": round(sum(r["turns"] for r in rs) / len(rs), 1),
            "first_attack_turn": round(sum(fa) / len(fa), 2) if fa else None,
            "pct_no_attack_game": round(1 - len(fa) / len(rs), 3),
            "pass_rate": round(sum(r["n_pass"] for r in rs)
                               / max(sum(r["n_main"] for r in rs), 1), 3),
            "attach_per_game": round(sum(r["attach"] for r in rs) / len(rs), 2),
            "max_bench": round(sum(r["max_bench"] for r in rs) / len(rs), 2),
            "bench_by_turn3": round(sum(bt3) / len(bt3), 2) if bt3 else None,
            "prizes_taken": round(sum(r["prizes_taken"] for r in rs) / len(rs), 2),
            "prizes_lost": round(sum(r["prizes_lost"] for r in rs) / len(rs), 2),
        }

    # the OOD gap: mirror (greedy/metal) vs mean over the off-deck panel
    summary = {}
    for s in SUBJECTS:
        mirror = per_opp[s].get("greedy/metal", {}).get("winrate")
        offdeck = [per_opp[s][o]["winrate"] for o in per_opp[s]
                   if o.startswith("greedy/") and o != "greedy/metal"]
        offmean = round(sum(offdeck) / len(offdeck), 3) if offdeck else None
        summary[s] = {
            "vs_greedy_mirror": mirror, "vs_offdeck_mean": offmean,
            "gap": round(mirror - offmean, 3)
            if (mirror is not None and offmean is not None) else None,
            "vs_all": wr(by_subj[s]),
        }

    out = {"subject_deck": SUBJECT_DECK, "games_per_pair": args.games,
           "summary": summary, "per_opponent": per_opp, "behaviour": behaviour}
    args.out.write_text(json.dumps(out, indent=2))
    print(f"-> {args.out}")
    for s in SUBJECTS:
        sm = summary[s]
        print(f"  {s:<22} mirror={sm['vs_greedy_mirror']}  "
              f"offdeck={sm['vs_offdeck_mean']}  gap={sm['gap']}")


if __name__ == "__main__":
    main()
