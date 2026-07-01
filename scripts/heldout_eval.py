"""Generalization vs a HELD-OUT opponent pool -> results/heldout.json.

Step 0 metric ③ (``docs/deck-search-redesign.md`` §1.4): score our candidate strategies
against the ~30 ``decklists/heldout/*.csv`` decks the QD search / its fitness gauntlet /
the NashConv population never touch. Opponents are piloted three ways -- ``greedy``,
``heuristic`` and a forced-go-first ``greedy`` -- and none of these (deck, pilot) combos
appear in the QD gauntlet or ``nashconv_eval.py`` STRATS, so the set is genuinely
held-out: a ladder proxy and the NashConv exploiter bank.

Mirrors ``scripts/deck_strength.py`` (same ``play_game`` / ``wilson_interval`` /
multiprocessing / JSON schema); only the opponent set + opponent pilots differ.

Native/Docker (imports cg). Run:
  uv run python scripts/heldout_eval.py --games 30
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
from src.agents.base import OPT_YES  # noqa: E402
from src.agents.recurrent_agent import RecurrentNetAgent  # noqa: E402
from src.deck import build_pool  # noqa: E402
from src.harness.stats import wilson_interval  # noqa: E402

CTX_IS_FIRST = 41  # SelectContext.IS_FIRST (mirrors nashconv_eval.py)
NET = "data/qdcoevo/run7/round_6/rl/paper_final.npz"

# Subjects we evaluate: (label, deck name, pilot). "net" uses the run7 checkpoint.
SUBJECTS: list[tuple[str, str, str]] = [
    ("net_run7|metal", "metal_aggro", "net"),
    ("greedy|metal", "metal_aggro", "greedy"),
    ("net_run7|run7_best", "run7_best", "net"),
    ("greedy|run7_best", "run7_best", "greedy"),
    ("net_run7|grass", "grass_aggro", "net"),
]
# Opponent pilots (held-out configs: heuristic & go-first never appear in QD/NashConv).
OPP_PILOTS = ("greedy", "heuristic", "greedyFF")

_G: dict = {}


class _ForcedFirst:
    """Wrap an agent so it always chooses to go FIRST when the engine asks IS_FIRST."""

    def __init__(self, inner: object) -> None:
        self.inner = inner

    def reset(self, seed: int) -> None:
        self.inner.reset(seed)

    def __call__(self, obs: dict) -> list[int]:
        sel = obs.get("select")
        if sel is not None and int(sel.get("context", -1)) == CTX_IS_FIRST:
            for i, o in enumerate(sel.get("option") or []):
                if int(o.get("type", -1)) == OPT_YES:
                    return [i]
        return self.inner(obs)


def _init(heldout_names: list[str], subject_names: list[str]) -> None:
    from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: PLC0415

    _G["engine"] = load_engine_data()
    _G["pool"] = build_pool()
    _G["net"] = RecurrentPolicyValueNet.load(str(ROOT / NET))
    _G["decks"] = {nm: read_deck(ROOT / "decklists" / f"{nm}.csv")
                   for nm in subject_names}
    _G["decks"].update({nm: read_deck(ROOT / "decklists" / "heldout" / f"{nm}.csv")
                        for nm in heldout_names})


def _subject(deck_name: str, pilot: str) -> object:
    deck = _G["decks"][deck_name]
    if pilot == "net":
        return RecurrentNetAgent(deck, _G["engine"], net=_G["net"],
                                 cb_pool=_G["pool"], build_deck_from_net=False,
                                 temperature=0.0)
    return build_agent("greedy", deck, _G["engine"])


def _opponent(deck_name: str, pilot: str) -> object:
    deck = _G["decks"][deck_name]
    if pilot == "greedyFF":
        return _ForcedFirst(build_agent("greedy", deck, _G["engine"]))
    return build_agent(pilot, deck, _G["engine"])


def _play(task: dict) -> dict:
    subj = _subject(task["deck"], task["pilot"])
    opp = _opponent(task["opp"], task["opp_pilot"])
    subj_first = task["subj_first"]
    p0, p1 = (subj, opp) if subj_first else (opp, subj)
    res = play_game(p0, p1, a_is_player0=subj_first, seed=task["seed"])
    return {"subject": task["label"], "opp_pilot": task["opp_pilot"],
            "opp": task["opp"], "won": int(res.a_won),
            "dec": int(res.a_won or res.b_won)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out generalization eval")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=Path, default=ROOT / "results/heldout.json")
    args = ap.parse_args()

    heldout = sorted(p.stem for p in (ROOT / "decklists" / "heldout").glob("*.csv"))
    subject_names = sorted({deck for _, deck, _ in SUBJECTS})

    tasks = [
        {"label": label, "deck": deck, "pilot": pilot, "opp": opp,
         "opp_pilot": op, "subj_first": k % 2 == 0,
         "seed": (si * 911 + oi) * 1000 + pi * 137 + k}
        for si, (label, deck, pilot) in enumerate(SUBJECTS)
        for pi, op in enumerate(OPP_PILOTS)
        for oi, opp in enumerate(heldout)
        for k in range(args.games)
    ]
    print(f"subjects={len(SUBJECTS)} heldout={len(heldout)} "
          f"opp_pilots={len(OPP_PILOTS)} total={len(tasks)}")

    with Pool(args.workers, initializer=_init,
              initargs=(heldout, subject_names)) as pp:
        rows = pp.map(_play, tasks)

    by: dict[str, list[dict]] = {}                  # subject -> rows
    by_pilot: dict[tuple[str, str], list[dict]] = {}   # (subject, opp_pilot) -> rows
    by_opp: dict[tuple[str, str], list[dict]] = {}     # (subject, opp) -> rows
    for r in rows:
        by.setdefault(r["subject"], []).append(r)
        by_pilot.setdefault((r["subject"], r["opp_pilot"]), []).append(r)
        by_opp.setdefault((r["subject"], r["opp"]), []).append(r)

    def wr(rs: list[dict]) -> dict:
        w, d = sum(x["won"] for x in rs), sum(x["dec"] for x in rs)
        p, lo, hi = wilson_interval(w, d)
        return {"winrate": round(p, 3), "ci": [round(lo, 3), round(hi, 3)], "n": d}

    out = {"heldout": heldout, "opp_pilots": list(OPP_PILOTS),
           "games_per_pair": args.games, "overall": {}, "per_pilot": {},
           "per_opp": {}}
    for label, rs in by.items():
        out["overall"][label] = wr(rs)
    for (label, op), rs in by_pilot.items():
        out["per_pilot"].setdefault(label, {})[op] = wr(rs)["winrate"]
    for (label, opp), rs in by_opp.items():
        out["per_opp"].setdefault(label, {})[opp] = wr(rs)["winrate"]

    args.out.write_text(json.dumps(out, indent=2))
    print(f"-> {args.out}")
    for label, _, _ in SUBJECTS:
        o = out["overall"][label]
        pp_ = out["per_pilot"][label]
        print(f"  {label:<22} held-out-winrate={o['winrate']} CI{o['ci']} "
              f"n={o['n']}  by-pilot={pp_}")


if __name__ == "__main__":
    main()
