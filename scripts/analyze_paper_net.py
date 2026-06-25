"""Deeper multi-angle analysis of the recurrent paper net (-> JSON).

Beyond the head-to-head/gauntlet eval, this isolates *why* the agent wins/loses:

- **deck x play 2x2**: net/greedy play crossed with the net's deck / a meta deck,
  to split the agent's edge into a deck part and a play part;
- **loss-cause / game length**: abort & draw rates, turn-length of wins vs losses
  (fast bricks vs slow out-plays);
- **value calibration**: does the value head's sign predict the outcome (overall +
  by game phase) -- it is the leaf evaluator any future search would lean on;
- **policy entropy** and **first/second-player (slot) bias**.

Parallel (one process per game, net cached per worker).

  uv run python scripts/analyze_paper_net.py            # full
  uv run python scripts/analyze_paper_net.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.run_eval import load_engine_data, play_game, read_deck  # noqa: E402
from src.agents import build_agent  # noqa: E402
from src.agents.recurrent_agent import RecurrentNetAgent  # noqa: E402
from src.deck import build_pool  # noqa: E402
from src.harness.stats import wilson_interval  # noqa: E402
from src.net.cb import build_deck  # noqa: E402
from src.net.features import CardFeatures, load_engine_json  # noqa: E402
from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: E402

FINAL = ROOT / "data/paperosfp/main/paper_final.npz"
METAL = ROOT / "decklists/metal_aggro.csv"
ENGINE_JSON = ROOT / "data/bc/engine.json"
DRAW = 2
ABORT = -1
SINGLE = 1

_G: dict = {}


def _init(net_path: str = str(FINAL)) -> None:
    _G["engine"] = load_engine_data()
    _G["pool"] = build_pool()
    _G["net"] = RecurrentPolicyValueNet.load(net_path)


def _agent(play: str, deck: list[int]) -> object:
    if play == "net":
        return RecurrentNetAgent(
            deck, _G["engine"], net=_G["net"], cb_pool=_G["pool"],
            build_deck_from_net=False, temperature=0.0,
        )
    return build_agent(play, deck, _G["engine"])


class _Rec:
    """Captures each net agent's single-select value / entropy, tagged by slot."""

    def __init__(self, agents: tuple) -> None:
        self.agents = agents
        self.dec: list[dict] = []

    def on_decision(self, slot: int, obs: dict, choice: list[int]) -> None:  # noqa: ARG002
        ag = self.agents[slot]
        sel = (obs.get("select") or {})
        if isinstance(ag, RecurrentNetAgent) and int(sel.get("maxCount", 0)) == SINGLE:
            self.dec.append({"slot": slot, "v": ag.last_value, "h": ag.last_entropy})

    def on_end(self, winner: int) -> None:
        pass


def _play(task: dict) -> dict:
    a = _agent(task["pa"], task["da"])
    b = _agent(task["pb"], task["db"])
    a_first = task["a_first"]
    p0, p1 = (a, b) if a_first else (b, a)
    rec = _Rec((p0, p1))
    res = play_game(p0, p1, a_is_player0=a_first, seed=task["seed"], recorder=rec)
    n = len(rec.dec)
    decs = [
        {"v": d["v"], "h": d["h"],
         "outcome": 1.0 if res.winner == d["slot"]
                    else -1.0 if res.winner == (1 - d["slot"]) else 0.0,
         "frac": i / max(n - 1, 1)}
        for i, d in enumerate(rec.dec)
    ]
    return {
        "m": task["m"], "a_won": int(res.a_won),
        "dec": int(res.a_won or res.b_won), "winner": res.winner, "turns": res.turns,
        "a_first": a_first, "a_won_raw": int(res.a_won), "decs": decs,
    }


def _wr(rows: list[dict]) -> dict:
    wins = sum(r["a_won"] for r in rows)
    dec = sum(r["dec"] for r in rows)
    p, lo, hi = wilson_interval(wins, dec)
    return {"winrate": round(p, 3), "ci": [round(lo, 3), round(hi, 3)],
            "decisive": dec, "games": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep analysis of the recurrent net")
    ap.add_argument("--net", type=Path, default=FINAL, help="net to analyse")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results/paper_analysis.json")
    args = ap.parse_args()
    n = 24 if args.quick else 160

    pool = build_pool()
    feats = CardFeatures(load_engine_json(ENGINE_JSON))
    net = RecurrentPolicyValueNet.load(args.net)
    net_deck = build_deck(net, pool, feats)
    metal = read_deck(METAL)

    # deck x play matchups (A's win rate tracked); mirror checks slot bias.
    specs = [
        {"m": "play_net_vs_greedy_on_netdeck", "pa": "net", "da": net_deck,
         "pb": "greedy", "db": net_deck},
        {"m": "play_net_vs_greedy_on_metadeck", "pa": "net", "da": metal,
         "pb": "greedy", "db": metal},
        {"m": "deck_net_vs_meta_netplay", "pa": "net", "da": net_deck,
         "pb": "net", "db": metal},
        {"m": "mirror", "pa": "net", "da": net_deck, "pb": "net", "db": net_deck},
    ]
    tasks = [
        {**s, "a_first": k % 2 == 0, "seed": si * 100000 + k}
        for si, s in enumerate(specs) for k in range(n)
    ]
    with Pool(args.workers, initializer=_init, initargs=(str(args.net),)) as pp:
        rows = pp.map(_play, tasks)

    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["m"], []).append(r)

    matchups = {m: _wr(rs) for m, rs in by.items()}

    # loss-cause / game length (over all games where the net is "A").
    allr = rows
    aborts = sum(r["winner"] == ABORT for r in allr) / len(allr)
    draws = sum(r["winner"] == DRAW for r in allr) / len(allr)
    win_turns = [r["turns"] for r in allr if r["a_won"]]
    loss_turns = [r["turns"] for r in allr if r["dec"] and not r["a_won"]]

    # slot bias from the mirror (A wins by first/second).
    mir = by["mirror"]
    first_wr = np.mean([r["a_won"] for r in mir if r["a_first"]])
    second_wr = np.mean([r["a_won"] for r in mir if not r["a_first"]])

    # value calibration + entropy from all net decisions.
    decs = [d for r in allr for d in r["decs"]]
    dec_arr = [(d["v"], d["outcome"], d["h"], d["frac"]) for d in decs if d["outcome"]]
    v = np.array([x[0] for x in dec_arr])
    o = np.array([x[1] for x in dec_arr])
    h = np.array([d["h"] for d in decs])
    frac = np.array([x[3] for x in dec_arr])
    sign_acc = float(np.mean(np.sign(v) == np.sign(o)))
    early = float(np.mean(np.sign(v[frac < 0.33]) == np.sign(o[frac < 0.33])))
    late = float(np.mean(np.sign(v[frac > 0.66]) == np.sign(o[frac > 0.66])))
    mean_abs_value = round(float(np.mean(np.abs(v))), 3)

    results = {
        "matchups": matchups,
        "loss_cause": {"abort_rate": round(aborts, 3), "draw_rate": round(draws, 3),
                       "avg_win_turns": round(float(np.mean(win_turns)), 1),
                       "avg_loss_turns": round(float(np.mean(loss_turns)), 1)},
        "slot_bias": {"mirror_first_winrate": round(float(first_wr), 3),
                      "mirror_second_winrate": round(float(second_wr), 3)},
        "value_calibration": {"sign_accuracy": round(sign_acc, 3),
                              "early_game": round(early, 3),
                              "late_game": round(late, 3),
                              "mean_abs_value": mean_abs_value,
                              "n_decisions": len(v)},
        "policy_entropy": {"mean": round(float(np.mean(h)), 3),
                           "median": round(float(np.median(h)), 3)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
