"""Does the net *learn to pilot* slow / single-prize decks over co-evo rounds?

The QD archive's fitness for a slow cell is winrate-vs-meta = (a) operating skill x
(b) deck power vs the meta. A flat fitness can't tell "the net can't learn to pilot it"
from "the deck just loses to meta regardless of pilot". This isolates (a): for a FIXED
single-prize deck and a FIXED ramp deck (extracted from the run's final archive), it
plays each round's net **vs greedy on the same deck** -- a rising net-vs-greedy means
the net is genuinely learning to pilot that archetype, even if its vs-meta stays low.
A strong (3,0) deck is included as a control (the harness/known-good signal).

Docker/native engine (imports cg). Run:
  uv run python scripts/slow_skill.py --run data/qdcoevo/run7 --rounds 6 \
      --init data/qdcoevo/run6/round_3/rl/paper_final.npz --games 40
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

from scripts.run_eval import load_engine_data, play_game  # noqa: E402
from src.agents import build_agent  # noqa: E402
from src.agents.recurrent_agent import RecurrentNetAgent  # noqa: E402
from src.deck import build_pool  # noqa: E402
from src.harness.stats import wilson_interval  # noqa: E402
from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: E402
from src.qd.deck_qd import behaviour_descriptor  # noqa: E402

_G: dict = {}


def _init(net_paths: dict[str, str]) -> None:
    _G["engine"] = load_engine_data()
    _G["pool"] = build_pool()
    _G["nets"] = {k: RecurrentPolicyValueNet.load(v) for k, v in net_paths.items()}


def _play(task: dict) -> dict:
    deck = task["deck"]
    net = RecurrentNetAgent(deck, _G["engine"], net=_G["nets"][task["net"]],
                            cb_pool=_G["pool"], build_deck_from_net=False,
                            temperature=0.0)
    greedy = build_agent("greedy", deck, _G["engine"])
    net_first = task["net_first"]
    p0, p1 = (net, greedy) if net_first else (greedy, net)
    res = play_game(p0, p1, a_is_player0=net_first, seed=task["seed"])
    return {"key": task["key"], "won": int(res.a_won),
            "dec": int(res.a_won or res.b_won)}


def _pick_decks(archive: Path, pool: object) -> dict[str, list[int]]:
    """Best deck per tracked archetype from the final archive (by fitness)."""
    cells = json.loads(archive.read_text())["cells"]
    best: dict[str, tuple[float, list[int]]] = {}
    for c in cells:
        pbin, sbin = behaviour_descriptor(c["deck"], pool)
        if pbin == 0:
            tag = "single_prize"  # pure single-prize (the empty niche)
        elif sbin >= 2:  # speed bin >= 2 = ramp
            tag = "ramp"
        elif (pbin, sbin) == (3, 0):
            tag = "strong_control"  # a strong cell as a known-good control
        else:
            continue
        if tag not in best or c["fitness"] > best[tag][0]:
            best[tag] = (c["fitness"], c["deck"])
    return {tag: deck for tag, (_, deck) in best.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Slow-deck piloting skill across rounds")
    ap.add_argument("--run", type=Path, default=ROOT / "data/qdcoevo/run7")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--init", type=Path,
                    default=ROOT / "data/qdcoevo/run6/round_3/rl/paper_final.npz")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=Path, default=ROOT / "results/run7_slow_skill.json")
    args = ap.parse_args()

    pool = build_pool()
    decks = _pick_decks(args.run / f"round_{args.rounds}/qd_archive.json", pool)
    net_paths = {"init": str(args.init)}
    for r in range(1, args.rounds + 1):
        net_paths[f"r{r}"] = str(args.run / f"round_{r}/rl/paper_final.npz")

    tasks = [
        {"key": f"{net}|{tag}", "net": net, "deck": deck, "net_first": k % 2 == 0,
         "seed": hash((net, tag)) % 99999 + k}
        for net in net_paths
        for tag, deck in decks.items()
        for k in range(args.games)
    ]

    with Pool(args.workers, initializer=_init, initargs=(net_paths,)) as pp:
        rows = pp.map(_play, tasks)

    agg: dict[str, list[dict]] = {}
    for r in rows:
        agg.setdefault(r["key"], []).append(r)
    out: dict[str, dict] = {}
    for key, rs in agg.items():
        wins = sum(r["won"] for r in rs)
        dec = sum(r["dec"] for r in rs)
        p, lo, hi = wilson_interval(wins, dec)
        out[key] = {"net_vs_greedy": round(p, 3), "ci": [round(lo, 3), round(hi, 3)],
                    "decisive": dec}

    args.out.write_text(json.dumps({"decks": list(decks), "results": out}, indent=2))
    print(f"tracked decks: {list(decks)}")
    print(f"{'deck':<16}" + "".join(f"{n:>9}" for n in net_paths))
    for tag in decks:
        row = "".join(f"{out[f'{n}|{tag}']['net_vs_greedy']:>9.3f}" for n in net_paths)
        print(f"{tag:<16}{row}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
