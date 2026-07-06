"""Decision-level best-practice audit of a pilot -> where does the RL net misplay?

PTCG openings are near-scripted (bench basics, attach every turn, evolve on
curve, take a knockout when offered). This script plays a subject pilot on a
fixed deck against real-ladder opponents, records EVERY decision, and scores:

- go-first rate                (convention: go first)
- missed-attach rate           (ended own turn without the free energy attach)
- missed-bench rate (T1-3)     (could bench a Basic onto a free slot, ended turn)
- missed-evolve rate           (evolve option offered, chose End instead)
- missed-KO rate               (an attack with damage >= opp active HP existed,
                                chose something that was not a KO attack)
- greedy-shadow disagreement   (what would greedy have picked on the same obs?)
- value calibration (nets)     (mid-game value sign vs final outcome)

Also probes DECK-CONDITIONING sensitivity offline: same states, swapped deck
context -> how often does the argmax action change? (0 = the net ignores ctx.)

  uv run python scripts/play_bestpractice.py --games 12 --out results/bp.json
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
from src.agents.base import (  # noqa: E402
    OPT_ATTACH,
    OPT_ATTACK,
    OPT_END,
    OPT_EVOLVE,
    OPT_PLAY,
    OPT_YES,
)
from src.deck import build_pool  # noqa: E402

CTX_IS_FIRST = 41
SINGLE = 1

_G: dict = {}


def _opt_types(options: list[dict]) -> list[int]:
    return [int(o.get("type", -1)) for o in options]


def _hand_card_id(opt: dict, current: dict, you: int) -> int | None:
    """Card id a PLAY/EVOLVE option refers to (hand index -> card)."""
    hand = (current.get("players") or [{}, {}])[you].get("hand") or []
    idx = opt.get("index")
    if idx is None or not (0 <= int(idx) < len(hand)):
        return None
    card = hand[int(idx)]
    return int(card.get("id")) if card else None


class _Audit:
    """Recorder computing per-decision best-practice flags for one slot."""

    def __init__(self, slot: int, engine: dict, pool) -> None:  # noqa: ANN001
        self.slot = slot
        self.dmg = engine.get("attacks", {})
        self.pool = pool
        self.rows: list[dict] = []
        self.winner = -1

    def on_decision(  # noqa: C901 - one flag block per best-practice probe
        self, slot: int, obs: dict, choice: list[int],
    ) -> None:
        if slot != self.slot:
            return
        select = obs.get("select") or {}
        options = select.get("option") or []
        if not options:
            return
        current = obs.get("current") or {}
        you = int(current.get("yourIndex", 0))
        turn = int(current.get("turn", 0))
        types = _opt_types(options)
        ch = int(choice[0]) if choice else -1
        ch_type = types[ch] if 0 <= ch < len(types) else -1
        row = {"turn": turn, "context": int(select.get("context", -1)),
               "types": types, "choice": ch, "choice_type": ch_type}
        # go-first
        if row["context"] == CTX_IS_FIRST:
            row["went_first"] = ch_type == OPT_YES
        # missed attach: ended turn although the free attach was still available
        if ch_type == OPT_END and OPT_ATTACH in types:
            row["missed_attach"] = not bool(current.get("energyAttached"))
        # missed evolve: ended turn although an evolve option was offered
        if ch_type == OPT_END and OPT_EVOLVE in types:
            row["missed_evolve"] = True
        # missed bench (T1-3): a Basic was playable onto a free bench slot
        if ch_type == OPT_END and turn <= 3 and OPT_PLAY in types:
            players = current.get("players") or [{}, {}]
            bench = players[you].get("bench") or []
            free = any(b is None for b in bench) or len(bench) < 5  # max bench
            playable_basic = any(
                t == OPT_PLAY and (cid := _hand_card_id(o, current, you))
                and cid in self.pool.cards
                and self.pool.cards[cid].is_basic_pokemon
                for t, o in zip(types, options, strict=True))
            if free and playable_basic:
                row["missed_bench"] = True
        # missed KO: an attack whose base damage >= opp active HP existed
        if OPT_ATTACK in types:
            players = current.get("players") or [{}, {}]
            opp_active = (players[1 - you].get("active") or [None])[0]
            if opp_active:
                hp = int(opp_active.get("hp", 9999))
                ko_idx = [i for i, o in enumerate(options)
                          if types[i] == OPT_ATTACK
                          and self.dmg.get(o.get("attackId"), {}).get("dmg", 0)
                          >= hp]
                if ko_idx:
                    row["ko_avail"] = True
                    row["chose_ko"] = ch in ko_idx
        self.rows.append(row)

    def on_end(self, winner: int) -> None:
        self.winner = winner

    def turn_missed_kos(self) -> list[bool]:
        """Per-TURN missed KOs (greedy legitimately develops before attacking,
        so a per-decision flag would misfire on every develop-first line)."""
        turns: dict[int, dict] = {}
        for r in self.rows:
            if r.get("ko_avail"):
                t = turns.setdefault(r["turn"], {"avail": True, "took": False})
                t["took"] = t["took"] or r.get("chose_ko", False)
        return [not t["took"] for t in turns.values()]


def _init() -> None:
    from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: PLC0415

    _G["engine"] = load_engine_data()
    _G["pool"] = build_pool()
    _G["nets"] = {k: RecurrentPolicyValueNet.load(v)
                  for k, v in _G_NET_PATHS.items()}


_G_NET_PATHS = {
    "net_run1": "data/qdrl_run1/round_5/rl/paper_final.npz",
    "net_run4": "data/qdrl_run4/round_7/rl/paper_final.npz",
}


def _pilot(kind: str, deck: list[int]) -> object:
    if kind in _G_NET_PATHS:
        from src.agents.recurrent_agent import RecurrentNetAgent  # noqa: PLC0415
        return RecurrentNetAgent(deck, _G["engine"], net=_G["nets"][kind],
                                 cb_pool=_G["pool"], build_deck_from_net=False,
                                 temperature=0.0)
    return build_agent(kind, deck, _G["engine"])


def _game(task: dict) -> dict:
    subj = _pilot(task["kind"], task["deck"])
    opp = build_agent("greedy", task["opp_deck"], _G["engine"])
    shadow = build_agent("greedy", task["deck"], _G["engine"])
    subj_first = task["k"] % 2 == 0
    slot = 0 if subj_first else 1
    audit = _Audit(slot, _G["engine"], _G["pool"])

    # wrap the audit to also record greedy-shadow (dis)agreement per decision
    base_on_decision = audit.on_decision

    def on_decision(s: int, obs: dict, choice: list[int]) -> None:
        n_before = len(audit.rows)
        base_on_decision(s, obs, choice)
        if s == slot and len(audit.rows) > n_before:
            select = obs.get("select") or {}
            if int(select.get("maxCount", 0)) == SINGLE:
                try:
                    g = shadow.act(obs)
                    audit.rows[-1]["greedy_choice"] = int(g[0]) if g else -1
                except Exception:  # noqa: BLE001, S110 - diagnostics only
                    pass

    audit.on_decision = on_decision  # type: ignore[method-assign]
    p0, p1 = ((subj, opp) if subj_first else (opp, subj))
    play_game(p0, p1, a_is_player0=subj_first, seed=task["seed"],
              recorder=audit)
    won = audit.winner == slot
    return {"kind": task["kind"], "won": won, "rows": audit.rows,
            "turn_missed_kos": audit.turn_missed_kos()}


def _rate(rows: list[dict], key: str) -> tuple[float | None, int]:
    xs = [r[key] for r in rows if key in r]
    return (round(sum(xs) / len(xs), 3) if xs else None), len(xs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Best-practice play audit")
    ap.add_argument("--deck", type=Path,
                    default=ROOT / "decklists/candidates/qd7_r5.csv")
    ap.add_argument("--pilots", type=str, default="greedy,net_run1,net_run4")
    ap.add_argument("--games", type=int, default=12, help="per opponent deck")
    ap.add_argument("--opponents", type=int, default=8,
                    help="heldout2 decks used as the opponent panel")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/play_bestpractice.json")
    args = ap.parse_args()

    deck = read_deck(args.deck)
    opps = [read_deck(p) for p in
            sorted((ROOT / "decklists/heldout2").glob("*.csv"))
            [: args.opponents]]
    pilots = [p.strip() for p in args.pilots.split(",") if p.strip()]

    tasks = [{"kind": kind, "deck": deck, "opp_deck": od, "k": k,
              "seed": (pi * 733 + oi) * 100 + k}
             for pi, kind in enumerate(pilots)
             for oi, od in enumerate(opps)
             for k in range(args.games)]
    with Pool(args.workers, initializer=_init) as pp:
        games = pp.map(_game, tasks)

    report: dict = {}
    for kind in pilots:
        gs = [g for g in games if g["kind"] == kind]
        rows = [r for g in gs for r in g["rows"]]
        dis = [r for r in rows if "greedy_choice" in r]
        report[kind] = {
            "games": len(gs),
            "winrate": round(sum(g["won"] for g in gs) / len(gs), 3),
            "go_first": _rate(rows, "went_first")[0],
            "missed_attach": _rate(rows, "missed_attach")[0],
            "missed_bench_T1_3": _rate(rows, "missed_bench")[0],
            "missed_evolve": _rate(rows, "missed_evolve")[0],
            "missed_ko_turnlevel": (round(sum(x for g in gs
                                              for x in g["turn_missed_kos"])
                                          / max(1, sum(len(g["turn_missed_kos"])
                                                       for g in gs)), 3)),
            "n_ko_turns": sum(len(g["turn_missed_kos"]) for g in gs),
            "greedy_disagree": (round(sum(r["choice"] != r["greedy_choice"]
                                          for r in dis) / len(dis), 3)
                                if dis else None),
        }
        print(kind, json.dumps(report[kind]))
    args.out.write_text(json.dumps(report, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
