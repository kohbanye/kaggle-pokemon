"""Tactical analysis of cached ladder replays -> how opponents actually play.

STATE-based (the engine event log is re-delivered per observation and cannot be
deduplicated reliably): walks each replay's board states (``current`` on agent
0's observations shows both players absolutely), samples one snapshot per turn,
and derives per-side behaviour -- prize-race pace, evolution progress, board
energy, bench width -- plus the win reason inferred from the terminal state
(prizes taken / deck-out / no-active). Aggregated into "how do opponents that
beat us play vs the ones we beat": the playstyle counterpart of
ladder_episodes.py's deck-archetype breakdown.

  uv run python scripts/ladder_tactics.py            # uses results/episodes/
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.deck import build_pool  # noqa: E402
from src.qd import card_stage  # noqa: E402

EP_DIR = ROOT / "results" / "episodes"
PRIZES = 6


def _board(player: dict) -> list[dict]:
    return [c for c in (player.get("active") or []) if c] + [
        c for c in (player.get("bench") or []) if c]


def _snapshot(player: dict, pool) -> dict:  # noqa: ANN001
    board = _board(player)
    return {
        "prizes_left": len(player.get("prize") or []),
        "deck": player.get("deckCount", 0),
        "bench": len([c for c in (player.get("bench") or []) if c]),
        "energy": sum(len(c.get("energyCards") or []) for c in board),
        "max_stage": max((card_stage(pool.cards[c["id"]])
                          for c in board if c.get("id") in pool.cards),
                         default=0),
        "has_active": bool(player.get("active") and any(player["active"])),
    }


def game_features(replay: dict, pool) -> dict | None:  # noqa: ANN001
    """Per-turn snapshots -> one feature dict per side + result/reason."""
    per_turn: dict[int, list[dict]] = {}
    for step in replay.get("steps", []):
        for agent in step:  # each agent only observes on its own decisions --
            cur = (agent.get("observation") or {}).get("current")  # merge both
            if not cur or not cur.get("players"):
                continue
            per_turn[cur.get("turn", 0)] = [
                _snapshot(cur["players"][p], pool) for p in (0, 1)]
    rewards = replay.get("rewards") or []
    if not per_turn or len(rewards) != 2 or rewards[0] == rewards[1]:
        return None  # no states, or a draw/error episode
    result = 0 if rewards[0] > rewards[1] else 1  # agent order == player order
    turns = sorted(per_turn)
    last = per_turn[turns[-1]]
    loser = 1 - result if result in (0, 1) else None
    reason = "draw"
    if loser is not None:
        if last[result]["prizes_left"] == 0:
            reason = "prizes"
        elif not last[loser]["has_active"]:
            reason = "no-active"
        elif last[loser]["deck"] == 0:
            reason = "deck-out"
        else:
            reason = "other"
    out: dict = {"result": result, "reason": reason, "turns": turns[-1]}
    for p in (0, 1):
        series = [(t, per_turn[t][p]) for t in turns]
        first_prize = next((t for t, s in series
                            if s["prizes_left"] < PRIZES), None)
        t10 = [s for t, s in series if t <= 10]
        out[f"p{p}"] = {
            "first_prize_turn": first_prize,
            "prizes_taken": PRIZES - last[p]["prizes_left"],
            "max_stage": max(s["max_stage"] for _, s in series),
            "stage_by_t6": max((s["max_stage"] for t, s in series if t <= 6),
                               default=0),
            "energy_peak": max(s["energy"] for _, s in series),
            "bench_t3": next((s["bench"] for t, s in series if t >= 3), 0),
            "prizes_by_t10": PRIZES - min((s["prizes_left"] for s in t10),
                                          default=PRIZES),
        }
    return out


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 2) if xs else None


def summarise(rows: list[dict], label: str) -> None:
    if not rows:
        print(f"== {label}: no games ==")
        return
    reasons = {k: sum(1 for r in rows if r["reason"] == k)
               for k in {r["reason"] for r in rows}}
    print(f"== {label} (n={len(rows)}) ==")
    print(f"  length {_mean([r['turns'] for r in rows])} turns; "
          f"end reasons {reasons}")
    for side, name in (("opp", "opponent"), ("me", "us")):
        f = [r[side] for r in rows]
        print(f"  {name:<9}: 1st-prize T{_mean([x['first_prize_turn'] for x in f])} "
              f"prizes@T10 {_mean([x['prizes_by_t10'] for x in f])} "
              f"max-stage {_mean([x['max_stage'] for x in f])} "
              f"stage@T6 {_mean([x['stage_by_t6'] for x in f])} "
              f"bench@T3 {_mean([x['bench_t3'] for x in f])} "
              f"energy-peak {_mean([x['energy_peak'] for x in f])}")


def main() -> None:
    pool = build_pool()
    matches = json.loads((EP_DIR / "ladder_matches.json").read_text())
    rows = []
    for m in matches:
        if m["reward"] not in (1, -1):
            continue
        path = EP_DIR / f"ep{m['episode']}.json"
        if not path.exists():
            continue
        g = game_features(json.loads(path.read_text()), pool)
        if g is None or g["result"] not in (0, 1):
            continue
        my_idx = g["result"] if m["reward"] == 1 else 1 - g["result"]
        rows.append({**{k: m[k] for k in ("sub", "episode", "reward", "opp_elo",
                                          "opp_team", "opp_archetype")},
                     "turns": g["turns"], "reason": g["reason"],
                     "me": g[f"p{my_idx}"], "opp": g[f"p{1 - my_idx}"]})

    (EP_DIR / "ladder_tactics.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} decisive games analysed\n")
    for sid in sorted({r["sub"] for r in rows}):
        sub = [r for r in rows if r["sub"] == sid]
        print(f"#### submission {sid}")
        summarise([r for r in sub if r["reward"] == -1], "LOSSES")
        summarise([r for r in sub if r["reward"] == 1], "WINS")
        print()


if __name__ == "__main__":
    main()
