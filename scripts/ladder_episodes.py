"""Pull + analyse real ladder episodes for our submissions -> per-opponent report.

Uses Kaggle's public Episode API (the same data the UI shows): ListEpisodes for
each of our submission ids (cached under ``results/episodes/``), then each
episode's replay JSON, from which both 60-card decks are extracted (the step-1
actions). Opponent decks are classified with the same decklist-only descriptor
the QD search uses, so ladder wins/losses can finally be broken down by real
opponent archetype -- and the ladder's deck distribution compared against our
held-out pool.

  uv run python scripts/ladder_episodes.py --sub 54367885 --sub 54266227
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.deck import build_pool  # noqa: E402
from src.qd import (  # noqa: E402
    colour_count,
    deck_stats,
    evolution_depth,
    prize_liability,
)
from src.qd.deck_qd import primary_colour  # noqa: E402

EP_DIR = ROOT / "results" / "episodes"
LIST_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
REPLAY_URL = "https://www.kaggleusercontent.com/episodes/{eid}.json"


def _curl(url: str, post: dict | None = None) -> bytes:
    cmd = ["curl", "-sL", url]
    if post is not None:
        cmd += ["-X", "POST", "-H", "Content-Type: application/json",
                "-d", json.dumps(post)]
    return subprocess.run(cmd, capture_output=True, check=True).stdout  # noqa: S603


def fetch_episode_list(sub_id: int) -> dict:
    out = EP_DIR / f"list_{sub_id}.json"
    if not out.exists():
        out.write_bytes(_curl(LIST_URL, {"submissionId": sub_id}))
    return json.loads(out.read_text())


def fetch_replay(eid: int) -> dict | None:
    out = EP_DIR / f"ep{eid}.json"
    if not out.exists():
        data = _curl(REPLAY_URL.format(eid=eid))
        if not data or data[:1] != b"{":
            return None
        out.write_bytes(data)
    return json.loads(out.read_text())


def replay_decks(replay: dict) -> list[list[int]] | None:
    """Both agents' 60-card decks (the step-1 deck-selection actions)."""
    steps = replay.get("steps") or []
    if len(steps) < 2:
        return None
    decks = []
    for agent in steps[1]:
        act = agent.get("action") or []
        if len(act) != 60:
            return None
        decks.append([int(c) for c in act])
    return decks


def classify(deck: list[int], pool) -> str:  # noqa: ANN001
    """Compact archetype tag from the decklist-only descriptor features."""
    s = deck_stats(deck, pool)
    col = primary_colour(deck, pool)
    pl = prize_liability(deck, pool)
    evo = evolution_depth(deck, pool)
    kind = "evo" if evo > 0.25 else ("splash" if evo > 0.05 else "basic")
    prize = "1pz" if pl < 1.3 else ("mid" if pl < 1.9 else "ex/mega")
    return (f"{col}/{kind}/{prize} P{s['pokemon']}T{s['trainer']}E{s['energy']}"
            f" c{colour_count(deck, pool)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ladder episode analysis")
    ap.add_argument("--sub", type=int, action="append", required=True,
                    help="our submission id (repeatable)")
    args = ap.parse_args()
    EP_DIR.mkdir(parents=True, exist_ok=True)
    pool = build_pool()

    rows = []
    for sid in args.sub:
        listing = fetch_episode_list(sid)
        teams = {t["id"]: t.get("teamName", "?") for t in listing.get("teams", [])}
        for ep in listing["episodes"]:
            if ep.get("state") != "COMPLETED":
                continue
            agents = ep["agents"]
            mine = next((a for a in agents if a.get("submissionId") == sid), None)
            opp = next((a for a in agents if a.get("submissionId") != sid), None)
            if mine is None or opp is None:
                continue
            replay = fetch_replay(ep["id"])
            decks = replay_decks(replay) if replay else None
            my_idx = mine.get("index", 0)
            opp_deck = decks[1 - my_idx] if decks else None
            rows.append({
                "sub": sid,
                "episode": ep["id"],
                "reward": mine.get("reward"),
                "elo_before": round(mine.get("initialScore") or 0, 1),
                "elo_after": round(mine.get("updatedScore") or 0, 1),
                "opp_elo": round(opp.get("initialScore") or 0, 1),
                "opp_team": teams.get(opp.get("teamId"), "?"),
                "opp_sub": opp.get("submissionId"),
                "opp_archetype": classify(opp_deck, pool) if opp_deck else "?",
                "opp_deck": opp_deck,
            })

    (EP_DIR / "ladder_matches.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} completed episodes -> {EP_DIR / 'ladder_matches.json'}\n")
    for sid in args.sub:
        sub_rows = [r for r in rows if r["sub"] == sid]
        w = sum(r["reward"] == 1 for r in sub_rows)
        losses = [r for r in sub_rows if r["reward"] == -1]
        print(f"== sub {sid}: {w}W/{len(losses)}L "
              f"(of {len(sub_rows)} decisive+ties) ==")
        for r in sorted(losses, key=lambda r: -r["opp_elo"]):
            print(f"  LOSS vs {r['opp_team'][:20]:<20} elo={r['opp_elo']:>6} "
                  f"{r['opp_archetype']}")
        arch = Counter(r["opp_archetype"].split(" ")[0] for r in sub_rows)
        print("  opponent archetypes:", dict(arch.most_common(8)))


if __name__ == "__main__":
    main()
