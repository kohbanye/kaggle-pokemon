"""Build the v2 evaluation pools from REAL ladder opponent decks.

The v1 held-out pool (``decklists/heldout/``) was synthetic basic-aggro --
0% evolution decks vs the real ladder's 95% (ladder_episodes.py finding), i.e.
it measured against the wrong meta. v2 splits the distinct opponent decks
harvested from our submissions' replays (``results/episodes/ladder_matches.json``)
into two DISJOINT sets:

- ``decklists/anchors/``   -- the strongest opponents (by Elo), used as permanent
  QD gauntlet anchors + round-1 opponents (training-side grounding).
- ``decklists/heldout2/``  -- an archetype-stratified sample of the rest, used
  ONLY for evaluation (the new ladder proxy). Never fed to the search.

A manifest records each deck's provenance (team, Elo, episode) so the split is
auditable and re-buildable as more episodes accumulate.

  uv run python scripts/ladder_episodes.py --sub <ids...>   # refresh matches
  uv run python scripts/build_heldout_v2.py --anchors 4 --heldout 30
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.deck import build_pool, legality_errors  # noqa: E402
from src.qd import evolution_depth, prize_liability  # noqa: E402
from src.qd.deck_qd import primary_colour  # noqa: E402

MATCHES = ROOT / "results" / "episodes" / "ladder_matches.json"


def archetype(deck: list[int], pool) -> str:  # noqa: ANN001
    """Coarse stratification key: colour x evo x prize band."""
    evo = "evo" if evolution_depth(deck, pool) > 0.25 else "basic"
    pl = prize_liability(deck, pool)
    prize = "1pz" if pl < 1.3 else ("mid" if pl < 1.9 else "big")
    return f"{primary_colour(deck, pool)}/{evo}/{prize}"


def main() -> None:  # noqa: C901 - linear build steps, not branching logic
    ap = argparse.ArgumentParser(description="Build ladder-derived eval pools")
    ap.add_argument("--anchors", type=int, default=4,
                    help="top-Elo decks reserved as QD anchors (training side)")
    ap.add_argument("--heldout", type=int, default=30,
                    help="archetype-stratified decks for the eval-only pool")
    args = ap.parse_args()
    pool = build_pool()

    rows = json.loads(MATCHES.read_text())
    best: dict[tuple, dict] = {}  # dedup identical decklists, keep max-Elo owner
    for r in rows:
        if not r.get("opp_deck"):
            continue
        deck = [int(c) for c in r["opp_deck"]]
        if legality_errors(deck, pool):
            continue
        key = tuple(sorted(deck))
        cur = best.get(key)
        if cur is None or r["opp_elo"] > cur["elo"]:
            best[key] = {"deck": deck, "elo": r["opp_elo"],
                         "team": r["opp_team"], "episode": r["episode"]}
    decks = sorted(best.values(), key=lambda d: -d["elo"])
    print(f"{len(decks)} distinct legal opponent decks")

    anchors = decks[: args.anchors]
    rest = decks[args.anchors:]
    # Stratified held-out: round-robin over archetypes, strongest first, so the
    # pool covers the meta's variety instead of one dominant shape.
    by_arch: dict[str, list[dict]] = defaultdict(list)
    for d in rest:
        by_arch[archetype(d["deck"], pool)].append(d)
    heldout: list[dict] = []
    while len(heldout) < min(args.heldout, len(rest)):
        for arch in sorted(by_arch, key=lambda a: -len(by_arch[a])):
            if by_arch[arch] and len(heldout) < args.heldout:
                heldout.append(by_arch[arch].pop(0))

    manifest = {"anchors": [], "heldout2": []}
    for name, group, out_dir in (("anchors", anchors, ROOT / "decklists/anchors"),
                                 ("heldout2", heldout,
                                  ROOT / "decklists/heldout2")):
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, d in enumerate(group):
            arch = archetype(d["deck"], pool).replace("/", "_")
            fn = f"lad{i:02d}_{arch}_e{int(d['elo'])}.csv"
            (out_dir / fn).write_text("\n".join(map(str, d["deck"])) + "\n")
            manifest[name].append({"file": fn, "elo": d["elo"],
                                   "team": d["team"], "episode": d["episode"],
                                   "archetype": archetype(d["deck"], pool)})
        print(f"{name}: {len(group)} decks -> {out_dir} "
              f"(elo {group[-1]['elo']:.0f}..{group[0]['elo']:.0f})")
    (ROOT / "decklists" / "ladder_pools_manifest.json").write_text(
        json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
