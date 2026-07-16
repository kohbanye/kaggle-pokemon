"""AZ play-net evaluation: DECOMPOSE deck vs play, and measure EXPLOITABILITY.

The co-evo heldout2 curve conflates two levers -- a better DECK (QD) and better PLAY
(the AZ net). A rising curve has repeatedly been mis-read as a play gain when it was a
deck gain ([[az-lstm-coevo-works]]). This script isolates PLAY:

    play_delta(deck) = winrate(net | deck) - winrate(greedy_plus | deck)

on the SAME deck vs the held-out real-ladder pool ``decklists/heldout2``. A positive,
CI-separated delta on a deck is a genuine play win; averaged over a deck panel it is the
headline "does the AZ pilot beat the heuristic" number. We also report an EXPLOITABILITY
proxy (worst-case held-out matchup: ``1 - min_opp winrate(net|deck)``) -- a robustness
health check that flags rigid quirks (the always-go-second failure mode).

Reuses ``heldout_eval.py`` verbatim (one shared-pool invocation for all subjects) so the
match harness / CIs / opponent pilots are identical to the co-evo gate.

  uv run python scripts/az_eval.py --net data/coevo_az/gen8/net.npz \
      --decks g8_e0,qdgp_best,metal_aggro,qd7_r5,grass_aggro --games 30
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DECKS = "g8_e0,qdgp_best,metal_aggro,qd7_r5,grass_aggro"


def _run_heldout(net: Path, decks: list[str], games: int,  # noqa: PLR0913
                 workers: int, pool: str, out: Path) -> dict:
    subjects = [f"{pilot}|{d}" for d in decks for pilot in ("net", "greedy_plus")]
    subprocess.run(  # noqa: S603
        ["uv", "run", "python", "scripts/heldout_eval.py",  # noqa: S607
         "--pool", pool, "--games", str(games), "--workers", str(workers),
         "--net", str(net), "--subjects", ",".join(subjects), "--out", str(out)],
        check=True, cwd=ROOT)
    return json.loads(out.read_text())


def _exploitability(per_opp: dict[str, float]) -> tuple[float, str]:
    """Worst-case held-out matchup for a subject: (1 - min winrate, that opponent)."""
    if not per_opp:
        return float("nan"), ""
    worst = min(per_opp, key=lambda k: per_opp[k])
    return round(1.0 - per_opp[worst], 3), worst


def main() -> None:
    ap = argparse.ArgumentParser(description="AZ play decomposition + exploitability")
    ap.add_argument("--net", type=Path, default=ROOT / "data/coevo_az/gen8/net.npz")
    ap.add_argument("--decks", default=DEFAULT_DECKS)
    ap.add_argument("--pool", default="heldout2")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=Path, default=ROOT / "results/az_eval.json")
    args = ap.parse_args()

    decks = [d for d in args.decks.split(",") if d]
    raw_path = args.out.with_name(args.out.stem + "_raw.json")
    raw = _run_heldout(args.net, decks, args.games, args.workers, args.pool, raw_path)

    rows = []
    for d in decks:
        net_o = raw["overall"].get(f"net|{d}")
        gp_o = raw["overall"].get(f"greedy_plus|{d}")
        if not net_o or not gp_o:
            continue
        delta = round(net_o["winrate"] - gp_o["winrate"], 3)
        expl, worst = _exploitability(raw["per_opp"].get(f"net|{d}", {}))
        # Non-overlapping Wilson CIs = a real, significant difference in that direction.
        sig = "+" if net_o["ci"][0] > gp_o["ci"][1] else (
            "-" if net_o["ci"][1] < gp_o["ci"][0] else "~")
        rows.append({"deck": d, "net_wr": net_o["winrate"], "net_ci": net_o["ci"],
                     "gp_wr": gp_o["winrate"], "gp_ci": gp_o["ci"],
                     "play_delta": delta, "sig": sig,
                     "exploitability": expl, "worst_opp": worst})

    deltas = [r["play_delta"] for r in rows]
    mean_delta = round(sum(deltas) / len(deltas), 3) if deltas else float("nan")
    n_win = sum(1 for r in rows if r["sig"] == "+")
    n_loss = sum(1 for r in rows if r["sig"] == "-")
    summary = {"net": str(args.net), "pool": args.pool, "games_per_pair": args.games,
               "mean_play_delta": mean_delta, "decks_net_beats_gp": n_win,
               "decks_gp_beats_net": n_loss, "n_decks": len(rows), "per_deck": rows}
    args.out.write_text(json.dumps(summary, indent=2))

    print(f"\n=== AZ play decomposition (net vs greedy_plus, {args.pool}) ===")
    print(f"{'deck':<14}{'net':>14}{'greedy_plus':>16}{'Δplay':>9}{'   ':>3}"
          f"{'exploit':>9} worst_opp")
    for r in rows:
        print(f"{r['deck']:<14}"
              f"{r['net_wr']:.3f} {r['net_ci']!s:>8}"
              f"{r['gp_wr']:.3f} {r['gp_ci']!s:>8}"
              f"{r['play_delta']:+.3f}  {r['sig']}"
              f"{r['exploitability']:>9.3f} {r['worst_opp']}")
    verdict = ("PLAY WIN" if mean_delta > 0 and n_win > n_loss else
               "no play gain" if mean_delta <= 0 else "mixed")
    print(f"\nmean Δplay={mean_delta:+.3f}  net>gp on {n_win}/{len(rows)} decks, "
          f"gp>net on {n_loss}  ->  {verdict}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
