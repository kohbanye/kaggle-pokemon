"""Increment-2 A/B: does the value-net 1-ply lookahead improve Archaludon's play?

Two SUBJECTS on the same Archaludon netdeck:
  * ``baseline``  = the bespoke rule-based ``ArchaludonAgent``
  * ``v``         = ``ArchaludonVAgent`` (rule agent + V-guided lookahead on the
                    ATTACK / Boss-target frames)

Each subject is played, slot-swapped, against the SAME realistic opponent set
  {bespoke Alakazam, greedy_plus on 4 heldout2 ladder decks},
for ``--games`` games per matchup (>=60), and we report per-opponent win-rate +
Wilson 95% CI for BOTH subjects, the DELTA (v - baseline) with a two-proportion
95% CI, and the direct head-to-head v vs baseline.

Run:  uv run python scripts/eval_v_lookahead.py --games 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.run_eval import load_engine_data, play_game  # noqa: E402
from src.agents import build_agent  # noqa: E402
from src.agents.alakazam_pilot import AlakazamAgent  # noqa: E402
from src.agents.archaludon_pilot import ArchaludonAgent  # noqa: E402
from src.agents.archaludon_v_pilot import ArchaludonVAgent  # noqa: E402
from src.harness.stats import wilson_interval  # noqa: E402

ARCH_DECK = ROOT / "decklists" / "archaludon_netdeck.csv"
ALA_DECK = ROOT / "decklists" / "alakazam_netdeck.csv"
HELDOUT = ROOT / "decklists" / "heldout2"
OPP_DECK_FILES = [
    ALA_DECK,
    HELDOUT / "lad00_P_evo_1pz_e790.csv",
    HELDOUT / "lad03_G_evo_1pz_e741.csv",
    HELDOUT / "lad04_F_evo_mid_e839.csv",
    HELDOUT / "lad07_M_evo_mid_e711.csv",
]


def _read(p: Path) -> list[int]:
    return [int(x) for x in p.read_text().split() if x.strip()]


def match(subject, opponent, *, games: int, base_seed: int) -> dict:
    """Slot-swapped subject-vs-opponent match. Subject is A."""
    a_wins = b_wins = draws = aborted = 0
    move_ms: list[float] = []
    max_ms = 0.0
    for g in range(games):
        seed = base_seed + g
        a_is_p0 = g % 2 == 0
        if a_is_p0:
            res = play_game(subject, opponent, a_is_player0=True, seed=seed)
        else:
            res = play_game(opponent, subject, a_is_player0=False, seed=seed)
        a_wins += res.a_won
        b_wins += res.b_won
        draws += res.is_draw
        aborted += res.is_aborted
        if res.moves_a:
            move_ms.append(1000.0 * res.agent_time_a / res.moves_a)
        max_ms = max(max_ms, 1000.0 * res.max_move_a)
    dec = a_wins + b_wins
    p, lo, hi = wilson_interval(a_wins, dec)
    return {
        "wins": a_wins, "losses": b_wins, "decisive": dec,
        "draws": draws, "aborted": aborted,
        "wr": p, "ci": (lo, hi),
        "avg_move_ms": float(np.mean(move_ms)) if move_ms else 0.0,
        "max_move_ms": max_ms,
    }


def diff_ci(w1: int, n1: int, w2: int, n2: int) -> tuple[float, float, float]:
    """(delta, lo, hi) 95% CI for p1 - p2 (independent proportions, normal approx)."""
    if n1 == 0 or n2 == 0:
        return (0.0, 0.0, 0.0)
    p1, p2 = w1 / n1, w2 / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    return (d, d - 1.959963985 * se, d + 1.959963985 * se)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60, help="games per matchup (>=60)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--n-det", type=int, default=6)
    ap.add_argument("--epsilon", type=float, default=0.03)
    ap.add_argument("--strict", action="store_true",
                    help="narrow override: only attack-type options + Turn-End")
    args = ap.parse_args()

    engine = load_engine_data()
    arch = _read(ARCH_DECK)
    opp_decks = [_read(p) for p in OPP_DECK_FILES]

    # subjects (same Archaludon deck)
    baseline = ArchaludonAgent(arch, engine)
    vguided = ArchaludonVAgent(
        arch, engine, opp_decks=opp_decks,
        n_det=args.n_det, epsilon=args.epsilon, strict_attack=args.strict,
    )

    # realistic opponent set
    opponents: list[tuple[str, object]] = [("alakazam", AlakazamAgent(_read(ALA_DECK)))]
    for p in OPP_DECK_FILES[1:]:
        opponents.append((f"gp:{p.stem}", build_agent("greedy_plus", _read(p), engine)))

    print("== Increment-2 A/B: V-lookahead vs baseline Archaludon ==")
    print(f"games/matchup={args.games} n_det={args.n_det} epsilon={args.epsilon} "
          f"seed={args.seed}")
    print(f"opponents: {[n for n, _ in opponents]}\n")

    t0 = time.perf_counter()
    rows = []
    for name, opp in opponents:
        rb = match(baseline, opp, games=args.games, base_seed=args.seed)
        rv = match(vguided, opp, games=args.games, base_seed=args.seed)
        d, dlo, dhi = diff_ci(rv["wins"], rv["decisive"], rb["wins"], rb["decisive"])
        rows.append((name, rb, rv, d, dlo, dhi))
        print(f"[{name}]")
        print(f"  baseline : {rb['wr']:.3f}  CI[{rb['ci'][0]:.3f},{rb['ci'][1]:.3f}]"
              f"  ({rb['wins']}/{rb['decisive']}, {rb['aborted']} aborted)")
        print(f"  v-guided : {rv['wr']:.3f}  CI[{rv['ci'][0]:.3f},{rv['ci'][1]:.3f}]"
              f"  ({rv['wins']}/{rv['decisive']}, {rv['aborted']} aborted)")
        sep = "CI-separated" if (dlo > 0 or dhi < 0) else "within noise"
        print(f"  delta    : {d:+.3f}  95%CI[{dlo:+.3f},{dhi:+.3f}]  ({sep})\n")

    # aggregate across opponents
    bw = sum(r[1]["wins"] for r in rows)
    bn = sum(r[1]["decisive"] for r in rows)
    vw = sum(r[2]["wins"] for r in rows)
    vn = sum(r[2]["decisive"] for r in rows)
    bp, blo, bhi = wilson_interval(bw, bn)
    vp, vlo, vhi = wilson_interval(vw, vn)
    d, dlo, dhi = diff_ci(vw, vn, bw, bn)
    print("== AGGREGATE (all opponents pooled) ==")
    print(f"  baseline : {bp:.3f}  CI[{blo:.3f},{bhi:.3f}]  ({bw}/{bn})")
    print(f"  v-guided : {vp:.3f}  CI[{vlo:.3f},{vhi:.3f}]  ({vw}/{vn})")
    sep = "CI-separated" if (dlo > 0 or dhi < 0) else "within noise"
    print(f"  delta    : {d:+.3f}  95%CI[{dlo:+.3f},{dhi:+.3f}]  ({sep})\n")

    # head-to-head
    hh = match(vguided, baseline, games=args.games, base_seed=args.seed + 99999)
    print("== HEAD-TO-HEAD (v-guided A vs baseline Archaludon B, same deck) ==")
    lo_h, hi_h = hh["ci"]
    print(f"  v-guided win-rate: {hh['wr']:.3f}  CI[{lo_h:.3f},{hi_h:.3f}]"
          f"  ({hh['wins']}/{hh['decisive']}, {hh['draws']} draws)\n")

    print("== V-lookahead cost / activity ==")
    print(f"  lookahead frames fired : {vguided.n_lookahead}")
    print(f"  overrides applied      : {vguided.n_override}")
    if vguided.lookahead_times_ms:
        la = vguided.lookahead_times_ms
        print(f"  lookahead ms  avg={np.mean(la):.1f}  max={np.max(la):.1f}")
    if vguided.move_times_ms:
        mv = vguided.move_times_ms
        print(f"  per-move ms   avg={np.mean(mv):.2f}  max={np.max(mv):.1f}")
    print(f"\ntotal wall {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
