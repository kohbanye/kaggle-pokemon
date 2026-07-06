"""QD<->RL co-evolution orchestrator (Step 5: both sides keep improving).

Each OUTER round alternates the two halves, threading the play net and the deck
archive through:

1. **QD** (``qd_deck_search --rounds K``): coevolving-gauntlet MAP-Elites (descriptor
   v3) piloted multi-pilot (greedy / current net / heuristic). Warm-started from the
   previous outer round's archive via ``--seed-archive`` (decks re-scored, hall-of-
   fame carried), so deck improvement compounds instead of restarting.
2. **RL play** (``train_paper_osfp --deck-pool``): self-play over the fresh archive's
   decks, training the play/value head only -> a stronger net, which then pilots
   (and thereby sharpens) the next QD round's fitness.

Deck-side opponents rise via the inner coevolution (elites + HoF); play-side rises
via RL on the ever-better archive -- the outer loop has no fixed component left to
equilibrate against, which is what makes more compute keep buying quality.

  uv run python scripts/train_qd_coevo.py --outer-rounds 6 --native --workers 14
  uv run python scripts/train_qd_coevo.py --smoke --native   # tiny end-to-end check
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(argv: list[str]) -> None:
    print(f"$ {' '.join(str(a) for a in argv)}", flush=True)
    subprocess.run(argv, check=True)  # noqa: S603


def main() -> None:  # noqa: C901, PLR0912, PLR0915 - CLI orchestrator
    ap = argparse.ArgumentParser(description="QD<->RL co-evolution (outer loop)")
    ap.add_argument("--init-weights", type=Path,
                    default=ROOT / "data/paperosfp/main/paper_final.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "data/qdrl")
    ap.add_argument("--outer-rounds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--native", action="store_true")
    # QD half (forwarded to qd_deck_search; inner gauntlet coevolution)
    ap.add_argument("--qd-rounds", type=int, default=8,
                    help="inner coevolution rounds per outer round")
    ap.add_argument("--qd-generations", type=int, default=25,
                    help="generations per inner round")
    ap.add_argument("--qd-init", type=int, default=64)
    ap.add_argument("--surrogate", action="store_true", default=True)
    ap.add_argument("--no-surrogate", dest="surrogate", action="store_false")
    ap.add_argument("--hof-size", type=int, default=32)
    ap.add_argument("--eval-timeout", type=float, default=45.0)
    ap.add_argument("--colour-penalty", type=float, default=0.03)
    ap.add_argument("--anchors", type=str, default="auto",
                    help="permanent gauntlet anchors (external grounding); "
                         "'auto' = every decklists/anchors/*.csv (the real "
                         "top-Elo ladder decks from build_heldout_v2) plus "
                         "our own ladder champion qd7_r5")
    ap.add_argument("--crossover-prob", type=float, default=0.0,
                    help="QD package-crossover child probability (see qd_deck_search)")
    ap.add_argument("--race-top", type=int, default=0,
                    help="QD racing finalists per generation (0 = off)")
    ap.add_argument("--race-factor", type=int, default=4)
    # RL half (forwarded to train_paper_osfp)
    ap.add_argument("--rl-iterations", type=int, default=300)
    ap.add_argument("--shaping-prize", type=float, default=0.0)
    ap.add_argument("--shaping-board", type=float, default=0.0)
    ap.add_argument("--deck-ctx-dim", type=int, default=0,
                    help="deck-conditioned play width for the RL half (0 = off); "
                         "the first round migrates the checkpoint zero-padded")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-archive", type=Path, default=None,
                    help="warm-start outer round 1's QD from this archive JSON "
                         "(e.g. a previous run's final round) -- improvement "
                         "compounds across runs, not just across rounds")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny sizes end-to-end (wiring check, ~minutes)")
    args = ap.parse_args()
    if args.anchors == "auto":
        ladder = sorted((ROOT / "decklists" / "anchors").glob("*.csv"))
        args.anchors = ",".join([str(p) for p in ladder]
                                + [str(ROOT / "decklists/candidates/qd7_r5.csv")])
    if args.smoke:
        args.outer_rounds, args.qd_rounds, args.qd_generations = 1, 2, 2
        args.qd_init, args.rl_iterations = 12, 0  # rl --smoke drives its own sizes
    args.out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    net = args.init_weights
    prev_archive: Path | None = args.seed_archive
    for r in range(1, args.outer_rounds + 1):
        rd = args.out / f"round_{r}"
        rd.mkdir(parents=True, exist_ok=True)
        archive = rd / "qd_archive.json"
        # 1) QD deck search (inner coevo gauntlet) piloted by the current net.
        qd = [py, str(ROOT / "scripts/qd_deck_search.py"),
              "--pilot", str(net), "--workers", str(args.workers),
              "--rounds", str(args.qd_rounds),
              "--generations", str(args.qd_generations),
              "--init", str(args.qd_init),
              "--hof-size", str(args.hof_size),
              "--eval-timeout", str(args.eval_timeout),
              "--colour-penalty", str(args.colour_penalty),
              "--seed", str(args.seed + r), "--out", str(archive)]
        if args.surrogate:
            qd.append("--surrogate")
        if args.anchors:
            qd += ["--anchors", args.anchors]
        if args.crossover_prob > 0:
            qd += ["--crossover-prob", str(args.crossover_prob)]
        if args.race_top > 0:
            qd += ["--race-top", str(args.race_top),
                   "--race-factor", str(args.race_factor)]
        if args.smoke:
            qd += ["--batch", "8", "--n-games", "2"]
        if prev_archive is not None:
            qd += ["--seed-archive", str(prev_archive)]
        _run(qd)
        prev_archive = archive
        # 2) RL play training on the archive decks (battle-only).
        rl_out = rd / "rl"
        rl = [py, str(ROOT / "scripts/train_paper_osfp.py"),
              "--weights", str(net), "--deck-pool", str(archive),
              "--workers", str(args.workers),
              "--iterations", str(args.rl_iterations),
              "--seed", str(args.seed + r), "--out", str(rl_out)]
        if args.native:
            rl.append("--native")
        if args.smoke:
            rl.append("--smoke")
        if args.deck_ctx_dim > 0:
            rl += ["--deck-ctx-dim", str(args.deck_ctx_dim)]
        if args.shaping_prize or args.shaping_board:
            rl += ["--shaping-prize", str(args.shaping_prize),
                   "--shaping-board", str(args.shaping_board)]
        _run(rl)
        net = rl_out / "paper_final.npz"
        print(f"== outer round {r} done: net={net} archive={archive} ==", flush=True)

    print(f"== QD<->RL co-evolution done: final net={net} "
          f"archive={prev_archive} ==")


if __name__ == "__main__":
    main()
