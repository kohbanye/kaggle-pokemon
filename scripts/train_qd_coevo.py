"""Co-evolution orchestrator: QD decks <-> RL play (the "Option 1" loop).

Each round alternates the two halves, threading the play net and the deck archive:

1. **QD** (``qd_deck_search``): MAP-Elites illuminates the deck space *piloted by the
   current play net* -> a fresh archive of diverse, strong-under-current-play decks.
2. **RL play** (``train_paper_osfp --deck-pool``): self-play where both sides draw
   from that archive; train the play+value head only (``--no-deck-arm``) so it learns
   to pilot the *diverse* decks -> a new play net.

Repeat: better play makes the QD fitness sharper, a more diverse archive makes the
play generalise. This breaks the two failure modes the eval found -- deck-diversity
collapse (the archive can't collapse) and deck-specialised play (it trains across the
archive). Each half shells its own script (native, parallel).

  uv run python scripts/train_qd_coevo.py --rounds 4 --native --workers 14
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


def main() -> None:
    ap = argparse.ArgumentParser(description="QD<->RL co-evolution")
    ap.add_argument("--init-weights", type=Path,
                    default=ROOT / "data/paperosfp/main/paper_final.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "data/qdcoevo")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--native", action="store_true")
    # per-round sizes (forwarded to the two stage scripts)
    ap.add_argument("--qd-generations", type=int, default=20)
    ap.add_argument("--qd-init", type=int, default=48)
    ap.add_argument("--colour-penalty", type=float, default=0.03,
                    help="QD soft colour penalty (see qd_deck_search)")
    ap.add_argument("--rl-iterations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    net = args.init_weights
    for r in range(1, args.rounds + 1):
        rd = args.out / f"round_{r}"
        rd.mkdir(parents=True, exist_ok=True)
        archive = rd / "qd_archive.json"
        # 1) QD deck search piloted by the current play net.
        _run([py, str(ROOT / "scripts/qd_deck_search.py"),
              "--pilot", str(net), "--workers", str(args.workers),
              "--generations", str(args.qd_generations), "--init", str(args.qd_init),
              "--colour-penalty", str(args.colour_penalty),
              "--seed", str(args.seed + r), "--out", str(archive)])
        # 2) RL play training on the archive decks (battle-only).
        rl_out = rd / "rl"
        rl = [py, str(ROOT / "scripts/train_paper_osfp.py"),
              "--weights", str(net), "--deck-pool", str(archive),
              "--workers", str(args.workers), "--iterations", str(args.rl_iterations),
              "--seed", str(args.seed + r), "--out", str(rl_out)]
        if args.native:
            rl.append("--native")
        _run(rl)
        net = rl_out / "paper_final.npz"
        print(f"== round {r} done: net={net} archive={archive} ==", flush=True)

    print(f"== co-evolution done: final net={net} ==")


if __name__ == "__main__":
    main()
