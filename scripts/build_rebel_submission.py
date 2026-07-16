"""Assemble the ReBeL Kaggle submission bundle: main.py + src/ + decklists/ + net + cg/.

Unlike the greedy bundle (build_submission.py), ReBeL is not a single inlined file -- it
ships the real ``src/`` packages, the belief hypothesis decklists (``decklists/`` +
``anchors`` + ``candidates``, read by src/rebel/belief.py), a value net
(``value_net.npz``), the ``deck.csv``, and ``cg/``. All at the archive ROOT so
``main.py`` can ``import src...`` / ``import cg`` from ``/kaggle_simulations/agent``.

  uv run python scripts/build_rebel_submission.py \
      --deck decklists/metal_aggro.csv --net data/rebel/vnd3_metal_r4.npz
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.deck import build_pool, legality_errors, load_deck_csv  # noqa: E402

SRC_MAIN = ROOT / "submission" / "main_rebel.py"
CG_SRC = ROOT / "data" / "sample_submission" / "cg"
# belief.py reads these (non-recursive glob of *.csv in each) -- ship exactly them.
HYP_DIRS = ("decklists", "decklists/anchors", "decklists/candidates")
OUT = ROOT / "build" / "submission_rebel"
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage the ReBeL submission bundle")
    ap.add_argument("--deck", type=Path, default=ROOT / "decklists" / "metal_aggro.csv")
    ap.add_argument("--net", type=Path,
                    default=ROOT / "data" / "rebel" / "vnd3_metal_r4.npz")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    deck = load_deck_csv(args.deck)
    errs = legality_errors(deck, build_pool())
    if errs:
        raise SystemExit(f"deck {args.deck} is illegal: {errs}")
    for p in (CG_SRC, SRC_MAIN, args.net):
        if not p.exists():
            raise SystemExit(f"missing required input: {p}")

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    shutil.copy(SRC_MAIN, args.out / "main.py")
    shutil.copy(args.deck, args.out / "deck.csv")
    shutil.copy(args.net, args.out / "value_net.npz")
    shutil.copytree(CG_SRC, args.out / "cg", ignore=_IGNORE)
    shutil.copytree(ROOT / "src", args.out / "src", ignore=_IGNORE)
    # belief hypothesis decklists (top-level csvs of each dir only; not the big subdirs)
    for d in HYP_DIRS:
        dst = args.out / d
        dst.mkdir(parents=True, exist_ok=True)
        for csv in sorted((ROOT / d).glob("*.csv")):
            shutil.copy(csv, dst / csv.name)

    n_hyp = sum(len(list((args.out / d).glob("*.csv"))) for d in HYP_DIRS)
    size_mb = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) / 1e6
    print(f"staged {args.out}")
    print(f"  deck={args.deck.name} ({len(deck)} cards, legal), net={args.net.name}, "
          f"{n_hyp} belief hypotheses, {size_mb:.1f} MB")
    print("verify it runs, then package + submit (needs your Kaggle credentials):")
    print(f"  tar -czf build/submission_rebel.tar.gz -C {args.out} .")
    print("  kaggle competitions submit -c pokemon-tcg-ai-battle "
          "-f build/submission_rebel.tar.gz -m 'ReBeL + metal_aggro'")


if __name__ == "__main__":
    main()
