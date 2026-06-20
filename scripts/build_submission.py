"""Assemble the Kaggle submission bundle: main.py + deck.csv + cg/ (native).

Copies the self-contained agent (``submission/main.py``), the chosen deck, and
the bundled engine into ``build/submission/`` ready to zip and upload. Validates
the deck is legal first. The actual upload is a manual step -- it needs your
Kaggle credentials -- so this only stages the bundle and prints the command.

  uv run python scripts/build_submission.py --deck decklists/metal_aggro.csv
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.deck import build_pool, legality_errors, load_deck_csv  # noqa: E402

SRC_MAIN = ROOT / "submission" / "main.py"
CG_SRC = ROOT / "data" / "sample_submission" / "cg"
DEFAULT_DECK = ROOT / "decklists" / "metal_aggro.csv"
OUT = ROOT / "build" / "submission"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage the submission bundle")
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    deck = load_deck_csv(args.deck)
    errors = legality_errors(deck, build_pool())
    if errors:
        raise SystemExit(f"deck {args.deck} is illegal: {errors}")
    if not CG_SRC.exists():
        raise SystemExit(f"engine not found at {CG_SRC} (run scripts/download_data.sh)")

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    shutil.copy(SRC_MAIN, args.out / "main.py")
    shutil.copy(args.deck, args.out / "deck.csv")
    shutil.copytree(
        CG_SRC, args.out / "cg", ignore=shutil.ignore_patterns("__pycache__"),
    )

    print(f"staged {args.out}  (deck={args.deck.name}, {len(deck)} cards, legal)")
    print("next (needs your Kaggle credentials):")
    print(f"  (cd {args.out} && zip -r ../submission.zip .)")
    print("  kaggle competitions submit -c pokemon-tcg-ai-battle "
          "-f build/submission.zip -m 'P1 greedy + metal_aggro'")


if __name__ == "__main__":
    main()
