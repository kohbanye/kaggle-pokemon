"""Generate the demonstration deck set and write it to ``decklists/`` (native).

Phase 1: builds the mono-type aggro archetypes (see ``src.deckbuild``) and writes
each as a one-id-per-line ``deck.csv`` for use as the deck-eval opponent pool and
the P4 behavioural-cloning prior. No engine needed.

  uv run python scripts/build_demo_decks.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.deck import build_pool, legality_errors  # noqa: E402
from src.deckbuild import build_demo_decks  # noqa: E402
from src.decklists import save_deck_csv  # noqa: E402

OUT_DIR = ROOT / "decklists"


def main() -> None:
    pool = build_pool()
    decks = build_demo_decks()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"writing {len(decks)} demo decks to {OUT_DIR}/")
    for name, deck in sorted(decks.items()):
        errors = legality_errors(deck, pool)
        if errors:
            raise SystemExit(f"{name} is illegal: {errors}")
        save_deck_csv(deck, OUT_DIR / f"{name}.csv")
        print(f"  {name}.csv  (60 cards, legal)")


if __name__ == "__main__":
    main()
