"""Probe the cg engine's deck-construction rules (Linux x86-64 only / Docker).

Phase 1 needs to *confirm* which deck rules the engine actually enforces (and the
meaning of ``StartData.errorType``), so ``src.deck`` encodes the right rules and
the CB-head legality mask is sound. We feed ``battle_start`` a legal sample deck
plus several deliberately-broken variants and report, for each, what our local
validator says vs. what the engine does (accept / reject + errorPlayer/errorType).

Run inside the Docker image (see README):
    docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim \
        python scripts/probe_deck_legality.py
"""

import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CG_PARENT = ROOT / "data" / "sample_submission"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CG_PARENT))

from cg.game import battle_finish, battle_start  # noqa: E402

from src.deck import (  # noqa: E402
    MAX_COPIES_BY_NAME,
    CardPool,
    build_pool,
    legality_errors,
    load_deck_csv,
    random_legal_deck,
)


def _with_extra_copy(deck: list[int], pool: CardPool) -> list[int]:
    """Make a 5th copy of a card already at the 4-copy cap (replacing an energy)."""
    counts = Counter(deck)
    target = next(
        c
        for c, n in counts.items()
        if n == MAX_COPIES_BY_NAME and not pool.cards[c].is_basic_energy
    )
    pos = next(i for i, c in enumerate(deck) if pool.cards[c].is_basic_energy)
    out = list(deck)
    out[pos] = target
    return out


def _no_basic_pokemon(deck: list[int], pool: CardPool) -> list[int]:
    """Replace every Basic Pokemon with a Basic Energy (keeps 60 cards)."""
    energy = next(c for c in deck if pool.cards[c].is_basic_energy)
    return [energy if pool.cards[c].is_basic_pokemon else c for c in deck]


def _two_ace_spec(deck: list[int], pool: CardPool) -> list[int]:
    """Add a second, distinct ACE SPEC (replacing an energy)."""
    present = set(deck)
    extra = next(
        c for c, info in pool.cards.items() if info.is_ace_spec and c not in present
    )
    pos = next(i for i, c in enumerate(deck) if pool.cards[c].is_basic_energy)
    out = list(deck)
    out[pos] = extra
    return out


def check(label: str, deck: list[int], pool: CardPool, opponent: list[int]) -> None:
    local = legality_errors(deck, pool)
    print(f"\n[{label}] len={len(deck)} local_legal={not local}")
    if local:
        print(f"  local errors: {local}")
    try:
        obs, start = battle_start(deck, opponent)
    except ValueError as exc:  # wrapper rejects (e.g. size != 60) before the engine
        print(f"  engine: battle_start raised ValueError: {exc}")
        return
    if obs is None:
        print(
            f"  engine: REJECTED  errorPlayer={start.errorPlayer} "
            f"errorType={start.errorType}",
        )
    else:
        print(f"  engine: ACCEPTED (errorType={start.errorType})")
        battle_finish()


def main() -> None:
    pool = build_pool()
    sample = load_deck_csv(CG_PARENT / "deck.csv")
    print(f"pool: {len(pool.cards)} cards; sample deck: {len(sample)} cards")

    check("legal sample", sample, pool, sample)
    check("5th copy by name", _with_extra_copy(sample, pool), pool, sample)
    check("no basic pokemon", _no_basic_pokemon(sample, pool), pool, sample)
    check("two ACE SPEC", _two_ace_spec(sample, pool), pool, sample)
    check("59 cards (size)", sample[:-1], pool, sample)
    rng = random.Random(0)  # noqa: S311 - gameplay randomness, not crypto
    check("random full-pool legal", random_legal_deck(pool, rng), pool, sample)


if __name__ == "__main__":
    main()
