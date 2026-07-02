"""Unit tests for the co-evolving gauntlet (src/qd/coevo.py) -- pure, no engine."""

from __future__ import annotations

import numpy as np

from src.qd import HallOfFame, MapElitesArchive, build_gauntlet

RNG = np.random.default_rng(0)


def _deck(base: int) -> list[int]:
    """A distinct fake 60-card list (contents only need to be hashable/comparable)."""
    return [base] * 60


def test_hof_caps_fifo() -> None:
    hof = HallOfFame(3)
    for i in range(5):
        hof.add(_deck(i), f"d{i}")
    assert [e.tag for e in hof.entries] == ["d2", "d3", "d4"]  # oldest evicted


def test_hof_add_copies_deck() -> None:
    hof = HallOfFame(2)
    deck = _deck(1)
    hof.add(deck, "a")
    deck[0] = 99  # caller mutates its list afterwards
    assert hof.entries[0].deck == _deck(1)


def test_hof_sample_without_replacement() -> None:
    hof = HallOfFame(10)
    for i in range(6):
        hof.add(_deck(i), f"d{i}")
    picked = hof.sample(4, RNG)
    assert len(picked) == 4
    assert len({e.tag for e in picked}) == 4
    # n >= len returns everything (order may be shuffled)
    assert {e.tag for e in hof.sample(99, RNG)} == {f"d{i}" for i in range(6)}


def _archive(fits: dict[int, float]) -> MapElitesArchive:
    arc = MapElitesArchive()
    for base, f in fits.items():
        arc.insert(_deck(base), f, descriptor=base)
    return arc


def test_build_gauntlet_elites_first_then_hof() -> None:
    arc = _archive({1: 0.9, 2: 0.5, 3: 0.7})
    hof = HallOfFame(8)
    for i in range(10, 14):
        hof.add(_deck(i), f"meta{i}")
    decks, tags = build_gauntlet(arc, hof, size=4, top_k=2, rng=RNG)
    assert len(decks) == len(tags) == 4
    # top-2 elites by fitness lead the gauntlet
    assert tags[0] == "elite:1"
    assert tags[1] == "elite:3"
    assert all(t.startswith("hof:meta") for t in tags[2:])


def test_build_gauntlet_dedups_hof_copies_of_elites() -> None:
    arc = _archive({1: 0.9})
    hof = HallOfFame(8)
    hof.add(_deck(1), "r1_best")  # the elite deck is also the newest HoF entry
    hof.add(_deck(2), "meta")
    decks, tags = build_gauntlet(arc, hof, size=4, top_k=2, rng=RNG)
    assert tags == ["elite:1", "hof:meta"]  # duplicate decklist skipped, no padding
    assert [d[0] for d in decks] == [1, 2]


def test_build_gauntlet_respects_size_cap() -> None:
    arc = _archive({i: i / 10 for i in range(1, 6)})
    hof = HallOfFame(16)
    for i in range(20, 32):
        hof.add(_deck(i), f"m{i}")
    decks, _tags = build_gauntlet(arc, hof, size=6, top_k=3, rng=RNG)
    assert len(decks) == 6
