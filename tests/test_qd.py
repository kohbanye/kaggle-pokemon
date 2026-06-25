"""Unit tests for the QD deck core (operators + MAP-Elites archive), no engine."""

from __future__ import annotations

import numpy as np

from src.deck import CardInfo, CardPool, legality_errors
from src.qd import (
    MapElitesArchive,
    behaviour_descriptor,
    deck_stats,
    mutate,
    random_legal_deck,
)
from src.qd.deck_qd import (
    colour_count,
    energy_bin,
    energy_count,
    primary_colour,
)


def _pool() -> CardPool:
    """A tiny but deck-buildable pool: basics (3 colours), trainers, basic energy.

    Each row is ``CardInfo(id, name, supertype, stage, basic_pokemon, basic_energy,
    ace_spec, colour)``.
    """
    infos = [
        CardInfo(1, "PkR", "Pokemon", "Basic", True, False, False, "R"),
        CardInfo(2, "PkW", "Pokemon", "Basic", True, False, False, "W"),
        CardInfo(3, "PkG", "Pokemon", "Basic", True, False, False, "G"),
        CardInfo(4, "PkR2", "Pokemon", "Basic", True, False, False, "R"),
        CardInfo(10, "TrItem", "Trainer", "Item", False, False, False, ""),
        CardInfo(11, "TrSup", "Trainer", "Supporter", False, False, False, ""),
        CardInfo(12, "AceX", "Trainer", "Item", False, False, True, ""),
        CardInfo(20, "Fire Energy", "Energy", "Basic Energy", False, True, False, "R"),
        CardInfo(21, "Water Energy", "Energy", "Basic Energy", False, True, False, "W"),
    ]
    return CardPool({info.card_id: info for info in infos})


def test_random_legal_deck_is_legal() -> None:
    pool = _pool()
    rng = np.random.default_rng(0)
    for _ in range(20):
        deck = random_legal_deck(pool, rng)
        assert len(deck) == 60
        assert legality_errors(deck, pool) == []


def test_mutate_stays_legal_and_local() -> None:
    pool = _pool()
    rng = np.random.default_rng(1)
    deck = random_legal_deck(pool, rng)
    for _ in range(20):
        child = mutate(deck, pool, rng, n_swaps=3)
        assert legality_errors(child, pool) == []
        # Differs from the parent by at most ~2*n_swaps cards (multiset symmetric diff).
        before, after = sorted(deck), sorted(child)
        diff = sum((np.bincount(before, minlength=64)
                    - np.bincount(after, minlength=64)) != 0)
        assert diff <= 8


def test_energy_bin_edges() -> None:
    assert energy_bin(0) == 0
    assert energy_bin(8) == 0
    assert energy_bin(9) == 1
    assert energy_bin(16) == 2
    assert energy_bin(21) == 4


def test_behaviour_descriptor() -> None:
    pool = _pool()
    # 5 red basics + 10 fire energy + 45 water energy -> colour R (pokemon), energy 55.
    deck = [1] * 4 + [4] * 1 + [20] * 10 + [21] * 45
    colour, ebin = behaviour_descriptor(deck, pool)
    assert colour == "R"  # dominant Pokemon colour
    assert energy_count(deck, pool) == 55
    assert ebin == 4  # 55 > all edges
    assert primary_colour(deck, pool) == "R"
    assert deck_stats(deck, pool)["energy"] == 55


def test_colour_count_distinct_pokemon_colours() -> None:
    pool = _pool()
    assert colour_count([1, 4], pool) == 1  # both R
    assert colour_count([1, 2, 3], pool) == 3  # R, W, G
    assert colour_count([20, 21], pool) == 0  # energy is not a Pokemon colour


def test_colour_count_excludes_colourless() -> None:
    # A colourless ("C") Pokemon must not count toward the rainbow penalty.
    infos = [
        CardInfo(1, "PkR", "Pokemon", "Basic", True, False, False, "R"),
        CardInfo(2, "PkC", "Pokemon", "Basic", True, False, False, "C"),
    ]
    pool = CardPool({info.card_id: info for info in infos})
    assert colour_count([1, 2, 2], pool) == 1  # only R; C excluded


def test_archive_keeps_best_per_niche() -> None:
    arc = MapElitesArchive()
    assert arc.insert([1], 0.5, ("R", 1)) is True  # new niche
    assert arc.insert([2], 0.7, ("R", 1)) is True  # improves the niche
    assert arc.insert([3], 0.6, ("R", 1)) is False  # worse -> rejected
    assert arc.insert([4], 0.4, ("W", 2)) is True  # different niche
    assert arc.coverage == 2
    assert arc.cells[("R", 1)].deck == [2]
    best = arc.best()
    assert best is not None
    assert best.fitness == 0.7
    assert abs(arc.mean_fitness() - 0.55) < 1e-9


def test_archive_sample_is_seeded() -> None:
    arc = MapElitesArchive()
    for i in range(5):
        arc.insert([i], 0.1 * i, ("R", i))
    rng = np.random.default_rng(3)
    picks = {tuple(arc.sample(rng).deck) for _ in range(20)}
    assert len(picks) > 1  # actually random over the archive
