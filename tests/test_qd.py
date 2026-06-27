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
    prize_bin,
    prize_points,
    ramp_ids,
    random_legal_deck_biased,
    setup_cost,
    single_prize_ids,
    speed_bin,
)


def _pool() -> CardPool:
    """A tiny but deck-buildable pool: basics (3 colours), trainers, basic energy.

    Basics carry prize-liability / setup-speed facts (``is_ex``/``is_mega``/
    ``min_attack_cost``) so the behaviour-descriptor axes can be exercised: ``PkR`` is a
    1-energy single-prize attacker, ``PkW`` a 2-energy ex, ``PkG`` a 3-energy Mega ex.
    """
    infos = [
        CardInfo(1, "PkR", "Pokemon", "Basic", True, False, False, "R",
                 min_attack_cost=1),
        CardInfo(2, "PkW", "Pokemon", "Basic", True, False, False, "W",
                 is_ex=True, min_attack_cost=2),
        CardInfo(3, "PkG", "Pokemon", "Basic", True, False, False, "G",
                 is_ex=True, is_mega=True, min_attack_cost=3),
        CardInfo(4, "PkR2", "Pokemon", "Basic", True, False, False, "R",
                 min_attack_cost=1),
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


def test_prize_points_and_bin() -> None:
    pool = _pool()
    # Pure single-prize deck (only PkR): 0 extra prize points -> bin 0.
    assert prize_points([1] * 4 + [20] * 56, pool) == 0
    assert prize_bin(0) == 0
    # 4 ex (PkW, +1 each) + 5 Mega (PkG, +2 each) = 4 + 10 = 14 extra points.
    assert prize_points([2] * 4 + [3] * 5, pool) == 14
    assert prize_bin(14) == 4  # > every edge (0,4,8,12)
    assert prize_bin(4) == 1  # 1..4 -> bin 1
    assert prize_bin(5) == 2


def test_setup_cost_and_speed_bin() -> None:
    pool = _pool()
    # Cheapest attacker present sets the speed: PkR (cost 1) -> aggro bin 0.
    assert setup_cost([1, 2, 3], pool) == 1
    assert speed_bin(1) == 0
    # Only the 3-energy Mega attacker -> ramp.
    assert setup_cost([3] * 4 + [20] * 56, pool) == 3
    assert speed_bin(3) == 2
    # No attacker at all (energy only) -> slowest niche.
    assert setup_cost([20, 21], pool) is None
    assert speed_bin(None) == len((1, 2, 3))


def test_single_prize_seed_reaches_prize_bin_0() -> None:
    pool = _pool()
    # PkR / PkR2 are the only non-ex / non-Mega Pokemon.
    assert set(single_prize_ids(pool)) == {1, 4}
    rng = np.random.default_rng(0)
    for _ in range(10):
        deck = random_legal_deck_biased(pool, rng, single_prize_ids(pool))
        assert legality_errors(deck, pool) == []
        assert prize_points(deck, pool) == 0  # no ex/Mega -> empty single-prize niche
        assert prize_bin(prize_points(deck, pool)) == 0


def test_ramp_seed_reaches_high_speed_bin() -> None:
    pool = _pool()
    # Only PkG (Mega, cheapest attack 3) qualifies as a ramp Pokemon at min_cost=3.
    assert set(ramp_ids(pool, min_cost=3)) == {3}
    rng = np.random.default_rng(1)
    for _ in range(10):
        deck = random_legal_deck_biased(pool, rng, ramp_ids(pool, min_cost=3))
        assert legality_errors(deck, pool) == []
        # cheapest attacker costs >= 3 -> ramp side (speed bin >= 2), not aggro bin 0.
        cost = setup_cost(deck, pool)
        assert cost is not None
        assert cost >= 3
        assert speed_bin(cost) >= 2


def test_behaviour_descriptor() -> None:
    pool = _pool()
    # Single-prize, 1-energy attacker deck -> (prize bin 0, speed bin 0).
    aggro = [1] * 4 + [4] * 1 + [20] * 55
    assert behaviour_descriptor(aggro, pool) == (0, 0)
    # Mega-heavy, 3-energy attacker deck -> high prize liability, slow.
    mega = [3] * 4 + [21] * 56
    pbin, sbin = behaviour_descriptor(mega, pool)
    assert pbin == 2  # 4 Mega -> 8 extra points; 8>0,8>4,not 8>8 -> bin 2
    assert sbin == 2  # cheapest attack costs 3 -> ramp
    assert deck_stats(aggro, pool)["prize_points"] == 0


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
