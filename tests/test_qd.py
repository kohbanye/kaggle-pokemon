"""Unit tests for the QD deck core (operators + MAP-Elites archive), no engine."""

from __future__ import annotations

from collections import Counter

import numpy as np

from src.deck import MAX_COPIES_BY_NAME, CardInfo, CardPool, legality_errors
from src.qd import (
    MapElitesArchive,
    behaviour_descriptor,
    card_role,
    deck_stats,
    mutate,
    random_legal_deck,
)
from src.qd.deck_qd import (
    _energy_block_adjust,
    _evo_line_edit,
    _package_swap,
    colour_count,
    crossover,
    energy_bin,
    energy_count,
    evo_bin,
    evo_line_ids,
    evolution_depth,
    pl_bin,
    prize_bin,
    prize_liability,
    prize_points,
    ramp_ids,
    random_legal_deck_biased,
    setup_cost,
    single_prize_ids,
    speed_bin,
    toolbox_bin,
    toolbox_breadth,
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


def _role_pool() -> CardPool:
    """A role-diverse pool for the heuristic-mutation tests (Step 3).

    Distinct from ``_pool()`` (whose ``single_prize_ids``/``ramp_ids`` set-equality
    tests pin the exact Pokemon), so adding cards here can't perturb those. Covers every
    ``card_role`` bucket: attacker vs support Basic, a Stage 1, all four Trainer
    sub-types, and Basic vs Special Energy. Basic Energy (4-copy-cap exempt) keeps a
    legal 60 always completable.
    """
    infos = [
        CardInfo(1, "AtkR", "Pokemon", "Basic Pokémon", True, False, False, "R",
                 min_attack_cost=1),
        CardInfo(2, "AtkW", "Pokemon", "Basic Pokémon", True, False, False, "W",
                 min_attack_cost=2),
        CardInfo(3, "Support", "Pokemon", "Basic Pokémon", True, False, False, "P"),
        CardInfo(4, "Evo1", "Pokemon", "Stage 1 Pokémon", False, False, False, "R",
                 min_attack_cost=2),
        CardInfo(10, "TrItem", "Trainer", "Item", False, False, False, ""),
        CardInfo(11, "TrSup", "Trainer", "Supporter", False, False, False, ""),
        CardInfo(12, "TrTool", "Trainer", "Pokémon Tool", False, False, False, ""),
        CardInfo(13, "TrStad", "Trainer", "Stadium", False, False, False, ""),
        CardInfo(20, "Fire Energy", "Energy", "Basic Energy", False, True, False, "R"),
        CardInfo(21, "Water Energy", "Energy", "Basic Energy", False, True, False, "W"),
        CardInfo(22, "Rainbow Energy", "Energy", "Special Energy", False, False, False,
                 "C"),
    ]
    return CardPool({info.card_id: info for info in infos})


def _evo_pool() -> CardPool:
    """A pool with a full Basic -> Stage 1 -> Stage 2 line (for the evo-line tests).

    Dedicated pool (like ``_role_pool``) so the evolution links can't perturb the
    ``_pool()`` set-equality tests. ``Zard`` evolves from ``Meleon`` evolves from
    ``Mander``; ``Solo`` is an unrelated Basic attacker; Basic Energy keeps any
    60 completable.
    """
    infos = [
        CardInfo(1, "Mander", "Pokemon", "Basic Pokémon", True, False, False, "R",
                 min_attack_cost=1),
        CardInfo(2, "Meleon", "Pokemon", "Stage 1 Pokémon", False, False, False, "R",
                 min_attack_cost=2, evolves_from="Mander"),
        CardInfo(3, "Zard", "Pokemon", "Stage 2 Pokémon", False, False, False, "R",
                 is_ex=True, min_attack_cost=3, evolves_from="Meleon"),
        CardInfo(4, "Solo", "Pokemon", "Basic Pokémon", True, False, False, "W",
                 min_attack_cost=1),
        CardInfo(10, "TrItem", "Trainer", "Item", False, False, False, ""),
        CardInfo(20, "Fire Energy", "Energy", "Basic Energy", False, True, False, "R"),
    ]
    return CardPool({info.card_id: info for info in infos})


def test_evo_line_ids_walks_the_chain() -> None:
    pool = _evo_pool()
    assert evo_line_ids(pool, 3) == [[1], [2], [3]]  # Basic first
    assert evo_line_ids(pool, 2) == [[1], [2]]
    assert evo_line_ids(pool, 1) == [[1]]  # a Basic is its own line


def test_evo_line_edit_adds_and_removes_coherent_lines() -> None:
    pool = _evo_pool()
    rng = np.random.default_rng(7)
    base = [4] * 4 + [20] * 56  # no evolution cards -> the op can only ADD a line
    added = removed = False
    for _ in range(60):
        out = _evo_line_edit(base, pool, rng)
        assert legality_errors(out, pool) == [] or len(out) < 60  # legal prefix
        c = Counter(out)
        if c[3]:  # Stage 2 present -> its whole chain must be present
            assert c[2], "orphan Stage 2 (Stage 1 missing)"
            assert c[1], "orphan Stage 2 (Basic missing)"
            added = True
    assert added
    with_line = [4] * 4 + [1, 1, 2, 2, 3, 3] + [20] * 50
    for _ in range(60):
        out = _evo_line_edit(with_line, pool, rng)
        c = Counter(out)
        if not c[3] and not c[2] and not c[1]:
            removed = True  # the whole line went, not a partial strand
        # never a partial removal that leaves an orphan Stage 2
        if c[3]:
            assert c[2]
            assert c[1]
        # attacker guard: some attacker always survives
        assert any(card_role(pool.cards[x])[2] for x in out)
    assert removed


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
    # Single-prize 1-energy Basic attackers, 2 species -> all-zero niche.
    aggro = [1] * 4 + [4] * 1 + [20] * 55
    assert behaviour_descriptor(aggro, pool) == (0, 0, 0, 0)
    # Mega-only attacker deck -> max prize liability (3.0), ramp speed.
    mega = [3] * 4 + [21] * 56
    pbin, sbin, ebin, tbin = behaviour_descriptor(mega, pool)
    assert pbin == 4  # role-weighted liability 3.0 > every PL edge
    assert sbin == 2  # cheapest attack costs 3 -> ramp
    assert ebin == 0  # all Basic
    assert tbin == 0  # single attacker species
    assert deck_stats(aggro, pool)["prize_points"] == 0


def test_prize_liability_role_weighting() -> None:
    """A support ex weighs 0.25 vs an attacker's 1.0 in the liability average."""
    infos = [
        CardInfo(1, "Atk1", "Pokemon", "Basic Pokémon", True, False, False, "R",
                 min_attack_cost=1),
        CardInfo(2, "SupEx", "Pokemon", "Basic Pokémon", True, False, False, "P",
                 is_ex=True),  # no attack -> support role
        CardInfo(3, "AtkEx", "Pokemon", "Basic Pokémon", True, False, False, "W",
                 is_ex=True, min_attack_cost=2),
    ]
    pool = CardPool({i.card_id: i for i in infos})
    assert prize_liability([1, 1], pool) == 1.0  # pure single-prize
    # 2 single-prize attackers + 2 support ex: (2*1 + 0.25*2*2)/(2 + 0.25*2) = 1.2
    assert abs(prize_liability([1, 1, 2, 2], pool) - 1.2) < 1e-9
    # The same ex as ATTACKERS doubles their pull: (2*1 + 2*2)/4 = 1.5
    assert abs(prize_liability([1, 1, 3, 3], pool) - 1.5) < 1e-9
    assert pl_bin(1.0) == 0
    assert pl_bin(1.5) == 1
    assert pl_bin(2.0) == 3
    assert pl_bin(3.0) == 4


def test_evolution_depth_and_bin() -> None:
    pool = _evo_pool()
    assert evolution_depth([1, 1, 20], pool) == 0.0  # Basics only
    # 2 Basic + 1 Stage1 + 1 Stage2 -> (0+0+1+2)/4 = 0.75
    assert abs(evolution_depth([1, 1, 2, 3], pool) - 0.75) < 1e-9
    assert evo_bin(0.0) == 0
    assert evo_bin(0.2) == 1
    assert evo_bin(0.4) == 2
    assert evo_bin(0.75) == 3


def test_toolbox_breadth_counts_attacker_species() -> None:
    pool = _pool()
    assert toolbox_breadth([1] * 4 + [20] * 10, pool) == 1
    assert toolbox_breadth([1, 2, 3, 4], pool) == 4  # PkR, PkW, PkG, PkR2
    assert toolbox_bin(1) == 0
    assert toolbox_bin(4) == 1
    assert toolbox_bin(8) == 2


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


def test_card_role_splits_by_stage_and_attack() -> None:
    pool = _role_pool()
    role = {cid: card_role(info) for cid, info in pool.cards.items()}
    # attacker vs support Basic Pokemon of the same stage are distinct roles.
    assert role[1] == ("Pokemon", "Basic Pokémon", True)
    assert role[3] == ("Pokemon", "Basic Pokémon", False)
    assert role[1] != role[3]
    # stage matters: a Stage 1 attacker != a Basic attacker.
    assert role[4] == ("Pokemon", "Stage 1 Pokémon", True)
    assert role[4] != role[1]
    # all four Trainer sub-types are distinct roles.
    assert len({role[10], role[11], role[12], role[13]}) == 4
    # Basic vs Special Energy are distinct roles.
    assert role[20] != role[22]


def test_mutate_heuristic_stays_legal() -> None:
    pool = _role_pool()
    rng = np.random.default_rng(2)
    deck = random_legal_deck(pool, rng)
    for _ in range(50):
        child = mutate(deck, pool, rng, n_swaps=4, strategy="heuristic")
        assert len(child) == 60
        assert legality_errors(child, pool) == []


def test_mutate_heuristic_locality_bounded() -> None:
    """Heuristic edits stay local, but looser than the random swap: a package op can
    move a whole 4-copy playset per unit, so the multiset diff is bounded by
    ``n_swaps * 2 * MAX_COPIES_BY_NAME`` (removals + insertions), not ``2 * n_swaps``.
    """
    pool = _role_pool()
    rng = np.random.default_rng(3)
    deck = random_legal_deck(pool, rng)
    n_swaps = 3
    bound = n_swaps * 2 * MAX_COPIES_BY_NAME
    for _ in range(30):
        child = mutate(deck, pool, rng, n_swaps=n_swaps, strategy="heuristic")
        before = np.bincount(deck, minlength=32)
        after = np.bincount(child, minlength=32)
        assert int(np.abs(before - after).sum()) <= bound


def test_mutate_random_strategy_unchanged() -> None:
    """``strategy="random"`` reproduces the Step-1 operator's tight locality bound."""
    pool = _role_pool()
    rng = np.random.default_rng(6)
    deck = random_legal_deck(pool, rng)
    for _ in range(20):
        child = mutate(deck, pool, rng, n_swaps=3, strategy="random")
        assert legality_errors(child, pool) == []
        before = np.bincount(deck, minlength=32)
        after = np.bincount(child, minlength=32)
        assert int((np.abs(before - after) != 0).sum()) <= 8  # ~2 * n_swaps positions


def test_package_swap_removes_whole_playset_and_keeps_attacker() -> None:
    pool = _role_pool()
    rng = np.random.default_rng(4)
    # 4-copy playset of the ONLY attacker (id 1); the rest support / energy.
    keep = [1, 1, 1, 1, 3, 3] + [20] * 54
    for _ in range(50):
        out = _package_swap(keep, pool, rng, swap_prob=0.5)
        before, after = Counter(keep), Counter(out)
        # A package op removes ALL copies of one id -- never a partial 1..3 left.
        for cid, cnt in before.items():
            if after.get(cid, 0) < cnt:
                assert after.get(cid, 0) == 0
        # No non-Basic-Energy id ever exceeds the 4-copy cap.
        for cid, cnt in after.items():
            if not pool.cards[cid].is_basic_energy:
                assert cnt <= MAX_COPIES_BY_NAME
        # Sole-attacker guard: an attacker-role card always remains.
        assert any(card_role(pool.cards[c])[2] for c in out)


def test_energy_block_adjust_trends_toward_range() -> None:
    pool = _role_pool()
    rng = np.random.default_rng(5)
    # Far above the 8-15 range: each call cuts energy (never adds while above).
    over = [20] * 44 + [1] * 4 + [10] * 4 + [11] * 4 + [12] * 4  # 60 cards, energy = 44
    start = energy_count(over, pool)
    prev = start
    for _ in range(20):
        over = _energy_block_adjust(over, pool, rng)
        cur = energy_count(over, pool)
        assert cur <= prev  # monotone non-increasing above the range
        prev = cur
    assert prev < start  # made downward progress
    # Below the range: adds Basic Energy (exercised on a crafted short list).
    under = [1, 2, 3, 4, 10, 11] + [20] * 2  # energy = 2, below lo = 8
    assert energy_count(_energy_block_adjust(under, pool, rng), pool) >= energy_count(
        under, pool,
    )


def test_crossover_legal_and_line_atomic() -> None:
    pool = _evo_pool()
    rng = np.random.default_rng(8)
    # Parent A: the evolution line + energy; parent B: Solo aggro + items.
    a = [1, 1, 2, 2, 3, 3] + [20] * 54
    b = [4, 4, 4, 4] + [10, 10, 10, 10] + [20] * 52
    for _ in range(40):
        child = crossover(a, b, pool, rng)
        assert len(child) == 60
        assert legality_errors(child, pool) == []
        c = Counter(child)
        # Line atomicity: a Stage 2 never arrives without its whole chain.
        if c[3]:
            assert c[2]
            assert c[1]


def test_crossover_mixes_parent_packages() -> None:
    pool = _evo_pool()
    rng = np.random.default_rng(9)
    a = [1, 1, 2, 2, 3, 3] + [20] * 54  # line, no items
    b = [4, 4, 4, 4] + [10, 10, 10, 10] + [20] * 52  # items, no line
    mixed = False
    for _ in range(60):
        c = Counter(crossover(a, b, pool, rng))
        if c[3] and c[10]:  # A's Stage-2 line AND B's item package together
            mixed = True
            break
    assert mixed, "crossover never combined packages from both parents"
