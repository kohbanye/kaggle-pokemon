"""Tests for the OSFP opponent pool (pure -- no engine, no torch).

Covers the meta-loop mechanics the Phase-5 self-play loop depends on: the
recency-weighted sampling (recent checkpoints heavier, scripted baselines kept at
a floor), the self-play probability, and the two checkpoint-admission paths
(beat-all-by-threshold and patience).
"""

import numpy as np

from src.net.osfp import OpponentPool


def test_baselines_only_uniform() -> None:
    pool = OpponentPool(["random", "greedy", "heuristic"], baseline_floor=0.1)
    weights = pool.weights(0)
    assert len(weights) == 3
    assert all(abs(p - 1 / 3) < 1e-9 for p in weights.values())
    assert pool.num_checkpoints == 0


def test_baseline_entry_shape() -> None:
    pool = OpponentPool(["random"])
    baselines = [e for e in pool.opponents() if e.kind == "baseline"]
    assert len(baselines) == 1
    assert baselines[0].ref == "random"
    assert baselines[0].iteration == -1  # baselines never decay


def test_recency_weight_favours_recent() -> None:
    pool = OpponentPool([], decay=0.5, self_play_prob=0.0, patience=1)
    assert pool.admit("old.npz", 1, {})  # patience=1 -> admit each iteration
    assert pool.admit("new.npz", 3, {})
    weights = pool.weights(3)
    old = next(p for e, p in weights.items() if e.ref == "old.npz")
    new = next(p for e, p in weights.items() if e.ref == "new.npz")
    assert new > old  # admitted more recently => heavier


def test_recency_off_is_uniform_over_checkpoints() -> None:
    # decay == 1 disables recency: every checkpoint weighs the same (ablation arm).
    pool = OpponentPool([], decay=1.0, self_play_prob=0.0, patience=1)
    pool.admit("a.npz", 1, {})
    pool.admit("b.npz", 3, {})
    values = list(pool.weights(3).values())
    assert abs(values[0] - values[1]) < 1e-9


def test_baseline_floor_retained_with_checkpoints() -> None:
    pool = OpponentPool(
        ["random"], decay=0.5, self_play_prob=0.0, baseline_floor=0.1, patience=1,
    )
    pool.admit("ckpt.npz", 1, {})
    weights = pool.weights(1)
    base = next(p for e, p in weights.items() if e.ref == "random")
    ckpt = next(p for e, p in weights.items() if e.ref == "ckpt.npz")
    # raw weights: baseline 0.1, fresh checkpoint 1.0 -> base normalises to 0.1/1.1.
    assert abs(base - 0.1 / 1.1) < 1e-9
    assert ckpt > base


def test_self_play_probability_honoured() -> None:
    pool = OpponentPool(["greedy"], self_play_prob=0.5)
    rng = np.random.default_rng(0)
    draws = [pool.sample(1, rng) for _ in range(2000)]
    none_frac = sum(d is None for d in draws) / len(draws)
    assert 0.45 < none_frac < 0.55  # ~half are self-play (None)


def test_sample_returns_opponent_when_no_self_play() -> None:
    pool = OpponentPool(["greedy", "random"], self_play_prob=0.0)
    rng = np.random.default_rng(0)
    picks = [pool.sample(1, rng) for _ in range(50)]
    assert all(p is not None for p in picks)
    assert {p.ref for p in picks if p is not None} <= {"greedy", "random"}


def test_admit_threshold_path() -> None:
    pool = OpponentPool(["greedy"], threshold=0.55, patience=100)
    # beats every opponent by > threshold -> admitted even though patience is far off.
    assert pool.admit("c1.npz", 1, {"greedy": 0.6, "random": 0.7})
    assert pool.num_checkpoints == 1
    # one opponent below threshold -> not admitted.
    assert not pool.admit("c2.npz", 2, {"greedy": 0.4, "random": 0.7})
    assert pool.num_checkpoints == 1


def test_admit_patience_path() -> None:
    pool = OpponentPool(["greedy"], threshold=0.99, patience=2)
    # iteration 1: 1 - 0 < 2 and win-rates don't clear 0.99 -> not admitted.
    assert not pool.admit("c1.npz", 1, {"greedy": 0.5})
    # iteration 2: 2 - 0 >= 2 -> patience admits regardless of win-rate.
    assert pool.admit("c2.npz", 2, {"greedy": 0.5})
    assert pool.num_checkpoints == 1


def test_empty_winrates_never_beat_all() -> None:
    pool = OpponentPool(["greedy"], threshold=0.55, patience=100)
    assert not pool.admit("c.npz", 1, {})  # no eval -> only patience can admit
    assert pool.num_checkpoints == 0
