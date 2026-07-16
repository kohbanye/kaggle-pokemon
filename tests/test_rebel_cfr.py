"""Gate B: the sampled MCCFR estimator is unbiased == it reaches the KNOWN Kuhn Nash.

If external-sampling MCCFR converges to ~0 exploitability (the equilibrium exact CFR
finds), the counterfactual-regret estimator -- reach weighting, per-player ranges,
chance/opponent sampling -- is sound. The precondition (Codex review) for wiring the
determinized cg engine onto the same `Game` interface. Kuhn poker is the canonical CFR
benchmark: chance deal, asymmetric reach, information-revealing bet, analytic Nash.
"""

from __future__ import annotations

from src.rebel.cfr import (
    exploitability,
    solve_cfr,
    solve_mccfr_belief,
    solve_mccfr_external,
)
from src.rebel.game import KuhnPoker

_KUHN_GAME_VALUE_P0 = -1.0 / 18.0  # known Kuhn equilibrium value to player 0


def test_kuhn_structure() -> None:
    """Sanity: terminals, payoffs, and infoset keys are what CFR expects."""
    g = KuhnPoker()
    # deal (K=2 vs J=0), p0 bets, p1 folds -> p0 wins the ante (+1).
    h = g.next(g.next(g.root(), (2, 0)), "b")
    h = g.next(h, "p")
    assert g.is_terminal(h)
    assert g.utility(h) == 1.0
    # showdown pp with higher card wins 1.
    h2 = g.next(g.next(g.next(g.root(), (0, 2)), "p"), "p")
    assert g.is_terminal(h2)
    assert g.utility(h2) == -1.0  # p0 has J, loses
    # infoset hides the opponent card: key = own card + public actions.
    hk = g.next(g.root(), (1, 2))
    assert g.infoset_key(hk) == "1:"


def test_exact_cfr_reaches_nash() -> None:
    """Exact CFR drives exploitability to ~0 and hits the known game value."""
    g = KuhnPoker()
    avg = solve_cfr(g, iters=3000)
    assert exploitability(g, avg) < 0.01
    # game value to p0 under the equilibrium average strategy.
    from src.rebel.cfr import _expected_u0  # noqa: PLC0415
    assert abs(_expected_u0(g, avg, avg) - _KUHN_GAME_VALUE_P0) < 0.03


def test_mccfr_matches_exact_cfr_nash() -> None:
    """THE differential test: sampled MCCFR reaches the same ~0-exploitability Nash."""
    g = KuhnPoker()
    avg = solve_mccfr_external(g, iters=40000, seed=0)
    expl = exploitability(g, avg)
    assert expl < 0.05, f"MCCFR exploitability {expl:.4f} too high -- estimator biased"


def test_mccfr_belief_world_major_reaches_nash() -> None:
    """The WORLD-MAJOR solver (root chance enumerated) also reaches Kuhn's Nash -- a
    sound speed variant for the engine belief subgame, not just a heuristic."""
    g = KuhnPoker()
    avg, root_value = solve_mccfr_belief(g, sweeps=4000, traversals_per_world=2, seed=0)
    expl = exploitability(g, avg)
    assert expl < 0.06, f"world-major MCCFR exploitability {expl:.4f} too high"
    # the returned root value tracks Kuhn's known game value (~ -1/18).
    assert abs(root_value - (-1.0 / 18.0)) < 0.03


def test_mccfr_seed_robustness() -> None:
    """Convergence is not a lucky seed: two seeds both reach low exploitability."""
    g = KuhnPoker()
    for seed in (1, 2):
        expl = exploitability(g, solve_mccfr_external(g, iters=40000, seed=seed))
        assert expl < 0.06, f"seed {seed}: exploitability {expl:.4f}"
