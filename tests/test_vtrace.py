"""Unit tests for the V-Trace correction + PPO surrogate (``src/net/vtrace.py``).

The engine is not needed -- this pins the pure-numpy RL math against its defining
properties (on-policy == Monte-Carlo, clipping, padding-invariance).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.net.vtrace import ppo_policy_loss, vtrace

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def _logp(probs: ArrayLike) -> NDArray[np.float64]:
    return np.log(np.asarray(probs, dtype=np.float64))


def test_on_policy_equals_monte_carlo_return() -> None:
    """ratio==1, gamma==1, terminal-only reward => every v_t is the MC return."""
    t = 5
    reward = np.zeros((1, t))
    reward[0, -1] = 1.0  # win at the last step
    values = np.array([[0.2, -0.1, 0.5, 0.3, 0.9]])
    logp = np.zeros((1, t))  # behaviour == target => ratio 1
    valid = np.ones((1, t), dtype=bool)

    out = vtrace(logp, logp, values, reward, valid, gamma=1.0)

    # MC return for terminal +1 with gamma 1 is +1 at every step.
    np.testing.assert_allclose(out.vs[0], np.ones(t), atol=1e-9)
    # PG advantage = return - V(s_t) = 1 - V.
    np.testing.assert_allclose(out.pg_advantages[0], 1.0 - values[0], atol=1e-9)


def test_on_policy_discounted_return() -> None:
    """ratio==1, gamma<1 => v_t is the discounted MC return of future rewards."""
    gamma = 0.9
    rewards = np.array([[0.0, 1.0, 0.0, 2.0]])
    valid = np.ones((1, 4), dtype=bool)
    values = np.zeros((1, 4))
    logp = np.zeros((1, 4))

    out = vtrace(logp, logp, values, rewards, valid, gamma=gamma)

    # v_t = sum_{k>=t} gamma^{k-t} r_k  (V==0, ratio==1).
    expected = np.array([
        1.0 * gamma + 2.0 * gamma**3,
        1.0 + 2.0 * gamma**2,
        2.0 * gamma,
        2.0,
    ])
    np.testing.assert_allclose(out.vs[0], expected, atol=1e-9)


def test_padding_invariance() -> None:
    """A padded short sequence must match computing it at its true length."""
    logp = np.zeros((1, 3))
    values = np.array([[0.1, -0.2, 0.4]])
    rewards = np.array([[0.0, 0.0, 1.0]])
    valid = np.ones((1, 3), dtype=bool)
    short = vtrace(logp, logp, values, rewards, valid, gamma=1.0)

    # Same sequence, padded to length 5.
    logp_p = np.zeros((1, 5))
    values_p = np.array([[0.1, -0.2, 0.4, 7.0, -3.0]])  # garbage in pad slots
    rewards_p = np.array([[0.0, 0.0, 1.0, 9.0, 9.0]])  # garbage in pad slots
    valid_p = np.array([[True, True, True, False, False]])
    padded = vtrace(logp_p, logp_p, values_p, rewards_p, valid_p, gamma=1.0)

    np.testing.assert_allclose(padded.vs[0, :3], short.vs[0], atol=1e-9)
    np.testing.assert_allclose(padded.pg_advantages[0, :3], short.pg_advantages[0])
    # Pad steps contribute nothing.
    assert padded.vs[0, 3:].tolist() == [0.0, 0.0]
    assert padded.pg_advantages[0, 3:].tolist() == [0.0, 0.0]


def test_rho_upper_clip_caps_correction() -> None:
    """A target far above behaviour saturates rho at clip_rho."""
    behaviour = _logp([[0.01, 0.01]])
    target = _logp([[0.99, 0.99]])  # ratio ~ 99 >> 1
    values = np.zeros((1, 2))
    rewards = np.array([[0.0, 1.0]])
    valid = np.ones((1, 2), dtype=bool)

    out = vtrace(behaviour, target, values, rewards, valid, clip_rho=1.0)
    np.testing.assert_allclose(out.clipped_rho[0], [1.0, 1.0], atol=1e-9)


def test_rho_lower_clip() -> None:
    """rho_min floors the correction so a tiny ratio can't vanish the trace."""
    behaviour = _logp([[0.99]])
    target = _logp([[0.001]])  # ratio ~ 0.001
    values = np.zeros((1, 1))
    rewards = np.array([[1.0]])
    valid = np.ones((1, 1), dtype=bool)

    floored = vtrace(behaviour, target, values, rewards, valid, rho_min=0.5)
    assert floored.clipped_rho[0, 0] == 0.5
    unfloored = vtrace(behaviour, target, values, rewards, valid, rho_min=0.0)
    assert unfloored.clipped_rho[0, 0] < 0.01


def test_two_sequences_independent() -> None:
    """Batched rows of different lengths don't bleed into each other."""
    logp = np.zeros((2, 4))
    values = np.zeros((2, 4))
    rewards = np.array([
        [0.0, 0.0, 1.0, 0.0],   # len 3, win
        [0.0, -1.0, 0.0, 0.0],  # len 2, loss
    ])
    valid = np.array([
        [True, True, True, False],
        [True, True, False, False],
    ])
    out = vtrace(logp, logp, values, rewards, valid, gamma=1.0)
    np.testing.assert_allclose(out.vs[0, :3], [1.0, 1.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(out.vs[1, :2], [-1.0, -1.0], atol=1e-9)


def test_ppo_surrogate_sign_and_clip() -> None:
    """PPO loss rewards raising prob on +adv and clips the ratio gain."""
    valid = np.ones((1, 1), dtype=bool)
    adv = np.array([[1.0]])
    # ratio 1 => loss = -adv = -1.
    same = _logp([[0.5]])
    assert ppo_policy_loss(same, same, adv, valid, clip_eps=0.2) == -1.0
    # ratio >> 1 with +adv is clipped to (1+eps)*adv => loss = -(1.2).
    hi = ppo_policy_loss(_logp([[0.9]]), _logp([[0.1]]), adv, valid, clip_eps=0.2)
    np.testing.assert_allclose(hi, -1.2, atol=1e-9)
