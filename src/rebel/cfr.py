"""Exact CFR and external-sampling MCCFR on the `Game` interface, plus exact
exploitability by best-response enumeration (Gate B kill-test).

The DIFFERENTIAL TEST demanded by the Codex review: the sampled estimator
(`solve_mccfr_external`) must converge to the SAME Nash as exact CFR (`solve_cfr`) --
i.e. its counterfactual-regret estimates are unbiased. We verify that on Kuhn poker
(`src.rebel.game.KuhnPoker`), whose Nash exploitability is 0. If MCCFR reaches ~0
exploitability the estimator math (reach weighting, per-player ranges, sampling
chance+opponent) is sound -- the precondition for wiring the engine onto this interface.

Pure numpy/Python, engine-free, unit-tested.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from src.rebel.game import CHANCE

if TYPE_CHECKING:
    from src.rebel.game import Game

Strategy = dict[str, dict[object, float]]


def _regret_match(regret: dict[object, float],
                  actions: list[object]) -> dict[object, float]:
    pos = {a: max(regret.get(a, 0.0), 0.0) for a in actions}
    s = sum(pos.values())
    if s <= 0:
        return {a: 1.0 / len(actions) for a in actions}
    return {a: pos[a] / s for a in actions}


def _average(strat_sum: dict[str, dict[object, float]]) -> Strategy:
    out: Strategy = {}
    for i, row in strat_sum.items():
        s = sum(row.values())
        out[i] = ({a: v / s for a, v in row.items()} if s > 0
                  else {a: 1.0 / len(row) for a in row})
    return out


# --- exact vanilla CFR --------------------------------------------------------

def solve_cfr(game: Game, iters: int) -> Strategy:
    """Exact (full-tree, full-chance) CFR; returns the average strategy."""
    regret: dict[str, dict[object, float]] = defaultdict(lambda: defaultdict(float))
    strat_sum: dict[str, dict[object, float]] = defaultdict(lambda: defaultdict(float))

    def walk(h: tuple, p0: float, p1: float, pc: float) -> float:
        if game.is_terminal(h):
            return game.utility(h)
        pl = game.current_player(h)
        if pl == CHANCE:
            return sum(pr * walk(game.next(h, a), p0, p1, pc * pr)
                       for a, pr in game.chance_outcomes(h))
        info = game.infoset_key(h)
        actions = game.legal_actions(h)
        sigma = _regret_match(regret[info], actions)
        util = {}
        v = 0.0
        for a in actions:
            nh = game.next(h, a)
            util[a] = (walk(nh, p0 * sigma[a], p1, pc) if pl == 0
                       else walk(nh, p0, p1 * sigma[a], pc))
            v += sigma[a] * util[a]
        reach_self, cf = (p0, p1 * pc) if pl == 0 else (p1, p0 * pc)
        for a in actions:
            # regret in the ACTING player's own utility (player 1's is -u0).
            gain = (util[a] - v) if pl == 0 else (v - util[a])
            regret[info][a] += cf * gain
            strat_sum[info][a] += reach_self * sigma[a]
        return v

    for _ in range(iters):
        walk(game.root(), 1.0, 1.0, 1.0)
    return _average(strat_sum)


# --- external-sampling MCCFR (the estimator under test) -----------------------

def solve_mccfr_external(game: Game, iters: int, seed: int = 0) -> Strategy:
    """External-sampling MCCFR (Lanctot 2009): sample chance + the opponent, traverse
    all of the traverser's actions. Average strategy accumulates at opponent infosets.
    Must converge to the same Nash as `solve_cfr` — that convergence IS the unbiasedness
    proof for the sampled counterfactual-regret estimator."""
    regret: dict[str, dict[object, float]] = defaultdict(lambda: defaultdict(float))
    strat_sum: dict[str, dict[object, float]] = defaultdict(lambda: defaultdict(float))
    rng = np.random.default_rng(seed)

    def walk(h: tuple, i: int) -> float:
        if game.is_terminal(h):
            u0 = game.utility(h)
            return u0 if i == 0 else -u0
        pl = game.current_player(h)
        if pl == CHANCE:
            outs = game.chance_outcomes(h)
            probs = np.array([p for _, p in outs])
            a = outs[int(rng.choice(len(outs), p=probs / probs.sum()))][0]
            return walk(game.next(h, a), i)
        info = game.infoset_key(h)
        actions = game.legal_actions(h)
        sigma = _regret_match(regret[info], actions)
        if pl == i:  # traverser: recurse ALL actions, update regret
            util = {}
            v = 0.0
            for a in actions:
                util[a] = walk(game.next(h, a), i)
                v += sigma[a] * util[a]
            for a in actions:
                regret[info][a] += util[a] - v
            return v
        # opponent: accumulate average strategy, sample one action
        for a in actions:
            strat_sum[info][a] += sigma[a]
        probs = np.array([sigma[a] for a in actions])
        a = actions[int(rng.choice(len(actions), p=probs / probs.sum()))]
        return walk(game.next(h, a), i)

    for _ in range(iters):
        walk(game.root(), 0)
        walk(game.root(), 1)
    return _average(strat_sum)


# --- world-major MCCFR (root chance enumerated) -------------------------------

@overload
def solve_mccfr_belief(
    game: Game, sweeps: int, *, traversals_per_world: int = ..., seed: int = ...,
    cfr_plus: bool = ..., exploit: bool = ...,
    return_world_values: Literal[False] = ...,
) -> tuple[Strategy, float]: ...
@overload
def solve_mccfr_belief(
    game: Game, sweeps: int, *, traversals_per_world: int = ..., seed: int = ...,
    cfr_plus: bool = ..., exploit: bool = ...,
    return_world_values: Literal[True],
) -> tuple[Strategy, float, list[float]]: ...
def solve_mccfr_belief(  # noqa: C901, PLR0913, PLR0915
    game: Game, sweeps: int, *, traversals_per_world: int = 8, seed: int = 0,
    cfr_plus: bool = False, exploit: bool = False,
    return_world_values: bool = False,
) -> tuple[Strategy, float] | tuple[Strategy, float, list[float]]:
    """External-sampling MCCFR with the ROOT chance node ENUMERATED (weighted), deeper
    chance/opponent SAMPLED. Requires ``game.root()`` to be a chance node whose outcomes
    are the "worlds" (a `BeliefSubgame`, or Kuhn's deal). Returns ``(average strategy,
    root value)`` -- the belief-weighted player-0 value at the root is the ReBeL value
    TARGET, computed for free during the solve (no extra engine traversals).

    Purpose is speed on the ENGINE subgame: `BeliefSubgame` re-``search_begin``s on each
    world switch, so we go WORLD-MAJOR -- enter each world once per (sweep, traverser),
    run ``traversals_per_world`` traversals while its engine session + state cache stay
    warm, then move on. Each world's regret/strategy contribution is scaled by its
    belief weight (the enumerated chance reach), so this converges to the same Nash as
    :func:`solve_mccfr_external` (checked on Kuhn), just with the engine cost amortised.

    ``cfr_plus``: CFR+ (regret-matching+ = clamp cumulative regret to >=0 each update;
    linear averaging = weight the average-strategy contribution by the sweep index t).
    CFR+ converges much faster per sweep -- the diagnostic showed plain MCCFR at low
    sweeps is far from equilibrium, so a converged solve becomes affordable at serving.

    ``exploit``: BEST-RESPOND to a FIXED scripted opponent instead of CFR-Nash. Opponent
    nodes deterministically follow ``game.scripted_action(h)`` (greedy_plus's move
    that determinized world) and only OUR infosets regret-minimise -- so the average
    strategy converges to a depth-limited best response to that opponent. Beats a strong
    NON-adaptive opponent that Nash play merely ties (diagnosis: ReBeL disagrees with
    greedy_plus 69% yet ties it -> it needs to EXPLOIT, not equilibrate).

    ``return_world_values``: also return a PER-WORLD player-0 root value list (aligned
    to ``game.chance_outcomes(root)`` order). These are the proper-ReBeL infostate CFV
    targets -- world k (drawn from hypothesis h_k) trains the value net's output[h_k],
    instead of collapsing every world into one belief-averaged scalar.
    """
    regret: dict[str, dict[object, float]] = defaultdict(lambda: defaultdict(float))
    strat_sum: dict[str, dict[object, float]] = defaultdict(lambda: defaultdict(float))
    rng = np.random.default_rng(seed)
    t_weight = [1.0]  # current sweep index (1-based) for CFR+ linear averaging

    def walk(h: tuple, i: int, w: float) -> float:  # noqa: C901
        if game.is_terminal(h):
            u0 = game.utility(h)
            return u0 if i == 0 else -u0
        pl = game.current_player(h)
        if pl == CHANCE:  # deeper chance (e.g. a coin) -> sample one outcome
            outs = game.chance_outcomes(h)
            probs = np.array([p for _, p in outs])
            a = outs[int(rng.choice(len(outs), p=probs / probs.sum()))][0]
            return walk(game.next(h, a), i, w)
        info = game.infoset_key(h)
        actions = game.legal_actions(h)
        sigma = _regret_match(regret[info], actions)
        if pl == i:  # traverser: recurse all actions, weight regret by world reach w
            util = {}
            v = 0.0
            for a in actions:
                util[a] = walk(game.next(h, a), i, w)
                v += sigma[a] * util[a]
            reg = regret[info]
            for a in actions:
                reg[a] += w * (util[a] - v)
                if cfr_plus and reg[a] < 0.0:  # regret-matching+: floor at 0
                    reg[a] = 0.0
            if exploit:  # BR: opp is fixed & only we traverse, so collect OUR avg here
                aw = w * t_weight[0] if cfr_plus else w
                for a in actions:
                    strat_sum[info][a] += aw * sigma[a]
            return v
        if exploit:  # opponent is FIXED (scripted): follow its move, best-respond to it
            a = game.scripted_action(h)  # type: ignore[attr-defined]
            if a not in actions:
                a = actions[0]
            return walk(game.next(h, a), i, w)
        aw = w * t_weight[0] if cfr_plus else w  # linear averaging weights by sweep t
        for a in actions:  # opponent: (weighted) average strategy, then sample one
            strat_sum[info][a] += aw * sigma[a]
        probs = np.array([sigma[a] for a in actions])
        a = actions[int(rng.choice(len(actions), p=probs / probs.sum()))]
        return walk(game.next(h, a), i, w)

    worlds = game.chance_outcomes(game.root())  # [(world_action, weight), ...]
    # EXPLOIT: only WE (the acting player at the root) best-respond; the opponent is
    # scripted, so we never traverse as them. players = just our index.
    if exploit:
        first_child = game.next(game.root(), worlds[0][0])
        us = game.current_player(first_child)
        players: tuple[int, ...] = (us,)
    else:
        players = (0, 1)
    root_val_sum = 0.0  # accumulate player-0's root value (the CFR value target)
    n_i0 = 0
    world_val_sum: dict[object, float] = defaultdict(float)  # per-world player-0 value
    world_val_cnt: dict[object, int] = defaultdict(int)
    for sweep in range(sweeps):
        t_weight[0] = float(sweep + 1)
        for i in players:
            for k, w in worlds:
                child = game.next(game.root(), k)  # world k (engine begins once)
                for _ in range(traversals_per_world):
                    v = walk(child, i, float(w))
                    if i == 0:
                        root_val_sum += float(w) * v  # belief-weighted player-0 value
                        n_i0 += 1
                        world_val_sum[k] += v
                        world_val_cnt[k] += 1
    # n_i0 = sweeps * worlds * tpw; Σ_k w_k = 1, so the mean belief value is
    # root_val_sum / (sweeps * tpw) = root_val_sum / n_i0 * len(worlds).
    root_value = root_val_sum / n_i0 * len(worlds) if n_i0 else 0.0
    if return_world_values:
        wv = [world_val_sum[k] / world_val_cnt[k] if world_val_cnt[k] else 0.0
              for k, _ in worlds]
        return _average(strat_sum), root_value, wv
    return _average(strat_sum), root_value


def game_value(
    game: Game, strategy: Strategy, rng: np.random.Generator, n_rollouts: int = 16,
) -> float:
    """Monte-Carlo estimate of player 0's value when both players follow ``strategy``
    (uniform on any unseen infoset). Used as the ReBeL value TARGET: the solved
    subgame's root value, recorded against its PBS features to train the leaf net."""
    total = 0.0
    for _ in range(n_rollouts):
        h = game.root()
        while not game.is_terminal(h):
            if game.current_player(h) == CHANCE:
                outs = game.chance_outcomes(h)
                probs = np.array([p for _, p in outs])
                a = outs[int(rng.choice(len(outs), p=probs / probs.sum()))][0]
            else:
                actions = game.legal_actions(h)
                sig = strategy.get(game.infoset_key(h))
                unif = np.full(len(actions), 1.0 / len(actions))
                if sig:
                    p = np.array([sig.get(a, 0.0) for a in actions])
                    p = p / p.sum() if p.sum() > 0 else unif
                else:
                    p = unif
                a = actions[int(rng.choice(len(actions), p=p))]
            h = game.next(h, a)
        total += game.utility(h)
    return total / max(n_rollouts, 1)


# --- exact exploitability via best-response enumeration -----------------------

def _collect_infosets(game: Game) -> tuple[dict[str, list[object]], dict[str, int]]:
    """Map every infoset -> its action list and owning player (full-tree walk)."""
    actions_of: dict[str, list[object]] = {}
    owner: dict[str, int] = {}

    def walk(h: tuple) -> None:
        if game.is_terminal(h):
            return
        pl = game.current_player(h)
        nexts = (game.chance_outcomes(h) if pl == CHANCE
                 else [(a, 0.0) for a in game.legal_actions(h)])
        if pl != CHANCE:
            info = game.infoset_key(h)
            actions_of[info] = game.legal_actions(h)
            owner[info] = pl
        for a, _ in nexts:
            walk(game.next(h, a))

    walk(game.root())
    return actions_of, owner


def _expected_u0(game: Game, s0: Strategy, s1: Strategy) -> float:
    """E[u0] with players following (possibly pure) strategies s0, s1; full chance."""
    def walk(h: tuple) -> float:
        if game.is_terminal(h):
            return game.utility(h)
        pl = game.current_player(h)
        if pl == CHANCE:
            return sum(pr * walk(game.next(h, a))
                       for a, pr in game.chance_outcomes(h))
        info = game.infoset_key(h)
        strat = (s0 if pl == 0 else s1)[info]
        return sum(p * walk(game.next(h, a)) for a, p in strat.items() if p > 0)
    return walk(game.root())


def exploitability(game: Game, strategy: Strategy) -> float:
    """Exact exploitability = BR0(vs s1) + BR1(vs s0) by enumerating each responder's
    PURE strategies over its infosets. 0 at Nash. Small games only (2^#infosets)."""
    actions_of, owner = _collect_infosets(game)
    infos = {0: [i for i in owner if owner[i] == 0],
             1: [i for i in owner if owner[i] == 1]}

    def pure_strategies(player: int) -> list[Strategy]:
        keys = infos[player]
        choices = [actions_of[k] for k in keys]
        out = []
        for combo in product(*choices):
            pairs = zip(keys, combo, strict=True)
            out.append({k: dict.fromkeys([act], 1.0) for k, act in pairs})
        return out

    br0 = max(_expected_u0(game, s0, strategy) for s0 in pure_strategies(0))
    br1 = max(-_expected_u0(game, strategy, s1) for s1 in pure_strategies(1))
    return br0 + br1
