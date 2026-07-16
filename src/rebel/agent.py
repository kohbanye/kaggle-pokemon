"""RebelAgent -- the approximate-ReBeL solver as a Kaggle agent (serving driver).

Ties the pieces together into a playable `act(obs) -> list[int]`: on a searchable
single-select it builds the opponent belief (`OpponentBelief`), samples belief worlds
(`build_worlds`), wraps them as a depth-limited `BeliefSubgame`, solves it with
external-sampling MCCFR, and plays the average strategy at our root-decision infoset.
Every other decision (and any error / time overrun) falls back to a scripted
`greedy_plus` move, so the agent never crashes.

The leaf value is still the crude prize-progress heuristic (`default_leaf_value`); this
makes ReBeL *measurable* end to end. Replacing the leaf with a trained infostate value
net is the next design step -- the agent picks it up through the `leaf_value` knob.

Engine-bound (imports `cg` transitively via `BeliefSubgame`); ty-excluded like the other
engine modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.agents.base import Agent, legal_fallback
from src.rebel.belief import OpponentBelief, build_worlds, build_worlds_hyp
from src.rebel.cfr import solve_mccfr_belief
from src.rebel.engine_game import BeliefSubgame, default_leaf_value

if TYPE_CHECKING:
    from collections.abc import Callable

SINGLE_SELECT = 1
_MIN_CHOICE = 2


class RebelAgent(Agent):
    """Approximate ReBeL (belief + depth-limited MCCFR subgame solve) as an agent."""

    name = "rebel"

    def __init__(  # noqa: PLR0913 - solver knobs are genuinely separate inputs
        self,
        deck: list[int],
        engine: dict,
        *,
        n_particles: int = 4,
        max_depth: int = 2,
        sweeps: int = 16,
        traversals_per_world: int = 4,
        seed: int = 0,
        sharpness: float = 6.0,
        temperature: float = 0.0,
        cfr_plus: bool = True,
        exploit: bool = False,
        exploit_opp: str = "greedy_plus",
        horizon_turns: int | None = None,
        top_k: int | None = None,
        leaf_value: Callable[[object], float] = default_leaf_value,
        value_net: object | None = None,
        recurrent_net: object | None = None,
        lstm_leaf: bool = True,
        vector_net: object | None = None,
        vector_record: bool = False,
        energy_shaping: float = 0.0,
        record: bool = False,
    ) -> None:
        super().__init__(deck)
        from src.agents import build_agent  # noqa: PLC0415

        self.engine = engine
        self.n_particles = n_particles
        self.max_depth = max_depth
        self.sweeps = sweeps
        self.traversals_per_world = traversals_per_world
        self.cfr_plus = cfr_plus
        self.horizon_turns = horizon_turns
        self.top_k = top_k
        self.exploit = exploit
        # in-search scripted opponent for EXPLOIT (best-response) mode
        self._opp_agent = build_agent(exploit_opp, deck, engine) if exploit else None
        self.temperature = float(temperature)
        self._rng = np.random.default_rng(seed)
        self._belief = OpponentBelief.from_dirs(sharpness=sharpness)
        self.leaf_value = leaf_value
        # HISTORY-aware (LSTM) leaf: running LSTM state over the agent's past
        # decision features (real game history), threaded per search-leaf; seq
        # recorded for training when `record`.
        self.recurrent_net = recurrent_net
        self.lstm_leaf = lstm_leaf  # False = record history, heuristic leaf
        self._hist_state = (recurrent_net.zero_state()
                            if recurrent_net is not None else None)
        self._hist_seq: list = []
        # PROPER-ReBeL infostate value VECTOR. `vector_net` (a VectorValueNet) becomes
        # the per-world leaf; `vector_record` logs the per-world CFR values as the
        # K-vector training target (mask on the hypotheses that appeared). Either turns
        # vector mode on: worlds carry their sampled hypothesis index, the solve returns
        # per-world values, records/serves per infostate not one belief-averaged scalar.
        self.vector_net = vector_net
        self.vector_mode = vector_net is not None or vector_record
        # Feature extractors (for the value-net leaf and/or recording training targets).
        self._feats = self._decks = self._ctxs = None
        if (value_net is not None or recurrent_net is not None
                or self.vector_mode or record):
            from src.net.features import CardFeatures  # noqa: PLC0415
            from src.rebel.value_net import build_hypothesis_ctxs  # noqa: PLC0415
            self._feats = CardFeatures(engine)
            self._decks = self._belief.hypotheses
            self._ctxs = build_hypothesis_ctxs(self._decks, self._feats)
        # A trained value net (if given) becomes the leaf; else the prize heuristic.
        if value_net is not None:
            from src.rebel.value_net import make_leaf_value  # noqa: PLC0415
            self.leaf_value = make_leaf_value(
                value_net, self._feats, self._decks, self._ctxs)
        # ENERGY-PROGRESS shaping (ladder-loss fix H1): the search under-attaches energy
        # so its attacker never reaches KO range; this dense term credits powering up.
        self.energy_shaping = energy_shaping
        if energy_shaping > 0.0:
            from src.rebel.engine_game import make_energy_shaped_leaf  # noqa: PLC0415
            self.leaf_value = make_energy_shaped_leaf(
                self.leaf_value, engine.get("cards", {}), engine.get("attacks", {}),
                weight=energy_shaping)
        # (F) self-play training: record (PBS features, subgame root value) per solve.
        self.record = record
        # (x, target) for scalar/LSTM; (x, target_vec, mask) for vector mode.
        self.samples: list[tuple] = []
        cards = engine.get("cards", {})
        self._basics = [c for c in deck if cards.get(c) and cards[c].get("basic")]
        # scripted fallback for non-searchable decisions and on any solver failure.
        self._fallback = build_agent("greedy_plus", deck, engine)
        self.solved = 0   # decisions where the ReBeL solve produced a move (diag)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._fallback.reset(seed)
        if self.recurrent_net is not None:  # new game -> fresh history
            self._hist_state = self.recurrent_net.zero_state()
            self._hist_seq = []

    def act(self, obs: dict) -> list[int]:
        select = obs.get("select") or {}
        cur = obs.get("current")
        options = select.get("option") or []
        single = int(select.get("maxCount", 0)) == SINGLE_SELECT and cur is not None
        searchable = (single and len(options) >= _MIN_CHOICE
                      and bool(obs.get("search_begin_input")))
        if searchable:
            try:
                move = self._solve(obs)
            except Exception:  # noqa: BLE001 - never crash a match; degrade to fallback
                move = None
            if move is not None:
                self.solved += 1
                return move
        return self._fallback_move(obs, select)

    def _fallback_move(self, obs: dict, select: dict) -> list[int]:
        try:
            return self._fallback.act(obs)
        except Exception:  # noqa: BLE001
            return legal_fallback(select)

    def _state_feat(self, state: object) -> object:
        """Engine SearchState -> value-net feature vector (for LSTM threading)."""
        from src.rebel.value_net import _as_dict, state_features  # noqa: PLC0415
        cur = state.observation.current  # type: ignore[attr-defined]
        if cur is None:
            from src.rebel.value_net import VALUE_IN_DIM  # noqa: PLC0415
            return np.zeros(VALUE_IN_DIM)
        d = _as_dict(cur)
        return state_features(d, int(d.get("yourIndex", 0)), self._feats,
                              self._decks, self._ctxs)

    def _solve(self, obs: dict) -> list[int] | None:
        cur = obs["current"]
        your = int(cur.get("yourIndex", 0))
        n_opts = len(obs["select"]["option"])
        hyps: list[int] = []
        if self.vector_mode:  # worlds carry their sampled hypothesis index
            worlds, hyps = build_worlds_hyp(
                cur, your, self.deck, self._belief, self.n_particles, self._rng,
                opp_basics=self._basics)
        else:
            worlds = build_worlds(cur, your, self.deck, self._belief, self.n_particles,
                                  self._rng, opp_basics=self._basics)
        rec = None
        if self.recurrent_net is not None and self.lstm_leaf:  # history-aware LSTM leaf
            rec = {"net": self.recurrent_net, "feature_fn": self._state_feat,
                   "root_state": self._hist_state}
        vec = None
        if self.vector_net is not None:  # per-world infostate-vector leaf
            vec = {"net": self.vector_net, "feature_fn": self._state_feat,
                   "world_hyp": hyps}
        game = BeliefSubgame(obs, worlds, max_depth=self.max_depth,
                             leaf_value=self.leaf_value, opp_agent=self._opp_agent,
                             horizon_turns=self.horizon_turns, top_k=self.top_k,
                             recurrent=rec, vector=vec)
        try:
            solve_seed = int(self._rng.integers(1 << 30))
            if self.vector_mode:
                avg, root_value, world_values = solve_mccfr_belief(
                    game, sweeps=self.sweeps,
                    traversals_per_world=self.traversals_per_world,
                    seed=solve_seed, cfr_plus=self.cfr_plus,
                    return_world_values=True)
            else:
                avg, root_value = solve_mccfr_belief(
                    game, sweeps=self.sweeps,
                    traversals_per_world=self.traversals_per_world,
                    seed=solve_seed, cfr_plus=self.cfr_plus, exploit=self.exploit)
            root_key = game.infoset_key((0,))  # our 1st decision (shared over worlds)
            strat = avg.get(root_key)
            if self.record and self._feats is not None:
                if self.vector_mode:
                    self._record_vector_sample(world_values, hyps, cur, your)
                else:
                    self._record_sample(root_value, cur, your)
        finally:
            game.close()
        # advance the running history LSTM state by THIS decision's features
        if self.recurrent_net is not None:
            from src.rebel.value_net import state_features  # noqa: PLC0415
            x = state_features(cur, your, self._feats, self._decks, self._ctxs)
            self._hist_seq.append(x)
            self._hist_state = self.recurrent_net.step(self._hist_state, x)
        if not strat:
            return None
        idx = self._pick(strat, n_opts)
        return None if idx is None else [idx]

    def _record_sample(self, root_value: float, cur: dict, your: int) -> None:
        """Log (root PBS features, acting-perspective subgame value) as target.

        ``root_value`` is player 0's CFR root value from the solve (free, low-variance):
        no extra engine rollouts (the old `game_value` MC estimate blew up on big
        late-game trees). Recorded from the acting player's view (negate for p1)."""
        from src.rebel.value_net import state_features  # noqa: PLC0415
        x = state_features(cur, your, self._feats, self._decks, self._ctxs)
        target = root_value if your == 0 else -root_value
        if self.recurrent_net is not None:
            # LSTM sample: feature SEQUENCE (history ++ this decision) + target.
            seq = np.stack([*self._hist_seq, x])
            self.samples.append((seq, target))
        else:
            self.samples.append((x, target))

    def _record_vector_sample(self, world_values: list[float], hyps: list[int],
                              cur: dict, your: int) -> None:
        """Log (public features, per-hypothesis CFV vector, mask) -- the proper-ReBeL
        infostate-value target. Each world's player-0 CFR value is converted to the
        acting player's view (like the scalar target) and written to its hypothesis
        slot; worlds sharing a hypothesis are averaged. The mask marks which hypotheses
        appeared this solve, so training only regresses the observed slots."""
        from collections import defaultdict  # noqa: PLC0415

        from src.rebel.value_net import state_features  # noqa: PLC0415
        x = state_features(cur, your, self._feats, self._decks, self._ctxs)
        k = len(self._decks)  # type: ignore[arg-type]
        agg: dict[int, list[float]] = defaultdict(list)
        for wv, hyp in zip(world_values, hyps, strict=True):
            agg[hyp].append(wv if your == 0 else -wv)  # acting-player perspective
        target = np.zeros(k)
        mask = np.zeros(k)
        for hyp, vals in agg.items():
            target[hyp] = float(np.mean(vals))
            mask[hyp] = 1.0
        self.samples.append((x, target, mask))

    def _pick(self, strat: dict[object, float], n_opts: int) -> int | None:
        """Map the strategy over option indices to a legal option to play."""
        items = [(int(a), p) for a, p in strat.items() if 0 <= int(a) < n_opts]
        if not items:
            return None
        if self.temperature <= 0.0:
            return max(items, key=lambda kv: kv[1])[0]
        acts = np.array([a for a, _ in items])
        probs = np.array([p for _, p in items], dtype=float)
        probs = probs ** (1.0 / self.temperature)
        probs /= probs.sum()
        return int(acts[int(self._rng.choice(len(acts), p=probs))])
