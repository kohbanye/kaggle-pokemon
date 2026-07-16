"""Wrap a determinized cg subgame as a `Game` so `src.rebel.cfr` can solve it (Gate C).

Gate B proved the MCCFR estimator is unbiased on Kuhn. Gate C wires the REAL engine onto
the same interface: a depth-limited subgame rooted at one determinized world becomes a
`Game`, driven by `search_begin`/`search_step` (immutable branching + per-world
determinism verified in `scripts/rebel_api_probe.py`).

**This first version is SINGLE-WORLD (perfect info)** -- it validates the engine<->CFR
plumbing (session, immutable-branching state cache, decision/chance/terminal
classification, depth limit, pluggable leaf value) with a path-based infoset key that is
exactly correct for one world. The BELIEF/multi-world case (infoset key = public history
+ the acting player's own private cards, per the Codex review) is next, atop this.

A history is a tuple of chosen option indices at the GENUINE decision nodes; forced and
multi-select nodes are auto-advanced (an action abstraction). Coin flips
(`SelectContext.COIN_HEAD`=46, exposed by `manual_coin=True`) are chance nodes.

Engine-bound (lazy `cg` import); ty-excluded like the other engine modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.rebel.game import CHANCE

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.search.determinize import Determinization

SINGLE_SELECT = 1
_MIN_CHOICE = 2
COIN_HEAD_CTX = 46          # SelectContext.COIN_HEAD ("choose heads?"), manual_coin
_DECISION, _CHANCE, _TERMINAL = "decision", "chance", "terminal"


def _advance_state(state: object, depth: int) -> tuple[object, str, int]:
    """Skip forced/multi-select nodes; stop at a decision, coin, or terminal.

    Returns (resolved state, kind, depth). ``depth`` counts genuine decisions taken so
    far; the depth limit itself is checked by the game's ``is_terminal``."""
    from cg.api import search_step  # noqa: PLC0415

    for _ in range(400):  # bound: a turn resolves in far fewer forced steps
        o = state.observation  # type: ignore[attr-defined]
        cur = o.current
        if cur is None or int(getattr(cur, "result", -1)) != -1:
            return state, _TERMINAL, depth
        sel = o.select
        opts = (sel.option or []) if sel is not None else []
        if not opts:
            return state, _TERMINAL, depth
        if int(getattr(sel, "context", -1)) == COIN_HEAD_CTX:
            return state, _CHANCE, depth
        if int(sel.maxCount) == SINGLE_SELECT and len(opts) >= _MIN_CHOICE:
            return state, _DECISION, depth
        # forced single option or multi-select -> auto-advance (abstraction)
        k = max(1, int(sel.maxCount))
        state = search_step(state.searchId, list(range(min(k, len(opts)))))  # type: ignore[attr-defined]
    return state, _TERMINAL, depth


def _state_turn(state: object) -> int:
    """The engine turn counter at a state (increments each turn); -1 if unavailable."""
    cur = state.observation.current  # type: ignore[attr-defined]
    return int(getattr(cur, "turn", -1)) if cur is not None else -1


def default_leaf_value(state: object) -> float:
    """Prize-progress heuristic in [-1, 1] from player 0's view (placeholder for the
    value net): + when player 0 has taken more prizes than player 1."""
    cur = state.observation.current  # type: ignore[attr-defined]
    if cur is None:
        return 0.0
    taken = [6 - len(cur.players[i].prize) for i in (0, 1)]
    return max(-1.0, min(1.0, (taken[0] - taken[1]) / 6.0))


def make_energy_shaped_leaf(
    base_leaf: object, cards: dict, attacks: dict, weight: float = 0.25,
) -> object:
    """Wrap a leaf with an ENERGY-PROGRESS reward term (ladder-loss fix H1).

    Ladder replay analysis: ReBeL loses because the depth-limited search UNDER-ATTACHES
    energy -- its attacker never reaches the cost of its damaging attack, so it can't KO
    and takes 0 prizes. Attaching is tempo-neutral with a KO payoff BEYOND the depth-3
    horizon, and the prize/value leaf can't see it. This adds, from player 0's view,
    ``weight * (our_progress - opp_progress)`` where progress = min(energy on the
    active / cheapest damaging-attack cost, 1) -- a dense gradient that makes the search
    value powering up its attacker even when the KO is past the horizon."""
    def _progress(active_list: object) -> float:
        if not active_list:
            return 0.0
        a = active_list[0]  # type: ignore[index]
        card = cards.get(int(getattr(a, "id", 0)))
        if not card:
            return 0.0
        costs = [len(attacks[aid]["cost"]) for aid in card.get("attacks", [])
                 if attacks.get(aid) and attacks[aid].get("dmg", 0) > 0
                 and attacks[aid].get("cost")]
        if not costs:
            return 0.0
        have = len(getattr(a, "energies", []) or [])
        return min(have / min(costs), 1.0)

    def leaf(state: object) -> float:
        base = base_leaf(state)  # type: ignore[operator]
        cur = state.observation.current  # type: ignore[attr-defined]
        if cur is None:
            return base
        try:
            shaped = weight * (_progress(cur.players[0].active)
                               - _progress(cur.players[1].active))
        except Exception:  # noqa: BLE001 - shaping must never break the leaf
            return base
        return max(-1.0, min(1.0, base + shaped))
    return leaf


class EngineSubgame:
    """A depth-limited single-world cg subgame as a `Game` (player 0 maximises)."""

    def __init__(
        self,
        obs: dict,
        det: Determinization,
        *,
        max_depth: int = 3,
        leaf_value: Callable[[object], float] = default_leaf_value,
    ) -> None:
        from cg.api import search_begin, to_observation_class  # noqa: PLC0415

        self._obs = obs
        self._det = det
        self.max_depth = max_depth
        self.leaf_value = leaf_value
        self._begin = search_begin
        raw_root = search_begin(
            to_observation_class(obs), det.your_deck, det.your_prize, det.opp_deck,
            det.opp_prize, det.opp_hand, det.opp_active, manual_coin=True)
        # history -> (resolved SearchState, kind, depth). () is the root.
        self._cache: dict[tuple, tuple[object, str, int]] = {}
        self._cache[()] = _advance_state(raw_root, 0)

    def close(self) -> None:
        from cg.api import search_end  # noqa: PLC0415
        search_end()

    def _resolve(self, h: tuple) -> tuple[object, str, int]:
        if h not in self._cache:
            from cg.api import search_step  # noqa: PLC0415
            parent, _, depth = self._resolve(h[:-1])
            raw = search_step(parent.searchId, [h[-1]])  # type: ignore[attr-defined]
            self._cache[h] = _advance_state(raw, depth + 1)
        return self._cache[h]

    # --- Game interface ------------------------------------------------------

    def root(self) -> tuple:
        return ()

    def is_terminal(self, h: tuple) -> bool:
        _, kind, depth = self._resolve(h)
        return kind == _TERMINAL or depth >= self.max_depth

    def current_player(self, h: tuple) -> int:
        state, kind, _ = self._resolve(h)
        if kind == _CHANCE:
            return CHANCE
        return int(state.observation.current.yourIndex)  # type: ignore[union-attr]

    def chance_outcomes(self, h: tuple) -> list[tuple[object, float]]:
        n = len(self._resolve(h)[0].observation.select.option)  # type: ignore[union-attr]
        return [(i, 1.0 / n) for i in range(n)]  # fair coin over the exposed options

    def legal_actions(self, h: tuple) -> list[object]:
        n = len(self._resolve(h)[0].observation.select.option)  # type: ignore[union-attr]
        return list(range(n))

    def next(self, h: tuple, action: object) -> tuple:
        return (*h, int(action))  # type: ignore[arg-type]

    def infoset_key(self, h: tuple) -> str:
        # SINGLE-WORLD: the path is the infoset (perfect info). Multi-world keying
        # (public history + acting player's own private cards) is the next step.
        state, _, _ = self._resolve(h)
        return f"{int(state.observation.current.yourIndex)}:{h}"  # type: ignore[union-attr]

    def utility(self, h: tuple) -> float:
        state, kind, _ = self._resolve(h)
        cur = state.observation.current  # type: ignore[attr-defined]
        if kind == _TERMINAL and cur is not None:
            res = int(getattr(cur, "result", -1))
            if res != -1:
                return 1.0 if res == 0 else -1.0
        return self.leaf_value(state)  # depth-limit leaf (or None-state fallback)


def infoset_signature(state: object) -> str:
    """WORLD-INDEPENDENT infoset key: a canonical serialization of the acting player's
    observation (which the engine already restricts to THEIR information set -- own hand
    set, opponent hand None). Two worlds that present the acting player the same
    observation therefore share an infoset, so regret is pooled correctly across the
    belief (avoids per-world strategy fusion). Caveat: this keys on the current
    observation, not the full observation HISTORY -- a mild imperfect-recall abstr,
    acceptable because the board+hand+discard is a near-sufficient statistic here."""
    import json  # noqa: PLC0415

    from src.search.ismcts import _plain  # noqa: PLC0415
    # Convert ONLY current+select (the two fields we key on) instead of the whole
    # observation via obs_to_dict -- byte-identical output, skips _plain over the
    # discarded fields. infoset_signature is the hottest function in a deep solve.
    obs = state.observation  # type: ignore[attr-defined]
    keep = {"current": _plain(getattr(obs, "current", None)),
            "select": _plain(getattr(obs, "select", None))}
    return json.dumps(keep, sort_keys=True, default=str)


class BeliefSubgame:
    """A depth-limited MULTI-WORLD cg subgame as a `Game` (the ReBeL PBS solve).

    The belief (a fixed list of ``(Determinization, weight)`` worlds) is the root CHANCE
    node: one world is dealt with probability proportional to its weight, so external-
    sampling MCCFR samples worlds from the belief and its counterfactual-regret estimate
    is over the belief RANGE (fixed per solve, as CFR requires). Within a world the tree
    is the determinized subgame (as `EngineSubgame`), but infosets are keyed by the
    world-independent :func:`infoset_signature`, so regret is shared across worlds that
    the acting player cannot distinguish. **All belief worlds are ``search_begin``-ed
    ONCE up front and their engine sessions COEXIST** (verified in the api probe: many
    search trees live at once under the global ``agent_ptr``, freely interleaved) -- so
    a solve does N begins total, not N-per-sweep, and no cache is ever cleared (~10x
    faster). One ``search_end`` at :meth:`close` tears them all down.
    """

    def __init__(  # noqa: PLR0913 - subgame config knobs are genuinely distinct
        self,
        obs: dict,
        worlds: list[tuple[Determinization, float]],
        *,
        max_depth: int = 3,
        leaf_value: Callable[[object], float] = default_leaf_value,
        opp_agent: object | None = None,
        horizon_turns: int | None = None,
        top_k: int | None = None,
        recurrent: object | None = None,
        vector: object | None = None,
    ) -> None:
        if not worlds:
            msg = "BeliefSubgame needs >=1 world"
            raise ValueError(msg)
        from cg.api import search_begin, to_observation_class  # noqa: PLC0415

        self._obs = obs
        self._worlds = [w for w, _ in worlds]
        total = sum(max(0.0, wt) for _, wt in worlds) or 1.0
        self._weights = [max(0.0, wt) / total for _, wt in worlds]
        self.max_depth = max_depth
        # HISTORY-aware (LSTM) leaf: `recurrent` is a dict with keys `net`
        # (RecurrentValueNet), `feature_fn` (state -> 231-vec), `root_state` (h,c after
        # the real game history). utility() then THREADS the LSTM through the path
        # (per-node state cached like _resolve) instead of a static leaf_value(state).
        self._rec = recurrent
        self._lstm: dict[tuple, tuple] = {}  # per-node (h,c) LSTM state cache
        # PROPER-ReBeL infostate VALUE VECTOR leaf: `vector` is a dict with keys `net`
        # (VectorValueNet: public features -> K per-hypothesis values), `feature_fn`
        # (state -> 231-vec), `world_hyp` (world index k -> hypothesis index h_k). A
        # leaf in world k reads net.value(x, h_k) -- the counterfactual value for THAT
        # infostate, not a belief-averaged scalar (removes the averaging approximation).
        self._vec = vector
        # Turn-boundary (public-event) horizon: stop the leaf at the end of the current
        # player's turn +N rather than after a fixed #decisions -- leaves then sit at
        # clean turn boundaries and the horizon adapts to turn length. max_depth stays a
        # hard safety cap; None => classic fixed-decision depth.
        self.horizon_turns = horizon_turns
        # Growing-tree width bound: expand only the top_k options (ranked by the cheap
        # prize heuristic of their children) at a decision. Bounds branching so a
        # deeper / turn horizon is affordable (full width blows up N^depth). None=all.
        self.top_k = top_k
        self._topk: dict[str, list[int]] = {}  # kept acts per infoset (CFR-consistent)
        self.leaf_value = leaf_value
        self._opp_agent = opp_agent  # scripted in-search opponent for EXPLOIT (BR) mode
        # Begin every world once; the sessions coexist. cache key = (world k, path).
        obs_obj = to_observation_class(obs)
        self._cache: dict[tuple, tuple[object, str, int]] = {}
        self._util: dict[tuple, float] = {}  # leaf-value cache by h (exact-node repeat)
        self._util_sig: dict[str, float] = {}  # leaf-value cache by infoset signature
        self._infokey: dict[tuple, str] = {}  # infoset-signature cache (56% of solve!)
        self._open = False
        for k, det in enumerate(self._worlds):
            raw = search_begin(
                obs_obj, det.your_deck, det.your_prize, det.opp_deck, det.opp_prize,
                det.opp_hand, det.opp_active, manual_coin=True)
            self._cache[(k,)] = _advance_state(raw, 0)
            self._open = True
        self._root_turn = _state_turn(self._cache[(0,)][0])

    def close(self) -> None:
        if self._open:
            from cg.api import search_end  # noqa: PLC0415
            search_end()
            self._open = False

    def _resolve(self, h: tuple) -> tuple[object, str, int]:
        # h = (k, i1, i2, ...); () is the belief root (handled by callers).
        if h not in self._cache:
            from cg.api import search_step  # noqa: PLC0415
            parent, _, depth = self._resolve(h[:-1])
            raw = search_step(parent.searchId, [h[-1]])  # type: ignore[attr-defined]
            self._cache[h] = _advance_state(raw, depth + 1)
        return self._cache[h]

    # --- Game interface ------------------------------------------------------

    def root(self) -> tuple:
        return ()

    def is_terminal(self, h: tuple) -> bool:
        if h == ():
            return False  # the belief chance root
        state, kind, depth = self._resolve(h)
        if kind == _TERMINAL or depth >= self.max_depth:
            return True
        if self.horizon_turns is not None:  # public-event horizon: stop at turn edge
            return _state_turn(state) - self._root_turn >= self.horizon_turns
        return False

    def current_player(self, h: tuple) -> int:
        if h == ():
            return CHANCE  # deal a world from the belief
        state, kind, _ = self._resolve(h)
        if kind == _CHANCE:
            return CHANCE
        return int(state.observation.current.yourIndex)  # type: ignore[union-attr]

    def chance_outcomes(self, h: tuple) -> list[tuple[object, float]]:
        if h == ():
            return [(k, w) for k, w in enumerate(self._weights) if w > 0]
        n = len(self._resolve(h)[0].observation.select.option)  # type: ignore[union-attr]
        return [(i, 1.0 / n) for i in range(n)]  # fair coin over exposed options

    def legal_actions(self, h: tuple) -> list[object]:
        state = self._resolve(h)[0]
        n = len(state.observation.select.option)  # type: ignore[union-attr]
        if self.top_k is None or n <= self.top_k:
            return list(range(n))
        # top_k pruning cached per INFOSET so all worlds/paths sharing it keep the SAME
        # action set (CFR pools regret by infoset -- inconsistent sets would break it).
        sig = self.infoset_key(h)
        cached = self._topk.get(sig)
        if cached is not None:
            return [a for a in cached if a < n] or [0]
        acting = int(state.observation.current.yourIndex)  # type: ignore[union-attr]
        def _score(a: int) -> float:
            lv = default_leaf_value(self._resolve((*h, a))[0])  # p0 view, cheap
            return lv if acting == 0 else -lv  # acting player's own value
        kept = sorted(range(n), key=_score, reverse=True)[: self.top_k]
        kept.sort()  # stable option order
        self._topk[sig] = kept
        return kept

    def next(self, h: tuple, action: object) -> tuple:
        return (*h, int(action))  # type: ignore[arg-type]

    def infoset_key(self, h: tuple) -> str:
        # world-INDEPENDENT: the acting player's observation, not the path/world.
        # Memoised on h: each node is revisited sweeps*tpw*2 times but its signature
        # (obs->dict->json.dumps) is fixed -- this cache was ~56% of the depth-4 solve.
        cached = self._infokey.get(h)
        if cached is not None:
            return cached
        key = infoset_signature(self._resolve(h)[0])
        self._infokey[h] = key
        return key

    def _lstm_state(self, h: tuple) -> tuple:
        """LSTM (h,c) at node h: thread the recurrent net along the search path, cached
        per-node (like _resolve). Base (k,) steps from the real-history root_state."""
        st = self._lstm.get(h)
        if st is not None:
            return st
        rec = self._rec
        x = rec["feature_fn"](self._resolve(h)[0])  # type: ignore[index]
        prev = rec["root_state"] if len(h) == 1 else self._lstm_state(h[:-1])  # type: ignore[index]
        st = rec["net"].step(prev, x)  # type: ignore[index]
        self._lstm[h] = st
        return st

    def utility(self, h: tuple) -> float:
        cached = self._util.get(h)
        if cached is not None:
            return cached
        state, kind, _ = self._resolve(h)
        cur = state.observation.current  # type: ignore[attr-defined]
        res = int(getattr(cur, "result", -1)) if cur is not None else -1
        if kind == _TERMINAL and res != -1:
            v = 1.0 if res == 0 else -1.0
        elif self._rec is not None:
            # HISTORY-aware leaf: thread the LSTM along the path (history root_state
            # + in-search steps), value from the leaf hidden. Per-node state is cached.
            v = float(self._rec["net"].value_from_h(self._lstm_state(h)[0]))
        elif self._vec is not None:
            # INFOSTATE-VECTOR leaf: world k -> hypothesis h_k, read output[h_k]. The
            # net predicts the ACTING player's value-vector (like the scalar net);
            # convert to player-0's view here (negate on p1). h[0] is the dealt world.
            acting = int(cur.yourIndex) if cur is not None else 0
            hyp = self._vec["world_hyp"][h[0]]  # type: ignore[index]
            x = self._vec["feature_fn"](state)  # type: ignore[index]
            vv = float(self._vec["net"].value(x, hyp))  # type: ignore[index]
            v = vv if acting == 0 else -vv
        else:
            # The value-net leaf (~51% of a depth-4 solve) depends ONLY on the acting
            # player's observation (belief hypotheses are fixed, world-independent), so
            # two h with the same infoset_signature give the SAME leaf value. Cache the
            # expensive leaf by signature to collapse it across worlds/paths (the
            # h-cache above still short-circuits repeat visits of the exact same node).
            sig = self.infoset_key(h)  # h-memoised, cheap
            sv = self._util_sig.get(sig)
            if sv is None:
                sv = self.leaf_value(state)
                self._util_sig[sig] = sv
            v = sv
        self._util[h] = v
        return v

    def scripted_action(self, h: tuple) -> int:
        """The scripted in-search opponent's option index at node h (EXPLOIT/BR mode).

        Builds the acting player's obs from the determinized state, asks ``opp_agent``
        (greedy_plus) for its move -- a solve best-responds to that FIXED opponent
        instead of a CFR-Nash one. Falls back to option 0 on any error."""
        state, _, _ = self._resolve(h)
        n = len(state.observation.select.option)  # type: ignore[union-attr]
        if self._opp_agent is None:
            return 0
        from src.search.ismcts import obs_to_dict  # noqa: PLC0415
        try:
            mv = self._opp_agent.act(obs_to_dict(state.observation))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return 0
        idx = int(mv[0]) if mv else 0
        return idx if 0 <= idx < n else 0
