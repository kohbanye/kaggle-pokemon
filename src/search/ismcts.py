"""Single-observer Information-Set MCTS (SO-ISMCTS), net-guided, for AlphaZero-style
training amplification.

Unlike PIMC (which solves each determinization independently and so suffers *strategy
fusion*), ISMCTS grows ONE tree keyed by our information set (our action sequence) and
re-samples a determinization every iteration, sharing visit statistics across worlds --
the correct object for hidden information. The opponent belief (opp_belief) drives which
deck each determinization draws, so the search integrates over *which deck* the opponent
is on from the cards they've revealed.

Leaf evaluation is controlled by ``rollout_cap`` (speedup lever (1)):
- ``0``   -> pure VALUE-HEAD leaf (AlphaZero): one net forward, no rollout. ~30x cheaper
             than rolling to terminal -- the dominant cost of the original design.
- ``k>0`` -> roll out at most k steps (warmup while the value head is weak), then value.
- large   -> the original full net-rollout-to-terminal (value head only as a fallback).

Set ``record=True`` to log, per searched decision, the root visit-count ``pi``
and value ``z`` -- the AlphaZero training targets.

Engine-driven (lazy ``cg`` import); ty-excluded like the other engine modules.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from src.agents.recurrent_agent import RecurrentNetAgent
from src.net.encode import encode_options, option_embed_rows
from src.search.determinize import sample_determinization
from src.search.opp_belief import sample_consistent_deck, seen_opponent_ids

if TYPE_CHECKING:
    from src.net.recurrent_model import RecurrentPolicyValueNet

SINGLE_SELECT = 1
_MIN_CHOICE = 2


def _plain(x: object) -> object:
    if is_dataclass(x) and not isinstance(x, type):
        return {f.name: _plain(getattr(x, f.name)) for f in fields(x)}
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, list):
        return [_plain(v) for v in x]
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    return x


def obs_to_dict(observation: object) -> dict:
    """Engine ``Observation`` dataclass -> plain obs dict (Enum -> its value)."""
    return {f.name: _plain(getattr(observation, f.name)) for f in fields(observation)}
_OPTION_KEYS = (
    "type", "number", "area", "index", "playerIndex", "toolIndex", "energyIndex",
    "count", "inPlayArea", "inPlayIndex", "attackId", "cardId", "serial",
    "specialConditionType",
)


def option_sig(opt: dict) -> tuple:
    """Order-independent identity of an option (so stats aggregate across worlds)."""
    return tuple(opt.get(k) for k in _OPTION_KEYS)


class _Node:
    """Per-information-set action stats: visits, value sum, availability, prior."""

    __slots__ = ("avail", "n", "p", "w")

    def __init__(self) -> None:
        self.n: dict[tuple, int] = {}
        self.w: dict[tuple, float] = {}
        self.avail: dict[tuple, int] = {}
        self.p: dict[tuple, float] = {}   # net policy prior P(a) for PUCT


class RecurrentIsmctsAgent(RecurrentNetAgent):
    """Recurrent net + SO-ISMCTS over single-select decisions, gated by confidence."""

    name = "recurrent_ismcts"

    def __init__(  # noqa: PLR0913 - search knobs are genuinely separate inputs
        self,
        deck: list[int],
        engine: dict | None,
        net: RecurrentPolicyValueNet,
        *,
        opp_prior: list[int],
        opp_basics: list[int] | None = None,
        opp_decks: list[list[int]] | None = None,
        iterations: int = 160,
        max_depth: int = 3,
        rollout_depth: int = 200,
        rollout_cap: int = 0,
        move_budget_s: float = 2.0,
        gate_margin: float = 0.10,
        c_puct: float = 1.5,
        record: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(deck, engine, net=net, **kwargs)  # type: ignore[arg-type]
        self.opp_prior = list(opp_prior)
        self.opp_basics = list(opp_basics or [])
        self.opp_decks = [list(d) for d in (opp_decks or [])]
        self.iterations = iterations
        self.max_depth = max_depth
        self.rollout_depth = rollout_depth
        self.rollout_cap = rollout_cap            # (1) 0 = value-head leaf (AlphaZero)
        self.move_budget_s = move_budget_s
        self.gate_margin = gate_margin
        self.c_puct = c_puct
        self.record = record
        self._rollers = (
            RecurrentNetAgent(deck, engine, net=net, cb_pool=None,
                              build_deck_from_net=False, temperature=0.0),
            RecurrentNetAgent(deck, engine, net=net, cb_pool=None,
                              build_deck_from_net=False, temperature=0.0),
        )
        self.searched = 0    # decisions where the gate let search override the net
        self.n_searches = 0  # decisions where the search actually ran
        self._last_pi: list[float] | None = None  # last search's visit-count policy
        # every single-select if record: {current, select, choice, pi|None}; pi is the
        # root visit-count on searched decisions (the AZ policy target), None otherwise.
        self.move_log: list[dict] = []

    def act(self, obs: dict) -> list[int]:
        base = super().act(obs)  # advances the real LSTM (one trajectory step)
        select = obs.get("select") or {}
        options = select.get("option") or []
        cur = obs.get("current")
        single = int(select.get("maxCount", 0)) == SINGLE_SELECT and cur is not None
        searchable = (single and len(options) >= _MIN_CHOICE
                      and bool(obs.get("search_begin_input")))
        self._last_pi = None
        choice = base
        if searchable:
            try:
                choice = self._ismcts(obs, options, base)
            except Exception:  # noqa: BLE001 - search must never crash a match
                choice = base
        # Record EVERY single-select (the LSTM stepped on it) so the trajectory h_t can
        # be replayed at training; pi (AZ policy target) is set only where search ran.
        if self.record and single and options:
            self.move_log.append({
                "current": copy.deepcopy(cur),
                "select": copy.deepcopy(select),
                "choice": [int(c) for c in choice],
                "pi": self._last_pi,
            })
        return choice

    # --- the search -----------------------------------------------------------

    def _ismcts(self, obs: dict, options: list, base: list[int]) -> list[int]:
        from cg.api import (  # noqa: PLC0415
            search_begin,
            search_end,
            to_observation_class,
        )

        cur = obs["current"]
        your = int(cur.get("yourIndex", 0))
        obs_obj = to_observation_class(obs)
        seen = seen_opponent_ids(cur, your) if self.opp_decks else []
        root_sigs = [option_sig(o) for o in options]
        # Pre-seed the ROOT node's PUCT prior from the REAL options (which encode
        # cleanly, unlike some determinized search-state options) + the real LSTM state,
        # so root-visit concentration -- and thus the extracted pi -- is well guided.
        root_node = _Node()
        root_prior = self._prior({"current": cur, "select": obs["select"]}, your,
                                  h=self._h)
        for s, pr in zip(root_sigs, root_prior, strict=False):
            root_node.p[s] = float(pr)
        tree: dict[tuple, _Node] = {(): root_node}
        self.n_searches += 1
        deadline = time.perf_counter() + self.move_budget_s

        for _ in range(self.iterations):
            if time.perf_counter() >= deadline:
                break
            prior = (
                sample_consistent_deck(self.opp_decks, seen, self._rng)
                if self.opp_decks else self.opp_prior
            )
            det = sample_determinization(
                cur, your, self.deck, prior, self._rng, opp_basics=self.opp_basics,
            )
            try:
                root = search_begin(
                    obs_obj, det.your_deck, det.your_prize, det.opp_deck,
                    det.opp_prize, det.opp_hand, det.opp_active,
                )
                self._iterate(root, your, tree)
                search_end()
            except Exception:  # noqa: BLE001 - drop a bad determinization
                search_end()
        return self._gated_choice(tree, root_sigs, base, obs)

    def _iterate(self, root: object, your: int, tree: dict[tuple, _Node]) -> None:
        from cg.api import search_step  # noqa: PLC0415

        for r in self._rollers:
            r.reset(0)
        self._rollers[your]._h = np.array(self._h, copy=True)  # noqa: SLF001
        self._rollers[your]._c = np.array(self._c, copy=True)  # noqa: SLF001

        st = root
        path: list[tuple] = []
        while True:
            st, terminal, result, obs_d = self._advance(st, your)
            if terminal:
                self._backprop(tree, path, 1.0 if result == your else -1.0)
                return
            options_d = obs_d["select"]["option"]
            node = tree.setdefault(tuple(path), _Node())
            sigs = [option_sig(o) for o in options_d]
            for s in sigs:
                node.avail[s] = node.avail.get(s, 0) + 1
            if not node.p:                      # first visit -> net policy prior (PUCT)
                prior = self._prior(obs_d, your)
                for s, pr in zip(sigs, prior, strict=False):
                    node.p[s] = float(pr)
            i = self._puct(node, sigs)
            st = search_step(st.searchId, [i])  # type: ignore[attr-defined]
            path.append(sigs[i])
            if sigs[i] not in node.n:            # newly expanded edge -> evaluate leaf
                node.n[sigs[i]] = 0
                node.w[sigs[i]] = 0.0
                self._backprop(tree, path, self._leaf(st, your))
                return
            if len(path) >= self.max_depth:
                self._backprop(tree, path, self._leaf(st, your))
                return

    def _prior(self, obs_d: dict, your: int,
               h: np.ndarray | None = None) -> np.ndarray:
        """Net policy prior P(a) over the node's options (softmax of policy logits from
        an LSTM hidden state ``h`` -- the real state at the root, the roller's deeper).
        Falls back to a uniform prior if the search-state options fail to encode (some
        carry ``None`` fields, as the roller tolerates via its legal fallback)."""
        opts = obs_d.get("select", {}).get("option") or []
        unif = np.full(len(opts), 1.0 / max(len(opts), 1))
        try:
            cur = obs_d["current"]
            of = encode_options(opts, cur, your, self.feats)
            orows = option_embed_rows(opts, cur, your, self._index)
            hh = h if h is not None else self._rollers[your]._h  # noqa: SLF001
            logits = self.net.policy_logits_from_h(hh, of, orows)
        except Exception:  # noqa: BLE001 - degrade to uniform prior, never fail search
            return unif
        if logits.shape[0] != len(opts):
            return unif
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def _advance(self, st: object, your: int) -> tuple[object, bool, int, dict]:
        """Step through terminal / non-searchable / opponent decisions via the net.

        Returns ``(state, terminal, result, obs_d)`` stopped at our next single decision
        (or terminal, ``obs_d={}``). Opponent + our forced/multi-selects play out
        via the net so the tree only branches on our genuine single-select choices.
        """
        from cg.api import search_step  # noqa: PLC0415

        for _ in range(self.rollout_depth):
            o = st.observation  # type: ignore[attr-defined]
            state = o.current
            if state is None:
                return st, True, -1, {}
            if getattr(state, "result", -1) != -1:
                return st, True, int(state.result), {}
            sel = o.select
            opts = (sel.option or []) if sel is not None else []
            if not opts:
                return st, True, -1, {}
            yidx = int(state.yourIndex)
            ours = (yidx == your and int(sel.maxCount) == SINGLE_SELECT
                    and len(opts) >= _MIN_CHOICE)
            if ours:
                return st, False, -1, obs_to_dict(o)
            choice = self._rollers[yidx].act(obs_to_dict(o))
            if not choice:
                return st, True, -1, {}
            st = search_step(st.searchId, choice)  # type: ignore[attr-defined]
        return st, True, -1, {}

    def _puct(self, node: _Node, sigs: list[tuple]) -> int:
        """AlphaZero PUCT: ``Q(a) + c_puct * P(a) * sqrt(Sum_b N) / (1 + N(a))`` --
        the net policy prior ``P`` focuses visits on promising moves (vs uniform UCB),
        so the root visit-count policy sharpens into a usable training target."""
        total = sum(node.n.get(s, 0) for s in sigs)
        sq = math.sqrt(total + 1)
        unif = 1.0 / max(len(sigs), 1)
        best_i, best_u = 0, -1e18
        for i, s in enumerate(sigs):
            n = node.n.get(s, 0)
            q = node.w[s] / n if n > 0 else 0.0
            u = q + self.c_puct * node.p.get(s, unif) * sq / (1 + n)
            if u > best_u:
                best_i, best_u = i, u
        return best_i

    def _leaf(self, st: object, your: int) -> float:
        """Leaf value (1): roll out at most ``rollout_cap`` steps (0 = none), then read
        the VALUE HEAD -- instead of always rolling to terminal. ``rollout_cap`` large
        recovers the original roll-to-terminal behaviour."""
        from cg.api import search_step  # noqa: PLC0415

        for _ in range(self.rollout_cap):
            o = st.observation  # type: ignore[attr-defined]
            state = o.current
            if state is None:
                break
            if getattr(state, "result", -1) != -1:
                return 1.0 if int(state.result) == your else -1.0
            sel = o.select
            if sel is None or not (sel.option or []):
                break
            choice = self._rollers[int(state.yourIndex)].act(obs_to_dict(o))
            if not choice:
                break
            st = search_step(st.searchId, choice)  # type: ignore[attr-defined]
        # value-head estimate at the (non-terminal) leaf, from `your` perspective.
        o = st.observation  # type: ignore[attr-defined]
        state = o.current
        if state is None:
            return 0.0
        if getattr(state, "result", -1) != -1:
            return 1.0 if int(state.result) == your else -1.0
        sel = o.select
        if sel is None or not (sel.option or []):
            return 0.0
        yidx = int(state.yourIndex)
        self._rollers[yidx].act(obs_to_dict(o))       # one net forward -> last_value
        v = float(self._rollers[yidx].last_value)
        return v if yidx == your else -v

    def _backprop(self, tree: dict[tuple, _Node], path: list[tuple], r: float) -> None:
        for k in range(len(path)):
            node = tree[tuple(path[:k])]
            sig = path[k]
            node.n[sig] = node.n.get(sig, 0) + 1
            node.w[sig] = node.w.get(sig, 0.0) + r

    def _gated_choice(
        self, tree: dict[tuple, _Node], root_sigs: list[tuple], base: list[int],
        obs: dict,  # noqa: ARG002 - kept for signature symmetry
    ) -> list[int]:
        """Set ``self._last_pi`` to the root visit-count policy (the AZ target) and
        override the net only if the best searched root move clearly beats it."""
        root = tree.get(())
        if root is None or not root.n:
            return base

        def mean(s: tuple) -> float:
            n = root.n.get(s, 0)
            return root.w[s] / n if n > 0 else -2.0

        tot = sum(root.n.get(s, 0) for s in root_sigs)
        if tot:
            self._last_pi = [round(root.n.get(s, 0) / tot, 5) for s in root_sigs]

        best_sig = max(root.n, key=mean)
        if best_sig not in root_sigs:
            return base
        net_idx = base[0] if base else -1
        net_mean = mean(root_sigs[net_idx]) if 0 <= net_idx < len(root_sigs) else -2.0
        if mean(best_sig) > net_mean + self.gate_margin:
            self.searched += 1
            return [root_sigs.index(best_sig)]
        return base
