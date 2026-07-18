"""Increment-2 V-GUIDED Archaludon pilot (Linux x86-64 native, imports ``cg``).

Wraps the bespoke rule-based :class:`ArchaludonAgent` and overrides ONLY the
highest-leverage *offensive-commitment* frames with a shallow, value-net-guided
1-ply lookahead:

* the **ATTACK** frame  (any single-select frame that offers an ``ATTACK`` option
  -- lets V weigh attacking now vs holding / setting up), and
* the **Boss's-Orders target** frame (``SWITCH`` / ``TO_ACTIVE`` where the choices
  are opponent Pokemon -- which benched threat to drag active).

On those frames it enumerates the legal candidate options, and for each one uses
the engine's ``search_begin`` / ``search_step`` lookahead (opponent hidden state
DETERMINIZED via the same belief machinery as ``src/search/ismcts.py``) to reach
the state ONE ply after applying that action, encodes it with the Increment-1
27-feature ``encode_state`` encoder, scores it with the trained value net
V(state)->P(Archaludon wins), and picks argmax-V. It falls back to the rule
agent's own choice on any error or when V is within ``epsilon`` of the rule
agent's pick (don't break good defaults on a near-tie).

Everything else delegates verbatim to the rule agent (setup ordering, lethal Boss
logic, energy attach, evolve). ``ty`` / ``ruff`` excluded like the other
engine-importing bespoke pilots.

Local A/B only (this increment is not a submission), so ``torch`` is fine here.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from scripts.collect_value_data import encode_state
from scripts.train_value_net import ValueMLP
from src.agents.archaludon_pilot import ArchaludonAgent
from src.agents.archaludon_pilot import agent as _arch_agent
from src.search.determinize import sample_determinization
from src.search.opp_belief import sample_consistent_deck, seen_opponent_ids

# raw-dict enum values (obs dict carries ints, not the OptionType/SelectContext enums)
_OPT_ATTACK = 13          # OptionType.ATTACK
_OPT_END = 14             # OptionType.END
_CTX_SWITCH = 3           # SelectContext.SWITCH
_CTX_TO_ACTIVE = 4        # SelectContext.TO_ACTIVE
_SINGLE = 1


def _plain_current(observation: object) -> dict | None:
    """Engine ``Observation`` dataclass -> its ``current`` state as a plain dict.

    Only the ``current`` sub-tree is needed (``encode_state`` reads
    ``players[*]`` and a couple of scalars), so we convert lazily and shallowly.
    """
    from dataclasses import asdict  # noqa: PLC0415

    cur = getattr(observation, "current", None)
    if cur is None:
        return None
    try:
        return asdict(cur)
    except Exception:  # noqa: BLE001 - degrade rather than crash the match
        return None


class ArchaludonVAgent:
    """Rule-based Archaludon + value-net 1-ply lookahead on offensive frames."""

    name = "archaludon_v"

    def __init__(  # noqa: PLR0913 - the search config is genuinely several knobs
        self,
        deck: list[int],
        engine: dict | None = None,
        *,
        weights: str = "data/value_net.pt",
        opp_decks: list[list[int]] | None = None,
        n_det: int = 6,
        epsilon: float = 0.03,
        seed: int = 0,
        strict_attack: bool = False,
    ) -> None:
        self._deck = list(deck)
        self._inner = ArchaludonAgent(deck, engine)
        self.n_det = n_det
        self.epsilon = epsilon
        # strict_attack=True: on an attack frame, only search the ATTACK-type
        # options (which attack / attack-vs-hold via END) instead of every legal
        # option -- a much narrower override surface.
        self.strict_attack = strict_attack
        self._rng = np.random.default_rng(seed)

        cards = (engine or {}).get("cards", {})
        self._ex_ids = {cid for cid, c in cards.items() if c.get("ex")}
        self._mega_ids = {cid for cid, c in cards.items() if c.get("mega")}
        self._opp_decks = [list(d) for d in (opp_decks or [])]
        basics: set[int] = set()
        for d in self._opp_decks:
            basics |= {cid for cid in d if cards.get(cid, {}).get("basic")}
        self._opp_basics = list(basics)

        blob = torch.load(weights, map_location="cpu", weights_only=False)
        self._mu = np.asarray(blob["mu"], dtype=np.float32)
        self._sd = np.asarray(blob["sd"], dtype=np.float32)
        self._net = ValueMLP(self._mu.shape[0])
        self._net.load_state_dict(blob["state_dict"])
        self._net.eval()

        # instrumentation
        self.move_times_ms: list[float] = []
        self.lookahead_times_ms: list[float] = []
        self.n_lookahead = 0
        self.n_override = 0

    def reset(self, seed: int = 0) -> None:
        self._inner.reset(seed)
        self._rng = np.random.default_rng(seed)

    # --- value net ------------------------------------------------------------
    def _value(self, feats: np.ndarray) -> float:
        x = (feats - self._mu) / self._sd
        with torch.no_grad():
            logit = self._net(torch.from_numpy(x.astype(np.float32)).unsqueeze(0))
            return float(torch.sigmoid(logit).item())

    def _value_of_state(self, st: object, your: int) -> float | None:
        """P(Archaludon wins) at the state one ply after our candidate action."""
        cur = _plain_current(st.observation)  # type: ignore[attr-defined]
        if cur is None or len(cur.get("players") or []) < 2:  # noqa: PLR2004
            return None
        res = cur.get("result", -1)
        if res != -1:
            return 1.0 if int(res) == your else 0.0
        try:
            feats = encode_state(cur, your, self._ex_ids, self._mega_ids)
        except Exception:  # noqa: BLE001
            return None
        return self._value(feats)

    # --- 1-ply lookahead ------------------------------------------------------
    def _lookahead(
        self, obs_dict: dict, candidates: list[int], your: int,
    ) -> dict[int, float]:
        from cg.api import (  # noqa: PLC0415
            search_begin,
            search_end,
            search_step,
            to_observation_class,
        )

        obs_obj = to_observation_class(obs_dict)
        cur = obs_dict["current"]
        seen = seen_opponent_ids(cur, your) if self._opp_decks else []
        acc: dict[int, list[float]] = {i: [] for i in candidates}
        for _ in range(self.n_det):
            prior = (
                sample_consistent_deck(self._opp_decks, seen, self._rng)
                if self._opp_decks else self._deck
            )
            det = sample_determinization(
                cur, your, self._deck, prior, self._rng,
                opp_basics=self._opp_basics,
            )
            for i in candidates:
                try:
                    root = search_begin(
                        obs_obj, det.your_deck, det.your_prize, det.opp_deck,
                        det.opp_prize, det.opp_hand, det.opp_active,
                    )
                    st = search_step(root.searchId, [i])
                    v = self._value_of_state(st, your)
                    search_end()
                except Exception:  # noqa: BLE001 - drop this (candidate, world)
                    try:
                        search_end()
                    except Exception:  # noqa: BLE001, S110
                        pass
                    continue
                if v is not None:
                    acc[i].append(v)
        return {i: float(np.mean(v)) for i, v in acc.items() if v}

    # --- frame classification -------------------------------------------------
    def _candidates(self, sel: dict, your: int) -> list[int] | None:
        """Return the candidate option indices to search, or None to defer."""
        opts = sel.get("option") or []
        if int(sel.get("maxCount", 0)) != _SINGLE or len(opts) < 2:  # noqa: PLR2004
            return None
        ctx = int(sel.get("context", -1))

        # ATTACK frame: any single-select frame offering an attack option.
        if any(int(o.get("type", -1)) == _OPT_ATTACK for o in opts):
            if self.strict_attack:
                # narrow: only the attack options + Turn-End (which attack / hold).
                cand = [i for i, o in enumerate(opts)
                        if int(o.get("type", -1)) in (_OPT_ATTACK, _OPT_END)]
                return cand if len(cand) >= 2 else None  # noqa: PLR2004
            # broad: all options, so V weighs attacking vs any alternative line.
            return list(range(len(opts)))

        # Boss's-Orders target: pick which OPPONENT Pokemon to drag active.
        if ctx in (_CTX_SWITCH, _CTX_TO_ACTIVE):
            opp = 1 - your
            cand = [i for i, o in enumerate(opts)
                    if o.get("playerIndex") is not None
                    and int(o["playerIndex"]) == opp]
            if len(cand) >= 2:  # noqa: PLR2004
                return cand
        return None

    # --- agent protocol -------------------------------------------------------
    def __call__(self, obs_dict: dict) -> list[int]:
        if obs_dict.get("select") is None:
            self.reset(0)
            return list(self._deck)

        t0 = time.perf_counter()
        # baseline choice (also runs the rule agent's opponent-attack tracking).
        try:
            base = _arch_agent(obs_dict)
        except Exception:  # noqa: BLE001
            base = None

        try:
            choice = self._maybe_override(obs_dict, base)
        except Exception:  # noqa: BLE001 - never crash a match
            choice = base
        self.move_times_ms.append(1000.0 * (time.perf_counter() - t0))
        return choice if choice is not None else base

    def _maybe_override(self, obs_dict: dict, base: list[int] | None) -> list[int] | None:
        sel = obs_dict.get("select") or {}
        cur = obs_dict.get("current")
        if cur is None or not obs_dict.get("search_begin_input"):
            return base
        your = int(cur.get("yourIndex", 0))
        candidates = self._candidates(sel, your)
        if candidates is None:
            return base

        t0 = time.perf_counter()
        means = self._lookahead(obs_dict, candidates, your)
        self.lookahead_times_ms.append(1000.0 * (time.perf_counter() - t0))
        self.n_lookahead += 1
        if len(means) < 2:  # noqa: PLR2004
            return base

        best_i = max(means, key=lambda i: means[i])
        base_i = base[0] if base and len(base) == 1 else None
        if base_i is None or base_i not in means:
            # rule agent picked outside the searched set -> trust it (conservative).
            return base
        if means[best_i] - means[base_i] > self.epsilon:
            self.n_override += 1
            return [best_i]
        return base
