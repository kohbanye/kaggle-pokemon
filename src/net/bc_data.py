"""Phase-4 behaviour-cloning dataset: teacher logs -> training batches.

Turns the JSONL decision logs produced by ``scripts/collect_bc.py`` (one game per
line, each carrying every decision's deep-copied observation + chosen action +
deciding slot, plus the game winner) into the exact 5-tuple
``(states, options, option_mask, targets, values)`` that
:class:`~src.net.lit.LitPolicyValue` trains on, and demo decklists into the
``(legal_mask, target)`` per-step supervision the CB head trains on.

The split mirrors the project's Docker/native boundary: the engine-bound *sim*
runs under Docker (collection), and this module (pure numpy + torch, no ``cg``)
runs on the host (encoding + batching). Observations are encoded here, not at
collection time, so the encoder/feature set can be ablated without re-simulating
(the engine RNG is unseedable -- re-running yields different games, so the cached
raw logs are the only way to hold the data fixed while varying the encoder).

Only **single-select** decisions become policy targets (``maxCount == 1``): the
multi-select sub-choices are mostly forced and the serving ``NetAgent`` derives
them as the top-``maxCount`` scored options anyway, so cloning the single-select
head transfers. Value targets are the deciding slot's game outcome (+1 win / -1
loss / 0 draw|abort), optionally discounted by ``gamma ** d`` where ``d`` is that
slot's number of remaining own decisions (the final-vs-discounted ablation knob).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from src.deck import legal_next_ids
from src.net.encode import OPTION_DIM, encode_options, encode_state

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from numpy.typing import NDArray

    from src.deck import CardPool
    from src.net.features import CardFeatures

# A selection the policy head clones is exactly one chosen option.
SINGLE_SELECT = 1


@dataclass
class PolicySample:
    """One cloned decision: encoded state + presented options + teacher pick."""

    state: NDArray[np.float64]  # (STATE_DIM,)
    options: NDArray[np.float64]  # (K, OPTION_DIM)
    target: int  # index of the teacher's chosen option, 0 <= target < K
    value: float  # outcome in [-1, 1] from the deciding slot's view


@dataclass
class CBSample:
    """One deck-building step: which pool cards are legal + the demo's pick.

    ``weight`` is ``1 / copies-of-target-in-deck`` so every *distinct* card in a
    deck contributes equal total loss: without it the per-step "next card" target
    is dominated by Basic Energy (~2/3 of an aggro deck), the CB logit for energy
    dwarfs everything, and greedy decode picks energy for all 60 slots. Equal
    per-distinct-card weight lets greedy decode 4-ofs of the real cards + energy
    fill (a meta-like deck).
    """

    legal_mask: NDArray[np.bool_]  # (N_pool,) True where the card is a legal next pick
    target_idx: int  # pool index of the demo deck's card at this step
    weight: float  # 1 / copies of the target card in its deck


# --- loading the collected logs --------------------------------------------


def load_engine_json(path: str | Path) -> dict:
    """Load the engine card/attack dump, restoring int keys (JSON stringifies them)."""
    raw = json.loads(Path(path).read_text())
    return {
        "cards": {int(k): v for k, v in raw.get("cards", {}).items()},
        "attacks": {int(k): v for k, v in raw.get("attacks", {}).items()},
    }


def game_files(data_dir: str | Path) -> list[Path]:
    """The JSONL game shards under ``<data_dir>/games/``."""
    return sorted((Path(data_dir) / "games").glob("*.jsonl"))


def iter_games(paths: Sequence[str | Path]) -> Iterator[dict]:
    """Yield game records (one JSON object per line) from the given JSONL files."""
    for path in paths:
        with Path(path).open() as handle:
            for raw in handle:
                text = raw.strip()
                if text:
                    yield json.loads(text)


# --- policy / value samples ------------------------------------------------


def _outcome(winner: int, slot: int) -> float:
    """Game result in [-1, 1] from ``slot``'s view (+1 win, -1 loss, 0 draw/abort)."""
    if winner == slot:
        return 1.0
    if winner == 1 - slot:
        return -1.0
    return 0.0


def _make_policy_sample(
    decision: dict,
    winner: int,
    dist: int,
    feats: CardFeatures,
    discount: float | None,
) -> PolicySample | None:
    """Encode one single-select decision into a sample, or None if not usable."""
    obs = decision.get("obs") or {}
    select = obs.get("select") or {}
    if int(select.get("maxCount", 0)) != SINGLE_SELECT:
        return None
    choice = decision.get("choice") or []
    if len(choice) != SINGLE_SELECT:
        return None
    target = int(choice[0])
    options = select.get("option") or []
    if not options or not 0 <= target < len(options):
        return None

    slot = int(decision.get("slot", 0))
    current = obs.get("current") or {}
    state = encode_state(current, slot, feats)
    option_feats = encode_options(options, current, slot, feats)

    value = _outcome(winner, slot)
    if discount is not None:
        value *= discount**dist
    return PolicySample(state, option_feats, target, value)


def build_policy_samples(
    games: Iterable[dict],
    feats: CardFeatures,
    *,
    teachers: set[str] | None = None,
    discount: float | None = None,
) -> list[PolicySample]:
    """Encode every (single-select) teacher decision into a :class:`PolicySample`.

    ``teachers`` restricts which agents' decisions are cloned (the logs may hold
    both players' moves); None keeps all. ``discount`` enables the discounted
    return value target (``gamma ** d``), None keeps the raw final outcome.
    """
    samples: list[PolicySample] = []
    for game in games:
        winner = int(game.get("winner", -1))
        decisions = game.get("decisions") or []

        # Distance to this slot's own game-end, per decision (for discounting).
        seen: dict[int, int] = {}
        dist = [0] * len(decisions)
        for i in reversed(range(len(decisions))):
            slot = int(decisions[i].get("slot", 0))
            dist[i] = seen.get(slot, 0)
            seen[slot] = dist[i] + 1

        for i, decision in enumerate(decisions):
            if teachers is not None and decision.get("agent") not in teachers:
                continue
            sample = _make_policy_sample(decision, winner, dist[i], feats, discount)
            if sample is not None:
                samples.append(sample)
    return samples


class PolicyDataset(Dataset):
    """A list of :class:`PolicySample`; pair with :func:`collate_policy`."""

    def __init__(self, samples: list[PolicySample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PolicySample:
        return self.samples[idx]


def collate_policy(
    batch: list[PolicySample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad to the batch-max option count and build the trainer's 5-tuple.

    Returns ``(states, options, option_mask, targets, values)`` with shapes
    ``(B, STATE_DIM) / (B, K, OPTION_DIM) / (B, K) / (B,) / (B,)`` and dtypes
    float32 / float32 / bool / long / float32 -- exactly what
    :meth:`~src.net.lit.LitPolicyValue.training_step` consumes.
    """
    bsz = len(batch)
    max_k = max(sample.options.shape[0] for sample in batch)
    states = torch.from_numpy(np.stack([sample.state for sample in batch])).float()
    options = torch.zeros(bsz, max_k, OPTION_DIM, dtype=torch.float32)
    option_mask = torch.zeros(bsz, max_k, dtype=torch.bool)
    for i, sample in enumerate(batch):
        k = sample.options.shape[0]
        options[i, :k] = torch.from_numpy(sample.options).float()
        option_mask[i, :k] = True
    targets = torch.tensor([sample.target for sample in batch], dtype=torch.long)
    values = torch.tensor([sample.value for sample in batch], dtype=torch.float32)
    return states, options, option_mask, targets, values


# --- CB (deck-construction) samples ----------------------------------------


def cb_supervision(
    decks: Sequence[Sequence[int]],
    pool: CardPool,
    feats: CardFeatures,
    rng: np.random.Generator,
    *,
    shuffles: int = 1,
) -> tuple[NDArray[np.float64], list[CBSample]]:
    """Per-step ``(legal_mask, target)`` supervision from demo decks.

    Returns the fixed ``(N_pool, CARD_FEAT_DIM)`` card-feature matrix (the CB head
    scores it once) and one :class:`CBSample` per build step: each demo deck is
    shuffled ``shuffles`` times and, at step ``t``, the legal next cards
    (:func:`~src.deck.legal_next_ids` over ``deck[:t]``) form the mask and the
    deck's card at ``t`` is the target -- always within the legal set by
    construction.
    """
    ids = sorted(pool.ids())
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    card_feats = np.stack([feats.vector(cid) for cid in ids])

    samples: list[CBSample] = []
    for deck in decks:
        counts = Counter(deck)
        for _ in range(shuffles):
            order = [deck[i] for i in rng.permutation(len(deck))]
            for t in range(len(order)):
                target = order[t]
                if target not in id_to_idx:
                    continue
                legal = legal_next_ids(order[:t], pool)
                if target not in legal:
                    continue
                mask = np.zeros(len(ids), dtype=np.bool_)
                for cid in legal:
                    idx = id_to_idx.get(cid)
                    if idx is not None:
                        mask[idx] = True
                samples.append(CBSample(mask, id_to_idx[target], 1.0 / counts[target]))
    return card_feats, samples


class CBDataset(Dataset):
    """A list of :class:`CBSample`; pair with :func:`collate_cb`."""

    def __init__(self, samples: list[CBSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CBSample:
        return self.samples[idx]


def collate_cb(
    batch: list[CBSample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack into ``(legal_mask (B, N_pool) bool, target (B,) long, weight (B,))``."""
    masks = torch.from_numpy(np.stack([sample.legal_mask for sample in batch]))
    targets = torch.tensor([sample.target_idx for sample in batch], dtype=torch.long)
    weights = torch.tensor([sample.weight for sample in batch], dtype=torch.float32)
    return masks, targets, weights
