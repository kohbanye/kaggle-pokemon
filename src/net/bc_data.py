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
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from src.deck import card_kind, legal_next_ids
from src.net.embedding import CardEmbeddingIndex
from src.net.encode import (
    OPTION_DIM,
    encode_options,
    encode_state,
    option_embed_rows,
    state_embed_rows,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from numpy.typing import NDArray

    from src.deck import CardPool
    from src.net.features import CardFeatures

# A selection the policy head clones is exactly one chosen option.
SINGLE_SELECT = 1


@dataclass
class PolicySample:
    """One cloned decision: encoded state + presented options + teacher pick.

    Carries the shared-card-embedding rows alongside the fixed features so the
    forward can look the live embedding up (and so the play loss trains the shared
    table): ``state_rows``/``state_mask`` are the four state slots
    (:func:`~src.net.encode.state_embed_rows`) and ``option_rows`` the per-option
    target-card rows.
    """

    state: NDArray[np.float64]  # (STATE_DIM,)
    state_rows: NDArray[np.intp]  # (STATE_EMBED_SLOTS, SLOT_MAX)
    state_mask: NDArray[np.bool_]  # (STATE_EMBED_SLOTS, SLOT_MAX)
    options: NDArray[np.float64]  # (K, OPTION_DIM)
    option_rows: NDArray[np.intp]  # (K,) target-card embedding rows
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


def _make_policy_sample(  # noqa: PLR0913 - encodes one decision from its many parts
    decision: dict,
    winner: int,
    dist: int,
    feats: CardFeatures,
    index: CardEmbeddingIndex | None,
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
    state_rows, state_mask = state_embed_rows(current, slot, index)
    option_rows = option_embed_rows(options, current, slot, index)

    value = _outcome(winner, slot)
    if discount is not None:
        value *= discount**dist
    return PolicySample(
        state, state_rows, state_mask, option_feats, option_rows, target, value,
    )


def build_policy_samples(  # noqa: PLR0913 - one builder with several optional knobs
    games: Iterable[dict],
    feats: CardFeatures,
    index: CardEmbeddingIndex | None = None,
    *,
    teachers: set[str] | None = None,
    discount: float | None = None,
    max_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[PolicySample]:
    """Encode (single-select) teacher decisions into :class:`PolicySample` objects.

    ``index`` maps target/board card ids to shared-embedding rows for the play head
    (None => every play-head card uses the UNK row). ``teachers`` restricts which
    agents' decisions are cloned (the logs may hold both players' moves); None keeps
    all. ``discount`` enables the discounted return value target (``gamma ** d``),
    None keeps the raw final outcome.

    ``max_samples`` (with ``rng``) **subsamples**: a self-play iteration produces ~200
    correlated decisions per game, far more than a policy-gradient step needs, and
    encoding them all dominates the loop. Decision references are gathered cheaply
    (no encoding) and randomly thinned to ``max_samples`` *before* encoding, so both
    the encode and the train cost scale with ``max_samples``, not the raw count.
    """
    refs: list[tuple[dict, int, int]] = []  # (decision, winner, dist) -- not encoded
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
            refs.append((decision, winner, dist[i]))

    if max_samples is not None and rng is not None and len(refs) > max_samples:
        keep = rng.choice(len(refs), size=max_samples, replace=False)
        refs = [refs[i] for i in keep]

    samples: list[PolicySample] = []
    for decision, winner, dist_i in refs:
        sample = _make_policy_sample(decision, winner, dist_i, feats, index, discount)
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
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Pad to the batch-max option count and build the trainer's 8-tuple.

    Returns ``(states, state_rows, state_mask, options, option_mask, option_rows,
    targets, values)`` with shapes ``(B, STATE_DIM) / (B, S, SLOT_MAX) /
    (B, S, SLOT_MAX) / (B, K, OPTION_DIM) / (B, K) / (B, K) / (B,) / (B,)`` -- what
    the play arm of :class:`~src.net.lit.LitJointPolicyGradient` (and
    :class:`LitPolicyValue`) consumes. Padded option rows are 0 (a valid embedding
    row; the option_mask zeroes their contribution to the loss anyway).
    """
    bsz = len(batch)
    max_k = max(sample.options.shape[0] for sample in batch)
    states = torch.from_numpy(np.stack([sample.state for sample in batch])).float()
    state_rows = torch.from_numpy(
        np.stack([sample.state_rows for sample in batch]),
    ).long()
    state_mask = torch.from_numpy(np.stack([sample.state_mask for sample in batch]))
    options = torch.zeros(bsz, max_k, OPTION_DIM, dtype=torch.float32)
    option_mask = torch.zeros(bsz, max_k, dtype=torch.bool)
    option_rows = torch.zeros(bsz, max_k, dtype=torch.long)
    for i, sample in enumerate(batch):
        k = sample.options.shape[0]
        options[i, :k] = torch.from_numpy(sample.options).float()
        option_mask[i, :k] = True
        option_rows[i, :k] = torch.from_numpy(sample.option_rows).long()
    targets = torch.tensor([sample.target for sample in batch], dtype=torch.long)
    values = torch.tensor([sample.value for sample in batch], dtype=torch.float32)
    return (
        states, state_rows, state_mask, options, option_mask, option_rows,
        targets, values,
    )


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


def cb_rl_samples(
    decks_with_returns: Sequence[tuple[Sequence[int], float]],
    pool: CardPool,
    feats: CardFeatures,
    *,
    normalize: bool = True,
) -> tuple[NDArray[np.float64], list[CBSample]]:
    """Per-deck-step REINFORCE samples from sampled decks + their returns (5b-ii).

    ``decks_with_returns`` is ``[(deck_in_pick_order, mean_return), ...]`` -- each
    deck was *sampled* from the CB head and scored by playing it K times. The
    advantage ``return - baseline`` (baseline = batch mean, optionally divided by
    the batch std) is shared across that deck's 60 build steps and stored as each
    :class:`CBSample`'s ``weight``. Legal masks are recomputed from the deck prefix
    (:func:`~src.deck.legal_next_ids`), exactly as the CB head saw them at sampling
    time -- so we don't have to log the masks, only the pick order. Row order is
    ``sorted(pool.ids())`` (the embedding's rows), via :class:`CardEmbeddingIndex`.
    """
    index = CardEmbeddingIndex(pool)
    card_feats = index.fixed_matrix(feats)
    returns = np.asarray([r for _, r in decks_with_returns], dtype=np.float64)
    adv = returns - returns.mean() if returns.size else returns
    if normalize and adv.size > 1:
        std = float(adv.std())
        if std > 1e-8:  # noqa: PLR2004 - tiny-variance guard
            adv = adv / std

    samples: list[CBSample] = []
    for (deck, _), advantage in zip(decks_with_returns, adv, strict=True):
        for t in range(len(deck)):
            target = deck[t]
            row = index.row(target)
            if row >= index.n_pool:  # unknown card id
                continue
            legal = legal_next_ids(list(deck[:t]), pool)
            if target not in legal:
                continue
            mask = np.zeros(index.n_pool, dtype=np.bool_)
            for cid in legal:
                r = index.row(cid)
                if r < index.n_pool:
                    mask[r] = True
            samples.append(CBSample(mask, row, float(advantage)))
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


# --- CB *sequences* for the LSTM deck head (Phase 5c) ----------------------


@dataclass
class CBSequence:
    """One deck as a build *sequence* (for the autoregressive LSTM CB head).

    A deck is processed in pick order through the LSTM: ``targets[t]`` is the pool
    row picked at step ``t`` (it is also the LSTM input at step ``t+1``);
    ``legal_masks[t]`` is the legal set at that step; ``weights[t]`` is the
    type-composition weight (BC) or the deck's advantage (RL, constant across steps).
    """

    targets: NDArray[np.int64]  # (T,) pick-order pool rows
    legal_masks: NDArray[np.bool_]  # (T, N_pool)
    weights: NDArray[np.float64]  # (T,)


def _step_mask(prefix: list[int], pool: CardPool, index: CardEmbeddingIndex) -> NDArray:
    """Boolean legal-next mask over the pool for a partial deck ``prefix``."""
    mask = np.zeros(index.n_pool, dtype=np.bool_)
    for cid in legal_next_ids(prefix, pool):
        row = index.row(cid)
        if row < index.n_pool:
            mask[row] = True
    return mask


def _deck_sequence(
    order: list[int],
    pool: CardPool,
    index: CardEmbeddingIndex,
    weight_at: Callable[[int, int], float],
) -> CBSequence | None:
    """Turn a pick-ordered deck into a :class:`CBSequence` (None if no valid steps)."""
    targets: list[int] = []
    masks: list[NDArray] = []
    weights: list[float] = []
    for t, card in enumerate(order):
        row = index.row(card)
        if row >= index.n_pool:  # unknown card -> can't supervise this step
            continue
        legal = legal_next_ids(order[:t], pool)
        if card not in legal:
            continue
        targets.append(row)
        masks.append(_step_mask(order[:t], pool, index))
        weights.append(weight_at(t, card))
    if not targets:
        return None
    return CBSequence(
        np.array(targets, dtype=np.int64),
        np.stack(masks),
        np.array(weights, dtype=np.float64),
    )


def _type_target_weights(deck: Sequence[int], pool: CardPool) -> dict[int, float]:
    """Per-pick CB-loss weight that targets the deck's TYPE composition.

    Flat inverse-copy weighting (``1 / copies``) gives every *distinct* card equal
    total weight, so each card type's share of the loss is its distinct-card count
    -- which collapses energy (1 distinct card, ~half the deck) to a tiny fraction
    of the signal, and the LSTM head learns to under-pick it (greedy decode then
    bricks at ~0 energy). Instead, target each type's PHYSICAL share of the deck:
    type ``T`` (``N_T`` physical, ``D_T`` distinct cards) gets total weight
    ``N_T / deck_size``, split equally across its distinct cards (so Pokemon /
    Trainer variety is preserved, the good part of inverse-copy) and then across
    copies. A pick of a card with ``c`` copies thus weighs
    ``(N_T / deck_size) / (D_T * c)``; the deck's weights sum to 1.
    """
    counts = Counter(deck)
    distinct_by_kind: dict[str, list[int]] = defaultdict(list)
    physical_by_kind: Counter[str] = Counter()
    for cid, copies in counts.items():
        kind = card_kind(pool, cid)
        distinct_by_kind[kind].append(cid)
        physical_by_kind[kind] += copies
    size = max(len(deck), 1)
    weights: dict[int, float] = {}
    for kind, ids in distinct_by_kind.items():
        per_distinct = (physical_by_kind[kind] / size) / len(ids)
        for cid in ids:
            weights[cid] = per_distinct / counts[cid]
    return weights


def cb_sequences(
    decks: Sequence[Sequence[int]],
    pool: CardPool,
    feats: CardFeatures,
    rng: np.random.Generator,
    *,
    shuffles: int = 1,
) -> tuple[NDArray[np.float64], list[CBSequence]]:
    """BC sequences: each demo deck (shuffled) becomes one teacher-forced sequence.

    Returns the fixed ``(N_pool, CARD_FEAT_DIM)`` feature matrix (the LSTM trainer
    concatenates the live embedding) and one :class:`CBSequence` per deck-shuffle,
    weighted by :func:`_type_target_weights` so the learned composition matches the
    demo decks' card-type proportions (energy is no longer starved of signal).
    """
    index = CardEmbeddingIndex(pool)
    card_feats = index.fixed_matrix(feats)
    seqs: list[CBSequence] = []
    for deck in decks:
        weights = _type_target_weights(deck, pool)
        for _ in range(shuffles):
            order = [deck[i] for i in rng.permutation(len(deck))]
            seq = _deck_sequence(
                order, pool, index, lambda _t, card, w=weights: w.get(card, 0.0),
            )
            if seq is not None:
                seqs.append(seq)
    return card_feats, seqs


def cb_rl_sequences(
    decks_with_returns: Sequence[tuple[Sequence[int], float]],
    pool: CardPool,
    feats: CardFeatures,
    *,
    normalize: bool = True,
) -> tuple[NDArray[np.float64], list[CBSequence]]:
    """RL sequences: each sampled deck (pick order) weighted by its advantage.

    Advantage = ``return - batch_mean`` (optionally / batch std), shared across the
    deck's steps — the REINFORCE signal for the LSTM CB head (Phase 5b-ii redux).
    """
    index = CardEmbeddingIndex(pool)
    card_feats = index.fixed_matrix(feats)
    returns = np.asarray([r for _, r in decks_with_returns], dtype=np.float64)
    adv = returns - returns.mean() if returns.size else returns
    if normalize and adv.size > 1:
        std = float(adv.std())
        if std > 1e-8:  # noqa: PLR2004 - tiny-variance guard
            adv = adv / std

    seqs: list[CBSequence] = []
    for (deck, _), advantage in zip(decks_with_returns, adv, strict=True):
        seq = _deck_sequence(
            list(deck), pool, index, lambda _t, _card, a=float(advantage): a,
        )
        if seq is not None:
            seqs.append(seq)
    return card_feats, seqs


class CBSequenceDataset(Dataset):
    """A list of :class:`CBSequence`; pair with :func:`collate_cb_seq`."""

    def __init__(self, sequences: list[CBSequence]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> CBSequence:
        return self.sequences[idx]


def collate_cb_seq(
    batch: list[CBSequence],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad decks to the batch-max length T.

    Returns ``(targets (B,T) long, legal_masks (B,T,N_pool) bool, weights (B,T),
    valid (B,T) bool)``. Padded steps get a dummy-legal row 0 (so the per-step
    softmax is never all ``-inf`` -> no nan) and ``valid = False`` (excluded from
    the loss). Demo decks are all 60 steps, so padding only bites mixed test decks.
    """
    bsz = len(batch)
    max_t = max(seq.targets.shape[0] for seq in batch)
    n_pool = batch[0].legal_masks.shape[1]
    targets = torch.zeros(bsz, max_t, dtype=torch.long)
    masks = torch.zeros(bsz, max_t, n_pool, dtype=torch.bool)
    weights = torch.zeros(bsz, max_t, dtype=torch.float32)
    valid = torch.zeros(bsz, max_t, dtype=torch.bool)
    for i, seq in enumerate(batch):
        t = seq.targets.shape[0]
        targets[i, :t] = torch.from_numpy(seq.targets)
        masks[i, :t] = torch.from_numpy(seq.legal_masks)
        weights[i, :t] = torch.from_numpy(seq.weights).float()
        valid[i, :t] = True
        masks[i, t:, 0] = True  # dummy-legal for padded steps (avoids all-(-inf))
    return targets, masks, weights, valid
