"""Encode an observation into fixed-length net inputs.

Two encoders, both pure ``dict -> np.ndarray`` and both defensive (any missing /
malformed field degrades to zeros rather than raising -- the agent must never
crash a match):

- :func:`encode_state` turns ``obs['current']`` (the ``State`` dict) into a
  fixed-length vector of length :data:`STATE_DIM`, oriented from the selecting
  player's view (me = ``players[yourIndex]``, opponent = the other). Each player
  contributes an Active block, a pooled Bench block and resource scalars; a small
  global block carries turn / per-turn-flag context.
- :func:`encode_option` turns one presented ``Option`` into a vector of length
  :data:`OPTION_DIM`: its option-type one-hot, the feature vector of the card it
  targets, where that target sits (active / bench / mine), and -- for attacks --
  the attack's damage and cost.

The encoders read card stats through :class:`~src.net.features.CardFeatures`,
which the runner builds from injected engine data, so nothing here imports ``cg``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.net.features import CARD_FEAT_DIM

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.net.embedding import CardEmbeddingIndex
    from src.net.features import CardFeatures

# AreaType (mirror of cg.api.AreaType) -- the in-play areas an option points at.
# Mirrored locally (like NUM_OPTION_TYPES below and features.py's enum sizes) so
# the net layer stays decoupled from src.agents (avoids an import cycle: agents
# imports the net, so the net must not import agents).
AREA_HAND = 2
AREA_ACTIVE = 4
AREA_BENCH = 5

# OptionType spans 0..16 (cg.api.OptionType); one-hot width for an option's type.
NUM_OPTION_TYPES = 17

# Per-block widths (concatenation order documented in the module docstring).
_ACTIVE_SCALAR_WIDTH = 3  # hp fraction, energy count, has-active
_SPECIAL_WIDTH = 5  # poisoned, burned, asleep, paralyzed, confused
_BENCH_SCALAR_WIDTH = 2  # bench size, bench energy total
_RESOURCE_WIDTH = 3  # prize, deck, hand counts
_GLOBAL_WIDTH = 5  # turn, supporter/stadium/energy/retreat flags
_OPTION_FLAG_WIDTH = 3  # targets active, targets bench, targets mine
_OPTION_ATTACK_WIDTH = 2  # attack damage, attack cost
_OPTION_NUMBER_WIDTH = 1

PLAYER_BLOCK_DIM = (
    CARD_FEAT_DIM  # active card features
    + _ACTIVE_SCALAR_WIDTH
    + _SPECIAL_WIDTH
    + CARD_FEAT_DIM  # pooled bench features
    + _BENCH_SCALAR_WIDTH
    + _RESOURCE_WIDTH
)
STATE_DIM = 2 * PLAYER_BLOCK_DIM + _GLOBAL_WIDTH
OPTION_DIM = (
    NUM_OPTION_TYPES
    + CARD_FEAT_DIM
    + _OPTION_FLAG_WIDTH
    + _OPTION_ATTACK_WIDTH
    + _OPTION_NUMBER_WIDTH
)

# Learned-embedding card slots in the state (shared card embedding, Phase 5d): the
# four board groups whose cards get a learned embedding fed into the play head.
# Each slot is a padded row list (+ mask); the forward masked-means each slot's
# embeddings, so a single-card slot (active) and a multi-card slot (bench) use the
# same machinery. SLOT_MAX caps cards per slot (bench is <=5 in standard rules; 8
# leaves headroom). Order: my active, opp active, my bench, opp bench.
STATE_EMBED_SLOTS = 4
SLOT_MAX = 8

# Normalisers (see features.py: scale only needs to be sane, not exact).
_ENERGY_NORM = 4.0
_BENCH_ENERGY_NORM = 8.0
_BENCH_SIZE_NORM = 5.0
_PRIZE_NORM = 6.0
_DECK_NORM = 60.0
_HAND_NORM = 10.0
_TURN_NORM = 20.0
_NUMBER_NORM = 10.0
_ATK_DAMAGE_NORM = 200.0
_ATK_COST_NORM = 5.0


def _active_pokemon(player: dict) -> dict | None:
    """The face-up Active Pokemon dict, or None (empty spot or face-down)."""
    spot = player.get("active") or []
    return spot[0] if spot and spot[0] is not None else None


def _card_id_at(player: dict, area: int, index: int) -> int | None:
    """Card id of the card at ``(area, index)`` in ``player`` (None if hidden)."""
    if index is None or index < 0:
        return None
    if area == AREA_ACTIVE:
        spot = player.get("active") or []
    elif area == AREA_BENCH:
        spot = player.get("bench") or []
    elif area == AREA_HAND:
        spot = player.get("hand")  # None for the opponent (hidden hand)
    else:
        return None
    if not spot or not 0 <= index < len(spot):
        return None
    card = spot[index]
    return None if card is None else card.get("id")


def _player_block(
    player: dict,
    feats: CardFeatures,
) -> NDArray[np.float64]:
    """Encode one player's board into a :data:`PLAYER_BLOCK_DIM` vector."""
    active = _active_pokemon(player)
    if active is not None:
        active_feat = feats.vector(active.get("id"))
        max_hp = active.get("maxHp") or 0
        hp_frac = active.get("hp", 0) / max_hp if max_hp > 0 else 0.0
        active_scalars = [
            hp_frac,
            len(active.get("energies") or []) / _ENERGY_NORM,
            1.0,
        ]
    else:
        active_feat = feats.vector(None)
        active_scalars = [0.0, 0.0, 0.0]

    special = [
        float(bool(player.get("poisoned"))),
        float(bool(player.get("burned"))),
        float(bool(player.get("asleep"))),
        float(bool(player.get("paralyzed"))),
        float(bool(player.get("confused"))),
    ]

    bench = player.get("bench") or []
    if bench:
        bench_feat = np.mean(
            [feats.vector(pk.get("id")) for pk in bench], axis=0,
        )
        bench_energy = sum(len(pk.get("energies") or []) for pk in bench)
    else:
        bench_feat = feats.vector(None)
        bench_energy = 0
    bench_scalars = [
        len(bench) / _BENCH_SIZE_NORM,
        bench_energy / _BENCH_ENERGY_NORM,
    ]

    resources = [
        len(player.get("prize") or []) / _PRIZE_NORM,
        player.get("deckCount", 0) / _DECK_NORM,
        player.get("handCount", 0) / _HAND_NORM,
    ]

    return np.concatenate([
        active_feat,
        np.asarray(active_scalars, dtype=np.float64),
        np.asarray(special, dtype=np.float64),
        bench_feat,
        np.asarray(bench_scalars, dtype=np.float64),
        np.asarray(resources, dtype=np.float64),
    ])


def encode_state(
    current: dict | None,
    your_index: int,
    feats: CardFeatures,
) -> NDArray[np.float64]:
    """Encode ``obs['current']`` into a :data:`STATE_DIM` vector (me-then-opp)."""
    if not current:
        return np.zeros(STATE_DIM, dtype=np.float64)
    players = current.get("players") or []
    if len(players) < 2:  # noqa: PLR2004 - the engine always sends exactly 2
        return np.zeros(STATE_DIM, dtype=np.float64)
    me = players[your_index]
    opp = players[1 - your_index]

    glob = [
        current.get("turn", 0) / _TURN_NORM,
        float(bool(current.get("supporterPlayed"))),
        float(bool(current.get("stadiumPlayed"))),
        float(bool(current.get("energyAttached"))),
        float(bool(current.get("retreated"))),
    ]
    return np.concatenate([
        _player_block(me, feats),
        _player_block(opp, feats),
        np.asarray(glob, dtype=np.float64),
    ])


def _option_target(
    option: dict,
    current: dict | None,
    your_index: int,
) -> tuple[int | None, int, int]:
    """Resolve the option's target ``(card_id, target_area, owner)``.

    Shared by :func:`encode_option` (fixed features) and :func:`option_card_rows`
    (embedding rows) so the two never disagree on which card an option acts on.
    """
    players = (current or {}).get("players") or []
    owner = int(option.get("playerIndex", your_index))
    # Prefer the on-field Pokemon the option acts on; fall back to its source card.
    in_area = option.get("inPlayArea")
    in_index = option.get("inPlayIndex")
    if in_area is not None and in_index is not None:
        target_area, target_index = int(in_area), int(in_index)
    else:
        target_area = int(option.get("area", -1))
        target_index = int(option.get("index", -1))

    target_id: int | None = None
    if 0 <= owner < len(players):
        target_id = _card_id_at(players[owner], target_area, target_index)
    return target_id, target_area, owner


def encode_option(
    option: dict,
    current: dict | None,
    your_index: int,
    feats: CardFeatures,
) -> NDArray[np.float64]:
    """Encode one presented ``Option`` into an :data:`OPTION_DIM` vector."""
    opt_type = int(option.get("type", -1))
    type_onehot = np.zeros(NUM_OPTION_TYPES, dtype=np.float64)
    if 0 <= opt_type < NUM_OPTION_TYPES:
        type_onehot[opt_type] = 1.0

    target_id, target_area, owner = _option_target(option, current, your_index)
    target_feat = feats.vector(target_id)

    flags = [
        float(target_area == AREA_ACTIVE),
        float(target_area == AREA_BENCH),
        float(owner == your_index),
    ]

    aid = option.get("attackId")
    attack = feats.attacks.get(aid) if isinstance(aid, int) else None
    attack_feats = [
        (attack.get("dmg", 0) if attack else 0) / _ATK_DAMAGE_NORM,
        (len(attack.get("cost", [])) if attack else 0) / _ATK_COST_NORM,
    ]

    number = option.get("number")
    number_feat = [(number or 0) / _NUMBER_NORM]

    return np.concatenate([
        type_onehot,
        target_feat,
        np.asarray(flags, dtype=np.float64),
        np.asarray(attack_feats, dtype=np.float64),
        np.asarray(number_feat, dtype=np.float64),
    ])


def encode_options(
    options: list[dict],
    current: dict | None,
    your_index: int,
    feats: CardFeatures,
) -> NDArray[np.float64]:
    """Stack the option encodings into a ``(len(options), OPTION_DIM)`` matrix."""
    if not options:
        return np.zeros((0, OPTION_DIM), dtype=np.float64)
    return np.stack([
        encode_option(opt, current, your_index, feats) for opt in options
    ])


# --- learned-embedding row indices (shared card embedding, Phase 5d) ---------
#
# These return the ``CardEmbeddingIndex`` rows for the cards the play head should
# embed; the forward looks the embedding up from ``cb_embed`` (differentiable in
# torch, so the play loss trains the *shared* table). They never raise -- a missing
# index or unknown id degrades to the UNK row (``index.n_pool``).


def _bench_ids(player: dict) -> list[int | None]:
    return [pk.get("id") for pk in (player.get("bench") or []) if pk is not None]


def state_embed_rows(
    current: dict | None,
    your_index: int,
    index: CardEmbeddingIndex | None,
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    """Padded embedding rows + mask for the four state slots.

    Returns ``rows`` ``(STATE_EMBED_SLOTS, SLOT_MAX)`` and ``mask`` (same shape,
    True where a real card sits). Slot order: my active, opp active, my bench, opp
    bench. With no ``index`` (or no state) every slot is empty (mask all False), so
    the masked-mean contributes zeros -- the pre-embedding behaviour.
    """
    rows = np.zeros((STATE_EMBED_SLOTS, SLOT_MAX), dtype=np.intp)
    mask = np.zeros((STATE_EMBED_SLOTS, SLOT_MAX), dtype=np.bool_)
    players = (current or {}).get("players") or []
    if index is None or len(players) < 2:  # noqa: PLR2004 - engine always sends 2
        return rows, mask
    me = players[your_index]
    opp = players[1 - your_index]
    me_active = _active_pokemon(me)
    opp_active = _active_pokemon(opp)
    slots: list[list[int | None]] = [
        [me_active.get("id")] if me_active is not None else [],
        [opp_active.get("id")] if opp_active is not None else [],
        _bench_ids(me),
        _bench_ids(opp),
    ]
    for s, ids in enumerate(slots):
        for j, cid in enumerate(ids[:SLOT_MAX]):
            rows[s, j] = index.row(cid)
            mask[s, j] = True
    return rows, mask


def option_embed_rows(
    options: list[dict],
    current: dict | None,
    your_index: int,
    index: CardEmbeddingIndex | None,
) -> NDArray[np.intp]:
    """Embedding row per presented option's target card -- ``(len(options),)``.

    Unknown / untargeted options (and a missing ``index``) map to the UNK row.
    """
    if not options:
        return np.zeros(0, dtype=np.intp)
    unk = index.n_pool if index is not None else 0
    if index is None:
        return np.full(len(options), unk, dtype=np.intp)
    return np.asarray(
        [
            index.row(_option_target(opt, current, your_index)[0])
            for opt in options
        ],
        dtype=np.intp,
    )
