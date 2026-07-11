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
_OPT_PLAY = 7  # play-a-card-from-hand (its ``index`` is a HAND slot, area implied)

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

# Rich target-STATE features (opt-in ``rich=True``; appended after the base block).
# The base encoding only sees the target card's PRINTED stats, so two options acting
# on same-name Pokemon in different board states encode identically -- measured to
# make 21.6% of the search-teacher's override decisions untrainable (bottleneck diag
# 2026-07-08). These capture the decision-relevant CURRENT state: energy progress
# toward attack costs (the attach-targeting signal), lethality/threat, HP.
_OPTION_RICH_WIDTH = 10
OPTION_DIM_RICH = OPTION_DIM + _OPTION_RICH_WIDTH

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
        # OPT_PLAY options carry only ``index`` (a hand slot; the engine leaves the
        # area implicit). Without this default the played card was UNRESOLVABLE --
        # zero card features + UNK embedding row for EVERY play option, i.e. the net
        # could never see WHICH card it was playing (bottleneck diag 2026-07-08).
        if target_area < 0 and target_index >= 0 and int(
                option.get("type", -1)) == _OPT_PLAY:
            target_area = AREA_HAND

    target_id: int | None = None
    if 0 <= owner < len(players):
        target_id = _card_id_at(players[owner], target_area, target_index)
    return target_id, target_area, owner


# --- rich target-state features (see OPTION_DIM_RICH) ------------------------

_ENERGY_COLORLESS = 0
_ENERGY_RAINBOW = 10
_RICH_DMG_NORM = 300.0


def _pokemon_dict_at(player: dict, area: int, index: int) -> dict | None:
    """The board/hand card dict an option points at (None when unresolvable)."""
    if index < 0:
        return None
    spot = {AREA_ACTIVE: player.get("active"), AREA_BENCH: player.get("bench"),
            AREA_HAND: player.get("hand")}.get(area) or []
    return spot[index] if 0 <= index < len(spot) else None


def _afford(cost: list[int], energies: list[int]) -> bool:
    pool: dict[int, int] = {}
    for e in energies:
        pool[e] = pool.get(e, 0) + 1
    colorless = 0
    for c in cost:
        if c == _ENERGY_COLORLESS:
            colorless += 1
        elif pool.get(c, 0) > 0:
            pool[c] -= 1
        elif pool.get(_ENERGY_RAINBOW, 0) > 0:
            pool[_ENERGY_RAINBOW] -= 1
        else:
            return False
    return sum(pool.values()) >= colorless


def _eff_dmg(attacker_type: int, defender: dict | None, dmg: int) -> int:
    if dmg > 0 and defender is not None and defender.get("weak") == attacker_type:
        return dmg * 2
    return dmg


def _card_of(pk: dict | None, feats: CardFeatures) -> dict | None:
    """Engine card stats for a board/hand card dict (None when unknown)."""
    cid = (pk or {}).get("id")
    return feats.cards.get(cid) if isinstance(cid, int) else None


def _best_dmg_now(pk: dict | None, defender: dict | None, feats: CardFeatures) -> int:
    """Best weakness-adjusted damage ``pk`` can deal NOW with attached energy."""
    card = _card_of(pk, feats)
    if pk is None or card is None:
        return 0
    energies = pk.get("energies") or []
    atype = card.get("type", _ENERGY_COLORLESS)
    best = 0
    for aid in card.get("attacks", []):
        info = feats.attacks.get(aid)
        if info is not None and _afford(info["cost"], energies):
            best = max(best, _eff_dmg(atype, defender, info["dmg"]))
    return best


def _rich_feats(
    option: dict,
    current: dict | None,
    your_index: int,
    feats: CardFeatures,
) -> NDArray[np.float64]:
    """The :data:`_OPTION_RICH_WIDTH` target-state features for one option."""
    out = np.zeros(_OPTION_RICH_WIDTH, dtype=np.float64)
    players = (current or {}).get("players") or []
    if len(players) < 2:  # noqa: PLR2004
        return out
    _tid, target_area, owner = _option_target(option, current, your_index)
    if not 0 <= owner < len(players):
        return out
    in_area, in_index = option.get("inPlayArea"), option.get("inPlayIndex")
    if in_area is not None and in_index is not None:
        pk = _pokemon_dict_at(players[owner], int(in_area), int(in_index))
    else:
        pk = _pokemon_dict_at(players[owner], target_area,
                              int(option.get("index", -1)))
    opp = players[1 - your_index]
    opp_active = (opp.get("active") or [None])[0]
    opp_card = _card_of(opp_active, feats)

    card = _card_of(pk, feats)
    energies = list(pk.get("energies") or []) if pk else []
    out[0] = len(energies) / _ENERGY_NORM
    if card is not None:
        costs = [len(feats.attacks[a]["cost"]) for a in card.get("attacks", [])
                 if a in feats.attacks]
        if costs:
            # energies still missing to the cheapest attack (0 = can pay something)
            out[1] = max(min(costs) - len(energies), 0) / 3.0
            # would ONE more (any-colour) energy newly afford some attack?
            ext = [*energies, _ENERGY_RAINBOW]
            out[2] = float(any(
                not _afford(feats.attacks[a]["cost"], energies)
                and _afford(feats.attacks[a]["cost"], ext)
                for a in card.get("attacks", []) if a in feats.attacks))
    best_now = _best_dmg_now(pk, opp_card, feats)
    out[3] = float(best_now > 0)  # can attack now
    # (4,5): this option's own attack when it IS an attack, else best-now
    aid = option.get("attackId")
    if isinstance(aid, int) and aid in feats.attacks and card is not None:
        eff = _eff_dmg(card.get("type", _ENERGY_COLORLESS), opp_card,
                       feats.attacks[aid]["dmg"])
    else:
        eff = best_now
    out[4] = eff / _RICH_DMG_NORM
    opp_hp = opp_active.get("hp") if opp_active else None
    out[5] = float(opp_hp is not None and eff >= opp_hp and eff > 0)  # lethal
    if pk is not None:
        hp, mx = pk.get("hp"), pk.get("maxHp") or 0
        if hp is None and card is not None:  # hand card: printed HP
            hp = mx = card.get("hp", 0)
        out[6] = (hp or 0) / mx if mx else 0.0
        my_card = _card_of(pk, feats)
        threat = _best_dmg_now(opp_active, my_card, feats)
        out[7] = threat / _RICH_DMG_NORM
        out[8] = float(hp is not None and threat >= (hp or 0) and threat > 0)
    if card is not None:
        out[9] = (3 if card.get("mega") else 2 if card.get("ex") else 1) / 3.0
    return out


def encode_option(
    option: dict,
    current: dict | None,
    your_index: int,
    feats: CardFeatures,
    *,
    rich: bool = False,
) -> NDArray[np.float64]:
    """Encode one presented ``Option`` into an :data:`OPTION_DIM` vector
    (:data:`OPTION_DIM_RICH` when ``rich`` -- appends target-state features)."""
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

    parts = [
        type_onehot,
        target_feat,
        np.asarray(flags, dtype=np.float64),
        np.asarray(attack_feats, dtype=np.float64),
        np.asarray(number_feat, dtype=np.float64),
    ]
    if rich:
        parts.append(_rich_feats(option, current, your_index, feats))
    return np.concatenate(parts)


def encode_options(
    options: list[dict],
    current: dict | None,
    your_index: int,
    feats: CardFeatures,
    *,
    rich: bool = False,
) -> NDArray[np.float64]:
    """Stack the option encodings into a ``(len(options), OPTION_DIM[_RICH])``."""
    if not options:
        return np.zeros((0, OPTION_DIM_RICH if rich else OPTION_DIM),
                        dtype=np.float64)
    return np.stack([
        encode_option(opt, current, your_index, feats, rich=rich) for opt in options
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


def deck_context(deck: list[int], feats: CardFeatures) -> NDArray[np.float64]:
    """Fixed deck-context vector: the mean card-feature vector over the 60 cards.

    The play net's observation only carries ``deckCount`` (a scalar), so without
    this the net cannot know at turn 1 whether it is piloting an aggro pile or a
    Stage-2 engine. Computed once per game (the deck is fully known to its owner)
    and fed to the deck-conditioned play LSTM (``RecurrentNetConfig.deck_ctx_dim``).
    Same encoder at training (trajectory_data) and serving (recurrent_agent).
    """
    if not deck:
        return feats.vector(None)
    return np.mean([feats.vector(cid) for cid in deck], axis=0)
