"""Observation-consistent determinization of the hidden state (PIMC sampling).

A determinization is a concrete guess of every hidden card, fed to the engine's
``search_begin`` so it can simulate forward. We keep it **consistent with what is
observed**, which splits cleanly by side:

* **Our side** is exact up to ordering -- we know our 60-card deck list, so the cards
  not currently visible (hand / board / discard / revealed prizes) are exactly the
  multiset in {deck, face-down prizes}; we only have to guess the *split* and order.
* **The opponent's side** is genuinely hidden: the cards they have *revealed* (active,
  bench, attached energy/tools, discard, face-up prizes) are already in the engine's
  state, so we never re-guess them; we only fill their **hidden** zones (deck, hand,
  face-down prizes, a face-down active) from a caller-supplied prior. That prior is the
  one real approximation -- everything else is pinned to the observation.

Pure ``dict`` in / ``list[int]`` out (engine-free): the obs is the Kaggle observation's
``current`` state dict, card ids are ``CardData`` ids matching deck lists.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# A card dict carries its CardData id under "id"; a Pokemon dict additionally carries
# attached cards (energy / tools / pre-evolutions), all of which are visible.
_ATTACH_KEYS = ("energyCards", "tools", "preEvolution")


def _pokemon_card_ids(pkmn: dict | None) -> list[int]:
    """Every visible CardData id a board Pokemon accounts for (itself + attached)."""
    if not pkmn:
        return []
    ids = [int(pkmn["id"])]
    for key in _ATTACH_KEYS:
        ids += [int(c["id"]) for c in (pkmn.get(key) or [])]
    return ids


def seen_card_ids(player: dict, *, include_hand: bool) -> list[int]:
    """All CardData ids whose location is **known** for a player.

    Active + bench (with everything attached), the discard pile, and any face-up
    prize. ``include_hand`` adds the hand (known for us, ``None``/hidden for the
    opponent). Face-down cards (``None``) contribute nothing.
    """
    ids: list[int] = []
    for slot in player.get("active") or []:
        ids += _pokemon_card_ids(slot)
    for slot in player.get("bench") or []:
        ids += _pokemon_card_ids(slot)
    ids += [int(c["id"]) for c in (player.get("discard") or [])]
    ids += [int(c["id"]) for c in (player.get("prize") or []) if c is not None]
    if include_hand:
        ids += [int(c["id"]) for c in (player.get("hand") or [])]
    return ids


def _multiset_remove(base: list[int], remove: list[int]) -> list[int]:
    """``base`` minus ``remove`` as multisets (each removal drops one occurrence)."""
    out = list(base)
    for cid in remove:
        # observed a card our deck list didn't contain -> ignore
        with contextlib.suppress(ValueError):
            out.remove(cid)
    return out


@dataclass
class Determinization:
    """The six predicted hidden-card lists ``search_begin`` consumes."""

    your_deck: list[int]
    your_prize: list[int]
    opp_deck: list[int]
    opp_prize: list[int]
    opp_hand: list[int]
    opp_active: list[int]


def _none_count(cards: list | None) -> int:
    return sum(1 for c in (cards or []) if c is None)


def _fill_prize(prize: list | None, hidden: list[int]) -> list[int]:
    """A full prize prediction: keep face-up ids, draw face-down ones from ``hidden``.

    ``hidden`` is consumed (pop) for each face-down slot; order within the unknown
    cards is arbitrary (we cannot observe it).
    """
    return [int(c["id"]) if c is not None else int(hidden.pop()) for c in prize or []]


def sample_determinization(  # noqa: PLR0913 - the hidden state legitimately has parts
    current: dict,
    your_index: int,
    your_deck_list: list[int],
    opp_prior: list[int],
    rng: np.random.Generator,
    *,
    opp_basics: list[int] | None = None,
) -> Determinization:
    """Sample one observation-consistent determinization.

    ``current`` is ``obs["current"]`` (the ``State`` dict). ``your_deck_list`` is our
    own 60 ids. ``opp_prior`` is a flat pool of candidate ids to draw the opponent's
    hidden cards from (e.g. concatenated meta decks). ``opp_basics`` (Basic-Pokemon
    ids) guarantees the opponent's predicted deck has the >=1 Basic the engine needs.
    """
    me = current["players"][your_index]
    opp = current["players"][1 - your_index]

    # --- our side: exact multiset, only the split/order is unknown ---------------
    our_unseen = _multiset_remove(
        your_deck_list, seen_card_ids(me, include_hand=True),
    )
    rng.shuffle(our_unseen)
    your_prize = _fill_prize(me.get("prize"), our_unseen)  # consumes face-down slots
    your_deck = our_unseen  # whatever remains is the deck (deckCount cards)

    # --- opponent: keep revealed cards, fill hidden zones from the prior ----------
    opp_face_down_prizes = _none_count(opp.get("prize"))
    opp_active = opp.get("active") or []
    need_active = len(opp_active) > 0 and opp_active[0] is None
    n_hidden = (
        int(opp["deckCount"]) + int(opp["handCount"])
        + opp_face_down_prizes + (1 if need_active else 0)
    )
    pool = list(opp_prior) if opp_prior else list(your_deck_list)
    draw = [int(pool[int(i)]) for i in rng.integers(0, len(pool), size=n_hidden)]

    opp_active_pred = [int(draw.pop())] if need_active else []
    opp_hand = [int(draw.pop()) for _ in range(int(opp["handCount"]))]
    # face-down prize slots first, then the deck takes the rest.
    opp_prize = _fill_prize(opp.get("prize"), draw)
    opp_deck = draw  # remaining == deckCount
    if opp_basics and not any(c in set(opp_basics) for c in opp_deck) and opp_deck:
        opp_deck[int(rng.integers(0, len(opp_deck)))] = int(
            opp_basics[int(rng.integers(0, len(opp_basics)))],
        )

    return Determinization(
        your_deck=your_deck, your_prize=your_prize,
        opp_deck=opp_deck, opp_prize=opp_prize,
        opp_hand=opp_hand, opp_active=opp_active_pred,
    )
