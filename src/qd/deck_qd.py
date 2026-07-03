"""Deck operators for MAP-Elites: random legal deck, mutation, archetype descriptor.

Pure (no engine): every deck is a 60-card list that passes
:func:`~src.deck.legality_errors` by construction -- both the random generator and
the mutator build/repair through :func:`~src.deck.legal_next_ids`, so the QD search
only ever proposes **legal** decks (the engine's only hard rule). Playability beyond
legality is left to fitness, not hand-coded constraints.

The **behaviour descriptor** (v3, Step 5) is the deck's archetype niche --
``(prize-liability, setup-speed, evolution-depth, toolbox-breadth)`` bins, four
genuine construction *trade-offs* of the modern Mega-era meta (grounded in the
2026-07 competitive-deckbuilding research doc):

- **prize liability** (attrition ↔ power): single-prize attackers lose fewer prizes
  per KO and win the prize race; ex (2) / Mega ex (3) hit harder. v3 measures the
  *role-weighted average* prizes-per-KO (attackers weigh 1.0, support Pokemon 0.25).
- **setup speed** (aggro ↔ ramp): the cheapest attack the deck can field.
- **evolution depth** (tempo ↔ ceiling): mean evolution stage over the Pokemon.
  Stage-2 engines (the dominant real-world archetypes) pay setup turns for a
  stronger board; all meta anchors are Basics-only, so this axis opens a region
  the search never explored.
- **toolbox breadth** (linear ↔ answer-rich): distinct attacker species -- a thick
  single line maximises consistency, many one-of attackers buy matchup coverage.

All are derived from the *decklist alone* (no engine). The earlier ``(colour,
energy-count)`` descriptor failed to illuminate the space -- colour duplicated the
soft colour penalty while every strong deck piled into the top energy bin. Colour
stays as the soft fitness *penalty* (see :func:`colour_count`), energy count as a
mutation target (:data:`ENERGY_TARGET`), not niche axes. MAP-Elites keeps the best
deck per niche.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from src.deck import DECK_SIZE, card_kind, legal_next_ids

if TYPE_CHECKING:
    import numpy as np

    from src.deck import CardInfo, CardPool

# Energy-count niche edges (upper-inclusive): aggro/thin .. energy-heavy/setup.
ENERGY_BIN_EDGES = (8, 12, 16, 20)
NO_COLOUR = "C"  # colourless / typeless fallback descriptor

# Prize-liability niche edges over "extra prize points" (ex=+1, Mega=+2 per copy):
# bin 0 == pure single-prize deck, rising to all-Mega. 5 bins.
PRIZE_BIN_EDGES = (0, 4, 8, 12)
# Setup-speed niche edges over the deck's cheapest attack cost: <=1 (aggro) .. >=4
# (ramp). 4 bins; a deck with no attacker at all falls into the slowest bin.
SPEED_BIN_EDGES = (1, 2, 3)
_EX_PRIZES = 2
_MEGA_PRIZES = 3

# ---- Descriptor v3 (Step 5) -- 4 axes grounded in competitive-TCG trade-offs ----
# (docs/deck-search-redesign.md + the 2026-07 deckbuilding research doc). All are
# decklist-only. Role-weighted prize liability replaces raw prize *points* as the
# archive axis: an ex/Mega ATTACKER is real liability, a support ex much less so.
PL_BIN_EDGES = (1.3, 1.6, 1.9, 2.2)  # avg prizes/KO over role-weighted Pokemon; 5 bins
_ATTACKER_WEIGHT = 1.0  # main-attacker role weight in the liability average
_SUPPORT_WEIGHT = 0.25  # non-attacking (engine/support) Pokemon weight
# Mean evolution stage over the deck's Pokemon (Basic=0 .. Stage 2=2): the setup-cost
# axis the meta anchors never explore (they are all-Basic; 37% of the pool evolves).
EVO_BIN_EDGES = (0.05, 0.25, 0.5)  # 4 bins: basics-only / splash / line / evo-core
# Distinct attacker species: linear (Ceruledge-like) vs toolbox (Pidgeot-like). 3 bins.
TOOLBOX_BIN_EDGES = (3, 7)

# Heuristic mutation (Step 3): role-aware operators + weights (see card_role / mutate).
ENERGY_TARGET = (8, 15)  # normal-Standard energy count (docs/deck-search-redesign §0.4)
_ENERGY_BLOCK = 2  # energy cards moved per energy-block adjustment
_TRAINER_REFILL_BIAS = 0.7  # P(refill a freed energy slot with a Trainer)
_OP_NAMES = ("same_role", "package", "energy_block", "evo_line", "random")
_OP_WEIGHTS = (0.35, 0.20, 0.15, 0.20, 0.10)  # free "random" = low explore floor
_EVO_LINE_ADD_PROB = 0.6  # evo_line op: P(add a line) vs remove an existing one
_EVO_LINE_COPIES = (2, 3)  # copies per line member when adding (2-2[-2] or 3-3[-3])


def random_legal_deck(pool: CardPool, rng: np.random.Generator) -> list[int]:
    """Build a uniformly-random **legal** 60-card deck (random legal pick per slot)."""
    deck: list[int] = []
    while len(deck) < DECK_SIZE:
        legal = sorted(legal_next_ids(deck, pool))
        if not legal:
            break
        deck.append(legal[int(rng.integers(len(legal)))])
    return deck


def single_prize_ids(pool: CardPool) -> list[int]:
    """Pokemon that give up a single prize (not ex / not Mega) -- the attrition side.

    Random init almost never produces a *pure* single-prize deck (the pool is dense
    with ex/Mega, so a random draw includes some), leaving the ``prize_bin == 0`` niche
    empty. Seeding from this subset reaches it.
    """
    return [
        cid
        for cid, info in pool.cards.items()
        if info.supertype == "Pokemon" and not info.is_ex and not info.is_mega
    ]


def ramp_ids(pool: CardPool, min_cost: int = 3) -> list[int]:
    """Pokemon whose cheapest attack costs ``>= min_cost`` (or that don't attack).

    Restricting the deck's Pokemon to these makes its *cheapest* attacker expensive, so
    it lands in a high ``speed_bin`` (ramp) -- the niche random init can't reach because
    a random deck almost always includes a 1-energy attacker.
    """
    return [
        cid
        for cid, info in pool.cards.items()
        if info.supertype == "Pokemon"
        and (info.min_attack_cost is None or info.min_attack_cost >= min_cost)
    ]


def random_legal_deck_biased(
    pool: CardPool,
    rng: np.random.Generator,
    allowed_pokemon: list[int],
) -> list[int]:
    """Random legal 60 whose *Pokemon* are drawn only from ``allowed_pokemon``.

    Non-Pokemon (energy / trainers) stay unrestricted so a legal 60 is always
    completable. Used to seed the descriptor's "exclusion" niches (single-prize / ramp)
    that uniform random init can't reach. Falls back to an unbiased deck if the filter
    leaves no Basic Pokemon to satisfy the >=1-Basic rule.
    """
    allowed = set(allowed_pokemon)
    basics = [cid for cid in allowed if pool.cards[cid].is_basic_pokemon]
    if not basics:
        return random_legal_deck(pool, rng)
    deck = [basics[int(rng.integers(len(basics)))]]  # seed a Basic so the corner binds
    while len(deck) < DECK_SIZE:
        legal = [
            cid
            for cid in legal_next_ids(deck, pool)
            if pool.cards[cid].supertype != "Pokemon" or cid in allowed
        ]
        if not legal:
            break
        deck.append(sorted(legal)[int(rng.integers(len(legal)))])
    return deck


def card_role(info: CardInfo) -> tuple[str, str, bool]:
    """Deckbuilding role of a card: ``(supertype, printed stage/type, is-attacker)``.

    Groups cards the way a deckbuilder treats them as interchangeable, from fields
    already on :class:`~src.deck.CardInfo` (no effect-text parsing): ``supertype`` + the
    printed ``stage_or_type`` (Basic/Stage 1/Stage 2 Pokemon; Item/Supporter/Pokemon
    Tool/Stadium Trainer; Basic/Special Energy), split by whether a Pokemon can attack
    (``min_attack_cost is not None`` -- the same "has an attack" fact :func:`setup_cost`
    keys off), so an attacker and a pure support Pokemon of the same stage are *not*
    interchangeable. Drives the heuristic mutation operator.
    """
    is_attacker = info.supertype == "Pokemon" and info.min_attack_cost is not None
    return info.supertype, info.stage_or_type, is_attacker


def _mutate_random_swap(
    deck: list[int],
    pool: CardPool,
    rng: np.random.Generator,
    n_swaps: int = 3,
) -> list[int]:
    """Remove ``n_swaps`` random cards and refill uniformly -- the Step-1 operator.

    Kept verbatim as the ``strategy="random"`` A/B baseline and as the heuristic
    operator's low-probability free-swap primitive. Removes ``n_swaps`` random cards
    (the ≤60 remainder is still a legal prefix) then re-fills to 60 through
    :func:`~src.deck.legal_next_ids`, so the result is always legal.
    """
    keep = list(deck)
    for _ in range(min(n_swaps, len(keep))):
        keep.pop(int(rng.integers(len(keep))))
    while len(keep) < DECK_SIZE:
        legal = sorted(legal_next_ids(keep, pool))
        if not legal:
            break
        keep.append(legal[int(rng.integers(len(legal)))])
    return keep


def _pick_removal_index(
    keep: list[int],
    pool: CardPool,
    rng: np.random.Generator,
    candidates: list[int] | None = None,
) -> int:
    """Index of a card to remove, weighted by how redundant its role is in the deck.

    The "active-gene" proxy: a role held by many copies (bulk energy, spare Items) is
    far likelier to be touched than the deck's sole attacker or a 1-of tech, so mutation
    edits the flexible slots and rarely deletes a load-bearing one. Decklist-only (this
    pure module has no board/engine telemetry). ``candidates`` restricts the choice to a
    subset of positions (redundancy is still counted over the whole deck).
    """
    idxs = list(range(len(keep))) if candidates is None else candidates
    counts = Counter(card_role(pool.cards[c]) for c in keep)
    weights = [counts[card_role(pool.cards[keep[i]])] for i in idxs]
    total = sum(weights)
    probs = [w / total for w in weights]
    return idxs[int(rng.choice(len(idxs), p=probs))]


def _single_random_swap(
    keep: list[int], pool: CardPool, rng: np.random.Generator,
) -> list[int]:
    """Free 1-card swap: the exploration floor kept inside the heuristic operator."""
    keep = list(keep)
    keep.pop(int(rng.integers(len(keep))))
    legal = sorted(legal_next_ids(keep, pool))
    if legal:
        keep.append(legal[int(rng.integers(len(legal)))])
    return keep


def _same_role_swap(
    keep: list[int], pool: CardPool, rng: np.random.Generator,
) -> list[int]:
    """Replace one card with a legal *same-role* card (falls back to any legal)."""
    keep = list(keep)
    removed = keep.pop(_pick_removal_index(keep, pool, rng))
    role = card_role(pool.cards[removed])
    legal = legal_next_ids(keep, pool)
    same = sorted(c for c in legal if card_role(pool.cards[c]) == role)
    cands = same or sorted(legal)
    if cands:
        keep.append(cands[int(rng.integers(len(cands)))])
    return keep


def _package_swap(
    keep: list[int],
    pool: CardPool,
    rng: np.random.Generator,
    swap_prob: float = 0.6,
) -> list[int]:
    """Remove a card's whole playset; with ``swap_prob`` refill it same-role.

    The "package add/remove" edit -- humans cut/add a playset at a time, not one copy.
    Targets a *named* card (a Pokemon / Trainer line, at most a 4-copy playset), never
    the Basic-Energy stack -- energy composition is :func:`_energy_block_adjust`'s job,
    and swapping a 14-card energy stack in one op is neither local nor useful. With
    ``1 - swap_prob`` the freed slots are left for later ops / the final top-up (a
    genuine package *removal*). Never zeroes the deck's attacker role: if this removes
    the last attacker, the swap branch is forced.
    """
    keep = list(keep)
    nbe = [i for i, c in enumerate(keep) if not pool.cards[c].is_basic_energy]
    if not nbe:  # nothing but Basic Energy (can't happen: >=1 Basic Pokemon) -- no-op
        return keep
    cid = keep[_pick_removal_index(keep, pool, rng, nbe)]
    role = card_role(pool.cards[cid])
    remaining = [c for c in keep if c != cid]
    n = len(keep) - len(remaining)
    removed_last_attacker = role[2] and not any(
        card_role(pool.cards[c])[2] for c in remaining
    )
    if not (removed_last_attacker or rng.random() < swap_prob):
        return remaining
    legal = legal_next_ids(remaining, pool)
    same = [c for c in legal if c != cid and card_role(pool.cards[c]) == role]
    if removed_last_attacker and not same:
        same = [c for c in legal if card_role(pool.cards[c])[2]]  # any attacker
    cands = sorted(same) if same else sorted(legal)
    if not cands:
        return remaining
    new_id = cands[int(rng.integers(len(cands)))]
    for _ in range(n):
        if new_id not in legal_next_ids(remaining, pool):
            break
        remaining.append(new_id)
    return remaining


def _drop_random(
    keep: list[int], cand_idx: list[int], k: int, rng: np.random.Generator,
) -> list[int]:
    """Drop ``k`` cards chosen uniformly at random from positions ``cand_idx``."""
    remove: set[int] = set()
    for _ in range(k):
        choices = [i for i in cand_idx if i not in remove]
        if not choices:
            break
        remove.add(choices[int(rng.integers(len(choices)))])
    return [c for i, c in enumerate(keep) if i not in remove]


def _energy_reduce(
    keep: list[int], pool: CardPool, rng: np.random.Generator, excess: int,
) -> list[int]:
    """Cut up to ``_ENERGY_BLOCK`` energy cards, refilling toward Trainers."""
    idxs = [i for i, c in enumerate(keep) if card_kind(pool, c) == "energy"]
    out = _drop_random(keep, idxs, min(_ENERGY_BLOCK, excess), rng)
    for _ in range(len(keep) - len(out)):
        legal = legal_next_ids(out, pool)
        if not legal:
            break
        trainers = sorted(c for c in legal if card_kind(pool, c) == "trainer")
        use_tr = bool(trainers) and rng.random() < _TRAINER_REFILL_BIAS
        cands = trainers if use_tr else sorted(legal)
        out.append(cands[int(rng.integers(len(cands)))])
    return out


def _energy_increase(
    keep: list[int], pool: CardPool, rng: np.random.Generator, deficit: int,
) -> list[int]:
    """Cut up to ``_ENERGY_BLOCK`` non-energy cards, refilling with Basic Energy."""
    idxs = [i for i, c in enumerate(keep) if card_kind(pool, c) != "energy"]
    out = _drop_random(keep, idxs, min(_ENERGY_BLOCK, deficit), rng)
    for _ in range(len(keep) - len(out)):
        legal = legal_next_ids(out, pool)
        if not legal:
            break
        basics = sorted(c for c in legal if pool.cards[c].is_basic_energy)
        cands = basics or sorted(legal)
        out.append(cands[int(rng.integers(len(cands)))])
    return out


def evo_line_ids(pool: CardPool, cid: int) -> list[list[int]]:
    """The full evolution line ending at ``cid``, as id-choices per member.

    Walks ``evolves_from`` names down to the Basic: for a Stage 2 this returns
    ``[[basic ids], [stage-1 ids], [cid]]`` (a name can have several printings, any
    of which completes the line). Returns ``[[cid]]`` for a Basic. Empty inner list
    if a previous-stage name has no card in the pool (line not buildable).
    """
    by_name: dict[str, list[int]] = {}
    for i, info in pool.cards.items():
        if info.supertype == "Pokemon":
            by_name.setdefault(info.name, []).append(i)
    chain: list[list[int]] = [[cid]]
    prev = pool.cards[cid].evolves_from
    while prev:
        ids = by_name.get(prev, [])
        chain.append(ids)
        prev = pool.cards[ids[0]].evolves_from if ids else ""
    chain.reverse()  # Basic first
    return chain


def _evo_line_remove(
    keep: list[int], pool: CardPool, rng: np.random.Generator,
    in_deck_evo: list[int],
) -> list[int] | None:
    """Remove one whole evolution line (every copy of every name in the chain).

    The removal set is closed *upward* too -- cutting a middle stage must not
    orphan the stages above it still in the deck. Returns ``None`` when the cut
    would zero the deck's attacker role (same guard as :func:`_package_swap`).
    """
    target = in_deck_evo[int(rng.integers(len(in_deck_evo)))]
    names = {pool.cards[ids[0]].name
             for ids in evo_line_ids(pool, target) if ids}
    changed = True
    while changed:
        changed = False
        for c in keep:
            info = pool.cards[c]
            if info.name not in names and info.evolves_from in names:
                names.add(info.name)
                changed = True
    remaining = [c for c in keep if pool.cards[c].name not in names]
    if any(card_role(pool.cards[c])[2] for c in remaining):
        return remaining
    return None


def _evo_line_edit(
    keep: list[int], pool: CardPool, rng: np.random.Generator,
) -> list[int]:
    """Add a coherent evolution LINE (Basic + Stage 1 [+ Stage 2] together) or
    remove an existing one whole -- the coordinated multi-name edit none of the
    single-name ops can make, and the operator that makes the descriptor's
    evolution-depth niches reachable with playable lines (an orphan Stage 2 is
    dead weight, which is exactly the junk-seeding trap the redesign doc warns
    about).
    """
    keep = list(keep)
    in_deck_evo = sorted({c for c in keep if card_stage(pool.cards[c]) > 0})
    if in_deck_evo and rng.random() >= _EVO_LINE_ADD_PROB:
        removed = _evo_line_remove(keep, pool, rng, in_deck_evo)
        if removed is not None:
            return removed
    # Add a line: pick a random evolved Pokemon from the pool with a buildable chain.
    evo_pool = [i for i, info in pool.cards.items()
                if card_stage(info) > 0 and _is_attacker(info)]
    if not evo_pool:
        return keep
    top = evo_pool[int(rng.integers(len(evo_pool)))]
    chain = evo_line_ids(pool, top)
    if any(not ids for ids in chain):
        return keep  # previous stage missing from the pool
    n = int(_EVO_LINE_COPIES[int(rng.integers(len(_EVO_LINE_COPIES)))])
    line = [ids[int(rng.integers(len(ids)))] for ids in chain]  # one printing each
    room = n * len(line)
    for _ in range(room):  # free the slots first (role-redundancy weighted)
        if len(keep) == 0:
            break
        keep.pop(_pick_removal_index(keep, pool, rng))
    for member in line:
        for _ in range(n):
            if member in legal_next_ids(keep, pool):
                keep.append(member)
    return keep  # any shortfall is topped up by mutate()'s defensive refill


def _energy_block_adjust(
    keep: list[int], pool: CardPool, rng: np.random.Generator,
) -> list[int]:
    """Nudge energy count toward the normal-Standard ``ENERGY_TARGET`` range.

    Targets the diagnosed anomaly (evolved decks ran ~27-31 energy vs 8-15 normal); the
    over-energy branch refills toward Trainers, easing the trainer-thin anomaly too.
    """
    n_energy = energy_count(keep, pool)
    lo, hi = ENERGY_TARGET
    if n_energy > hi:
        return _energy_reduce(keep, pool, rng, n_energy - hi)
    if n_energy < lo:
        return _energy_increase(keep, pool, rng, lo - n_energy)
    return list(keep)


def mutate(
    deck: list[int],
    pool: CardPool,
    rng: np.random.Generator,
    n_swaps: int = 3,
    *,
    strategy: str = "random",
) -> list[int]:
    """Vary a deck into a legal neighbour. ``strategy`` selects the operator.

    ``"random"`` (default, backward-compatible) removes ``n_swaps`` random cards and
    refills uniformly -- the Step-1 operator. ``"heuristic"`` (Step 3) applies
    ``n_swaps`` role-aware unit-ops (same-role substitution / package add-remove /
    energy-block adjustment, plus a low-probability free swap as an exploration floor),
    reaching coherent engine decks the uniform refill can't. Both stay legal and route
    every card through :func:`~src.deck.legal_next_ids`; the space is unrestricted
    (every deck stays reachable), only the operator is smarter.

    Crossover (swapping role packages between two archive cells) is a deliberate
    follow-on, deferred to Step 4's population-level work per the redesign doc.
    """
    if strategy == "random":
        return _mutate_random_swap(deck, pool, rng, n_swaps)
    if strategy != "heuristic":
        msg = f"unknown mutation strategy: {strategy!r}"
        raise ValueError(msg)
    keep = list(deck)
    for _ in range(n_swaps):
        name = _OP_NAMES[int(rng.choice(len(_OP_NAMES), p=list(_OP_WEIGHTS)))]
        if name == "same_role":
            keep = _same_role_swap(keep, pool, rng)
        elif name == "package":
            keep = _package_swap(keep, pool, rng)
        elif name == "energy_block":
            keep = _energy_block_adjust(keep, pool, rng)
        elif name == "evo_line":
            keep = _evo_line_edit(keep, pool, rng)
        else:
            keep = _single_random_swap(keep, pool, rng)
    while len(keep) < DECK_SIZE:  # defensive top-up (ops may leave freed slots)
        legal = sorted(legal_next_ids(keep, pool))
        if not legal:
            break
        keep.append(legal[int(rng.integers(len(legal)))])
    return keep


def primary_colour(deck: list[int], pool: CardPool) -> str:
    """The deck's dominant Pokemon energy colour (its archetype colour)."""
    colours: Counter[str] = Counter()
    for cid in deck:
        info = pool.cards.get(cid)
        if info is not None and info.supertype == "Pokemon" and info.card_type:
            colours[info.card_type] += 1
    colours.pop(NO_COLOUR, None)  # colourless is not an archetype identity
    return colours.most_common(1)[0][0] if colours else NO_COLOUR


def colour_count(deck: list[int], pool: CardPool) -> int:
    """Distinct *coloured* Pokemon types in the deck (its rainbow-ness).

    Colourless (``NO_COLOUR``) is excluded: colourless attackers run on any energy,
    so they do not demand an extra colour. Used by the QD soft colour penalty to bias
    the archive toward fewer-colour (more consistent) decks without forbidding any.
    """
    colours: set[str] = set()
    for cid in deck:
        info = pool.cards.get(cid)
        if info is not None and info.supertype == "Pokemon" and info.card_type:
            colours.add(info.card_type)
    colours.discard(NO_COLOUR)
    return len(colours)


def energy_count(deck: list[int], pool: CardPool) -> int:
    """Number of Energy cards in the deck (Basic + Special)."""
    return sum(card_kind(pool, cid) == "energy" for cid in deck)


def energy_bin(n: int) -> int:
    """Bin an energy count into a niche index ``0..len(ENERGY_BIN_EDGES)``."""
    return sum(n > edge for edge in ENERGY_BIN_EDGES)


def prize_value(info: object | None) -> int:
    """Prizes a single Pokemon gives up when KO'd: Mega ex 3, ex 2, other Pokemon 1.

    Non-Pokemon (energy / trainers) give up nothing toward the liability.
    """
    if info is None or getattr(info, "supertype", "") != "Pokemon":
        return 0
    if getattr(info, "is_mega", False):
        return _MEGA_PRIZES
    if getattr(info, "is_ex", False):
        return _EX_PRIZES
    return 1


def prize_points(deck: list[int], pool: CardPool) -> int:
    """Total *extra* prizes the deck can give up (ex = +1, Mega = +2 per copy).

    0 for a pure single-prize deck; grows with multi-prize density and severity --
    the attrition ↔ power axis of the prize race.
    """
    return sum(max(prize_value(pool.cards.get(cid)) - 1, 0) for cid in deck)


def prize_bin(points: int) -> int:
    """Bin prize points into ``0..len(PRIZE_BIN_EDGES)`` (0 == pure single-prize)."""
    return sum(points > edge for edge in PRIZE_BIN_EDGES)


def setup_cost(deck: list[int], pool: CardPool) -> int | None:
    """Cheapest attack the deck can field (min attack cost over its Pokemon).

    ``None`` if no card in the deck has an attack (treated as the slowest niche).
    """
    costs = [
        info.min_attack_cost
        for cid in deck
        if (info := pool.cards.get(cid)) is not None
        and info.min_attack_cost is not None
    ]
    return min(costs) if costs else None


def speed_bin(cost: int | None) -> int:
    """Bin the cheapest attack cost into ``0..len(SPEED_BIN_EDGES)`` (0 = aggro)."""
    if cost is None:
        return len(SPEED_BIN_EDGES)  # no attacker -> slowest niche
    return sum(cost > edge for edge in SPEED_BIN_EDGES)


def card_stage(info: CardInfo) -> int:
    """Evolution stage of a Pokemon card: Basic 0, Stage 1 -> 1, Stage 2 -> 2."""
    if info.stage_or_type.startswith("Stage 1"):
        return 1
    if info.stage_or_type.startswith("Stage 2"):
        return 2
    return 0


def _is_attacker(info: CardInfo) -> bool:
    return info.supertype == "Pokemon" and info.min_attack_cost is not None


def prize_liability(deck: list[int], pool: CardPool) -> float:
    """Role-weighted average prizes-per-KO over the deck's Pokemon (1.0 .. 3.0).

    ``sum(copies * prizes * role) / sum(copies * role)`` with attackers at weight
    1.0 and non-attacking (engine/support) Pokemon at 0.25 -- a support ex is far
    less exposed than an ex attacker that must sit in the Active. 1.0 = pure
    single-prize; a Mega-attacker core approaches 3.0. Decks with no Pokemon
    (impossible: legality needs a Basic) default to 1.0.
    """
    num = den = 0.0
    for cid in deck:
        info = pool.cards.get(cid)
        if info is None or info.supertype != "Pokemon":
            continue
        w = _ATTACKER_WEIGHT if _is_attacker(info) else _SUPPORT_WEIGHT
        num += w * prize_value(info)
        den += w
    return num / den if den else 1.0


def pl_bin(liability: float) -> int:
    """Bin role-weighted prize liability into ``0..len(PL_BIN_EDGES)``."""
    return sum(liability > edge for edge in PL_BIN_EDGES)


def evolution_depth(deck: list[int], pool: CardPool) -> float:
    """Mean evolution stage over the deck's Pokemon (0.0 all-Basic .. 2.0)."""
    stages = [
        card_stage(info)
        for cid in deck
        if (info := pool.cards.get(cid)) is not None and info.supertype == "Pokemon"
    ]
    return sum(stages) / len(stages) if stages else 0.0


def evo_bin(depth: float) -> int:
    """Bin evolution depth into ``0..len(EVO_BIN_EDGES)`` (0 = basics-only)."""
    return sum(depth > edge for edge in EVO_BIN_EDGES)


def toolbox_breadth(deck: list[int], pool: CardPool) -> int:
    """Distinct attacker species in the deck (linear core vs toolbox answers)."""
    return len({
        info.name
        for cid in deck
        if (info := pool.cards.get(cid)) is not None and _is_attacker(info)
    })


def toolbox_bin(n: int) -> int:
    """Bin attacker-species count into ``0..len(TOOLBOX_BIN_EDGES)``."""
    return sum(n > edge for edge in TOOLBOX_BIN_EDGES)


def behaviour_descriptor(
    deck: list[int], pool: CardPool,
) -> tuple[int, int, int, int]:
    """Archetype niche (descriptor v3): ``(prize-liability, setup-speed,
    evolution-depth, toolbox-breadth)`` bins -- max 5*4*4*3 = 240 cells.

    v2 was ``(prize-points bin, speed bin)`` (20 cells); v3 keeps the two validated
    trade-offs (liability now role-weighted) and adds the two axes the 2026-07
    research ranked highest among the decklist-computable ones: evolution depth
    (the setup-cost axis; every meta anchor is all-Basic, so the region real
    Stage-2 archetypes live in was structurally unexplored) and toolbox breadth
    (linear vs answer-rich construction).
    """
    return (
        pl_bin(prize_liability(deck, pool)),
        speed_bin(setup_cost(deck, pool)),
        evo_bin(evolution_depth(deck, pool)),
        toolbox_bin(toolbox_breadth(deck, pool)),
    )


def deck_stats(deck: list[int], pool: CardPool) -> dict[str, int | float | None]:
    """Coarse composition + niche features (for logging / inspection)."""
    kinds = Counter(card_kind(pool, cid) for cid in deck)
    return {
        "energy": kinds.get("energy", 0),
        "pokemon": kinds.get("pokemon", 0),
        "trainer": kinds.get("trainer", 0),
        "distinct": len(set(deck)),
        "prize_points": prize_points(deck, pool),
        "prize_liability": round(prize_liability(deck, pool), 2),
        "evo_depth": round(evolution_depth(deck, pool), 2),
        "toolbox": toolbox_breadth(deck, pool),
        "min_attack_cost": setup_cost(deck, pool),
    }
