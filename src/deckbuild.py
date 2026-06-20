"""Coherent demonstration-deck builder (PLAN.md Phase 1).

The OSFP CB head warm-starts (P4) from *demonstration* decks: legal, coherent,
better-than-random decks. Rather than scrape external metagame lists (fragile to
map onto this engine's own 1267-card pool), we build mono-type aggro archetypes
directly from the pool: a few strong Basic Pokemon attacker lines + matching
Basic Energy + a generic draw/search trainer engine (mirroring the sample deck's
shape). They are engine-legal by construction (validated against ``src.deck``),
and give a diverse, weakness-structured opponent pool / BC prior.

Pure/native (reads the card CSVs via ``src.cards``); the deck-eval harness under
Docker confirms they actually play (beat random, show matchup structure).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.cards import load_cards, load_moves, parse_cost
from src.deck import DECK_SIZE, build_pool, is_legal

if TYPE_CHECKING:
    from pathlib import Path

    from src.deck import CardPool

_BASIC_POKEMON_STAGE = "Basic Pokémon"
_BASIC_ENERGY_STAGE = "Basic Energy"

COPIES = 4  # standard per-name cap (we run attacker lines at the cap)
TOP_SPECIES = 3  # distinct attacker species per deck
MIN_POKEMON = 8  # add a colorless filler attacker below this
MAX_ATTACK_COST = 3  # only count cheap, castable attacks when ranking attackers

# Colorless Basic ex attacker (Mega Kangaskhan ex): plays off any energy, so it
# is a safe filler when a colour is thin on castable Basic attackers.
COLORLESS_FILLER = 756

# Generic draw/search engine reused from the sample deck (id -> copies). Maximum
# Belt (1158) is an ACE SPEC, hence a single copy.
ENGINE: dict[int, int] = {1205: 4, 1227: 4, 1235: 4, 1145: 4, 1158: 1}

COLOR_NAMES = {
    "R": "fire", "W": "water", "L": "lightning", "P": "psychic",
    "F": "fighting", "G": "grass", "D": "darkness", "M": "metal",
}


def _basic_energy_ids(cards: list[dict]) -> dict[str, int]:
    """Map energy colour code -> Basic Energy card id (e.g. ``"R" -> 2``)."""
    out: dict[str, int] = {}
    for row in cards:
        if row["stage_or_type"] == _BASIC_ENERGY_STAGE:
            code = row["type_code"]
            if isinstance(code, str):
                out[code] = int(row["card_id"])
    return out


def _top_attackers(cards: list[dict], moves: list[dict]) -> dict[str, list[int]]:
    """Per colour, the best Basic Pokemon attackers (mono/colourless, cheap)."""
    basics = {
        int(r["card_id"]) for r in cards if r["stage_or_type"] == _BASIC_POKEMON_STAGE
    }
    best: dict[int, tuple[str, int]] = {}  # card_id -> (colour, best damage)
    for row in moves:
        card_id = int(row["card_id"])
        if card_id not in basics:
            continue
        dmg = row["damage_value"]
        if dmg is None or (isinstance(dmg, float) and math.isnan(dmg)):
            continue
        cost = parse_cost(row["cost"])
        total = sum(cost.values())
        colours = [k for k in cost if k != "C"]
        if total == 0 or total > MAX_ATTACK_COST or len(colours) > 1:
            continue
        colour = colours[0] if colours else "C"
        damage = int(dmg)
        if card_id not in best or damage > best[card_id][1]:
            best[card_id] = (colour, damage)

    by_colour: dict[str, list[tuple[int, int]]] = {}
    for card_id, (colour, damage) in best.items():
        by_colour.setdefault(colour, []).append((card_id, damage))
    ranked: dict[str, list[int]] = {}
    for colour, lst in by_colour.items():
        lst.sort(key=lambda t: t[1], reverse=True)
        ranked[colour] = [card_id for card_id, _ in lst[:TOP_SPECIES]]
    return ranked


def build_mono_aggro(
    attacker_ids: list[int], energy_id: int, pool: CardPool,
) -> list[int]:
    """Assemble one mono-type aggro deck: attackers + engine + Basic Energy."""
    pokemon: list[int] = []
    for card_id in attacker_ids[:TOP_SPECIES]:
        pokemon.extend([card_id] * COPIES)
    if len(pokemon) < MIN_POKEMON and COLORLESS_FILLER in pool:
        pokemon.extend([COLORLESS_FILLER] * COPIES)

    trainers = [card_id for card_id, n in ENGINE.items() for _ in range(n)]
    energy = [energy_id] * (DECK_SIZE - len(pokemon) - len(trainers))
    return pokemon + trainers + energy


def build_demo_decks(
    lang: str = "EN", data_dir: Path | None = None,
) -> dict[str, list[int]]:
    """Build the legal mono-type aggro demonstration set, keyed by deck name."""
    cards = load_cards(lang=lang, data_dir=data_dir).to_dict(orient="records")
    moves = load_moves(lang=lang, data_dir=data_dir).to_dict(orient="records")
    pool = build_pool(lang=lang, data_dir=data_dir)
    energy_ids = _basic_energy_ids(cards)
    attackers = _top_attackers(cards, moves)

    decks: dict[str, list[int]] = {}
    for code, name in COLOR_NAMES.items():
        if code not in attackers or code not in energy_ids:
            continue
        deck = build_mono_aggro(attackers[code], energy_ids[code], pool)
        if is_legal(deck, pool):
            decks[f"{name}_aggro"] = deck
    return decks
