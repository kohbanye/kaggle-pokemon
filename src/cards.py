"""Loader and parsing helpers for the Pokemon TCG AI Battle card data.

The competition ships two human-readable exports, ``EN_Card_Data.csv`` and
``JP_Card_Data.csv``. Each *row* describes a single move/ability, so a card with
N attacks spans N rows that share the same ``Card ID``. This module turns those
raw rows into tidy frames:

- :func:`load_moves`  -> one row per move/ability (cleaned dtypes)
- :func:`load_cards`  -> one row per unique card (deck-building view)

Energy notation: ``{G}`` etc. are colored energies, ``●`` is a colorless
(any-type) requirement. ``parse_cost("{D}●●")`` -> {"D": 1, "C": 2}.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Energy symbol -> short code. ``●`` is a colorless requirement on attack costs.
ENERGY_SYMBOLS = {
    "G": "Grass",
    "R": "Fire",
    "W": "Water",
    "L": "Lightning",
    "P": "Psychic",
    "F": "Fighting",
    "D": "Darkness",
    "M": "Metal",
    "C": "Colorless",
    "N": "Dragon",  # {N} shows up for Dragon-type in some exports
    "A": "Special",
}

_RAW_TO_SNAKE = {
    "Card ID": "card_id",
    "Card Name": "name",
    "Expansion": "expansion",
    "Collection No.": "collection_no",
    "Stage (Pokémon)/Type (Energy and Trainer)": "stage_or_type",
    "Rule": "rule",
    "Category": "category",
    "Previous stage": "previous_stage",
    "HP": "hp",
    "Type": "type",
    "Weakness": "weakness",
    "Resistance (Type)": "resistance",
    "Retreat": "retreat",
    "Move Name": "move_name",
    "Cost": "cost",
    "Damage": "damage",
    "Effect Explanation": "effect",
}

# Coarse supertype derived from the stage_or_type column.
_POKEMON_STAGES = {"Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon"}
_TRAINER_STAGES = {"Item", "Supporter", "Pokémon Tool", "Stadium"}
_ENERGY_STAGES = {"Basic Energy", "Special Energy"}


def _symbol(value: object) -> str | None:
    """Extract the single energy code from a ``{X}`` cell (or None)."""
    if not isinstance(value, str):
        return None
    m = re.search(r"\{([A-Z])\}", value)
    return m.group(1) if m else None


def parse_cost(cost: object) -> dict[str, int]:
    """Parse an attack cost string into ``{energy_code: count}``.

    ``"{P}{D}{M}●"`` -> ``{"P": 1, "D": 1, "M": 1, "C": 1}``.
    ``●`` (colorless) is counted under the ``"C"`` key.
    """
    counts: dict[str, int] = {}
    if not isinstance(cost, str):
        return counts
    for code in re.findall(r"\{([A-Z])\}", cost):
        counts[code] = counts.get(code, 0) + 1
    colorless = cost.count("●")
    if colorless:
        counts["C"] = counts.get("C", 0) + colorless
    return counts


def cost_total(cost: object) -> int:
    """Total energy count required for an attack cost."""
    return sum(parse_cost(cost).values())


def parse_damage(damage: object) -> int | None:
    """Base numeric damage, ignoring modifiers like ``×``, ``+``, ``-``.

    ``"30×"`` -> 30, ``"120"`` -> 120, ``NaN`` -> None.
    """
    if isinstance(damage, (int, float)) and pd.notna(damage):
        return int(damage)
    if not isinstance(damage, str):
        return None
    m = re.search(r"\d+", damage)
    return int(m.group()) if m else None


def supertype(stage_or_type: object) -> str:
    """Map the stage/type cell to Pokemon / Trainer / Energy / Unknown."""
    if stage_or_type in _POKEMON_STAGES:
        return "Pokemon"
    if stage_or_type in _TRAINER_STAGES:
        return "Trainer"
    if stage_or_type in _ENERGY_STAGES:
        return "Energy"
    return "Unknown"


def load_moves(lang: str = "EN", data_dir: Path | None = None) -> pd.DataFrame:
    """Load the raw per-move table with cleaned column names and dtypes."""
    data_dir = data_dir or DATA_DIR
    path = data_dir / f"{lang}_Card_Data.csv"
    df = pd.read_csv(path).rename(columns=_RAW_TO_SNAKE)

    # "n/a" sentinels appear in energy/trainer rows; treat as missing.
    df = df.replace("n/a", pd.NA)

    df["hp"] = pd.to_numeric(df["hp"], errors="coerce")
    df["retreat"] = pd.to_numeric(df["retreat"], errors="coerce")
    df["supertype"] = df["stage_or_type"].map(supertype)
    df["type_code"] = df["type"].map(_symbol)
    df["weakness_code"] = df["weakness"].map(_symbol)
    df["cost_total"] = df["cost"].map(cost_total)
    df["damage_value"] = df["damage"].map(parse_damage)
    df["is_attack"] = df["move_name"].notna() & df["cost"].notna()
    return df


def load_cards(lang: str = "EN", data_dir: Path | None = None) -> pd.DataFrame:
    """Collapse moves into one row per unique card (deck-building view).

    Move-level fields are aggregated: ``move_names`` is a list, ``max_damage``
    and ``max_cost`` summarize the card's attacks.
    """
    moves = load_moves(lang=lang, data_dir=data_dir)

    first = (
        moves.sort_values("card_id")
        .groupby("card_id", as_index=False)
        .first()[
            [
                "card_id",
                "name",
                "expansion",
                "supertype",
                "stage_or_type",
                "rule",
                "category",
                "previous_stage",
                "hp",
                "type_code",
                "weakness_code",
                "retreat",
            ]
        ]
    )

    agg = moves.groupby("card_id").agg(
        n_moves=("move_name", lambda s: s.notna().sum()),
        move_names=("move_name", lambda s: [m for m in s if pd.notna(m)]),
        max_damage=("damage_value", "max"),
        max_cost=("cost_total", "max"),
    )

    # Cheapest *attack* the card can field (energy cost) -- a deck's setup speed
    # descriptor: aggro decks have a 1-energy attacker, ramp decks need 3-4. NaN for
    # cards with no attack (ability-only / evolution fodder / energy / trainers).
    attacks = moves[moves["is_attack"]]
    atk_agg = attacks.groupby("card_id").agg(min_attack_cost=("cost_total", "min"))

    cards = first.merge(agg, on="card_id", how="left").merge(
        atk_agg, on="card_id", how="left",
    )

    rule = cards["rule"].fillna("")
    cards["is_ex"] = rule.str.contains("ex", case=False)
    cards["is_mega"] = rule.str.contains("Mega", case=False)
    cards["is_ace_spec"] = rule.str.contains("ACE SPEC", case=False)
    return cards
