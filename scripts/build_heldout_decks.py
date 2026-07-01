"""Generate the held-out opponent deck pool -> ``decklists/heldout/`` (native).

Step 0 of the deck-search redesign (``docs/deck-search-redesign.md`` §1.4): a pool of
~30 decks that the QD search, its fitness gauntlet, and the NashConv population *never*
touch -- the generalization yardstick (metric ③) and the NashConv exploiter bank. They
live in the ``heldout/`` subdirectory so the root ``decklists/*.csv`` glob (the QD
gauntlet, ``deck_strength.py``/``nashconv_eval.py``) never picks them up.

Built like ``src.deckbuild`` (a focused Basic-ex attacker line + a generic draw/search
shell + matching Basic Energy), but authored by *name* against the actual SV-era pool
(many real-meta names are absent here, so each deck is real-archetype-*inspired* yet
grounded in cards that exist). A spread of tiers: ~18 "strong" (top Basic-ex) + ~12
"medium" (weaker / off-type / two-type splits) so the held-out set spans power levels.

Names resolve through a normalisation pass (case / curly-vs-straight apostrophes /
accents folded) so authoring need not match each card's exact punctuation. Legality is
validated against ``src.deck`` before writing; the actual id-list write reuses
``src.decklists.save_deck_csv``.

  uv run python scripts/build_heldout_decks.py
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cards import load_cards  # noqa: E402
from src.deck import DECK_SIZE, build_pool, legality_errors  # noqa: E402
from src.decklists import save_deck_csv  # noqa: E402

OUT_DIR = ROOT / "decklists" / "heldout"

# Generic draw/search shells (17 trainers each). "proven" mirrors the demo-deck
# ENGINE (the shape that gives greedy|metal ~0.66); "std" is a more conventional
# ball+supporter line for variety. Both are ACE-SPEC-safe (<=1 ACE SPEC).
SHELL_PROVEN: list[tuple[int, str]] = [
    (4, "Cyrano"), (4, "Lillie's Determination"), (4, "Waitress"),
    (4, "Mega Signal"), (1, "Maximum Belt"),
]
SHELL_STD: list[tuple[int, str]] = [
    (4, "Ultra Ball"), (3, "Buddy-Buddy Poffin"), (2, "Boss's Orders"),
    (2, "Judge"), (2, "Carmine"), (2, "Pokegear 3.0"),
    (1, "Switch"), (1, "Night Stretcher"),
]

# Short aliases -> exact-ish card names (resolved through _norm, so punctuation-loose).
ATK = {
    "pika": "Pikachu ex", "zekrom": "Zekrom ex", "thorns": "Iron Thorns ex",
    "gouging": "Gouging Fire ex", "volcanion": "Volcanion ex",
    "hearthflame": "Hearthflame Mask Ogerpon ex", "mawile": "Mega Mawile ex",
    "zacian": "Zacian ex", "hopzacian": "Hop's Zacian ex", "genesect": "Genesect ex",
    "koraidon": "Koraidon ex", "regirock": "Regirock ex",
    "cornerstone": "Cornerstone Mask Ogerpon ex", "zygarde": "Mega Zygarde ex",
    "yveltal": "Yveltal ex", "okidogi": "Okidogi ex", "munkidori": "Munkidori ex",
    "absol": "Mega Absol ex", "latias": "Latias ex",
    "mewtwo": "Team Rocket's Mewtwo ex", "diancie": "Mega Diancie ex",
    "blackkyurem": "Black Kyurem ex", "kyurem": "Kyurem ex",
    "regice": "Regice ex", "dondozo": "Dondozo ex", "heracross": "Mega Heracross ex",
    "ironleaves": "Iron Leaves ex", "durant": "Durant ex",
    "teal": "Teal Mask Ogerpon ex",
    "kanga": "Mega Kangaskhan ex", "terapagos": "Terapagos ex",
    "ursaluna": "Bloodmoon Ursaluna ex", "snorlax": "Snorlax",
}

# Each deck: name -> (energy split [(code, copies)] or single code, shell, 3 attacker
# aliases). Each attacker runs at 4 copies (12 Pokemon); energy fills the rest to 60.
P, S = "proven", "std"
DECKS: dict[str, tuple] = {
    # ---- STRONG: top Basic-ex, mono-energy, proven shell -------------------
    "ho_lightning_pikachu":   ("L", P, ["pika", "zekrom", "thorns"]),
    "ho_fire_gouging":        ("R", P, ["gouging", "volcanion", "hearthflame"]),
    "ho_metal_mawile":        ("M", P, ["mawile", "zacian", "genesect"]),
    "ho_metal_zacian":        ("M", P, ["hopzacian", "zacian", "genesect"]),
    "ho_fighting_koraidon":   ("F", P, ["koraidon", "regirock", "cornerstone"]),
    "ho_fighting_zygarde":    ("F", P, ["zygarde", "koraidon", "regirock"]),
    "ho_dark_yveltal":        ("D", P, ["yveltal", "okidogi", "munkidori"]),
    "ho_dark_absol":          ("D", P, ["absol", "yveltal", "okidogi"]),
    "ho_psychic_latias":      ("P", P, ["latias", "mewtwo", "diancie"]),
    "ho_psychic_mewtwo":      ("P", P, ["mewtwo", "latias", "diancie"]),
    "ho_water_kyurem":        ("W", P, ["blackkyurem", "kyurem", "regice"]),
    "ho_grass_heracross":     ("G", P, ["heracross", "ironleaves", "durant"]),
    "ho_grass_ironleaves":    ("G", P, ["ironleaves", "heracross", "teal"]),
    "ho_colorless_kanga":     ("F", P, ["kanga", "terapagos", "snorlax"]),
    "ho_colorless_terapagos": ("F", P, ["terapagos", "kanga", "ursaluna"]),
    "ho_lightning_zekrom":    ("L", P, ["zekrom", "pika", "thorns"]),
    "ho_fire_volcanion":      ("R", P, ["volcanion", "gouging", "hearthflame"]),
    "ho_water_blackkyurem":   ("W", P, ["blackkyurem", "regice", "dondozo"]),
    # ---- MEDIUM: weaker attackers / std shell ------------------------------
    "md_water_regice":        ("W", S, ["regice", "dondozo", "kyurem"]),
    "md_water_dondozo":       ("W", S, ["dondozo", "regice", "snorlax"]),
    "md_grass_durant":        ("G", S, ["durant", "teal", "kanga"]),
    "md_fighting_regirock":   ("F", S, ["regirock", "cornerstone", "snorlax"]),
    "md_metal_genesect":      ("M", S, ["genesect", "zacian", "kanga"]),
    "md_psychic_diancie":     ("P", S, ["diancie", "latias", "snorlax"]),
    "md_dark_okidogi":        ("D", S, ["okidogi", "munkidori", "snorlax"]),
    "md_fire_ogerpon":        ("R", S, ["hearthflame", "volcanion", "snorlax"]),
    "md_lightning_thorns":    ("L", S, ["thorns", "zekrom", "snorlax"]),
    "md_stall_snorlax":       ("F", S, ["snorlax", "ursaluna", "kanga"]),
    # ---- MEDIUM: two-type splits (harder to cast on purpose) ---------------
    "md2_dark_psychic": ([("D", 16), ("P", 15)], S, ["munkidori", "latias", "snorlax"]),
    "md2_fight_metal":  ([("F", 16), ("M", 15)], S, ["koraidon", "zacian", "genesect"]),
}


def _norm(name: str) -> str:
    """Fold case, accents and curly apostrophes so authoring is punctuation-loose."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("\u2019", "'").casefold().strip()  # curly -> straight apostrophe


def _resolvers() -> tuple[dict[str, int], dict[str, int]]:
    """(normalised name -> lowest card id, energy code -> Basic Energy id)."""
    cards = load_cards().to_dict(orient="records")
    name_ids: dict[str, int] = {}
    energy_ids: dict[str, int] = {}
    for row in cards:
        cid = int(row["card_id"])
        key = _norm(str(row["name"]))
        if key not in name_ids or cid < name_ids[key]:
            name_ids[key] = cid
        if row["stage_or_type"] == "Basic Energy" and isinstance(row["type_code"], str):
            energy_ids[row["type_code"]] = cid
    return name_ids, energy_ids


def _resolve(name: str, name_ids: dict[str, int]) -> int:
    cid = name_ids.get(_norm(name))
    if cid is None:
        msg = f"unknown card name: {name!r}"
        raise KeyError(msg)
    return cid


ATTACKER_COPIES = 4  # each of the 3 species at the 4-copy cap (12 Pokemon)


def _build_deck(spec: tuple, name_ids: dict[str, int],
                energy_ids: dict[str, int]) -> list[int]:
    energy_spec, shell_kind, attackers = spec
    shell = SHELL_PROVEN if shell_kind == "proven" else SHELL_STD
    ids: list[int] = []
    for alias in attackers:
        ids += [_resolve(ATK[alias], name_ids)] * ATTACKER_COPIES
    for copies, card in shell:
        ids += [_resolve(card, name_ids)] * copies
    n_energy = DECK_SIZE - len(ids)
    if isinstance(energy_spec, str):
        ids += [energy_ids[energy_spec]] * n_energy
    else:  # explicit split [(code, copies), ...]; pad the remainder onto the first
        total = sum(c for _, c in energy_spec)
        for i, (code, copies) in enumerate(energy_spec):
            extra = (n_energy - total) if i == 0 else 0
            ids += [energy_ids[code]] * (copies + extra)
    return ids


def main() -> None:
    pool = build_pool()
    name_ids, energy_ids = _resolvers()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"writing {len(DECKS)} held-out decks to {OUT_DIR}/")
    for name, spec in sorted(DECKS.items()):
        deck = _build_deck(spec, name_ids, energy_ids)
        errors = legality_errors(deck, pool)
        if errors:
            raise SystemExit(f"{name} is illegal: {errors}")
        save_deck_csv(deck, OUT_DIR / f"{name}.csv")
        print(f"  {name}.csv  (60 cards, legal)")


if __name__ == "__main__":
    main()
