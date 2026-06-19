"""Generate notebooks/01_card_data_eda.ipynb from source cells.

Keeping the notebook content in a plain .py builder makes it easy to review in
diffs and regenerate. Run: ``uv run python scripts/build_eda_notebook.py``.
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "01_card_data_eda.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(
    """
# Pokémon TCG AI Battle — Card Data EDA

Competition: [pokemon-tcg-ai-battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
(Simulation Track). We submit an **agent** (`agent(obs) -> list[int]`) plus a
60-card `deck.csv`; matches are scored by **Elo** on a ladder. Community wisdom:
*deck choice dominates agent quality*, so understanding the card pool is step one.

This notebook explores the ~1,250-card Standard pool shipped as
`data/EN_Card_Data.csv` (one row per move/ability). Parsing lives in
`src/cards.py`.
""",
)

code(
    """
import sys
from pathlib import Path

# Make `import src...` work whether the notebook runs from repo root or notebooks/.
ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.cards import load_moves, load_cards, parse_cost, ENERGY_SYMBOLS

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
""",
)

code(
    """
moves = load_moves("EN")   # one row per move/ability
cards = load_cards("EN")   # one row per unique card
print(f"move rows: {len(moves):,}   unique cards: {len(cards):,}")
cards.head()
""",
)

md("## 1. Card pool composition")

code(
    """
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
cards["supertype"].value_counts().plot.bar(ax=ax[0], color="#4c72b0")
ax[0].set_title("Cards by supertype")
ax[0].set_ylabel("count")
cards["stage_or_type"].value_counts().plot.barh(ax=ax[1], color="#55a868")
ax[1].set_title("Cards by stage / type")
ax[1].invert_yaxis()
plt.tight_layout()
plt.show()

cards["supertype"].value_counts()
""",
)

code(
    """
# Expansions present in the pool (set rotation matters for deck building).
ax = cards["expansion"].value_counts().plot.bar(figsize=(13, 4), color="#c44e52")
ax.set_title("Cards per expansion")
ax.set_ylabel("count")
plt.tight_layout()
plt.show()
""",
)

md(
    """
## 2. Pokémon: HP, type, stage, retreat

These drive the "engine" of a deck: how tanky attackers are, what energy colors
you must run, and how easy Pokémon are to retreat.
""",
)

code(
    """
pk = cards[cards["supertype"] == "Pokemon"].copy()

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(data=pk, x="hp", bins=30, ax=ax[0], color="#4c72b0")
ax[0].set_title("Pokémon HP distribution")

order = ["Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon"]
sns.boxplot(data=pk, x="stage_or_type", y="hp", order=order, ax=ax[1])
ax[1].set_title("HP by stage")
ax[1].set_xlabel("")
plt.tight_layout()
plt.show()

pk["hp"].describe()
""",
)

code(
    """
# Energy-color identity of the Pokémon pool + most common weaknesses.
color_name = lambda s: s.map(lambda c: ENERGY_SYMBOLS.get(c, c))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
color_name(pk["type_code"]).value_counts().plot.bar(ax=ax[0], color="#8172b3")
ax[0].set_title("Pokémon by type (energy color)")
color_name(pk["weakness_code"]).value_counts().plot.bar(ax=ax[1], color="#ccb974")
ax[1].set_title("Pokémon by weakness")
plt.tight_layout()
plt.show()
""",
)

code(
    """
# Retreat cost: cheap-retreat attackers and good "pivot" Pokémon are valuable.
ax = pk["retreat"].value_counts().sort_index().plot.bar(
    figsize=(8, 4), color="#64b5cd"
)
ax.set_title("Retreat cost distribution (Pokémon)")
ax.set_xlabel("retreat energy")
plt.tight_layout()
plt.show()
""",
)

md(
    """
### ex / Mega / ACE SPEC

`ex` Pokémon give up an **extra prize** when KO'd (Mega ex give up more), so
they trade raw power for prize risk. ACE SPEC cards are limited to **one per
deck**. These flags shape both deck slots and ladder risk.
""",
)

code(
    """
print("ex:", int(pk["is_ex"].sum()),
      "| mega:", int(pk["is_mega"].sum()),
      "| ace spec:", int(cards["is_ace_spec"].sum()))

fig, ax = plt.subplots(figsize=(8, 4))
sns.kdeplot(data=pk, x="hp", hue="is_ex", fill=True, common_norm=False, ax=ax)
ax.set_title("HP: ex vs non-ex Pokémon")
plt.tight_layout()
plt.show()
""",
)

md(
    """
## 3. Attacks: cost vs damage

Damage-per-energy is the crude "tempo" measure. Watch for cheap high-damage
attacks (likely have drawbacks in the effect text) and for which energy colors
attacks demand.
""",
)

code(
    """
atk = moves[moves["is_attack"]].copy()
atk = atk[atk["damage_value"].notna() & (atk["cost_total"] > 0)]
atk["dmg_per_energy"] = atk["damage_value"] / atk["cost_total"]

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.stripplot(data=atk, x="cost_total", y="damage_value", ax=ax[0],
              alpha=0.4, color="#4c72b0", jitter=0.25)
ax[0].set_title("Attack damage vs energy cost")
ax[0].set_xlabel("total energy cost")
ax[0].set_ylabel("base damage")
sns.histplot(data=atk, x="dmg_per_energy", bins=30, ax=ax[1], color="#55a868")
ax[1].set_title("Damage per energy")
plt.tight_layout()
plt.show()
""",
)

code(
    """
# Most energy-efficient attacks (raw base damage / cost). Effects often explain
# why these are "too good" — read the effect column before trusting the number.
cols = ["name", "move_name", "cost", "damage", "cost_total",
        "damage_value", "dmg_per_energy", "effect"]
atk.nlargest(12, "dmg_per_energy")[cols].reset_index(drop=True)
""",
)

code(
    """
# Aggregate energy-color demand across every attack cost in the pool.
# Tells you which colors the format is built around.
demand = {}
for cost in moves["cost"].dropna():
    for code, n in parse_cost(cost).items():
        demand[code] = demand.get(code, 0) + n
demand = (
    pd.Series(demand)
    .rename(index=lambda c: ENERGY_SYMBOLS.get(c, c))
    .sort_values(ascending=False)
)
ax = demand.plot.bar(figsize=(9, 4), color="#937860")
ax.set_title("Total energy symbols required across all attacks")
ax.set_ylabel("count of symbols")
plt.tight_layout()
plt.show()
demand
""",
)

md("## 4. Trainers & Energy")

code(
    """
trainers = cards[cards["supertype"] == "Trainer"]
print(trainers["stage_or_type"].value_counts().to_string())
print()
print("Energy cards:")
print(cards[cards["supertype"] == "Energy"][["card_id", "name", "stage_or_type"]]
      .to_string(index=False))
""",
)

code(
    """
# Supporters are the deck's draw/disruption engine — skim the pool by name.
sup = trainers[trainers["stage_or_type"] == "Supporter"]
sorted(sup["name"].unique().tolist())
""",
)

md(
    """
## 5. Deck-building lens — decode the sample deck

The sample submission ships a 60-card `deck.csv` (one card ID per line). Decoding
it shows a real, legal deck and its energy identity — a useful baseline before we
build our own.
""",
)

code(
    """
deck_path = ROOT / "data" / "sample_submission" / "deck.csv"
deck_ids = [int(x) for x in deck_path.read_text().split() if x.strip()]
print(f"deck size: {len(deck_ids)} (must be 60)")

id_to = cards.set_index("card_id")
deck = (
    pd.Series(deck_ids, name="card_id")
    .map(id_to["name"])
    .value_counts()
    .rename_axis("name")
    .reset_index(name="count")
)
deck = deck.merge(
    cards[["name", "supertype", "stage_or_type", "type_code"]].drop_duplicates("name"),
    on="name", how="left",
)
deck.sort_values(["supertype", "count"], ascending=[True, False]).reset_index(drop=True)
""",
)

code(
    """
# Composition of the sample deck: Pokémon vs Trainer vs Energy, and color identity.
by_super = (
    pd.Series(deck_ids).map(id_to["supertype"]).value_counts()
)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
by_super.plot.bar(ax=ax[0], color="#4c72b0")
ax[0].set_title("Sample deck: cards by supertype")

pk_ids = [i for i in deck_ids if id_to.loc[i, "supertype"] == "Pokemon"]
pd.Series(pk_ids).map(id_to["type_code"]).map(
    lambda c: ENERGY_SYMBOLS.get(c, c)
).value_counts().plot.bar(ax=ax[1], color="#8172b3")
ax[1].set_title("Sample deck: Pokémon color identity")
plt.tight_layout()
plt.show()
""",
)

md(
    """
## Takeaways & next steps

- The pool is **~1,250 Standard cards**: mostly Pokémon, with a focused set of
  Trainers (draw/disruption) and a handful of Energy cards.
- Deck building is constrained by **energy color identity**, **stage lines**
  (Basic → Stage 1 → Stage 2), and **prize risk** (ex/Mega give up extra prizes).
- High damage-per-energy attacks almost always carry a drawback in the effect
  text — read it before trusting the number.

**Next:**
1. Pick / refine a deck archetype (the single biggest lever on ladder Elo).
2. Stand up the **simulator** (Linux x86-64 only — see README) to run battles and
   the `search_*` lookahead API.
3. Build a first heuristic `agent()` that beats the random baseline, then iterate
   against the live ladder (5 submissions/day).
""",
)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
