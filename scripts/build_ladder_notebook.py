"""Generate notebooks/04_ladder_episodes.ipynb from source cells.

Parameterised ladder-match analysis: set ``SUBMISSION_IDS`` in the first code
cell and re-run to pull that submission's real Kaggle episodes (opponents, Elo
trajectory, results) and the opponents' actual 60-card decks from the replays,
then break wins/losses down by real opponent archetype and compare the ladder's
deck distribution against our held-out pool. All fetching/classification logic
lives in ``scripts/ladder_episodes.py`` (cached under ``results/episodes/``);
the notebook only renders. Regenerate + execute:

  uv run python scripts/build_ladder_notebook.py
  uv run jupyter nbconvert --to notebook --execute --inplace \
      notebooks/04_ladder_episodes.ipynb
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "04_ladder_episodes.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(
    """
# Ladder episode analysis — who did a submission actually play, and how did it go?

Set `SUBMISSION_IDS` below (from `kaggle competitions submissions`) and run all
cells. Episodes and replays are fetched once and cached in `results/episodes/`;
opponent decks are extracted from the replays and classified with the same
decklist-only archetype features the QD search uses.
""",
)

code(
    """
# --- config: which of our submissions to analyse -----------------------------
SUBMISSION_IDS = [54367885, 54266227, 54297327]
""",
)

code(
    """
import sys, json
from pathlib import Path
from collections import Counter

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.deck import build_pool
from src.qd import deck_stats, evolution_depth, prize_liability
from scripts.ladder_episodes import (
    classify, fetch_episode_list, fetch_replay, replay_decks,
)
from scripts.run_eval import read_deck

pool = build_pool()
pd.set_option("display.width", 160)
""",
)

md("## 1. Fetch episodes + opponent decks (cached)")

code(
    """
rows = []
for sid in SUBMISSION_IDS:
    listing = fetch_episode_list(sid)
    teams = {t["id"]: t.get("teamName", "?") for t in listing.get("teams", [])}
    for ep in listing["episodes"]:
        if ep.get("state") != "COMPLETED":
            continue
        agents = ep["agents"]
        mine = next((a for a in agents if a.get("submissionId") == sid), None)
        opp = next((a for a in agents if a.get("submissionId") != sid), None)
        if mine is None or opp is None:
            continue
        replay = fetch_replay(ep["id"])
        decks = replay_decks(replay) if replay else None
        my_idx = mine.get("index", 0)
        rows.append({
            "sub": sid,
            "time": ep.get("endTime", ""),
            "reward": mine.get("reward"),
            "elo_before": mine.get("initialScore"),
            "elo_after": mine.get("updatedScore"),
            "opp_elo": opp.get("initialScore"),
            "opp_team": teams.get(opp.get("teamId"), "?"),
            "opp_sub": opp.get("submissionId"),
            "opp_deck": decks[1 - my_idx] if decks else None,
        })

df = pd.DataFrame(rows).sort_values(["sub", "time"]).reset_index(drop=True)
df["result"] = df["reward"].map({1: "W", -1: "L", 0: "T"})
df["opp_archetype"] = df["opp_deck"].map(
    lambda d: classify(d, pool) if d else "?")
print(f"{len(df)} completed episodes across {len(SUBMISSION_IDS)} submissions")
df.groupby("sub")["result"].value_counts().unstack(fill_value=0)
""",
)

md("## 2. Elo trajectory per submission")

code(
    """
fig, ax = plt.subplots(figsize=(9, 4))
for sid, g in df.groupby("sub"):
    g = g.reset_index(drop=True)
    ax.plot(g.index, g["elo_after"], marker="o", ms=3, label=str(sid))
ax.set_xlabel("episode #")
ax.set_ylabel("Elo after episode")
ax.legend(title="submission")
ax.grid(alpha=0.3)
plt.tight_layout()
""",
)

md("## 3. Win rate by opponent Elo bracket")

code(
    """
dec = df[df["reward"].isin([1, -1])].copy()
dec["bracket"] = pd.cut(dec["opp_elo"], [0, 450, 550, 650, 2000],
                        labels=["<450", "450-550", "550-650", "650+"])
(dec.groupby(["sub", "bracket"], observed=True)["reward"]
    .agg(games="count", winrate=lambda r: (r == 1).mean().round(2)))
""",
)

md("## 4. Wins / losses by real opponent archetype")

code(
    """
dec["arch"] = dec["opp_archetype"].str.split(" ").str[0]
pivot = (dec.groupby(["arch"])["reward"]
         .agg(games="count", winrate=lambda r: round((r == 1).mean(), 2))
         .sort_values("games", ascending=False))
pivot
""",
)

code(
    """
# every loss, hardest opponents first
cols = ["sub", "opp_team", "opp_elo", "opp_archetype"]
dec[dec["reward"] == -1].sort_values("opp_elo", ascending=False)[cols]
""",
)

md(
    """
## 5. The real ladder meta vs our held-out pool

Distinct opponent decks (deduped by opponent submission), composition and
evolution share, against `decklists/heldout/`.
""",
)

code(
    """
seen = {}
for _, r in df.iterrows():
    if r["opp_deck"]:
        seen[r["opp_sub"]] = r["opp_deck"]
ladder_decks = list(seen.values())

def pool_stats(decks):
    comp = np.array([[deck_stats(d, pool)[k]
                      for k in ("pokemon", "trainer", "energy")] for d in decks])
    evo = np.array([evolution_depth(d, pool) for d in decks])
    return {
        "n": len(decks),
        "evo-deck share": round(float((evo > 0.25).mean()), 2),
        "P median": float(np.median(comp[:, 0])),
        "T median": float(np.median(comp[:, 1])),
        "E median": float(np.median(comp[:, 2])),
    }

heldout = [read_deck(p) for p in sorted((ROOT / "decklists/heldout").glob("*.csv"))]
pd.DataFrame({"ladder (real)": pool_stats(ladder_decks),
              "held-out (ours)": pool_stats(heldout)})
""",
)

code(
    """
# strongest distinct opponents -- candidates for a real-meta eval pool / anchors
opp_best = (df.dropna(subset=["opp_deck"])
            .sort_values("opp_elo", ascending=False)
            .drop_duplicates("opp_sub"))
opp_best[["opp_team", "opp_elo", "opp_archetype"]].head(20)
""",
)


md(
    """
## 6. Tactics — how the games were actually played

State-based per-turn features from the replays (`scripts/ladder_tactics.py`):
prize-race pace, evolution progress, bench width, and the terminal reason.
Run `uv run python scripts/ladder_tactics.py` first to (re)build
`results/episodes/ladder_tactics.json`.
""",
)

code(
    """
tac = pd.DataFrame(json.loads(
    (ROOT / "results/episodes/ladder_tactics.json").read_text()))
tac["result"] = tac["reward"].map({1: "W", -1: "L"})
me = pd.json_normalize(tac["me"]).add_prefix("me_")
op = pd.json_normalize(tac["opp"]).add_prefix("opp_")
tac = pd.concat([tac.drop(columns=["me", "opp"]), me, op], axis=1)
tac.groupby(["sub", "result"])[
    ["turns", "opp_prizes_by_t10", "me_prizes_by_t10",
     "opp_stage_by_t6", "me_stage_by_t6", "opp_bench_t3", "me_bench_t3"]
].mean().round(2)
""",
)

code(
    """
# how do games END? (the engine meta in one table)
tac.groupby(["result", "reason"]).size().unstack(fill_value=0)
""",
)

nb = nbf.v4.new_notebook(cells=cells)
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT}")
