"""Generate notebooks/03_paper_net_eval.ipynb from source cells (results only).

Renders the multi-faceted evaluation of the canonical recurrent paper net from
``results/paper_eval.json`` (produced by ``scripts/run_paper_eval.py``) and the
training log: training curve, head-to-head, checkpoint progression, the meta-deck
gauntlet, deck composition / diversity, and inference cost. Tables + plots only --
no prose conclusions. Keep the content in this builder (reviewable, regenerable):

  uv run python scripts/run_paper_eval.py        # refresh results/paper_eval.json
  uv run python scripts/build_eval_notebook.py
  uv run jupyter nbconvert --to notebook --execute --inplace \
      notebooks/03_paper_net_eval.ipynb
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "03_paper_net_eval.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(
    """
# Pokémon TCG — recurrent paper net evaluation

`paper_final` = recurrent V-Trace/PPO OSFP net, 5000 iterations. Source data:
`results/paper_eval.json` (`scripts/run_paper_eval.py`) and
`data/paperosfp/main/train.log`. All win rates are slot-swapped with Wilson 95% CI.
""",
)

code(
    """
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "results").exists())
E = json.loads((ROOT / "results" / "paper_eval.json").read_text())
M = E["matches"]
print("final:", E["final"])
print("phase5d ref:", E["phase5d"])
print("matches:", len(M))
""",
)

md("## Training curve — gate win rate vs iteration (vs metal_aggro)")
code(
    """
log = (ROOT / "data/paperosfp/main/train.log").read_text().splitlines()
gate = []
for ln in log:
    m = re.search(r"paperiter (\\d+).*gate=([0-9.]+)", ln)
    if m:
        gate.append((int(m.group(1)), float(m.group(2))))
gx = [a for a, _ in gate]
gy = [b for _, b in gate]
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(gx, gy, marker=".", lw=1)
ax.axhline(0.5, color="gray", ls="--", lw=0.8)
ax.set_xlabel("iteration")
ax.set_ylabel("gate win rate")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()
""",
)

md("## Head-to-head vs baselines")
code(
    """
labels = [
    ("vs_phase5d", "vs Phase-5d (full: own deck + play)"),
    ("vs_greedy_samedeck", "vs greedy (shared deck: play only)"),
    ("vs_random_samedeck", "vs random (shared deck: play only)"),
]
rows = [
    [lbl, round(M[k]["winrate"], 3),
     f"[{M[k]['ci_lo']:.3f}, {M[k]['ci_hi']:.3f}]", M[k]["decisive"],
     round(M[k]["avg_turns"], 1), round(M[k]["avg_move_ms"], 3)]
    for k, lbl in labels if k in M
]
pd.DataFrame(
    rows,
    columns=["matchup", "winrate", "95% CI", "decisive", "avg turns", "avg move ms"],
)
""",
)

md("## Checkpoint progression — win rate vs Phase-5d by iteration")
code(
    """
its = [it for it in E["checkpoint_iters"] if f"ckpt_{it}" in M]
ys = np.array([M[f"ckpt_{it}"]["winrate"] for it in its])
lo = np.array([M[f"ckpt_{it}"]["ci_lo"] for it in its])
hi = np.array([M[f"ckpt_{it}"]["ci_hi"] for it in its])
fig, ax = plt.subplots(figsize=(9, 4))
ax.errorbar(
    its, ys, yerr=[np.clip(ys - lo, 0, None), np.clip(hi - ys, 0, None)],
    marker="o", capsize=3,
)
ax.axhline(0.5, color="gray", ls="--", lw=0.8)
ax.set_ylim(0, 1)
ax.set_xlabel("checkpoint iteration")
ax.set_ylabel("win rate vs Phase-5d")
plt.tight_layout()
plt.show()
""",
)

md("## Gauntlet — the net's deck vs each meta archetype (greedy-piloted)")
code(
    """
g = sorted((k.replace("gauntlet_", ""), M[k]) for k in M if k.startswith("gauntlet_"))
names = [n for n, _ in g]
wr = np.array([s["winrate"] for _, s in g])
lo = np.array([s["ci_lo"] for _, s in g])
hi = np.array([s["ci_hi"] for _, s in g])
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(names, wr, color=["#4c72b0" if w >= 0.5 else "#c44e52" for w in wr])
ax.errorbar(
    names, wr, yerr=[np.clip(wr - lo, 0, None), np.clip(hi - wr, 0, None)],
    fmt="none", ecolor="black", capsize=3,
)
ax.axhline(0.5, color="gray", ls="--", lw=0.8)
ax.set_ylim(0, 1)
ax.set_ylabel("win rate")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
pd.DataFrame(
    [(n, round(s["winrate"], 3), f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]", s["decisive"])
     for n, s in g],
    columns=["meta deck", "winrate", "95% CI", "decisive"],
)
""",
)

md("## Deck composition (greedy build) + top cards")
code(
    """
d = E["deck"]
print("type composition:", d["greedy_comp"], " | distinct:", d["greedy_distinct"])
pd.DataFrame(d["greedy_top"], columns=["card id", "copies"])
""",
)

md("## Deck composition over training iterations")
code(
    """
ev = E.get("deck_evolution", [])
its = [r["iter"] for r in ev]
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].stackplot(
    its, [r["energy"] for r in ev], [r["pokemon"] for r in ev],
    [r["trainer"] for r in ev], labels=["energy", "pokemon", "trainer"], alpha=0.85,
)
axs[0].set_title("greedy deck composition")
axs[0].set_xlabel("iteration")
axs[0].set_ylabel("cards")
axs[0].legend(loc="upper right")
axs[1].plot(its, [r["sampled_distinct"] for r in ev], marker="o", label="distinct")
axs[1].plot(its, [r["sampled_energy"] for r in ev], marker="s", label="energy")
axs[1].set_title("sampled-deck diversity / energy")
axs[1].set_xlabel("iteration")
axs[1].legend()
plt.tight_layout()
plt.show()
pd.DataFrame(ev)
""",
)

md("## Sampled-deck diversity (final net, 30 samples)")
code(
    """
d = E["deck"]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.5))
axs[0].hist(d["sampled_distinct"], bins=12)
axs[0].set_title("distinct cards / deck")
axs[0].set_xlabel("distinct")
axs[1].hist(d["sampled_energy"], bins=12)
axs[1].set_title("energy count / deck")
axs[1].set_xlabel("energy")
plt.tight_layout()
plt.show()
""",
)

md("## Inference cost (per move, recurrent serving)")
code(
    """
rows = [
    [k, round(M[k]["avg_move_ms"], 3), round(M[k]["max_move_ms"], 1)]
    for k in ("vs_phase5d", "vs_greedy_samedeck", "vs_random_samedeck") if k in M
]
pd.DataFrame(rows, columns=["match", "avg move ms", "max move ms"])
""",
)


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
