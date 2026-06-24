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


md("## Deck vs play decomposition (2x2) + mirror")
code(
    """
A = json.loads((ROOT / "results" / "paper_analysis.json").read_text())
mu = A["matchups"]
labels = {
    "play_net_vs_greedy_on_netdeck": "net play vs greedy  (on the NET's deck)",
    "play_net_vs_greedy_on_metadeck": "net play vs greedy  (on a META deck)",
    "deck_net_vs_meta_netplay": "net deck vs meta deck  (same net play)",
    "mirror": "mirror  (net vs net)",
}
ks = list(labels)
wr = [mu[k]["winrate"] for k in ks]
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.barh([labels[k] for k in ks], wr,
        color=["#4c72b0" if w >= 0.5 else "#c44e52" for w in wr])
ax.axvline(0.5, color="gray", ls="--", lw=0.8)
ax.set_xlim(0, 1)
ax.set_xlabel("A win rate")
plt.tight_layout()
plt.show()
pd.DataFrame(
    [(labels[k], mu[k]["winrate"], f"[{mu[k]['ci'][0]}, {mu[k]['ci'][1]}]",
      mu[k]["decisive"]) for k in ks],
    columns=["matchup", "A winrate", "95% CI", "decisive"],
)
""",
)

md("## Diagnostics — value calibration, loss-cause, slot bias, policy entropy")
code(
    """
A = json.loads((ROOT / "results" / "paper_analysis.json").read_text())
vc, lc, sb, pe = (A["value_calibration"], A["loss_cause"], A["slot_bias"],
                  A["policy_entropy"])
pd.DataFrame([
    ("value sign-acc (overall / early / late)",
     f"{vc['sign_accuracy']} / {vc['early_game']} / {vc['late_game']}"),
    ("value mean|v|", vc["mean_abs_value"]),
    ("abort / draw rate", f"{lc['abort_rate']} / {lc['draw_rate']}"),
    ("avg turns (win / loss)", f"{lc['avg_win_turns']} / {lc['avg_loss_turns']}"),
    ("mirror winrate (first / second)",
     f"{sb['mirror_first_winrate']} / {sb['mirror_second_winrate']}"),
    ("policy entropy (mean / median)", f"{pe['mean']} / {pe['median']}"),
], columns=["metric", "value"])
""",
)


md("## Strength over iterations — checkpoint round-robin (Bradley-Terry Elo)")
code(
    """
S = json.loads((ROOT / "results" / "strength.json").read_text())
its = S["iters"]
elo = S["elo"]
W = np.array([[np.nan if x is None else x for x in row] for row in S["win_matrix"]])
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
axs[0].plot(its, elo, marker="o")
axs[0].set_xlabel("iteration")
axs[0].set_ylabel("Bradley-Terry Elo")
axs[0].set_title("strength vs iteration (round-robin)")
im = axs[1].imshow(W, cmap="RdBu_r", vmin=0, vmax=1)
axs[1].set_xticks(range(len(its)))
axs[1].set_xticklabels(its, rotation=45)
axs[1].set_yticks(range(len(its)))
axs[1].set_yticklabels(its)
axs[1].set_title("row beats column (win rate)")
axs[1].set_xlabel("opponent iteration")
axs[1].set_ylabel("iteration")
fig.colorbar(im, ax=axs[1], fraction=0.046)
plt.tight_layout()
plt.show()
pd.DataFrame({"iter": its, "elo": elo})
""",
)

md("## Explored deck space — t-SNE of sampled decks, coloured by iteration")
code(
    """
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

S = json.loads((ROOT / "results" / "strength.json").read_text())
sd = S["sampled_decks"]
metas = {p.stem: [int(x) for x in p.read_text().split()]
         for p in sorted((ROOT / "decklists").glob("*.csv"))}
vocab = sorted(
    {c for decks in sd.values() for d in decks for c in d}
    | {c for d in metas.values() for c in d}
)
col = {c: i for i, c in enumerate(vocab)}


def vec(deck):
    v = np.zeros(len(vocab))
    for c in deck:
        v[col[c]] += 1
    return v


X, it_col = [], []
for it in S["iters"]:
    for d in sd[str(it)]:
        X.append(vec(d))
        it_col.append(it)
meta_start = len(X)
for d in metas.values():
    X.append(vec(d))
X = np.array(X)
xp = PCA(n_components=min(30, X.shape[0] - 1)).fit_transform(X)
z = TSNE(n_components=2, perplexity=20, init="pca", random_state=0).fit_transform(xp)
fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(z[:meta_start, 0], z[:meta_start, 1], c=it_col, cmap="viridis",
                s=18, alpha=0.8)
ax.scatter(z[meta_start:, 0], z[meta_start:, 1], c="red", marker="*", s=220,
           edgecolor="black", label="meta decks")
for k, name in enumerate(metas):
    ax.annotate(name, (z[meta_start + k, 0], z[meta_start + k, 1]), fontsize=8)
fig.colorbar(sc, ax=ax, label="iteration")
ax.legend()
ax.set_title("explored deck space (t-SNE); colour = training iteration")
plt.tight_layout()
plt.show()
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
