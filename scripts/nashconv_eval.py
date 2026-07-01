"""NashConv (restricted) evaluation over a strategy population -> results/nashconv.json.

NashConv(profile) = sum over players of how much each player gains by switching to a
best response. We use the RESTRICTED form: the best response is taken over a FIXED
strategy population (an empirical / EGTA meta-game), so it is a *lower bound* on true
exploitability -- only as strong as the population (hence the exploiter strategies).

A "strategy" = a (pilot, deck) pair (optionally a forced-go-first wrapper). For each
strategy i we play every other strategy j (slot-swapped) to fill the win matrix
W[i][j] = P(i beats j). Then for the symmetric profile "everyone plays i":

    exploitability(i) = max_j W[j][i] - 0.5 = 0.5 - min_j W[i][j]   (worst matchup)
    NashConv(i)       = 2 * exploitability(i)

Lower NashConv = harder to exploit = more robust (what the ladder rewards). We also
solve the population's symmetric Nash mixture (clone-invariant; Balduzzi-style) and
report its support = the strategies that actually survive.

Native/Docker (imports cg).  uv run python scripts/nashconv_eval.py --games 30
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.run_eval import load_engine_data, play_game, read_deck  # noqa: E402
from src.agents import build_agent  # noqa: E402
from src.agents.base import OPT_YES  # noqa: E402
from src.agents.recurrent_agent import RecurrentNetAgent  # noqa: E402
from src.deck import build_pool  # noqa: E402

CTX_IS_FIRST = 41  # SelectContext.IS_FIRST
INIT = "data/paperosfp/main/paper_final.npz"
RUN3 = "data/qdcoevo/run3/round_6/rl/paper_final.npz"
RUN7 = "data/qdcoevo/run7/round_6/rl/paper_final.npz"

# label -> (kind, deck, ckpt|None, force_first).  Covers play-on-metal robustness,
# real submission units, and exploiter probes (forced-go-first, type variety).
STRATS: dict[str, tuple[str, str, str | None, bool]] = {
    "greedy|metal":       ("greedy", "metal_aggro", None, False),   # ladder-best ref
    "net_init|metal":     ("net", "metal_aggro", INIT, False),
    "net_run3|metal":     ("net", "metal_aggro", RUN3, False),
    "net_run7|metal":     ("net", "metal_aggro", RUN7, False),
    "random|metal":       ("random", "metal_aggro", None, False),   # floor
    "net_run7|run7_best": ("net", "run7_best", RUN7, False),        # pending submission
    "net_run3|grass":     ("net", "grass_aggro", RUN3, False),      # the 451 submission
    "greedy|run7_best":   ("greedy", "run7_best", None, False),
    "greedy|fire":        ("greedy", "fire_aggro", None, False),    # type variety
    "greedy|psychic":     ("greedy", "psychic_aggro", None, False),
    "greedyFF|metal":     ("greedy", "metal_aggro", None, True),    # go-2nd exploiter
}

_G: dict = {}


class _ForcedFirst:
    """Wrap an agent so it always chooses to go FIRST when the engine asks IS_FIRST."""

    def __init__(self, inner: object) -> None:
        self.inner = inner

    def reset(self, seed: int) -> None:
        self.inner.reset(seed)

    def __call__(self, obs: dict) -> list[int]:
        sel = obs.get("select")
        if sel is not None and int(sel.get("context", -1)) == CTX_IS_FIRST:
            for i, o in enumerate(sel.get("option") or []):
                if int(o.get("type", -1)) == OPT_YES:
                    return [i]
        return self.inner(obs)


def _init(net_paths: dict[str, str], deck_names: list[str]) -> None:
    from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: PLC0415

    _G["engine"] = load_engine_data()
    _G["pool"] = build_pool()
    _G["nets"] = {p: RecurrentPolicyValueNet.load(p) for p in set(net_paths.values())}
    _G["decks"] = {nm: read_deck(ROOT / "decklists" / f"{nm}.csv") for nm in deck_names}


def _agent(label: str) -> object:
    kind, deck_name, ckpt, force_first = STRATS[label]
    deck = _G["decks"][deck_name]
    if kind == "net":
        net = _G["nets"][str(ROOT / ckpt)]
        base = RecurrentNetAgent(deck, _G["engine"], net=net,
                                 cb_pool=_G["pool"], build_deck_from_net=False,
                                 temperature=0.0)
    else:
        base = build_agent(kind, deck, _G["engine"])
    return _ForcedFirst(base) if force_first else base


def _play(task: dict) -> dict:
    a, b = _agent(task["i"]), _agent(task["j"])
    i_first = task["i_first"]
    p0, p1 = (a, b) if i_first else (b, a)
    res = play_game(p0, p1, a_is_player0=i_first, seed=task["seed"])
    return {"i": task["i"], "j": task["j"],
            "i_won": int(res.a_won), "dec": int(res.a_won or res.b_won)}


def _nash_mixture(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric Nash of the antisymmetric meta-game M=W-0.5 via LP; returns (p, BR)."""
    from scipy.optimize import linprog  # noqa: PLC0415

    n = w.shape[0]
    m = w - 0.5
    # maximise v s.t. for all j: sum_i p_i M[i][j] >= v ; sum p = 1 ; p >= 0
    c = np.zeros(n + 1)
    c[-1] = -1.0
    a_ub = np.column_stack([-m.T, np.ones(n)])          # -M[i][j]*p_i + v <= 0
    b_ub = np.zeros(n)
    a_eq = np.zeros((1, n + 1))
    a_eq[0, :n] = 1.0
    bounds = [(0.0, 1.0)] * n + [(None, None)]
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0], bounds=bounds)
    p = np.clip(res.x[:n], 0, None)
    p = p / p.sum() if p.sum() > 0 else np.ones(n) / n
    br = float(max((w[i] @ p) for i in range(n)))        # best-response winrate vs p
    return p, br


def main() -> None:
    ap = argparse.ArgumentParser(description="Restricted-NashConv evaluation")
    ap.add_argument("--games", type=int, default=30, help="games per unordered pair")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=Path, default=ROOT / "results/nashconv.json")
    args = ap.parse_args()

    labels = list(STRATS)
    n = len(labels)
    net_paths = {lbl: str(ROOT / s[2]) for lbl, s in STRATS.items() if s[2]}
    deck_names = sorted({s[1] for s in STRATS.values()})

    tasks = [
        {"i": labels[i], "j": labels[j], "i_first": k % 2 == 0,
         "seed": (i * 131 + j) * 1000 + k}
        for i in range(n) for j in range(i + 1, n) for k in range(args.games)
    ]
    print(f"strategies={n} pairs={n * (n - 1) // 2} games/pair={args.games} "
          f"total={len(tasks)}")

    with Pool(args.workers, initializer=_init,
              initargs=(net_paths, deck_names)) as pp:
        rows = pp.map(_play, tasks)

    idx = {lbl: i for i, lbl in enumerate(labels)}
    wins = np.zeros((n, n))
    dec = np.zeros((n, n))
    for r in rows:
        i, j = idx[r["i"]], idx[r["j"]]
        wins[i, j] += r["i_won"]
        dec[i, j] += r["dec"]
    w = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(i + 1, n):
            wij = wins[i, j] / dec[i, j] if dec[i, j] else 0.5
            w[i, j] = wij
            w[j, i] = 1 - wij

    # per-strategy restricted NashConv (worst matchup)
    summary = []
    for i, lbl in enumerate(labels):
        others = [j for j in range(n) if j != i]
        worst_j = min(others, key=lambda j: w[i, j])
        worst_wr = float(w[i, worst_j])
        summary.append({
            "strategy": lbl, "nashconv": round(2 * (0.5 - worst_wr), 3),
            "exploitability": round(0.5 - worst_wr, 3),
            "worst_vs": labels[worst_j], "worst_winrate": round(worst_wr, 3),
            "mean_winrate": round(float(w[i, others].mean()), 3),
        })
    summary.sort(key=lambda d: d["nashconv"])

    p, br = _nash_mixture(w)
    nash = {labels[i]: round(float(p[i]), 3) for i in range(n) if p[i] > 1e-3}
    nashconv_pop = round(2 * (br - 0.5), 3)

    out = {"labels": labels, "games_per_pair": args.games,
           "win_matrix": [[round(float(w[i, j]), 3) for j in range(n)]
                          for i in range(n)],
           "per_strategy_nashconv": summary,
           "nash_mixture": nash, "nashconv_of_nash": nashconv_pop}
    args.out.write_text(json.dumps(out, indent=2))
    print(f"-> {args.out}\n")
    print(f"{'strategy':<20}{'NashConv':>10}{'exploit':>9}{'worst_vs':>20}{'wr':>7}")
    for d in summary:
        print(f"{d['strategy']:<20}{d['nashconv']:>10}{d['exploitability']:>9}"
              f"{d['worst_vs']:>20}{d['worst_winrate']:>7}")
    print(f"\nNash support (clone-invariant robust core): {nash}")
    print(f"NashConv of the Nash mixture (~0 check): {nashconv_pop}")


if __name__ == "__main__":
    main()
