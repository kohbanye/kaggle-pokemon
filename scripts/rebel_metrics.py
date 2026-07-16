"""Multi-metric learning curve for self-play ReBeL checkpoints (Koh: "more games ->
better on MULTIPLE metrics, not just win-rate").

For each value-net checkpoint, on a FIXED representative-PBS set (no win-rate),
compute the metrics that should IMPROVE as self-play collects more games:
  * calib_kl  : KL(rollout-leaf policy || net-leaf policy) -- lower = the net's values
                agree more with a greedy_plus rollout (better calibrated leaf).
  * calib_dV  : |rollout root value - net root value| -- value-scale calibration.
  * conv_kl   : KL(net@lo-sweeps || net@hi-sweeps) -- solver convergence health.
  * entropy   : mean root-policy entropy (decisiveness; informational).
Plots each metric vs checkpoint so a rising #games -> falling calib/conv is visible.
Uses the fast depth-3 + top_k config (ties gp at 0.1s/solve) so many PBS are cheap.

  uv run python scripts/rebel_metrics.py --nets a.npz,b.npz,c.npz --n-pbs 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.rebel_diag import collect_pbs, make_rollout_leaf, solve_root  # noqa: E402
from src.rebel.belief import OpponentBelief  # noqa: E402


def _entropy(p: dict[int, float]) -> float:
    vals = [v for v in p.values() if v > 0]
    return float(-sum(v * np.log(v) for v in vals)) if vals else 0.0


def _kl(p: dict[int, float], q: dict[int, float]) -> float:
    eps = 1e-9
    out = 0.0
    for k in set(p) | set(q):
        pk = p.get(k, 0.0)
        if pk > 0:
            out += pk * np.log(pk / max(q.get(k, 0.0), eps))
    return float(out)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="metal_aggro")
    ap.add_argument("--nets", required=True, help="comma-sep checkpoint .npz")
    ap.add_argument("--n-pbs", type=int, default=12)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--particles", type=int, default=4)
    ap.add_argument("--tpw", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--sweeps-lo", type=int, default=4)
    ap.add_argument("--sweeps-hi", type=int, default=32)
    ap.add_argument("--out", default="results/rebel_metrics.json")
    args = ap.parse_args()

    from scripts.run_eval import load_engine_data, resolve_deck  # noqa: PLC0415
    engine = load_engine_data()
    deck = resolve_deck(args.deck)
    cards = engine.get("cards", {})
    basics = [c for c in deck if cards.get(c) and cards[c].get("basic")]
    belief = OpponentBelief.from_dirs()

    from src.net.features import CardFeatures  # noqa: PLC0415
    from src.rebel.value_net import (  # noqa: PLC0415
        ValueNet,
        build_hypothesis_ctxs,
        make_leaf_value,
    )
    feats = CardFeatures(engine)
    decks = belief.hypotheses
    ctxs = build_hypothesis_ctxs(decks, feats)
    leaf_roll = make_rollout_leaf(deck, engine)

    print(f"collecting {args.n_pbs} fixed PBS ...", flush=True)
    pbs = collect_pbs(deck, engine, args.n_pbs, seed=0)
    print(f"  got {len(pbs)} PBS", flush=True)

    tk = None if args.top_k <= 0 else args.top_k
    def _solve(obs: dict, leaf: object, sweeps: int, sd: int) -> tuple[dict, float]:
        return solve_root(obs, deck, belief, basics, leaf, args.particles, args.depth,
                          sweeps, args.tpw, sd, cfr_plus=True, top_k=tk)

    nets = [p for p in args.nets.split(",") if p]
    rows = []
    t0 = time.perf_counter()
    for ci, npath in enumerate(nets):
        full = npath if Path(npath).is_absolute() else str(ROOT / npath)
        net = ValueNet.load(full)
        leaf_net = make_leaf_value(net, feats, decks, ctxs)
        calib_kl, calib_dv, conv_kl, ent = [], [], [], []
        for i, obs in enumerate(pbs):
            sd = 300 + i
            try:
                p_net, v_net = _solve(obs, leaf_net, args.sweeps_hi, sd)
                p_lo, _ = _solve(obs, leaf_net, args.sweeps_lo, sd)
                p_roll, v_roll = _solve(obs, leaf_roll, args.sweeps_hi, sd)
            except Exception as e:  # noqa: BLE001
                print(f"    ckpt{ci} pbs{i} skip ({type(e).__name__})", flush=True)
                continue
            calib_kl.append(_kl(p_roll, p_net))
            calib_dv.append(abs(v_roll - v_net))
            conv_kl.append(_kl(p_lo, p_net))
            ent.append(_entropy(p_net))
        row = {
            "ckpt": Path(npath).stem,
            "calib_kl": float(np.mean(calib_kl)) if calib_kl else None,
            "calib_dV": float(np.mean(calib_dv)) if calib_dv else None,
            "conv_kl": float(np.mean(conv_kl)) if conv_kl else None,
            "entropy": float(np.mean(ent)) if ent else None,
            "n": len(calib_kl),
        }
        rows.append(row)
        print(f"  [{ci}] {row['ckpt']}: calib_kl={row['calib_kl']:.3f} "
              f"calib_dV={row['calib_dV']:.3f} conv_kl={row['conv_kl']:.3f} "
              f"ent={row['entropy']:.3f} ({time.perf_counter()-t0:.0f}s)", flush=True)

    Path(ROOT / args.out).write_text(json.dumps(rows, indent=2))
    print("\n=== multi-metric learning curve (lower calib/conv = better) ===")
    print(f"{'ckpt':<20}{'calib_kl':>10}{'calib_dV':>10}{'conv_kl':>10}{'entropy':>10}")
    for r in rows:
        print(f"{r['ckpt']:<20}{r['calib_kl']:>10.3f}{r['calib_dV']:>10.3f}"
              f"{r['conv_kl']:>10.3f}{r['entropy']:>10.3f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
