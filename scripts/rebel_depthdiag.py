"""Under converged CFR+, does the root policy change as search goes DEEPER? (cheap)

The earlier "depth3 == depth4" (win-rate) was measured with UNCONVERGED sweeps=4 and no
width bound. If a CONVERGED (CFR+) deeper solve gives ~the same root policy as depth-3,
the horizon truly doesn't matter -> ship the depth-3 CFR+ Nash tie. If it diverges, the
horizon is the real lever. Measured on fixed PBS via KL + top-action swap vs the SHALLOW
baseline -- noise-free (no win-rate) -- plus per-depth solve time so we know what is
servable. The ``--top-k`` width bound is what makes depth 4-6 affordable (it caps
branching), so this is the tool for the deeper-search perf phase.

  uv run python scripts/rebel_depthdiag.py --n-pbs 12 --sweeps 24 --depths 3,4,5,6 \
      --top-k 3
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

from scripts.rebel_diag import _kl, _top, collect_pbs, solve_root  # noqa: E402
from src.rebel.belief import OpponentBelief  # noqa: E402


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="metal_aggro")
    ap.add_argument("--net", default="data/rebel/vnd3_metal_r4.npz")
    ap.add_argument("--n-pbs", type=int, default=12)
    ap.add_argument("--particles", type=int, default=4)
    ap.add_argument("--tpw", type=int, default=4)
    ap.add_argument("--sweeps", type=int, default=24)
    ap.add_argument("--depths", default="3,4",
                    help="comma list; each is compared to the FIRST (shallow baseline)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="growing-tree width bound (makes depth 4-6 affordable)")
    ap.add_argument("--out", default="results/rebel_depthdiag.json")
    args = ap.parse_args()
    depths = [int(d) for d in args.depths.split(",") if d]

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
    net = ValueNet.load(str(ROOT / args.net))
    leaf_net = make_leaf_value(net, feats, decks, ctxs)

    print(f"collecting {args.n_pbs} PBS ...", flush=True)
    pbs = collect_pbs(deck, engine, args.n_pbs, seed=0)
    print(f"  got {len(pbs)} PBS", flush=True)

    base = depths[0]  # shallow baseline; every other depth is compared to it
    kls: dict[int, list[float]] = {d: [] for d in depths[1:]}
    swaps: dict[int, int] = dict.fromkeys(depths[1:], 0)
    times: dict[int, list[float]] = {d: [] for d in depths}
    n = 0
    for i, obs in enumerate(pbs):
        sd = 200 + i
        try:  # a determinized world can be invalid (bad opp_active) -> skip that PBS
            pols = {}
            for d in depths:
                ts = time.perf_counter()
                pols[d], _ = solve_root(obs, deck, belief, basics, leaf_net,
                                        args.particles, d, args.sweeps, args.tpw, sd,
                                        cfr_plus=True, top_k=args.top_k)
                times[d].append(time.perf_counter() - ts)
        except Exception as e:  # noqa: BLE001
            print(f"  pbs {i}: skipped ({type(e).__name__})", flush=True)
            continue
        n += 1
        msg = []
        for d in depths[1:]:
            kl = _kl(pols[d], pols[base])
            sw = int(_top(pols[d]) != _top(pols[base]))
            kls[d].append(kl)
            swaps[d] += sw
            msg.append(f"d{d}:KL={kl:.3f},sw={sw}")
        print(f"  pbs {i}: {' '.join(msg)}", flush=True)

    per_depth = {
        d: {"mean_solve_s": round(float(np.mean(times[d])), 3) if times[d] else 0.0,
            "KL_vs_base": round(float(np.mean(kls[d])), 3) if kls.get(d) else 0.0,
            "topswap_vs_base": round(swaps[d] / n, 3) if d in swaps and n else 0.0}
        for d in depths}
    summary = {
        "n_pbs": n, "sweeps": args.sweeps, "cfr_plus": True, "top_k": args.top_k,
        "base_depth": base, "per_depth": per_depth,
        # ref (rebel_diag_cfrplus): leaf-swap KL ~5, converged mid-KL ~0.19
    }
    Path(ROOT / args.out).write_text(json.dumps(summary, indent=2))
    print(f"\n=== deeper-search root policy vs depth-{base} (CFR+ converged, "
          f"top_k={args.top_k}) ===")
    print(json.dumps(summary, indent=2))
    print("\nREAD: KL~0 & low topswap at every depth => horizon does NOT matter under "
          "CFR+ (ceiling confirmed, ship the Nash tie). Large/rising KL/topswap => "
          "deeper search changes play => the horizon is the lever (then win-rate it). "
          "mean_solve_s tells us which depths are servable (<~1-2s/solve).")

if __name__ == "__main__":
    main()
