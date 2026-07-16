"""Offline forensic on collected AZ data: why the LSTM fails to beat greedy_plus.

Engine-free (replays recorded trajectories through the net; no games). Answers three
diagnostic questions that localise the failure to a link in the AZ chain:

1. DISTILLATION FIT (teacher->student): on searched steps, does the trained net's policy
   match the ISMCTS visit-count target pi? Reports policy CE(net||pi), argmax-agreement,
   and the pi/net entropies. High CE / low agreement = the LSTM did NOT fit the teacher
   (capacity/optim). Low CE but still weak play = the TEACHER (pi) is the problem.
2. VALUE CALIBRATION: replay -> V_t per step; correlate with outcome z, stratified
   by turn (early/mid/late). If V is ~0 / uninformative early, the advantage signal that
   should teach tempo is absent (opening-collapse). If V is well-calibrated, the
   value head is fine and the bias is in the policy target.
3. RECURRENCE CONTRIB: replay each trajectory twice -- normal h_t vs h FROZEN to zero
   each step (memoryless) -- and measure how often the argmax move and V differ. ~0
   difference => the LSTM recurrence is degenerate (not using history) = "the LSTM isn't
   really learning" in the most literal sense.

  uv run python scripts/az_target_diag.py --net data/coevo_az_v2/gen9/net.npz \
      --data data/coevo_az_v2/gen9/play_g9_e0.jsonl --games 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.run_eval import load_engine_data  # noqa: E402
from src.deck import build_pool  # noqa: E402
from src.net.embedding import CardEmbeddingIndex  # noqa: E402
from src.net.encode import (  # noqa: E402
    encode_options,
    encode_state,
    option_embed_rows,
    state_embed_rows,
)
from src.net.features import CardFeatures  # noqa: E402
from src.net.opp_context import build_hypotheses, opp_context  # noqa: E402
from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: E402

SINGLE_SELECT = 1
_EPS = 1e-9


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _replay(game: dict, net: RecurrentPolicyValueNet, feats: CardFeatures,  # noqa: PLR0913
            index: CardEmbeddingIndex, hyp: tuple, *, frozen: bool) -> list[dict]:
    """Replay one game's single-select trajectory; return per-step diagnostics.

    ``frozen=True`` zeroes the LSTM state each step (memoryless ablation)."""
    h, c = net.initial_state()
    use_opp = net.config.opp_ctx_dim > 0
    steps = []
    for mv in game.get("moves") or []:
        cur = mv.get("current") or {}
        sel = mv.get("select") or {}
        options = sel.get("option") or []
        if int(sel.get("maxCount", 0)) != SINGLE_SELECT or not options or not cur:
            continue
        slot = int(cur.get("yourIndex", 0))
        x = encode_state(cur, slot, feats)
        rows, mask = state_embed_rows(cur, slot, index)
        of = encode_options(options, cur, slot, feats)
        orows = option_embed_rows(options, cur, slot, index)
        opp_ctx = (net.opp_ctx(opp_context(cur, slot, hyp[0], hyp[1]))
                   if use_opp else None)
        h_in, c_in = (net.initial_state() if frozen else (h, c))
        logits, value, h, c = net.step(x, rows, mask, of, orows, h_in, c_in,
                                       opp_ctx=opp_ctx)
        if logits.shape[0] != len(options):
            continue
        p = _softmax(logits)
        pi = mv.get("pi")
        pi_arr = (np.asarray(pi, dtype=np.float64)
                  if pi and len(pi) == len(options) else None)
        steps.append({"turn": int(cur.get("turn", 0)), "value": float(value),
                      "argmax": int(p.argmax()), "p": p, "pi": pi_arr,
                      "entropy": float(-(p * np.log(p + _EPS)).sum())})
    return steps


def main() -> None:  # noqa: PLR0915, C901
    ap = argparse.ArgumentParser(description="Offline AZ target/distillation forensic")
    ap.add_argument("--net", type=Path, default=ROOT / "data/coevo_az_v2/gen9/net.npz")
    ap.add_argument("--data", type=Path,
                    default=ROOT / "data/coevo_az_v2/gen9/play_g9_e0.jsonl")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "results/az_target_diag.json")
    args = ap.parse_args()

    engine = load_engine_data()
    feats = CardFeatures(engine)
    index = CardEmbeddingIndex(build_pool())
    net = RecurrentPolicyValueNet.load(args.net)
    hyp = build_hypotheses(feats)
    games = [json.loads(ln) for ln in args.data.read_text().splitlines() if ln.strip()]
    games = games[:args.games]
    print(f"net opp_ctx_dim={net.config.opp_ctx_dim} "
          f"play_lstm_hidden={net.config.play_lstm_hidden}; {len(games)} games")

    # --- 1+2: distillation fit + value calibration (normal recurrence) ---
    ce_sum = agree = n_pi = 0.0
    pi_ent = net_ent_on_pi = 0.0
    v_all: list[tuple[int, float, float]] = []  # (turn, V, z)
    # --- 3: recurrence contribution (normal vs frozen) ---
    diff_argmax = n_steps = 0
    v_gap = 0.0
    for g in games:
        z = float(g.get("z", 0.0))
        norm = _replay(g, net, feats, index, hyp, frozen=False)
        froz = _replay(g, net, feats, index, hyp, frozen=True)
        for s in norm:
            v_all.append((s["turn"], s["value"], z))
            if s["pi"] is not None:
                p = s["p"]
                ce_sum += -float((s["pi"] * np.log(p + _EPS)).sum())
                agree += float(s["argmax"] == int(s["pi"].argmax()))
                pi_ent += float(-(s["pi"] * np.log(s["pi"] + _EPS)).sum())
                net_ent_on_pi += s["entropy"]
                n_pi += 1
        for a, b in zip(norm, froz, strict=False):
            n_steps += 1
            diff_argmax += int(a["argmax"] != b["argmax"])
            v_gap += abs(a["value"] - b["value"])

    def buck(turn: int) -> str:
        return "early(t<=2)" if turn <= 2 else ("mid(3-5)" if turn <= 5 else "late(6+)")

    strat: dict[str, list[tuple[float, float]]] = {}
    for turn, v, z in v_all:
        strat.setdefault(buck(turn), []).append((v, z))

    def corr(pairs: list[tuple[float, float]]) -> float:
        if len(pairs) < 3:
            return float("nan")
        v = np.array([p[0] for p in pairs])
        zz = np.array([p[1] for p in pairs])
        if v.std() < _EPS or zz.std() < _EPS:
            return 0.0
        return float(np.corrcoef(v, zz)[0, 1])

    out = {
        "net": str(args.net), "n_games": len(games), "n_pi_steps": int(n_pi),
        "distillation": {
            "policy_CE_net_vs_pi": round(ce_sum / max(n_pi, 1), 4),
            "argmax_agreement": round(agree / max(n_pi, 1), 4),
            "pi_entropy": round(pi_ent / max(n_pi, 1), 4),
            "net_entropy_on_searched": round(net_ent_on_pi / max(n_pi, 1), 4),
        },
        "value_calibration": {
            b: {"n": len(p), "mean_V": round(float(np.mean([x[0] for x in p])), 4),
                "mean_z": round(float(np.mean([x[1] for x in p])), 4),
                "corr_V_z": round(corr(p), 4)}
            for b, p in sorted(strat.items())
        },
        "recurrence": {
            "n_steps": n_steps,
            "argmax_changed_when_frozen": round(diff_argmax / max(n_steps, 1), 4),
            "mean_abs_V_gap_frozen": round(v_gap / max(n_steps, 1), 4),
        },
    }
    args.out.write_text(json.dumps(out, indent=2))
    d, vc, rec = out["distillation"], out["value_calibration"], out["recurrence"]
    print("\n[1] DISTILLATION FIT (student net vs teacher pi, searched steps)")
    print(f"    policy CE(net||pi) = {d['policy_CE_net_vs_pi']}  "
          f"argmax-agreement = {d['argmax_agreement']}")
    print(f"    pi entropy = {d['pi_entropy']}  net entropy = "
          f"{d['net_entropy_on_searched']}   (low CE+high agree => net fit teacher)")
    print("\n[2] VALUE CALIBRATION (V vs outcome z, by turn)")
    for b, s in vc.items():
        print(f"    {b:<12} n={s['n']:<6} mean_V={s['mean_V']:+.3f} "
              f"mean_z={s['mean_z']:+.3f} corr(V,z)={s['corr_V_z']:+.3f}")
    print("\n[3] RECURRENCE CONTRIBUTION (normal vs frozen/memoryless h)")
    print(f"    argmax changed when frozen = {rec['argmax_changed_when_frozen']}  "
          f"mean|ΔV| = {rec['mean_abs_V_gap_frozen']}")
    print("    (~0 => the LSTM recurrence is degenerate / not using history)")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
