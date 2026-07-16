"""AlphaZero trainer for the RECURRENT (LSTM) policy/value net.

Trains ``RecurrentPolicyValueNet`` on ISMCTS ``(state, pi, z)`` targets by replaying
each game's FULL single-select trajectory through ``net.play_sequence`` (so the LSTM
state ``h_t`` is reconstructed correctly), then applying:
- **policy** cross-entropy: masked ``log_softmax(logits)`` to the visit-count ``pi`` on
  searched steps (``pi_valid``);
- **value** MSE: ``V(s_t)`` to the game outcome ``z`` on every valid step.

Warm-starts from a numpy checkpoint (the paper net) and exports a numpy ``.npz`` for
serving / the next ISMCTS round. Trains on the A100 when available.

  uv run python scripts/train_az_lstm.py --data data/az/r0.jsonl \
      --warm data/qdcoevo/run7/round_6/rl/paper_final.npz --out data/az/lstm_r0.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from scripts.run_eval import load_engine_data  # noqa: E402
from src.deck import build_pool  # noqa: E402
from src.net.az_data import build_az_episodes, collate_az  # noqa: E402
from src.net.embedding import CardEmbeddingIndex  # noqa: E402
from src.net.features import CardFeatures  # noqa: E402

_NEG_INF = -1e9


def _batches(episodes: list, size: int) -> list[list]:
    """Group episodes by length (fewer pad steps) into batches of ``size``."""
    order = sorted(range(len(episodes)), key=lambda i: len(episodes[i].steps))
    return [[episodes[i] for i in order[b:b + size]]
            for b in range(0, len(order), size)]


def train(args: argparse.Namespace) -> None:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.nn.functional as torch_f  # noqa: PLC0415

    from src.net.opp_context import build_hypotheses  # noqa: PLC0415
    from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: PLC0415
    from src.net.recurrent_torch import from_numpy_recurrent  # noqa: PLC0415

    engine = load_engine_data()
    feats = CardFeatures(engine)
    pool = build_pool()
    index = CardEmbeddingIndex(pool)

    # (item 1) enable opponent-belief conditioning: migrate the warm net (adds a zero,
    # behaviour-preserving channel) and build the SAME hypothesis set serving uses.
    warm = RecurrentPolicyValueNet.load(args.warm)
    hyp = None
    if args.opp_ctx_dim > 0:
        warm = warm.enable_opp_ctx(np.random.default_rng(0), args.opp_ctx_dim)
        hyp = build_hypotheses(feats)
        print(f"opp-belief conditioning ON: dim={args.opp_ctx_dim}, "
              f"{len(hyp[0])} hypothesis decks")

    files = [Path(p) for p in str(args.data).split(",") if p]
    games = [json.loads(ln) for f in files
             for ln in f.read_text().splitlines() if ln.strip()]
    episodes = build_az_episodes(games, feats, index, hyp)
    n_steps = sum(len(e.steps) for e in episodes)
    n_pi = sum(1 for e in episodes for s in e.steps if s.pi is not None)
    print(f"AZ(LSTM): {len(episodes)} episodes, {n_steps} steps, {n_pi} pi-targets")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = from_numpy_recurrent(warm).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    batches = _batches(episodes, args.batch)

    for ep in range(args.epochs):
        tot_p = tot_v = n_p = n_v = 0.0
        for group in batches:
            b = {k: v.to(dev) for k, v in collate_az(group).items()}
            ctx = net.deck_ctx(b["deck_vec"]) if net.config.deck_ctx_dim > 0 else None
            octx = (net.opp_ctx(b["opp_vec"])
                    if net.config.opp_ctx_dim > 0 and "opp_vec" in b else None)
            logits, values = net.play_sequence(
                b["states"], b["state_rows"], b["state_mask"],
                b["options"], b["option_rows"], deck_ctx=ctx, opp_ctx=octx)
            logits = logits.masked_fill(~b["option_mask"], _NEG_INF)
            logp = torch_f.log_softmax(logits, dim=-1)                 # (B,T,K)
            pmask = b["pi_valid"] & b["valid"]                          # (B,T)
            ce = -(b["pi"] * logp).sum(-1)                             # (B,T)
            ploss = (ce * pmask).sum() / pmask.sum().clamp_min(1)
            vmask = b["valid"]
            vloss = (((values - b["value_target"]) ** 2) * vmask).sum(
                ) / vmask.sum().clamp_min(1)
            loss = ploss + args.value_coef * vloss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_p += float(ploss) * float(pmask.sum())
            tot_v += float(vloss) * float(vmask.sum())
            n_p += float(pmask.sum())
            n_v += float(vmask.sum())
        print(f"  epoch {ep + 1}: policy_CE={tot_p / max(n_p, 1):.4f} "
              f"value_MSE={tot_v / max(n_v, 1):.4f}")

    net.to_numpy_net().save(args.out)
    print(f"saved -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default="data/az/r0.jsonl")
    ap.add_argument("--warm", type=str,
                    default="data/qdcoevo/run7/round_6/rl/paper_final.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "data/az/lstm_r0.npz")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--value-coef", type=float, default=1.0)
    ap.add_argument("--opp-ctx-dim", type=int, default=0,
                    help="(item 1) opponent-belief conditioning width (0 = off)")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
