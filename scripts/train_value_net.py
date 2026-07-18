"""Increment-1 value-model TRAINING + HONEST BASELINE COMPARISON.

Trains a small MLP V(state)->P(win) on the ``.npz`` produced by
``collect_value_data.py`` and reports whether it beats two trivial baselines:
  (a) always-predict-base-rate (train-set win rate)
  (b) a prize-differential logistic (P(win) from opp_prize - my_prize only)

Train/val split is BY GAME (not by sample) so samples from one game never
straddle the split -- otherwise the net could memorise a game's outcome from an
early state and "predict" it on a late state of the same game (leakage).

Run:  uv run python scripts/train_value_net.py --data data/value_ds.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parent.parent


class ValueMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits


def bce(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def acc(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p >= 0.5).astype(np.float32) == y))


def report(tag: str, p: np.ndarray, y: np.ndarray) -> None:
    print(f"  {tag:28s} acc={acc(p, y):.4f}  auc={roc_auc_score(y, p):.4f}  "
          f"bce={bce(p, y):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "value_ds.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--class-weight", action="store_true",
                    help="up-weight the rare loss class (better recall, "
                         "uncalibrated probs -> worse BCE)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "value_net.pt")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    x_all = d["features"].astype(np.float32)
    y = d["labels"].astype(np.float32)
    gids = d["game_ids"]
    my_prize = d["my_prize"].astype(np.float32)
    opp_prize = d["opp_prize"].astype(np.float32)
    names = list(d["feature_names"])
    print(f"loaded {x_all.shape[0]} samples, {x_all.shape[1]} feats, "
          f"{len(np.unique(gids))} games, win-rate={y.mean():.3f}")

    # --- split BY GAME (no sample from one game straddles the split) ----------
    rng = np.random.default_rng(args.seed)
    uniq = np.unique(gids)
    rng.shuffle(uniq)
    n_val = int(len(uniq) * args.val_frac)
    val_games = set(uniq[:n_val].tolist())
    val_mask = np.array([g in val_games for g in gids])
    tr = ~val_mask
    print(f"train games={len(uniq) - n_val} samples={tr.sum()} | "
          f"val games={n_val} samples={val_mask.sum()}")

    # --- standardise using TRAIN stats ---------------------------------------
    mu = x_all[tr].mean(0)
    sd = x_all[tr].std(0) + 1e-6
    x_norm = (x_all - mu) / sd
    x_tr = torch.tensor(x_norm[tr])
    y_tr = torch.tensor(y[tr])
    x_va = torch.tensor(x_norm[val_mask])
    yva = y[val_mask]

    # Default: plain BCE -> calibrated P(win), fair BCE vs baselines. Optional
    # --class-weight up-weights the rare loss class (better minority recall but
    # uncalibrated probabilities, so BCE gets worse).
    torch.manual_seed(args.seed)
    model = ValueMLP(x_all.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if args.class_weight:
        pos_w = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    # Early stopping on val BCE (keep the best-calibrated epoch): with plain BCE
    # the net otherwise grows over-confident and its BCE drifts ABOVE base-rate
    # even while AUC stays high -- early stopping gives an honest BCE.
    n = x_tr.shape[0]
    bs = 512
    best_bce = float("inf")
    best_state = None
    best_ep = -1
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(x_va)).numpy()
        vb = bce(pv, yva)
        if vb < best_bce:
            best_bce, best_ep = vb, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"early-stop: best val-BCE {best_bce:.4f} at epoch {best_ep}")

    model.eval()
    with torch.no_grad():
        p_val = torch.sigmoid(model(x_va)).numpy()

    # --- baselines ------------------------------------------------------------
    base_rate = float(y[tr].mean())
    p_base = np.full(val_mask.sum(), base_rate, dtype=np.float32)

    # prize-differential logistic: single feature = opp_prize - my_prize
    pd_tr = (opp_prize[tr] - my_prize[tr]).reshape(-1, 1)
    pd_va = (opp_prize[val_mask] - my_prize[val_mask]).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(pd_tr, y[tr])
    p_pd = lr.predict_proba(pd_va)[:, 1]

    print("\n== VAL metrics (higher acc/auc better, lower bce better) ==")
    report("learned V (MLP, all feats)", p_val, yva)
    report("baseline: base-rate", p_base, yva)
    report("baseline: prize-diff logistic", p_pd, yva)

    auc_v = roc_auc_score(yva, p_val)
    auc_pd = roc_auc_score(yva, p_pd)
    print(f"\nAUC lift (learned V - prize-diff) = {auc_v - auc_pd:+.4f}")
    verdict = ("BEATS" if auc_v - auc_pd > 0.02
               else "~matches" if auc_v - auc_pd > -0.02 else "LOSES to")
    print(f"VERDICT: learned V {verdict} the prize-differential baseline.")

    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                "feature_names": names}, args.out)
    print(f"saved weights -> {args.out}")


if __name__ == "__main__":
    main()
