"""Held-out value-MSE learning curve: does the value net LEARN as self-play adds games?

The cleanest answer to "more games -> better?" free of win-rate noise AND of the belief-
averaging artifact in rebel_metrics' calib_kl. Collect ONCE a fixed held-out set of
(state_features x, actual game outcome z) from greedy_plus self-play (x from the acting
player's view, z=+1 if that player won else -1 -- exactly what `rebel_selfplay --target
outcome` records). Then for each checkpoint compute MSE(net.value(x), z) and Pearson r.
Pure numpy forward passes, no engine, no belief averaging. Falling MSE across rounds ==
the net genuinely learns value with more games; flat == it does not.

  uv run python scripts/rebel_valcurve.py --nets a.npz,b.npz --games 40
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

from scripts.run_eval import load_engine_data, play_game, resolve_deck  # noqa: E402
from src.agents import build_agent  # noqa: E402
from src.agents.base import Agent  # noqa: E402
from src.rebel.belief import OpponentBelief  # noqa: E402

SINGLE_SELECT = 1
_MIN_CHOICE = 2


class _FeatRec(Agent):
    """Wrap greedy_plus; record (state_features, marker) at its decisions."""

    name = "featrec"

    def __init__(self, inner: Agent, store: list, feats: object, decks: list,
                 ctxs: object) -> None:
        super().__init__(inner.deck)
        self.inner = inner
        self.store = store
        self._feats, self._decks, self._ctxs = feats, decks, ctxs

    def reset(self, seed: int) -> None:
        self.inner.reset(seed)

    def act(self, obs: dict) -> list[int]:
        from src.rebel.value_net import state_features  # noqa: PLC0415
        cur = obs.get("current")
        select = obs.get("select") or {}
        if cur is not None and int(select.get("maxCount", 0)) == SINGLE_SELECT \
                and len(select.get("option") or []) >= _MIN_CHOICE:
            your = int(cur.get("yourIndex", 0))
            x = state_features(cur, your, self._feats, self._decks, self._ctxs)
            self.store.append([x, None])  # z filled in after the game
        return self.inner.act(obs)


def collect_heldout(deck: list[int], engine: dict, games: int, seed: int,  # noqa: PLR0913
                    feats: object, decks: list, ctxs: object) -> tuple:
    """Fixed held-out (X, z): greedy_plus decisions labelled by its game result."""
    xs, zs = [], []
    for g in range(games):
        store: list = []
        gp = build_agent("greedy_plus", deck, engine)
        rec = _FeatRec(gp, store, feats, decks, ctxs)
        opp = build_agent("greedy", deck, engine)
        sf = g % 2 == 0
        p0, p1 = (rec, opp) if sf else (opp, rec)
        res = play_game(p0, p1, a_is_player0=sf, seed=9000 + seed + g)
        z = 1.0 if res.a_won else -1.0  # acting player = greedy_plus (its own outcome)
        for x, _ in store:
            xs.append(x)
            zs.append(z)
    return np.stack(xs), np.array(zs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="metal_aggro")
    ap.add_argument("--nets", required=True, help="comma-sep checkpoint .npz")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--out", default="results/rebel_valcurve.json")
    args = ap.parse_args()

    engine = load_engine_data()
    deck = resolve_deck(args.deck)
    belief = OpponentBelief.from_dirs()
    from src.net.features import CardFeatures  # noqa: PLC0415
    from src.rebel.value_net import ValueNet, build_hypothesis_ctxs  # noqa: PLC0415
    feats = CardFeatures(engine)
    decks = belief.hypotheses
    ctxs = build_hypothesis_ctxs(decks, feats)

    print(f"collecting held-out (x,z) from {args.games} games ...", flush=True)
    xs, zs = collect_heldout(deck, engine, args.games, 0, feats, decks, ctxs)
    print(f"  {len(zs)} samples, z mean={zs.mean():.3f} "
          f"(gp win-rate {(zs > 0).mean():.3f})", flush=True)

    rows = []
    for npath in (p for p in args.nets.split(",") if p):
        full = npath if Path(npath).is_absolute() else str(ROOT / npath)
        net = ValueNet.load(full)
        preds = np.array([net.value(x) for x in xs])
        mse = float(np.mean((preds - zs) ** 2))
        # Pearson r between prediction and outcome (higher = better ranking of states)
        pc = float(np.corrcoef(preds, zs)[0, 1]) if preds.std() > 0 else 0.0
        rows.append({"ckpt": Path(npath).stem, "val_mse": mse, "pearson_r": pc,
                     "pred_mean": float(preds.mean()), "pred_std": float(preds.std())})
        print(f"  {Path(npath).stem:<20} val_mse={mse:.4f} r={pc:+.3f} "
              f"pred_mean={preds.mean():+.3f} std={preds.std():.3f}", flush=True)

    Path(ROOT / args.out).write_text(json.dumps(
        {"n_samples": len(zs), "gp_winrate": float((zs > 0).mean()),
         "rows": rows}, indent=2))
    print("\n=== held-out value-MSE curve (lower MSE / higher r = net learns) ===")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
