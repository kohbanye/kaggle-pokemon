"""Does HISTORY help value prediction? LSTM vs scalar MLP on the SAME supervised data.

Sidesteps the self-play collapse (fresh LSTM death-spirals) AND the depth-3 win-rate
ceiling: collect BALANCED (feature-sequence, game-outcome z) from greedy_plus self-play
(~50/50 outcomes), split train/held-out, then fit BOTH an LSTM (sees the whole history)
and a scalar MLP (sees only the LAST state) on identical data and compare held-out MSE +
Pearson r. If the LSTM predicts outcomes better, self+opponent HISTORY genuinely helps.

  uv run python scripts/rebel_histcompare.py --games 120 --hidden 128
"""

from __future__ import annotations

import argparse
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


class _SeqRec(Agent):
    """Wrap greedy_plus; record the growing feature SEQUENCE per decision."""

    name = "seqrec"

    def __init__(self, inner: Agent, store: list, feats: object, decks: list,
                 ctxs: object) -> None:
        super().__init__(inner.deck)
        self.inner, self.store = inner, store
        self._feats, self._decks, self._ctxs = feats, decks, ctxs
        self._seq: list = []

    def reset(self, seed: int) -> None:
        self.inner.reset(seed)
        self._seq = []

    def act(self, obs: dict) -> list[int]:
        from src.rebel.value_net import state_features  # noqa: PLC0415
        cur = obs.get("current")
        select = obs.get("select") or {}
        if cur is not None and int(select.get("maxCount", 0)) == SINGLE_SELECT \
                and len(select.get("option") or []) >= _MIN_CHOICE:
            your = int(cur.get("yourIndex", 0))
            x = state_features(cur, your, self._feats, self._decks, self._ctxs)
            self._seq.append(x)
            self.store.append([np.stack(self._seq[-24:]), None])  # last<=24 steps
        return self.inner.act(obs)


def collect(deck: list[int], engine: dict, games: int, feats: object,  # noqa: PLR0913
            decks: list, ctxs: object) -> list:
    out: list = []
    for g in range(games):
        store: list = []
        gp = build_agent("greedy_plus", deck, engine)
        rec = _SeqRec(gp, store, feats, decks, ctxs)
        opp = build_agent("greedy", deck, engine)
        sf = g % 2 == 0
        p0, p1 = (rec, opp) if sf else (opp, rec)
        res = play_game(p0, p1, a_is_player0=sf, seed=8000 + g)
        z = 1.0 if res.a_won else -1.0
        for s in store:
            s[1] = z
            out.append((s[0], z))
    return out


def _pad(seqs: list, in_dim: int) -> tuple:
    import torch  # noqa: PLC0415
    lengths = torch.tensor([len(s) for s in seqs])
    tmax = int(lengths.max())
    xb = np.zeros((len(seqs), tmax, in_dim))
    for i, s in enumerate(seqs):
        xb[i, : len(s)] = s
    return torch.tensor(xb), lengths


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="metal_aggro")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save", default=None,
                    help="save the supervised-trained LSTM numpy net to this .npz")
    args = ap.parse_args()

    import torch  # noqa: PLC0415

    from src.net.features import CardFeatures  # noqa: PLC0415
    from src.rebel.recurrent_value_net_torch import (  # noqa: PLC0415
        TorchRecurrentValueNet,
    )
    from src.rebel.value_net import (  # noqa: PLC0415
        VALUE_IN_DIM,
        build_hypothesis_ctxs,
    )
    from src.rebel.value_net_torch import TorchValueNet  # noqa: PLC0415

    engine = load_engine_data()
    deck = resolve_deck(args.deck)
    belief = OpponentBelief.from_dirs()
    feats = CardFeatures(engine)
    decks = belief.hypotheses
    ctxs = build_hypothesis_ctxs(decks, feats)

    print(f"collecting {args.games} greedy_plus games ...", flush=True)
    data = collect(deck, engine, args.games, feats, decks, ctxs)
    rng = np.random.default_rng(0)
    rng.shuffle(data)
    n_test = max(1, len(data) // 5)
    test, train = data[:n_test], data[n_test:]
    zc = np.array([z for _, z in train])
    print(f"  {len(data)} samples ({len(train)} train / {len(test)} test), "
          f"train z-mean={zc.mean():.3f} (gp wr {(zc > 0).mean():.2f})", flush=True)

    ytr = torch.tensor(np.array([z for _, z in train]))
    yte = torch.tensor(np.array([z for _, z in test]))

    def _fit(net: object, xtr: object, xte: object, *, seq: bool) -> tuple:
        opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
        ltr = xtr[1] if seq else None
        lte = xte[1] if seq else None
        for _ in range(args.epochs):
            opt.zero_grad()
            pred = net(xtr[0], ltr) if seq else net(xtr)
            loss = ((pred - ytr) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pte = (net(xte[0], lte) if seq else net(xte)).numpy()
        z = yte.numpy()
        mse = float(np.mean((pte - z) ** 2))
        r = float(np.corrcoef(pte, z)[0, 1]) if pte.std() > 0 else 0.0
        return mse, r

    # LSTM (history) on padded sequences
    xtr_s = _pad([s for s, _ in train], VALUE_IN_DIM)
    xte_s = _pad([s for s, _ in test], VALUE_IN_DIM)
    lstm = TorchRecurrentValueNet(VALUE_IN_DIM, args.hidden).double()
    mse_l, r_l = _fit(lstm, xtr_s, xte_s, seq=True)

    # scalar MLP (no history) on the LAST state of each sequence
    xtr_m = torch.tensor(np.stack([s[-1] for s, _ in train]))
    xte_m = torch.tensor(np.stack([s[-1] for s, _ in test]))
    mlp = TorchValueNet(VALUE_IN_DIM, (args.hidden,)).double()
    mse_m, r_m = _fit(mlp, xtr_m, xte_m, seq=False)

    print("\n=== HISTORY vs NO-HISTORY (held-out) ===")
    print(f"  LSTM (history)   : MSE={mse_l:.4f}  r={r_l:+.3f}")
    print(f"  MLP  (last state): MSE={mse_m:.4f}  r={r_m:+.3f}")
    verdict = ("HISTORY HELPS" if mse_l < mse_m - 0.005 else
               "no clear gain from history")
    print(f"  -> {verdict}")
    if args.save:
        lstm.to_numpy_net().save(args.save)
        print(f"  saved supervised LSTM -> {args.save}")


if __name__ == "__main__":
    main()
