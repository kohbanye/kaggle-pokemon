"""Train the Phase-4 BC warm-start net (host / native; no engine needed).

Reads the teacher logs + engine dump from ``scripts/collect_bc.py``, clones the
policy head and regresses the value head (:class:`LitPolicyValue`), then clones
the CB head on the demo decklists (:class:`LitCB`), and exports the trained
weights to a single numpy ``.npz`` the submission / ``NetAgent`` loads. Train in
torch + Lightning, serve in numpy (plan SS D). Prints held-out policy top-1
accuracy, value sign accuracy, and a CB legality / demo-overlap sanity check.

  uv run python scripts/train_bc.py --data data/bc --out data/bc/bc_net.npz
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # make `src` importable when run as a script

import lightning as L  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.deck import build_pool, is_legal, load_deck_csv  # noqa: E402
from src.net.bc_data import (  # noqa: E402
    CBSequenceDataset,
    PolicyDataset,
    PolicySample,
    build_policy_samples,
    cb_sequences,
    collate_cb_seq,
    collate_policy,
    game_files,
    iter_games,
    load_engine_json,
)
from src.net.cb import build_deck  # noqa: E402
from src.net.features import CardFeatures  # noqa: E402
from src.net.lit import LitCBSeq, LitPolicyValue  # noqa: E402
from src.net.model import NetConfig  # noqa: E402


def _trainer(epochs: int) -> L.Trainer:
    return L.Trainer(
        max_epochs=epochs, accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False,
    )


def evaluate(lit: LitPolicyValue, samples: list[PolicySample]) -> tuple[float, float]:
    """Held-out policy top-1 accuracy and value sign accuracy (on decisive games)."""
    if not samples:
        return 0.0, 0.0
    states, options, mask, targets, values = collate_policy(samples)
    with torch.no_grad():
        logits = lit.net.policy_logits(states, options)
        logits = logits.masked_fill(~mask, float("-inf"))
        pol_acc = (logits.argmax(dim=1) == targets).float().mean().item()
        vpred = lit.net.value(states)
    decisive = values != 0
    if decisive.any():
        agree = (vpred[decisive] > 0) == (values[decisive] > 0)
        val_acc = agree.float().mean().item()
    else:
        val_acc = 0.0
    return pol_acc, val_acc


def _overlap(deck: list[int], demo: list[int]) -> float:
    """Multiset overlap fraction |deck ∩ demo| / len(deck) between two decks."""
    inter = sum((Counter(deck) & Counter(demo)).values())
    return inter / max(len(deck), 1)


def main() -> None:  # noqa: PLR0915 - a training CLI legitimately threads its config
    parser = argparse.ArgumentParser(description="Train the Phase-4 BC net")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "bc")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "bc" / "bc_net.npz")
    parser.add_argument("--decks", type=Path, default=ROOT / "decklists")
    parser.add_argument("--teachers", default="heuristic", help="comma-separated")
    parser.add_argument(
        "--discount", type=float, default=None,
        help="discounted-return gamma; omit for the raw final outcome",
    )
    parser.add_argument("--epochs", type=int, default=20)
    # The LSTM deck head needs more CB BC than the old flat head to learn a balanced
    # composition (energy/attacker ratio); under-training over-picks one type.
    parser.add_argument("--cb-epochs", type=int, default=150)
    parser.add_argument("--cb-shuffles", type=int, default=12)
    parser.add_argument("--embed-dim", type=int, default=16, help="CB card embedding")
    parser.add_argument("--lstm-hidden", type=int, default=64, help="deck LSTM hidden")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    engine = load_engine_json(args.data / "engine.json")
    feats = CardFeatures(engine)
    teachers = {t for t in args.teachers.split(",") if t}

    games = list(iter_games(game_files(args.data)))
    samples = build_policy_samples(
        games, feats, teachers=teachers, discount=args.discount,
    )
    if not samples:
        raise SystemExit(f"no policy samples for teachers={teachers} under {args.data}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(samples)).tolist()
    n_val = int(len(samples) * args.val_frac)
    val_set = {perm[i] for i in range(n_val)}
    train = [s for i, s in enumerate(samples) if i not in val_set]
    val = [samples[i] for i in perm[:n_val]]
    print(
        f"games={len(games)} policy samples={len(samples)} "
        f"(train {len(train)}, val {len(val)}) teachers={teachers} "
        f"discount={args.discount}",
    )

    torch.manual_seed(args.seed)
    pool = build_pool()  # built early: the net's card embedding is sized to the pool
    config = NetConfig(
        n_cards=len(pool.ids()), embed_dim=args.embed_dim,
        lstm_hidden=args.lstm_hidden,
    )
    lit = LitPolicyValue(config, lr=args.lr)
    loader = DataLoader(
        PolicyDataset(train), batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_policy,
    )
    _trainer(args.epochs).fit(lit, loader)
    pol_acc, val_acc = evaluate(lit, val)
    print(f"val policy top-1 acc: {pol_acc:.3f}  value sign acc: {val_acc:.3f}")

    # CB head: behaviour-clone the demo decklists with the autoregressive LSTM deck
    # head (cb_lstm/cb_start/cb1/cb2/cb_embed update; trunk/policy/value frozen).
    demo_decks = [load_deck_csv(p) for p in sorted(args.decks.glob("*.csv"))]
    card_feats, cb_seqs = cb_sequences(
        demo_decks, pool, feats, np.random.default_rng(args.seed),
        shuffles=args.cb_shuffles,
    )
    print(f"CB sequences: {len(cb_seqs)} from {len(demo_decks)} demo decks")
    litcb = LitCBSeq(lit.net, card_feats, lr=args.lr)
    cb_loader = DataLoader(
        CBSequenceDataset(cb_seqs), batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_cb_seq,
    )
    _trainer(args.cb_epochs).fit(litcb, cb_loader)

    net_np = lit.net.double().to_numpy_net()
    deck = build_deck(net_np, pool, feats)
    best = max((_overlap(deck, d) for d in demo_decks), default=0.0)
    n_energy = sum(1 for c in deck if pool.cards[c].is_basic_energy)
    n_poke = sum(1 for c in deck if pool.cards[c].supertype == "Pokemon")
    print(
        f"CB greedy deck: legal={is_legal(deck, pool)} "
        f"best_overlap_vs_demo={best:.2f} distinct={len(set(deck))} "
        f"energy={n_energy} pokemon={n_poke} (functional if energy in ~[15,35])",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    net_np.save(args.out)
    print(f"saved {args.out}  params={net_np.param_count()}")


if __name__ == "__main__":
    main()
