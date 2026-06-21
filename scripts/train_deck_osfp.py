"""OSFP deck self-play orchestrator (native; shells Docker for the sim).

Learns the **CB (deck) head by self-play**: each iteration samples a batch of decks
from the current net, scores each by playing it against decks drawn from the net's
own distribution -- the learner itself (``self_play_prob``) or a past checkpoint
sampled from the recency-weighted :class:`~src.net.osfp.OpponentPool`. Both sides
use the same frozen play head, so the game is decided by the deck. The deck returns
become a REINFORCE advantage (``cb_rl_sequences``) and update **only** the CB head +
card embedding (``LitCBSeqPolicyGradient``; play/value frozen).

This replaces the Phase-5b-ii ``train_cb`` design, which scored decks against a
*fixed* reference deck -- so every learned deck lost and the advantage signal
vanished. Here the opponent is the net's own deck distribution, so the matchup is
~even by symmetry and the advantage is always informative (the self-play fix).

A periodic **yardstick** (read-only) tracks the greedy deck's win rate vs a fixed
reference (e.g. metal_aggro) to see whether self-play is making the deck stronger;
it does not feed training. Like the Phase-5a loop, ``run_deck_osfp`` takes injected
``generate`` / ``evaluate`` callables, so it is unit-tested natively with a
synthetic generator and no Docker.

  uv run python scripts/train_deck_osfp.py --smoke
  uv run python scripts/train_deck_osfp.py --iterations 40 --decks 24
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lightning as L  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.deck import build_pool  # noqa: E402
from src.net.bc_data import (  # noqa: E402
    CBSequenceDataset,
    cb_rl_sequences,
    collate_cb_seq,
    load_engine_json,
)
from src.net.features import CardFeatures  # noqa: E402
from src.net.lit import LitCBSeqPolicyGradient  # noqa: E402
from src.net.model import PolicyValueNet  # noqa: E402
from src.net.osfp import OpponentPool  # noqa: E402
from src.net.torch_model import from_numpy_net  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.deck import CardPool
    from src.net.osfp import PoolEntry

DOCKER_PREFIX = (
    "docker", "run", "--platform=linux/amd64", "--rm",
    "-v", f"{ROOT}:/work", "-w", "/work", "ptcg-sim", "python",
)


@dataclass
class DeckOsfpConfig:
    """Everything the deck self-play loop needs (one object, no long arg lists)."""

    init_weights: Path
    iter_dir: Path
    gate_deck: Path  # yardstick only (not training)
    iterations: int = 20
    decks_per_iter: int = 16
    games_per_deck: int = 16
    lr: float = 1e-3
    epochs: int = 1
    batch_size: int = 256
    eval_every: int = 5
    eval_games: int = 200
    seed: int = 0
    decay: float = 0.5
    self_play_prob: float = 0.5
    threshold: float = 0.55
    patience: int = 3


@dataclass
class DeckGenSpec:
    """One deck self-play collection request (``opp`` ``None`` => self-play)."""

    weights: Path
    opp: PoolEntry | None
    n_decks: int
    games_per_deck: int
    seed: int
    out: Path


@dataclass
class DeckEvalSpec:
    """One yardstick request: greedy deck vs the fixed gate deck."""

    weights: Path
    games: int
    seed: int
    out: Path


@dataclass
class DeckIterStat:
    iteration: int
    opponent: str
    n_decks: int
    n_samples: int
    mean_winrate: float
    gate: float | None
    admitted: bool


@dataclass
class DeckOsfpResult:
    final_weights: Path
    iterations: list[DeckIterStat]
    pool: OpponentPool


def _trainer(epochs: int) -> L.Trainer:
    return L.Trainer(
        max_epochs=epochs, accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False,
    )


def _deck_return(record: dict) -> float:
    """Deck score in [-1, 1]: (wins - losses) / decisive games."""
    dec = record["wins"] + record["losses"]
    return (record["wins"] - record["losses"]) / dec if dec else 0.0


def _mean_winrate(records: list[dict]) -> float:
    decisive = [r for r in records if r["wins"] + r["losses"] > 0]
    if not decisive:
        return 0.0
    return sum(r["wins"] / (r["wins"] + r["losses"]) for r in decisive) / len(decisive)


def _cb_train_step(
    net_np: PolicyValueNet,
    card_feats: object,
    samples: list,
    cfg: DeckOsfpConfig,
) -> PolicyValueNet:
    torch_net = from_numpy_net(net_np)
    lit = LitCBSeqPolicyGradient(torch_net, card_feats, lr=cfg.lr)
    loader = DataLoader(
        CBSequenceDataset(samples), batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_cb_seq,
    )
    _trainer(cfg.epochs).fit(lit, loader)
    return lit.net.double().to_numpy_net()


def _opp_label(opp: PoolEntry | None) -> str:
    return "self" if opp is None else Path(opp.ref).name


def run_deck_osfp(
    cfg: DeckOsfpConfig,
    pool: CardPool,
    feats: CardFeatures,
    *,
    generate: Callable[[DeckGenSpec], list[dict]],
    evaluate: Callable[[DeckEvalSpec], float] | None = None,
) -> DeckOsfpResult:
    """Run the deck self-play OSFP loop; ``generate`` / ``evaluate`` are injected."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    cfg.iter_dir.mkdir(parents=True, exist_ok=True)
    # No scripted-deck baselines: opponents are the learner (self) or CB checkpoints.
    history = OpponentPool(
        [], decay=cfg.decay, self_play_prob=cfg.self_play_prob,
        threshold=cfg.threshold, patience=cfg.patience,
    )
    net_np = PolicyValueNet.load(cfg.init_weights)

    stats: list[DeckIterStat] = []
    for n in range(1, cfg.iterations + 1):
        weights_path = cfg.iter_dir / f"deckiter_{n}.npz"
        net_np.save(weights_path)  # save BEFORE collecting -- the collector loads this

        opp = history.sample(n, rng)
        spec = DeckGenSpec(
            weights=weights_path, opp=opp, n_decks=cfg.decks_per_iter,
            games_per_deck=cfg.games_per_deck, seed=cfg.seed + n * 1000,
            out=cfg.iter_dir / f"decks_{n}",
        )
        records = generate(spec)
        scored = [(r["deck"], _deck_return(r)) for r in records
                  if r["wins"] + r["losses"] > 0]
        card_feats, samples = cb_rl_sequences(scored, pool, feats, normalize=True)
        if samples:
            net_np = _cb_train_step(net_np, card_feats, samples, cfg)

        gate: float | None = None
        if evaluate is not None and n % cfg.eval_every == 0:
            gate = evaluate(DeckEvalSpec(
                weights=weights_path, games=cfg.eval_games, seed=cfg.seed,
                out=cfg.iter_dir / f"gate_{n}",
            ))
        # Admit on the yardstick (beats the reference) or patience -- keeps the
        # opponent pool refreshing even while self-play win rate hovers at ~0.5.
        admitted = history.admit(
            str(weights_path), n, {"gate": gate} if gate is not None else {},
        )

        label = _opp_label(opp)
        mean_wr = _mean_winrate(records)
        stats.append(DeckIterStat(
            n, label, len(records), len(samples), mean_wr, gate, admitted,
        ))
        print(
            f"[deckiter {n}] opp={label} decks={len(records)} samples={len(samples)} "
            f"mean_wr={mean_wr:.3f} gate={gate} admitted={admitted} "
            f"ckpts={history.num_checkpoints}",
        )

    final = cfg.iter_dir / "deck_final.npz"
    net_np.save(final)
    run_history = {
        "config": {
            "iterations": cfg.iterations, "decks_per_iter": cfg.decks_per_iter,
            "games_per_deck": cfg.games_per_deck, "lr": cfg.lr, "seed": cfg.seed,
            "self_play_prob": cfg.self_play_prob, "init_weights": str(cfg.init_weights),
            "gate_deck": str(cfg.gate_deck),
        },
        "final_weights": str(final),
        "checkpoints": history.num_checkpoints,
        "iterations_log": [asdict(s) for s in stats],
    }
    (cfg.iter_dir / "history.json").write_text(json.dumps(run_history, indent=2))
    print(f"== done: {final}  checkpoints={history.num_checkpoints} ==")
    return DeckOsfpResult(final, stats, history)


# --- production wiring (shells Docker) -------------------------------------


def _in_container(path: Path) -> str:
    return f"/work/{path.resolve().relative_to(ROOT)}"


def _run(argv: list[str], what: str) -> None:
    try:
        subprocess.run(argv, check=True, capture_output=True, text=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "")[-2000:]
        msg = f"docker {what} failed (exit {exc.returncode}): {tail}"
        raise RuntimeError(msg) from exc


def _read_records(out: Path) -> list[dict]:
    shard = next((out / "decks").glob("*.jsonl"))
    return [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]


_SEED_STRIDE = 1_000_000  # space shard seeds far apart so they sample different decks


def _split(total: int, parts: int) -> list[int]:
    """Split ``total`` into ``parts`` near-even chunks (some may be 0)."""
    base, rem = divmod(total, max(parts, 1))
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _make_docker_collect(workers: int) -> Callable[[DeckGenSpec], list[dict]]:
    """Build a ``generate`` that fans the deck batch across ``workers`` containers.

    The engine is single-threaded, so parallelism is N independent ``docker run``
    processes. Shards are seeded far apart (different decks) and a failed shard is
    skipped (its decks are lost for that iteration) so one flaky container can't
    abort a long unattended run.

    MEASURED (2026-06-22): on this Mac's **x86-emulated** Docker, workers > 1 is
    *slower* (w1=9.6 g/s, w2=4.1, w6=3.8 on 15 cores) -- the emulation layer
    contends, one container already saturates it. So ``--workers`` defaults to 1
    here; > 1 pays off only on **native x86** (a Linux box / cloud), where there is
    no emulation and the processes parallelise.
    """
    def _shard(spec: DeckGenSpec, k: int, n_decks: int) -> list[dict]:
        out = spec.out / f"w{k}"
        argv = [
            *DOCKER_PREFIX, "scripts/collect_deck_selfplay.py",
            "--weights", _in_container(spec.weights),
            "--decks", str(n_decks), "--games-per-deck", str(spec.games_per_deck),
            "--seed", str(spec.seed + k * _SEED_STRIDE), "--out", _in_container(out),
        ]
        if spec.opp is None:
            argv.append("--self-play")
        else:
            argv += ["--opp-weights", _in_container(Path(spec.opp.ref))]
        try:
            _run(argv, f"collect_deck w{k}")
            return _read_records(out)
        except (RuntimeError, StopIteration) as exc:
            print(f"  [warn] shard {k} failed, skipping its decks: {exc}")
            return []

    def collect(spec: DeckGenSpec) -> list[dict]:
        sizes = [(k, n) for k, n in enumerate(_split(spec.n_decks, workers)) if n > 0]
        if len(sizes) <= 1:  # nothing to parallelise
            return _shard(spec, *sizes[0]) if sizes else []
        with ThreadPoolExecutor(max_workers=len(sizes)) as ex:
            shards = ex.map(lambda kn: _shard(spec, kn[0], kn[1]), sizes)
        return [record for shard in shards for record in shard]

    return collect


def _make_docker_gate(gate_deck: Path) -> Callable[[DeckEvalSpec], float]:
    def evaluate(spec: DeckEvalSpec) -> float:
        argv = [
            *DOCKER_PREFIX, "scripts/collect_deck_selfplay.py",
            "--weights", _in_container(spec.weights),
            "--gate-deck", _in_container(gate_deck),
            "--games-per-deck", str(spec.games), "--seed", str(spec.seed),
            "--out", _in_container(spec.out),
        ]
        _run(argv, "deck gate")
        records = _read_records(spec.out)
        return _deck_return(records[0]) / 2 + 0.5 if records else 0.0  # -> win rate

    return evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="OSFP deck self-play training")
    parser.add_argument(
        "--weights", type=Path, default=ROOT / "data/bc/bc_net_lstm.npz",
    )
    parser.add_argument(
        "--engine-json", type=Path, default=ROOT / "data/bc/engine.json",
    )
    parser.add_argument(
        "--gate-deck", type=Path, default=ROOT / "decklists/metal_aggro.csv",
        help="yardstick reference deck (eval only, not training)",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data/deckosfp")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--decks", type=int, default=16)
    parser.add_argument("--games-per-deck", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--self-play-prob", type=float, default=0.5)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="parallel collector containers; >1 ONLY helps on native x86 (under "
             "this Mac's x86-emulated Docker, extra containers contend and slow down)",
    )
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="3 tiny iterations")
    args = parser.parse_args()

    if args.smoke:
        args.iterations, args.decks, args.games_per_deck, args.eval_games = 3, 4, 4, 20

    cfg = DeckOsfpConfig(
        init_weights=args.weights, iter_dir=args.out, gate_deck=args.gate_deck,
        iterations=args.iterations, decks_per_iter=args.decks,
        games_per_deck=args.games_per_deck, lr=args.lr,
        self_play_prob=args.self_play_prob, eval_every=args.eval_every,
        eval_games=args.eval_games, seed=args.seed,
    )
    feats = CardFeatures(load_engine_json(args.engine_json))
    pool = build_pool()
    evaluate = None if args.no_eval else _make_docker_gate(args.gate_deck)
    print(f"deck self-play: {args.workers} parallel collector(s)")
    run_deck_osfp(
        cfg, pool, feats,
        generate=_make_docker_collect(args.workers), evaluate=evaluate,
    )


if __name__ == "__main__":
    main()
