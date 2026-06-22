"""Paper-faithful OSFP actor-learner loop (V-Trace + PPO + recurrent net).

The orchestrator for the rewrite: each iteration the actors collect recurrent
self-play **trajectories** (``collect_paper_selfplay``, native subprocesses), the
episodes feed a bounded **FIFO queue** (the IMPALA-style replay buffer; V-Trace
corrects the staleness of the older episodes still in it), and the learner does one
:class:`~src.net.lit_vtrace.LitVtracePPO` update over a sample of the queue. The
opponent each iteration is drawn from the OSFP :class:`~src.net.osfp.OpponentPool`
(self-play with probability ``self_play_prob``, else a recency-weighted past
checkpoint); strong checkpoints are admitted back into the pool.

This replaces ``train_joint_osfp`` (plain REINFORCE, memoryless, synchronous). The
collector / FIFO / V-Trace together are the paper's actor-learner; the OpponentPool
is the OSFP meta-loop (last-iterate -> submit the final checkpoint directly).

  uv run python scripts/train_paper_osfp.py --smoke --native
  uv run python scripts/train_paper_osfp.py --native --workers 14 --iterations 500
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lightning as L  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.deck import build_pool  # noqa: E402
from src.net.bc_data import load_engine_json  # noqa: E402
from src.net.embedding import CardEmbeddingIndex  # noqa: E402
from src.net.features import CardFeatures  # noqa: E402
from src.net.lit_vtrace import LitVtracePPO  # noqa: E402
from src.net.osfp import OpponentPool  # noqa: E402
from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: E402
from src.net.recurrent_torch import TorchRecurrentNet  # noqa: E402
from src.net.trajectory_data import (  # noqa: E402
    EpisodeDataset,
    build_episodes,
    collate_episodes,
)

_SCRIPT = "scripts/collect_paper_selfplay.py"


@dataclass
class Config:
    """The loop's knobs (one object, no long arg lists)."""

    init_weights: Path
    out_dir: Path
    engine_json: Path
    gate_deck: Path
    iterations: int = 50
    games_per_iter: int = 64
    temperature: float = 1.0
    queue_episodes: int = 4096  # FIFO replay capacity (episodes)
    train_episodes: int = 512  # episodes sampled from the queue per update
    batch_size: int = 32
    epochs: int = 1
    lr: float = 1e-3
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    deck_entropy_coef: float = 0.01
    clip_eps: float = 0.2
    clip_rho: float = 1.0
    clip_c: float = 1.0
    rho_min: float = 0.05  # the paper's rho lower-clip (improved technique)
    workers: int = 1
    native: bool = False
    eval_every: int = 10
    eval_games: int = 200
    seed: int = 0
    self_play_prob: float = 0.5
    decay: float = 0.5
    threshold: float = 0.55
    patience: int = 3


def _trainer(epochs: int) -> L.Trainer:
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    return L.Trainer(
        max_epochs=epochs, accelerator=accelerator, devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False,
    )


def _argv(cfg: Config, args: list[str]) -> list[str]:
    if cfg.native:
        return [sys.executable, str(ROOT / _SCRIPT), *args]
    return [
        "docker", "run", "--platform=linux/amd64", "--rm",
        "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
        "-e", "MKL_NUM_THREADS=1", "-v", f"{ROOT}:/work", "-w", "/work",
        "ptcg-sim", "python", _SCRIPT, *args,
    ]


def _path(cfg: Config, p: Path) -> str:
    return str(p.resolve()) if cfg.native else f"/work/{p.resolve().relative_to(ROOT)}"


def _env(cfg: Config) -> dict[str, str] | None:
    if not cfg.native:
        return None
    return {
        **os.environ, "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }


def _run(argv: list[str], env: dict[str, str] | None) -> None:
    try:
        subprocess.run(argv, check=True, capture_output=True, text=True, env=env)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        msg = f"collector failed (exit {exc.returncode}): {(exc.stderr or '')[-2000:]}"
        raise RuntimeError(msg) from exc


def _read_records(out: Path) -> list[dict]:
    shard = next((out / "games").glob("*.jsonl"))
    return [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]


def _split(total: int, parts: int) -> list[int]:
    base, rem = divmod(total, max(parts, 1))
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _collect(  # noqa: PLR0913 - one collection request's parameters
    cfg: Config,
    weights: Path,
    opp: Path | None,
    games: int,
    seed: int,
    out: Path,
) -> list[dict]:
    """Fan ``games`` across ``cfg.workers`` collector processes; gather their lines."""
    def shard(k: int, n: int) -> list[dict]:
        sub = out / f"w{k}"
        args = [
            "--weights", _path(cfg, weights), "--games", str(n),
            "--temperature", str(cfg.temperature),
            "--seed", str(seed + k * 1_000_000), "--out", _path(cfg, sub),
        ]
        args += ["--self-play"] if opp is None else ["--opp-weights", _path(cfg, opp)]
        try:
            _run(_argv(cfg, args), _env(cfg))
            return _read_records(sub)
        except (RuntimeError, StopIteration) as exc:
            print(f"  [warn] shard {k} failed: {exc}")
            return []

    sizes = [(k, n) for k, n in enumerate(_split(games, cfg.workers)) if n > 0]
    if len(sizes) <= 1:
        return shard(*sizes[0]) if sizes else []
    with ThreadPoolExecutor(max_workers=len(sizes)) as ex:
        shards = ex.map(lambda kn: shard(*kn), sizes)
    return [rec for s in shards for rec in s]


def _gate(cfg: Config, weights: Path, out: Path) -> float | None:
    """Greedy deck vs the fixed gate deck -> win rate (read-only yardstick)."""
    args = [
        "--weights", _path(cfg, weights), "--gate-deck", _path(cfg, cfg.gate_deck),
        "--games", str(cfg.eval_games), "--seed", str(cfg.seed),
        "--out", _path(cfg, out),
    ]
    _run(_argv(cfg, args), _env(cfg))
    gate = next((r for r in _read_records(out) if r.get("type") == "gate"), None)
    if gate is None:
        return None
    dec = gate["wins"] + gate["losses"]
    return gate["wins"] / dec if dec else 0.0


def _update(
    cfg: Config,
    net_np: RecurrentPolicyValueNet,
    episodes: list,
    card_feats: object,
) -> RecurrentPolicyValueNet:
    """One V-Trace/PPO update over a sample of the FIFO queue; return the new net."""
    if not episodes:
        return net_np
    torch_net = TorchRecurrentNet(net_np.config)
    torch_net.load_numpy_params(net_np.params)
    lit = LitVtracePPO(
        torch_net, card_feats, lr=cfg.lr, value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef, deck_entropy_coef=cfg.deck_entropy_coef,
        clip_eps=cfg.clip_eps, clip_rho=cfg.clip_rho, clip_c=cfg.clip_c,
        rho_min=cfg.rho_min,
    )
    loader = DataLoader(
        EpisodeDataset(episodes), batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_episodes,
    )
    _trainer(cfg.epochs).fit(lit, loader)
    return lit.net.double().to_numpy_net()


def run(cfg: Config) -> None:
    """Run the paper-faithful OSFP actor-learner loop."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    pool = build_pool()
    feats = CardFeatures(load_engine_json(cfg.engine_json))
    card_feats = CardEmbeddingIndex(pool).fixed_matrix(feats)
    index = CardEmbeddingIndex(pool)

    history = OpponentPool(
        [], decay=cfg.decay, self_play_prob=cfg.self_play_prob,
        threshold=cfg.threshold, patience=cfg.patience,
    )
    net_np = RecurrentPolicyValueNet.load(cfg.init_weights)
    queue: deque = deque(maxlen=cfg.queue_episodes)

    for n in range(1, cfg.iterations + 1):
        weights_path = cfg.out_dir / f"paperiter_{n}.npz"
        net_np.save(weights_path)  # the collector loads this

        opp_entry = history.sample(n, rng)
        opp = Path(opp_entry.ref) if opp_entry is not None else None
        out = cfg.out_dir / f"games_{n}"
        records = _collect(
            cfg, weights_path, opp, cfg.games_per_iter, cfg.seed + n * 1000, out,
        )
        episodes = build_episodes(records, feats, index, pool)
        queue.extend(episodes)

        sample = list(queue)
        if len(sample) > cfg.train_episodes:
            pick = rng.choice(len(sample), size=cfg.train_episodes, replace=False)
            sample = [sample[i] for i in pick]
        net_np = _update(cfg, net_np, sample, card_feats)

        gate = None
        if n % cfg.eval_every == 0:
            gate = _gate(cfg, weights_path, cfg.out_dir / f"gate_{n}")
        admitted = history.admit(
            str(weights_path), n, {"gate": gate} if gate is not None else {},
        )
        label = "self" if opp is None else opp.name
        print(
            f"[paperiter {n}] opp={label} new_eps={len(episodes)} "
            f"queue={len(queue)} trained={len(sample)} gate={gate} "
            f"admitted={admitted} ckpts={history.num_checkpoints}",
        )
        shutil.rmtree(out, ignore_errors=True)
        shutil.rmtree(cfg.out_dir / f"gate_{n}", ignore_errors=True)

    final = cfg.out_dir / "paper_final.npz"
    net_np.save(final)
    print(f"== done: {final} checkpoints={history.num_checkpoints} ==")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-faithful OSFP training")
    parser.add_argument("--weights", type=Path, default=ROOT / "data/paper/init.npz")
    parser.add_argument(
        "--engine-json", type=Path, default=ROOT / "data/bc/engine.json",
    )
    parser.add_argument(
        "--gate-deck", type=Path, default=ROOT / "decklists/metal_aggro.csv",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data/paperosfp/run1")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games-per-iter", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-play-prob", type=float, default=0.5)
    parser.add_argument("--smoke", action="store_true", help="3 tiny iterations")
    args = parser.parse_args()

    cfg = Config(
        init_weights=args.weights, out_dir=args.out, engine_json=args.engine_json,
        gate_deck=args.gate_deck, iterations=args.iterations,
        games_per_iter=args.games_per_iter, temperature=args.temperature, lr=args.lr,
        workers=args.workers, native=args.native, eval_every=args.eval_every,
        eval_games=args.eval_games, seed=args.seed, self_play_prob=args.self_play_prob,
    )
    if args.smoke:
        cfg.iterations, cfg.games_per_iter = 3, 4
        cfg.train_episodes, cfg.batch_size = 16, 8
        cfg.eval_every, cfg.eval_games = 2, 20
    run(cfg)


if __name__ == "__main__":
    main()
