"""Joint OSFP self-play orchestrator (Phase 5d): πBT + πCB, shared embedding.

Mirrors the ByteDance Hearthstone paper: each iteration plays self-play games where
the **deck** is sampled from the CB head (πCB) and the **battle** is played by the
shared play head (πBT), then does ONE update that improves the play, value and deck
heads -- and the **shared card embedding both heads read** -- together
(:class:`~src.net.lit.LitJointPolicyGradient`). This replaces the two separate
loops (play-only ``train_osfp`` and deck-only ``train_deck_osfp``); the embedding is
now genuinely shared rather than trained by one head with the other frozen.

From one batch of games the collector emits both kinds of samples (see
``scripts/collect_joint_selfplay.py``): per-game ``"game"`` lines -> play samples
(:func:`~src.net.bc_data.build_policy_samples`), per-deck ``"deck"`` lines -> deck
sequences (:func:`~src.net.bc_data.cb_rl_sequences`). The two dataloaders are zipped
with a :class:`CombinedLoader` so a single ``trainer.fit`` applies both losses.

A read-only **gate** yardstick (greedy deck vs ``metal_aggro``, both sides using the
trained play head) tracks whether the agent is getting stronger -- it now reflects
*both* a better deck and a better play head. Collection shells out per batch (Docker
or, with ``--native``, a plain subprocess), parallelised across ``--workers`` with
BLAS pinned to one thread each (see :class:`_Collector`).

  uv run python scripts/train_joint_osfp.py --smoke --native
  uv run python scripts/train_joint_osfp.py --native --workers 14 --iterations 200
"""

from __future__ import annotations

import argparse
import json
import os
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
from lightning.pytorch.utilities.combined_loader import CombinedLoader  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.deck import build_pool  # noqa: E402
from src.net.bc_data import (  # noqa: E402
    CBSequenceDataset,
    PolicyDataset,
    build_policy_samples,
    cb_rl_sequences,
    collate_cb_seq,
    collate_policy,
    load_engine_json,
)
from src.net.embedding import CardEmbeddingIndex  # noqa: E402
from src.net.features import CardFeatures  # noqa: E402
from src.net.lit import LitJointPolicyGradient  # noqa: E402
from src.net.model import PolicyValueNet  # noqa: E402
from src.net.osfp import OpponentPool  # noqa: E402
from src.net.torch_model import from_numpy_net  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.deck import CardPool
    from src.net.osfp import PoolEntry

LEARNER = "learner"  # the agent tag the play arm trains on (see collect_joint_selfplay)

DOCKER_PREFIX = (
    "docker", "run", "--platform=linux/amd64", "--rm",
    # Pin each container's numpy/BLAS to a single thread (see _Collector): without
    # this, N parallel collectors each spawn ncores BLAS threads and contend.
    "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
    "-e", "MKL_NUM_THREADS=1",
    "-v", f"{ROOT}:/work", "-w", "/work", "ptcg-sim", "python",
)


@dataclass
class JointOsfpConfig:
    """Everything the joint self-play loop needs (one object, no long arg lists)."""

    init_weights: Path
    iter_dir: Path
    gate_deck: Path  # yardstick only (not training)
    iterations: int = 20
    decks_per_iter: int = 16
    games_per_deck: int = 16
    temperature: float = 1.0
    lr: float = 1e-3
    epochs: int = 1
    batch_size: int = 256
    max_play_samples: int = 20000  # subsample play decisions per iter (see build_...)
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    eval_every: int = 5
    eval_games: int = 200
    seed: int = 0
    decay: float = 0.5
    self_play_prob: float = 0.5
    threshold: float = 0.55
    patience: int = 3


@dataclass
class JointGenSpec:
    """One joint self-play collection request (``opp`` ``None`` => self-play)."""

    weights: Path
    opp: PoolEntry | None
    n_decks: int
    games_per_deck: int
    temperature: float
    seed: int
    out: Path
    max_play_samples: int = 20000


@dataclass
class JointEvalSpec:
    """One yardstick request: greedy deck vs the fixed gate deck."""

    weights: Path
    games: int
    seed: int
    out: Path


@dataclass
class JointIterStat:
    iteration: int
    opponent: str
    n_games: int
    n_decks: int
    n_play_samples: int
    n_deck_samples: int
    mean_winrate: float
    gate: float | None
    admitted: bool


@dataclass
class JointOsfpResult:
    final_weights: Path
    iterations: list[JointIterStat]
    pool: OpponentPool


def _trainer(epochs: int) -> L.Trainer:
    # Use the GPU for the joint update when one is present (the collectors stay CPU
    # numpy in their workers); fall back to CPU otherwise.
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    return L.Trainer(
        max_epochs=epochs, accelerator=accelerator, devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False,
    )


def _deck_return(record: dict) -> float:
    """Deck score in [-1, 1]: (wins - losses) / decisive games."""
    dec = record["wins"] + record["losses"]
    return (record["wins"] - record["losses"]) / dec if dec else 0.0


def _mean_winrate(decks: list[dict]) -> float:
    decisive = [r for r in decks if r["wins"] + r["losses"] > 0]
    if not decisive:
        return 0.0
    return sum(r["wins"] / (r["wins"] + r["losses"]) for r in decisive) / len(decisive)


def _split_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition the mixed shard lines into (game records, deck records)."""
    games = [r for r in records if r.get("type") == "game"]
    decks = [r for r in records if r.get("type") == "deck"]
    return games, decks


def _joint_train_step(
    net_np: PolicyValueNet,
    play_samples: list,
    deck_card_feats: object,
    deck_samples: list,
    cfg: JointOsfpConfig,
) -> PolicyValueNet:
    """One joint update over both arms via a CombinedLoader (no head frozen)."""
    torch_net = from_numpy_net(net_np)
    lit = LitJointPolicyGradient(
        torch_net, deck_card_feats, lr=cfg.lr,
        value_coef=cfg.value_coef, entropy_coef=cfg.entropy_coef,
    )
    loaders: dict[str, DataLoader] = {}
    if play_samples:
        loaders["play"] = DataLoader(
            PolicyDataset(play_samples), batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate_policy,
        )
    if deck_samples:
        loaders["deck"] = DataLoader(
            CBSequenceDataset(deck_samples), batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate_cb_seq,
        )
    if not loaders:
        return net_np
    combined = CombinedLoader(loaders, mode="max_size_cycle")
    _trainer(cfg.epochs).fit(lit, combined)
    return lit.net.double().to_numpy_net()


def _opp_label(opp: PoolEntry | None) -> str:
    return "self" if opp is None else Path(opp.ref).name


def run_joint_osfp(
    cfg: JointOsfpConfig,
    pool: CardPool,
    feats: CardFeatures,
    *,
    generate: Callable[[JointGenSpec], list[dict]],
    evaluate: Callable[[JointEvalSpec], float] | None = None,
) -> JointOsfpResult:
    """Run the joint self-play OSFP loop; ``generate`` / ``evaluate`` are injected."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    cfg.iter_dir.mkdir(parents=True, exist_ok=True)
    index = CardEmbeddingIndex(pool)  # card id -> shared-embedding row (play arm)
    history = OpponentPool(
        [], decay=cfg.decay, self_play_prob=cfg.self_play_prob,
        threshold=cfg.threshold, patience=cfg.patience,
    )
    net_np = PolicyValueNet.load(cfg.init_weights)

    stats: list[JointIterStat] = []
    for n in range(1, cfg.iterations + 1):
        weights_path = cfg.iter_dir / f"jointiter_{n}.npz"
        net_np.save(weights_path)  # save BEFORE collecting -- the collector loads this

        opp = history.sample(n, rng)
        spec = JointGenSpec(
            weights=weights_path, opp=opp, n_decks=cfg.decks_per_iter,
            games_per_deck=cfg.games_per_deck, temperature=cfg.temperature,
            seed=cfg.seed + n * 1000, out=cfg.iter_dir / f"games_{n}",
            max_play_samples=cfg.max_play_samples,
        )
        games, decks = _split_records(generate(spec))

        play_samples = build_policy_samples(
            games, feats, index, teachers={LEARNER}, discount=None,
            max_samples=cfg.max_play_samples, rng=rng,
        )
        scored = [(r["deck"], _deck_return(r)) for r in decks
                  if r["wins"] + r["losses"] > 0]
        card_feats, deck_samples = cb_rl_sequences(scored, pool, feats, normalize=True)
        net_np = _joint_train_step(
            net_np, play_samples, card_feats, deck_samples, cfg,
        )

        gate: float | None = None
        if evaluate is not None and n % cfg.eval_every == 0:
            gate = evaluate(JointEvalSpec(
                weights=weights_path, games=cfg.eval_games, seed=cfg.seed,
                out=cfg.iter_dir / f"gate_{n}",
            ))
        admitted = history.admit(
            str(weights_path), n, {"gate": gate} if gate is not None else {},
        )

        label = _opp_label(opp)
        mean_wr = _mean_winrate(decks)
        stats.append(JointIterStat(
            n, label, len(games), len(decks), len(play_samples), len(deck_samples),
            mean_wr, gate, admitted,
        ))
        print(
            f"[jointiter {n}] opp={label} games={len(games)} decks={len(decks)} "
            f"play={len(play_samples)} deck={len(deck_samples)} mean_wr={mean_wr:.3f} "
            f"gate={gate} admitted={admitted} ckpts={history.num_checkpoints}",
        )

    final = cfg.iter_dir / "joint_final.npz"
    net_np.save(final)
    run_history = {
        "config": {
            "iterations": cfg.iterations, "decks_per_iter": cfg.decks_per_iter,
            "games_per_deck": cfg.games_per_deck, "temperature": cfg.temperature,
            "lr": cfg.lr, "seed": cfg.seed, "self_play_prob": cfg.self_play_prob,
            "init_weights": str(cfg.init_weights), "gate_deck": str(cfg.gate_deck),
        },
        "final_weights": str(final),
        "checkpoints": history.num_checkpoints,
        "iterations_log": [asdict(s) for s in stats],
    }
    (cfg.iter_dir / "history.json").write_text(json.dumps(run_history, indent=2))
    print(f"== done: {final}  checkpoints={history.num_checkpoints} ==")
    return JointOsfpResult(final, stats, history)


# --- production wiring (shells the collector: Docker, or native subprocess) --


def _in_container(path: Path) -> str:
    return f"/work/{path.resolve().relative_to(ROOT)}"


def _run(argv: list[str], what: str, env: dict[str, str] | None = None) -> None:
    try:
        subprocess.run(  # noqa: S603
            argv, check=True, capture_output=True, text=True, env=env,
        )
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "")[-2000:]
        msg = f"{what} failed (exit {exc.returncode}): {tail}"
        raise RuntimeError(msg) from exc


def _read_records(out: Path) -> list[dict]:
    shard = next((out / "games").glob("*.jsonl"))
    return [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]


_SEED_STRIDE = 1_000_000  # space shard seeds far apart so they sample different decks
_DECISIONS_PER_GAME = 195  # rough self-play decisions/game, for the keep-prob estimate


def _split(total: int, parts: int) -> list[int]:
    """Split ``total`` into ``parts`` near-even chunks (some may be 0)."""
    base, rem = divmod(total, max(parts, 1))
    return [base + (1 if i < rem else 0) for i in range(parts)]


@dataclass(frozen=True)
class _Collector:
    """How to launch one collector batch: native subprocess or Docker container.

    ``native`` skips the container -- valid only on a real Linux x86-64 host, where
    ``collect_joint_selfplay`` imports ``cg`` directly. It avoids per-launch
    container startup and docker-daemon contention, so ``--workers`` keeps scaling
    past ~12. Both backends pin BLAS to one thread per worker so parallel collectors
    don't oversubscribe the CPU (un-pinned, 8 workers measured ~15x slower each).
    """

    native: bool
    _SCRIPT = "scripts/collect_joint_selfplay.py"

    def argv(self, args: list[str]) -> list[str]:
        if self.native:
            return [sys.executable, str(ROOT / self._SCRIPT), *args]
        return [*DOCKER_PREFIX, self._SCRIPT, *args]

    def path(self, p: Path) -> str:
        # native runs in-tree (real paths); docker sees the repo mounted at /work.
        return str(p.resolve()) if self.native else _in_container(p)

    def env(self) -> dict[str, str] | None:
        if not self.native:
            return None  # DOCKER_PREFIX already passes the thread pins via -e
        return {
            **os.environ, "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        }

    @property
    def label(self) -> str:
        return "native subprocess" if self.native else "docker container"


def _make_collect(
    workers: int, backend: _Collector,
) -> Callable[[JointGenSpec], list[dict]]:
    """Build a ``generate`` that fans the deck batch across ``workers`` collectors.

    The engine is single-threaded, so parallelism is N independent collector
    processes. Shards are seeded far apart (different decks) and a failed shard is
    skipped (its decks/games are lost for that iteration) so one flaky worker can't
    abort a long run. Pinned, a 16-core native x86 box scaled ~linearly to ~12
    workers; under ARM x86-emulated Docker extra workers contend, so ``--workers``
    defaults to 1 -- raise it on a native box (ideally with ``--native``).
    """
    def _shard(spec: JointGenSpec, k: int, n_decks: int) -> list[dict]:
        out = spec.out / f"w{k}"
        # Global keep-prob so the collectors deepcopy ~max_play_samples decisions in
        # total (instead of all ~200/game) -- the single biggest collection cost.
        total_decisions = spec.n_decks * spec.games_per_deck * _DECISIONS_PER_GAME
        keep_prob = min(1.0, spec.max_play_samples / max(total_decisions, 1))
        args = [
            "--weights", backend.path(spec.weights),
            "--decks", str(n_decks), "--games-per-deck", str(spec.games_per_deck),
            "--temperature", str(spec.temperature), "--play-keep-prob", str(keep_prob),
            "--seed", str(spec.seed + k * _SEED_STRIDE), "--out", backend.path(out),
        ]
        if spec.opp is None:
            args.append("--self-play")
        else:
            args += ["--opp-weights", backend.path(Path(spec.opp.ref))]
        try:
            _run(backend.argv(args), f"collect_joint w{k}", backend.env())
            return _read_records(out)
        except (RuntimeError, StopIteration) as exc:
            print(f"  [warn] shard {k} failed, skipping its decks: {exc}")
            return []

    def collect(spec: JointGenSpec) -> list[dict]:
        sizes = [(k, n) for k, n in enumerate(_split(spec.n_decks, workers)) if n > 0]
        if len(sizes) <= 1:  # nothing to parallelise
            return _shard(spec, *sizes[0]) if sizes else []
        with ThreadPoolExecutor(max_workers=len(sizes)) as ex:
            shards = ex.map(lambda kn: _shard(spec, kn[0], kn[1]), sizes)
        return [record for shard in shards for record in shard]

    return collect


def _make_gate(
    gate_deck: Path, backend: _Collector,
) -> Callable[[JointEvalSpec], float]:
    def evaluate(spec: JointEvalSpec) -> float:
        args = [
            "--weights", backend.path(spec.weights),
            "--gate-deck", backend.path(gate_deck),
            "--games-per-deck", str(spec.games), "--seed", str(spec.seed),
            "--out", backend.path(spec.out),
        ]
        _run(backend.argv(args), "joint gate", backend.env())
        _, decks = _split_records(_read_records(spec.out))
        return _deck_return(decks[0]) / 2 + 0.5 if decks else 0.0  # -> win rate

    return evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint OSFP self-play training")
    parser.add_argument(
        "--weights", type=Path, default=ROOT / "data/bc/bc_net_joint.npz",
    )
    parser.add_argument(
        "--engine-json", type=Path, default=ROOT / "data/bc/engine.json",
    )
    parser.add_argument(
        "--gate-deck", type=Path, default=ROOT / "decklists/metal_aggro.csv",
        help="yardstick reference deck (eval only, not training)",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data/jointosfp")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--decks", type=int, default=16)
    parser.add_argument("--games-per-deck", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--self-play-prob", type=float, default=0.5)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="parallel collectors; >1 ONLY helps on native x86 (under ARM "
             "x86-emulated Docker, extra workers contend). Set ~cores-2 on a box.",
    )
    parser.add_argument(
        "--native", action="store_true",
        help="run collectors as native subprocesses (no Docker). Requires a real "
             "Linux x86-64 host where `import cg` works; faster and scales further.",
    )
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-play-samples", type=int, default=20000,
        help="subsample play decisions per iter before encoding/training",
    )
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="3 tiny iterations")
    args = parser.parse_args()

    if args.smoke:
        args.iterations, args.decks, args.games_per_deck, args.eval_games = 3, 4, 4, 20

    cfg = JointOsfpConfig(
        init_weights=args.weights, iter_dir=args.out, gate_deck=args.gate_deck,
        iterations=args.iterations, decks_per_iter=args.decks,
        games_per_deck=args.games_per_deck, temperature=args.temperature, lr=args.lr,
        self_play_prob=args.self_play_prob, eval_every=args.eval_every,
        eval_games=args.eval_games, seed=args.seed,
        max_play_samples=args.max_play_samples,
    )
    feats = CardFeatures(load_engine_json(args.engine_json))
    pool = build_pool()
    backend = _Collector(native=args.native)
    evaluate = None if args.no_eval else _make_gate(args.gate_deck, backend)
    print(f"joint self-play: {args.workers} parallel collector(s) via {backend.label}")
    run_joint_osfp(
        cfg, pool, feats,
        generate=_make_collect(args.workers, backend), evaluate=evaluate,
    )


if __name__ == "__main__":
    main()
