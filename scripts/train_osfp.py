"""OSFP self-play orchestrator (Phase 5a, native; shells Docker for the sim).

The outer loop the paper calls Algorithm 1, scaled to one machine. Each iteration:

  1. save the current net to ``iter_N.npz`` (the exact file the collector loads);
  2. sample an opponent from the :class:`~src.net.osfp.OpponentPool` (or self-play);
  3. **generate** self-play trajectories (Docker: ``scripts/collect_selfplay.py``);
  4. keep only decisive games and turn the learner's decisions into RL samples
     (``build_policy_samples(teachers={"learner"})`` -- target = sampled action,
     value = game return) -- the same encoder Phase-4 BC used;
  5. one policy-gradient pass (:class:`~src.net.lit.LitPolicyGradient`, CB frozen);
  6. periodically **evaluate** vs the baselines + BC and feed the win-rates to the
     pool's admission rule.

The loop itself (:func:`run_osfp`) is pure: it takes ``generate`` / ``evaluate``
callables, so tests drive the whole thing with a synthetic generator and no Docker
(the engine is Linux-only and its RNG is unseedable, so the Docker path gets only
a smoke, never a numeric assertion). The production wiring (``_docker_collect`` /
``_docker_eval``) shells ``docker run`` exactly like the Phase-4 boundary.

  uv run python scripts/train_osfp.py --smoke         # 3 iters, tiny, needs Docker
  uv run python scripts/train_osfp.py --iterations 30 --games 128
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # make `src` importable when run as a script

import lightning as L  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.net.bc_data import (  # noqa: E402
    PolicyDataset,
    build_policy_samples,
    collate_policy,
    game_files,
    iter_games,
    load_engine_json,
)
from src.net.features import CardFeatures  # noqa: E402
from src.net.lit import LitPolicyGradient  # noqa: E402
from src.net.model import PolicyValueNet  # noqa: E402
from src.net.osfp import OpponentPool  # noqa: E402
from src.net.torch_model import from_numpy_net  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.net.osfp import PoolEntry

DOCKER_PREFIX = (
    "docker", "run", "--platform=linux/amd64", "--rm",
    "-v", f"{ROOT}:/work", "-w", "/work", "ptcg-sim", "python",
)


@dataclass
class OsfpConfig:
    """Everything the OSFP loop needs (one object so no call has a long arg list)."""

    deck: Path
    bc_weights: Path
    iter_dir: Path
    baselines: list[str]
    iterations: int = 10
    games_per_iter: int = 64
    temperature: float = 1.0
    epochs: int = 1
    batch_size: int = 256
    lr: float = 1e-3
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    eval_every: int = 5
    eval_games: int = 100
    seed: int = 0
    decay: float = 0.5
    self_play_prob: float = 0.3
    baseline_floor: float = 0.1
    threshold: float = 0.55
    patience: int = 3


@dataclass
class GenSpec:
    """One self-play collection request (opp ``None`` => self-play the learner)."""

    learner_weights: Path
    opp: PoolEntry | None
    deck: Path
    games: int
    temperature: float
    seed: int
    out: Path


@dataclass
class EvalSpec:
    """One evaluation request for the current learner weights."""

    weights: Path
    games: int
    seed: int


@dataclass
class IterStat:
    """Per-iteration record (returned for inspection / tests / logging)."""

    iteration: int
    opponent: str
    n_games: int
    n_samples: int
    winrates: dict[str, float]
    admitted: bool


@dataclass
class OsfpResult:
    final_weights: Path
    iterations: list[IterStat]
    pool: OpponentPool


# --- the pure loop ---------------------------------------------------------


def _trainer(epochs: int) -> L.Trainer:
    return L.Trainer(
        max_epochs=epochs, accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False,
    )


def _train_step(
    net_np: PolicyValueNet,
    samples: list,
    cfg: OsfpConfig,
) -> PolicyValueNet:
    """One policy-gradient pass; returns the updated numpy serving net."""
    torch_net = from_numpy_net(net_np)
    lit = LitPolicyGradient(
        torch_net, lr=cfg.lr, value_coef=cfg.value_coef, entropy_coef=cfg.entropy_coef,
    )
    loader = DataLoader(
        PolicyDataset(samples), batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_policy,
    )
    _trainer(cfg.epochs).fit(lit, loader)
    return lit.net.double().to_numpy_net()


def _opp_label(opp: PoolEntry | None) -> str:
    if opp is None:
        return "self"
    return Path(opp.ref).name if opp.kind == "checkpoint" else opp.ref


def run_osfp(
    cfg: OsfpConfig,
    feats: CardFeatures,
    *,
    generate: Callable[[GenSpec], list[dict]],
    evaluate: Callable[[EvalSpec], dict[str, float]] | None = None,
) -> OsfpResult:
    """Run the OSFP self-play loop; ``generate`` / ``evaluate`` are injected."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    cfg.iter_dir.mkdir(parents=True, exist_ok=True)
    pool = OpponentPool(
        cfg.baselines, decay=cfg.decay, self_play_prob=cfg.self_play_prob,
        baseline_floor=cfg.baseline_floor, threshold=cfg.threshold,
        patience=cfg.patience,
    )
    net_np = PolicyValueNet.load(cfg.bc_weights)

    stats: list[IterStat] = []
    for n in range(1, cfg.iterations + 1):
        weights_path = cfg.iter_dir / f"iter_{n}.npz"
        net_np.save(weights_path)  # save BEFORE collecting -- the collector loads this

        opp = pool.sample(n, rng)
        spec = GenSpec(
            learner_weights=weights_path, opp=opp, deck=cfg.deck,
            games=cfg.games_per_iter, temperature=cfg.temperature,
            seed=cfg.seed + n * 1000, out=cfg.iter_dir / f"games_{n}",
        )
        # Keep only decisive games: _outcome can't tell a draw from an abort, so an
        # aborted trajectory would leak in as a spurious value=0 sample.
        games = [g for g in generate(spec) if int(g.get("winner", -1)) in (0, 1)]
        samples = build_policy_samples(
            games, feats, teachers={"learner"}, discount=None,  # gamma = 1
        )
        if samples:
            net_np = _train_step(net_np, samples, cfg)

        winrates: dict[str, float] = {}
        if evaluate is not None and n % cfg.eval_every == 0:
            winrates = evaluate(
                EvalSpec(weights=weights_path, games=cfg.eval_games, seed=cfg.seed),
            )
        admitted = pool.admit(str(weights_path), n, winrates)

        label = _opp_label(opp)
        stats.append(IterStat(n, label, len(games), len(samples), winrates, admitted))
        print(
            f"[iter {n}] opp={label} games={len(games)} samples={len(samples)} "
            f"admitted={admitted} ckpts={pool.num_checkpoints} winrates={winrates}",
        )

    final = cfg.iter_dir / "final.npz"
    net_np.save(final)
    print(f"== done: {final}  checkpoints={pool.num_checkpoints} ==")
    return OsfpResult(final, stats, pool)


# --- production wiring (shells Docker) -------------------------------------


def _in_container(path: Path) -> str:
    """Map a host path under the repo to its bind-mounted ``/work/...`` path."""
    return f"/work/{path.resolve().relative_to(ROOT)}"


def _run(argv: list[str], what: str) -> None:
    try:
        subprocess.run(argv, check=True, capture_output=True, text=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "")[-2000:]
        msg = f"docker {what} failed (exit {exc.returncode}): {tail}"
        raise RuntimeError(msg) from exc


def _docker_collect(spec: GenSpec) -> list[dict]:
    argv = [
        *DOCKER_PREFIX, "scripts/collect_selfplay.py",
        "--learner-weights", _in_container(spec.learner_weights),
        "--deck", _in_container(spec.deck),
        "--games", str(spec.games), "--temperature", str(spec.temperature),
        "--seed", str(spec.seed), "--out", _in_container(spec.out),
    ]
    if spec.opp is None:
        argv.append("--self-play")
    elif spec.opp.kind == "baseline":
        argv += ["--opp-agent", spec.opp.ref]
    else:
        argv += ["--opp-weights", _in_container(Path(spec.opp.ref))]
    _run(argv, "collect")
    return list(iter_games(game_files(spec.out)))


def _make_docker_eval(
    deck: Path,
    opponents: list[tuple[str, str, Path | None]],
    out_dir: Path,
) -> Callable[[EvalSpec], dict[str, float]]:
    """Build an ``evaluate`` that runs ``run_eval`` vs each opponent under Docker."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(spec: EvalSpec) -> dict[str, float]:
        results: dict[str, float] = {}
        for label, agent, weights in opponents:
            tag = f"eval_{label}"
            argv = [
                *DOCKER_PREFIX, "scripts/run_eval.py",
                "--a", "net", "--a-weights", _in_container(spec.weights),
                "--b", agent, "--deck", _in_container(deck),
                "--games", str(spec.games), "--seed", str(spec.seed),
                "--out", _in_container(out_dir), "--tag", tag, "--progress-every", "0",
            ]
            if weights is not None:
                argv += ["--b-weights", _in_container(weights)]
            _run(argv, f"eval-{label}")
            summary = json.loads((out_dir / f"{tag}.json").read_text())
            results[label] = float(summary["a_winrate"])
        return results

    return evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="OSFP self-play training (Phase 5a)")
    parser.add_argument("--bc-weights", type=Path, default=ROOT / "data/bc/bc_net.npz")
    parser.add_argument(
        "--engine-json", type=Path, default=ROOT / "data/bc/engine.json",
    )
    parser.add_argument("--deck", type=Path, default=ROOT / "decklists/metal_aggro.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data/osfp")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="3 tiny iterations")
    args = parser.parse_args()

    if args.smoke:
        args.iterations, args.games, args.eval_games = 3, 8, 20

    cfg = OsfpConfig(
        deck=args.deck, bc_weights=args.bc_weights, iter_dir=args.out,
        baselines=["random", "greedy", "heuristic"], iterations=args.iterations,
        games_per_iter=args.games, temperature=args.temperature, epochs=args.epochs,
        lr=args.lr, entropy_coef=args.entropy_coef, eval_every=args.eval_every,
        eval_games=args.eval_games, seed=args.seed,
    )
    feats = CardFeatures(load_engine_json(args.engine_json))
    opponents: list[tuple[str, str, Path | None]] = [
        ("random", "random", None),
        ("greedy", "greedy", None),
        ("heuristic", "heuristic", None),
        ("bc", "net", args.bc_weights),
    ]
    evaluate = (
        None if args.no_eval
        else _make_docker_eval(args.deck, opponents, args.out / "eval")
    )
    run_osfp(cfg, feats, generate=_docker_collect, evaluate=evaluate)


if __name__ == "__main__":
    main()
