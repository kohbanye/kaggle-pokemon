"""CB-head self-play RL orchestrator (Phase 5b-ii, native; shells Docker for sim).

Each iteration: save the current net, sample a batch of decks from its CB head,
score each by playing it K times vs a fixed reference deck (Docker:
``scripts/collect_cb.py``), turn the deck returns into a REINFORCE advantage
(``cb_rl_samples``), and update **only** the CB head + card embedding
(``LitCBPolicyGradient``; the play/value heads stay frozen at the Phase-5a/BC
result). Periodically run the **gate**: does the greedy learned deck beat the
fixed reference (metal_aggro)?

Like ``train_osfp.run_osfp``, the loop (:func:`run_cb_rl`) takes injected
``generate`` / ``evaluate`` callables, so it is unit-tested natively with a
synthetic generator and no Docker. This is a **research bet** (one ±1 outcome must
disentangle deck quality from play and engine RNG); the gate decides keep/drop.

  uv run python scripts/train_cb.py --smoke
  uv run python scripts/train_cb.py --iterations 30 --decks 24 --games-per-deck 16
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
sys.path.insert(0, str(ROOT))

import lightning as L  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.deck import build_pool  # noqa: E402
from src.net.bc_data import (  # noqa: E402
    CBDataset,
    cb_rl_samples,
    collate_cb,
    load_engine_json,
)
from src.net.features import CardFeatures  # noqa: E402
from src.net.lit import LitCBPolicyGradient  # noqa: E402
from src.net.model import PolicyValueNet  # noqa: E402
from src.net.torch_model import from_numpy_net  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.deck import CardPool

DOCKER_PREFIX = (
    "docker", "run", "--platform=linux/amd64", "--rm",
    "-v", f"{ROOT}:/work", "-w", "/work", "ptcg-sim", "python",
)


@dataclass
class CBConfig:
    init_weights: Path
    iter_dir: Path
    opp_deck: Path
    iterations: int = 20
    decks_per_iter: int = 16
    games_per_deck: int = 16
    lr: float = 1e-3
    epochs: int = 1
    batch_size: int = 256
    eval_every: int = 5
    eval_games: int = 200
    seed: int = 0


@dataclass
class CBGenSpec:
    weights: Path
    n_decks: int
    games_per_deck: int
    opp_deck: Path
    seed: int
    out: Path
    greedy: bool = False


@dataclass
class CBIterStat:
    iteration: int
    n_decks: int
    n_samples: int
    mean_winrate: float
    gate_winrate: float | None


@dataclass
class CBResult:
    final_weights: Path
    iterations: list[CBIterStat]


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


def _cb_train_step(
    net_np: PolicyValueNet,
    card_feats: object,
    samples: list,
    cfg: CBConfig,
) -> PolicyValueNet:
    torch_net = from_numpy_net(net_np)
    lit = LitCBPolicyGradient(torch_net, card_feats, lr=cfg.lr)
    loader = DataLoader(
        CBDataset(samples), batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_cb,
    )
    _trainer(cfg.epochs).fit(lit, loader)
    return lit.net.double().to_numpy_net()


def run_cb_rl(
    cfg: CBConfig,
    pool: CardPool,
    feats: CardFeatures,
    *,
    generate: Callable[[CBGenSpec], list[dict]],
    evaluate: Callable[[CBGenSpec], float] | None = None,
) -> CBResult:
    """Run the CB-head RL loop; ``generate`` / ``evaluate`` are injected."""
    torch.manual_seed(cfg.seed)
    cfg.iter_dir.mkdir(parents=True, exist_ok=True)
    net_np = PolicyValueNet.load(cfg.init_weights)

    stats: list[CBIterStat] = []
    for n in range(1, cfg.iterations + 1):
        weights_path = cfg.iter_dir / f"cbiter_{n}.npz"
        net_np.save(weights_path)

        spec = CBGenSpec(
            weights=weights_path, n_decks=cfg.decks_per_iter,
            games_per_deck=cfg.games_per_deck, opp_deck=cfg.opp_deck,
            seed=cfg.seed + n * 1000, out=cfg.iter_dir / f"decks_{n}",
        )
        records = generate(spec)
        scored = [(r["deck"], _deck_return(r)) for r in records
                  if r["wins"] + r["losses"] > 0]
        card_feats, samples = cb_rl_samples(scored, pool, feats, normalize=True)
        if samples:
            net_np = _cb_train_step(net_np, card_feats, samples, cfg)

        mean_wr = (
            sum(r["wins"] / (r["wins"] + r["losses"])
                for r in records if r["wins"] + r["losses"] > 0)
            / max(sum(1 for r in records if r["wins"] + r["losses"] > 0), 1)
        )
        gate: float | None = None
        if evaluate is not None and n % cfg.eval_every == 0:
            gate = evaluate(CBGenSpec(
                weights=weights_path, n_decks=1, games_per_deck=cfg.eval_games,
                opp_deck=cfg.opp_deck, seed=cfg.seed, out=cfg.iter_dir / f"gate_{n}",
                greedy=True,
            ))
        stats.append(CBIterStat(n, len(records), len(samples), mean_wr, gate))
        print(
            f"[cbiter {n}] decks={len(records)} samples={len(samples)} "
            f"mean_wr={mean_wr:.3f} gate={gate}",
        )

    final = cfg.iter_dir / "cb_final.npz"
    net_np.save(final)
    print(f"== done: {final} ==")
    return CBResult(final, stats)


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


def _collect_argv(spec: CBGenSpec) -> list[str]:
    argv = [
        *DOCKER_PREFIX, "scripts/collect_cb.py",
        "--weights", _in_container(spec.weights),
        "--opp-deck", _in_container(spec.opp_deck),
        "--decks", str(spec.n_decks), "--games-per-deck", str(spec.games_per_deck),
        "--seed", str(spec.seed), "--out", _in_container(spec.out),
    ]
    if spec.greedy:
        argv.append("--greedy")
    return argv


def _read_records(out: Path) -> list[dict]:
    shard = next((out / "decks").glob("*.jsonl"))
    return [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]


def _docker_cb_collect(spec: CBGenSpec) -> list[dict]:
    _run(_collect_argv(spec), "collect_cb")
    return _read_records(spec.out)


def _docker_cb_gate(spec: CBGenSpec) -> float:
    _run(_collect_argv(spec), "collect_cb gate")
    records = _read_records(spec.out)
    return _deck_return(records[0]) / 2 + 0.5 if records else 0.0  # return -> win rate


def main() -> None:
    parser = argparse.ArgumentParser(description="CB-head RL training (Phase 5b-ii)")
    parser.add_argument("--weights", type=Path, default=ROOT / "data/bc/bc_net_emb.npz")
    parser.add_argument(
        "--engine-json", type=Path, default=ROOT / "data/bc/engine.json",
    )
    parser.add_argument(
        "--opp-deck", type=Path, default=ROOT / "decklists/metal_aggro.csv",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data/cb")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--decks", type=int, default=16)
    parser.add_argument("--games-per-deck", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="3 tiny iterations")
    args = parser.parse_args()

    if args.smoke:
        args.iterations, args.decks, args.games_per_deck, args.eval_games = 3, 4, 4, 20

    cfg = CBConfig(
        init_weights=args.weights, iter_dir=args.out, opp_deck=args.opp_deck,
        iterations=args.iterations, decks_per_iter=args.decks,
        games_per_deck=args.games_per_deck, lr=args.lr, eval_every=args.eval_every,
        eval_games=args.eval_games, seed=args.seed,
    )
    feats = CardFeatures(load_engine_json(args.engine_json))
    pool = build_pool()
    evaluate = None if args.no_eval else _docker_cb_gate
    run_cb_rl(cfg, pool, feats, generate=_docker_cb_collect, evaluate=evaluate)


if __name__ == "__main__":
    main()
