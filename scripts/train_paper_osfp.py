"""Paper-faithful OSFP actor-learner loop (V-Trace + PPO + recurrent net).

The orchestrator for the rewrite: each iteration the actors collect recurrent
self-play **trajectories** (``collect_paper_selfplay``, native subprocesses), the
episodes feed a bounded **FIFO queue** (the IMPALA-style replay buffer; V-Trace
corrects the staleness of the older episodes still in it), and the learner does one
:class:`~src.net.lit_vtrace.LitVtracePPO` update over a sample of the queue. The
opponent each iteration is drawn from the OSFP :class:`~src.net.osfp.OpponentPool`
(self-play with probability ``self_play_prob``, else a recency-weighted past
checkpoint); strong checkpoints are admitted back into the pool.

The collector / FIFO / V-Trace together are the paper's actor-learner; the
OpponentPool is the OSFP meta-loop (last-iterate -> submit the final checkpoint
directly).

  uv run python scripts/train_paper_osfp.py --smoke --native
  uv run python scripts/train_paper_osfp.py --native --workers 14 --iterations 500
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.deck import build_pool  # noqa: E402
from src.net.embedding import CardEmbeddingIndex  # noqa: E402
from src.net.features import CardFeatures, load_engine_json  # noqa: E402
from src.net.lit_vtrace import LitVtracePPO  # noqa: E402
from src.net.osfp import OpponentPool, PoolEntry  # noqa: E402
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
    # QD mode: decks come from this archive (a JSON deck list) and the deck (CB) head
    # is NOT trained -- only the play head learns to pilot the diverse archive decks.
    deck_pool: Path | None = None
    train_deck: bool = True
    # Async actor-learner: collect+encode iteration n+1 on a background thread while
    # the learner updates on iteration n. The actor's weights then lag the learner by
    # one step -- V-Trace corrects that off-policy staleness (it is the IMPALA design).
    # Numerics-affecting (unlike the other speedups), so opt-in until validated.
    pipeline: bool = False


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


# --- parallel episode encoding ----------------------------------------------
# build_episodes (records -> encoded Episodes) was ~30% of an iteration and ran
# single-threaded in the master. It is embarrassingly parallel over games, so a
# persistent worker pool (each worker holds its own feats/index/pool) encodes
# chunks of the records concurrently. The pool is forked BEFORE CUDA is touched
# (see run()) so the workers never inherit a CUDA context.
_BG: dict = {}


def _build_init(engine_json: str) -> None:
    pool = build_pool()
    _BG["feats"] = CardFeatures(load_engine_json(Path(engine_json)))
    _BG["index"] = CardEmbeddingIndex(pool)
    _BG["pool"] = pool


def _build_chunk(games: list[dict]) -> list:
    return build_episodes(games, _BG["feats"], _BG["index"], _BG["pool"])


def _build_episodes_par(  # noqa: PLR0913 - encode-request parameters
    pp: Pool | None,
    records: list[dict],
    parts: int,
    feats: CardFeatures,
    index: CardEmbeddingIndex,
    pool: object,
) -> list:
    """Encode ``records`` into Episodes, fanning chunks across ``pp`` (or inline)."""
    if pp is None or len(records) < parts * 2:  # too few to bother parallelising
        return build_episodes(records, feats, index, pool)
    sizes = _split(len(records), parts)
    chunks, i = [], 0
    for n in sizes:
        chunks.append(records[i:i + n])
        i += n
    return [ep for part in pp.map(_build_chunk, chunks) for ep in part]


def _opp_args(cfg: Config, opp: PoolEntry | None) -> list[str]:
    """Collector opponent flag: self-play / a meta-deck baseline / a past checkpoint."""
    if cfg.deck_pool is not None:  # QD mode: both sides draw from the archive
        return ["--deck-pool", _path(cfg, cfg.deck_pool)]
    if opp is None:
        return ["--self-play"]
    if opp.kind == "baseline":  # a fixed meta deck (external deck-quality pressure)
        return ["--opp-deck", _path(cfg, Path(opp.ref))]
    return ["--opp-weights", _path(cfg, Path(opp.ref))]  # a past checkpoint net


def _collect(  # noqa: PLR0913 - one collection request's parameters
    cfg: Config,
    weights: Path,
    opp: PoolEntry | None,
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
            *_opp_args(cfg, opp),
        ]
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


class _Learner:
    """Resident V-Trace/PPO learner: the torch net + Adam live on the GPU for the
    whole run and are stepped in place each iteration.

    The old per-iteration path rebuilt a Lightning ``Trainer`` (and re-loaded the net
    numpy->torch) every update -- ~50% of the iteration's wall-time was that fixed
    setup/teardown, not compute. Here the net is built once; each update is a plain
    manual optimiser loop over the same :class:`LitVtracePPO` loss (identical maths,
    no Trainer). ``self.log*`` is neutered because those need an attached Trainer.
    """

    def __init__(self, cfg: Config, net_np: RecurrentPolicyValueNet,
                 card_feats: object) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_net = TorchRecurrentNet(net_np.config)
        torch_net.load_numpy_params(net_np.params)
        self.lit = LitVtracePPO(
            torch_net, card_feats, lr=cfg.lr, value_coef=cfg.value_coef,
            entropy_coef=cfg.entropy_coef, deck_entropy_coef=cfg.deck_entropy_coef,
            clip_eps=cfg.clip_eps, clip_rho=cfg.clip_rho, clip_c=cfg.clip_c,
            rho_min=cfg.rho_min, train_deck=cfg.train_deck,
        ).to(self.device)
        self.lit.log = lambda *_a, **_k: None  # type: ignore[method-assign]
        self.lit.log_dict = lambda *_a, **_k: None  # type: ignore[method-assign]
        self.opt = torch.optim.Adam(self.lit.net.parameters(), lr=cfg.lr)

    def step(self, episodes: list) -> None:
        """One V-Trace/PPO update over a sample of the FIFO queue (in place)."""
        if not episodes:
            return
        self.lit.train()
        loader = DataLoader(
            EpisodeDataset(episodes), batch_size=self.cfg.batch_size, shuffle=True,
            collate_fn=collate_episodes,
        )
        dev = self.device
        for _ in range(self.cfg.epochs):
            for battle, deck in loader:
                b = {k: v.to(dev) for k, v in battle.items()}
                d = {k: v.to(dev) for k, v in deck.items()}
                self.opt.zero_grad()
                loss = self.lit.training_step((b, d), 0)
                loss.backward()
                self.opt.step()

    def to_numpy(self) -> RecurrentPolicyValueNet:
        """Export the live weights as a numpy serving net (float64, on a copy so the
        resident float32 net + its optimiser state are left untouched)."""
        return copy.deepcopy(self.lit.net).double().to_numpy_net()


def _collect_build(  # noqa: PLR0913 - one collect+encode request's parameters
    cfg: Config,
    build_pp: Pool | None,
    weights_path: Path,
    opp_entry: PoolEntry | None,
    seed: int,
    out: Path,
    feats: CardFeatures,
    index: CardEmbeddingIndex,
    pool: object,
) -> list:
    """Collect self-play games with ``weights_path`` and encode them to Episodes."""
    records = _collect(cfg, weights_path, opp_entry, cfg.games_per_iter, seed, out)
    return _build_episodes_par(build_pp, records, cfg.workers, feats, index, pool)


def run(cfg: Config) -> None:  # noqa: C901, PLR0915 - orchestrator: launch/consume loop
    """Run the paper-faithful OSFP actor-learner loop."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    pool = build_pool()
    feats = CardFeatures(load_engine_json(cfg.engine_json))
    card_feats = CardEmbeddingIndex(pool).fixed_matrix(feats)
    index = CardEmbeddingIndex(pool)

    # Seed the OSFP pool with the meta decklists as permanent baselines (paper
    # Alg.1): the learner's sampled deck plays them piloted by its own play head, so
    # a degenerate (e.g. zero-energy) deck is punished by a real deck -- the external
    # pressure self-play alone can't provide. Recency-weighted checkpoints are added
    # on top as the run proceeds.
    baselines = [str(p) for p in sorted((ROOT / "decklists").glob("*.csv"))]
    history = OpponentPool(
        baselines, decay=cfg.decay, self_play_prob=cfg.self_play_prob,
        threshold=cfg.threshold, patience=cfg.patience,
    )
    # Persistent episode-encoding pool, forked HERE -- before _Learner touches CUDA --
    # so its workers never inherit a CUDA context (fork-after-CUDA would hang/crash).
    build_pp = (
        Pool(cfg.workers, initializer=_build_init, initargs=(str(cfg.engine_json),))
        if cfg.workers > 1 else None
    )
    net_np = RecurrentPolicyValueNet.load(cfg.init_weights)
    learner = _Learner(cfg, net_np, card_feats)  # torch net + Adam, resident on GPU
    queue: deque = deque(maxlen=cfg.queue_episodes)
    actor = ThreadPoolExecutor(max_workers=1) if cfg.pipeline else None

    def launch(n: int) -> dict:
        """Save the current (actor) weights and kick off iteration ``n``'s collect."""
        wp = cfg.out_dir / f"paperiter_{n}.npz"
        net_np.save(wp)  # the collector loads this; stable file before the thread runs
        opp = history.sample(n, rng)
        out = cfg.out_dir / f"games_{n}"
        args = (cfg, build_pp, wp, opp, cfg.seed + n * 1000, out, feats, index, pool)
        fut = actor.submit(_collect_build, *args) if actor else None
        return {"n": n, "wp": wp, "opp": opp, "out": out, "fut": fut, "args": args}

    def consume(job: dict, episodes: list) -> None:
        """Learner update on ``episodes`` + gate/admit bookkeeping for ``job``."""
        nonlocal net_np
        queue.extend(episodes)
        sample = list(queue)
        if len(sample) > cfg.train_episodes:
            pick = rng.choice(len(sample), size=cfg.train_episodes, replace=False)
            sample = [sample[i] for i in pick]
        if sample:
            learner.step(sample)
            net_np = learner.to_numpy()  # export for the next collect + gate
        n = job["n"]
        gate = None
        if n % cfg.eval_every == 0:
            gate = _gate(cfg, job["wp"], cfg.out_dir / f"gate_{n}")
        admitted = history.admit(
            str(job["wp"]), n, {"gate": gate} if gate is not None else {},
        )
        label = "self" if job["opp"] is None else Path(job["opp"].ref).name
        print(
            f"[paperiter {n}] opp={label} new_eps={len(episodes)} "
            f"queue={len(queue)} trained={len(sample)} gate={gate} "
            f"admitted={admitted} ckpts={history.num_checkpoints}",
        )
        shutil.rmtree(job["out"], ignore_errors=True)
        shutil.rmtree(cfg.out_dir / f"gate_{n}", ignore_errors=True)

    # Pipeline: actor runs one iteration ahead of the learner. Sync: launch and
    # consume the same iteration before moving on (the actor uses the just-updated net).
    job = launch(1)
    for n in range(1, cfg.iterations + 1):
        episodes = job["fut"].result() if actor else _collect_build(*job["args"])
        if cfg.pipeline and n < cfg.iterations:
            job_next = launch(n + 1)  # uses the pre-update (actor-lagged) weights
            consume(job, episodes)
            job = job_next
        else:
            consume(job, episodes)
            if n < cfg.iterations:
                job = launch(n + 1)

    if actor is not None:
        actor.shutdown(wait=True)
    if build_pp is not None:
        build_pp.close()
        build_pp.join()
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
    parser.add_argument(
        "--deck-pool", type=Path, default=None,
        help="QD archive (JSON deck list): decks come from it, only play is trained",
    )
    parser.add_argument(
        "--no-deck-arm", action="store_true",
        help="don't train the CB/deck head (set automatically with --deck-pool)",
    )
    parser.add_argument(
        "--pipeline", action="store_true",
        help="async actor-learner: collect iter n+1 while training iter n "
             "(actor lags one step; V-Trace corrects). Faster; opt-in until validated",
    )
    parser.add_argument("--smoke", action="store_true", help="3 tiny iterations")
    args = parser.parse_args()

    cfg = Config(
        init_weights=args.weights, out_dir=args.out, engine_json=args.engine_json,
        gate_deck=args.gate_deck, iterations=args.iterations,
        games_per_iter=args.games_per_iter, temperature=args.temperature, lr=args.lr,
        workers=args.workers, native=args.native, eval_every=args.eval_every,
        eval_games=args.eval_games, seed=args.seed, self_play_prob=args.self_play_prob,
        deck_pool=args.deck_pool, pipeline=args.pipeline,
        train_deck=not (args.no_deck_arm or args.deck_pool is not None),
    )
    if args.smoke:
        cfg.iterations, cfg.games_per_iter = 3, 4
        cfg.train_episodes, cfg.batch_size = 16, 8
        cfg.eval_every, cfg.eval_games = 2, 20
    run(cfg)


if __name__ == "__main__":
    main()
