"""ReBeL self-play value-net training loop (design step F).

Round 0 collects (PBS features, subgame root value) targets from `RebelAgent` solves
that use the PRIZE-HEURISTIC leaf (so the first targets anchor to a non-self-referential
signal, dodging the pure-bootstrap trap Codex warned about), then trains a `ValueNet` on
them. Each later round re-collects with the freshly-trained net AS the leaf and retrains
(warm-started) -- the ReBeL value-iteration. Exports the numpy `ValueNet` per round.

Slow (each solve drives the engine, plus rollouts for the value target), so defaults are
tiny; scale --games / --rounds when perf work (step G) lands.

  uv run python scripts/rebel_selfplay.py --rounds 2 --games 3 --out data/rebel/vn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from multiprocessing import Pool  # noqa: E402

from scripts.run_eval import load_engine_data, play_game, resolve_deck  # noqa: E402
from src.agents import build_agent  # noqa: E402
from src.rebel.agent import RebelAgent  # noqa: E402
from src.rebel.value_net import VALUE_IN_DIM, ValueNet  # noqa: E402

_W: dict = {}


def _winit(deck_name: str, params: dict, net_path: str | None,  # noqa: PLR0913
           opps: list[str], target: str, recurrent: bool,  # noqa: FBT001
           vector: bool = False) -> None:  # noqa: FBT001, FBT002
    _W["engine"] = load_engine_data()
    _W["deck"] = resolve_deck(deck_name)
    _W["params"] = params
    _W["recurrent"] = recurrent
    _W["vector"] = vector
    if net_path and vector:
        from src.rebel.vector_value_net import VectorValueNet  # noqa: PLC0415
        _W["net"] = VectorValueNet.load(net_path)
    elif net_path and recurrent:
        from src.rebel.recurrent_value_net import RecurrentValueNet  # noqa: PLC0415
        _W["net"] = RecurrentValueNet.load(net_path)
    else:
        _W["net"] = ValueNet.load(net_path) if net_path else None
    _W["opps"] = opps
    _W["target"] = target


def _wgame(task: dict) -> list[tuple]:
    from scripts.heldout_eval import _ForcedFirst  # noqa: PLC0415
    if _W.get("vector"):  # vector_net leaf (if trained yet) + record vector targets
        net_kw: dict = {"vector_record": True}
        if _W["net"] is not None:
            net_kw["vector_net"] = _W["net"]
    elif _W.get("recurrent"):
        net_kw = {"recurrent_net": _W["net"]}
    else:
        net_kw = {"value_net": _W["net"]}
    agent = RebelAgent(_W["deck"], _W["engine"], record=True,
                       seed=task["seed"], **net_kw, **_W["params"])
    name = _W["opps"][task["seed"] % len(_W["opps"])]  # round-robin diverse opponents
    if name == "greedyFF":
        opp: object = _ForcedFirst(build_agent("greedy", _W["deck"], _W["engine"]))
    else:
        opp = build_agent(name, _W["deck"], _W["engine"])
    sf = task["sf"]
    p0, p1 = (agent, opp) if sf else (opp, agent)
    res = play_game(p0, p1, a_is_player0=sf, seed=task["gseed"])
    if _W.get("target") == "outcome" and not _W.get("vector"):
        # TD(1): GROUND every recorded state in the ACTUAL game result (acting player =
        # the agent's fixed seat), not the self-bootstrapped CFR root value. z in the
        # acting-player perspective the samples already use (see _record_sample).
        z = 1.0 if res.a_won else -1.0
        return [(x, z) for x, _ in agent.samples]
    return agent.samples


def _collect(deck_name: str, net_path: str | None, games: int,  # noqa: PLR0913
             params: dict, seed0: int, workers: int,
             opps: list[str], target: str,
             recurrent: bool = False,  # noqa: FBT001,FBT002
             vector: bool = False) -> list[tuple]:  # noqa: FBT001,FBT002
    tasks = [{"seed": seed0 + g, "sf": g % 2 == 0, "gseed": 7000 + seed0 + g}
             for g in range(games)]
    with Pool(workers, initializer=_winit,
              initargs=(deck_name, params, net_path, opps, target, recurrent,
                        vector)) as pp:
        return [s for game in pp.map(_wgame, tasks) for s in game]


def _train(samples: list[tuple[np.ndarray, float]], warm: ValueNet | None,  # noqa: PLR0913
           epochs: int, lr: float, hidden: tuple[int, ...],
           weight_decay: float = 0.0) -> ValueNet:
    import torch  # noqa: PLC0415

    from src.rebel.value_net_torch import (  # noqa: PLC0415
        TorchValueNet,
        from_numpy_value_net,
    )

    x = torch.tensor(np.stack([s[0] for s in samples]))
    y = torch.tensor(np.array([s[1] for s in samples]))
    net = (from_numpy_value_net(warm) if warm is not None
           else TorchValueNet(VALUE_IN_DIM, hidden).double())
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(epochs):
        opt.zero_grad()
        loss = ((net(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        if ep == 0 or ep == epochs - 1:
            print(f"    epoch {ep}: MSE={float(loss):.4f}")
    return net.to_numpy_net()


def _train_recurrent(samples: list, warm: object, epochs: int,  # noqa: PLR0913
                     lr: float, hidden: int, weight_decay: float,
                     max_len: int = 24) -> object:
    """Train the LSTM value net on (feature-seq, target) samples (pad+lengths)."""
    import torch  # noqa: PLC0415

    from src.rebel.recurrent_value_net import RecurrentValueNet  # noqa: PLC0415
    from src.rebel.recurrent_value_net_torch import (  # noqa: PLC0415
        TorchRecurrentValueNet,
        from_numpy_recurrent,
    )

    seqs = [s[0][-max_len:] for s in samples]  # truncate to last max_len steps
    lengths = torch.tensor([len(s) for s in seqs])
    tmax = int(lengths.max())
    xb = np.zeros((len(seqs), tmax, VALUE_IN_DIM))
    for i, s in enumerate(seqs):
        xb[i, : len(s)] = s
    x = torch.tensor(xb)
    y = torch.tensor(np.array([s[1] for s in samples]))
    net = (from_numpy_recurrent(warm) if isinstance(warm, RecurrentValueNet)
           else TorchRecurrentValueNet(VALUE_IN_DIM, hidden).double())
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(epochs):
        opt.zero_grad()
        loss = ((net(x, lengths) - y) ** 2).mean()
        loss.backward()
        opt.step()
        if ep == 0 or ep == epochs - 1:
            print(f"    epoch {ep}: MSE={float(loss):.4f}")
    return net.to_numpy_net()


def _train_vector(samples: list, warm: object, epochs: int,  # noqa: PLR0913
                  lr: float, hidden: tuple[int, ...],
                  weight_decay: float) -> object:
    """Train the infostate VECTOR net on (public x, per-hyp target, mask) with a MASKED
    MSE -- only hypotheses that appeared in a solve contribute their loss."""
    import torch  # noqa: PLC0415

    from src.rebel.vector_value_net import VALUE_IN_DIM, VectorValueNet  # noqa: PLC0415
    from src.rebel.vector_value_net_torch import (  # noqa: PLC0415
        TorchVectorValueNet,
        from_numpy_vector,
    )

    x = torch.tensor(np.stack([s[0] for s in samples]))
    y = torch.tensor(np.stack([s[1] for s in samples]))
    m = torch.tensor(np.stack([s[2] for s in samples]))
    k = int(y.shape[1])
    net = (from_numpy_vector(warm) if isinstance(warm, VectorValueNet)
           else TorchVectorValueNet(VALUE_IN_DIM, k, hidden).double())
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(epochs):
        opt.zero_grad()
        se = ((net(x) - y) ** 2) * m  # masked squared error
        loss = se.sum() / m.sum().clamp(min=1.0)
        loss.backward()
        opt.step()
        if ep == 0 or ep == epochs - 1:
            print(f"    epoch {ep}: masked-MSE={float(loss):.4f}")
    return net.to_numpy_net()


def main() -> None:  # noqa: PLR0915, PLR0912, C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="metal_aggro")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--particles", type=int, default=3)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--tpw", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=None,
                    help="growing-tree width bound during collection (fast, ~ties gp)")
    ap.add_argument("--opps", default="greedy,greedy_plus,heuristic,greedyFF",
                    help="comma opponent pilots, round-robin (diverse = generalises)")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--init-net", type=Path, default=None,
                    help="warm-start value-iteration from this net (round 0 uses it as "
                         "the leaf) -- to CONTINUE a rising learning curve")
    ap.add_argument("--target", choices=("cfr", "outcome"), default="cfr",
                    help="value target: cfr root value (bootstrap) or outcome=TD(1) "
                         "grounded in the actual game result (grounds the net)")
    ap.add_argument("--hidden", default="64",
                    help="value-net hidden sizes, comma-sep (e.g. 256,256 for bigger)")
    ap.add_argument("--replay", type=int, default=1,
                    help="sliding replay: train on the last N rounds' samples")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="Adam weight decay (L2) -- regularise bigger nets")
    ap.add_argument("--recurrent", action="store_true",
                    help="history-aware LSTM value net (h_t threading)")
    ap.add_argument("--vector", action="store_true",
                    help="proper-ReBeL infostate VALUE VECTOR net (per-hypothesis CFV; "
                         "round 0 uses the heuristic leaf and records vector targets)")
    ap.add_argument("--out", type=Path, default=ROOT / "data/rebel/vn")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    params = {"n_particles": args.particles, "max_depth": args.depth,
              "sweeps": args.sweeps, "traversals_per_world": args.tpw,
              "top_k": args.top_k}
    opps = [o for o in args.opps.split(",") if o]
    hidden = tuple(int(h) for h in str(args.hidden).split(",") if h)
    net: object | None = None
    net_path: str | None = str(args.init_net) if args.init_net else None
    if args.vector:
        # round 0 net=None => heuristic leaf, still records vector targets; the net's
        # out_dim (K = #hypotheses) is set from the first round's target width at train.
        if args.init_net:
            from src.rebel.vector_value_net import VectorValueNet  # noqa: PLC0415
            net = VectorValueNet.load(args.init_net)
    elif args.recurrent:
        from src.rebel.recurrent_value_net import RecurrentValueNet  # noqa: PLC0415
        if net_path is None:  # round-0 = random LSTM net (heuristic-led)
            net = RecurrentValueNet.random(np.random.default_rng(0), hidden=hidden[0])
            net_path = str(args.out.with_name(f"{args.out.name}_init.npz"))
            net.save(net_path)
        else:
            net = RecurrentValueNet.load(args.init_net)
    elif args.init_net:
        net = ValueNet.load(args.init_net)

    from collections import deque  # noqa: PLC0415
    replay: deque = deque(maxlen=max(1, args.replay))  # sliding buffer of round samples
    for r in range(args.rounds):
        # recurrent round 0 (fresh, untrained LSTM) uses the HEURISTIC leaf so play is
        leaf = ("lstm" if args.recurrent and params.get("lstm_leaf") else
                "heuristic")
        if args.recurrent:
            params["lstm_leaf"] = args.init_net is not None or r > 0
        leaf = ("lstm" if args.recurrent and params.get("lstm_leaf")
                else "heuristic")
        if args.vector:
            leaf = "vector" if (net is not None) else "heuristic"
        print(f"[round {r}] collect {args.games} games x{args.workers}w "
              f"(leaf={leaf}, opps={opps}, target={args.target})", flush=True)
        samples = _collect(args.deck, net_path, args.games, params,
                           seed0=r * 100, workers=args.workers, opps=opps,
                           target=args.target, recurrent=args.recurrent,
                           vector=args.vector)
        if not samples:
            print("  no samples collected; increase games/decisions")
            return
        replay.append(samples)
        train_set = [s for batch in replay for s in batch]  # sliding-replay union
        if args.vector:  # masked stats over observed hypothesis slots
            tv = np.stack([s[1] for s in samples])
            mv = np.stack([s[2] for s in samples])
            obs_mean = float((tv * mv).sum() / max(mv.sum(), 1.0))
            print(f"  {len(samples)} samples (train on {len(train_set)}), "
                  f"K={tv.shape[1]}, ~{mv.sum(1).mean():.1f} hyps/solve, "
                  f"target mean={obs_mean:.3f}", flush=True)
        else:
            ys = np.array([s[1] for s in samples])
            print(f"  {len(samples)} samples (train on {len(train_set)} from "
                  f"{len(replay)} rounds), target mean={ys.mean():.3f} "
                  f"std={ys.std():.3f}", flush=True)
        if args.vector:
            net = _train_vector(train_set, net, args.epochs, args.lr, hidden,
                                args.weight_decay)
        elif args.recurrent:
            net = _train_recurrent(train_set, net, args.epochs, args.lr, hidden[0],
                                   args.weight_decay)
        else:
            net = _train(train_set, net, args.epochs, args.lr, hidden,
                         args.weight_decay)
        out = args.out.with_name(f"{args.out.name}_r{r}.npz")
        net.save(out)
        net_path = str(out)
        print(f"  saved -> {out}", flush=True)
    print("ReBeL self-play value-net loop OK")


if __name__ == "__main__":
    main()
