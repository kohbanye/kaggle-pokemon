"""Three-way co-evolution with AlphaZero: QD (deck) x LSTM-AZ (play) x ISMCTS (search).

Per generation g (all three couplings, AlphaZero substrate):
  A. QD    : qd_deck_search --pilots net --pilot net_{g-1} (warm-start prev archive)
             -> archive_g                                    [couplings 1 + 3]
  B. extract: top-K elite decks -> decklists/coevo/
  C. collect: net_{g-1}-guided PUCT ISMCTS pilots the top-M elites vs the QD population
             (elites + anchors); logs (state, pi, z)         [couplings 1 + 2]
  D. train : LSTM-AZ (policy CE to pi + value MSE to z) over the CUMULATIVE buffer,
             warm-started from net_{g-1} -> net_g            [the AZ apprentice]
  E. gate  : heldout2 (real ladder) with net_g piloting the best elite.

The play net is the recurrent LSTM (RecurrentPolicyValueNet) throughout -- it pilots QD
fitness, guides its own ISMCTS next round, and is what serves. Native/Docker (engine).

  uv run python scripts/coevo_az.py --generations 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

ANCHORS = ["lad00_M_evo_1pz_e1003", "lad01_D_evo_1pz_e905",
           "lad02_P_evo_1pz_e890", "lad03_F_evo_mid_e843"]
COEVO_DIR = ROOT / "decklists" / "coevo"
INIT_NET = ROOT / "data/qdcoevo/run7/round_6/rl/paper_final.npz"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT)  # noqa: S603


def _run_qd(net: Path, seed_archive: Path | None, out: Path,  # noqa: PLR0913
            gens: int, rounds: int, batch: int, n_games: int,
            workers: int, seed: int) -> None:
    cmd = ["uv", "run", "python", "scripts/qd_deck_search.py",
           "--pilots", "net", "--pilot", str(net),
           "--generations", str(gens), "--rounds", str(rounds),
           "--batch", str(batch), "--n-games", str(n_games),
           "--anchors", ",".join(f"decklists/anchors/{a}.csv" for a in ANCHORS),
           "--workers", str(workers), "--seed", str(seed), "--out", str(out)]
    if seed_archive and seed_archive.exists():
        cmd += ["--seed-archive", str(seed_archive)]
    _run(cmd)


def _extract_elites(archive: Path, g: int, k: int) -> list[str]:
    a = json.loads(archive.read_text())
    cells = a["cells"] if isinstance(a, dict) else a
    cells = list(cells.values()) if isinstance(cells, dict) else cells
    cells = [c for c in cells if c.get("deck")]
    cells.sort(key=lambda c: c.get("fitness", -1), reverse=True)
    COEVO_DIR.mkdir(parents=True, exist_ok=True)
    names = []
    for i, c in enumerate(cells[:k]):
        name = f"g{g}_e{i}"
        (COEVO_DIR / f"{name}.csv").write_text(
            "\n".join(str(x) for x in c["deck"]) + "\n")
        names.append(name)
    return names


def _collect(deck: str, net: Path, opps: list[str], out: Path,  # noqa: PLR0913
             games: int, iters: int, workers: int) -> None:
    _run(["uv", "run", "python", "scripts/collect_ismcts.py",
          "--deck", deck, "--net", str(net), "--games", str(games),
          "--iterations", str(iters), "--workers", str(workers),
          "--opp-pilots", "greedy,greedy_plus,heuristic",
          "--opp-decks", ",".join(opps), "--out", str(out)])


def _gate(deck: str, net: Path, games: int, workers: int) -> dict:
    out = ROOT / "results" / f"_az_gate_{deck}.json"
    _run(["uv", "run", "python", "scripts/heldout_eval.py",
          "--pool", "heldout2", "--games", str(games), "--workers", str(workers),
          "--net", str(net), "--subjects", f"net|{deck}", "--out", str(out)])
    d = json.loads(out.read_text())
    o = d["overall"][f"net|{deck}"]
    return {"winrate": o["winrate"], "ci": o["ci"], "n": o["n"],
            "by_pilot": d["per_pilot"][f"net|{deck}"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="QD x LSTM-AZ x ISMCTS co-evolution")
    ap.add_argument("--init-net", type=Path, default=INIT_NET)
    ap.add_argument("--seed-archive", type=Path,
                    default=ROOT / "data/qd_gp/qd_archive.json")
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--qd-gens", type=int, default=6)
    ap.add_argument("--qd-rounds", type=int, default=1)
    ap.add_argument("--qd-batch", type=int, default=8)
    ap.add_argument("--qd-n-games", type=int, default=3)
    ap.add_argument("--elites", type=int, default=6)
    ap.add_argument("--pilot-decks", type=int, default=2)
    ap.add_argument("--play-games", type=int, default=300)
    ap.add_argument("--iters", type=int, default=96, help="ISMCTS iterations")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work", type=Path, default=ROOT / "data/coevo_az")
    ap.add_argument("--out", type=Path, default=ROOT / "results/coevo_az.json")
    args = ap.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    net = args.init_net
    prev_archive = args.seed_archive
    buffer: list[Path] = []
    curve: list[dict] = []

    for g in range(1, args.generations + 1):
        gdir = args.work / f"gen{g}"
        gdir.mkdir(parents=True, exist_ok=True)
        archive = gdir / "archive.json"
        print(f"[g{g}] QD: evolve decks piloted by {net.name} "
              f"(warm {prev_archive.name if prev_archive else 'none'})", flush=True)
        _run_qd(net, prev_archive, archive, args.qd_gens, args.qd_rounds,
                args.qd_batch, args.qd_n_games, args.workers, args.seed + g)
        elites = _extract_elites(archive, g, args.elites)
        pilots = elites[:args.pilot_decks]
        opps = elites + ANCHORS
        print(f"[g{g}] net pilots {pilots} vs opps={opps}", flush=True)
        for d in pilots:
            data = gdir / f"play_{d}.jsonl"
            _collect(d, net, opps, data, args.play_games, args.iters, args.workers)
            buffer.append(data)
        new_net = gdir / "net.npz"
        _run(["uv", "run", "python", "scripts/train_az_lstm.py",
              "--data", ",".join(str(p) for p in buffer), "--warm", str(net),
              "--out", str(new_net), "--epochs", str(args.epochs)])
        best = pilots[0]
        ev = _gate(best, new_net, args.eval_games, args.workers)
        row = {"gen": g, "best_deck": best, "net": str(new_net),
               "elites": elites, **ev}
        curve.append(row)
        print(f"[g{g}] DONE best={best} heldout2_wr={ev['winrate']} CI{ev['ci']} "
              f"by_pilot={ev['by_pilot']}", flush=True)
        args.out.write_text(json.dumps({"gate": "heldout2", "curve": curve}, indent=2))
        net, prev_archive = new_net, archive

    print("\n=== QD x LSTM-AZ co-evolution curve (heldout2) ===")
    for row in curve:
        print(f"  g{row['gen']}  best={row['best_deck']:<10} "
              f"wr={row['winrate']:.3f} CI{row['ci']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
