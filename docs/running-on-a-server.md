# Running on a server (native x86) — deck self-play OSFP

This is a runbook for cloning the repo onto another machine and running the
**deck self-play OSFP** loop (`scripts/train_deck_osfp.py`) at speed.

## Why a different machine

The simulator (`cg`) ships only as `libcg.so` (Linux **x86-64**). On Apple Silicon
it runs under x86 **emulation**, which is a throughput wall, and running parallel
collector containers there was measured *slower* (`--workers` 1/2/6 → 9.6/4.1/3.8
games/s). On a **native x86-64 Linux** box there is no emulation and `--workers N`
**scales near-linearly** across cores — that is the whole reason to move.

> ⚠️ **The scaling only happens with BLAS pinned to 1 thread per container.**
> numpy's OpenBLAS defaults to *all cores per process*, so N collector containers
> each spawn ~`ncores` BLAS threads and oversubscribe the CPU — they then contend
> instead of scaling (this, not emulation alone, is also what hurt the Mac).
> `train_deck_osfp.py` now pins `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1` in
> every `docker run`. **Verified on a 16-core native x86 box (2026-06-22):**
> 64-games/container, identical work, scaled 1→2→4→8→12→16 containers as
> 4.9/9.1/18.3/34.1/48.0/56.9 games/s (≈ linear to ~12, ~11.6× at 16). The *same*
> 8 containers **unpinned** were ~15× slower each (15s → 247s wall) — i.e. without
> the pin, a native box does **not** scale either.

> TL;DR for a native x86-64 Linux server: install Docker + uv, clone, download the
> competition data, build the image, make a BC net, then run `train_deck_osfp.py`
> with `--workers ≈ cores−2`.

---

## 0. Prerequisites

- **A native x86-64 Linux host** (this is the point — avoid ARM/emulation). Check:
  ```bash
  uname -m          # want: x86_64
  nproc             # number of cores (drives --workers)
  ```
- **Docker** (runs the Linux x86-64 sim). `docker run hello-world` should work
  without sudo (add your user to the `docker` group, or prefix commands with sudo).
- **git**, **unzip**, **curl**. (`sudo apt-get install -y git unzip curl`)
- **A Kaggle account that has accepted the competition rules** (required for the
  data download, else `403`): <https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules>
- ~16 GB RAM is comfortable.

---

## 1. Clone

```bash
git clone git@github.com:kohbanye/kaggle-pokemon.git
cd kaggle-pokemon
# until this is merged to main, use the deck-self-play branch:
git checkout feat/phase5d-deck-selfplay
```

## 2. Python env (training runs natively, needs torch+Lightning)

The submission/inference path is pure numpy, but **training** uses torch+Lightning,
which are **dev** dependencies — so install with `--dev`.

```bash
# install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh
exec "$SHELL"                       # pick up uv on PATH

uv sync --dev                       # create the venv (torch, lightning, etc.)
uv run pytest -q                    # sanity: all tests pass natively (no engine needed)
```

## 3. Download the competition data (the `cg` engine)

The Docker sim imports `cg` from `data/sample_submission/cg/` — that comes from the
competition download, so this step is required even though it has nothing to train.

```bash
# Kaggle CLI + credentials
uv run pip install kaggle           # or: pipx install kaggle
mkdir -p ~/.kaggle
# put your token at ~/.kaggle/kaggle.json (Account → Create New API Token), then:
chmod 600 ~/.kaggle/kaggle.json

bash scripts/download_data.sh       # 403? -> accept the rules on the website first
```

This populates `data/sample_submission/cg/libcg.so` (+ `cg/*.py`) and the card CSVs.
`data/` is gitignored — it never gets committed.

## 4. Build the sim image

On native x86-64, `--platform=linux/amd64` is the host platform, so this is a
normal (fast) build — no emulation.

```bash
docker build --platform=linux/amd64 -t ptcg-sim .

# smoke: the engine loads and a game runs
docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim \
    python scripts/sim_smoke.py
```

---

## 5. Make a BC (behaviour-cloning) net — the starting point for self-play

Joint self-play starts from a BC-warm-started net (`data/bc/bc_net_joint.npz`) plus
the engine dump (`data/bc/engine.json`). This net has the **shared card embedding**
wired into *both* the deck head and the play head (Phase 5d), so it must be trained
with the current code — older `bc_net_lstm.npz` files are a different architecture
and won't load.

```bash
# 1) generate teacher self-play games + engine.json (Docker; ~400 games)
docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim \
    python scripts/collect_bc.py --teacher heuristic --games 400 --out data/bc

# 2) train the play+value heads AND the LSTM deck head (native; needs --dev deps)
CUDA_VISIBLE_DEVICES="" uv run python scripts/train_bc.py \
    --data data/bc --out data/bc/bc_net_joint.npz
```

The last line prints a line like
`CB greedy deck: legal=True ... energy=35 pokemon=8 ...` — a balanced, functional
deck means the LSTM deck head trained correctly.

---

## 6. Run joint self-play OSFP (πBT + πCB)

`train_joint_osfp.py` trains the **battle policy (πBT), the value head AND the deck
policy (πCB) together**, sharing the card embedding both heads read (the ByteDance
Hearthstone setup). Each iteration plays games where the deck is sampled from πCB
and the battle is played by πBT, then does one update over both arms. It runs
natively and shells out a collector per batch. On native x86, set `--workers` to
roughly `cores − 2`, and make `--decks ≥ --workers` (the deck batch is split across
workers).

**Use `--native` on this box.** It runs each collector as a plain subprocess
(`import cg` directly) instead of a Docker container, so there is no per-launch
container startup and no docker-daemon contention. Measured here it is faster at
every worker count and is what lets `--workers` keep scaling past ~12. (Drop
`--native` only on an ARM/dev host that needs the x86 container.)

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/train_joint_osfp.py --native \
    --weights data/bc/bc_net_joint.npz \
    --iterations 200 \
    --decks 48 --games-per-deck 16 \
    --workers 12 \
    --out data/jointosfp/run1
```

Quick end-to-end check first (3 tiny iterations):

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/train_joint_osfp.py --smoke --native \
    --weights data/bc/bc_net_joint.npz --out data/jointosfp/smoke
```

### What you get

- `data/jointosfp/run1/joint_final.npz` — the trained net (play + deck + embedding).
- `data/jointosfp/run1/history.json` — per-iteration log.
- Per-iteration stdout: `play`/`deck` sample counts, `mean_wr` (deck self-play win
  rate, ~0.5 by symmetry — that it **varies** is the REINFORCE signal) and `gate`
  (a read-only yardstick: the greedy deck's win rate vs `decklists/metal_aggro.csv`,
  both sides using the trained play head, every `--eval-every`). **Watch the `gate`
  trend** — it now reflects *both* a stronger deck and a stronger play head.

### Tuning knobs

| flag | meaning |
|---|---|
| `--native` | run collectors as subprocesses, no Docker (native x86 only). Faster; scales further. |
| `--workers` | parallel collectors; ≈ `cores − 2` on native x86. |
| `--decks` / `--games-per-deck` | decks sampled per iter / games each is scored over. Bigger `--games-per-deck` = less noisy deck scores. Keep `--decks ≥ --workers`. |
| `--temperature` | play-head sampling temperature for exploration (the play-arm signal). |
| `--self-play-prob` | fraction of games vs the learner itself (rest vs past checkpoints from the OSFP pool). |
| `--eval-every` / `--eval-games` | yardstick cadence / sample size. |
| `--no-eval` | skip the yardstick (pure self-play, fastest). |

---

## 7. Measuring throughput on the new box

```bash
# time one collection batch (128 games) to see games/s and pick --workers
# (--out must be INSIDE the repo: Docker mounts the repo root at /work)
start=$(date +%s)
docker run --platform=linux/amd64 --rm -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
    -v "$PWD":/work -w /work ptcg-sim \
    python scripts/collect_joint_selfplay.py --weights data/bc/bc_net_joint.npz \
      --self-play --decks 8 --games-per-deck 16 --out data/_bench
echo "128 games in $(( $(date +%s) - start ))s"
```

Then sweep `--workers` on a short `train_joint_osfp.py --no-eval --iterations 1` run
to confirm it scales (it pins BLAS internally, so it should). If you hand-roll a
parallel bench with raw `docker run`, add `-e OPENBLAS_NUM_THREADS=1 -e
OMP_NUM_THREADS=1` to **each** container or the processes will fight over cores and
*not* speed up. Context: the reference paper (arXiv:2303.05197) used ~3.2×10⁸ games
— so more cores / more boxes is the lever that matters here.

---

## Notes / troubleshooting

- **Train torch, serve numpy.** Only training needs `uv sync --dev` (torch). The net
  is exported to a numpy `.npz`; a parity test keeps the two forwards identical.
- **`403` on `download_data.sh`.** Accept the competition rules on the website (and
  the API token must be for that same account). It can also be a transient `403` —
  just retry.
- **`docker: permission denied`.** Add your user to the `docker` group
  (`sudo usermod -aG docker $USER`; re-login) or prefix commands with `sudo`.
- **No `cg` / `libcg.so` import errors.** Step 3 didn't complete — the engine lives
  at `data/sample_submission/cg/libcg.so`.
- **Training (`train_bc.py` / `train_joint_osfp.py`) crashes with "The NVIDIA driver
  on your system is too old" / "Cannot re-initialize CUDA in forked subprocess".**
  The box has a GPU whose driver is older than the CUDA the bundled torch was built
  for, so torch half-detects CUDA then dies. Training is CPU-only here anyway — force
  it: prefix the command with `CUDA_VISIBLE_DEVICES=""` (e.g.
  `CUDA_VISIBLE_DEVICES="" uv run python scripts/train_bc.py ...`).
- **The `--platform=linux/amd64` flags** are harmless on a native x86-64 host (they
  match it); they only mean emulation on ARM.
