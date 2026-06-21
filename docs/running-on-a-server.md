# Running on a server (native x86) — deck self-play OSFP

This is a runbook for cloning the repo onto another machine and running the
**deck self-play OSFP** loop (`scripts/train_deck_osfp.py`) at speed.

## Why a different machine

The simulator (`cg`) ships only as `libcg.so` (Linux **x86-64**). On Apple Silicon
it runs under x86 **emulation**, which is the throughput wall: ~10 games/s single
stream, and running parallel collector containers is *slower* (the emulation layer
contends — measured `--workers` 1/2/6 → 9.6/4.1/3.8 games/s). On a **native
x86-64 Linux** box there is no emulation, each container runs at full speed, **and
`--workers N` actually scales** across cores. That is the whole reason to move.

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

Deck self-play starts from a BC-warm-started LSTM net (`data/bc/bc_net_lstm.npz`)
plus the engine dump (`data/bc/engine.json`). Two ways to get them:

### Option A — copy from another machine (fastest)

If you already trained it elsewhere (e.g. the Mac), copy both files over; you can
then skip to step 6. (You still need step 3 for the `cg` engine.)

```bash
# from the machine that has them:
scp data/bc/bc_net_lstm.npz data/bc/engine.json  user@server:~/kaggle-pokemon/data/bc/
```

### Option B — regenerate from scratch (canonical)

```bash
# 1) generate teacher self-play games + engine.json (Docker; ~400 games)
docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim \
    python scripts/collect_bc.py --teacher heuristic --games 400 --out data/bc

# 2) train the play+value heads AND the LSTM deck head (native; needs --dev deps)
uv run python scripts/train_bc.py --data data/bc --out data/bc/bc_net_lstm.npz
```

The last line prints a line like
`CB greedy deck: legal=True ... energy=35 pokemon=8 ...` — a balanced, functional
deck means the LSTM deck head trained correctly.

---

## 6. Run deck self-play OSFP

`train_deck_osfp.py` runs natively and shells Docker for each game-collection
batch. On native x86, set `--workers` to roughly `cores − 2`, and make
`--decks ≥ --workers` (the deck batch is split across workers).

```bash
uv run python scripts/train_deck_osfp.py \
    --weights data/bc/bc_net_lstm.npz \
    --iterations 200 \
    --decks 48 --games-per-deck 16 \
    --workers 12 \
    --out data/deckosfp/run1
```

Quick end-to-end check first (3 tiny iterations):

```bash
uv run python scripts/train_deck_osfp.py --smoke \
    --weights data/bc/bc_net_lstm.npz --out data/deckosfp/smoke
```

### What you get

- `data/deckosfp/run1/deck_final.npz` — the trained net (deck head learned).
- `data/deckosfp/run1/history.json` — per-iteration log.
- Per-iteration stdout: `mean_wr` (self-play win rate, ~0.5 by symmetry — that it
  **varies** is what gives the REINFORCE signal) and `gate` (a read-only yardstick:
  the greedy deck's win rate vs `decklists/metal_aggro.csv`, every `--eval-every`).
  **Watch the `gate` trend** — that is "is the learned deck getting stronger?".

### Tuning knobs

| flag | meaning |
|---|---|
| `--workers` | parallel collector containers; ≈ `cores − 2` on native x86. |
| `--decks` / `--games-per-deck` | decks sampled per iter / games each is scored over. Bigger `--games-per-deck` = less noisy deck scores. Keep `--decks ≥ --workers`. |
| `--self-play-prob` | fraction of games vs the learner itself (rest vs past checkpoints from the OSFP pool). |
| `--eval-every` / `--eval-games` | yardstick cadence / sample size. |
| `--no-eval` | skip the yardstick (pure self-play, fastest). |

---

## 7. Measuring throughput on the new box

```bash
# time one collection batch (128 games) to see games/s and pick --workers
start=$(date +%s)
docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim \
    python scripts/collect_deck_selfplay.py --weights data/bc/bc_net_lstm.npz \
      --self-play --decks 8 --games-per-deck 16 --out /tmp/bench
echo "128 games in $(( $(date +%s) - start ))s"
```

Then sweep `--workers` on a short `train_deck_osfp.py --no-eval --iterations 1` run
to confirm it scales (unlike the emulated Mac, it should). Context: the reference
paper (arXiv:2303.05197) used ~3.2×10⁸ games — so more cores / more boxes is the
lever that matters here.

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
- **The `--platform=linux/amd64` flags** are harmless on a native x86-64 host (they
  match it); they only mean emulation on ARM.
