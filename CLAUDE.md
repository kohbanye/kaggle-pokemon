# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

This repo is for the Kaggle **Pokémon TCG AI Battle** competition (Simulation Track,
deadline 2026-08-16). It is **not a prediction competition** — you submit an *agent*
that plays the Pokémon Trading Card Game, bundled with a 60-card deck and the provided
simulator, and it is scored by **Elo on a ladder** (up to 5 submissions/team/day).

The agent contract is `agent(obs_dict) -> list[int]`: given the game state, return the
chosen option indices. On the initial selection (`obs.select is None`) return the 60
deck card IDs instead. The agent **must never crash** (always return a legal fallback)
and respect a per-move time limit. The evaluation sandbox is **CPU-only, offline (no
internet), ~10 min/game** — so external LLM/API calls at match time are impossible, and
heavy local models are impractical. See `README.md`, `PLAN.md`, and
`docs/research/game-ai-survey.md` for the full strategy.

## Commands

Environment is managed with **uv** (Python 3.12).

```bash
uv sync --dev                         # create/refresh the venv

# Lint / type-check / test (same three checks as CI)
uv run ruff check .
uv run ty check
uv run pytest -q
uv run pytest tests/test_cards.py::test_parse_cost_only_colorless   # single test

# Data (requires accepting competition rules on the website first, else 403)
bash scripts/download_data.sh

# EDA notebook: edit the BUILDER, not the .ipynb, then regenerate + execute
uv run python scripts/build_eda_notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_card_data_eda.ipynb
```

## The simulator is Linux x86-64 only

The `cg` engine ships only as `cg.dll` (Windows) and `libcg.so` (Linux x86-64) — **there
is no macOS build**. Anything that imports `cg` (running battles, the `search_*` lookahead
API, `all_card_data()`) must run under Docker on a non-amd64 host:

```bash
docker build --platform=linux/amd64 -t ptcg-sim .
docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim python scripts/sim_smoke.py

# Eval harness: two agents head-to-head, slot-swapped, win rate + Wilson CI
docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim \
    python scripts/run_eval.py --a greedy --b random --games 500 --seed 0
```

Card-data analysis (`src/cards.py`, the EDA notebook) does **not** need the engine and
runs natively. Because of this split, `ty` is scoped to `src`/`tests` only (scripts that
import the gitignored `cg` are excluded), and `ruff` excludes `notebooks/`.

## Submitting to the ladder

Scoring is an **Elo ladder of agents**, not a fixed test set: a submission first plays a
*Validation Episode* (vs copies of itself, `PENDING`→`COMPLETE`), then keeps playing real
opponents so its **score drifts over hours** — re-check, don't treat the first number as
final. **5 submissions/day.**

The bundle is `submission/main.py` + a deck + `cg/`. `submission/main.py` is a
**self-contained** greedy agent (mirrors `src/agents/greedy_agent.py`; reads `deck.csv`,
builds attack damages from the bundled `cg.all_attack()` at startup, legal fallback on any
error). `ruff`/`ty` skip `submission/` (standalone, deliberately crash-proof).

```bash
# 1. stage build/submission/ = main.py + deck.csv + cg/  (native; --deck picks the deck)
uv run python scripts/build_submission.py --deck decklists/metal_aggro.csv

# 2. package: tar.gz with the files at the archive ROOT (not inside a wrapping dir)
tar -czf build/submission.tar.gz -C build/submission .

# 3. submit  (needs network + an authenticated Kaggle CLI; run OUTSIDE any sandbox)
kaggle competitions submit -c pokemon-tcg-ai-battle -f build/submission.tar.gz -m "msg"

# status / rank
kaggle competitions submissions -c pokemon-tcg-ai-battle
kaggle competitions leaderboard pokemon-tcg-ai-battle --download --path /tmp  # grep your TeamName
```

- **Auth**: `kaggle auth login` (OAUTH, stored at `~/.kaggle/credentials.json`) or a classic
  `~/.kaggle/kaggle.json` API token. `build/` is gitignored.
- **Transient 403**: `CreateSubmission` sometimes returns `403 Forbidden` spuriously even when
  authed + rules-accepted + phone-verified — **just retry**, it goes through. Don't assume it's
  a rules/permission failure.
- Method is **file/tar.gz** (the competition ships a `sample_submission/` *folder*); Notebook
  submission also exists but we use the CLI.

## Architecture / where things live

- `data/` — **gitignored** competition download. Not present until `download_data.sh` runs.
  - `EN_Card_Data.csv` / `JP_Card_Data.csv` — ~1,250 cards, **one row per move/ability**
    (a card with N attacks spans N rows sharing a `Card ID`).
  - `sample_submission/` — `main.py` (reference agent) + `deck.csv` + `cg/` (the engine).
- `src/cards.py` — the card-data loader. Turns the raw per-move CSV into two tidy views:
  `load_moves()` (one row per move) and `load_cards()` (one row per unique card, for
  deckbuilding). Also parses energy notation: `{X}` = colored energy, `●` = colorless
  (`parse_cost("{D}●●") -> {"D":1,"C":2}`). The pure parsers are unit-tested in `tests/`.
- `src/agents/` — swappable policies, all **pure `dict -> list[int]`** (the exact Kaggle
  `agent(obs)` contract) so they import and unit-test natively with no `cg` engine. Engine-
  derived data (e.g. attack damages) is *injected* by the runner, never imported here.
  `base.py` holds the option/select-type int constants and the legal-fallback/legality
  helpers; `REGISTRY` (in `__init__.py`) maps names used by `--a`/`--b`. Baselines:
  `random` and `greedy` (develop-the-board-then-attack; it must develop, not attack ASAP —
  attacking with an empty bench loses on "no Active Pokémon").
- `src/harness/` — **pure** result attribution (`result.py`) and Wilson-CI win-rate
  aggregation (`stats.py`); unit-tested. The engine-touching match driver is **not** here.
- `scripts/run_eval.py` — the Phase-0 battle runner (imports `cg`, **Docker-only**): drives
  N games with first/second slot swap, routes each selection by `current.yourIndex`, seeds
  agent RNG per game, enforces legality (never crashes), and writes CSV/JSON to `results/`.
  ⚠️ The engine RNG is **not** seedable (no public API), so games aren't bit-reproducible —
  comparisons rely on large N + swap + CI, not per-game determinism.
- `cg/api.py` (in the download) is the source of truth for the engine: `Observation`/
  `SelectData`/`Option` dataclasses, the `Select*Context`/`OptionType` enums, and the
  `search_begin`/`search_step`/`search_end` lookahead API used for determinized tree search.
- `notebooks/01_card_data_eda.ipynb` — **generated**, do not hand-edit; the source of
  truth is `scripts/build_eda_notebook.py`.
- `PLAN.md` — the phased, **ablation-driven** plan (eval harness → deck → heuristic →
  search/PIMC → ISMCTS → learned value → distill). New methods are added one at a time and
  kept only if they beat the prior baseline by a measured margin.

## Working conventions

- **Don't commit `data/`** (CSVs, the large card PDFs, and `*.so`/`*.dll` are gitignored).
  The agent submission must be a self-contained bundle, but that bundle is built from the
  downloaded `cg/`, not from tracked files.
- Two-tier evaluation: the local sim is for fast relative comparisons, but it is known to
  **mispredict the real ladder** — validate big decisions on the ladder. Deck choice is the
  single biggest lever on Elo.
- `ruff` runs with `select = ["ALL"]`; prefer fixing or a scoped `per-file-ignores` entry
  over broad suppressions.
