# Pokémon TCG AI Battle Challenge

Work for the Kaggle [**Pokémon TCG AI Battle**](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
competition (Simulation Track). Host: Kaggle × The Pokémon Company × Matsuo Lab × HEROZ.

## What the competition is

- **Not a prediction task.** You submit an **agent** that plays the Pokémon TCG,
  bundled with a 60-card deck and the provided simulator.
- **Agent contract:** `agent(obs_dict) -> list[int]` — given the game state,
  return the chosen option indices. On the initial selection (`obs.select is None`)
  return your 60 card IDs (the deck). The agent must **never crash** (always return
  a legal fallback) and respect a per-move time limit.
- **Scoring:** Elo on a ladder — your agent plays games against similarly-rated
  agents. **Up to 5 submissions per team per day.**
- **Deadline:** 2026-08-16 (Simulation Track). A separate Strategy Track (report,
  prize pool) closes 2026-09-13.
- **Community wisdom:** *deck choice dominates agent quality*, and *the local
  simulator mispredicts ladder results* — the live ladder is the real judge.

## Repo layout

```
.
├── data/                       # downloaded competition data (gitignored)
│   ├── EN_Card_Data.csv        #   ~1,250 cards, one row per move/ability
│   ├── JP_Card_Data.csv
│   └── sample_submission/      #   main.py + deck.csv + cg/ (the engine)
├── notebooks/
│   └── 01_card_data_eda.ipynb  # card-pool EDA (built from the script below)
├── src/
│   ├── cards.py                # card CSV loader + energy/cost/damage parsing
│   ├── agents/                 # swappable policies (pure dict->list[int])
│   │   ├── random_agent.py     #   random legal-move baseline
│   │   └── greedy_agent.py     #   develop-then-attack baseline
│   └── harness/                # Wilson CI + win-rate aggregation (pure)
├── scripts/
│   ├── download_data.sh        # fetch competition data (after accepting rules)
│   ├── build_eda_notebook.py   # regenerate the EDA notebook
│   ├── sim_smoke.py            # verify the simulator runs (Linux only)
│   └── run_eval.py             # battle runner / eval harness (Linux only)
├── results/                    # eval summaries (JSON committed, CSVs gitignored)
└── Dockerfile                  # linux/amd64 box to run the simulator
```

## Setup

```bash
uv sync                         # create the env from pyproject/uv.lock
bash scripts/download_data.sh   # after accepting the rules on the website
```

Accept the competition rules first or downloads 403:
<https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules>

## EDA

```bash
uv run python scripts/build_eda_notebook.py                       # (re)build
uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/01_card_data_eda.ipynb                              # run it
```

Or open `notebooks/01_card_data_eda.ipynb` in Jupyter/VS Code.

## Running the simulator (Linux x86-64 only)

The engine ships only as `cg.dll` (Windows) and `libcg.so` (Linux x86-64) — there
is **no macOS build**. On a Mac / non-amd64 host, run it under Docker emulation:

```bash
docker build --platform=linux/amd64 -t ptcg-sim .
docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim \
    python scripts/sim_smoke.py
```

The engine exposes battle play (`cg.game`), full card/attack data
(`all_card_data()`, `all_attack()`), and a lookahead **search API**
(`search_begin` / `search_step`) usable for MCTS-style planning.

## Evaluation harness (Phase 0)

Play two registered agents head-to-head with first/second slot swapping and a
win-rate + Wilson 95% CI verdict — the "ruler" every later ablation is measured
on (also Linux x86-64, so run it under Docker):

```bash
docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim \
    python scripts/run_eval.py --a greedy --b random --games 500 --seed 0
```

Agents are pure `dict -> list[int]` policies registered in `src/agents/`; add
one to the registry and it's selectable by `--a` / `--b`. Summaries land in
`results/` (per-game CSV + summary JSON).

**Reproducibility caveat:** the engine's RNG (shuffles, coin flips) is *not*
exposed, so individual games can't be replayed bit-for-bit. The harness seeds
agent randomness and relies on large N + slot swap + Wilson CI for
*statistically* reproducible comparisons (which is what keep/drop decisions
need). Measured baseline: `greedy` beats `random` **0.908 [0.879, 0.930]** over
500 games; `random` vs `random` sits at **0.498 / 0.512** (harness is fair).

## Plan & research

- **[PLAN.md](PLAN.md)** — phased, ablation-driven attack plan (eval harness → deck →
  heuristic → search → ISMCTS → learned value → distill), with keep/drop criteria.
- **[docs/research/game-ai-survey.md](docs/research/game-ai-survey.md)** — cited survey of
  game-AI algorithms (MCTS/ISMCTS, MuZero family, CFR/ReBeL/DeepNash) and their fit here.

## Status / next steps

- [x] Data downloaded, card-pool EDA notebook.
- [x] Simulator verified under Docker (`scripts/sim_smoke.py` plays a full game).
- [x] Methods survey + phased plan written.
- [x] Phase 0 — eval harness + baselines (random / greedy); greedy beats random
      0.908 [0.879, 0.930] over 500 games, harness calibrated.
- [ ] Phase 1 — choose deck archetype (biggest lever on Elo).
- [ ] Phases 2+ — heuristic → search → ablations against the live ladder (5 subs/day).
