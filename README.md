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
├── data/                       # downloaded data + trained nets/archives (gitignored)
│   └── sample_submission/      #   reference main.py + deck.csv + cg/ (the engine)
├── decklists/                  # 60-card decks: metas + anchors/ + candidates/ + coevo/
├── notebooks/01_card_data_eda.ipynb   # card-pool EDA (generated from a builder script)
├── src/
│   ├── cards.py                # card CSV loader + energy/cost/damage parsing
│   ├── deck.py                 # card pool + legal deck building
│   ├── agents/                 # swappable policies (pure dict->list[int]); REGISTRY
│   │   ├── greedy_agent.py     #   develop-then-attack baseline
│   │   ├── heuristic_agent.py  #   tempo/positional heuristic (greedy_plus config)
│   │   └── recurrent_agent.py  #   the LSTM policy/value net, numpy serving
│   ├── net/                    # policy/value/deck net: encode → serve(numpy)/train(torch)
│   ├── search/                 # ISMCTS (AlphaZero), determinizer, opponent belief
│   ├── qd/                     # QD / MAP-Elites deck search library
│   └── harness/                # Wilson CI + win-rate aggregation (pure)
├── scripts/                    # entry points -- see the "Current pipeline" section below
├── results/                    # eval summaries (JSON committed, CSVs gitignored)
└── Dockerfile                  # linux/amd64 box to run the simulator
```

## Current pipeline (QD × AlphaZero)

A submission is `(deck, pilot)`. Two levers, co-evolved:
**deck** via QD/MAP-Elites, **play** via an LSTM AlphaZero net trained on ISMCTS search.

```
scripts/coevo_az.py         ── the flywheel: for each generation
  ├─ qd_deck_search.py      ──   QD evolves decks, piloted by the current net   [deck]
  ├─ collect_ismcts.py      ──   net-guided PUCT ISMCTS self-play → (state,π,z)  [search]
  ├─ train_az_lstm.py       ──   LSTM-AZ: policy CE(π) + value MSE(z)            [play]
  └─ heldout_eval.py        ──   gate on the real-ladder held-out pool (heldout2)
scripts/build_submission.py ── bundle main.py + deck.csv + cg/ for the ladder
```

Everything else in `scripts/` is tooling: `run_eval.py` (base battle harness),
`nashconv_eval.py` (exploitability), `ismcts_timing.py` (search perf), `ladder_episodes.py`
+ `build_heldout_v2.py` (refresh the eval pool from ladder replays), `build_eda_notebook.py`.
**New here? Start with `CLAUDE.md`, then read `scripts/coevo_az.py`.**

## Setup

```bash
uv sync                         # create the env from pyproject/uv.lock
bash scripts/download_data.sh   # after accepting the rules on the website
```

Accept the competition rules first or downloads 403:
<https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules>

To clone and run the deck self-play loop on another (native x86-64) machine for
speed, see [docs/running-on-a-server.md](docs/running-on-a-server.md).

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

## Submitting to the ladder

Build the self-contained bundle (`submission/main.py` + a deck + `cg/`), package
it as a `tar.gz`, and submit with the Kaggle CLI (authenticated, online):

```bash
uv run python scripts/build_submission.py --deck decklists/metal_aggro.csv  # -> build/submission/
tar -czf build/submission.tar.gz -C build/submission .
kaggle competitions submit -c pokemon-tcg-ai-battle -f build/submission.tar.gz -m "msg"
kaggle competitions submissions -c pokemon-tcg-ai-battle                    # check status/score
```

Scoring is an **Elo ladder** (5 subs/day): a submission validates (`PENDING`→
`COMPLETE`) then keeps playing, so the score drifts — re-check it. See **CLAUDE.md
→ "Submitting to the ladder"** for auth setup and the transient-403 retry note.

## Plan & research

- **[PLAN.md](PLAN.md)** — phased, ablation-driven attack plan converging on the OSFP
  target (eval harness → deck-space scaffolding → heuristic → net skeleton (CB+BT heads)
  → BC warm-start → OSFP self-play → test-time search → distill), with keep/drop criteria.
- **[docs/research/osfp-cardgame-2303.05197.md](docs/research/osfp-cardgame-2303.05197.md)** —
  the **target paper** (Hearthstone via end-to-end policy + Optimistic Smooth Fictitious
  Play): OSFP, the no-search architecture, improved techniques, and how it maps to this comp.
- **[docs/research/game-ai-survey.md](docs/research/game-ai-survey.md)** — cited survey of
  game-AI algorithms (MCTS/ISMCTS, MuZero family, CFR/ReBeL/DeepNash) and their fit here.

## Status / next steps

- [x] Data downloaded, card-pool EDA notebook.
- [x] Simulator verified under Docker (`scripts/sim_smoke.py` plays a full game).
- [x] Methods survey + phased plan written.
- [x] Phase 0 — eval harness + baselines (random / greedy); greedy beats random
      0.908 [0.879, 0.930] over 500 games, harness calibrated.
- [x] Phase 1 — deck-space scaffolding: legality + legal-deck mask (`src/deck.py`,
      engine-confirmed), coherent demo decks (`src/deckbuild.py` → `decklists/`),
      name→id importer (`src/decklists.py`), round-robin deck-eval
      (`scripts/run_deck_eval.py`; deck spread 0.683 ≫ ~0.50 agent diff), and a
      self-contained greedy submission bundle (`submission/`, Docker-smoked).
      Remaining: the actual ladder submission (`scripts/build_submission.py` →
      `kaggle competitions submit`, needs credentials).
- [ ] Phase 3+ — net skeleton (CB+BT heads) → BC warm-start → OSFP self-play →
      test-time search → distill, validated against the live ladder (5 subs/day).
