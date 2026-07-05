# Candidate decks (Phase 1)

Each file here is one candidate deck: a whitespace-separated list of **exactly 60
card IDs** (same format as `data/sample_submission/deck.csv`), named by its file
stem. `scripts/run_tournament.py` reads every `*.txt` / `*.csv` in this
directory (plus the sample deck) and runs an agent-fixed round robin.

```
# decks/fire-aggro.txt  (example — real IDs come from the card pool)
12 12 12 12 34 34 ...   # 60 IDs total
```

Legality (enforced by `src.eval.deck.validate_deck`, card-aware sets built from
`src.cards`):

- exactly 60 cards,
- at most 4 copies of any card **except basic energy**,
- at most 1 ACE SPEC card.

> Real archetype decks need the competition card pool (card IDs), which lives in
> the gitignored `data/` download. Author decks here once
> `bash scripts/download_data.sh` has run and the EDA has surfaced viable
> archetypes.
