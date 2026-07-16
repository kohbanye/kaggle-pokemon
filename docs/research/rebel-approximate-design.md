# Approximate ReBeL for Pokémon TCG, combined with QD deck search

Status: **design** (2026-07-11). Supersedes the AlphaZero-style play arm, which was shown
to be a self-confirming fixed point on this imperfect-information game (see
`memory: az-lstm-learning-failure-rootcause`, and `docs/research/paper-faithful-rewrite.md`
for the AZ arm it replaces). This document specifies an **approximate ReBeL** play solver
(MCCFR + particle/archetype-bucketed belief + a batched infostate-value network) and how it
composes with the **QD deck search** (which is our only empirically-proven Elo lever) as a
two-level PSRO.

The goal is not to reproduce exact ReBeL (intractable here — see §2) but to keep its two
properties that fix our failure: **(i) the opponent minimizes regret *inside* the search**
(so "fast punishes slow" appears as counterfactual regret, not a hand-tuned roller — this
preserves self-play and does not cap the ceiling at greedy_plus), and **(ii) the value
target is a local subgame infostate value, not a broadcast terminal ±1** (so the early-game
signal has SNR).

---

## 1. Why ReBeL, and what exactly it changes

Our AZ arm failed because, on an imperfect-information game, determinized MCTS is not an
improvement operator: the in-search opponent was the weak net itself, the value target was
a broadcast ±1, and the distilled π collapsed to the (tempo-biased) prior. ReBeL replaces
determinized-PUCT with **CFR over Public Belief States (PBS)** and, in 2p zero-sum, provably
converges to a Nash equilibrium (Brown et al. 2020, arXiv:2007.13544; in perfect-info it
reduces to AlphaZero).

| current AZ arm | approximate ReBeL |
|---|---|
| ISMCTS determinization (one world) | **PBS**: public state + belief over both players' private infostates |
| in-search opponent = fixed weak net roller | **CFR** solves *both* players' strategies in the subgame |
| PUCT visit counts | CFR cumulative regret / **average strategy** |
| scalar value V(s) trained on broadcast ±1 z | **infostate value vector** = CFR subgame root values |
| π = MCTS visit distribution | π = **CFR average strategy** (net is only a warm start) |
| LSTM implicitly remembers opponent info | belief is an **explicit PBS input** (LSTM optional) |

Effect on the diagnosed links (`az-lstm-learning-failure-rootcause`):
- **A** (uninformative early value) → value target is the local subgame infostate value.
- **B** (tempo bias from weak opponent) → the opponent regret-minimizes in-search; slow-loses-to-fast is discovered as its counterfactual regret. **No hand-tuned roller, ceiling not capped.**
- **C** (visit-prior distillation, degenerate recurrence) → policy target is CFR average strategy; belief is explicit so recurrent memory is optional, not load-bearing.

---

## 2. The tractability crux and our approximations

Exact ReBeL needs a **tractable enumeration of each player's private infostates at a public
state** (poker: ~1,326 hands/player). Pokémon TCG's private state = *hand contents + hidden
deck ORDER + face-down prize contents + the opponent's unknown decklist* — astronomically
large. So **exact PBS/CFR is impossible**; we approximate on four axes, each with a knob:

1. **Opponent decklist belief → archetype buckets.** Instead of a distribution over 60-card
   lists, keep a categorical belief over a fixed hypothesis set of archetype decklists
   (reuse `src/net/opp_context.build_hypotheses` + `src/search/opp_belief`), Bayes-updated
   by revealed cards. Knob: number of buckets `B` (~10–40).
2. **Hidden card assignment → weighted particles.** Within a bucket, a *particle* is a full
   world: (opp hand cards, opp prize cards, opp deck order, our prize cards, our deck order).
   A particle maps **directly** onto one `cg.search_begin(obs, your_deck, your_prize,
   opp_deck, opp_prize, opp_hand, opp_active)` call — the engine is already a fully
   determinized world simulator, which is exactly what particle CFR needs. Knob: particle
   count `P` per solve (~32–256).
3. **Depth-limited public subgames.** Solve only a shallow public subgame (e.g. to the next
   turn boundary, or `D` public decisions ahead); beyond the leaf, call the **infostate
   value network**. Knob: subgame depth `D` (~2–8 decisions / 1 turn).
4. **Action abstraction.** The raw option list per decision is large; optionally bucket
   equivalent options (e.g. "attach energy to the same-role target"). Knob: on/off + rules.

These give up ReBeL's exactness guarantee — this is **approximate** CFR (closer to
Online-Outcome-Sampling ExIt / particle CFR than to tabular ReBeL). §8 is the validation
plan that keeps us honest (NashConv must actually fall).

---

## 3. Public / private decomposition (grounded in `cg/api.py`)

From `PlayerState` / `State`:

- **Public** (both players observe): `active`/`bench` Pokémon with attached energy, damage,
  tools; `benchMax`; `deckCount` and `handCount` (counts only); the **full `discard`
  list**; face-up prizes and prize *count*; status conditions; `turn`, `turnActionCount`,
  `firstPlayer`, `supporterPlayed`/`stadiumPlayed`/`energyAttached`/`retreated`; `stadium`.
- **Private** (per player): own `hand` contents (opponent's `hand` is `None`); own deck
  **order** (future draws — a chance source; only `deckCount` public); own face-down `prize`
  contents; and the **opponent's decklist identity** (never directly observed).

Consequence that makes belief cheap for *our* side: we know our own 60-card list, so our
hidden state is only a *partition + ordering* of `(deck ∪ hand ∪ prizes) = 60 − board −
discard` cards. Discard being public sharply constrains beliefs as the game proceeds.

**PBS definition** (from the acting player `i`'s perspective):

```
PBS β = ( public_state s_pub ,
          belief over particles  { (bucket b, world w) ↦ weight } )
```

where a *world* `w` fully specifies both players' hidden state and is `search_begin`-able.

---

## 4. Components

### 4.1 Belief tracker (`src/rebel/belief.py`, new — engine-free, unit-testable)
- Maintains the archetype-bucket posterior (reuse `opp_belief.consistency_scores`) and a
  set of weighted particles consistent with `s_pub` (discard, counts, revealed cards).
- **Public-action Bayes update**: after player `j`'s public action `a`, reweight particle
  `w` by `σ_j(a | I_j(w))` — "if you held this hidden state, how often does your current
  strategy play `a`" — using the *current CFR strategy* (this is the ReBeL coupling: belief
  moves with the solved strategy, not a fixed roller).
- **Resampling** when effective particle count drops (SIR particle filter).
- Pure Python over card multisets; no `cg`.

### 4.2 Subgame solver — external-sampling MCCFR (`src/rebel/mccfr.py`, new; engine-bound)
Depth-limited public subgame rooted at `β`. Per CFR iteration `t`:
1. Sample a world `w ~ belief`, `search_begin(...)` it (determinized engine handle).
2. **Traverse** the public subgame with external sampling: at the traverser's infostate
   nodes, evaluate all legal (abstracted) actions via `search_step`; at the opponent /
   chance nodes, sample one action from the current strategy `σ_j` / chance.
3. At a **leaf** (depth `D` or turn boundary), instead of rolling to terminal, query the
   **infostate value net** (§4.3) for the leaf PBS — this replaces the AZ value-head-at-leaf
   but returns a *vector over infostates*, not a scalar.
4. Update counterfactual **regrets** and the **average strategy** (Linear CFR weighting).
Return `(average_strategy σ̄, root infostate value vector v̄)`. MCCFR gives regret updates
that are unbiased vs full CFR in expectation, so trajectory sampling scales where full-tree
traversal cannot (Lanctot et al.; Waugh MCCFR).

> The single hardest implementation point is that `cg` only advances *determinized* worlds
> via `search_step`; there is no native public-tree/infostate API. We therefore build the
> public tree ourselves by (a) fixing a world per traversal, (b) keying regret tables by the
> *observable* infostate signature (public state + the traverser's own private cards), and
> (c) sharing regret across worlds that share an infostate — this is what makes it CFR over
> information sets rather than per-world PIMC (avoids strategy fusion).

### 4.3 Infostate value network (`src/rebel/value_net.py` — torch train / numpy serve)
- **Input**: a PBS embedding = `public_encoder(s_pub)` ⊕ `belief_encoder(Σ weight ·
  particle/bucket embedding)`. Reuse `src/net/features.py` + `src/net/encode.py` for the
  public/card features and `opp_context` for the belief summary.
- **Output**: a **value per infostate** for a queried player. Because the infostate set is
  large, use the **query form**: `V(pbs_embedding, player, private_state) → v ∈ [−1,1]`,
  batched over the private states the CFR traversal actually needs (GPU batch per leaf).
- **Target**: the CFR **root infostate values** `v̄(β)` from solving deeper subgames /
  (Phase 1) exact solve — NOT the broadcast ±1. This is what fixes link A.
- Keep the numpy-serve / torch-train split and a parity test, exactly as the current net
  (`src/net/recurrent_torch.py` ↔ `recurrent_model.py`) does.

### 4.4 Policy net (optional warm start)
CFR's average strategy is the policy; a policy net is trained on `σ̄(β)` only to **warm-start**
CFR (fewer iterations to converge) — it is not the played policy. Can reuse the current
`RecurrentPolicyValueNet` policy head initially.

---

## 5. Self-play loop (ReBeL-recursive) — `scripts/rebel_selfplay.py` (new)
```
β ← initial_pbs(deck_matchup)
while not terminal(β):
    σ̄, v̄ = solve_subgame(β, value_net, policy_net, iters=T)     # §4.2
    value_buffer.add(β, v̄)                                        # value targets
    policy_buffer.add(β, σ̄)                                       # warm-start targets
    t ~ sample_cfr_iteration(weight="linear")                     # ReBeL: random iter
    β ← sample_leaf_pbs(β, strategy_at(t))                        # recurse from that iter
train(value_net, value_buffer); train(policy_net, policy_buffer)
```
Sampling the next PBS from a **random CFR iteration** (not just the average) is ReBeL's
trick that makes the value net accurate on the beliefs the solver actually visits, and is
part of safe search at test time.

**At serve time** the agent runs the same depth-limited solve at the real root and plays
`σ̄`; belief is tracked online (§4.1). No determinization fallback bug (unlike current
ISMCTS, which throws on unknown decks — the bucketed belief always yields legal worlds).

---

## 6. Combining with QD deck search — a two-level PSRO

QD deck search is our **proven** Elo lever; ReBeL is a **play** solver for a *fixed* deck
matchup distribution. Composing them naively (optimize deck and play separately) invites
non-transitive deck-play cycles. The principled structure is **PSRO / double-oracle** over
`(deck, play)` oracles:

```
Outer (meta-game):  strategies o_k = (deck d_k, play π_k)
                    payoff M[i][j] = U(o_i, o_j)  via the eval harness (run_eval)
                    meta-solver → mixture ν over strategies   (Nash / α-rank)
Two best-response oracles add to the population each epoch:
  • DECK oracle  = QD/MAP-Elites best-response deck vs ν      (reuse qd_deck_search.py)
  • PLAY oracle  = ReBeL solver best-response play vs ν       (§4–5, deck fixed to top decks)
Gate = exploitability where the opponent picks BOTH deck and play (az_nashconv.py).
```

- **Deck belief inside ReBeL**: since the opponent's deck is hidden, its archetype bucket is
  part of the PBS private state (§3). The outer PSRO population *defines the bucket set* — so
  QD-discovered decks and anchors become the belief hypotheses the ReBeL solver reasons over.
  This is the clean coupling: **QD explores decks; those decks become ReBeL's opponent-belief
  support; ReBeL best-responds in play; the meta-solver keeps it non-exploitable.**
- The existing `same-deck net − greedy_plus` (`az_eval.py`) stays as a *play* diagnostic; the
  *final* gate is the two-sided exploitability, which QD-only cannot game.

---

## 7. What we reuse vs build

Reuse: `cg` determinized engine (`search_begin/step`), `src/net/features.py` + `encode.py`,
`src/search/opp_belief.py` + `opp_context.build_hypotheses` (archetype belief), `qd_deck_search.py`,
`run_eval.py`, and the new eval suite (`az_eval.py`, `az_nashconv.py`, `az_target_diag.py`).

Build (new `src/rebel/`): `belief.py` (particle filter + bucket posterior), `mccfr.py`
(depth-limited external-sampling CFR over the determinized engine), `value_net.py`
(infostate-value query net, torch/numpy split + parity), `pbs.py` (PBS encoding), and the
orchestrators `scripts/rebel_selfplay.py`, `scripts/rebel_psro.py`.

---

## 8. Phased roadmap with validation gates (each gate must pass before the next)

> **Update 2026-07-11 (post-Codex review + API kill-gate).** Codex's review corrected the
> gate ordering: the tabular-exact-CFR PoC (old Phase 1) validates CFR *math* but NOT the
> dangerous assumption — whether this black-box determinized API yields *sound, per-player
> counterfactual* estimates at shared infosets. Revised order:
> - **Gate A — API contract (DONE, PASSED)**: `scripts/rebel_api_probe.py` proved immutable
>   branching, per-particle determinism, `manual_coin` chance control, and correct per-player
>   observation. The engine *can* back CFR traversal; perfect recall is obtained by keying
>   infosets on the tree **path** (public history), not the board snapshot.
> - **Gate B — estimator differential test (NEXT, the real kill gate)**: a tiny 2-world
>   asymmetric-reach game with a KNOWN exact CFR solution; implement the sampled counterfactual
>   estimator *with explicit reach/importance weights `π_{-i}π_c/q` and a fixed per-player range
>   per solved public state*; verify sampled regret increments converge to exact CFR for BOTH
>   players, with and without SIR. If the estimator is biased, particle-ReBeL is not sound here.
> - Only after Gate B do the value net / recursion phases below run, with an **immutable target
>   network + oracle-anchored labels + cross-fitting** to avoid re-introducing the AZ bootstrap
>   fixed point, and (§6) evaluating **complete `(deck, policy)` pairs** rather than coordinate
>   best responses, gated on restricted-PSRO exploitability with CIs + held-out QD elites.


**Phase 0 — pre-flight teacher ablation (cheap, no rewrite).** Confirm the diagnosis and get
a baseline: fix the `teacher_strength.py` serving bug (determinized ISMCTS throws ~12k
"invalid index" on unknown decks); run the 2×2 `{opponent: det-reset vs policy-mixture-memory}`
× `{leaf: value-head vs multi-continuation avg}` paired ablation on fixed roots; re-diagnose
value with many continuations from the *same* infostate (z̄(I,a), MSE vs constant, action-rank
corr — corr(V,z)≈0.34 alone doesn't prove uninformative). *Gate: identify which link a
better teacher actually moves.* Useful regardless of ReBeL.

**Phase 1 — tiny EXACT CFR proof-of-concept.** A small self-built exact subgame (fixed small
decks, aggressive action abstraction, tabular Linear CFR, exact joint belief, NO value net).
*Gate: NashConv(ReBeL) < NashConv(ISMCTS) on the same tiny game, and the tempo-bias
disappears.* If this fails, belief/CFR is wrong — stop before adding function approximation.

**Phase 2 — depth-limited solving with an oracle leaf.** Add subgame boundaries + an oracle
(exact deep solve) leaf value on the small game. *Gate: CFR iterations ↑ ⇒ exploitability ↓;
Bayes belief update verified; results independent of determinization order.*

**Phase 3 — infostate value network.** Train `value_net` on `(PBS, player, infostate-value)`
from Phase-2 oracle solves. *Gate: net-leaf solve ≈ oracle-leaf solve on held-out PBS.*

**Phase 4 — recursive self-play** (§5), cumulative buffer with a **freshness window** (we
already saw cumulative dilution kill learning). *Gate: play improves on `az_eval.py`
same-deck decomposition — the metric AZ never passed.*

**Phase 5 — approximate scaling** to the full game: external-sampling MCCFR, particle belief,
archetype bucketing, batched leaf eval, warm-start caching. *Gate: `az_nashconv.py`
exploitability of `(ReBeL, deck)` below `greedy_plus × best-deck`.*

**Phase 6 — outer PSRO with QD** (§6). *Gate: two-sided exploitability keeps falling as
oracles are added; ladder submission of the meta-mixture's top oracle.*

---

## 9. Risks & honest caveats

- **Approximation ≠ ReBeL guarantee.** Bucketing + particles + depth limit forfeit exact
  Nash convergence; Phase 1/2 gates exist to catch belief/CFR bugs before approximation hides
  them.
- **Engine cost.** `search_begin/step` per particle per CFR iteration is expensive; leaf-value
  batching + warm starts + MCCFR sampling are the levers. Budget an in-search move at seconds,
  not the ~10-min/game ladder cap.
- **Belief consistency.** Particles must stay consistent with public discard/counts and remain
  jointly valid (card exclusion couples the two players' hidden states); a naive independent
  product `P(x1)P(x2)` is wrong. Use SIR resampling on joint particles.
- **Timeline.** Competition deadline 2026-08-16 (~5 weeks). This is a multi-week research
  build; run it as the research track while **shipping `greedy_plus × best-QD-deck` to the
  ladder in parallel** (deck is the proven lever). Phases 0–2 are the cheap, high-information
  down-payment that decides whether to commit the rest.
