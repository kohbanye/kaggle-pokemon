# Paper-faithful OSFP rewrite (branch `feat/osfp-paper-faithful`)

Goal: re-implement the training core to match the ByteDance Hearthstone OSFP paper
(arXiv:2303.05197) as closely as the Pokémon engine allows, **filling the gaps** the
current Phase-5d stack leaves open. We reuse the already-paper-aligned infra and
replace the simplified RL with the paper's actual algorithm. Hearthstone-specific
pieces (hero-separate models, the cheat/c5 variant) are dropped.

> Scope decision (user): "fill the gaps to paper-compliant" — keep numpy serving +
> torch↔numpy parity, the obs encoder, `OpponentPool`, and the autoregressive CB
> legal-mask machinery; rewrite the learner + data + loop. Drop the Pokémon-specific
> deck-collapse regularisers (deck entropy + BC-KL anchor).

## What the current stack does vs the paper

| Axis | Paper (2303.05197) | Current Phase-5d | This rewrite |
|---|---|---|---|
| RL objective | **V-Trace + PPO** (IMPALA actor-learner) | plain REINFORCE + value baseline | **V-Trace + PPO** (`src/net/vtrace.py`, `LitVtracePPO`) |
| Hidden info | **LSTM over obs history** (h=256) | memoryless MLP (play side) | recurrent play head + numpy parity |
| Episode | deckbuild+battle = one MDP | two separate REINFORCE arms | one trajectory: CB picks then BT moves, shared value + LSTM |
| Off-policy data | FIFO queue, producer/consumer ≈1 | synchronous collect→train | FIFO trajectory queue, V-Trace corrects staleness |
| OSFP | Alg.1 (recency mix, last-iterate) | `OpponentPool` (kept) | `OpponentPool` (reused) |
| γ / reward | 1.0, terminal ±1 | 1.0, terminal ±1 (kept) | 1.0, terminal ±1 |
| deck collapse | n/a (30 cards, no degenerate) | entropy + KL anchor hacks | dropped — value baseline handles it |

## Build stages (each verifiable before the next)

1. **V-Trace + PPO math** — `src/net/vtrace.py`, pure numpy, unit-tested. ✅
2. **Recurrent net** — play-side LSTM in torch + numpy serving + parity test; stateful `NetAgent`.
3. **Trajectory data** — collector records behaviour log-probs + step order + rewards; one episode = CB picks ⊕ BT moves; padded-sequence dataset.
4. **Learner** — `LitVtracePPO`: recurrent forward → V-Trace targets → PPO clipped surrogate + entropy + value MSE, one update over both heads + shared trunk/LSTM/embedding.
5. **Actor-learner loop** — `scripts/train_paper_osfp.py`: FIFO queue of trajectories, producer/consumer balance, OSFP opponent sampling via `OpponentPool`.

## V-Trace notes (stage 1)

Standard IMPALA V-Trace with the paper's lower-clip on ρ (the "ρ-clip" improved
technique that stops the trace vanishing):

- ratio `= π(a|s)/μ(a|s)`, `ρ_t = clip(ratio, ρ_min, ρ̄)`, `c_t = min(ratio, c̄)`
- `δ_t = ρ_t (r_t + γ V_{t+1} − V_t)`
- `v_t = V_t + δ_t + γ c_t (v_{t+1} − V_{t+1})` (backward; `v_{T} = bootstrap`)
- PG advantage `Â_t = ρ_t (r_t + γ v_{t+1} − V_t)` (detached)

PPO surrogate uses the V-Trace advantage with ratio clipping `±ε`. On-policy
(ratio≡1) V-Trace collapses to the Monte-Carlo return — the unit test pins this.

## Episode / learner design (stages 3–4)

One **episode** = one player's trajectory in one game: the 60 deck picks (CB head)
followed by the battle decisions (BT head), terminal reward ±1, γ=1.

- **Battle arm** — full V-Trace over the battle steps: the recurrent play LSTM
  gives a per-step value `V(h_t)`; V-Trace turns (behaviour log-probs, target
  log-probs, values, terminal reward) into value targets `vs` and PG advantages.
  PPO clipped surrogate + value MSE + entropy. This is the faithful core.
- **Deck arm** — the deck LSTM has no per-pick value head, so the deck picks share
  one baseline: `advantage = return − V(battle-start)` (the play value at the first
  battle decision, `values[:,0]`). This is REINFORCE-with-a-learned-baseline (the
  shared value), PPO-clipped against the pick behaviour log-probs, plus a small
  entropy bonus. It **replaces** the Phase-5d batch-mean baseline + the deck KL/
  entropy anti-collapse hacks: a real value baseline is what those hacks approximated.

A standard entropy bonus is kept on both arms — that is the paper's SBR entropy
regularisation, not a Pokémon-specific hack. What we drop are the deck-collapse KL
anchor and the hand-tuned deck-entropy coefficient.

**Behaviour log-probs are recorded at collection time** (the actor's `μ(a|s)` for
each sampled action and each deck pick), so the learner can form the importance
ratio. Battle trajectories are kept whole (no per-decision subsampling — that would
break the recurrence); cost is controlled by sampling whole games instead.
</content>
