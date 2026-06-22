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
</content>
