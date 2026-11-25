# Minimum-step (quasimetric) goal cost on PushT

**Status:** trained once (`results/pusht_quasimetric_cost/seed_20261201`, commit
`f121655`) and analysed on the dev banks and branching copies in
`code/debug/temporal_cost.ipynb`. It failed the branching check, so by the plan
below it stops here. Its λ = 0.2 blend with MSE is the strongest dev result on the
branch, but nothing has been run on the evaluation seeds.

## Why

The temporal cost (`docs/pusht_temporal_cost.md`) was basically training on elapsed time
between two frames of one recorded episode. It counted steps well along a
recording but never beat MSE at planning. One probably cause of this is the labels that
were used in training:

- **Elapsed time is only an upper bound.** A recording can pause or detour, so
  the gap between two of its frames can be larger than the steps needed.
- **Labels never compare branches.** A gap exists only for two frames of the
  same recording. Planning asks which of several branching candidates is nearer
  the goal, which is exactly where temporal lost to MSE.

This cost is to target the _minimum_ number of steps needed to get to a goal state
from a current state `t`. 

## Method

`QuasimetricCost` in `code/jepa_planning/quasimetric_cost.py` follows
quasimetric RL (Wang et al., 2023). It has the same inputs and call as
`TemporalCost`, so the probe and closed-loop scripts can score with it unchanged:

```text
candidate [16, 16, 384] -> encoder --\
                                      interval quasimetric -> steps
goal      [16, 16, 384] -> encoder --/      (same encoder)
```

- **Encoder.** Layer norm, a 1x1 projection, two stride-2 convolutions and an
  MLP, as in `TemporalCost`, ending in 32 components of 8 latent coordinates.
- **Distance.** Interval Quasimetric Embedding. In each component, every
  coordinate where the candidate lies below the goal adds the interval between
  them, and the component's distance is the length of the union. Components are
  mixed as a learned blend of their maximum and mean. The distance is zero from a
  state to itself, never negative and can differ by direction. It obeys the
  triangle inequality by construction, for any weights.

`QuasimetricObjective` trains it with two rules:

1. A recorded step costs at most one. For consecutive cached frames `s, s'`,
   the mean of `relu(d(s, s') - 1)²` must stay below $\text{slack}^2$ (default $0.25^2$).
   A learned Lagrange multiplier enforces this. It rises while the rule is
   broken and falls once it holds.
2. Otherwise, distances go up. Between random pairs of frames, the loss
   rewards larger distances, which saturates past `horizon` so that
   unreachable pairs stay finite.

Together these give the shortest path. Any chain of recorded steps from `s` to
`g` bounds `d(s, g)` by its length, through the triangle inequality, and the
push raises every distance to that bound. Chains may stitch pieces of different
recordings, which elapsed-time labels cannot express.

`sample_transitions` draws recorded transitions, each equally likely, and
`sample_frame_pairs` draws random pairs. Both use the frame-cache layout of
`code/experiments/train_pusht_temporal_cost.py`.

### Why stitching should help on `pusht_noise`

The data has 185 scenes with 101 copies each, branching from nearly one start
(Analysis in the temporal doc). Many recordings pass through nearly the same
states, which is where shortest paths can switch between them. Exact repeats are
rare in continuous states, so stitching relies on the encoder mapping similar
states close together. Whether that happens is the first thing to check.

### Design consequences

- One encoder for both sides. The triangle inequality needs candidate and
  goal in the same space. `TemporalCost` used separate projections so it could
  discount the agent in the goal photo. This cost counts the steps to reach the
  photo's full state, agent included, which should add a few steps at most.
  Photo agreement on the probe measures whether that matters. (Observed: it
  ignores the photo's agent almost entirely; see Results.)
- Steps should run about 1.25, not 1. The slack lets the average step slightly
  exceed one, so all distances are inflated by about the same factor. Rankings,
  which are all the planner uses, are unaffected. (Observed: about 1.6, because
  the step rule never fully held; see Results.)
- Distances start near 0.01. The push term must grow the latent scale early
  in training. Watch this in the first validation rows.

## Training

`code/experiments/train_pusht_quasimetric_cost.py` uses the same data and frame
cache as the temporal cost (`code/configs/pusht_quasimetric_cost.yaml`; same
seed, episodes, frame stride, network width, batch size, steps and learning
rate). Each step draws 128 recorded transitions and 128 random frame pairs.

Check the pipeline first, then train. On the RTX 5070 Ti use
`~/jepa-venv-cu128/bin/python` instead of `.venv/bin/python`
(`docs/reproduction.md`):

```bash
.venv/bin/python code/experiments/train_pusht_quasimetric_cost.py \
  --device cpu --smoke --output artifacts/quasimetric_smoke

.venv/bin/python code/experiments/train_pusht_quasimetric_cost.py \
  --device cuda --cache artifacts/pusht_feature_cache \
  --output results/pusht_quasimetric_cost/seed_20261201
```

- **Feature cache.** Encoding all 18,685 episodes takes about 90 GB. With
  `--cache`, the features are kept and reused by later runs with the same
  episodes, frame stride and data; a cache built for other settings is refused.
  Without it, the cache goes in the output directory and is deleted at the end.
- **Outputs.** `quasimetric_cost.pt` (tagged `kind: quasimetric`) and
  `metadata.json`, which records the config, Git revision and dirty status, the
  cache used, and the history. Output directories must be new.
- **The final weights are saved.** Validation is logged but does not pick a
  checkpoint, since the temporal study showed step counting along recordings
  does not predict planning. The weights at every logged step are also kept in
  `checkpoints/`, so earlier states can still be analysed if the multiplier
  overshoots.
- **The multiplier has its own learning rate** (`multiplier_learning_rate`).
  Under Adam its raw value moves by about that much per step. The first run used
  the network rate, 3e-4, which proved too slow (Results). The retrain,
  `code/configs/pusht_quasimetric_cost_fast_multiplier.yaml`, uses 3e-3 and
  changes nothing else.
- **Validation** uses 4,096 fixed transitions and same-episode pairs (gaps up
  to 24) from the validation split:
  - `val_step_distance` should be about one, and `val_step_p10/p50/p90`
    show whether every step is, not just the average;
  - `val_gap_ratio`, distance over recorded gap, should be at most about
    `1 + slack`, and below one where the cost found shorter routes;
  - `val_order_accuracy` is how often the next recorded frame scores nearer
    the same goal.

`horizon: 64` has not been checked against the episode lengths in model steps.
It only sets where pushing distances apart fades, and it is tuned on the dev
banks.

## Tests

`code/tests/test_quasimetric_cost.py` runs on CPU in about a minute:

- Union-of-intervals distance: hand-worked cases and the triangle inequality
  on random latents.
- The cost: zero to itself, never negative, triangle inequality, asymmetry
  and one-goal broadcasting.
- The multiplier: it rises only while recorded steps cost more than one.
- A shortcut no single recording shows. Two toy recordings, 0-1-2-3-4-5 and
  0-6-5. Elapsed-time labels put state 5 five steps from 0, further than state
  4. After training, `d(0, 5)` matches the two-step route through 6 and is below
  `d(0, 4)`. `d(5, 0)` stays larger than the whole long route, since no recording
  goes back.
- Samplers: transitions stay within episodes and are drawn in proportion to
  episode length, random pairs stay in the cache, and both repeat under a seed.

`code/tests/test_train_pusht_quasimetric_cost.py` covers the training script:
config validation, building the cost from the config, the feature cache (built
once, reused, refused for other settings, rebuilt after an interruption), the
validation pairs, validation scores of exactly one for a cost that counts frames
perfectly, and a short training loop on a toy cache. Run both files with:

```bash
.venv/bin/python -m pytest code/tests/test_quasimetric_cost.py \
  code/tests/test_train_pusht_quasimetric_cost.py -q
```

## Evaluation plan

Same data, frozen model and protocol as the temporal and multi-goal studies, in
this order. Steps 1 and 2 are done (Results); step 2 triggered the stop rule, so
steps 3 and 4 have not been run.

1. Train on the `pusht_noise` training split and validate on the
   validation split. Check that steps stay near one and that `d(s, g)` stays at
   or below the recorded gap for same-episode pairs.
2. Branching check first (notebook step 9). Rank copies of a scene against a
   frame further along one copy. This is the test the temporal cost failed
   (rank correlation 0.10 among the 16 nearest, vs 0.32 for MSE). If this cost
   does not beat the temporal cost here, stop.
3. Dev banks 20261301–03 for any tuning (`horizon`, `slack`, model size).
4. One frozen run on the evaluation seeds: probe 20261008–12 and closed loop
   20261108–12. The pass rule is written down beforehand, the same as before: an
   interval above zero and at least 3 of 5 banks favoring the new cost over MSE.

## Results

One model, `seed_20261201`: 20,000 steps on all 18,685 training episodes, 5.6 hours
on the RTX 5070 Ti. All numbers come from `code/debug/temporal_cost.ipynb` (dev banks
20261301–03 and training scenes only) and are in `results/tables/`. Intervals are
95% hierarchical bootstraps over banks (or scenes), then tasks.

### Training: the step rule never fully held

| Step | Step length, validation (target 1 to 1.25) | Multiplier | Distance ÷ recorded gap | Next frame scored closer |
|---|---|---|---|---|
| 1,000 | 2.70 | 0.84 | 1.52 | 0.64 |
| 5,000 | 1.93 | 1.53 | 1.06 | 0.70 |
| 10,000 | 2.02 | 2.56 | 1.20 | 0.73 |
| 20,000 | 1.62 | 4.74 | 0.93 | 0.74 |

The multiplier shares the network's learning rate (3e-4) and rose too slowly, so
recorded steps still cost about 1.6 at the end, ranging from 0.5 to 3. Unequal
step costs loosen the shortest-path property this cost depends on. It does find
routes shorter than the recordings (distance ÷ gap 0.93 while a step costs 1.6).

### Ordering along recordings

Frames *k* steps nearer the same goal scored lower, 21 validation episodes:

| | MSE | Temporal | Quasimetric |
|---|---|---|---|
| 1 step apart | 0.69 | **0.80** | 0.75 |
| 2 steps apart | 0.75 | **0.86** | 0.79 |
| 4 steps apart | 0.80 | **0.91** | 0.83 |

The temporal cost is trained on exactly this and wins.

### Branching check: fails the stop rule

216 tasks over 8 training scenes: candidates are 100 copies of a scene at time
*t*, the goal is one copy's frame at *t + g*, graded by T error.

| | MSE | Temporal | Temporal + ordering loss | Quasimetric |
|---|---|---|---|---|
| Regret | **2.21** | 4.77 | 4.59 | 5.32 |
| Rank correlation, all | **0.46** | 0.41 | 0.39 | 0.36 |
| Rank correlation, 16 nearest | **0.32** | 0.10 | 0.15 | 0.10 |

Against the temporal cost on the 16 nearest: +0.003 [−0.081, +0.084], 4 of 8
scenes. It does not beat the temporal cost, so the plan's stop rule applies.

### Probe on the dev banks

| | MSE | Temporal | Quasimetric | Quasimetric vs temporal |
|---|---|---|---|---|
| Regret | **7.30** | 10.41 | 10.54 | −0.13 [−3.99, +3.12], 1 of 3 |
| Success | **0.81** | 0.64 | 0.74 | +0.10 [−0.02, +0.21], 3 of 3 |
| Rank correlation, 16 nearest | **0.68** | 0.49 | 0.49 | +0.00 [−0.13, +0.12], 2 of 3 |
| Photo agreement | 0.69 | 0.57 | **0.77** | +0.19 [+0.10, +0.28], 3 of 3 |
| `agent_weight` | 0.35 | 0.20 | **−0.01** | |

- **It ignores the agent in the goal photo.** The fit of cost on T error and
  agent distance gives `agent_weight` ≈ 0 in all three banks, the opposite of what
  the shared encoder suggested. That is why photo agreement beats MSE's: the
  photos differ only in the agent. It still follows the T less than MSE
  (`t_weight` 0.56 vs 0.65).
- **It cannot use an agent-matched photo.** With the goal photo's agent placed at
  the best candidate's agent, MSE's regret drops from 7.3 to 1.5 and the temporal
  cost's to 5.1; the quasimetric cost stays at 10.5.
- **Predicted features are not the main problem.** Scoring encoded true frames
  instead lowers its regret to 8.8, still far behind MSE's 3.9.

### Blend with MSE

$(1 - \lambda)\, z(\text{MSE}) + \lambda\, z(\text{quasimetric})$, standardized per goal photo:

| λ | 0 (MSE) | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 1 (quasimetric) |
|---|---|---|---|---|---|---|---|
| Regret | 7.30 | 6.22 | **5.33** | 5.75 | 6.60 | 8.03 | 10.54 |
| Success | 0.81 | 0.82 | **0.84** | 0.82 | 0.82 | 0.81 | 0.74 |
| Photo agreement | 0.69 | 0.71 | 0.71 | 0.74 | 0.77 | **0.78** | 0.77 |

At λ = 0.2, regret improves on MSE by 1.96 [0.40, 4.11], better in 3 of 3
banks, with success also better in 3 of 3. It is the first dev result on this
branch with an interval above zero. But λ is the best of five values on three
banks, and the temporal blend (λ = 0.5: +1.68 [−0.25, +4.41] on these banks)
failed on the evaluation seeds.

## Conclusion

Targeting the fewest steps instead of elapsed time did not fix branch
comparison in this run: on real branching trajectories the quasimetric cost ranks
the nearest candidates no better than the temporal cost, and both trail MSE. It
did change one thing. It ignores the agent in the goal photo, which raises photo
agreement and success over the temporal cost, but loses the agent-matched case.

Two things remain open:

1. **The step rule.** Recorded steps cost about 1.6, not 1, so the shortest-path
   property was never really in place, and the branching check is not yet a
   fair test of the idea. A retrain with a faster multiplier (its own learning
   rate, or a higher start) can reuse the feature cache.
2. **The λ = 0.2 blend.** It would need a frozen, one-time run on the evaluation
   seeds, with the pass rule written down first.

<!-- ## Figures

`code/scripts/plot_results.py` draws these from `results/tables/`.

**Training curves.** The step length never reaches the target band.

![Quasimetric training curves](../results/figures/pusht_quasimetric_cost/training.png)

**Every method on the dev banks and branching copies.**

![Dev-bank comparison](../results/figures/overview/method_comparison_dev.png)

**Ranking candidates** on the probe and on branching copies.

![Rank correlation, all and nearest candidates](../results/figures/pusht_temporal_cost/ranking.png)

**The blend sweep**, for all three learned costs.

![Dev-bank metrics against blend weight](../results/figures/pusht_temporal_cost/blend_sweep.png) -->
