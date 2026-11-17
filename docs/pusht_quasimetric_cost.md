# Minimum-step (quasimetric) goal cost on PushT

**Status:** the head, the training objective, the training script and unit tests
have been built. Training has not been run yet.

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
  Photo agreement on the probe measures whether that matters.
- Steps run about 1.25, not 1. The slack lets the average step slightly
  exceed one, so all distances are inflated by about the same factor. Rankings,
  which are all the planner uses, are unaffected.
- Distances start near 0.01. The push term must grow the latent scale early
  in training. Watch this in the first validation rows.

## Training

`code/experiments/train_pusht_quasimetric_cost.py` uses the same data and frame
cache as the temporal cost (`code/configs/pusht_quasimetric_cost.yaml`; same
seed, episodes, frame stride, network width, batch size, steps and learning
rate). Each step draws 128 recorded transitions and 128 random frame pairs.

Check the pipeline first, then train:

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
  does not predict planning.
- **Validation** uses 4,096 fixed transitions and same-episode pairs (gaps up
  to 24) from the validation split:
  - `val_step_distance` should be about one;
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

## Planned evaluation

Same data, frozen model and protocol as the temporal and multi-goal studies, in
this order:

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
