# Learned temporal goal cost on PushT

## Idea and scope

Feature MSE asks how similar an endpoint looks to a goal photo. Since goal photos
show the agent too, it is possible for MSE to prefer a worse action just because 
the agent resembles the photo. Multi-goal averaging and the learned linear head 
address this problem using information about which photos show the same target.

This cost asks a different question: how many model steps from the goal is the agent?
Frame gaps from recorded trajectories supply labels, so this function does not use
equivalence groups or object coordinates. This experiment compares learning task
relevance from transitions to learning from task equivalence:

> Can a temporal cost become insensitive to the agent shown in the goal photo
> while remaining sensitive to useful agent positioning in the candidate state?

This is not forced in training. A cost trained to reach a future image has a reason
to care about the agent in the image because moving the agent take time. This experiment
can show what information transitions can and cannot supply.

## Method

`TemporalCost` in `code/jepa_planning/temporal_cost.py` predicts model steps from
frozen candidate and goal patch features. Separate projections act as inpute to a shared convolutional trunk:

```text
candidate [16, 16, 384] -> 1x1 projection --\
                                             concat -> conv stride 2 x 2 -> MLP -> steps
goal      [16, 16, 384] -> 1x1 projection --/
```

Design choices:

- **Asymmetric inputs.** A shared embedding distance `||F(x) - F(g)||` treats both
  sides identically, so it could choose to not ignore the goal agent while valuing the
  candidate agent.
- **Patch layout is kept.** Features are compared per location before pooling, so
  small object displacements are not averaged away.
- **Labels are model steps.** Frames are kept every five environment steps.
- **Far pairs.** A quarter of training pairs take the goal from another episode
  and receive `max_gap`. Same-episode gaps are capped at `max_gap`.

Two additions were tested after the first results (see Results):

- **One-step ordering loss.** With `ordering_weight > 0`, training adds a hinge
  loss on frames one step apart that share a goal. The frame nearer the goal must
  score at least one step lower. `ordering_weight: 0` trains on regression only.
- **Blend with MSE.** `blend_costs` scores candidates with
  `(1 - λ) z(MSE) + λ z(temporal)`. Each cost is standardized over the
  candidates for one goal photo, because their units are unrelated. λ = 0 picks
  what MSE picks, and λ = 1 what the temporal cost picks.

Labels measure time elapsed in recorded trajectories. A segment can contain pauses and detours, so this is a learned temporal cost rather than an estimate of optimal effort.

Only `TemporalCost` is trained, the world model stays frozen.

## Experiment

`code/experiments/run_pusht_temporal_probe.py` reuses the bank generation and
config of the multi-goal probe, so a seed gives identical tasks, candidates, and
goal photos. MSE and the temporal cost each score candidates against one goal
photo at a time and reports the following:

| Metric | Question |
|---|---|
| `regret` | Is the selected candidate physically good? |
| `rank_correlation` | Does the cost order candidates by physical error? |
| `photo_agreement` | Do photos of the same target select the same candidate? |
| `target_response` | Does changing the object target change the selection? |

Raw cost changes are not comparable between MSE units and predicted steps. We measure
sensitivity through selected actions. Agreement only counts alongside
target response because a constant cost agrees perfectly and responds to nothing.

## Commands

Follow `docs/reproduction.md` to create the environment, download the PushT
checkpoint, and download `pusht_noise`. On a GPU the pinned build does not support,
use the matching environment described there. Check the pipeline first:

```bash
.venv/bin/python code/experiments/train_pusht_temporal_cost.py \
  --device cpu --smoke --output artifacts/temporal_smoke

.venv/bin/python code/experiments/run_pusht_temporal_probe.py \
  --device cpu --smoke --cost artifacts/temporal_smoke/temporal_cost.pt \
  --output artifacts/temporal_probe_smoke
```

The smoke settings only validate the pipeline and are not scientific evidence.
Configs and settings are chosen on dev banks 20261301–03. The evaluation seeds
below are run once, for a frozen setting. Train, then evaluate each seed of the
multi-goal probe:

```bash
.venv/bin/python code/experiments/train_pusht_temporal_cost.py \
  --device cuda --output results/pusht_temporal_cost/seed_20261201

for seed in 20261008 20261009 20261010 20261011 20261012; do
  .venv/bin/python code/experiments/run_pusht_temporal_probe.py --device cuda \
    --cost results/pusht_temporal_cost/seed_20261201/temporal_cost.pt \
    --seed "$seed" --output "results/pusht_temporal_probe/seed_$seed"
done
```

Closed loop uses the multi-goal closed-loop tasks and CEM random numbers, and
`summarize_pusht_temporal_closed_loop.py` pairs the result with those runs:

```bash
for seed in 20261108 20261109 20261110 20261111 20261112; do
  .venv/bin/python code/experiments/run_pusht_temporal_closed_loop.py --device cuda \
    --cost results/pusht_temporal_cost/seed_20261201/temporal_cost.pt \
    --blend-weight 0.5 --seed "$seed" --output "results/pusht_temporal_closed_loop/seed_$seed"
done

.venv/bin/python code/experiments/summarize_pusht_temporal_closed_loop.py \
  --input results/pusht_temporal_closed_loop \
  --reference results/pusht_multi_goal_closed_loop \
  --output results/pusht_temporal_closed_loop/aggregate
```

The summary refuses to run unless the `mse` rows reproduce `l2_g1` exactly.

The default config now includes the ordering loss (`ordering_weight: 1.0`). Set it
to 0 for a regression-only cost. Add `--blend-weight 0.5` to the probe to also
score the blend with λ = 0.5. Its rows are labelled `blend`.

Training writes `temporal_cost.pt` and `metadata.json`, which records the config,
Git revision and dirty status at the start, the loss history, and `best_step`.
The saved weights are the checkpoint with the best score on a fixed sample of
4,096 validation pairs. Each probe run writes `task_rows.csv`
and `metadata.json`. Output directories must be new, and generated files are
ignored by Git. Training on all 18,685 episodes needs about 90 GB of disk for its temporary
feature cache, which is deleted when training finishes.

Run the unit tests with:

```bash
.venv/bin/python -m pytest code/tests/test_temporal_cost.py \
  code/tests/test_train_pusht_temporal_cost.py \
  code/tests/test_run_pusht_temporal_probe.py -q
```

## Validation of the setup

- **Baseline reproduction.** The multi-goal probe and closed-loop runs were
  repeated on an RTX 5070 Ti with the CUDA 12.8 environment. One-photo regret was
  5.300 (reported 5.309), four-photo regret 3.267 (3.270), and closed-loop final
  error 43.932 to 30.749 (43.932 to 30.786).
- **Training images match evaluation images.** Rendering dataset states with the
  simulator settings used by the probes differs from the dataset video by a mean
  of 0.30 out of 255 per pixel.
- **The T moves in the data.** It moves in about 35% of training frames, and 0.4%
  of episodes leave it essentially unmoved.
- **Goal-photo coverage.** Comparing agent-to-T distances, with T positions
  approximated by the start-state range, 97.7% of goal photos fall within the 99th
  percentile of the data. About half lie beyond the 90th percentile, so goal photos
  sit in the upper tail of what training saw.

## Results

We use a fixed-bank probe, five banks (seeds 20261008–12), with 30 tasks per bank. Differences are paired by task and signed so that positive favors the temporal cost.

| Metric | MSE | Temporal, 2,000 episodes | Temporal, 18,685 episodes |
|---|---|---|---|
| Regret (lower is better) | 5.300 | 10.362 | 13.462 |
| Combined error (lower is better) | 15.534 | 20.596 | 23.696 |
| Success | 0.801 | 0.670 | 0.570 |
| Rank correlation | 0.520 | 0.453 | 0.507 |
| Photo agreement | 0.746 | 0.605 | 0.549 |
| Target response | 0.916 | 0.925 | 0.915 |

| Paired difference, 18,685 episodes | Mean | 95% interval | Banks favoring temporal |
|---|---|---|---|
| Regret | −8.162 | [−10.757, −5.646] | 0 of 5 |
| Success | −0.231 | [−0.318, −0.144] | 0 of 5 |
| Rank correlation | −0.013 | [−0.087, 0.061] | 2 of 5 |
| Photo agreement | −0.196 | [−0.267, −0.127] | 0 of 5 |
| Target response | −0.001 | [−0.052, 0.052] | 3 of 5 |

The temporal cost function performs worse than MSE on regret, success, and photo
agreement in every tested seed. This implementation was supposed to make the model
care less about where the agent is positioned in the goal image, but the lower photo
agreement shows that this is not the case. However, it still corresponds to the target as well as 
the MSE cost did, so what we are seeing is not a total collapse. It is also less stable
than MSE, which is the opposite of what we intended.

Using all available episodes to train on as opposed to a smaller subset of 2,000 episodes
(initially used to test that our cost function actually learns by overfitting) brought
rank correlation to level with MSE, but degraded the performance on the selected action.
The 2,000 episodes cover only 20 of the 185 scenes in the data. The analysis below
explains why better overall ranking does not give better choices.

Validation loss stayed near 3 while training loss fell to about 1.4, indicating that our
function is still overfitting. Validation uses one 128-pair batch from 21 episodes
and is noisy. Each configuration was trained once, so the difference between the
two temporal columns includes training variance. Closed loop is reported
below, under Closed loop.

The 18,685-episode cost is `results/pusht_temporal_cost/seed_20261201`. Its
recorded revision was captured at the end of training. The 2,000-episode
artifacts were deleted after its numbers were recorded, so that column cannot be
recomputed.

## Analysis

`code/debug/temporal_cost.ipynb` investigates the gap step by step on dev banks
20261301–03, never the evaluation seeds. Each step states what it measures and
what it found. In short:

| Question | Finding |
|---|---|
| Does it count steps on unseen episodes? | Moderately worse than on training episodes. Frames one step apart are ordered correctly 80% of the time on validation episodes and 84% on training episodes |
| How varied is the data? | 185 scenes with 101 copies each. Copy 0 is the reference, and later copies drift further from it (about 0.3 px for copies 1–10, 58 px for copy 100). Validation and probe starts are as close to a training scene as training scenes are to each other |
| Worse everywhere or near the goal? | Near the goal. Among the 16 best candidates, rank correlation is 0.49 vs 0.68 for MSE |
| Does it follow the agent? | Less than MSE, but it also follows the T less. Not the cause |
| Predicted features or the goal photo? | Neither is specific to the temporal cost |
| Does it compare branching trajectories? | **No.** Copies of a scene branch from one start. Ranking them against a frame further along one copy, the temporal cost loses to MSE without the world model, without synthetic goal photos and on training scenes |

The last row is the main finding. A step-count label exists only for two frames
on the same trajectory, so training never compares different trajectories.
Planning asks only that: which of several branching candidates is nearer the
goal.

## Follow-up runs

**One-step ordering loss.** This targets the near-goal weakness. It has the same
data, network and seed, with `ordering_weight: 1.0` and the best checkpoint by
validation score (step 11,000). The model is
`results/pusht_temporal_cost/ordering_seed_20261201`. It learned the skill: on
validation episodes, one-step ordering accuracy rose from 0.80 to 0.85. On the dev
banks it planned worse than the regression-only cost:

| Dev banks | MSE | Temporal | Temporal with ordering loss |
|---|---|---|---|
| Regret | 7.295 | 10.408 | 12.695 |
| Success | 0.806 | 0.640 | 0.589 |
| Rank correlation, 16 best candidates | 0.682 | 0.490 | 0.401 |

Ordering against the regression-only cost changes regret by −2.287
[−5.324, 0.315], worse in 3 of 3 banks. Improving the skill along one
trajectory did not transfer, which fits the branching finding.

**Blend with MSE.** On the dev banks, the blend with the regression-only cost had
its lowest regret at λ = 0.5 (5.614 vs 7.295 for MSE, better in 3 of 3 banks).
The setting was frozen and run once on the evaluation seeds. The pass rule was
written down first, in `results/pusht_temporal_probe_blend/preregistration.txt`.
It is the multi-goal study's rule: a regret interval above zero and at least 3 of
5 banks favoring the blend.

| Evaluation seeds 20261008–12 | MSE | Blend, λ = 0.5 | Paired difference, favoring blend | Banks favoring blend |
|---|---|---|---|---|
| Regret | 5.300 | 7.077 | −1.776 [−3.330, −0.246] | 1 of 5 |
| Success | 0.801 | 0.749 | −0.052 [−0.107, −0.003] | 1 of 5 |
| Rank correlation | 0.520 | 0.592 | +0.072 [+0.036, +0.108] | 5 of 5 |
| Photo agreement | 0.746 | 0.705 | −0.041 [−0.097, 0.017] | 1 of 5 |

The blend fails. The dev-bank gain did not hold: λ was the best of ten settings
on three banks, and banks vary a lot (MSE regret is 7.30 on the dev banks and 5.30
here). The blend does rank all candidates better in every bank, but it still picks
a worse best candidate. This is the same pattern as the 18,685-episode run.

## Closed loop

The fixed-bank probe scores one step, so it could penalize preparatory moves that
pay off later. Closed-loop CEM is the decisive test in the multi-goal study, so
the frozen costs were run on its five seeds, 20261108–12. Each seed has 6 episodes
and 4 replans. The episodes, targets and CEM random numbers are identical to the
multi-goal runs, and every method plans against the first goal photo. Our `mse`
reproduces `l2_g1` exactly, so all four methods are paired episode by episode.
The settings and pass rule were written down first, in
`results/pusht_temporal_closed_loop/preregistration.txt`.

| Method | Final error (lower is better) | Ever succeeded |
|---|---|---|
| `mse`, one photo (`l2_g1`) | 43.93 | 0.40 |
| `l2_g4`, multi-goal, four photos | **30.75** | **0.47** |
| Temporal | 60.62 | 0.23 |
| Blend, λ = 0.5 | 37.42 | 0.43 |

| Paired final-error improvement | Mean | 95% interval | Banks positive | Passes |
|---|---|---|---|---|
| Temporal over `mse` | −16.68 | [−42.36, 11.26] | 1 of 5 | No |
| Blend over `mse` | +6.51 | [−8.72, 23.45] | 3 of 5 | No |
| Temporal over `l2_g4` | −29.87 | [−49.83, −9.21] | 0 of 5 | No |
| Blend over `l2_g4` | −6.67 | [−20.68, 5.28] | 2 of 5 | No |
| `l2_g4` over `mse` (reference) | +13.18 | [+2.46, +26.94] | 5 of 5 | Yes |

The temporal cost alone plans worse than MSE in 4 of 5 banks. The blend leans
better than MSE (3 of 5 banks) but the interval includes zero, so it does not pass.
It also trails multi-goal averaging. With only 30 episodes the intervals are wide,
so the blend's closed-loop result is inconclusive, not a demonstrated gain. Its
probe result on the evaluation seeds was a clear loss. Multi-goal averaging
remains the only method here that beats one-photo MSE.

## Conclusion

A cost trained on elapsed time within recorded trajectories learns to order
frames along a path, and gets better at it when trained for it. Planning instead
compares candidates on different trajectories against a goal on none of them. No
version tested beat MSE:

- trained on 2,000 or 18,685 episodes,
- with the ordering loss,
- blended with MSE, in the probe or in closed loop.

The analysis points to the training labels as the cause: they never compare
branches. In `pusht_noise`, the only labels that could compare branches are
object poses. That would train on what the probe grades and drop the idea of a
temporal cost.

In closed loop the temporal cost alone was clearly worse. The blend was
inconclusive and also behind multi-goal averaging.

## Limitations

- Training uses `pusht_noise`, which the frozen model has probably also seen.
- The cost is trained on encoded frames but applied to predicted features.
- CEM searches for low-cost actions and may exploit errors in a learned cost.
- Goal photos place the agent without an overlap check, as in the multi-goal
  probes, so a small share may disturb the T.
- Each model was trained once, so differences between models include training
  variance.
- Configs were compared on three dev banks, and Analysis step 9 uses eight
  scenes. Intervals over so few banks are coarse.
- Closed loop has 6 episodes per seed, so its intervals are wide.
- Results cover PushT only. Meta-World Push has not been tested.
