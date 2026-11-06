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
Train, then evaluate each seed of the multi-goal probe:

```bash
.venv/bin/python code/experiments/train_pusht_temporal_cost.py \
  --device cuda --output results/pusht_temporal_cost/seed_20261201

for seed in 20261008 20261009 20261010 20261011 20261012; do
  .venv/bin/python code/experiments/run_pusht_temporal_probe.py --device cuda \
    --cost results/pusht_temporal_cost/seed_20261201/temporal_cost.pt \
    --seed "$seed" --output "results/pusht_temporal_probe/seed_$seed"
done
```

Training writes `temporal_cost.pt` and `metadata.json`, which records the config,
Git revision, dirty status, and loss history. Each probe run writes `task_rows.csv`
and `metadata.json`. Output directories must be new, and generated files are
ignored by Git. Training needs about 10 GB of disk for its temporary feature cache.

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
care less about where the agent is positioned in the goal image, but the higher regret
shows that this is not the case. However, it still corresponds to the target as well as 
the MSE cost did, so what we are seeing is not a total collapse. It is also less stable
than MSE, which is the opposite of what we intended.

Using all available episodes to train on as opposed to a smaller subset of ____ episodes 
(initially used to test that our cost actually function learns by overfitting) brought
rank correlation to level with MSE, but degraded the performance on the selected action.
*NEED TO TEST WHY*.

Validation loss stayed near 3 while training loss fell to about 1.4, indicating that our
function is still overfitting. Validation uses one 128-pair batch from 21 episodes
and is noisy. Each configuration was trained once, so the difference between the
two temporal columns includes training variance. The closed-loop experiment has
not been run.

The 18,685-episode cost is `results/pusht_temporal_cost/seed_20261201`. Its
recorded revision was captured at the end of training. The 2,000-episode
artifacts were deleted after its numbers were recorded, so that column cannot be
recomputed.

## Limitations

- Training uses `pusht_noise`, which the frozen model has probably also seen.
- The cost is trained on encoded frames but applied to predicted features.
- CEM searches for low-cost actions and may exploit errors in a learned cost.
- Goal photos place the agent without an overlap check, as in the multi-goal
  probes, so a small share may disturb the T.
- Results cover PushT only. Meta-World Push has not been tested.
