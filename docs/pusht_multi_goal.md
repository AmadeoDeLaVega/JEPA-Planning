# Equivalent-goal averaging on PushT

## Idea and scope

A single goal photo contains both the task target—the T-shaped object's pose—and
nuisance content such as the agent's position. A frozen visual encoder does not
guarantee that these factors are separated. Consequently, standard terminal
feature MSE can rank action candidates according to accidental details in one
photo.

The multi-goal baseline asks for several valid photos of the same object target
with different agent poses and scores candidate endpoint feature `x` by

```text
cost(x) = mean_k MSE(x, goal_k).
```

This is a training-free estimator, not a learned task metric. It is motivated by
the proposal's grouped equivalent observations, but it does not implement the
proposal's learned low-rank matrix. Algebraically, mean squared distance to
several goals has the same candidate ranking as distance to their mean feature;
the benefit comes from estimating a less noisy goal prototype.

## Experiments

The fixed-bank probe isolates candidate ranking. For each task it executes 64
shared five-control chunks in the simulator, predicts the same endpoints with
the released frozen PushT DINO-WM, and renders eight goal photos with a common
object pose and independently varied agent poses. It evaluates all subsets for
1, 2, and 4 goals (plus the full eight-goal set), then averages subsets within
each task. The primary metric is physical regret relative to the best candidate
in the same bank.

The closed-loop experiment is a paired reduced-budget CEM test. Targets are
selected from independent simulator rollouts before either planner runs. The
one-goal and four-goal planners restore the same initial state and receive the
same random CEM samples at every iteration and replan. Only the terminal cost
changes.

## Commands

After following `docs/reproduction.md` to create `.venv` and download the PushT
checkpoint, run small CPU integration checks:

```bash
.venv/bin/python code/experiments/run_pusht_multi_goal_probe.py \
  --device cpu --smoke --output artifacts/pusht_multi_goal_probe_smoke

.venv/bin/python code/experiments/run_pusht_multi_goal_closed_loop.py \
  --device cpu --smoke --output artifacts/pusht_multi_goal_closed_loop_smoke
```

The smoke settings only validate the pipeline and are not scientific evidence.
Full five-seed runs use Slurm:

```bash
sbatch code/scripts/run_pusht_multi_goal_probe.slurm
sbatch code/scripts/run_pusht_multi_goal_closed_loop.slurm
```

Aggregate completed run directories with:

```bash
.venv/bin/python code/experiments/summarize_pusht_multi_goal.py \
  --kind probe --input results/pusht_multi_goal_probe \
  --output results/pusht_multi_goal_probe/aggregate

.venv/bin/python code/experiments/summarize_pusht_multi_goal.py \
  --kind closed-loop --input results/pusht_multi_goal_closed_loop \
  --output results/pusht_multi_goal_closed_loop/aggregate
```

Outputs record the effective config, checkpoint path, Git revision and dirty
status, software versions, command, runtime, GPU, and Slurm identifiers. Output
directories must be new; scripts refuse to overwrite existing runs.

## Existing Local evidence

The earlier Local implementation produced the following preliminary results.
They motivate this cleaned implementation but are not copied into tracked
`results/` and should be regenerated from this repository before final reporting.

- Fixed action bank: five seeds × 30 tasks, 64 candidates, eight equivalent
  goals. Predicted-latent regret decreased from 5.300 with one goal to 3.267
  with four and 2.876 with eight. The paired four-versus-one improvement was
  2.034 with hierarchical 95% interval [1.169, 2.990], positive in all five
  seed banks.
- Paired closed-loop CEM: five seeds × six episodes. Mean final physical error
  decreased from 43.932 to 30.749. The paired improvement was 13.184 with
  hierarchical 95% interval [2.119, 26.838], positive in all five banks. The
  success-rate improvement was not statistically resolved.

These findings support multi-goal averaging as a suitable course-project
baseline. They do not establish broad task generality, and the success metric
needs larger or stronger closed-loop evaluation.
