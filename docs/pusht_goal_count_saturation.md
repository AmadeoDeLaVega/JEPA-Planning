# PushT goal-count saturation

## Question

The confirmed multi-goal experiment improved monotonically from one through
eight equivalent goal photographs, but the marginal gain decreased. This study
tests whether predicted-action selection continues to improve at 16 and 32
goals and measures convergence to the full goal prototype.

## Frozen protocol

- Five independent seed banks: 20261121–20261125.
- Thirty tasks per bank and 64 shared five-control candidates per task.
- Thirty-two goal photographs with identical object pose and independently
  sampled agent positions.
- Goal counts: 1, 2, 4, 8, 16, and 32.
- At most 64 deterministic uniformly sampled subsets per count and task; all
  combinations are retained when there are at most 64.
- Frozen PushT DINO-WM checkpoint and the same physical regret/success metrics
  as the original confirmation.

Every feature source sees the same tasks, candidates, goals, and goal subsets.
Subsets are averaged within a physical task before banks/tasks are bootstrapped.
The primary scaling comparison is action-predicted regret at 16 versus 8 goals;
32 versus 16 is the next incremental comparison. A positive mean alone is not
called confirmation: the paired interval must be above zero and at least three
of five bank means must be positive. A nonsignificant increment is not treated
as statistical equivalence or proof of saturation.

Convergence diagnostics are computed against the full 32-goal prototype:

- mean-squared prototype difference;
- candidate-cost rank correlation; and
- selected-action agreement.

## Results

All five banks completed. The primary action-predicted results are:

| Goals | Physical regret ↓ | Physical rank corr. ↑ | Cost rank corr. to 32 ↑ | Selection matches 32 ↑ |
|---:|---:|---:|---:|---:|
| 1 | 6.692 [5.123, 8.551] | 0.510 | 0.843 | 0.804 |
| 2 | 4.886 [3.524, 6.550] | 0.535 | 0.891 | 0.873 |
| 4 | 4.506 [3.039, 6.234] | 0.552 | 0.940 | 0.916 |
| 8 | 4.467 [3.042, 6.169] | 0.562 | 0.973 | 0.939 |
| 16 | 4.481 [3.038, 6.255] | 0.568 | 0.990 | 0.959 |
| 32 | 4.376 [2.837, 6.174] | 0.571 | 1.000 | 1.000 |

The paired adjacent-count comparisons isolate the marginal value of additional
goal photographs:

| Larger vs. smaller set | Regret improvement ↑ | Positive banks |
|---|---:|---:|
| 2 vs. 1 | 1.805 [1.270, 2.431] | 5/5 |
| 4 vs. 2 | 0.381 [0.143, 0.677] | 5/5 |
| 8 vs. 4 | 0.039 [-0.087, 0.184] | 3/5 |
| 16 vs. 8 | -0.014 [-0.136, 0.087] | 3/5 |
| 32 vs. 16 | 0.105 [-0.069, 0.351] | 4/5 |

The frozen primary 16-versus-8 comparison did not pass. The 32-versus-16
comparison also did not establish an incremental gain. All larger counts still
beat one goal: 32 versus one improved regret by 2.316 [1.549, 3.209], with all
five bank means positive.

Actual encoded endpoints show the same diminishing-return pattern. Their mean
regret was 4.276, 3.112, 2.786, 2.717, 2.690, and 2.681 for 1 through 32 goals.
Only the 1-to-2 and 2-to-4 adjacent improvements had intervals above zero.

## Interpretation

Four equivalent goals are the practical knee of this curve. More goals make
the latent prototype and candidate ranking progressively closer to the full
32-goal reference, but that representation-level convergence no longer yields
a reliable physical-selection improvement. At four goals, the selected action
already agrees with the 32-goal action for 91.6% of sampled subsets; at eight
goals agreement is 93.9%.

This is evidence of strong diminishing returns, not a formal equivalence claim.
The new banks also have a harder one-goal baseline than the earlier study, so
absolute regret values should not be compared across studies; all scientific
comparisons here are paired within these new tasks.

The next useful experiment is therefore goal-set diversity at fixed count,
rather than increasing the count beyond 32.

## Validation and resources

- Slurm array: `8501848`; all five tasks exited zero.
- Hardware/QoS: one NVIDIA RTX A4000 per task, `vulcan-default`, nodes
  `vulcan39` and `vulcan41`.
- Per-bank model/simulator runtime: 70–105 seconds.
- Checkpoint SHA-256:
  `8ec9cb05f22812d7f12e3c216b0637f41641055c0653e503e2746edb981b550f`.
- Generated result storage: approximately 16 MB.
- Repository tests after implementation: 95 passed.
- CPU smoke output: `artifacts/pusht_goal_count_saturation_smoke_20261121/`.
- Full outputs: `results/pusht_goal_count_saturation/`.

## Commands

Run a CPU smoke test:

```bash
.venv/bin/python code/experiments/run_pusht_goal_count_saturation.py \
  --device cpu --smoke --output artifacts/pusht_goal_count_saturation_smoke
```

Submit the five-bank GPU array:

```bash
sbatch code/scripts/run_pusht_goal_count_saturation.slurm
```

Aggregate completed runs:

```bash
.venv/bin/python code/experiments/summarize_pusht_goal_count_saturation.py \
  --input results/pusht_goal_count_saturation \
  --output results/pusht_goal_count_saturation/aggregate
```
