# Documentation Index

Research reports, architecture notes, and reproduction guides for the
JEPA-Planning project. Raw logs, model checkpoints, and generated outputs live
outside Git under `results/` and `checkpoints/`.

## Unified Capstone Report

- [`unified_capstone_results.md`](unified_capstone_results.md) — End-to-end
  synthesis across all four tracks with locked held-out `test` benchmarks for
  2D PushT (`N = 24`), zero-privileged-state 2-stage visual waypoint chaining
  (`N = 8`), and 3D Meta-World Sawyer `Push-v3` (`N = 72` audit, `N = 12`
  candidate bank & `H = 6` closed loop), embedded figures/GIFs, and
  reproducibility commands.

## Setup & Baseline Reproduction

- [`reproduction.md`](reproduction.md) — Environment setup, checkpoint
  downloads and checksums, Meta-World Push baseline protocol (`run_baseline.py`),
  and platform notes (`MUJOCO_GL=glfw` on macOS, `egl` on Linux).
- [`goal_photo_sensitivity.md`](goal_photo_sensitivity.md) — Controlled
  goal-photo nuisance intervention quantifying how plain feature MSE reacts to
  repositioned effectors versus object displacements.

## Track 1 — Multi-Goal Prototype Averaging & Saturation

- [`pusht_multi_goal.md`](pusht_multi_goal.md) — Fixed-bank and closed-loop
  evaluation of averaging equivalent goal photos on PushT (`K = 4` reduces
  closed-loop physical error from `43.932` to `30.786 px` across five seeds).
- [`pusht_goal_count_saturation.md`](pusht_goal_count_saturation.md) —
  Scaling study from `K = 1` to `K = 32` equivalent goal photos on 150 PushT
  tasks, establishing `K = 4` as the practical knee.

## Track 2 — Linear Goal Heads, Spectral Isotropy & `k`-Fantope Projection

- [`linear_head_plan.md`](linear_head_plan.md) — Design and budget allocation
  plan ($B = M \times K$) for trace-normalized linear goal heads $L \in \mathbb{R}^{r \times 384}$
  across PushT and Meta-World Sawyer Push-v3 (note: for 3D Sawyer Push-v3, use
  `code/configs/metaworld_goal_groups_calibrated.yaml` for reachable mocap hand
  banks).
- [`pusht_linear_head.md`](pusht_linear_head.md) — Full PushT linear-head
  sweep across ranks `4/16/32/128`, 9 allocations, shuffled/random controls,
  fixed-bank regret, and paired closed-loop CEM evaluation.

## Track 3 — Temporal, Quasimetric & Visual Waypoint Objectives

- [`pusht_temporal_cost.md`](pusht_temporal_cost.md) — Learned elapsed-step
  temporal goal cost trained from trajectory frame gaps on PushT.
- [`pusht_quasimetric_cost.md`](pusht_quasimetric_cost.md) — Interval
  Quasimetric Embedding (IQE) minimum-step goal cost and branching diagnostics;
  complements unified calibrated objectives (`calibrated_objectives.py`) and
  multi-stage visual waypoint chaining (`waypoint_mpc.py`,
  `run_pusht_waypoint_demo.py`).

## Track 4 — Barrier-Certified Trust-MPC & Interactive Planning Studio

- [`pusht_trust_mpc.md`](pusht_trust_mpc.md) — Root-cause spectral analysis of
  rank-1 InfoNCE collapse at `B = 1024`, log-det row-isotropy and `k`-Fantope
  spectral regularization, Jensen-certified mode splitting, stay-gate drift
  suppression, and held-out paired closed-loop confirmation on PushT (extended
  to 3D Sawyer Push-v3 in `metaworld_planning.py`, `run_metaworld_trust_mpc.py`,
  and the Interactive Visual World-Model Planning Studio in `studio.py`).
