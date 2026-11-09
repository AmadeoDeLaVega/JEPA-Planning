# Plan: learned linear goal head L (proposal §3–5)

Status: in progress (Meta-World on `feature/linear-goal-head`; PushT on `feature/pusht-linear-goal-head`, results below).
Done: step 1 (`goal_groups.py`, `make_goal_groups.py`, `configs/goal_groups.yaml`),
the head, loss, sampler and control matrices (`linear_head.py`),
`goal_costs.linear_goal_distance`, and CPU tests. The group generator has not yet
run in the simulator. Defaults chosen for open decisions 1-2: expert-trajectory
targets and an xy-only hand bank.

## PushT first (branch `feature/pusht-linear-goal-head`)

The pipeline is run on PushT before Meta-World Push, the proposal's environment.
All images are rendered with the jepa-wms PushT env (`third_party/jepa-wms`,
commit `13cf1d9`, `PushTEnv(render_size=224, with_target=True)`), the same env
the PushT checkpoint and the multi-goal experiments use.

- Targets: T poses built like the closed-loop multi-goal targets (random start,
  128 random chunks, farthest endpoint); train/val/test use separate seeded
  streams.
- Agents: a fixed bank of 64 positions uniform over [45, 467]² px, split
  32/16/16 train/val/test, independent of the target. A photo is rejected if
  the env's reset step moved the T or the agent (overlap).
- Val/test groups also get photos with the T shifted 5/10/20 px or rotated
  5/10/20° for the sensitivity check.
- Code: `pusht_goal_groups.py`, `make_pusht_goal_groups.py` (render + fp16
  feature cache), `train_linear_head.py` (shared τ/lr grid on val, 9
  allocations × 3 resamples, shuffled-label and projection controls).
- Compression sweep (added beyond the proposal): the head's output rank r is
  varied over {4, 8, 16, 32, 64, 128, 384}, i.e. compression 384/r from 96x to
  1x (learned full-rank metric). Each rank gets its own τ/lr choice on val, the
  full 9 allocations × 3 resamples, the shuffled-label control, and random /
  coordinate-subset controls of the same rank. Outputs carry `rank` and
  `compression` columns; `--ranks` splits ranks across Slurm jobs. About 180
  fits per rank (15 grid × 9 + 45); time one rank before launching all.
- Planner integration: `run_pusht_head_bank.py` (steps 6.2/6.3),
  `run_pusht_head_closed_loop.py` (step 6.4), `summarize_pusht_heads.py` (6.5).

### PushT results so far (M2 Mac, MPS; generated files under `results/`, git-ignored)

Data: 4096 train / 512 val / 800 test photos (512 / 64 / 100 targets × 8 agents).
Heads: ranks 4, 16, 32, 128; 9 allocations × 3 resamples each, plus shuffled-label
heads and random-projection / coordinate-subset controls. τ/lr chosen on val.

**6.1 Photo level (test, no refitting).** Every learned head has AUROC 1.000
(plain MSE 0.995). Sensitivity check passes: moving the T by 20 px raises a
rank-32 head's distance to 2.4× the usual spread between photos of one target
(plain MSE 0.33×; random/coordinate controls about the same as MSE). Rotating it
20° gives 3.9× (MSE 0.65×).

**6.2/6.3 Fixed bank, test (100 tasks, 64 candidates, 2 goal photos).**
Week-4 gate passes. Regret in px against the best candidate in the bank:

| | actual endpoints | predicted endpoints |
|---|---:|---:|
| plain MSE | 10.6 | 10.8 |
| learned heads (all ranks/allocations) | 5.7 | 7.0 |
| random projection | 13.4 | 13.3 |
| coordinate subset | 12.2 | 11.8 |
| shuffled-label heads | 35.7 | 36.4 |

Rank 32, predicted endpoints: paired difference vs MSE −2.6 to −4.3 px per
allocation; the 95% task-bootstrap CI excludes 0 for 8 of 9 (B1024 M256 K4:
[−5.7, +0.4]). Swapping goal photo 0 for photo 1 changes the selected candidate
in 26% of tasks with plain MSE and 4–12% with heads.

**6.4 Closed loop, test (100 tasks × 2 goal photos, identical CEM noise/budget).**
Done so far: plain MSE, rank-32 r0 of all 9 allocations, rank-32 random s0.
Mean final T position error in px (paired vs MSE, 95% CI):

| rank 32, r0 | K=2 | K=4 | K=8 |
|---|---:|---:|---:|
| B=64 | 19.5 (−13.7 [−18.4, −9.2]) | 19.5 (−13.7 [−18.3, −9.4]) | 20.7 (−12.5 [−17.1, −8.1]) |
| B=256 | 36.9 (+3.7 [−2.1, 9.6]) | 19.5 (−13.7 [−18.4, −9.2]) | 20.6 (−12.6 [−17.2, −8.1]) |
| B=1024 | 35.4 (+2.2 [−3.5, 8.1]) | 31.9 (−1.3 [−7.2, 4.5]) | 31.5 |

Plain MSE 33.2 (success ≤20 px 0.47); good heads ≈ 0.63. Random rank-32
projection 40.2, so the gain is not from compression alone. Every B=1024 head
is near MSE; one untested possibility is that all allocations get the same 2000
steps, so B=1024 sees each image fewer times. Only one resample per allocation
so far; r1/r2 are queued.

Remaining closed-loop queue (`code/configs/pusht_head_closed_loop_order.txt`):
the rank-32 shuffled control, r1 and r2 of the 9 allocations, the remaining
rank-32 controls, then the val-best allocation at ranks 4/16/128. About 55 min
per head on the Mac; it runs overnight only:
`caffeinate -dimsu bash code/scripts/run_pusht_head_closed_loop_night.sh`
(resumable; skips finished heads).

**MPS bug (fixed in `model.py`).** Upstream `encode` copies normalized proprio
to the device with `non_blocking=True`; on MPS the temporary can be freed before
the copy reads it, so the predictor got garbage proprio (up to 1e32) and
predicted endpoints were wrong. Encoder features were unaffected, so head
training and 6.1 are valid. The first bank run's predicted results were invalid
and are kept in `results/pusht_head_bank_INVALID_mps_proprio_bug/`.
`model.py` also loads DINOv2 from the local torch-hub cache, so a GitHub outage
no longer blocks model loading.

## What we are building

A frozen DINO-WM (Meta-World checkpoint) plus one trainable matrix
`L ∈ R^{32×384}`, shared across all N = 256 patches (12,288 weights,
`||L||_F = 1`). L is used only in the goal cost; rollouts keep the original
features.

- Planning cost (Eq. 3): `J_L(A, g) = mean_i ||L (ẑ_i(A) − g_i)||²`
- Training distance (Eq. 4): `D_L(g, h) = mean_i ||L (g_i − h_i)||²`
- Loss (Eq. 4): InfoNCE with one positive (same object goal, other hand) and
  7 negatives (other goal groups), temperature τ.
- After each gradient step: `L ← L / ||L||_F`; skip the step if the norm is 0.

Research question: at a fixed budget `B = M·K`, B ∈ {64, 256, 1024},
K ∈ {2, 4, 8}, is it better to have more targets (M) or more hand configurations
per target (K), measured by planning performance on held-out targets?

## What already exists and gets reused

| Need | Existing code |
|---|---|
| Seeded Push tasks, expert, snapshots, puck/hand readout | `code/jepa_planning/metaworld_push.py` |
| Same-puck/different-hand goal photos | `mp.move_hand_keep_puck`, `baseline.prepare_task` |
| Frozen model loading, `encode`, `unroll` | `code/jepa_planning/model.py` |
| Plain feature cost | `code/jepa_planning/goal_costs.py` |
| CEM closed loop, puck error, success at 1/2/5 cm, timing | `code/jepa_planning/baseline.py`, `code/experiments/run_baseline.py` |
| Fixed-bank candidate test (selection changed, regret) | `code/experiments/goal_photo_sensitivity.py` |
| Run metadata and no-overwrite outputs | pattern in `run_pusht_multi_goal_*.py` |

## Steps

### 1. Goal-image groups (data)
New `code/jepa_planning/goal_groups.py` and `code/experiments/make_goal_groups.py`.

- **Targets:** for each seed, reset, run the expert for 40 steps, then 30 more
  steps (same as `prepare_task`), and take that puck pose as the target. This
  keeps the training targets in the same distribution as the evaluation goals.
  Train, val and test use disjoint seed ranges, and test = the baseline's
  60000–60099.
- **Hand configurations:** a fixed bank of absolute hand positions sampled
  uniformly over the reachable workspace, **independent of the target**. The
  bank is split into train/val/test hands. Each image moves the hand to the
  bank position (`move_hand_keep_puck` with offset = bank − current hand) and
  restores the puck pose.
- **Acceptance checks (per image):** puck pose exactly equal to the group's
  pose (as `prepare_task` asserts) and hand at least 6 cm from the puck in xy.
  Log rejection counts. Rejection adds a small dependence between hand and
  target, so we report it.
- **Pool size:** generate one nested training pool large enough for every
  allocation (512 targets × 8 hands), then subsample: for (M, K), take the
  first M targets and K hands from each. All allocations then draw from the
  same pool, and the 3 resamples use different subsampling seeds.
- **Stored per image:** 224×224 frame, puck pose, hand position, seed, split,
  group id. Also log images/sec and the rejection rate ("label collection
  cost", §3.2).

### 2. Feature cache
`code/experiments/encode_goal_groups.py`: encode every image once with the
frozen encoder and store `[n, 256, 384]` in fp16 (~200 MB per 1024 images).
Fitting L then never touches the model.

### 3. Head and training
New `code/jepa_planning/linear_head.py`:
- `LinearGoalHead`: parameter L, `distance(a, b)` computed as
  `((a − b) @ L.T).square().sum(-1).mean(patches)`, and `renormalize_()`.
- `info_nce_loss(anchor, positive, negatives, L, tau)`.
- `AllocationSampler(pool, M, K, seed)`: exactly B images. It picks a group
  uniformly, two of its K images, and 7 negatives from other training groups
  (at B = 64, K = 8 that is all 7 other groups).
- Training: the same number of updates for every allocation, Adam on L, and
  renormalization after each step. Initialize L as a random Gaussian,
  normalized and seeded.
- τ and learning rate are chosen on **validation groups** with one shared grid
  across allocations and fixed before looking at test.

`code/experiments/train_linear_head.py`: loops over 9 allocations × 3
resamples and saves each L plus its config and train/val curves.

### 4. Cost integration
- `goal_costs.linear_goal_distance(candidates, goal, L)` (Eq. 3).
- `VisualCEMPlanner` takes a cost callable instead of hard-coding MSE, so every
  cost runs the same CEM code (Algorithm 1, step 5).
- `run_baseline.py --head path/to/L.pt` (default = plain MSE, unchanged).

### 5. Controls (§5)
Each control is a 32×384 matrix with Frobenius norm 1, so all of them plug
into the same cost:
1. Plain MSE (no head).
2. Random 32-dim projections, repeated over several seeds.
3. A fixed coordinate subset (32 of the 384 dims).
4. The same head trained with **shuffled pair labels**.

Later, only if the head passes the gates in step 6: a nonlinear head,
VICReg/SIGReg losses, and a coordinate readout and object-mask cost (with their
extra supervision reported). The PushT multi-goal averaging is also a cheap
extra baseline.

### 6. Evaluation, in order, with gates
1. **Photo level (cheap, cached features):** on held-out groups and hands, test
   whether same-goal pairs rank closer than different-goal pairs (retrieval
   accuracy and AUROC), and run a **sensitivity check**: does D_L still grow
   when the puck target moves slightly (1, 2, 5 cm)? This catches a head that
   simply collapses everything.
2. **Fixed candidate bank, actual endpoints:** reuse the
   `goal_photo_sensitivity.py` protocol on the dev seeds, with the cost as a
   flag. Measure selection changed L→R, success changed, regret (Eq. 5), and
   puck error to q.
3. **Same bank, predicted endpoints.** This is the **week-4 gate** from the
   proposal: does the head improve decisions made from predicted features? If
   not, stop and diagnose before adding methods. The pilot's rank-8 projection
   removed 51.8% of the variation without changing decisions, so decision
   metrics are what count, not feature variance.
4. **Closed-loop CEM:** 100 held-out test seeds × 2 goal photos × 3
   resamples, with identical planner budgets and CEM seeds across costs.
   Report puck error, success within 1/2/5 cm, native success separately, and
   planning time.
5. **Uncertainty:** bootstrap over tasks, keeping both photos of a task
   together (Agarwal et al.). Report IQM and confidence intervals.

### 7. Tests (`code/tests/`)
- `L = I_384 / sqrt(384)` reproduces `mean_squared_goal_distance` exactly.
- `||L||_F = 1` after every step, and a zero-norm update is rejected.
- The InfoNCE loss matches a brute-force computation.
- The sampler uses exactly B images, M×K structure, and negatives only from
  the training pool.
- Train/val/test target groups and hand banks are disjoint.
- The planner with the default cost gives the same output as now.

## Compute (from §3.4)
Data generation and encoding ~30 GPU-h, head fitting ~20 (mostly CPU, from the
cache), planning evaluation ~120, confirmation ~30. Time one small closed-loop
run first, before fixing the full run count.

## Open decisions for the team
1. Targets from expert-trajectory endpoints (recommended, matches evaluation)
   or uniform puck placement (wider coverage)?
2. Hand bank: xy only, or also height and gripper state?
3. Number of training updates per allocation, and the τ/learning-rate grid.
4. Who owns data/head training, planning integration, and evaluation (§7: each
   part is reviewed by another member).
