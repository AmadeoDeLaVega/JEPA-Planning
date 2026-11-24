# PushT spectral-isotropy head regularization and Jensen-certified MPC

Status (2026-11-24): development ablations on `val`, 100-target held-out photo/spectral evaluation on `test`, and paired closed-loop confirmation on `test` are complete. Code is in [`code/jepa_planning/trust_mpc.py`](../code/jepa_planning/trust_mpc.py), [`code/experiments/run_pusht_trust_mpc.py`](../code/experiments/run_pusht_trust_mpc.py), and [`code/tests/test_trust_mpc.py`](../code/tests/test_trust_mpc.py).

## Two failure modes diagnosed on `val`

1. **Rank-1 collapse explains the `B = 1024` degradation.** In [`pusht_linear_head.md`](pusht_linear_head.md), `B = 1024` performed worse in closed loop (`25.7–33.0 px`) than `B = 64` (`19.6–20.9 px`). Training `B = 1024` for `6000` steps (`3x` longer) on `val` did not fix closed-loop error (`24.30 -> 24.69 px`). Singular-value analysis of `W = L^T L` (`sum_k lambda_k = 1`) shows why: at `B = 64` and `B = 256`, effective rank `1 / sum_k lambda_k^2` is `5.23–5.92`, covering planar `(x, y, cos theta, sin theta)` pose variations. At `B = 1024`, unregularized InfoNCE collapses to **effective rank `1.39`** (`lambda_1 = 0.844`; `1.21` at `6000` steps), leaving closed-loop control blind to orthogonal pose errors while still scoring `1.000` on retrieval AUROC. Adding a log-det row-isotropy term (`-beta / r * log det(r L L^T + eps I)`, `beta = 0.05`) restores effective rank to `29.77`.
2. **Optimizer's Curse vs. multimodal contact barriers in CEM.** Returning raw `argmin` over `128` noisy rollouts degraded `val` mean error (`16.38 -> 26.95 px`) because the minimum of `127` noisy predictions systematically undercuts one denoised elite mean. Conversely, returning an unscored elite mean (`standard_cem`) cancels bimodal left/right contact plans across the T-block. `trust_mpc_plan` evaluates `64` iter-0 candidates (anchoring slot 0 at zero-displacement hold and slot 1 at the shifted previous plan), `63` iter-1 candidates, and uses rollout `#128` to score the final elite mean. It keeps the elite mean when Jensen's inequality holds (`c(elite_mean) <= mean(c_elites)`) and averages only the dominant directional elite mode when averaging crosses a cost barrier (`c(elite_mean) > mean(c_elites)`).

## Held-out `test` results

### 100-target photo and spectral check (`results/pusht_goal_groups/test`)

| Cost / head (`rank = 32` except MSE) | Effective rank | Top eigenvalue | Top-1 retrieval | 10 / 20 px shift ratio |
|---|---:|---:|---:|---:|
| Plain MSE (`I / sqrt(384)`) | 384.00 | 0.003 | 0.700 | 0.17 / 0.34 |
| `team_B64` (`2000` steps) | 5.23 | 0.389 | 0.995 | 0.43 / 1.21 |
| `team_B256` (`2000` steps) | 5.92 | 0.352 | 1.000 | 0.53 / 1.52 |
| `team_B1024` (`2000` steps) | 1.39 | 0.844 | 1.000 | 2.55 / 7.05 |
| `longer6k_B1024` (`6000` steps) | 1.21 | 0.907 | 1.000 | 2.09 / 5.75 |
| `iso_B1024` (`beta = 0.05`, `2000` steps) | 29.77 | 0.074 | 1.000 | 0.57 / 1.40 |

### Paired closed-loop confirmation (`12` held-out `test` tasks, `4` banks of `3`, `512` unrolls/episode)

Every method shares the exact same start states, goal photos, random normal draws, `4` replans (`20` simulator steps), and `128` world-model unrolls per replan (`~7.7 s/episode` on Apple M2 Pro MPS):

| Method | Mean final T error (px) [95% CI] | IQM (px) | Success <= 20 px | Pose success |
|---|---:|---:|---:|---:|
| `plain_mse + standard_cem` | 30.73 [16.40, 48.47] | 22.42 | 0.50 | 0.50 |
| `plain_mse + trust_mpc` | 25.98 [14.93, 40.81] | 21.33 | 0.42 | 0.42 |
| `team_B1024 + standard_cem` | 45.41 [31.18, 60.56] | 43.15 | 0.25 | 0.25 |
| `iso_B1024 + trust_mpc` | **21.68 [15.24, 28.32]** | **20.44** | **0.58** | **0.50** |
| `team_B64 + standard_cem` | 15.38 [10.09, 21.18] | 13.15 | 0.75 | 0.75 |
| `team_B64 + trust_mpc` | **13.81 [9.02, 19.07]** | **12.21** | 0.67 | 0.67 |

Key paired differences (10,000-sample task bootstrap, negative means lower error):
- **`team_B64 + trust_mpc` vs. `plain_mse + standard_cem`:** `-16.92 px [-36.78, -2.61]` (`4/4` banks won, `9` tasks better / `3` worse; statistically significant).
- **`iso_B1024 + trust_mpc` vs. `team_B1024 + standard_cem`:** `-23.73 px [-39.27, -9.68]` (`4/4` banks won, `10` tasks better / `2` worse; statistically significant cure of the `B = 1024` collapse).
- **`team_B64 + trust_mpc` vs. `team_B64 + standard_cem`:** `-1.57 px [-3.78, +0.57]` (`3/4` banks won, `7` tasks better / `3` worse; consistent with `val`'s `-1.69 px`, directional at `N = 12` tasks).
