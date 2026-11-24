"""Trust-regularized model predictive control and head calibration for frozen DINO-WM.

This module addresses two coupled failure modes in visual world-model planning:
1. Selecting InfoNCE temperatures by validation AUROC saturates near 1.0 and
   under-resolves small object displacements on large goal budgets.
2. Standard CEM returns an unscored elite mean and cold-starts around zero,
   which can miss multimodal contact early and knock a solved object off its
   target on later replans.
"""

from __future__ import annotations

import numpy as np
import torch

from jepa_planning.goal_costs import linear_goal_distance


def zero_displacement_chunk(
    action_mean: np.ndarray,
    action_std: np.ndarray,
    controls_per_chunk: int,
    action_clip: float,
) -> np.ndarray:
    """Normalized action chunk corresponding to zero raw controller displacement."""
    single_step_hold = np.clip(-action_mean / action_std, -action_clip, action_clip)
    return np.tile(single_step_hold.astype(np.float32), controls_per_chunk)


def select_calibrated_hyperparameters(
    grid_rows: list[dict],
    min_top1: float = 0.98,
    metric: str = "sensitivity_shift_px_10",
) -> tuple[float, float]:
    """Pick (tau, lr) on validation groups using local object-shift sensitivity.

    Validation AUROC saturates near 1.0 across temperatures. Maximizing the
    median 10 px shift-to-nuisance ratio among pairs that maintain strong
    same-target retrieval keeps local pose gradients sharp at large budgets.
    """
    grouped: dict[tuple[float, float], list[dict]] = {}
    for row in grid_rows:
        key = (float(row["tau"]), float(row["lr"]))
        grouped.setdefault(key, []).append(row)

    eligible = [
        (key, rows)
        for key, rows in grouped.items()
        if np.mean([r["top1"] for r in rows]) >= min_top1
    ]
    pool = eligible if eligible else list(grouped.items())

    best_key, _ = max(
        pool,
        key=lambda item: float(np.mean([r[metric] for r in item[1]])),
    )
    return best_key


@torch.inference_mode()
def rollout_head_costs(
    model,
    encoded_state,
    candidate_actions: np.ndarray,
    goal_features: torch.Tensor,
    head: torch.Tensor,
    batch_size: int = 64,
    running_weight: float = 0.0,
) -> np.ndarray:
    """Predict rollouts for candidate_actions [N, H, A] and score them under head L."""
    goal = goal_features[None].to(device=model.device, dtype=torch.float32)
    head_device = head.to(device=model.device, dtype=torch.float32)
    num_candidates = len(candidate_actions)
    feature_dim = goal_features.shape[-1]
    all_costs = []

    for start in range(0, num_candidates, batch_size):
        chunk_batch = candidate_actions[start : start + batch_size]
        action_tensor = torch.from_numpy(chunk_batch).permute(1, 0, 2).to(model.device)
        predicted_visual = model.unroll(encoded_state, action_tensor)["visual"]

        terminal_features = predicted_visual[-1].reshape(len(chunk_batch), -1, feature_dim).float()
        cost = linear_goal_distance(terminal_features, goal, head_device)

        if running_weight > 0.0 and predicted_visual.shape[0] > 2:
            mid_features = predicted_visual[-2].reshape(len(chunk_batch), -1, feature_dim).float()
            mid_cost = linear_goal_distance(mid_features, goal, head_device)
            cost = (1.0 - running_weight) * cost + running_weight * mid_cost

        all_costs.append(cost.cpu().numpy())

    return np.concatenate(all_costs)


def action_trust_penalty(
    candidate_actions: np.ndarray,
    hold_sequence: np.ndarray,
    action_weight: float,
    smoothness_weight: float,
) -> np.ndarray:
    """Dimensionless effort and inter-chunk smoothness penalty around hold_sequence."""
    num_candidates, horizon, _ = candidate_actions.shape
    displacement = (candidate_actions - hold_sequence[None, :]).reshape(num_candidates, -1)
    penalty = action_weight * np.mean(displacement ** 2, axis=1)

    if smoothness_weight > 0.0 and horizon > 1:
        chunk_steps = (candidate_actions[:, 1:, :] - candidate_actions[:, :-1, :]).reshape(num_candidates, -1)
        penalty = penalty + smoothness_weight * np.mean(chunk_steps ** 2, axis=1)

    return penalty.astype(np.float32)


@torch.inference_mode()
def trust_mpc_plan(
    model,
    observation: dict,
    goal_features: torch.Tensor,
    head: torch.Tensor,
    normals: np.ndarray,
    hold_chunk: np.ndarray,
    previous_plan: np.ndarray | None = None,
    *,
    num_elites: int = 8,
    action_clip: float = 2.0,
    batch_size: int = 64,
    action_weight: float = 0.015,
    smoothness_weight: float = 0.01,
    stay_margin: float = 0.03,
    running_weight: float = 0.0,
    std_floor: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Plan one receding-horizon action sequence under a matched rollout budget.

    Given standard-normal noise of shape [2, N, H, A], iteration 0 evaluates N
    candidates (anchoring slot 0 at zero-displacement hold and slot 1 at the
    shifted previous plan when available). Iteration 1 evaluates N - 1 samples
    around the elite distribution and reserves the final rollout to explicitly
    score the final elite mean, keeping total unrolls equal to 2 * N.
    """
    visual = torch.from_numpy(observation["visual"]).permute(2, 0, 1)[None, None]
    proprio = torch.from_numpy(np.asarray(observation["proprio"], dtype=np.float32))[None, None]
    encoded_state = model.encode({"visual": visual, "proprio": proprio})

    _, samples_per_iter, horizon, action_dim = normals.shape
    hold_sequence = np.tile(hold_chunk[None, :], (horizon, 1)).astype(np.float32)

    mean = np.zeros((horizon, action_dim), dtype=np.float32)
    std = np.ones_like(mean)

    # Iteration 0: Gaussian exploration plus deterministic hold and warm-start anchors.
    iter0_actions = np.clip(mean[None] + std[None] * normals[0], -action_clip, action_clip).astype(np.float32)
    iter0_actions[0] = hold_sequence
    if previous_plan is not None and samples_per_iter > 1:
        shifted_plan = np.vstack([previous_plan[1:], hold_chunk[None, :]]).astype(np.float32)
        iter0_actions[1] = np.clip(shifted_plan, -action_clip, action_clip)

    iter0_raw_costs = rollout_head_costs(
        model, encoded_state, iter0_actions, goal_features, head, batch_size, running_weight
    )
    hold_cost = max(float(iter0_raw_costs[0]), 1e-6)

    iter0_objective = (iter0_raw_costs / hold_cost) + action_trust_penalty(
        iter0_actions, hold_sequence, action_weight, smoothness_weight
    )
    elite_indices = np.argsort(iter0_objective)[:num_elites]
    elite_mean = iter0_actions[elite_indices].mean(axis=0)
    elite_std = np.maximum(iter0_actions[elite_indices].std(axis=0), std_floor)

    # Iteration 1: evaluate N - 1 candidates so rollout #2N can certify the final elite mean.
    iter1_count = samples_per_iter - 1
    iter1_actions = np.clip(
        elite_mean[None] + elite_std[None] * normals[1, :iter1_count],
        -action_clip,
        action_clip,
    ).astype(np.float32)
    iter1_actions[0] = elite_mean
    iter1_raw_costs = rollout_head_costs(
        model, encoded_state, iter1_actions, goal_features, head, batch_size, running_weight
    )

    pooled_actions = np.concatenate([iter0_actions, iter1_actions], axis=0)
    pooled_raw_costs = np.concatenate([iter0_raw_costs, iter1_raw_costs], axis=0)
    pooled_objective = (pooled_raw_costs / hold_cost) + action_trust_penalty(
        pooled_actions, hold_sequence, action_weight, smoothness_weight
    )

    final_elites = np.argsort(pooled_objective)[:num_elites]
    final_elite_mean = np.clip(
        pooled_actions[final_elites].mean(axis=0), -action_clip, action_clip
    ).astype(np.float32)[None, :, :]
    final_mean_raw_cost = rollout_head_costs(
        model, encoded_state, final_elite_mean, goal_features, head, batch_size, running_weight
    )

    certified_actions = np.concatenate([pooled_actions, final_elite_mean], axis=0)
    certified_raw_costs = np.concatenate([pooled_raw_costs, final_mean_raw_cost], axis=0)
    certified_objective = (certified_raw_costs / hold_cost) + action_trust_penalty(
        certified_actions, hold_sequence, action_weight, smoothness_weight
    )

    best_index = int(np.argmin(certified_objective))
    used_stay_gate = False
    if stay_margin > 0.0 and certified_raw_costs[best_index] > (1.0 - stay_margin) * float(iter0_raw_costs[0]):
        best_index = 0
        used_stay_gate = True

    selected_plan = certified_actions[best_index]
    diagnostics = {
        "unrolls": int(len(certified_actions)),
        "hold_cost": float(iter0_raw_costs[0]),
        "selected_cost": float(certified_raw_costs[best_index]),
        "selected_mean_proposal": bool(best_index == len(certified_actions) - 1),
        "used_stay_gate": used_stay_gate,
    }
    return selected_plan[0], selected_plan, diagnostics
