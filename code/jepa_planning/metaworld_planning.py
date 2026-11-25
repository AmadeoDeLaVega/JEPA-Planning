"""Box-constrained CEM, Jensen-certified Trust-MPC, and metrics for Meta-World Push.

Extends the frozen DINO-WM linear-head and Trust-MPC methods from 2D PushT to
3D articulated Sawyer manipulation (`Push-v3`), handling three key differences:

1. Asymmetric normalized action bounds: in Meta-World Push the dataset action
   mean is substantially nonzero, so raw ``[-1, 1]`` control limits map to
   coordinate-specific normalized bounds ``[lower, upper]`` and true zero
   displacement lies at ``-action_mean / action_std`` rather than ``0``.
2. Multi-goal prototypes and dense-horizon scoring: candidate rollouts can be
   scored against single ``[P, D]`` or multi-photo ``[K, P, D]`` goal banks and
   optional nonnegative horizon weights ``step_weights`` via
   :mod:`jepa_planning.calibrated_objectives`.
3. Dual-certified Trust-MPC: under exact ``2 * N`` unroll budget parity, both
   the uniform elite mean and the dominant half-space mode mean are evaluated
   through the world model so no uncertified trajectory is ever returned.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from jepa_planning.calibrated_objectives import (
    dense_horizon_head_distance,
    projected_multi_goal_distance,
)
from jepa_planning.spatial_channel_head import spatial_patch_scale


def metaworld_action_bounds(
    action_mean: np.ndarray,
    action_std: np.ndarray,
    controls_per_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized ``(lower, upper, hold_chunk)`` for one Meta-World action chunk.

    Maps raw environment limits ``[-1, +1]`` and zero displacement ``0`` into
    normalized action space and tiles them across ``controls_per_chunk`` steps.
    """
    if isinstance(controls_per_chunk, bool) or not isinstance(
        controls_per_chunk, (int, np.integer)
    ) or int(controls_per_chunk) < 1:
        raise ValueError("controls_per_chunk must be a positive integer")

    mean = np.asarray(action_mean, dtype=np.float64)
    std = np.asarray(action_std, dtype=np.float64)
    if mean.ndim != 1 or mean.size == 0 or mean.shape != std.shape:
        raise ValueError("action_mean and action_std must be non-empty 1D arrays of matching shape")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("action_mean and action_std must contain finite values")
    if np.any(std <= 0.0):
        raise ValueError("action_std must be strictly positive")

    lower_step = (-1.0 - mean) / std
    upper_step = (1.0 - mean) / std
    hold_step = np.clip(-mean / std, lower_step, upper_step)

    repeats = int(controls_per_chunk)
    return (
        np.tile(lower_step, repeats).astype(np.float32),
        np.tile(upper_step, repeats).astype(np.float32),
        np.tile(hold_step, repeats).astype(np.float32),
    )


def puck_metrics_cm(
    candidate_puck_xyz: np.ndarray,
    goal_puck_xyz: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute 3D/XY puck position errors in cm and 5/3/2 cm success flags."""
    candidates = np.asarray(candidate_puck_xyz, dtype=np.float64)
    goal = np.asarray(goal_puck_xyz, dtype=np.float64)
    if candidates.ndim < 1 or candidates.shape[-1] != 3 or candidates.size == 0:
        raise ValueError("candidate_puck_xyz must be non-empty with shape [..., 3]")
    if goal.shape != (3,):
        raise ValueError("goal_puck_xyz must have shape [3]")
    if not np.all(np.isfinite(candidates)) or not np.all(np.isfinite(goal)):
        raise ValueError("puck coordinates must be finite")

    diff = candidates - goal
    error_3d_cm = np.asarray(100.0 * np.linalg.norm(diff, axis=-1), dtype=np.float64)
    error_xy_cm = np.asarray(100.0 * np.linalg.norm(diff[..., :2], axis=-1), dtype=np.float64)
    return {
        "error_3d_cm": error_3d_cm,
        "error_xy_cm": error_xy_cm,
        "success_5cm": np.asarray(error_3d_cm <= 5.0, dtype=bool),
        "success_3cm": np.asarray(error_3d_cm <= 3.0, dtype=bool),
        "success_2cm": np.asarray(error_3d_cm <= 2.0, dtype=bool),
    }


def render_puck_sensitivity_pairs(
    env,
    goal_snap: dict,
    hand_xy: np.ndarray,
    shifts_m: Sequence[float] = (0.02, 0.05),
) -> dict[str, np.ndarray]:
    """Render reference and x-shifted puck frames at a fixed Sawyer arm pose.

    Restores ``goal_snap``, moves the 7-DoF Sawyer gripper to ``hand_xy``
    without disturbing the puck, renders the reference ``224x224`` RGB frame and
    4D proprioception, then translates the puck's ``x`` coordinate
    (``qpos[9] += shift``) for each ``shift`` in ``shifts_m`` while keeping the
    arm configuration identical.
    """
    from jepa_planning import metaworld_push as mp

    target_xy = np.asarray(hand_xy, dtype=np.float64)
    if target_xy.shape != (2,) or not np.all(np.isfinite(target_xy)):
        raise ValueError("hand_xy must be a finite [2] array")
    shifts = np.asarray(list(shifts_m), dtype=np.float64)
    if shifts.ndim != 1 or shifts.size == 0 or not np.all(np.isfinite(shifts)):
        raise ValueError("shifts_m must be a non-empty sequence of finite distances in metres")

    mp.restore(env, goal_snap)
    offset = np.array(
        [target_xy[0] - mp.hand_pos(env)[0], target_xy[1] - mp.hand_pos(env)[1], 0.0],
        dtype=np.float64,
    )
    mp.move_hand_keep_puck(env, offset)

    unwrapped = env.unwrapped
    base_qpos = unwrapped.data.qpos.copy()
    base_qvel = unwrapped.data.qvel.copy()

    reference_visual = env.render().copy()
    reference_proprio = unwrapped._get_curr_obs_combined_no_goal()[:4].astype(np.float32)
    reference_hand_pos = mp.hand_pos(env)
    reference_puck_pose = mp.puck_pose(env)

    visuals, proprios, hand_positions, puck_poses = [], [], [], []
    for shift in shifts:
        qpos = base_qpos.copy()
        qvel = base_qvel.copy()
        qpos[9] += float(shift)
        qvel[9:15] = 0.0
        unwrapped.set_state(qpos, qvel)
        visuals.append(env.render().copy())
        proprios.append(unwrapped._get_curr_obs_combined_no_goal()[:4].astype(np.float32))
        hand_positions.append(mp.hand_pos(env))
        puck_poses.append(mp.puck_pose(env))

    return {
        "reference_visual": reference_visual,
        "reference_proprio": reference_proprio,
        "reference_hand_pos": reference_hand_pos,
        "reference_puck_pose": reference_puck_pose,
        "visual": np.stack(visuals),
        "proprio": np.stack(proprios).astype(np.float32),
        "hand_pos": np.stack(hand_positions),
        "puck_pose": np.stack(puck_poses),
        "shifts_m": shifts,
        "kind": np.array(["shift_cm"] * len(shifts)),
        "amount": np.round(shifts * 100.0, 6),
    }


def _encode_observation(model, observation: dict):
    """Encode either numpy ``[H, W, 3]``/``[4]`` or pre-batched ``mp.observe`` tensors."""
    raw_visual = observation["visual"]
    raw_proprio = observation["proprio"]

    if isinstance(raw_visual, torch.Tensor) and raw_visual.ndim == 5:
        visual = raw_visual
    else:
        visual_tensor = torch.as_tensor(raw_visual)
        if visual_tensor.ndim != 3:
            raise ValueError("observation['visual'] must have shape [H, W, 3] or [1, 1, 3, H, W]")
        visual = visual_tensor.permute(2, 0, 1)[None, None]

    if isinstance(raw_proprio, torch.Tensor) and raw_proprio.ndim == 3:
        proprio = raw_proprio.float()
    else:
        proprio_arr = np.asarray(raw_proprio, dtype=np.float32)
        if proprio_arr.ndim != 1:
            raise ValueError("observation['proprio'] must have shape [P_dim] or [1, 1, P_dim]")
        proprio = torch.from_numpy(proprio_arr)[None, None]

    return model.encode({"visual": visual, "proprio": proprio})


def _validate_box(lower: np.ndarray, upper: np.ndarray, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate and cast coordinate-wise action bounds to float32 ``[A]``."""
    lo = np.asarray(lower, dtype=np.float32)
    hi = np.asarray(upper, dtype=np.float32)
    if lo.shape != (action_dim,) or hi.shape != (action_dim,):
        raise ValueError(f"lower and upper must have shape ({action_dim},)")
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("lower and upper bounds must be finite")
    if np.any(lo > hi):
        raise ValueError("lower bound must not exceed upper bound")
    return lo, hi


@torch.inference_mode()
def rollout_metaworld_costs(
    model,
    encoded_state,
    candidate_actions: np.ndarray,
    goal_features: torch.Tensor,
    head: torch.Tensor,
    batch_size: int = 32,
    step_weights: Sequence[float] | None = None,
    spatial_weights: torch.Tensor | None = None,
) -> np.ndarray:
    """Predict rollouts for ``[N, H, A]`` actions and score under head ``L``.

    Supports single goals ``[P, D]``, multi-goal banks ``[K, P, D]``, optional
    nonnegative horizon ``step_weights`` of length ``H``, and optional simplex
    ``spatial_weights`` of length ``P``.
    """
    if candidate_actions.ndim != 3 or min(candidate_actions.shape) == 0:
        raise ValueError("candidate_actions must be non-empty with shape [N, H, A]")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    device = torch.device(model.device)
    goal = goal_features.to(device=device, dtype=torch.float32)
    head_device = head.to(device=device, dtype=torch.float32)
    feature_dim = goal.shape[-1]
    horizon = candidate_actions.shape[1]

    patch_scale: torch.Tensor | None = None
    if spatial_weights is not None:
        if goal.ndim not in (2, 3):
            raise ValueError("goal_features must have shape [P, D] or [K, P, D]")
        num_patches = goal.shape[-2]
        patch_scale = spatial_patch_scale(
            spatial_weights, num_patches, device=device, dtype=torch.float32
        )
        if goal.ndim == 2:
            goal = goal * patch_scale[:, None]
        else:
            goal = goal * patch_scale[None, :, None]

    all_costs = []
    for start in range(0, len(candidate_actions), batch_size):
        chunk_batch = np.ascontiguousarray(candidate_actions[start : start + batch_size], dtype=np.float32)
        action_tensor = torch.from_numpy(chunk_batch).permute(1, 0, 2).to(device)
        predicted_visual = model.unroll(encoded_state, action_tensor)["visual"]
        if step_weights is None:
            terminal_features = predicted_visual[-1].reshape(len(chunk_batch), -1, feature_dim).float()
            if patch_scale is not None:
                terminal_features = terminal_features * patch_scale[None, :, None]
            cost = projected_multi_goal_distance(terminal_features, goal, head_device, exact_mean=False)
        else:
            rollout_steps = predicted_visual[-horizon:]
            rollout_features = rollout_steps.reshape(horizon, len(chunk_batch), -1, feature_dim).float()
            if patch_scale is not None:
                rollout_features = rollout_features * patch_scale[None, None, :, None]
            cost = dense_horizon_head_distance(
                rollout_features, goal, head_device, step_weights=step_weights
            )
        all_costs.append(cost.detach().cpu().numpy())

    return np.concatenate(all_costs)


@torch.inference_mode()
def metaworld_cem_plan(
    model,
    observation: dict,
    goal_features: torch.Tensor,
    head: torch.Tensor,
    normals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    num_elites: int = 8,
    batch_size: int = 32,
    step_weights: Sequence[float] | None = None,
    init_mean: np.ndarray | None = None,
    init_std: float = 1.0,
    spatial_weights: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Box-constrained CEM planner supporting multi-goal, dense-horizon, and spatial-weighted scoring."""
    normals = np.asarray(normals, dtype=np.float32)
    if normals.ndim != 4 or min(normals.shape) == 0:
        raise ValueError("normals must have shape [iterations, samples, horizon, action_dim]")
    iterations, samples_per_iter, horizon, action_dim = normals.shape
    lo, hi = _validate_box(lower, upper, action_dim)
    if not 1 <= num_elites <= samples_per_iter:
        raise ValueError("num_elites must be between 1 and samples_per_iter")
    if not np.isfinite(init_std) or init_std <= 0.0:
        raise ValueError("init_std must be finite and positive")

    encoded_state = _encode_observation(model, observation)

    if init_mean is None:
        mean = np.clip(np.zeros((horizon, action_dim), dtype=np.float32), lo[None, :], hi[None, :])
    else:
        mean_arr = np.asarray(init_mean, dtype=np.float32)
        if mean_arr.shape == (action_dim,):
            mean_arr = np.tile(mean_arr[None, :], (horizon, 1))
        elif mean_arr.shape != (horizon, action_dim):
            raise ValueError(f"init_mean must have shape ({action_dim},) or ({horizon}, {action_dim})")
        mean = np.clip(mean_arr, lo[None, :], hi[None, :]).astype(np.float32)
    std = np.full((horizon, action_dim), float(init_std), dtype=np.float32)

    total_unrolls = 0
    best_cost = float("inf")
    elite_mean_cost = float("nan")
    for iteration in range(iterations):
        actions = np.clip(
            mean[None] + std[None] * normals[iteration],
            lo[None, None, :],
            hi[None, None, :],
        ).astype(np.float32)
        actions[0] = mean
        costs = rollout_metaworld_costs(
            model,
            encoded_state,
            actions,
            goal_features,
            head,
            batch_size=batch_size,
            step_weights=step_weights,
            spatial_weights=spatial_weights,
        )
        total_unrolls += len(actions)
        elites = np.argsort(costs)[:num_elites]
        mean = np.clip(actions[elites].mean(axis=0), lo[None, :], hi[None, :]).astype(np.float32)
        std = actions[elites].std(axis=0).astype(np.float32)
        best_cost = float(costs[elites[0]])
        elite_mean_cost = float(costs[elites].mean())

    selected_plan = np.clip(mean, lo[None, :], hi[None, :]).astype(np.float32)
    diagnostics = {
        "unrolls": int(total_unrolls),
        "best_cost": best_cost,
        "elite_mean_cost": elite_mean_cost,
        "selected_cost": best_cost,
    }
    return selected_plan[0], selected_plan, diagnostics


@torch.inference_mode()
def metaworld_trust_mpc_plan(
    model,
    observation: dict,
    goal_features: torch.Tensor,
    head: torch.Tensor,
    normals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    hold_chunk: np.ndarray,
    previous_plan: np.ndarray | None = None,
    *,
    num_elites: int = 8,
    batch_size: int = 32,
    stay_margin: float = 0.02,
    step_weights: Sequence[float] | None = None,
    dual_certify: bool = True,
    init_std: float = 1.0,
    spatial_weights: torch.Tensor | None = None,
    sample_around_prior: bool = False,
    step_chunks: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Box-constrained Jensen-certified Trust-MPC with exact ``2 * N`` rollout parity.

    Iteration 0 evaluates ``N`` candidates centered at ``hold_sequence`` (or at
    normalized ``0`` clipped to ``[lower, upper]`` when
    ``sample_around_prior=True``), anchoring slot 0 at ``hold_sequence`` and
    slot 1 at the ``step_chunks``-shifted warm-start plan when available.
    Iteration 1 evaluates ``N - 2`` samples when ``dual_certify=True`` (or
    ``N - 1`` when ``dual_certify=False``) and spends the remaining rollout(s)
    to certify ``elite_mean`` (and the dominant half-space ``mode_mean`` when
    ``dual_certify=True``).
    """
    normals = np.asarray(normals, dtype=np.float32)
    if normals.ndim != 4 or normals.shape[0] < 2 or min(normals.shape[1:]) == 0:
        raise ValueError("normals must have shape [>=2, samples, horizon, action_dim]")
    _, samples_per_iter, horizon, action_dim = normals.shape
    lo, hi = _validate_box(lower, upper, action_dim)

    hold_arr = np.asarray(hold_chunk, dtype=np.float32)
    if hold_arr.shape != (action_dim,) or not np.all(np.isfinite(hold_arr)):
        raise ValueError(f"hold_chunk must be a finite array of shape ({action_dim},)")
    hold_clipped = np.clip(hold_arr, lo, hi).astype(np.float32)
    hold_sequence = np.tile(hold_clipped[None, :], (horizon, 1)).astype(np.float32)

    if isinstance(step_chunks, bool) or not isinstance(step_chunks, (int, np.integer)) or int(step_chunks) < 1:
        raise ValueError("step_chunks must be a positive integer")
    shift_steps = int(step_chunks)

    cert_count = 2 if dual_certify else 1
    iter1_count = samples_per_iter - cert_count
    if iter1_count < 1:
        raise ValueError(f"samples_per_iter ({samples_per_iter}) must exceed certification count ({cert_count})")
    if not 1 <= num_elites <= iter1_count:
        raise ValueError(f"num_elites ({num_elites}) must be between 1 and {iter1_count}")
    if not np.isfinite(init_std) or init_std <= 0.0:
        raise ValueError("init_std must be finite and positive")

    encoded_state = _encode_observation(model, observation)

    if sample_around_prior:
        iter0_center = np.clip(
            np.zeros((horizon, action_dim), dtype=np.float32),
            lo[None, :],
            hi[None, :],
        ).astype(np.float32)
    else:
        iter0_center = hold_sequence

    std0 = np.full((horizon, action_dim), float(init_std), dtype=np.float32)
    iter0_actions = np.clip(
        iter0_center[None] + std0[None] * normals[0],
        lo[None, None, :],
        hi[None, None, :],
    ).astype(np.float32)
    iter0_actions[0] = hold_sequence
    if previous_plan is not None and samples_per_iter > 1:
        prev = np.asarray(previous_plan, dtype=np.float32)
        if prev.shape != (horizon, action_dim):
            raise ValueError(f"previous_plan must have shape ({horizon}, {action_dim})")
        pad_count = min(shift_steps, horizon)
        tail = np.tile(hold_clipped[None, :], (pad_count, 1))
        shifted_plan = np.vstack([prev[shift_steps:], tail])[:horizon].astype(np.float32)
        iter0_actions[1] = np.clip(shifted_plan, lo[None, :], hi[None, :])

    iter0_costs = rollout_metaworld_costs(
        model,
        encoded_state,
        iter0_actions,
        goal_features,
        head,
        batch_size=batch_size,
        step_weights=step_weights,
        spatial_weights=spatial_weights,
    )
    hold_cost = float(iter0_costs[0])

    iter0_elites = np.argsort(iter0_costs)[:num_elites]
    mean1 = np.clip(iter0_actions[iter0_elites].mean(axis=0), lo[None, :], hi[None, :]).astype(np.float32)
    std1 = iter0_actions[iter0_elites].std(axis=0).astype(np.float32)

    iter1_actions = np.clip(
        mean1[None] + std1[None] * normals[1, :iter1_count],
        lo[None, None, :],
        hi[None, None, :],
    ).astype(np.float32)
    iter1_actions[0] = mean1
    iter1_costs = rollout_metaworld_costs(
        model,
        encoded_state,
        iter1_actions,
        goal_features,
        head,
        batch_size=batch_size,
        step_weights=step_weights,
        spatial_weights=spatial_weights,
    )

    final_elites = np.argsort(iter1_costs)[:num_elites]
    elite_actions = iter1_actions[final_elites]
    elite_costs = iter1_costs[final_elites]
    elite_mean = np.clip(elite_actions.mean(axis=0), lo[None, :], hi[None, :]).astype(np.float32)

    lead_direction = elite_actions[0, 0] - hold_clipped
    if float(np.dot(lead_direction, lead_direction)) > 1e-12:
        alignment = np.sum(
            (elite_actions[:, 0] - hold_clipped[None, :]) * lead_direction[None, :],
            axis=1,
        )
        dominant_mode = elite_actions[alignment >= 0.0]
    else:
        dominant_mode = elite_actions[: max(1, len(elite_actions) // 2)]

    mode_mean = (
        dominant_mode[0]
        if len(dominant_mode) <= 1
        else np.clip(dominant_mode.mean(axis=0), lo[None, :], hi[None, :]).astype(np.float32)
    )

    if dual_certify:
        cert_actions = np.stack([elite_mean, mode_mean], axis=0)
        cert_costs = rollout_metaworld_costs(
            model,
            encoded_state,
            cert_actions,
            goal_features,
            head,
            batch_size=batch_size,
            step_weights=step_weights,
            spatial_weights=spatial_weights,
        )
        mean_cost = float(cert_costs[0])
        mode_cost = float(cert_costs[1])
    else:
        mean_cost = float(
            rollout_metaworld_costs(
                model,
                encoded_state,
                elite_mean[None, :, :],
                goal_features,
                head,
                batch_size=batch_size,
                step_weights=step_weights,
                spatial_weights=spatial_weights,
            )[0]
        )
        mode_cost = float("nan")

    elite_cost_mean = float(elite_costs.mean())
    jensen_violation = mean_cost > elite_cost_mean
    if not jensen_violation:
        selected_plan = elite_mean
        selected_cost = mean_cost
    elif dual_certify:
        if mode_cost <= elite_cost_mean or mode_cost <= float(elite_costs[0]):
            selected_plan = mode_mean
            selected_cost = mode_cost
        else:
            selected_plan = elite_actions[0]
            selected_cost = float(elite_costs[0])
    else:
        selected_plan = mode_mean
        selected_cost = float(elite_costs[0])

    used_stay_gate = False
    if (
        stay_margin > 0.0
        and selected_cost > (1.0 - stay_margin) * hold_cost
        and float(elite_costs[0]) > (1.0 - stay_margin) * hold_cost
    ):
        selected_plan = hold_sequence
        selected_cost = hold_cost
        used_stay_gate = True

    diagnostics = {
        "unrolls": int(len(iter0_actions) + len(iter1_actions) + cert_count),
        "hold_cost": hold_cost,
        "elite_mean_cost": mean_cost,
        "mode_mean_cost": mode_cost,
        "best_elite_cost": float(elite_costs[0]),
        "selected_cost": selected_cost,
        "jensen_violation": bool(jensen_violation),
        "used_stay_gate": used_stay_gate,
        "dual_certify": bool(dual_certify),
    }
    return selected_plan[0], selected_plan, diagnostics
