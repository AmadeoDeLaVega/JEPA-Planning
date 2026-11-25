"""Jensen-certified MPC and spectral head fitting for frozen DINO-WM.

Consolidates three empirical findings from the PushT development split:
1. At large contrastive budgets (B = 1024), trace-normalized InfoNCE collapses
   effective rank toward 1 (top eigenvalue > 0.84), blinding closed-loop control
   to orthogonal pose coordinates. Log-det spectral isotropy and exact k-Fantope
   spectral projection preserve multi-dimensional pose sensitivity.
2. Comparing 127 noisy Gaussian rollouts directly against 1 denoised elite mean
   suffers from Optimizer's Curse, while returning an unscored elite mean fails
   across non-convex contact barriers. Evaluating the final elite mean (and,
   optionally, the dominant-mode split mean) within the exact 2 * N rollout
   budget retains elite-mean denoising on unimodal basins and certifies mode
   splitting only on strict Jensen barrier violations.
3. Averaging K equivalent goal photos into a projected prototype reduces
   multi-goal scoring to a single centroid comparison with zero extra rollouts.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import torch
import torch.nn.functional as F

from jepa_planning import linear_head as lh
from jepa_planning.calibrated_objectives import dense_horizon_head_distance
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


def effective_rank(head: torch.Tensor) -> float:
    """Participation-ratio effective rank 1 / sum_k lambda_k^2 of W = L^T L."""
    singular_values = torch.linalg.svdvals(head.float())
    eigenvalues = singular_values.square()
    eigenvalues = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    return float(1.0 / eigenvalues.square().sum())


def project_capped_simplex(
    values: np.ndarray,
    cap: float,
    total: float = 1.0,
) -> np.ndarray:
    """Exact Euclidean projection onto ``{lam : 0 <= lam_i <= cap, sum(lam) = total}``."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("values must be a non-empty 1D array")
    if not np.isfinite(arr).all():
        raise ValueError("values must contain only finite numbers")
    if isinstance(cap, bool) or not math.isfinite(float(cap)) or float(cap) <= 0.0:
        raise ValueError("cap must be a finite positive number")
    if isinstance(total, bool) or not math.isfinite(float(total)) or float(total) < 0.0:
        raise ValueError("total must be a finite nonnegative number")

    cap_f = float(cap)
    total_f = float(total)
    if cap_f * len(arr) < total_f - 1e-9:
        raise ValueError(
            f"infeasible capped simplex: cap * len(values) = {cap_f * len(arr)} < total = {total_f}"
        )
    if total_f == 0.0:
        return np.zeros_like(arr)
    if cap_f * len(arr) <= total_f:
        return np.full(len(arr), cap_f, dtype=np.float64)

    lo = float(arr.min()) - cap_f
    hi = float(arr.max())
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(np.clip(arr - mid, 0.0, cap_f).sum()) > total_f:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * max(1.0, abs(lo), abs(hi)):
            break
    return np.clip(arr - 0.5 * (lo + hi), 0.0, cap_f)


def project_fantope_(weight: torch.Tensor, k_fantope: float) -> None:
    """Project ``weight`` [r, D] in-place so squared singular values lie in the k-Fantope.

    Enforces ``sum(s^2) = 1`` and ``max(s^2) <= 1 / k_fantope``, guaranteeing
    ``||L||_F = 1``, ``||L^T L||_op <= 1 / k_fantope``, and
    ``effective_rank(L) >= k_fantope`` while allowing trailing singular values
    to vanish.
    """
    if weight.ndim != 2 or min(weight.shape) == 0:
        raise ValueError("weight must be a non-empty 2D tensor [r, D]")
    if isinstance(k_fantope, bool) or not math.isfinite(float(k_fantope)):
        raise ValueError("k_fantope must be a finite number")
    k_val = float(k_fantope)
    max_rank = min(weight.shape)
    if not 1.0 <= k_val <= float(weight.shape[0]) or k_val > float(max_rank):
        raise ValueError(f"k_fantope must satisfy 1.0 <= k_fantope <= {max_rank}, got {k_val}")

    with torch.no_grad():
        work = weight if weight.dtype in (torch.float32, torch.float64) else weight.float()
        u, singular_values, vh = torch.linalg.svd(work, full_matrices=False)
        eigenvalues = singular_values.square().detach().cpu().numpy()
        projected_eigenvalues = project_capped_simplex(
            eigenvalues,
            cap=1.0 / k_val,
            total=1.0,
        )
        new_singular_values = torch.from_numpy(
            np.sqrt(np.maximum(projected_eigenvalues, 0.0))
        ).to(device=weight.device, dtype=u.dtype)
        reconstructed = (u * new_singular_values.unsqueeze(0)).matmul(vh)
        weight.copy_(reconstructed.to(dtype=weight.dtype))


def _batch_infonce_loss(
    subset: torch.Tensor,
    local_map: np.ndarray,
    draw: dict[str, np.ndarray],
    head_weight: torch.Tensor,
    *,
    batch_size: int,
    patches: int,
    rank: int,
    tau: float,
    targets: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute batch-indexed InfoNCE loss without projecting unused pool images."""
    anc_idx = torch.as_tensor(local_map[draw["anchor"]], dtype=torch.long, device=device)
    pos_idx = torch.as_tensor(local_map[draw["positive"]], dtype=torch.long, device=device)
    neg_idx = torch.as_tensor(local_map[draw["negatives"]].ravel(), dtype=torch.long, device=device)

    head_t = head_weight.transpose(0, 1)
    anchor = torch.index_select(subset, 0, anc_idx).matmul(head_t).unsqueeze(1)
    positive = torch.index_select(subset, 0, pos_idx).matmul(head_t).unsqueeze(1)
    negatives = (
        torch.index_select(subset, 0, neg_idx)
        .matmul(head_t)
        .reshape(batch_size, -1, patches, rank)
    )

    candidates = torch.cat([positive, negatives], dim=1)
    distances = (candidates - anchor).square().sum(dim=-1).mean(dim=-1)
    return F.cross_entropy(-distances / tau, targets)


def fit_isotropy_head(
    features: torch.Tensor,
    sampler: lh.AllocationSampler,
    *,
    steps: int = 2000,
    batch_size: int = 64,
    tau: float = 0.3,
    lr: float = 0.003,
    isotropy_weight: float = 0.05,
    seed: int = 0,
    rank: int = lh.HEAD_RANK,
) -> torch.Tensor:
    """Fit linear head L with batch-indexed InfoNCE plus a log-det row-isotropy regularizer."""
    device = features.device
    pool_indices = sampler.images
    subset = features[torch.as_tensor(pool_indices, dtype=torch.long, device=device)].float()

    local_map = np.empty(len(features), dtype=np.int64)
    local_map[pool_indices] = np.arange(len(pool_indices), dtype=np.int64)

    head = lh.LinearGoalHead(lh.random_head(rank, features.shape[-1], seed)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))

    patches = subset.shape[1]
    eye_jitter = torch.eye(rank, device=device) * 1e-4
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)

    for _ in range(steps):
        draw = sampler.sample(batch_size, rng)
        optimizer.zero_grad(set_to_none=True)
        nce_loss = _batch_infonce_loss(
            subset,
            local_map,
            draw,
            head.weight,
            batch_size=batch_size,
            patches=patches,
            rank=rank,
            tau=tau,
            targets=targets,
            device=device,
        )

        gram = rank * head.weight.matmul(head.weight.transpose(0, 1)) + eye_jitter
        isotropy_loss = -torch.linalg.slogdet(gram)[1] / rank
        loss = nce_loss + isotropy_weight * isotropy_loss

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            head.weight.div_(torch.linalg.matrix_norm(head.weight).clamp_min(1e-12))

    return head.weight.detach().cpu()


def fit_fantope_head(
    features: torch.Tensor,
    sampler: lh.AllocationSampler,
    *,
    k_fantope: float = 4.0,
    steps: int = 2000,
    batch_size: int = 64,
    tau: float = 0.3,
    lr: float = 0.003,
    seed: int = 0,
    rank: int = lh.HEAD_RANK,
    project_every: int = 5,
) -> torch.Tensor:
    """Fit linear head L with batch-indexed InfoNCE and periodic k-Fantope projection."""
    if isinstance(project_every, bool) or not isinstance(project_every, int) or project_every < 1:
        raise ValueError("project_every must be a positive integer")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")

    device = features.device
    pool_indices = sampler.images
    subset = features[torch.as_tensor(pool_indices, dtype=torch.long, device=device)].float()

    local_map = np.empty(len(features), dtype=np.int64)
    local_map[pool_indices] = np.arange(len(pool_indices), dtype=np.int64)

    head = lh.LinearGoalHead(lh.random_head(rank, features.shape[-1], seed)).to(device)
    project_fantope_(head.weight, k_fantope)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))

    patches = subset.shape[1]
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)

    for step in range(1, steps + 1):
        draw = sampler.sample(batch_size, rng)
        optimizer.zero_grad(set_to_none=True)
        loss = _batch_infonce_loss(
            subset,
            local_map,
            draw,
            head.weight,
            batch_size=batch_size,
            patches=patches,
            rank=rank,
            tau=tau,
            targets=targets,
            device=device,
        )
        loss.backward()
        optimizer.step()

        if step % project_every == 0 or step == steps:
            project_fantope_(head.weight, k_fantope)
        else:
            with torch.no_grad():
                head.weight.div_(torch.linalg.matrix_norm(head.weight).clamp_min(1e-12))

    return head.weight.detach().cpu()


def _reduce_goal_prototype(goal_features: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Validate and collapse a single [P, D] goal or [K, P, D] goal bank into [1, P, D]."""
    if goal_features.ndim == 2:
        if min(goal_features.shape) == 0:
            raise ValueError("goal_features must be non-empty")
        return goal_features.to(device=device, dtype=torch.float32).unsqueeze(0)
    if goal_features.ndim == 3:
        if min(goal_features.shape) == 0:
            raise ValueError("goal_features must be non-empty")
        return goal_features.to(device=device, dtype=torch.float32).mean(dim=0, keepdim=True)
    raise ValueError("goal_features must have shape [P, D] or [K, P, D]")


@torch.inference_mode()
def rollout_head_costs(
    model,
    encoded_state,
    candidate_actions: np.ndarray,
    goal_features: torch.Tensor,
    head: torch.Tensor,
    batch_size: int = 64,
    step_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Predict rollouts for candidate_actions [N, H, A] and score them under head L.

    Supports single-photo goals ``[P, D]`` and multi-goal equivalence banks
    ``[K, P, D]`` (collapsed once to their centroid before unrolling), as well
    as optional dense-horizon ``step_weights`` of length ``H``.
    """
    goal = _reduce_goal_prototype(goal_features, model.device)
    head_device = head.to(device=model.device, dtype=torch.float32)
    feature_dim = goal.shape[-1]
    horizon = candidate_actions.shape[1]
    all_costs = []

    for start in range(0, len(candidate_actions), batch_size):
        chunk_batch = candidate_actions[start : start + batch_size]
        action_tensor = torch.from_numpy(chunk_batch).permute(1, 0, 2).to(model.device)
        predicted_visual = model.unroll(encoded_state, action_tensor)["visual"]
        if step_weights is None:
            terminal_features = predicted_visual[-1].reshape(len(chunk_batch), -1, feature_dim).float()
            cost = linear_goal_distance(terminal_features, goal, head_device)
        else:
            rollout_features = (
                predicted_visual[-horizon:]
                .reshape(horizon, len(chunk_batch), -1, feature_dim)
                .float()
            )
            cost = dense_horizon_head_distance(
                rollout_features, goal, head_device, step_weights=step_weights
            )
        all_costs.append(cost.cpu().numpy())

    return np.concatenate(all_costs)


def _dominant_mode_mean(
    elite_actions: np.ndarray,
    hold_chunk: np.ndarray,
    action_clip: float,
) -> np.ndarray:
    """Average elites aligned with the leading elite direction relative to hold_chunk."""
    lead_direction = elite_actions[0, 0] - hold_chunk
    alignment = np.sum(
        (elite_actions[:, 0] - hold_chunk[None, :]) * lead_direction[None, :],
        axis=1,
    )
    dominant_mode = elite_actions[alignment >= 0.0]
    if len(dominant_mode) <= 1:
        return elite_actions[0]
    return np.clip(dominant_mode.mean(axis=0), -action_clip, action_clip).astype(np.float32)


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
    stay_margin: float = 0.02,
    certify_barrier_split: bool = False,
    step_weights: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Jensen-certified mode-consistent CEM under an exact 2 * N rollout budget.

    Iteration 0 evaluates N candidates (anchoring slot 0 at zero-displacement
    hold and slot 1 at the shifted previous plan when available).

    When ``certify_barrier_split=False`` (default), iteration 1 evaluates N - 1
    elite-centered samples and uses rollout #2N to certify the final elite mean.
    When ``certify_barrier_split=True``, iteration 1 evaluates N - 2 samples and
    evaluates ``[elite_mean, mode_mean]`` together in a 2-candidate batch
    (rollouts #2N - 1 and #2N). Whenever Jensen's inequality holds
    (``c(elite_mean) <= mean(c_elites)``), ``elite_mean`` is always retained;
    only on a strict Jensen barrier violation does the certified split select
    ``mode_mean`` if ``c(mode_mean) <= elite_costs[0]`` and otherwise fall back
    to ``elite_actions[0]``.
    """
    _, samples_per_iter, horizon, action_dim = normals.shape
    if certify_barrier_split and samples_per_iter < 3:
        raise ValueError("certify_barrier_split requires samples_per_iter >= 3")

    goal_prototype = _reduce_goal_prototype(goal_features, model.device)[0]

    visual = torch.from_numpy(observation["visual"]).permute(2, 0, 1)[None, None]
    proprio = torch.from_numpy(np.asarray(observation["proprio"], dtype=np.float32))[None, None]
    encoded_state = model.encode({"visual": visual, "proprio": proprio})

    hold_sequence = np.tile(hold_chunk[None, :], (horizon, 1)).astype(np.float32)

    mean = np.zeros((horizon, action_dim), dtype=np.float32)
    std = np.ones_like(mean)

    iter0_actions = np.clip(mean[None] + std[None] * normals[0], -action_clip, action_clip).astype(np.float32)
    iter0_actions[0] = hold_sequence
    if previous_plan is not None and samples_per_iter > 1:
        shifted_plan = np.vstack([previous_plan[1:], hold_chunk[None, :]]).astype(np.float32)
        iter0_actions[1] = np.clip(shifted_plan, -action_clip, action_clip)

    iter0_costs = rollout_head_costs(
        model,
        encoded_state,
        iter0_actions,
        goal_prototype,
        head,
        batch_size,
        step_weights=step_weights,
    )
    hold_cost = float(iter0_costs[0])

    iter0_elites = np.argsort(iter0_costs)[:num_elites]
    mean1 = iter0_actions[iter0_elites].mean(axis=0)
    std1 = iter0_actions[iter0_elites].std(axis=0)

    cert_count = 2 if certify_barrier_split else 1
    iter1_count = samples_per_iter - cert_count
    iter1_actions = np.clip(
        mean1[None] + std1[None] * normals[1, :iter1_count],
        -action_clip,
        action_clip,
    ).astype(np.float32)
    iter1_actions[0] = mean1
    iter1_costs = rollout_head_costs(
        model,
        encoded_state,
        iter1_actions,
        goal_prototype,
        head,
        batch_size,
        step_weights=step_weights,
    )

    final_elites = np.argsort(iter1_costs)[:num_elites]
    elite_actions = iter1_actions[final_elites]
    elite_costs = iter1_costs[final_elites]
    elite_mean = np.clip(elite_actions.mean(axis=0), -action_clip, action_clip).astype(np.float32)

    mode_cost: float | None = None
    if certify_barrier_split:
        mode_mean = _dominant_mode_mean(elite_actions, hold_chunk, action_clip)
        cert_candidates = np.stack([elite_mean, mode_mean], axis=0)
        cert_costs = rollout_head_costs(
            model,
            encoded_state,
            cert_candidates,
            goal_prototype,
            head,
            batch_size,
            step_weights=step_weights,
        )
        mean_cost = float(cert_costs[0])
        mode_cost = float(cert_costs[1])
        jensen_violation = mean_cost > float(elite_costs.mean())
        if not jensen_violation:
            selected_plan = elite_mean
            selected_cost = mean_cost
        elif mode_cost <= float(elite_costs[0]):
            selected_plan = mode_mean
            selected_cost = mode_cost
        else:
            selected_plan = elite_actions[0]
            selected_cost = float(elite_costs[0])
    else:
        mean_cost = float(
            rollout_head_costs(
                model,
                encoded_state,
                elite_mean[None, :, :],
                goal_prototype,
                head,
                batch_size,
                step_weights=step_weights,
            )[0]
        )
        jensen_violation = mean_cost > float(elite_costs.mean())
        if not jensen_violation:
            selected_plan = elite_mean
            selected_cost = mean_cost
        else:
            selected_plan = _dominant_mode_mean(elite_actions, hold_chunk, action_clip)
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
        "selected_cost": selected_cost,
        "jensen_violation": bool(jensen_violation),
        "used_stay_gate": used_stay_gate,
    }
    if mode_cost is not None:
        diagnostics["mode_mean_cost"] = mode_cost
    return selected_plan[0], selected_plan, diagnostics
