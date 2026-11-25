"""Unified calibrated planning objectives for frozen DINO-WM features.

This module connects four complementary planning-objective extensions developed
on PushT:

1. Multi-goal prototypes under a learned linear head ``L``:
   :func:`projected_multi_goal_distance` exploits the exact bias-variance
   identity ``mean_k D_L(z, g_k) = D_L(z, g_bar) + mean_k D_L(g_k, g_bar)`` so
   candidate scoring stays ``O(N)`` regardless of the goal-bank size ``K``.
2. IIA-preserving multi-objective blending:
   :func:`anchor_calibrated_blend` scales each cost by a deterministic anchor
   candidate (such as the zero-displacement hold trajectory) instead of batch
   z-scoring, eliminating decoy-induced rank reversals.
3. Dense-horizon rollout scoring:
   :func:`dense_horizon_head_distance` evaluates full ``[H, N, P, D]`` world-model
   rollouts with arbitrary nonnegative horizon weights.
4. Spatial patch attribution:
   :func:`patch_energy_map` decomposes plain MSE or projected head distance into
   per-patch squared energy maps ``[..., G, G]`` (or ``[..., P]``).
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import torch

from jepa_planning.goal_costs import linear_goal_distance


def _normalize_goal_features(
    goal_features: torch.Tensor,
    *,
    expected_trailing: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Promote a single ``[*trailing]`` goal to ``[1, *trailing]`` and validate."""
    if goal_features.ndim == len(expected_trailing):
        goals = goal_features.unsqueeze(0)
    elif goal_features.ndim == len(expected_trailing) + 1:
        goals = goal_features
    else:
        raise ValueError(
            "goal_features must have shape [K, *trailing] or [*trailing] matching candidate_features"
        )
    if goals.shape[0] == 0:
        raise ValueError("goal_features batch must not be empty")
    if tuple(goals.shape[1:]) != expected_trailing:
        raise ValueError("goal_features trailing dimensions must match candidate_features")
    if goals.device != device:
        raise ValueError("candidate and goal features must be on the same device")
    if goals.dtype != dtype:
        raise ValueError("candidate and goal features must have the same dtype")
    return goals


def _validate_head(
    head: torch.Tensor,
    *,
    feature_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Validate that a linear head matrix matches feature dimension, device, and dtype."""
    if head.ndim != 2 or head.shape[0] == 0 or head.shape[1] != feature_dim:
        raise ValueError("head must have shape [r, D] with positive rank matching feature dimension")
    if head.device != device or head.dtype != dtype:
        raise ValueError("head must match feature device and dtype")


def projected_multi_goal_distance(
    candidate_features: torch.Tensor,
    goal_features: torch.Tensor,
    head: torch.Tensor,
    exact_mean: bool = False,
) -> torch.Tensor:
    """Return multi-goal projected distance under linear head ``L`` in ``O(N)``.

    Args:
        candidate_features: Candidate features of shape ``[N, P, D]`` (or
            ``[N, ..., D]`` with at least 3 dimensions).
        goal_features: Equivalent goal features of shape ``[K, P, D]``, or a
            single goal photo of shape ``[P, D]`` (treated as ``K = 1``).
        head: Linear projection matrix ``L`` of shape ``[r, D]``.
        exact_mean: If ``False`` (default), returns ``D_L(z, g_bar)``, which
            differs from ``mean_k D_L(z, g_k)`` by a candidate-independent
            constant and yields identical candidate rankings. If ``True``, adds
            the exact within-bank variance ``mean_k D_L(g_k, g_bar)`` so the
            result equals ``mean_k D_L(z, g_k)`` to floating-point precision.

    Returns:
        A tensor of shape ``[N]`` on the same device and dtype as the inputs.
    """
    if candidate_features.ndim < 3:
        raise ValueError("candidate_features must have shape [N, P, D] (at least 3 dimensions)")
    if candidate_features.shape[0] == 0 or any(dim == 0 for dim in candidate_features.shape[1:]):
        raise ValueError("candidate_features must be non-empty")

    goals = _normalize_goal_features(
        goal_features,
        expected_trailing=tuple(candidate_features.shape[1:]),
        device=candidate_features.device,
        dtype=candidate_features.dtype,
    )
    _validate_head(
        head,
        feature_dim=candidate_features.shape[-1],
        device=candidate_features.device,
        dtype=candidate_features.dtype,
    )

    goal_bar = goals.mean(dim=0, keepdim=True)
    centroid_distance = linear_goal_distance(candidate_features, goal_bar, head)
    if not exact_mean or goals.shape[0] == 1:
        return centroid_distance

    # Because L is linear, the cross term sum_k <L(z - g_bar), L(g_bar - g_k)> vanishes.
    goal_variance = linear_goal_distance(goals, goal_bar, head).mean()
    return centroid_distance + goal_variance


def anchor_calibrated_blend(
    primary_costs: np.ndarray,
    secondary_costs: np.ndarray,
    weight: float,
    anchor_index: int = 0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Blend two cost arrays using deterministic anchor scaling instead of batch z-scores.

    Each cost array is divided by ``max(|costs[anchor_index]|, eps)`` before
    convex combination:

        (1 - weight) * (primary / p_anchor) + weight * (secondary / s_anchor)

    Because the normalizing scale depends only on the fixed anchor candidate
    (e.g., the zero-displacement hold action at index 0), adding or removing
    other candidates never alters the scores or relative ranking of existing
    candidates (strict independence of irrelevant alternatives).

    Args:
        primary_costs: Array of shape ``[N]`` or ``[N, G]``.
        secondary_costs: Array with the exact same shape as ``primary_costs``.
        weight: Convex combination weight in ``[0, 1]`` on ``secondary_costs``.
        anchor_index: Index of the deterministic anchor candidate in ``[0, N)``.
        eps: Positive lower clamp on anchor magnitudes to avoid division by zero.

    Returns:
        Float64 array of blended costs with the same shape as the inputs.
    """
    primary = np.asarray(primary_costs, dtype=np.float64)
    secondary = np.asarray(secondary_costs, dtype=np.float64)

    if primary.shape != secondary.shape:
        raise ValueError("primary_costs and secondary_costs must have the same shape")
    if primary.ndim not in (1, 2):
        raise ValueError("cost arrays must be 1D [N] or 2D [N, G]")
    if primary.size == 0:
        raise ValueError("cost arrays must not be empty")
    if isinstance(weight, bool) or not np.isfinite(weight) or not 0.0 <= float(weight) <= 1.0:
        raise ValueError("weight must be a finite number in [0, 1]")
    if isinstance(anchor_index, bool) or not isinstance(anchor_index, (int, np.integer)):
        raise ValueError("anchor_index must be an integer")
    anchor = int(anchor_index)
    if not 0 <= anchor < primary.shape[0]:
        raise ValueError(f"anchor_index {anchor} is out of range for {primary.shape[0]} candidates")
    if isinstance(eps, bool) or not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite number")

    p_anchor = np.maximum(np.abs(primary[anchor]), eps)
    s_anchor = np.maximum(np.abs(secondary[anchor]), eps)
    w = float(weight)
    return (1.0 - w) * (primary / p_anchor) + w * (secondary / s_anchor)


def dense_horizon_head_distance(
    rollout_visuals: torch.Tensor,
    goal_features: torch.Tensor,
    head: torch.Tensor,
    step_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Score full rollout trajectories ``[H, N, P, D]`` under linear head ``L``.

    Args:
        rollout_visuals: Predicted visual patch features with shape ``[H, N, P, D]``.
        goal_features: Target features of shape ``[P, D]`` or ``[K, P, D]``
            (collapsed via centroid).
        head: Linear projection matrix ``L`` of shape ``[r, D]``.
        step_weights: Optional nonnegative weights of length ``H`` (normalized
            to sum to 1). When ``None``, only the terminal step ``H - 1`` is
            scored (equivalent to ``[0, ..., 0, 1.0]``).

    Returns:
        A tensor of shape ``[N]`` containing the horizon-weighted costs.
    """
    if rollout_visuals.ndim != 4:
        raise ValueError("rollout_visuals must have shape [H, N, P, D]")
    horizon, num_candidates, num_patches, feature_dim = rollout_visuals.shape
    if min(horizon, num_candidates, num_patches, feature_dim) == 0:
        raise ValueError("rollout_visuals dimensions must all be positive")

    goals = _normalize_goal_features(
        goal_features,
        expected_trailing=(num_patches, feature_dim),
        device=rollout_visuals.device,
        dtype=rollout_visuals.dtype,
    )
    _validate_head(
        head,
        feature_dim=feature_dim,
        device=rollout_visuals.device,
        dtype=rollout_visuals.dtype,
    )

    if step_weights is None:
        return projected_multi_goal_distance(
            rollout_visuals[-1], goals, head, exact_mean=False
        )

    weight_list = list(step_weights)
    if len(weight_list) != horizon:
        raise ValueError(f"step_weights must have length {horizon}, got {len(weight_list)}")
    weights = torch.as_tensor(
        weight_list, device=rollout_visuals.device, dtype=rollout_visuals.dtype
    )
    if weights.ndim != 1 or not torch.isfinite(weights).all():
        raise ValueError("step_weights must contain finite 1D numeric values")
    if bool((weights < 0).any()):
        raise ValueError("step_weights must be nonnegative")
    weight_sum = weights.sum()
    if float(weight_sum) <= 0.0:
        raise ValueError("step_weights must have a positive sum")
    weights = weights / weight_sum

    goal_bar = goals.mean(dim=0, keepdim=True)
    flat_visuals = rollout_visuals.reshape(horizon * num_candidates, num_patches, feature_dim)
    step_costs = linear_goal_distance(flat_visuals, goal_bar, head).reshape(
        horizon, num_candidates
    )
    return (weights[:, None] * step_costs).sum(dim=0)


def patch_energy_map(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    head: torch.Tensor | None = None,
    grid_size: int | None = None,
) -> torch.Tensor:
    """Compute per-patch squared energy for spatial interpretability heatmaps.

    Args:
        features_a: Patch features with shape ``[..., P, D]``.
        features_b: Patch features broadcastable with ``features_a`` and matching
            trailing ``[P, D]`` dimensions.
        head: Optional linear head ``L`` of shape ``[r, D]``. When ``None``,
            computes plain per-channel MSE ``(a - b).square().mean(dim=-1)``.
            When provided, computes ``((a - b) @ L.T).square().sum(dim=-1)``.
        grid_size: Optional spatial side length ``G`` with ``G * G == P``. When
            ``None`` and ``P`` is a perfect square, reshapes the patch axis to
            ``[..., isqrt(P), isqrt(P)]``; otherwise preserves ``[..., P]``.

    Returns:
        Per-patch squared energy tensor of shape ``[..., G, G]`` or ``[..., P]``.
    """
    if features_a.ndim < 2 or features_b.ndim < 2:
        raise ValueError("features_a and features_b must have shape [..., P, D]")
    if features_a.device != features_b.device:
        raise ValueError("features_a and features_b must be on the same device")
    if features_a.dtype != features_b.dtype:
        raise ValueError("features_a and features_b must have the same dtype")
    if features_a.shape[-2:] != features_b.shape[-2:]:
        raise ValueError("features_a and features_b must share trailing [P, D] dimensions")
    num_patches, feature_dim = features_a.shape[-2], features_a.shape[-1]
    if num_patches == 0 or feature_dim == 0:
        raise ValueError("patch and feature dimensions must be positive")

    try:
        a_broadcast, b_broadcast = torch.broadcast_tensors(features_a, features_b)
    except RuntimeError as exc:
        raise ValueError("features_a and features_b have incompatible broadcast shapes") from exc

    diff = a_broadcast - b_broadcast
    if head is None:
        energy = diff.square().mean(dim=-1)
    else:
        _validate_head(
            head,
            feature_dim=feature_dim,
            device=features_a.device,
            dtype=features_a.dtype,
        )
        energy = diff.matmul(head.transpose(0, 1)).square().sum(dim=-1)

    if grid_size is not None:
        if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size <= 0:
            raise ValueError("grid_size must be a positive integer")
        if grid_size * grid_size != num_patches:
            raise ValueError(
                f"grid_size={grid_size} is incompatible with {num_patches} patches"
            )
        return energy.reshape(*energy.shape[:-1], grid_size, grid_size)

    root = math.isqrt(num_patches)
    if root * root == num_patches:
        return energy.reshape(*energy.shape[:-1], root, root)
    return energy
