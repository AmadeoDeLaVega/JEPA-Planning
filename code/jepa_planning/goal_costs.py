"""Feature-space goal costs used by JEPA-Planning experiments.

The functions in this module operate on batches whose first dimension indexes
candidate endpoints. For example, call
``mean_squared_goal_distance(candidates, goal)`` with candidate features of
shape ``[N, ...]`` and either one broadcast goal of shape ``[1, ...]`` or one
goal per candidate of shape ``[N, ...]``. For several equivalent goal photos,
``mean_multi_goal_distance(candidates, goals)`` averages the squared distance
over the goal set without constructing an ``[N, K, ...]`` temporary tensor.
Both functions return one scalar cost per candidate; lower is better.
"""

from __future__ import annotations

import torch


def _validate_feature_shapes(
    candidate_features: torch.Tensor,
    goal_features: torch.Tensor,
) -> None:
    """Validate dimensions shared by the single- and multi-goal costs."""
    if candidate_features.ndim < 2 or goal_features.ndim < 2:
        raise ValueError(
            "candidate and goal features must include batch and feature dimensions"
        )
    if candidate_features.ndim != goal_features.ndim:
        raise ValueError(
            "candidate and goal features must have the same number of dimensions"
        )
    if candidate_features.shape[0] == 0:
        raise ValueError("candidate feature batch must not be empty")
    if goal_features.shape[0] == 0:
        raise ValueError("goal feature batch must not be empty")
    if candidate_features.shape[1:] != goal_features.shape[1:]:
        raise ValueError("candidate and goal feature dimensions must match")
    if candidate_features.device != goal_features.device:
        raise ValueError("candidate and goal features must be on the same device")
    if candidate_features.dtype != goal_features.dtype:
        raise ValueError("candidate and goal features must have the same dtype")


def mean_squared_goal_distance(
    candidate_features: torch.Tensor,
    goal_features: torch.Tensor,
) -> torch.Tensor:
    """Return the mean squared feature distance from each candidate to its goal.

    Args:
        candidate_features: Candidate endpoint features with shape ``[N, ...]``.
        goal_features: Features with the same trailing dimensions and a leading
            dimension of either ``1`` (one shared goal) or ``N`` (paired goals).

    Returns:
        A tensor of shape ``[N]`` on the same device as the inputs.

    Raises:
        ValueError: If the tensors do not have compatible batch or feature
            dimensions, or if the candidate batch is empty.
    """
    _validate_feature_shapes(candidate_features, goal_features)
    if goal_features.shape[0] not in (1, candidate_features.shape[0]):
        raise ValueError("goal batch must contain one goal or one goal per candidate")

    squared_error = (candidate_features - goal_features).square()
    return squared_error.flatten(start_dim=1).mean(dim=1)


def mean_multi_goal_distance(
    candidate_features: torch.Tensor,
    goal_features: torch.Tensor,
) -> torch.Tensor:
    """Average feature-space MSE from each candidate to equivalent goals.

    Args:
        candidate_features: Candidate endpoint features with shape ``[N, ...]``.
        goal_features: A non-empty set of equivalent goal features with shape
            ``[K, ...]``. Goals should depict the same task-relevant target
            while varying nuisance content such as the agent position.

    Returns:
        A tensor of shape ``[N]`` containing the mean over all ``K`` goals and
        all non-batch feature dimensions.

    Notes:
        The implementation uses
        ``E_g ||x-g||^2 = ||x||^2 - 2 x E[g] + E[||g||^2]``. This is exactly
        the brute-force mean but avoids materializing all candidate-goal pairs.
        The final clamp only removes tiny negative values caused by floating-
        point roundoff.
    """
    return pairwise_squared_goal_distance(candidate_features, goal_features).mean(dim=1)


def pairwise_squared_goal_distance(
    candidate_features: torch.Tensor,
    goal_features: torch.Tensor,
) -> torch.Tensor:
    """Return feature MSE for every candidate-goal pair as an ``[N, K]`` tensor.

    This matrix is useful when an experiment evaluates many subsets of the
    same goal set: compute each pair once, then average the selected columns.
    It uses the same algebraic expansion as :func:`mean_multi_goal_distance`.
    """
    _validate_feature_shapes(candidate_features, goal_features)
    candidates = candidate_features.flatten(start_dim=1)
    goals = goal_features.flatten(start_dim=1)
    feature_count = candidates.shape[1]

    candidate_norm = candidates.square().mean(dim=1, keepdim=True)
    goal_norm = goals.square().mean(dim=1).unsqueeze(0)
    cross_term = candidates.matmul(goals.transpose(0, 1)) / feature_count
    return (candidate_norm - 2 * cross_term + goal_norm).clamp_min(0)


def linear_goal_distance(
    candidate_features: torch.Tensor,
    goal_features: torch.Tensor,
    head: torch.Tensor,
) -> torch.Tensor:
    """Return the learned-head goal cost ``mean_i ||L (z_i - g_i)||^2`` (Eq. 3).

    Args:
        candidate_features: Candidate endpoint features ``[N, ..., D]``.
        goal_features: Goal features with leading dimension ``1`` or ``N`` and
            the same trailing dimensions.
        head: The head matrix ``L`` with shape ``[r, D]``. It is shared by all
            patches, so it acts only on the last (channel) dimension.

    Returns:
        A tensor of shape ``[N]``: squared projected distance summed over the
        ``r`` head outputs and averaged over every other non-batch dimension
        (the patches). With ``L = I_D / sqrt(D)`` this equals
        :func:`mean_squared_goal_distance`.
    """
    _validate_feature_shapes(candidate_features, goal_features)
    if goal_features.shape[0] not in (1, candidate_features.shape[0]):
        raise ValueError("goal batch must contain one goal or one goal per candidate")
    if head.ndim != 2 or head.shape[1] != candidate_features.shape[-1]:
        raise ValueError("head must have shape [r, D] matching the feature dimension")
    if head.device != candidate_features.device or head.dtype != candidate_features.dtype:
        raise ValueError("head must match the feature device and dtype")

    # [N, ..., D] @ [D, r] applies L to every patch; then sum over the r outputs
    # and average over all patch dimensions.
    projected = (candidate_features - goal_features).matmul(head.transpose(0, 1))
    return projected.square().sum(dim=-1).flatten(start_dim=1).mean(dim=1)
