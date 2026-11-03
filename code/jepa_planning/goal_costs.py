"""Feature-space goal costs used by JEPA-Planning experiments.

The functions in this module operate on batches whose first dimension indexes
candidate endpoints. For example, call
``mean_squared_goal_distance(candidates, goal)`` with candidate features of
shape ``[N, ...]`` and either one broadcast goal of shape ``[1, ...]`` or one
goal per candidate of shape ``[N, ...]``. The result is one scalar cost per
candidate; lower values indicate greater feature similarity to the goal.
"""

from __future__ import annotations

import torch


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
    if candidate_features.shape[1:] != goal_features.shape[1:]:
        raise ValueError("candidate and goal feature dimensions must match")
    if goal_features.shape[0] not in (1, candidate_features.shape[0]):
        raise ValueError("goal batch must contain one goal or one goal per candidate")

    squared_error = (candidate_features - goal_features).square()
    return squared_error.flatten(start_dim=1).mean(dim=1)
