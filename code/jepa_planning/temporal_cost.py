"""Learned temporal goal cost for frozen DINO-WM features.

Feature MSE asks how similar an endpoint looks to a goal photo. This cost asks
how many model steps separate them, learned from the frame gaps inside recorded
trajectories. It needs no equivalence groups and no object coordinates.

The candidate and goal pass through separate projections because they play
different roles: the candidate agent can matter for reaching the goal, while the
agent shown in a goal photo should not redefine the target. A shared embedding
distance would have to treat both sides identically.

Features use the flattened [N, grid * grid * feature_dim] layout returned by
``pusht_multi_goal.encode_observations``.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
from torch import nn


class TemporalCost(nn.Module):
    """Predict the model steps from candidate features to goal features."""

    def __init__(self, feature_dim: int = 384, grid: int = 16, width: int = 64, hidden: int = 256):
        super().__init__()
        if grid % 4:
            raise ValueError("grid must be divisible by four")
        self.feature_dim = feature_dim
        self.grid = grid
        self.candidate_projection = nn.Conv2d(feature_dim, width, kernel_size=1)
        self.goal_projection = nn.Conv2d(feature_dim, width, kernel_size=1)
        channels = 2 * width
        self.trunk = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(channels * (grid // 4) ** 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )

    def _as_grid(self, features: torch.Tensor) -> torch.Tensor:
        """Convert flattened features to normalized [N, D, grid, grid] maps."""
        maps = features.float().reshape(-1, self.grid, self.grid, self.feature_dim)
        maps = nn.functional.layer_norm(maps, (self.feature_dim,))
        return maps.permute(0, 3, 1, 2)

    def forward(self, candidates: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        """Return [N] predicted steps; ``goals`` holds one goal or one per candidate."""
        if len(goals) not in (1, len(candidates)):
            raise ValueError("goal batch must contain one goal or one goal per candidate")
        candidate_maps = self.candidate_projection(self._as_grid(candidates))
        goal_maps = self.goal_projection(self._as_grid(goals)).expand_as(candidate_maps)
        return self.trunk(torch.cat((candidate_maps, goal_maps), dim=1)).squeeze(1)


def sample_pairs(
    frame_counts: np.ndarray,
    count: int,
    max_gap: int,
    far_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample start frames, goal frames, and step labels from cached episodes.

    Frames are indexed globally, with episodes stored contiguously. Same-episode
    pairs are labelled with their gap. A ``far_fraction`` of pairs take the goal
    from another episode and receive ``max_gap``, the label for "far away".
    """
    counts = np.asarray(frame_counts)
    if len(counts) < 2 or counts.min() < 1:
        raise ValueError("need at least two non-empty episodes")
    if count < 1 or max_gap < 1 or not 0 <= far_fraction <= 1:
        raise ValueError("count and max_gap must be positive; far_fraction must be in [0, 1]")
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))

    episodes = rng.integers(len(counts), size=count)
    lengths = counts[episodes]
    gaps = rng.integers(0, np.minimum(lengths - 1, max_gap) + 1)
    starts = rng.integers(0, lengths - gaps)
    start_frames = offsets[episodes] + starts
    goal_frames = start_frames + gaps

    far = rng.random(count) < far_fraction
    others = (episodes + rng.integers(1, len(counts), size=count)) % len(counts)
    far_frames = offsets[others] + rng.integers(0, counts[others])
    goal_frames = np.where(far, far_frames, goal_frames)
    labels = np.where(far, max_gap, gaps).astype(np.float32)
    return start_frames, goal_frames, labels


def one_step_closer(
    start_frames: np.ndarray,
    goal_frames: np.ndarray,
    frame_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find same-episode pairs whose goal lies ahead, and the frame one step closer.

    Returns the indices of those pairs and, for each, the frame after its start.
    A good cost scores that frame lower than the start, since it is one step
    nearer the same goal.
    """
    ends = np.cumsum(frame_counts)
    same = np.searchsorted(ends, start_frames, side="right") == np.searchsorted(ends, goal_frames, side="right")
    ordered = np.flatnonzero(same & (goal_frames > start_frames))
    return ordered, start_frames[ordered] + 1


def photo_agreement(costs: np.ndarray) -> float:
    """Return how often two goal photos select the same candidate from [N, K] costs."""
    selected = np.asarray(costs).argmin(axis=0)
    if len(selected) < 2:
        raise ValueError("photo agreement needs at least two goal photos")
    return float(np.mean([first == second for first, second in combinations(selected, 2)]))


def target_response(costs: np.ndarray, other_target_costs: np.ndarray) -> float:
    """Return how often changing the object target changes the selected candidate."""
    first, second = np.asarray(costs), np.asarray(other_target_costs)
    if first.shape != second.shape:
        raise ValueError("cost matrices must have the same shape")
    return float(np.mean(first.argmin(axis=0) != second.argmin(axis=0)))
