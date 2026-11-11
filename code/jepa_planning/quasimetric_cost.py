"""Quasimetric goal cost: 

Labels should guarantee the fewest model steps from a candidate to a goal.

Temporal cost was basically elapsed gap between two frames of one recorded
episode. That gap is only an upper bound on the steps neededr, and it never relates
frames from different episodes. Planning compares branches from one start, so the cost
has to rank how close each branch can get, not how long a recording happened to take.

This cost follows quasimetric RL (Wang et al., 2023). It is trained with two rules:

* every recorded transition costs at most one step, ``d(s, s') <= 1``;
* subject to that, distances between random pairs of frames are pushed up.

The distance is a quasimetric by construction (zero to itself, never negative,
triangle inequality, not necessarily symmetric), so any chain of recorded steps
from ``s`` to ``g``, even one stitched across episodes, bounds ``d(s, g)`` by its
length. Pushing every distance up to that bound leaves the shortest chain:
the minimum number of steps the data shows is possible.

The candidate and goal share one encoder since the triangle inequality needs both in
the same space. The distance counts the steps to reach the goal photo's
state, including agent position. The old cost used separate projections
to discount the goal's agent. Here, that costs a few steps at most and it can be
checked on the probe's photo agreement.

Features use the flattened [N, grid * grid * feature_dim] layout returned by
``pusht_multi_goal.encode_observations``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional


def interval_quasimetric(starts: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
    """Return the Interval Quasimetric Embedding distance per component.

    ``starts`` and ``goals`` are [..., components, dim]. For each component, every
    coordinate where the start lies below the goal contributes the interval
    [start, goal]; the distance is the total length of the union of those
    intervals. It is zero when the start is at or above the goal in every
    coordinate. Returns [..., components].
    """
    starts, goals = torch.broadcast_tensors(starts, goals)
    open_ = (starts < goals).to(starts.dtype)
    points = torch.cat((starts, goals), dim=-1)
    # +1 where an interval opens, -1 where it closes; empty intervals add nothing.
    steps = torch.cat((open_, -open_), dim=-1)
    points, order = points.sort(dim=-1)
    covered = steps.gather(-1, order).cumsum(dim=-1)[..., :-1] > 0.5
    return ((points[..., 1:] - points[..., :-1]) * covered).sum(dim=-1)


class QuasimetricCost(nn.Module):
    """Predict the fewest model steps from candidate features to goal features."""

    def __init__(
        self,
        feature_dim: int = 384,
        grid: int = 16,
        width: int = 64,
        hidden: int = 256,
        components: int = 32,
        component_dim: int = 8,
    ):
        super().__init__()
        if grid % 4:
            raise ValueError("grid must be divisible by four")
        self.feature_dim = feature_dim
        self.grid = grid
        self.components = components
        self.component_dim = component_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(feature_dim, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(width * (grid // 4) ** 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, components * component_dim),
        )
        # Components are combined as a learned mix of their maximum and mean.
        # Both keep the triangle inequality, so any mix of them does too.
        self.max_weight_logit = nn.Parameter(torch.zeros(()))

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        """Map flattened features to [N, components, component_dim] latents."""
        maps = features.float().reshape(-1, self.grid, self.grid, self.feature_dim)
        maps = functional.layer_norm(maps, (self.feature_dim,)).permute(0, 3, 1, 2)
        return self.encoder(maps).reshape(-1, self.components, self.component_dim)

    def distance(self, starts: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        """Return [N] distances between latents from :meth:`embed`."""
        per_component = interval_quasimetric(starts, goals)
        max_weight = torch.sigmoid(self.max_weight_logit)
        return max_weight * per_component.amax(dim=-1) + (1 - max_weight) * per_component.mean(dim=-1)

    def forward(self, candidates: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        """Return [N] predicted steps; ``goals`` holds one goal or one per candidate."""
        if len(goals) not in (1, len(candidates)):
            raise ValueError("goal batch must contain one goal or one goal per candidate")
        return self.distance(self.embed(candidates), self.embed(goals))


class QuasimetricObjective(nn.Module):
    """The constrained QRL objective, with a learned Lagrange multiplier.

    Minimizing the returned loss pushes distances between random frames up
    toward ``horizon`` (beyond it the push fades, so unreachable pairs stay
    finite), while the multiplier grows whenever recorded transitions cost more
    than one step on average, by a squared excess above ``slack ** 2``.
    Train the multiplier with the cost, in the same optimizer.
    """

    def __init__(self, horizon: float, slack: float = 0.25):
        super().__init__()
        if not horizon > 1 or not slack > 0:
            raise ValueError("horizon must exceed one step and slack must be positive")
        self.horizon = float(horizon)
        self.slack = float(slack)
        self.raw_multiplier = nn.Parameter(torch.zeros(()))

    @property
    def multiplier(self) -> torch.Tensor:
        return functional.softplus(self.raw_multiplier)

    def forward(
        self,
        cost: QuasimetricCost,
        frames: torch.Tensor,
        next_frames: torch.Tensor,
        sources: torch.Tensor,
        goals: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the loss and its parts for one batch of features.

        ``frames``/``next_frames`` are recorded transitions; ``sources``/``goals``
        are random frame pairs to push apart.
        """
        step = cost(frames, next_frames)
        violation = functional.relu(step - 1).square().mean()
        spread = cost(sources, goals)
        # Roughly horizon - d for close pairs; flattens out past the horizon.
        push = functional.softplus(self.horizon - spread, beta=10 / self.horizon).mean()
        multiplier = self.multiplier
        # The cost sees the multiplier as a constant penalty weight; the multiplier
        # sees the constraint with the sign flipped, so descent raises it while
        # the constraint is violated (gradient ascent on the Lagrangian).
        loss = push + multiplier.detach() * violation - multiplier * (violation.detach() - self.slack**2)
        stats = {
            "push": push.item(),
            "violation": violation.item(),
            "multiplier": multiplier.item(),
            "step_distance": step.mean().item(),
            "pair_distance": spread.mean().item(),
        }
        return loss, stats


def sample_transitions(
    frame_counts: np.ndarray, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample recorded transitions: global frame indices and the frame after each.

    Frames are indexed globally, with episodes stored contiguously, as in
    ``temporal_cost.sample_pairs``. Single-frame episodes have no transitions.
    """
    counts = np.asarray(frame_counts)
    if count < 1:
        raise ValueError("count must be positive")
    usable = np.flatnonzero(counts >= 2)
    if not len(usable):
        raise ValueError("need an episode with at least two frames")
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    # Weight episodes by their transitions so every transition is equally likely.
    weights = (counts[usable] - 1) / (counts[usable] - 1).sum()
    episodes = rng.choice(usable, size=count, p=weights)
    frames = offsets[episodes] + rng.integers(0, counts[episodes] - 1)
    return frames, frames + 1


def sample_frame_pairs(
    frame_counts: np.ndarray, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample independent, uniformly random (source, goal) frames to push apart."""
    total = int(np.sum(frame_counts))
    if total < 1 or count < 1:
        raise ValueError("need frames and a positive count")
    return rng.integers(total, size=count), rng.integers(total, size=count)
