"""Reusable PushT utilities for equivalent-goal planning experiments.

The multi-goal experiments compare a conventional cost to one goal image with
the mean feature-space cost to several images that share the same object target
but vary the agent pose. This module contains the experiment-independent parts:
state sampling, physical metrics, fair goal-subset enumeration, candidate-bank
evaluation, observation encoding, and common CEM random numbers.

Simulator orchestration and output writing stay in ``code/experiments``. This
separation lets the numerical protocol be unit tested without loading a DINO-WM
checkpoint or constructing a PushT environment.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr
import torch

from jepa_planning.goal_costs import pairwise_squared_goal_distance


def angle_error(first: np.ndarray, second: np.ndarray | float) -> np.ndarray:
    """Return the absolute wrapped angular difference in radians."""
    return np.abs((first - second + np.pi) % (2 * np.pi) - np.pi)


def physical_errors(
    candidate_poses: np.ndarray,
    goal_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measure PushT object error independently of the agent position.

    Poses contain object ``(x, y, angle)``. The combined metric maps one radian
    to 50 pixels, matching the protocol used for the Local confirmation runs.
    Success requires position error below 20 pixels and angle error below 20°.
    """
    candidates = np.asarray(candidate_poses)
    goal = np.asarray(goal_pose)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate poses must have shape [N, 3]")
    if len(candidates) == 0:
        raise ValueError("candidate pose batch must not be empty")
    if goal.shape != (3,):
        raise ValueError("goal pose must have shape [3]")
    position = np.linalg.norm(candidates[:, :2] - goal[None, :2], axis=1)
    angle = angle_error(candidates[:, 2], goal[2])
    combined = np.sqrt(np.square(position) + np.square(50.0 * angle))
    success = (position < 20.0) & (angle < np.pi / 9.0)
    return position, angle, combined, success


def state_from(agent_pose: np.ndarray, object_pose: np.ndarray) -> np.ndarray:
    """Build the seven-value state expected by the vendored PushT simulator."""
    agent = np.asarray(agent_pose)
    obj = np.asarray(object_pose)
    if agent.shape != (2,) or obj.shape != (3,):
        raise ValueError("agent and object poses must have shapes [2] and [3]")
    return np.asarray([agent[0], agent[1], *obj, 0.0, 0.0], dtype=np.float32)


def sample_initial_state(rng: np.random.Generator) -> np.ndarray:
    """Sample a valid initial state with the agent near the PushT object."""
    obj = np.asarray(
        [rng.uniform(180.0, 332.0), rng.uniform(180.0, 332.0), rng.uniform(0, 2 * np.pi)]
    )
    direction = rng.uniform(0, 2 * np.pi)
    agent = obj[:2] + 65.0 * np.asarray([np.cos(direction), np.sin(direction)])
    return state_from(np.clip(agent, 35.0, 477.0), obj)


def sample_action_chunks(
    rng: np.random.Generator,
    count: int,
    horizon: int,
    controls_per_chunk: int,
    action_clip: float = 2.0,
) -> np.ndarray:
    """Sample smooth normalized action chunks with shape ``[N, H, C, 2]``."""
    if min(count, horizon, controls_per_chunk) < 1:
        raise ValueError("count, horizon, and controls_per_chunk must be positive")
    if not np.isfinite(action_clip) or action_clip <= 0:
        raise ValueError("action_clip must be finite and positive")
    base = rng.normal(size=(count, horizon, 1, 2))
    noise = rng.normal(scale=0.35, size=(count, horizon, controls_per_chunk, 2))
    return np.clip(base + noise, -action_clip, action_clip).astype(np.float32)


def goal_subsets(total_goals: int, goal_counts: Sequence[int]) -> list[tuple[int, ...]]:
    """Enumerate every requested subset, preventing favorable goal ordering."""
    if total_goals < 1:
        raise ValueError("total_goals must be positive")
    counts = tuple(goal_counts)
    if not counts or any(type(count) is not int or not 1 <= count <= total_goals for count in counts):
        raise ValueError("goal counts must be integers between one and total_goals")
    if len(set(counts)) != len(counts):
        raise ValueError("goal counts must not contain duplicates")
    return [subset for count in counts for subset in combinations(range(total_goals), count)]


def selection_record(
    costs: np.ndarray,
    candidate_poses: np.ndarray,
    goal_pose: np.ndarray,
) -> dict[str, float | int]:
    """Select the minimum-cost candidate and report object-space performance."""
    scores = np.asarray(costs)
    if scores.ndim != 1 or len(scores) != len(candidate_poses):
        raise ValueError("costs must contain one value per candidate pose")
    if not np.all(np.isfinite(scores)):
        raise ValueError("candidate costs must be finite")
    position, angle, combined, success = physical_errors(candidate_poses, goal_pose)
    selected = int(np.argmin(scores))
    correlation = (
        0.0
        if np.ptp(scores) == 0 or np.ptp(combined) == 0
        else spearmanr(scores, combined).statistic
    )
    return {
        "selected": selected,
        "position_error": float(position[selected]),
        "angle_error": float(angle[selected]),
        "combined_error": float(combined[selected]),
        "regret": float(combined[selected] - combined.min()),
        "success": int(success[selected]),
        "rank_correlation": float(correlation) if np.isfinite(correlation) else 0.0,
        "oracle_success_available": int(success.any()),
    }


def evaluate_goal_counts(
    candidate_features: torch.Tensor,
    goal_features: torch.Tensor,
    candidate_poses: np.ndarray,
    goal_pose: np.ndarray,
    goal_counts: Sequence[int],
) -> list[dict[str, object]]:
    """Evaluate every subset for each goal count on one shared candidate bank."""
    if len(candidate_features) != len(candidate_poses):
        raise ValueError("candidate features and poses must have the same length")
    pairwise = pairwise_squared_goal_distance(candidate_features, goal_features)
    records: list[dict[str, object]] = []
    for subset in goal_subsets(len(goal_features), goal_counts):
        costs = pairwise[:, subset].mean(dim=1).detach().cpu().numpy()
        records.append(
            {
                "goal_count": len(subset),
                "goal_subset": ";".join(str(index) for index in subset),
                **selection_record(costs, candidate_poses, goal_pose),
            }
        )
    return records


def batches(length: int, batch_size: int) -> Iterable[slice]:
    """Yield non-empty contiguous slices covering ``range(length)``."""
    if length < 0 or batch_size < 1:
        raise ValueError("length must be nonnegative and batch_size must be positive")
    for start in range(0, length, batch_size):
        yield slice(start, min(start + batch_size, length))


@torch.inference_mode()
def encode_observations(
    model,
    frames: np.ndarray,
    proprioception: np.ndarray,
    batch_size: int,
) -> torch.Tensor:
    """Encode RGB/proprio observations and return flattened CPU features."""
    if len(frames) != len(proprioception) or len(frames) == 0:
        raise ValueError("frames and proprioception must be non-empty and aligned")
    output = []
    for selection in batches(len(frames), batch_size):
        visual = torch.from_numpy(frames[selection]).permute(0, 3, 1, 2)[:, None]
        proprio = torch.from_numpy(proprioception[selection])[:, None]
        encoded = model.encode({"visual": visual, "proprio": proprio})["visual"]
        output.append(encoded[:, -1].flatten(start_dim=1).cpu())
    return torch.cat(output)


def cem_standard_normals(
    seed: int,
    episode: int,
    replan: int,
    iterations: int,
    samples: int,
    horizon: int,
    action_dim: int,
) -> np.ndarray:
    """Generate deterministic common random numbers for paired CEM planners."""
    if min(iterations, samples, horizon, action_dim) < 1:
        raise ValueError("CEM dimensions must be positive")
    combined_seed = seed * 100_000 + episode * 100 + replan
    rng = np.random.default_rng(combined_seed)
    return rng.standard_normal((iterations, samples, horizon, action_dim)).astype(np.float32)
