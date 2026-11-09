"""PushT goal-image groups for fitting the linear goal head.

A group is several photos of one T-object target pose ``(x, y, angle)`` that
differ only in the agent position. Targets are built like the closed-loop
evaluation targets in ``run_pusht_multi_goal_closed_loop.generate_tasks``:
sample an initial state, run ``target_candidates`` random action chunks, and
take the endpoint farthest from the start. Each split (train/val/test) draws
its targets from its own seeded random stream.

Agent positions come from one fixed bank of absolute positions, sampled
uniformly over the board independently of the targets and split into disjoint
train/val/test agents. For each target, bank agents are tried in a seeded
order. A photo is kept only if the agent disc is at least ``min_agent_clearance`` px
from the T (checked geometrically) and, after the simulator's reset step, the
object is still at the target pose and the agent at its bank position. Rejections are counted
because they make agents and targets slightly dependent.

For the photo-level sensitivity check, val/test groups can also get photos with
the object shifted by a few pixels or rotated by a few degrees. These photos
use the agent from the group's first photo.

The simulator functions take the environment as an argument, so tests can use
a stand-in without pymunk.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from jepa_planning.pusht_multi_goal import (
    physical_errors,
    sample_action_chunks,
    sample_initial_state,
    state_from,
)

SPLITS = ("train", "val", "test")
# T geometry in the env's block frame (add_tee, scale 30): world = position + R(angle) @ local.
T_RECTANGLES = ((-60.0, 60.0, 0.0, 30.0), (-15.0, 15.0, 30.0, 120.0))  # (x0, x1, y0, y1)
AGENT_RADIUS = 15.0
SPLIT_STREAM = {"train": 0, "val": 1, "test": 2}


def validate_config(cfg: dict) -> None:
    fields = {"seed", "num_targets", "agents_per_target", "target", "agent_bank", "pose_tolerance",
              "min_agent_clearance", "sensitivity"}
    if set(cfg) != fields:
        raise ValueError(f"Config fields must be {sorted(fields)}")
    for name in ("num_targets", "agents_per_target"):
        if set(cfg[name]) != set(SPLITS):
            raise ValueError(f"{name} needs an entry for each of {SPLITS}")
        if any(type(v) is not int or v < 1 for v in cfg[name].values()):
            raise ValueError(f"{name} values must be positive integers")
    target = cfg["target"]
    if set(target) != {"target_candidates", "horizon", "controls_per_chunk", "action_clip"}:
        raise ValueError("target needs target_candidates, horizon, controls_per_chunk, action_clip")
    bank = cfg["agent_bank"]
    if set(bank) != {"size", "fractions", "low", "high"} or set(bank["fractions"]) != set(SPLITS):
        raise ValueError("agent_bank needs size, fractions (per split), low and high")
    if not bank["low"] < bank["high"]:
        raise ValueError("agent_bank low must be below high")
    if abs(sum(bank["fractions"].values()) - 1) > 1e-9:
        raise ValueError("agent_bank fractions must sum to 1")
    sizes = bank_split_sizes(bank)
    for split in SPLITS:
        if not 2 <= cfg["agents_per_target"][split] <= sizes[split]:
            raise ValueError(f"{split}: need 2 <= agents_per_target <= {sizes[split]} bank agents")
    tolerance = cfg["pose_tolerance"]
    if set(tolerance) != {"position", "angle"} or min(tolerance.values()) <= 0:
        raise ValueError("pose_tolerance needs positive position (px) and angle (rad)")
    if type(cfg["min_agent_clearance"]) not in (int, float) or cfg["min_agent_clearance"] < 0:
        raise ValueError("min_agent_clearance must be a non-negative number of pixels")
    sensitivity = cfg["sensitivity"]
    if set(sensitivity) != {"shifts_px", "rotations_deg"}:
        raise ValueError("sensitivity needs shifts_px and rotations_deg lists")


def bank_split_sizes(bank: dict) -> dict[str, int]:
    sizes = {split: int(round(bank["size"] * bank["fractions"][split])) for split in SPLITS[1:]}
    sizes["train"] = bank["size"] - sum(sizes.values())
    return sizes


def agent_bank(cfg: dict) -> dict[str, np.ndarray]:
    """Absolute agent positions (px) per split: ``{split: [n, 2]}``, disjoint by construction."""
    bank = cfg["agent_bank"]
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 0]))
    positions = rng.uniform(bank["low"], bank["high"], size=(bank["size"], 2))
    sizes = bank_split_sizes(bank)
    out, start = {}, 0
    for split in SPLITS:
        out[split] = positions[start : start + sizes[split]]
        start += sizes[split]
    return out


def target_rng(cfg: dict, split: str, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([cfg["seed"], 1, SPLIT_STREAM[split], index]))


def agent_order(cfg: dict, split: str, index: int, bank_size: int) -> np.ndarray:
    """Seeded order in which one target tries the agents of its split's bank."""
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 2, SPLIT_STREAM[split], index]))
    return rng.permutation(bank_size)


def sample_target(env, rng: np.random.Generator, cfg: dict, action_mean: np.ndarray,
                  action_std: np.ndarray) -> np.ndarray:
    """Target pose ``[3]``: farthest endpoint of random chunks from a random start."""
    target = cfg["target"]
    initial = sample_initial_state(rng)
    chunks = sample_action_chunks(rng, target["target_candidates"], target["horizon"],
                                  target["controls_per_chunk"], target["action_clip"])
    endpoints = []
    for sequence in chunks:
        env.reset_to_state = initial
        env.reset()
        for normalized_action in sequence.reshape(-1, 2):
            _, _, _, info = env.step(normalized_action * action_std + action_mean)
        endpoints.append(np.asarray(info["state"][2:5], dtype=np.float32))
    endpoints = np.stack(endpoints)
    _, _, distance_from_start, _ = physical_errors(endpoints, initial[2:5])
    return endpoints[int(np.argmax(distance_from_start))]


def agent_clearance(agent: np.ndarray, object_pose: np.ndarray) -> float:
    """Gap in px between the agent disc and the T (negative = they overlap)."""
    x, y, angle = (float(v) for v in object_pose)
    dx, dy = float(agent[0]) - x, float(agent[1]) - y
    cos, sin = np.cos(angle), np.sin(angle)
    local_x, local_y = cos * dx + sin * dy, -sin * dx + cos * dy  # R(-angle) @ (agent - position)
    distance = min(
        np.hypot(max(x0 - local_x, 0.0, local_x - x1), max(y0 - local_y, 0.0, local_y - y1))
        for x0, x1, y0, y1 in T_RECTANGLES
    )
    return float(distance - AGENT_RADIUS)


def render_state(env, agent: np.ndarray, object_pose: np.ndarray, tolerance: dict, min_clearance: float = 0.0):
    """Render one state; return ``(observation, reason)`` where reason is None if accepted.

    States where the agent disc is closer than ``min_clearance`` px to the T are
    rejected before rendering (``observation`` is then None): the env's single
    reset step barely separates overlapping bodies, so the pose check alone
    would accept an agent drawn on top of the T.
    """
    if agent_clearance(agent, object_pose) < min_clearance:
        return None, "agent_on_object"
    env.reset_to_state = state_from(agent, object_pose)
    observation, state = env.reset()
    state = np.asarray(state, dtype=np.float64)
    position, angle, _, _ = physical_errors(state[None, 2:5], np.asarray(object_pose, dtype=np.float64))
    if position[0] > tolerance["position"] or angle[0] > tolerance["angle"]:
        return observation, "object_moved"
    if np.linalg.norm(state[:2] - agent) > tolerance["position"]:
        return observation, "agent_moved"
    return observation, None


def render_group(env, object_pose: np.ndarray, bank: np.ndarray, order: np.ndarray, needed: int,
                 tolerance: dict, min_clearance: float = 0.0) -> tuple[list[dict], Counter]:
    """Up to ``needed`` accepted photos of one target, trying bank agents in ``order``."""
    accepted, rejected = [], Counter()
    for index in order:
        if len(accepted) == needed:
            break
        observation, reason = render_state(env, bank[index], object_pose, tolerance, min_clearance)
        if reason:
            rejected[reason] += 1
        else:
            accepted.append({"visual": observation["visual"], "proprio": observation["proprio"],
                             "agent_bank_index": int(index), "agent": np.asarray(bank[index], dtype=np.float32)})
    return accepted, rejected


def perturbed_poses(object_pose: np.ndarray, cfg: dict, rng: np.random.Generator) -> list[tuple[str, float, np.ndarray]]:
    """Object poses shifted (random direction) or rotated by each configured amount."""
    out = []
    for shift in cfg["sensitivity"]["shifts_px"]:
        direction = rng.uniform(0, 2 * np.pi)
        pose = np.array(object_pose, dtype=np.float64)
        pose[:2] += shift * np.array([np.cos(direction), np.sin(direction)])
        out.append(("shift_px", float(shift), pose))
    for degrees in cfg["sensitivity"]["rotations_deg"]:
        pose = np.array(object_pose, dtype=np.float64)
        pose[2] = (pose[2] + rng.choice([-1, 1]) * np.deg2rad(degrees)) % (2 * np.pi)
        out.append(("rotation_deg", float(degrees), pose))
    return out
