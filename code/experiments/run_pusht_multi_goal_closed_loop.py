#!/usr/bin/env python3
"""Run a paired closed-loop CEM comparison of one versus four PushT goals.

Each target is selected from simulator-reachable action sequences before either
planner runs. Both planners restore the same initial state and receive identical
CEM standard-normal samples at every iteration and replan; only the terminal
goal cost changes. ``l2_g1`` uses the first valid photo, while ``l2_g4`` averages
four photos with the same object target and independently sampled agent poses.

Run on a Slurm-allocated GPU after setup::

    .venv/bin/python code/experiments/run_pusht_multi_goal_closed_loop.py \
      --device cuda --output results/pusht_multi_goal_closed_loop/seed_20261108
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa_planning import ROOT  # noqa: E402
from jepa_planning.goal_costs import mean_multi_goal_distance  # noqa: E402
from jepa_planning.model import load_world_model  # noqa: E402
from jepa_planning.pusht_multi_goal import (  # noqa: E402
    cem_standard_normals,
    encode_observations,
    physical_errors,
    sample_action_chunks,
    sample_initial_state,
    state_from,
)
from scripts.download_checkpoint import sha256  # noqa: E402


CONFIG_FIELDS = {
    "seed",
    "episodes",
    "goals_per_episode",
    "target_candidates",
    "controls_per_chunk",
    "cem_horizon",
    "replans",
    "cem_samples",
    "cem_elites",
    "cem_iterations",
    "batch_size",
    "action_clip",
}
EPISODE_METRICS = (
    "final_error",
    "best_error",
    "final_position_error",
    "final_angle_error",
    "error_reduction",
    "ever_success",
)


def validate_config(config: dict) -> None:
    """Validate the paired closed-loop protocol before allocating a GPU."""
    if not isinstance(config, dict) or set(config) != CONFIG_FIELDS:
        raise ValueError(f"config fields must be {sorted(CONFIG_FIELDS)}")
    for field in CONFIG_FIELDS - {"action_clip"}:
        minimum = 0 if field == "seed" else 1
        if type(config[field]) is not int or config[field] < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    if config["goals_per_episode"] != 4:
        raise ValueError("this preregistered comparison requires exactly four goals")
    if not 2 <= config["cem_elites"] < config["cem_samples"]:
        raise ValueError("require 2 <= cem_elites < cem_samples")
    if type(config["action_clip"]) not in (int, float) or not math.isfinite(config["action_clip"]):
        raise ValueError("action_clip must be finite and positive")
    if config["action_clip"] <= 0:
        raise ValueError("action_clip must be finite and positive")
    if config["seed"] >= 2**32:
        raise ValueError("seed must fit uint32")


def _make_environment():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv

    return PushTEnv(render_size=224, with_target=True, with_velocity=True)


def _render_states(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    env = _make_environment()
    frames = []
    proprioception = []
    try:
        env.seed(0)
        for state in states:
            env.reset_to_state = state
            observation, _ = env.reset()
            frames.append(observation["visual"])
            proprioception.append(observation["proprio"])
    finally:
        env.close()
    return np.stack(frames), np.stack(proprioception)


def generate_tasks(config: dict, action_mean: np.ndarray, action_std: np.ndarray) -> dict:
    """Construct reachable targets independently of either tested planner."""
    rng = np.random.default_rng(config["seed"] + 1000)
    env = _make_environment()
    env.seed(config["seed"])
    initial_states = []
    goal_poses = []
    expert_actions = []
    goal_states = []
    try:
        for _ in range(config["episodes"]):
            initial = sample_initial_state(rng)
            candidates = sample_action_chunks(
                rng,
                config["target_candidates"],
                config["cem_horizon"],
                config["controls_per_chunk"],
                config["action_clip"],
            )
            endpoints = []
            for sequence in candidates:
                env.reset_to_state = initial
                env.reset()
                for normalized_action in sequence.reshape(-1, 2):
                    executed = normalized_action * action_std + action_mean
                    _, _, _, info = env.step(executed)
                endpoints.append(np.asarray(info["state"][2:5], dtype=np.float32))
            endpoints = np.stack(endpoints)
            _, _, distance_from_start, _ = physical_errors(endpoints, initial[2:5])
            selected = int(np.argmax(distance_from_start))
            goal_pose = endpoints[selected]
            initial_states.append(initial)
            goal_poses.append(goal_pose)
            expert_actions.append(candidates[selected])
            goal_states.extend(
                state_from(rng.uniform(45.0, 467.0, size=2), goal_pose)
                for _ in range(config["goals_per_episode"])
            )
    finally:
        env.close()
    return {
        "initial_states": np.stack(initial_states),
        "goal_poses": np.stack(goal_poses),
        "expert_actions": np.stack(expert_actions),
        "goal_states": np.stack(goal_states),
    }


@torch.inference_mode()
def terminal_costs(
    model,
    encoded_observation: dict,
    actions: np.ndarray,
    goal_features: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    """Evaluate terminal multi-goal feature MSE for a CEM population."""
    costs = []
    for start in range(0, len(actions), batch_size):
        stop = min(start + batch_size, len(actions))
        action_tensor = torch.from_numpy(actions[start:stop]).permute(1, 0, 2).to(model.device)
        prediction = model.unroll(encoded_observation, action_tensor)["visual"][-1]
        flattened = prediction.flatten(start_dim=1)
        batch_cost = mean_multi_goal_distance(
            flattened, goal_features.to(device=flattened.device, dtype=flattened.dtype)
        )
        costs.append(batch_cost.cpu().numpy())
    return np.concatenate(costs)


@torch.inference_mode()
def cem_plan(
    model,
    observation: dict,
    goal_features: torch.Tensor,
    normals: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, float]:
    """Plan one normalized action chunk with reduced-budget CEM."""
    visual = torch.from_numpy(observation["visual"]).permute(2, 0, 1)[None, None]
    proprio = torch.from_numpy(observation["proprio"])[None, None]
    encoded = model.encode({"visual": visual, "proprio": proprio})
    action_dim = config["controls_per_chunk"] * 2
    if model.action_dim != action_dim:
        raise ValueError(f"model action_dim={model.action_dim}, expected {action_dim}")
    mean = np.zeros((config["cem_horizon"], action_dim), dtype=np.float32)
    std = np.ones_like(mean)
    minimum_cost = float("nan")
    for iteration in range(config["cem_iterations"]):
        actions = mean[None] + std[None] * normals[iteration]
        actions = np.clip(actions, -config["action_clip"], config["action_clip"]).astype(np.float32)
        actions[0] = mean
        costs = terminal_costs(model, encoded, actions, goal_features, config["batch_size"])
        elite_indices = np.argsort(costs)[: config["cem_elites"]]
        elites = actions[elite_indices]
        mean = elites.mean(axis=0)
        std = elites.std(axis=0)
        minimum_cost = float(costs[elite_indices[0]])
    return np.clip(mean[0], -config["action_clip"], config["action_clip"]), minimum_cost


def metric_row(state: np.ndarray, goal_pose: np.ndarray) -> dict[str, float | int]:
    """Report physical object error for one simulator state."""
    position, angle, combined, success = physical_errors(state[None, 2:5], goal_pose)
    return {
        "position_error": float(position[0]),
        "angle_error": float(angle[0]),
        "combined_error": float(combined[0]),
        "success": int(success[0]),
    }


def run_paired_episodes(config: dict, model, tasks: dict, goal_features: torch.Tensor) -> tuple[list[dict], list[dict]]:
    """Run both methods with matched state and CEM randomness."""
    action_mean = model.preprocessor.action_mean.cpu().numpy()
    action_std = model.preprocessor.action_std.cpu().numpy()
    env = _make_environment()
    env.seed(config["seed"])
    trajectories = []
    episodes = []
    try:
        for episode in range(config["episodes"]):
            initial = tasks["initial_states"][episode]
            goal_pose = tasks["goal_poses"][episode]
            start = episode * config["goals_per_episode"]
            all_goals = goal_features[start : start + config["goals_per_episode"]]
            initial_metrics = metric_row(initial, goal_pose)
            for method, goal_count in (("l2_g1", 1), ("l2_g4", 4)):
                env.reset_to_state = initial
                observation, state = env.reset()
                best_error = initial_metrics["combined_error"]
                ever_success = initial_metrics["success"]
                trajectories.append(
                    {"episode": episode, "method": method, "replan": 0, "planner_cost": "", **initial_metrics}
                )
                for replan in range(config["replans"]):
                    normals = cem_standard_normals(
                        config["seed"],
                        episode,
                        replan,
                        config["cem_iterations"],
                        config["cem_samples"],
                        config["cem_horizon"],
                        model.action_dim,
                    )
                    action, planner_cost = cem_plan(
                        model, observation, all_goals[:goal_count], normals, config
                    )
                    for normalized_action in action.reshape(config["controls_per_chunk"], 2):
                        executed = normalized_action * action_std + action_mean
                        observation, _, _, info = env.step(executed)
                    state = np.asarray(info["state"])
                    metrics = metric_row(state, goal_pose)
                    best_error = min(best_error, metrics["combined_error"])
                    ever_success = max(ever_success, metrics["success"])
                    trajectories.append(
                        {"episode": episode, "method": method, "replan": replan + 1, "planner_cost": planner_cost, **metrics}
                    )
                final_metrics = metric_row(np.asarray(state), goal_pose)
                episodes.append(
                    {
                        "episode": episode,
                        "method": method,
                        "initial_error": initial_metrics["combined_error"],
                        "final_position_error": final_metrics["position_error"],
                        "final_angle_error": final_metrics["angle_error"],
                        "final_error": final_metrics["combined_error"],
                        "best_error": best_error,
                        "error_reduction": initial_metrics["combined_error"] - final_metrics["combined_error"],
                        "ever_success": ever_success,
                    }
                )
    finally:
        env.close()
    return trajectories, episodes


def summarize(episodes: list[dict]) -> list[dict]:
    """Compute per-method means for one seed bank."""
    output = []
    for method in ("l2_g1", "l2_g4"):
        selected = [row for row in episodes if row["method"] == method]
        if not selected:
            raise ValueError(f"missing rows for {method}")
        output.append(
            {
                "method": method,
                "episodes": len(selected),
                **{metric: float(np.mean([row[metric] for row in selected])) for metric in EPISODE_METRICS},
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "code/configs/pusht_multi_goal_closed_loop.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pusht_multi_goal_closed_loop")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    for field in ("seed", "episodes"):
        if getattr(args, field) is not None:
            config[field] = getattr(args, field)
    if args.smoke:
        config.update(episodes=1, target_candidates=4, replans=1, cem_samples=4, cem_elites=2, cem_iterations=1, batch_size=2)
    validate_config(config)
    if args.output.exists():
        raise FileExistsError(f"choose a new output directory: {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a smoke test")

    started = perf_counter()
    torch.manual_seed(config["seed"])
    model = load_world_model("pusht", str(device)).requires_grad_(False).eval()
    action_mean = model.preprocessor.action_mean.cpu().numpy()
    action_std = model.preprocessor.action_std.cpu().numpy()
    tasks = generate_tasks(config, action_mean, action_std)
    goal_frames, goal_proprioception = _render_states(tasks["goal_states"])
    goals = encode_observations(model, goal_frames, goal_proprioception, config["batch_size"])
    trajectories, episodes = run_paired_episodes(config, model, tasks, goals)
    summary = summarize(episodes)

    args.output.mkdir(parents=True)
    write_csv(args.output / "trajectories.csv", trajectories)
    write_csv(args.output / "episodes.csv", episodes)
    write_csv(args.output / "summary.csv", summary)
    np.savez_compressed(
        args.output / "tasks.npz",
        initial_states=tasks["initial_states"],
        goal_poses=tasks["goal_poses"],
        expert_actions=tasks["expert_actions"],
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metadata = {
        "config": config,
        "checkpoint": str(ROOT / "checkpoints/dino_wm_pusht.pth.tar"),
        "checkpoint_sha256": sha256(ROOT / "checkpoints/dino_wm_pusht.pth.tar"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_revision": revision,
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy")},
        "command": sys.argv,
        "elapsed_seconds": perf_counter() - started,
        "slurm": {name: os.environ.get(name) for name in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_QOS", "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS")},
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
