#!/usr/bin/env python3
"""Run paired closed-loop CEM with the temporal cost on the multi-goal tasks.

Tasks, goal photos, and CEM random numbers are generated exactly as in
``run_pusht_multi_goal_closed_loop.py``, so a seed gives the same episodes, and
``mse`` reproduces that script's ``l2_g1`` rows. Every method plans against the
first goal photo, and only the terminal cost changes:

- ``mse``: feature MSE, the same cost as ``l2_g1``.
- ``temporal``: the learned temporal cost.
- ``blend``: ``(1 - weight) * z(MSE) + weight * z(temporal)``, standardized over
  the CEM population.

From the repository root::

    .venv/bin/python code/experiments/run_pusht_temporal_closed_loop.py --device cuda \
      --cost results/pusht_temporal_cost/seed_20261201/temporal_cost.pt \
      --output results/pusht_temporal_closed_loop/seed_20261108
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_pusht_multi_goal_closed_loop import (  # noqa: E402
    EPISODE_METRICS,
    _make_environment,
    _render_states,
    generate_tasks,
    metric_row,
    validate_config,
    write_csv,
)
from experiments.run_pusht_temporal_probe import load_cost  # noqa: E402
from jepa_planning import ROOT  # noqa: E402
from jepa_planning.goal_costs import mean_multi_goal_distance  # noqa: E402
from jepa_planning.model import load_world_model  # noqa: E402
from jepa_planning.pusht_multi_goal import cem_standard_normals, encode_observations  # noqa: E402
from jepa_planning.temporal_cost import blend_costs  # noqa: E402

METHODS = ("mse", "temporal", "blend")


@torch.inference_mode()
def population_costs(model, encoded: dict, actions: np.ndarray, goal: torch.Tensor, cost, batch_size: int) -> tuple:
    """Return MSE and temporal costs to one goal for every action sequence in a CEM population."""
    mse, temporal = [], []
    for start in range(0, len(actions), batch_size):
        action_tensor = torch.from_numpy(actions[start : start + batch_size]).permute(1, 0, 2).to(model.device)
        prediction = model.unroll(encoded, action_tensor)["visual"][-1].flatten(start_dim=1)
        mse.append(mean_multi_goal_distance(prediction, goal.to(device=prediction.device, dtype=prediction.dtype)).cpu().numpy())
        temporal.append(cost(prediction, goal.to(prediction.device)).cpu().numpy())
    return np.concatenate(mse), np.concatenate(temporal)


def method_costs(method: str, mse: np.ndarray, temporal: np.ndarray, blend_weight: float) -> np.ndarray:
    """Return the terminal cost a method plans with."""
    if method == "mse":
        return mse
    if method == "temporal":
        return temporal
    if method == "blend":
        return blend_costs(mse[:, None], temporal[:, None], blend_weight)[:, 0]
    raise ValueError(f"unknown method {method}")


@torch.inference_mode()
def cem_plan(model, observation: dict, goal: torch.Tensor, normals: np.ndarray, config: dict, score) -> tuple[np.ndarray, float]:
    """Plan one normalized action chunk with reduced-budget CEM; ``score`` maps MSE and temporal costs to costs."""
    visual = torch.from_numpy(observation["visual"]).permute(2, 0, 1)[None, None]
    proprio = torch.from_numpy(observation["proprio"])[None, None]
    encoded = model.encode({"visual": visual, "proprio": proprio})
    mean = np.zeros((config["cem_horizon"], config["controls_per_chunk"] * 2), dtype=np.float32)
    std = np.ones_like(mean)
    minimum_cost = float("nan")
    for iteration in range(config["cem_iterations"]):
        actions = mean[None] + std[None] * normals[iteration]
        actions = np.clip(actions, -config["action_clip"], config["action_clip"]).astype(np.float32)
        actions[0] = mean
        costs = score(*population_costs(model, encoded, actions, goal, score.cost, config["batch_size"]))
        elites = actions[np.argsort(costs)[: config["cem_elites"]]]
        mean, std = elites.mean(axis=0), elites.std(axis=0)
        minimum_cost = float(costs.min())
    return np.clip(mean[0], -config["action_clip"], config["action_clip"]), minimum_cost


class Scorer:
    """Bind a method and the temporal cost for ``cem_plan``."""

    def __init__(self, method: str, cost, blend_weight: float):
        self.method, self.cost, self.blend_weight = method, cost, blend_weight

    def __call__(self, mse: np.ndarray, temporal: np.ndarray) -> np.ndarray:
        return method_costs(self.method, mse, temporal, self.blend_weight)


def run_paired_episodes(config: dict, model, tasks: dict, goals: torch.Tensor, cost, blend_weight: float) -> list[dict]:
    """Run every method from the same states with the same CEM random numbers."""
    action_mean = model.preprocessor.action_mean.cpu().numpy()
    action_std = model.preprocessor.action_std.cpu().numpy()
    env = _make_environment()
    env.seed(config["seed"])
    episodes = []
    try:
        for episode in range(config["episodes"]):
            initial, goal_pose = tasks["initial_states"][episode], tasks["goal_poses"][episode]
            goal = goals[episode * config["goals_per_episode"]][None]  # the photo l2_g1 uses
            initial_metrics = metric_row(initial, goal_pose)
            for method in METHODS:
                env.reset_to_state = initial
                observation, state = env.reset()
                best_error, ever_success = initial_metrics["combined_error"], initial_metrics["success"]
                for replan in range(config["replans"]):
                    normals = cem_standard_normals(
                        config["seed"], episode, replan, config["cem_iterations"],
                        config["cem_samples"], config["cem_horizon"], model.action_dim,
                    )
                    action, _ = cem_plan(model, observation, goal, normals, config, Scorer(method, cost, blend_weight))
                    for normalized_action in action.reshape(config["controls_per_chunk"], 2):
                        observation, _, _, info = env.step(normalized_action * action_std + action_mean)
                    state = np.asarray(info["state"])
                    metrics = metric_row(state, goal_pose)
                    best_error = min(best_error, metrics["combined_error"])
                    ever_success = max(ever_success, metrics["success"])
                final = metric_row(np.asarray(state), goal_pose)
                episodes.append({
                    "episode": episode,
                    "method": method,
                    "initial_error": initial_metrics["combined_error"],
                    "final_position_error": final["position_error"],
                    "final_angle_error": final["angle_error"],
                    "final_error": final["combined_error"],
                    "best_error": best_error,
                    "error_reduction": initial_metrics["combined_error"] - final["combined_error"],
                    "ever_success": ever_success,
                })
    finally:
        env.close()
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "code/configs/pusht_multi_goal_closed_loop.yaml")
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--blend-weight", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    if args.smoke:
        config.update(episodes=1, target_candidates=4, replans=1, cem_samples=4, cem_elites=2, cem_iterations=1, batch_size=2)
    validate_config(config)
    if not 0 <= args.blend_weight <= 1:
        raise ValueError("blend weight must be in [0, 1]")
    if args.output.exists():
        raise FileExistsError(f"choose a new output directory: {args.output}")

    started = perf_counter()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    device = torch.device(args.device)
    torch.manual_seed(config["seed"])
    model = load_world_model("pusht", str(device)).requires_grad_(False).eval()
    tasks = generate_tasks(config, model.preprocessor.action_mean.cpu().numpy(), model.preprocessor.action_std.cpu().numpy())
    goal_frames, goal_proprioception = _render_states(tasks["goal_states"])
    goals = encode_observations(model, goal_frames, goal_proprioception, config["batch_size"])
    episodes = run_paired_episodes(config, model, tasks, goals, load_cost(args.cost, device), args.blend_weight)

    args.output.mkdir(parents=True)
    write_csv(args.output / "episodes.csv", episodes)
    summary = {
        method: {metric: float(np.mean([row[metric] for row in episodes if row["method"] == method])) for metric in EPISODE_METRICS}
        for method in METHODS
    }
    metadata = {
        "config": config,
        "cost": str(args.cost),
        "blend_weight": args.blend_weight,
        "git_revision": revision,
        "git_dirty": dirty,
        "torch": torch.__version__,
        "command": sys.argv,
        "elapsed_seconds": perf_counter() - started,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
