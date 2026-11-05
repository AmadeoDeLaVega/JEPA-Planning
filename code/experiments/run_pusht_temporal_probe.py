#!/usr/bin/env python3
"""Compare the temporal cost with one-photo feature MSE on the fixed PushT bank.

The bank, goal photos, and config are the same as the multi-goal probe, so seeds
give identical tasks. Each method scores candidates against one goal photo at a
time, and three questions are answered per task:

- regret: is the selected candidate physically good?
- photo_agreement: do photos of the same target select the same candidate?
- target_response: does changing the object target change the selection?

Agreement alone is not enough, since a constant cost agrees perfectly. It only
counts alongside a high target response.

From the repository root::

    .venv/bin/python code/experiments/run_pusht_temporal_probe.py --device cuda \
      --cost results/pusht_temporal_cost/seed_20261201/temporal_cost.pt \
      --output results/pusht_temporal_probe/seed_20261008
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_pusht_multi_goal_probe import (  # noqa: E402
    generate_action_bank,
    predict_action_candidates,
    validate_config,
    write_csv,
)
from jepa_planning import ROOT  # noqa: E402
from jepa_planning.goal_costs import pairwise_squared_goal_distance  # noqa: E402
from jepa_planning.model import load_world_model  # noqa: E402
from jepa_planning.pusht_multi_goal import encode_observations, selection_record  # noqa: E402
from jepa_planning.temporal_cost import TemporalCost, photo_agreement, target_response  # noqa: E402

METRICS = ("combined_error", "regret", "success", "rank_correlation")


def load_cost(path: Path, device: torch.device) -> TemporalCost:
    """Load a trained temporal cost for evaluation."""
    saved = torch.load(path, map_location=device, weights_only=True)
    config = saved["config"]
    cost = TemporalCost(width=config["width"], hidden=config["hidden"]).to(device)
    cost.load_state_dict(saved["state_dict"])
    return cost.eval()


@torch.inference_mode()
def temporal_costs(cost: TemporalCost, candidates: torch.Tensor, goals: torch.Tensor) -> np.ndarray:
    """Return a [N, K] matrix of predicted steps to each goal photo."""
    device = next(cost.parameters()).device
    candidates = candidates.to(device)
    return torch.stack([cost(candidates, goal[None].to(device)) for goal in goals], dim=1).cpu().numpy()


def mse_costs(candidates: torch.Tensor, goals: torch.Tensor) -> np.ndarray:
    """Return a [N, K] matrix of feature MSE to each goal photo."""
    return pairwise_squared_goal_distance(candidates, goals).numpy()


def task_record(costs: np.ndarray, other_costs: np.ndarray, poses: np.ndarray, goal_pose: np.ndarray) -> dict:
    """Average per-photo selection metrics and add the two sensitivity measures."""
    records = [selection_record(column, poses, goal_pose) for column in costs.T]
    return {
        **{metric: float(np.mean([row[metric] for row in records])) for metric in METRICS},
        "photo_agreement": photo_agreement(costs),
        "target_response": target_response(costs, other_costs),
    }


def evaluate(methods: dict, features: torch.Tensor, goals: torch.Tensor, bank: dict, config: dict) -> list[dict]:
    """Score every task with every method. The next task supplies the other target."""
    candidates_per_task, goals_per_task = config["candidates_per_task"], config["goals_per_task"]
    rows = []
    for task in range(config["tasks"]):
        candidate_slice = slice(task * candidates_per_task, (task + 1) * candidates_per_task)
        goal_slice = slice(task * goals_per_task, (task + 1) * goals_per_task)
        other = (task + 1) % config["tasks"]
        other_slice = slice(other * goals_per_task, (other + 1) * goals_per_task)
        for method, score in methods.items():
            record = task_record(
                score(features[candidate_slice], goals[goal_slice]),
                score(features[candidate_slice], goals[other_slice]),
                bank["candidate_poses"][candidate_slice],
                bank["goal_poses"][task],
            )
            rows.append({"method": method, "task": task, **record})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "code/configs/pusht_multi_goal_probe.yaml")
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    if args.smoke:
        config.update(tasks=2, candidates_per_task=4, goals_per_task=2, goal_counts=[1, 2], batch_size=2)
    validate_config(config)
    if args.output.exists():
        raise FileExistsError(f"choose a new output directory: {args.output}")

    device = torch.device(args.device)
    torch.manual_seed(config["seed"])
    model = load_world_model("pusht", str(device)).requires_grad_(False).eval()
    action_mean = model.preprocessor.action_mean.cpu().numpy()
    action_std = model.preprocessor.action_std.cpu().numpy()
    bank = generate_action_bank(config, action_mean, action_std)
    goals = encode_observations(model, bank["goal_frames"], bank["goal_proprioception"], config["batch_size"])
    predicted = predict_action_candidates(model, bank, config)

    cost = load_cost(args.cost, device)
    methods = {"mse": mse_costs, "temporal": lambda x, g: temporal_costs(cost, x, g)}
    rows = evaluate(methods, predicted, goals, bank, config)

    args.output.mkdir(parents=True)
    write_csv(args.output / "task_rows.csv", rows)
    (args.output / "metadata.json").write_text(json.dumps({"config": config, "cost": str(args.cost)}, indent=2) + "\n")
    means = {
        method: {key: float(np.mean([row[key] for row in rows if row["method"] == method])) for key in rows[0] if key not in ("method", "task")}
        for method in methods
    }
    print(json.dumps(means, indent=2))


if __name__ == "__main__":
    main()
