#!/usr/bin/env python3
"""Evaluate one versus several equivalent goal photos on a shared PushT bank.

Hypothesis: averaging frozen DINO-WM feature MSE across goal photos that share
the same object target but vary the nuisance agent pose reduces physical
candidate-selection regret. Every method sees the same initial states, action
chunks, predicted endpoints, and physical oracle. For goal counts below the
full set, every subset is evaluated so a lucky goal ordering cannot determine
the result.

Run from the repository root after environment/checkpoint setup::

    .venv/bin/python code/experiments/run_pusht_multi_goal_probe.py \
      --device cuda --output results/pusht_multi_goal_probe/seed_20261008

Generated outputs are ignored by Git. Use a fresh output path for each run.
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
from jepa_planning.model import load_world_model  # noqa: E402
from jepa_planning.pusht_multi_goal import (  # noqa: E402
    encode_observations,
    evaluate_goal_counts,
    physical_errors,
    sample_action_chunks,
    sample_initial_state,
    state_from,
)
from scripts.download_checkpoint import sha256  # noqa: E402


CONFIG_FIELDS = {
    "seed",
    "tasks",
    "candidates_per_task",
    "goals_per_task",
    "goal_counts",
    "controls_per_candidate",
    "batch_size",
    "action_clip",
}
METRICS = (
    "position_error",
    "angle_error",
    "combined_error",
    "regret",
    "success",
    "rank_correlation",
    "oracle_success_available",
)


def validate_config(config: dict) -> None:
    """Reject incomplete or ambiguous experiment configurations."""
    if not isinstance(config, dict) or set(config) != CONFIG_FIELDS:
        raise ValueError(f"config fields must be {sorted(CONFIG_FIELDS)}")
    integer_fields = (
        "seed",
        "tasks",
        "candidates_per_task",
        "goals_per_task",
        "controls_per_candidate",
        "batch_size",
    )
    for field in integer_fields:
        minimum = 0 if field == "seed" else 1
        if type(config[field]) is not int or config[field] < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    counts = config["goal_counts"]
    if (
        not isinstance(counts, list)
        or not counts
        or any(type(count) is not int or not 1 <= count <= config["goals_per_task"] for count in counts)
        or len(set(counts)) != len(counts)
    ):
        raise ValueError("goal_counts must be unique integers in [1, goals_per_task]")
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
    """Render deterministic observations for explicit PushT states."""
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


def generate_action_bank(config: dict, action_mean: np.ndarray, action_std: np.ndarray) -> dict:
    """Generate shared simulator action banks and equivalent goal photos."""
    rng = np.random.default_rng(config["seed"] + 1000)
    env = _make_environment()
    env.seed(config["seed"])
    initial_frames = []
    initial_proprioception = []
    initial_states = []
    normalized_actions = []
    candidate_frames = []
    candidate_proprioception = []
    candidate_states = []
    candidate_poses = []
    goal_poses = []
    try:
        for _ in range(config["tasks"]):
            initial = sample_initial_state(rng)
            chunks = sample_action_chunks(
                rng,
                config["candidates_per_task"],
                horizon=1,
                controls_per_chunk=config["controls_per_candidate"],
                action_clip=config["action_clip"],
            )[:, 0]
            env.reset_to_state = initial
            initial_observation, _ = env.reset()
            initial_frames.append(initial_observation["visual"])
            initial_proprioception.append(initial_observation["proprio"])
            initial_states.append(initial)

            endpoints = []
            for chunk in chunks:
                env.reset_to_state = initial
                env.reset()
                for normalized_action in chunk:
                    executed_action = normalized_action * action_std + action_mean
                    observation, _, _, info = env.step(executed_action)
                state = np.asarray(info["state"], dtype=np.float32)
                candidate_frames.append(observation["visual"])
                candidate_proprioception.append(observation["proprio"])
                candidate_states.append(state)
                candidate_poses.append(state[2:5])
                endpoints.append(state)

            endpoints = np.stack(endpoints)
            displacement = np.linalg.norm(endpoints[:, 2:4] - initial[None, 2:4], axis=1)
            angle_change = np.abs(
                (endpoints[:, 4] - initial[4] + np.pi) % (2 * np.pi) - np.pi
            )
            reachable_index = int(np.argmax(np.sqrt(displacement**2 + (50 * angle_change) ** 2)))
            goal_pose = endpoints[reachable_index, 2:5].copy()
            # Avoid an exact feature/pixel match while keeping the target near
            # a simulator-demonstrated endpoint.
            goal_pose += rng.normal(scale=(8.0, 8.0, 0.10), size=3)
            goal_pose[:2] = np.clip(goal_pose[:2], 105.0, 407.0)
            goal_pose[2] %= 2 * np.pi
            goal_poses.append(goal_pose)
            normalized_actions.append(chunks)
    finally:
        env.close()

    goal_states = [
        state_from(rng.uniform(45.0, 467.0, size=2), goal_pose)
        for goal_pose in goal_poses
        for _ in range(config["goals_per_task"])
    ]
    goal_frames, goal_proprioception = _render_states(np.stack(goal_states))
    return {
        "initial_frames": np.stack(initial_frames),
        "initial_proprioception": np.stack(initial_proprioception),
        "initial_states": np.stack(initial_states),
        "actions": np.stack(normalized_actions),
        "candidate_frames": np.stack(candidate_frames),
        "candidate_proprioception": np.stack(candidate_proprioception),
        "candidate_states": np.stack(candidate_states),
        "candidate_poses": np.stack(candidate_poses),
        "goal_frames": goal_frames,
        "goal_proprioception": goal_proprioception,
        "goal_poses": np.stack(goal_poses),
    }


@torch.inference_mode()
def predict_action_candidates(model, bank: dict, config: dict) -> torch.Tensor:
    """Unroll the frozen predictor for every candidate action chunk."""
    predicted = []
    candidate_count = config["candidates_per_task"]
    for task in range(config["tasks"]):
        visual = torch.from_numpy(bank["initial_frames"][task]).permute(2, 0, 1)[None, None]
        proprio = torch.from_numpy(bank["initial_proprioception"][task])[None, None]
        encoded = model.encode({"visual": visual, "proprio": proprio})
        for start in range(0, candidate_count, config["batch_size"]):
            stop = min(start + config["batch_size"], candidate_count)
            actions = bank["actions"][task, start:stop].reshape(stop - start, -1)
            action_tensor = torch.from_numpy(actions).unsqueeze(0).to(model.device)
            features = model.unroll(encoded, action_tensor)["visual"][-1]
            predicted.append(features.flatten(start_dim=1).cpu())
    return torch.cat(predicted)


def evaluate_bank(config: dict, sources: dict[str, torch.Tensor], goals: torch.Tensor, bank: dict) -> list[dict]:
    """Evaluate all tasks and fair goal subsets for each feature source."""
    records = []
    candidates_per_task = config["candidates_per_task"]
    goals_per_task = config["goals_per_task"]
    for source, features in sources.items():
        for task in range(config["tasks"]):
            candidate_slice = slice(task * candidates_per_task, (task + 1) * candidates_per_task)
            goal_slice = slice(task * goals_per_task, (task + 1) * goals_per_task)
            task_records = evaluate_goal_counts(
                features[candidate_slice],
                goals[goal_slice],
                bank["candidate_poses"][candidate_slice],
                bank["goal_poses"][task],
                config["goal_counts"],
            )
            records.extend({"source": source, "task": task, **row} for row in task_records)
    return records


def summarize(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Average subsets within tasks before averaging across tasks."""
    task_rows = []
    groups = sorted({(row["source"], row["goal_count"], row["task"]) for row in records})
    for source, goal_count, task in groups:
        selected = [
            row
            for row in records
            if row["source"] == source and row["goal_count"] == goal_count and row["task"] == task
        ]
        task_rows.append(
            {
                "source": source,
                "goal_count": goal_count,
                "task": task,
                "subsets": len(selected),
                **{metric: float(np.mean([row[metric] for row in selected])) for metric in METRICS},
            }
        )
    summary = []
    for source, goal_count in sorted({(row["source"], row["goal_count"]) for row in task_rows}):
        selected = [row for row in task_rows if row["source"] == source and row["goal_count"] == goal_count]
        summary.append(
            {
                "source": source,
                "goal_count": goal_count,
                "tasks": len(selected),
                **{metric: float(np.mean([row[metric] for row in selected])) for metric in METRICS},
            }
        )
    return task_rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a non-empty list of dictionaries with a stable header."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "code/configs/pusht_multi_goal_probe.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pusht_multi_goal_probe")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tasks", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    for field in ("seed", "tasks"):
        if getattr(args, field) is not None:
            config[field] = getattr(args, field)
    if args.smoke:
        config.update(tasks=1, candidates_per_task=4, goals_per_task=2, goal_counts=[1, 2], batch_size=2)
    validate_config(config)
    if args.output.exists():
        raise FileExistsError(f"choose a new output directory: {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a smoke test")

    started = perf_counter()
    torch.manual_seed(config["seed"])
    model = load_world_model("pusht", str(device)).requires_grad_(False).eval()
    if model.action_dim != 2 * config["controls_per_candidate"]:
        raise ValueError("controls_per_candidate does not match the checkpoint action dimension")
    action_mean = model.preprocessor.action_mean.cpu().numpy()
    action_std = model.preprocessor.action_std.cpu().numpy()
    bank = generate_action_bank(config, action_mean, action_std)
    actual = encode_observations(
        model, bank["candidate_frames"], bank["candidate_proprioception"], config["batch_size"]
    )
    goals = encode_observations(
        model, bank["goal_frames"], bank["goal_proprioception"], config["batch_size"]
    )
    predicted = predict_action_candidates(model, bank, config)
    records = evaluate_bank(
        config, {"actual_encoded": actual, "action_predicted": predicted}, goals, bank
    )
    task_rows, summary = summarize(records)

    args.output.mkdir(parents=True)
    write_csv(args.output / "records.csv", records)
    write_csv(args.output / "task_rows.csv", task_rows)
    write_csv(args.output / "summary.csv", summary)
    np.savez_compressed(
        args.output / "bank.npz",
        initial_states=bank["initial_states"],
        actions=bank["actions"],
        candidate_states=bank["candidate_states"],
        goal_poses=bank["goal_poses"],
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
