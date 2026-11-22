#!/usr/bin/env python3
"""Measure where equivalent-goal averaging saturates on PushT.

Each task uses one shared 64-candidate action bank and 32 goal photographs
with the same object pose and independently varied agent positions. Goal counts
1, 2, 4, 8, 16, and 32 are compared. When a subset family is too large to
enumerate, a fixed number of uniformly sampled subsets is used with deterministic
per-task seeds.

Besides physical candidate-selection metrics, the experiment measures how each
partial goal prototype converges to the full 32-goal prototype: prototype MSE,
candidate-cost rank correlation, and selected-action agreement.

Run from the repository root::

    .venv/bin/python code/experiments/run_pusht_goal_count_saturation.py \
      --device cuda --output results/pusht_goal_count_saturation/seed_20261121

Generated outputs are ignored by Git and existing output directories are never
overwritten.
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

from experiments.run_pusht_multi_goal_probe import (  # noqa: E402
    generate_action_bank,
    predict_action_candidates,
)
from jepa_planning import ROOT  # noqa: E402
from jepa_planning.model import load_world_model  # noqa: E402
from jepa_planning.pusht_multi_goal import (  # noqa: E402
    encode_observations,
    evaluate_goal_count_saturation,
)
from scripts.download_checkpoint import sha256  # noqa: E402


CONFIG_FIELDS = {
    "seed",
    "tasks",
    "candidates_per_task",
    "goals_per_task",
    "goal_counts",
    "max_subsets_per_count",
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
    "prototype_mse_to_full",
    "cost_rank_correlation_to_full",
    "selection_matches_full",
)


def validate_config(config: dict) -> None:
    """Reject configurations that cannot define an ordered saturation curve."""
    if not isinstance(config, dict) or set(config) != CONFIG_FIELDS:
        raise ValueError(f"config fields must be {sorted(CONFIG_FIELDS)}")
    integer_fields = (
        "seed",
        "tasks",
        "candidates_per_task",
        "goals_per_task",
        "max_subsets_per_count",
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
        or counts != sorted(counts)
        or len(set(counts)) != len(counts)
        or counts[0] != 1
        or counts[-1] != config["goals_per_task"]
        or any(type(count) is not int or count < 1 for count in counts)
    ):
        raise ValueError(
            "goal_counts must be sorted unique integers from 1 through goals_per_task"
        )
    action_clip = config["action_clip"]
    if type(action_clip) not in (int, float) or not math.isfinite(action_clip) or action_clip <= 0:
        raise ValueError("action_clip must be finite and positive")
    if config["seed"] >= 2**32:
        raise ValueError("seed must fit uint32")


def evaluate_bank(
    config: dict,
    sources: dict[str, torch.Tensor],
    goals: torch.Tensor,
    bank: dict,
) -> list[dict]:
    """Evaluate identical deterministic goal subsets for every feature source."""
    records = []
    candidates_per_task = config["candidates_per_task"]
    goals_per_task = config["goals_per_task"]
    for source, features in sources.items():
        for task in range(config["tasks"]):
            candidate_slice = slice(task * candidates_per_task, (task + 1) * candidates_per_task)
            goal_slice = slice(task * goals_per_task, (task + 1) * goals_per_task)
            task_records = evaluate_goal_count_saturation(
                features[candidate_slice],
                goals[goal_slice],
                bank["candidate_poses"][candidate_slice],
                bank["goal_poses"][task],
                config["goal_counts"],
                config["max_subsets_per_count"],
                subset_seed=(config["seed"] + task) % 2**32,
            )
            records.extend({"source": source, "task": task, **row} for row in task_records)
    return records


def summarize(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Average sampled subsets within tasks before averaging tasks."""
    task_rows = []
    groups = sorted({(row["source"], row["goal_count"], row["task"]) for row in records})
    for source, goal_count, task in groups:
        selected = [
            row
            for row in records
            if row["source"] == source
            and row["goal_count"] == goal_count
            and row["task"] == task
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
        selected = [
            row
            for row in task_rows
            if row["source"] == source and row["goal_count"] == goal_count
        ]
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
    """Write non-empty dictionaries with a stable header."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "code/configs/pusht_goal_count_saturation.yaml",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/pusht_goal_count_saturation"
    )
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
        config.update(
            tasks=1,
            candidates_per_task=4,
            goals_per_task=4,
            goal_counts=[1, 2, 4],
            max_subsets_per_count=3,
            batch_size=2,
        )
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
        goal_states=bank["goal_states"],
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
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        ),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy")
        },
        "command": sys.argv,
        "elapsed_seconds": perf_counter() - started,
        "slurm": {
            name: os.environ.get(name)
            for name in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_PARTITION",
                "SLURM_JOB_QOS",
                "SLURM_JOB_NODELIST",
                "SLURM_JOB_GPUS",
            )
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
