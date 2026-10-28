#!/usr/bin/env python3
"""Run paired-goal DINO-WM + CEM evaluation; see docs/reproduction.md."""

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

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jepa_planning import ROOT


def validate_config(cfg):
    fields = {"seed", "first_seed", "num_tasks", "warmup_steps", "goal_steps", "max_steps",
              "hand_separation", "planner"}
    planner_fields = {"horizon", "num_act_stepped", "iterations", "num_samples", "num_elites",
                      "var_scale", "batch_size"}
    if not isinstance(cfg, dict) or set(cfg) != fields:
        raise ValueError(f"Config fields must be {sorted(fields)}")
    planner = cfg["planner"]
    if not isinstance(planner, dict) or set(planner) != planner_fields:
        raise ValueError(f"Planner fields must be {sorted(planner_fields)}")
    for settings, names in [(cfg, fields - {"hand_separation", "planner"}), (planner, planner_fields - {"var_scale"})]:
        for name in names:
            minimum = 0 if name in {"seed", "first_seed", "warmup_steps"} else 1
            if type(settings[name]) is not int or settings[name] < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in [("hand_separation", cfg["hand_separation"]), ("var_scale", planner["var_scale"])]:
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 2 <= planner["num_elites"] <= planner["num_samples"]:
        raise ValueError("Require 2 <= num_elites <= num_samples")
    if planner["num_act_stepped"] > planner["horizon"]:
        raise ValueError("num_act_stepped must not exceed horizon")
    if cfg["warmup_steps"] + max(cfg["goal_steps"], cfg["max_steps"]) > 500:
        raise ValueError("Warmup plus goal/evaluation steps must fit the 500-step simulator limit")
    if cfg["seed"] >= 2**32 or cfg["first_seed"] + cfg["num_tasks"] > 2**32:
        raise ValueError("Seeds must fit uint32")


def summarize(rows):
    metrics = ["puck_error_cm", "success_1cm", "success_2cm", "success_5cm", "native_success",
               "native_success_any", "planning_seconds", "episode_seconds", "peak_cuda_mb"]
    def means(group):
        return {key: sum(row[key] for row in group) / len(group) for key in metrics}

    pairs = [(rows[i], rows[i + 1]) for i in range(0, len(rows), 2)]
    return {
        "num_tasks": len(pairs),
        "num_episodes": len(rows),
        "mean": means(rows),
        "by_goal": {goal: means([row for row in rows if row["goal"] == goal]) for goal in ("L", "R")},
        "paired_native_success_changed": sum(
            left["native_success"] != right["native_success"] for left, right in pairs
        ),
        "paired_mean_error_change_cm": sum(
            abs(left["puck_error_cm"] - right["puck_error_cm"]) for left, right in pairs
        ) / len(pairs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "code/configs/baseline.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/baseline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--first-seed", type=int)
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    validate_config(cfg)
    for name in ("first_seed", "num_tasks"):
        if getattr(args, name) is not None:
            cfg[name] = getattr(args, name)
    if args.smoke:
        cfg.update(num_tasks=1, max_steps=5)
        cfg["planner"].update(iterations=2, num_samples=8, num_elites=2, batch_size=4)
    validate_config(cfg)
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

    import numpy as np
    from PIL import Image
    import torch

    from jepa_planning import metaworld_push as mp
    from jepa_planning.baseline import prepare_task, run_episode
    from jepa_planning.model import load_world_model
    from scripts.download_checkpoint import sha256

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a smoke test")
    torch.manual_seed(cfg["seed"])
    started = perf_counter()
    model = load_world_model("metaworld", str(device)).requires_grad_(False).eval()
    if model.action_dim != 4 * mp.FRAMESKIP:
        raise ValueError("Expected five 4D simulator actions per model action")
    load_seconds = perf_counter() - started
    args.output.mkdir(parents=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    manifest = {
        "config": cfg, "smoke": args.smoke, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(), "git_revision": revision, "git_dirty": dirty,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "metaworld", "mujoco")},
        "checkpoint_sha256": sha256(ROOT / "checkpoints/dino_wm_metaworld.pth.tar"),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [
            Path(__file__).resolve(), *(ROOT / "code/jepa_planning").glob("*.py")
        ]},
        "model_load_seconds": load_seconds,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    env = mp.make_push_env()
    try:
        with (args.output / "episodes.csv").open("w", newline="") as handle:
            writer = None
            for seed in range(cfg["first_seed"], cfg["first_seed"] + cfg["num_tasks"]):
                task_dir = args.output / f"seed{seed}"
                task_dir.mkdir()
                start, goal_puck, photos, hands = prepare_task(env, seed, cfg)
                np.savez_compressed(task_dir / "task.npz", **start, goal_puck=goal_puck, hands=hands,
                                    native_target=env.unwrapped._target_pos.copy())
                Image.fromarray(np.concatenate(photos, axis=1)).save(task_dir / "goals_L_R.png")
                planner_seed = (cfg["seed"] + seed) % 2**32
                for label, photo in zip(("L", "R"), photos):
                    metrics, trace = run_episode(env, model, start, photo, goal_puck, cfg, planner_seed)
                    row = {"seed": seed, "goal": label,
                           "hand_separation_cm": float(100 * np.linalg.norm(hands[0] - hands[1])), **metrics}
                    np.savez_compressed(task_dir / f"{label}.npz", **trace)
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        env.close()
    summary = summarize(rows)
    summary["total_seconds"] = perf_counter() - started
    text = json.dumps(summary, indent=2)
    (args.output / "summary.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
