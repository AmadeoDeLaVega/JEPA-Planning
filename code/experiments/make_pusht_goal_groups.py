#!/usr/bin/env python3
"""Render PushT goal-image groups for one split (features: encode_goal_groups.py).

Writes to a new output directory (never overwrites):
  images.npz       visual [n, 224, 224, 3] uint8, proprio, group, slot, agent,
                   agent_bank_index, object_pose [n, 3]; plus target_poses [M, 3]
  sensitivity.npz  (val/test) visual, proprio, group, kind, amount, object_pose
                   for photos with the object slightly moved
  summary.json     images/sec, rejection counts by reason, targets short of agents
  metadata.json    config, checkpoint, git revision, versions, Slurm ids

Rendering and encoding run as separate processes: on macOS the decord wheel
the model code imports bundles its own SDL2, which clashes with pygame's.

Usage (repository root):
  .venv/bin/python code/experiments/make_pusht_goal_groups.py --split train \
    --output results/pusht_goal_groups/train
  .venv/bin/python code/experiments/encode_goal_groups.py results/pusht_goal_groups/train --device cuda
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa_planning import ROOT  # noqa: E402
from jepa_planning import pusht_goal_groups as pg  # noqa: E402


def make_environment():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv

    env = PushTEnv(render_size=224, with_target=True, with_velocity=True)
    env.seed(0)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=pg.SPLITS, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "code/configs/pusht_goal_groups.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-targets", type=int, help="override for smoke runs")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.num_targets is not None:
        cfg["num_targets"][args.split] = args.num_targets
    pg.validate_config(cfg)
    output = args.output or ROOT / "results" / "pusht_goal_groups" / args.split
    if output.exists():
        raise FileExistsError(f"{output} exists; choose a new --output")

    from app.plan_common.datasets import get_data_stats

    stats = get_data_stats("pusht")
    action_mean, action_std = np.asarray(stats["action_mean"]), np.asarray(stats["action_std"])
    env = make_environment()
    bank = pg.agent_bank(cfg)[args.split]
    needed = cfg["agents_per_target"][args.split]
    columns = {key: [] for key in ("visual", "proprio", "group", "slot", "agent", "agent_bank_index", "object_pose")}
    sensitivity = {key: [] for key in ("visual", "proprio", "group", "kind", "amount", "object_pose")}
    targets, rejected, sensitivity_rejected, short = [], Counter(), Counter(), []
    started = perf_counter()
    try:
        for group in range(cfg["num_targets"][args.split]):
            rng = pg.target_rng(cfg, args.split, group)
            pose = pg.sample_target(env, rng, cfg, action_mean, action_std)
            targets.append(pose)
            order = pg.agent_order(cfg, args.split, group, len(bank))
            photos, reasons = pg.render_group(env, pose, bank, order, needed, cfg["pose_tolerance"],
                                              cfg["min_agent_clearance"])
            rejected.update(reasons)
            if len(photos) < needed:
                short.append({"group": group, "accepted": len(photos)})
            for slot, photo in enumerate(photos):
                for key, value in [("visual", photo["visual"]), ("proprio", photo["proprio"]), ("group", group),
                                   ("slot", slot), ("agent", photo["agent"]),
                                   ("agent_bank_index", photo["agent_bank_index"]), ("object_pose", pose)]:
                    columns[key].append(value)
            if args.split != "train" and photos:
                for kind, amount, moved in pg.perturbed_poses(pose, cfg, rng):
                    observation, reason = pg.render_state(env, photos[0]["agent"], moved, cfg["pose_tolerance"],
                                                         cfg["min_agent_clearance"])
                    if reason:
                        sensitivity_rejected[reason] += 1
                        continue
                    for key, value in [("visual", observation["visual"]), ("proprio", observation["proprio"]),
                                       ("group", group), ("kind", kind), ("amount", amount), ("object_pose", moved)]:
                        sensitivity[key].append(value)
    finally:
        env.close()
    elapsed = perf_counter() - started

    arrays = {key: np.stack(value) for key, value in columns.items()}
    arrays["proprio"] = arrays["proprio"].astype(np.float32)
    output.mkdir(parents=True)
    np.savez(output / "images.npz", target_poses=np.stack(targets), **arrays)
    if sensitivity["visual"]:
        np.savez(output / "sensitivity.npz", **{key: np.stack(value) for key, value in sensitivity.items()})
    images = len(arrays["group"])
    tried = images + sum(rejected.values())
    summary = {
        "split": args.split,
        "num_targets": cfg["num_targets"][args.split],
        "num_images": images,
        "num_sensitivity_images": len(sensitivity["visual"]),
        "render_seconds": elapsed,
        "images_per_second": images / elapsed,
        "agent_attempts": tried,
        "rejection_rate": sum(rejected.values()) / tried if tried else 0.0,
        "rejected": dict(rejected),
        "sensitivity_rejected": dict(sensitivity_rejected),
        "targets_short_of_agents": short,
    }

    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    metadata = {
        "command": sys.argv,
        "config": cfg,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pymunk")},
        "slurm": {name: os.environ.get(name) for name in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST")},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
