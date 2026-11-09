#!/usr/bin/env python3
"""Render goal-image groups for one split (docs/linear_head_plan.md, step 1).

Writes to a new output directory (never overwrites):
  images.npz      frames [n, 224, 224, 3] uint8, group (target index), slot
                  (order within the group), target_seed, hand_bank_index,
                  hand_pos [n, 3], puck_pose [n, 7]
  hand_bank.npy   the split's bank xy positions
  summary.json    images/sec, rejection counts by reason, targets short of hands
  metadata.json   config, git revision, versions

Usage:
  MUJOCO_GL=egl .venv/bin/python code/experiments/make_goal_groups.py --split train
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

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from jepa_planning import ROOT  # noqa: E402
from jepa_planning import goal_groups as gg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=gg.SPLITS, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "code/configs/goal_groups.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-targets", type=int, help="override for smoke runs")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.num_targets is not None:
        cfg["num_targets"][args.split] = args.num_targets
    gg.validate_config(cfg)
    output = args.output or ROOT / "results" / "goal_groups" / args.split
    if output.exists():
        raise FileExistsError(f"{output} exists; choose a new --output")

    from jepa_planning import metaworld_push as mp

    env = mp.make_push_env()
    bank = gg.hand_bank(cfg)[args.split]
    needed = cfg["hands_per_target"][args.split]
    columns = {key: [] for key in ("frames", "group", "slot", "target_seed", "hand_bank_index", "hand_pos",
                                   "puck_pose")}
    rejected, short = Counter(), []
    started = perf_counter()
    for group, seed in enumerate(gg.target_seeds(cfg, args.split)):
        goal_snap, goal_pose = gg.make_target(env, int(seed), cfg)
        order = gg.hand_order(cfg, int(seed), len(bank))
        photos, reasons = gg.render_group(env, goal_snap, goal_pose, bank, order, needed, cfg)
        rejected.update(reasons)
        if len(photos) < needed:
            short.append({"target_seed": int(seed), "accepted": len(photos)})
        for slot, photo in enumerate(photos):
            columns["frames"].append(photo["frame"])
            columns["group"].append(group)
            columns["slot"].append(slot)
            columns["target_seed"].append(int(seed))
            columns["hand_bank_index"].append(photo["hand_bank_index"])
            columns["hand_pos"].append(photo["hand_pos"])
            columns["puck_pose"].append(goal_pose)
    elapsed = perf_counter() - started

    output.mkdir(parents=True)
    np.savez(output / "images.npz", **{key: np.stack(value) for key, value in columns.items()})
    np.save(output / "hand_bank.npy", bank)
    images = len(columns["frames"])
    tried = images + sum(rejected.values())
    summary = {
        "split": args.split,
        "num_targets": cfg["num_targets"][args.split],
        "num_images": images,
        "seconds": elapsed,
        "images_per_second": images / elapsed,
        "hand_attempts": tried,
        "rejection_rate": sum(rejected.values()) / tried if tried else 0.0,
        "rejected": dict(rejected),
        "targets_short_of_hands": short,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    metadata = {
        "command": sys.argv,
        "config": cfg,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "metaworld", "mujoco")},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
