#!/usr/bin/env python3
"""Paired closed-loop evaluation comparing standard CEM and trust-regularized MPC on PushT.

Uses the same start states, goal photos, world-model unroll budget (128 per replan),
and standard-normal random streams across planners and goal heads so that differences
reflect only the metric and search rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa_planning import ROOT  # noqa: E402
from jepa_planning import linear_head as lh  # noqa: E402
from jepa_planning import pusht_goal_groups as pg  # noqa: E402
from jepa_planning import pusht_head_planning as hp  # noqa: E402
from jepa_planning import trust_mpc as tm  # noqa: E402
from jepa_planning.pusht_multi_goal import cem_standard_normals, sample_initial_state  # noqa: E402
from experiments.run_pusht_head_closed_loop import cem_plan  # noqa: E402


def make_environment():
    """PushT environment matching the frozen DINO-WM checkpoint configuration."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv

    env = PushTEnv(render_size=224, with_target=True, with_velocity=True)
    env.seed(0)
    return env


def load_tasks_from_groups(groups_dir: Path, split: str, goal_photos: int, max_tasks: int | None = None):
    """Load target poses, reconstructed start states, and cached goal features for a split."""
    group_cfg = json.loads((groups_dir / "metadata.json").read_text())["config"]
    images = np.load(groups_dir / "images.npz")
    targets = images["target_poses"]

    photo_index = np.full((len(targets), goal_photos), -1)
    for row_idx, (group, slot) in enumerate(zip(images["group"], images["slot"])):
        if slot < goal_photos:
            photo_index[group, slot] = row_idx

    keep = np.flatnonzero((photo_index >= 0).all(axis=1))
    if max_tasks is not None:
        keep = keep[:max_tasks]

    initial_states = np.stack([sample_initial_state(pg.target_rng(group_cfg, split, int(g))) for g in keep])
    feature_bank = torch.load(groups_dir / "features.pt", map_location="cpu", weights_only=True)["groups"]
    kept_indices = photo_index[keep]
    goals = feature_bank[torch.from_numpy(kept_indices.ravel())].reshape(
        *kept_indices.shape, *feature_bank.shape[1:]
    )
    tasks = {"group": keep, "initial": initial_states, "target": targets[keep]}
    return tasks, goals


def run_episode_suite(
    env,
    model,
    head: torch.Tensor,
    head_name: str,
    planner_name: str,
    tasks: dict,
    goals: torch.Tensor,
    cfg: dict,
) -> list[dict]:
    """Run paired closed-loop episodes for one (head, planner) pair."""
    action_mean = model.preprocessor.action_mean.cpu().numpy()
    action_std = model.preprocessor.action_std.cpu().numpy()
    hold_chunk = tm.zero_displacement_chunk(
        action_mean, action_std, cfg["controls_per_chunk"], cfg["action_clip"]
    )
    head = head.to(model.device).float()
    rows = []

    for task_idx in range(len(tasks["target"])):
        target = tasks["target"][task_idx]
        start_metrics = {
            k: float(v[0])
            for k, v in hp.physical_metrics(tasks["initial"][task_idx][None, 2:5], target).items()
        }
        for photo_idx in range(cfg["goal_photos"]):
            goal = goals[task_idx, photo_idx].to(model.device).float()
            env.reset_to_state = tasks["initial"][task_idx]
            observation, _ = env.reset()

            best_position_error = start_metrics["position_error"]
            previous_plan = None
            unrolls = 0
            sim_steps = 0
            started = perf_counter()

            for replan in range(cfg["replans"]):
                normals = cem_standard_normals(
                    cfg["seed"],
                    int(tasks["group"][task_idx]),
                    replan,
                    cfg["cem_iterations"],
                    cfg["cem_samples"],
                    cfg["cem_horizon"],
                    model.action_dim,
                )
                if planner_name == "standard_cem":
                    action_chunk, _ = cem_plan(model, observation, goal, head, normals, cfg)
                    unrolls += cfg["cem_iterations"] * cfg["cem_samples"]
                else:
                    action_chunk, previous_plan, info_plan = tm.trust_mpc_plan(
                        model,
                        observation,
                        goal,
                        head,
                        normals,
                        hold_chunk,
                        previous_plan,
                        num_elites=cfg["cem_elites"],
                        action_clip=cfg["action_clip"],
                        batch_size=cfg["batch_size"],
                        stay_margin=cfg.get("stay_margin", 0.02),
                    )
                    unrolls += info_plan["unrolls"]

                for control in action_chunk.reshape(cfg["controls_per_chunk"], 2):
                    observation, _, _, step_info = env.step(control * action_std + action_mean)
                    sim_steps += 1

                step_metrics = {
                    k: float(v[0])
                    for k, v in hp.physical_metrics(np.asarray(step_info["state"])[None, 2:5], target).items()
                }
                best_position_error = min(best_position_error, step_metrics["position_error"])

            rows.append(
                {
                    "head": head_name,
                    "planner": planner_name,
                    "task": task_idx,
                    "group": int(tasks["group"][task_idx]),
                    "photo": photo_idx,
                    "initial_position_error": start_metrics["position_error"],
                    "final_position_error": step_metrics["position_error"],
                    "best_position_error": best_position_error,
                    "post_peak_drift": step_metrics["position_error"] - best_position_error,
                    "final_combined_error": step_metrics["combined_error"],
                    "success_20px": step_metrics["success_20px"],
                    "pose_success": step_metrics["pose_success"],
                    "unrolls": unrolls,
                    "sim_steps": sim_steps,
                    "planning_seconds": perf_counter() - started,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True, help="Goal-group split directory")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--head-files", type=Path, nargs="*", default=[], help="Saved .pt head files")
    parser.add_argument("--planners", nargs="+", default=["standard_cem", "trust_mpc"])
    parser.add_argument("--config", type=Path, default=ROOT / "code/configs/pusht_head_closed_loop.yaml")
    parser.add_argument("--tasks", type=int, default=24)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)

    from jepa_planning.model import load_world_model

    tasks, goals = load_tasks_from_groups(args.groups, args.split, cfg["goal_photos"], args.tasks)
    heads = [("identity", lh.identity_head(lh.FEATURE_DIM))]
    for path in args.head_files:
        matrix, _ = lh.load_head(path)
        heads.append((path.stem, matrix))

    model = load_world_model("pusht", args.device).requires_grad_(False).eval()
    env = make_environment()
    all_rows = []
    try:
        for head_name, matrix in heads:
            for planner_name in args.planners:
                rows = run_episode_suite(env, model, matrix, head_name, planner_name, tasks, goals, cfg)
                all_rows.extend(rows)
                mean_err = float(np.mean([r["final_position_error"] for r in rows]))
                print(f"{head_name} + {planner_name}: mean final error {mean_err:.2f} px", flush=True)
    finally:
        env.close()

    with (args.output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()
