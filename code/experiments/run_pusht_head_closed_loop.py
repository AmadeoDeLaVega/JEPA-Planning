#!/usr/bin/env python3
"""Paired closed-loop CEM on PushT with each goal head as the terminal cost (plan step 6.4).

Tasks are the same held-out tasks as a run_pusht_head_bank.py output (``bank.npz``):
initial state, photographed target pose, and two equivalent goal photos per task.
For every head and every goal photo, CEM plans from the same start state. It gets
the same standard-normal samples at each (task, replan, iteration), the same
budget, and the same executed chunk length. Only the terminal cost (Eq. 3) changes.

Physical metrics are measured against the photographed target (``pusht_head_planning``):
final T position error, combined error, success within 10/20/50 px, pose success,
native (coverage) success, ever-success, and planning time.

Each head's episodes are written to ``episodes/<head>.csv`` as soon as that head
finishes, so ``--resume`` continues an interrupted run with heads not yet done.

Usage (repository root):
  .venv/bin/python code/experiments/run_pusht_head_closed_loop.py \
    --bank results/pusht_head_bank/test --runs results/pusht_linear_head_mac/rank32 \
    --heads identity rank32_B256_M128_K2_r0 --output results/pusht_head_closed_loop/test --device mps
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa_planning import ROOT  # noqa: E402
from jepa_planning import linear_head as lh  # noqa: E402
from jepa_planning import pusht_head_planning as hp  # noqa: E402
from jepa_planning.pusht_multi_goal import cem_standard_normals  # noqa: E402


def make_environment():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv

    env = PushTEnv(render_size=224, with_target=True, with_velocity=True)
    env.seed(0)
    return env


@torch.inference_mode()
def cem_plan(model, observation: dict, goal: torch.Tensor, head: torch.Tensor, normals: np.ndarray,
             cfg: dict) -> tuple[np.ndarray, float]:
    """One CEM plan; return the first normalized action chunk and the best elite cost."""
    visual = torch.from_numpy(observation["visual"]).permute(2, 0, 1)[None, None]
    proprio = torch.from_numpy(np.asarray(observation["proprio"], dtype=np.float32))[None, None]
    encoded = model.encode({"visual": visual, "proprio": proprio})
    action_dim = cfg["controls_per_chunk"] * 2
    mean = np.zeros((cfg["cem_horizon"], action_dim), dtype=np.float32)
    std = np.ones_like(mean)
    best = float("nan")
    for iteration in range(cfg["cem_iterations"]):
        actions = np.clip(mean[None] + std[None] * normals[iteration], -cfg["action_clip"], cfg["action_clip"])
        actions = actions.astype(np.float32)
        actions[0] = mean
        costs = []
        for start in range(0, len(actions), cfg["batch_size"]):
            batch = torch.from_numpy(actions[start : start + cfg["batch_size"]]).permute(1, 0, 2).to(model.device)
            features = model.unroll(encoded, batch)["visual"][-1]
            features = features.reshape(features.shape[0], -1, features.shape[-1]).float()
            costs.append(hp.head_costs(features, goal, head).cpu().numpy())
        costs = np.concatenate(costs)
        elites = np.argsort(costs)[: cfg["cem_elites"]]
        mean, std = actions[elites].mean(axis=0), actions[elites].std(axis=0)
        best = float(costs[elites[0]])
    return np.clip(mean[0], -cfg["action_clip"], cfg["action_clip"]), best


def run_head(env, model, head: torch.Tensor, tasks: dict, goals: torch.Tensor, cfg: dict) -> list[dict]:
    action_mean = model.preprocessor.action_mean.cpu().numpy()
    action_std = model.preprocessor.action_std.cpu().numpy()
    head = head.to(model.device)
    rows = []
    for task in range(len(tasks["target"])):
        target = tasks["target"][task]
        initial = {k: float(v[0]) for k, v in hp.physical_metrics(tasks["initial"][task][None, 2:5], target).items()}
        for photo in range(cfg["goal_photos"]):
            goal = goals[task, photo].to(model.device).float()
            env.reset_to_state = tasks["initial"][task]
            observation, _ = env.reset()
            ever_native, ever_pose, best_combined = initial["native_success"], initial["pose_success"], initial["combined_error"]
            started = perf_counter()
            for replan in range(cfg["replans"]):
                normals = cem_standard_normals(cfg["seed"], int(tasks["group"][task]), replan, cfg["cem_iterations"],
                                               cfg["cem_samples"], cfg["cem_horizon"], model.action_dim)
                action, _ = cem_plan(model, observation, goal, head, normals, cfg)
                for normalized_action in action.reshape(cfg["controls_per_chunk"], 2):
                    observation, _, _, info = env.step(normalized_action * action_std + action_mean)
                step = {k: float(v[0]) for k, v in hp.physical_metrics(np.asarray(info["state"])[None, 2:5], target).items()}
                ever_native = max(ever_native, step["native_success"])
                ever_pose = max(ever_pose, step["pose_success"])
                best_combined = min(best_combined, step["combined_error"])
            rows.append({"task": task, "group": int(tasks["group"][task]), "photo": photo,
                         "initial_position_error": initial["position_error"],
                         "initial_combined_error": initial["combined_error"],
                         **{f"final_{k}": v for k, v in step.items()},
                         "ever_native_success": ever_native, "ever_pose_success": ever_pose,
                         "best_combined_error": best_combined, "planning_seconds": perf_counter() - started})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True, help="run_pusht_head_bank.py output with bank.npz")
    parser.add_argument("--runs", type=Path, nargs="*", default=[], help="train_linear_head.py outputs")
    parser.add_argument("--heads", nargs="+", required=True, help="head names (see linear_head.head_catalog)")
    parser.add_argument("--config", type=Path, default=ROOT / "code/configs/pusht_head_closed_loop.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", type=int, help="first N tasks only (timing / smoke)")
    parser.add_argument("--resume", action="store_true", help="continue in an existing output directory")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"{args.output} exists; choose a new --output or pass --resume")

    from jepa_planning.model import load_world_model
    from scripts.download_checkpoint import sha256

    bank = np.load(args.bank / "bank.npz")
    bank_meta = json.loads((args.bank / "metadata.json").read_text())
    tasks = {key: bank[key] for key in ("group", "initial", "target", "photo_index")}
    if args.tasks:
        tasks = {key: value[: args.tasks] for key, value in tasks.items()}
    features = torch.load(Path(bank_meta["groups"]) / "features.pt", map_location="cpu", weights_only=True)["groups"]
    photo_index = tasks["photo_index"][:, : cfg["goal_photos"]]
    goals = features[torch.from_numpy(photo_index.ravel())].reshape(*photo_index.shape, *features.shape[1:])
    catalog = {info["head"]: (info, head) for info, head in lh.head_catalog(args.runs)}
    missing = [name for name in args.heads if name not in catalog]
    if missing:
        raise KeyError(f"unknown heads: {missing}")

    (args.output / "episodes").mkdir(parents=True, exist_ok=True)
    meta_path = args.output / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {"invocations": []}
    if metadata.get("config", cfg) != cfg or metadata.get("bank", str(args.bank)) != str(args.bank):
        raise ValueError("--resume must use the same config and bank")
    checkpoint = ROOT / "checkpoints/dino_wm_pusht.pth.tar"
    metadata.update(config=cfg, bank=str(args.bank), checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
    metadata["invocations"].append({
        "command": sys.argv, "device": args.device,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))})
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")

    model = load_world_model("pusht", args.device).requires_grad_(False).eval()
    if model.action_dim != 2 * cfg["controls_per_chunk"]:
        raise ValueError("controls_per_chunk does not match the checkpoint action dimension")
    env = make_environment()
    try:
        for name in args.heads:
            path = args.output / "episodes" / f"{name}.csv"
            if path.exists():
                print(f"skip {name} (done)", flush=True)
                continue
            info, head = catalog[name]
            started = perf_counter()
            rows = [{**info, **row} for row in run_head(env, model, head, tasks, goals, cfg)]
            temporary = path.with_suffix(".tmp")
            with temporary.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.rename(path)
            final = np.mean([row["final_position_error"] for row in rows])
            print(f"{name}: {len(rows)} episodes in {perf_counter() - started:.0f}s, "
                  f"mean final position error {final:.1f}px", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
